"""Placement engine for the living taxonomy pilot (LLM-minimal).

For each candidate leaf we embed its descriptor with the same encoder the
production index uses (``mxbai-embed-large-v1``), score it against every
leaf-hosting branch of the seed tree, and decide by similarity threshold:

    sim >= ATTACH_THRESHOLD   -> attach_leaf   (no LLM)
    sim <= EXCEPTION_THRESHOLD-> exception      (no LLM; candidate new branch)
    otherwise                 -> gray_zone      (optional Gemini adjudication)

Thresholds are calibrated on the pilot, then the cheap path runs at scale.
Generative LLM calls (gray-zone adjudication, naming a new branch) go
through Gemini on the NYU gateway, in batch mode when volume is high.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

# mxbai recipe — must match src/askchem/embeddings_v2.py so pilot vectors
# live in the same space as the deployed corpus.
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# NYU Gemini gateway (mirrors src/classify_papers.py).
GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

ATTACH_THRESHOLD = 0.55
EXCEPTION_THRESHOLD = 0.42

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        device = "mps" if _mps_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def _mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _embed(texts, is_query=False):
    """Embed a list of texts -> (n, d) float32, L2-normalized."""
    model = _get_model()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    vecs = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        batch_size=32, show_progress_bar=False,
    )
    return vecs.astype(np.float32)


@dataclass
class Placement:
    claim_id: str
    doi: str
    title: str
    year: int
    text: str
    decision: str            # attach_leaf | exception | gray_zone
    branch_path: list        # nearest branch path
    score: float
    runner_up: tuple = ()     # (branch_name, score)
    current_path: list = field(default_factory=list)
    llm_verdict: str = ""     # filled only for adjudicated gray-zone leaves


def place_leaves(leaves, branches, attach=ATTACH_THRESHOLD,
                 exception=EXCEPTION_THRESHOLD):
    """Place candidate leaves against seed-tree branches by embedding sim.

    ``branches`` is ``[(path, name, desc), ...]`` from seed_trees.leaf_branches.
    Returns a list of :class:`Placement`.
    """
    if not leaves:
        return []
    branch_paths = [b[0] for b in branches]
    branch_descs = [f"{b[1]}. {b[2]}" for b in branches]
    branch_vecs = _embed(branch_descs, is_query=False)

    leaf_vecs = _embed([lf["text"] for lf in leaves], is_query=True)
    sims = leaf_vecs @ branch_vecs.T  # (n_leaves, n_branches), cosine

    placements = []
    for i, lf in enumerate(leaves):
        order = np.argsort(-sims[i])
        best = int(order[0])
        best_score = float(sims[i, best])
        runner = ()
        if len(order) > 1:
            r = int(order[1])
            runner = (branches[r][1], float(sims[i, r]))

        if best_score >= attach:
            decision = "attach_leaf"
        elif best_score <= exception:
            decision = "exception"
        else:
            decision = "gray_zone"

        placements.append(Placement(
            claim_id=lf["claim_id"], doi=lf["doi"], title=lf["title"],
            year=lf["year"], text=lf["text"], decision=decision,
            branch_path=branch_paths[best], score=best_score,
            runner_up=runner, current_path=lf.get("current_path", []),
        ))
    return placements


# ── optional LLM gray-zone adjudication (low volume -> sync calls) ────────────

def _gemini_chat(system, user, max_time=60):
    """Single Gemini chat completion via the NYU Portkey gateway (curl)."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY not set; cannot call Gemini.")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    cmd = [
        "curl", "-s", "--max-time", str(max_time), "-X", "POST",
        "-H", f"x-portkey-api-key: {api_key}",
        "-H", f"x-portkey-provider: {PROVIDER}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
        f"{GATEWAY}/chat/completions",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 30)
    try:
        out = json.loads(res.stdout)
        return out["choices"][0]["message"]["content"]
    except Exception:
        return ""


_ADJUDICATE_SYS = (
    "You are organizing a chemistry knowledge tree. Decide whether a new "
    "observation belongs under the proposed branch, or is an exception that "
    "needs a new branch. Answer with a single word: ATTACH or EXCEPTION, "
    "then a short reason."
)


def adjudicate_gray_zone(placements, branches):
    """Resolve gray-zone placements with Gemini (low volume -> sync)."""
    by_name = {b[1]: b for b in branches}
    for p in placements:
        if p.decision != "gray_zone":
            continue
        branch_name = p.branch_path[-1]
        branch = by_name.get(branch_name)
        desc = branch[2] if branch else ""
        user = (
            f"Proposed branch: {branch_name}\nBranch scope: {desc}\n\n"
            f"New observation: {p.text}\n\n"
            "Does the observation belong under this branch?"
        )
        verdict = _gemini_chat(_ADJUDICATE_SYS, user)
        p.llm_verdict = verdict.strip()
        head = verdict.strip().upper()
        if head.startswith("ATTACH"):
            p.decision = "attach_leaf"
        elif head.startswith("EXCEPTION"):
            p.decision = "exception"
    return placements
