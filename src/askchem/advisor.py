"""Advisor + paper-intelligence for the living tree.

AskChem acts as an advisor, not an answer engine: for a paper at its node it asks
sharp positioning questions, critiques whether its claims are supported by its own
evidence, and states its structural contribution relative to the host principle and
branch neighbors.

These three analyses are normally PRE-COMPUTED in batch (see
living_taxonomy/precompute_analysis.py) and stored in `paper_analysis`, so the UI is
instant. This module owns the shared prompt + parsing, exposes a live fallback used
only when a row is missing, and serves stored results first.

LLM = Gemini via the NYU gateway (needs PORTKEY_API_KEY) for the live fallback only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone

from askchem import db, ltree

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

# The formalized question set the advisor instantiates (grounded; skip if N/A).
QUESTION_TYPES = (
    "differentiation: how does this paper differ from its branch neighbors?",
    "novelty: what genuinely new insight does it add beyond the branch?",
    "contradiction_exception: does it conflict with specific neighbor claims, "
    "and what would that indicate (mechanism revision / scope limit / split)?",
    "positioning: is it an outlier, does it straddle branches, or suggest a new "
    "sub-branch?",
    "evidence_strength: is its central claim as well-supported as the branch norm?",
)

# System prompt for the combined three-analysis pass (advisor + critique + contribution).
ANALYSIS_SYS = (
    "You analyze ONE paper at its position in a chemistry knowledge tree, grounded "
    "ONLY in the claims, verbatim quotes, and neighbor claims provided - never invent "
    "evidence. Be factual and neutral: map each claim to the evidence stated in the "
    "extracted text. Do NOT pass judgment on the authors or call work good/bad; simply "
    "note which claims have explicit supporting evidence in the text and which do not.")


def _gemini_chat(system, user, max_time=120):
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY not set")
    payload = {"model": MODEL, "temperature": 0.2,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "response_format": {"type": "json_object"}}
    cmd = ["curl", "-s", "--max-time", str(max_time), "-X", "POST",
           "-H", f"x-portkey-api-key: {api_key}",
           "-H", f"x-portkey-provider: {PROVIDER}",
           "-H", "Content-Type: application/json",
           "-d", json.dumps(payload), f"{GATEWAY}/chat/completions"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 30)
    out = json.loads(res.stdout)
    return out["choices"][0]["message"]["content"]


def _parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", (text or "").strip()).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def _siblings(view_id, node_id, doi, limit=24):
    with db.get_conn() as c:
        rows = c.execute(
            "SELECT l.doi AS doi, s.title AS title, cl.claim_type AS ctype, "
            "cl.verbatim_quote AS quote "
            "FROM taxonomy_leaves l JOIN claims cl ON cl.claim_id=l.claim_id "
            "LEFT JOIN sources s ON s.doi=l.doi "
            "WHERE l.view_id=? AND l.node_id=? AND l.doi<>? "
            "ORDER BY s.citation_count DESC LIMIT ?",
            (view_id, node_id, doi, limit)).fetchall()
    return [{"doi": r["doi"], "title": r["title"], "type": r["ctype"],
             "quote": r["quote"]} for r in rows]


# ── shared context + combined prompt (used by live fallback AND batch) ───────────

def gather_context(view_id, node_id, doi):
    paper = ltree.get_paper_claims(view_id, node_id, doi)
    node = dict(ltree.get_node(view_id, node_id, depth=0) or {})
    path = ltree.get_path(view_id, node_id)
    siblings = _siblings(view_id, node_id, doi)
    branch = " > ".join(p["name"] for p in path) or node.get("name", node_id)
    node["_title"] = paper.get("title")
    return paper, node, branch, siblings


def build_analysis_user(paper, node, branch, siblings, doi):
    paper_block = "\n".join(
        f"- ({c['claim_type']}) {c['label']}: {(c['quote'] or '')[:220]}"
        for c in paper.get("claims", [])) or "(none)"
    sib_block = "\n".join(
        f"- [{s['doi']}] ({s['type']}) {(s['quote'] or s['title'] or '')[:200]}"
        for s in siblings) or "(no other papers under this branch yet)"
    law = node.get("definition") or "(no stated principle)"
    proposed = ("\nNote: this branch was auto-proposed (the paper did not fit an "
                "existing branch cleanly)." if node.get("proposed") else "")
    return (
        f"BRANCH (tree path): {branch}{proposed}\n"
        f"HOST PRINCIPLE ({node.get('name', '')}): {law}\n\n"
        f"PAPER under review: {paper.get('title') or doi} ({doi})\n"
        f"Its claims placed in this branch:\n{paper_block}\n\n"
        f"NEIGHBOR papers' claims in the same branch:\n{sib_block}\n\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  "advisor": {"questions": [{"type": "differentiation|novelty|'
        'contradiction_exception|positioning|evidence_strength", "question": "...", '
        '"grounded_on": ["<neighbor DOI from the list>"]}]},\n'
        '  "critique": {"overall": "<neutral one-line summary of how claims map to '
        'the evidence stated in the extracted text - no verdict, no praise/blame>", '
        '"supported": ["<claim that has explicit supporting evidence in its quote>"], '
        '"weak": [{"claim": "<claim with no explicit supporting evidence in the '
        'extracted text>"}]},\n'
        '  "contribution": {"relation_to_principle": "<how it instantiates/uses the '
        'host principle>", "extends": "<what it adds>", "challenges": "<what it '
        'questions, or empty>", "vs_neighbors": "<how it differs from neighbors>", '
        '"significance": "<incremental | notable | potentially paradigm-shifting, with '
        'reason>"}\n'
        "}\n"
        "Ground advisor questions only in the neighbor DOIs listed; omit any advisor "
        "type you cannot ground. Keep every field concise.")


def split_analysis(view_id, node_id, doi, branch, node, siblings, parsed):
    """One combined LLM result -> (advisor_json, critique_json, contribution_json)."""
    adv = parsed.get("advisor") if isinstance(parsed, dict) else {}
    adv = adv or {}
    raw_questions = adv.get("questions") or []
    sib_by_doi = {s["doi"]: s for s in siblings}
    # The LLM occasionally returns a question as a bare string (or the whole
    # "grounded_on" as a string). Normalize to the expected dict shape so one
    # malformed response can't crash the batch.
    questions = []
    for q in raw_questions:
        if isinstance(q, str):
            q = {"type": "other", "question": q, "grounded_on": []}
        elif not isinstance(q, dict):
            continue
        g = q.get("grounded_on") or []
        if isinstance(g, str):
            g = [g]
        q["grounded_on"] = [{"doi": d, "title": (sib_by_doi.get(d) or {}).get("title", "")}
                            for d in g if isinstance(d, str) and d]
        questions.append(q)
    advisor_json = {"view_id": view_id, "node_id": node_id, "branch": branch,
                    "doi": doi, "title": node.get("_title"), "n_siblings": len(siblings),
                    "proposed_branch": bool(node.get("proposed")),
                    "questions": questions, "error": None}
    return (json.dumps(advisor_json),
            json.dumps(parsed.get("critique") or {}),
            json.dumps(parsed.get("contribution") or {}))


def _store(view_id, node_id, doi, advisor_j, critique_j, contribution_j):
    with db.get_conn(readonly=False) as c:
        c.execute(
            "INSERT OR REPLACE INTO paper_analysis(view_id,node_id,doi,advisor_json,"
            "critique_json,contribution_json,generated_at) VALUES (?,?,?,?,?,?,?)",
            (view_id, node_id, doi, advisor_j, critique_j, contribution_j,
             datetime.now(timezone.utc).isoformat()))
        c.commit()


def live_analysis(view_id, node_id, doi, store=True):
    """Compute all three analyses live (fallback when no precomputed row exists)."""
    paper, node, branch, siblings = gather_context(view_id, node_id, doi)
    parsed = _parse_json(_gemini_chat(ANALYSIS_SYS,
                                      build_analysis_user(paper, node, branch, siblings, doi)))
    a, cr, co = split_analysis(view_id, node_id, doi, branch, node, siblings, parsed)
    if store:
        try:
            _store(view_id, node_id, doi, a, cr, co)
        except Exception:
            pass
    return {"advisor": json.loads(a), "critique": json.loads(cr),
            "contribution": json.loads(co)}


# ── served accessors: stored-first, live fallback ────────────────────────────────

def _live_enabled() -> bool:
    """Whether to attempt the live LLM fallback for a missing row.

    Prod (the DigitalOcean VPS) cannot reach the NYU gateway (it resolves to a
    private 10.x address, VPN-only), so attempting a live call there would just
    stall until the curl timeout. We gate it on PORTKEY_API_KEY being present
    AND CHEMTREE_ADVISOR_NO_LIVE being unset, so prod serves purely from the
    precomputed `paper_analysis` rows and fails fast for anything not stored.
    """
    return bool(os.environ.get("PORTKEY_API_KEY")) and not os.environ.get(
        "CHEMTREE_ADVISOR_NO_LIVE")


def _missing_advisor(view_id, node_id, doi, branch, err):
    return {"view_id": view_id, "node_id": node_id, "branch": branch, "doi": doi,
            "title": None, "n_siblings": 0, "proposed_branch": False,
            "questions": [], "error": err}


def advise(view_id: str, node_id: str, doi: str) -> dict:
    """Advisor questions: stored if precomputed, else live (and store)."""
    stored = ltree.get_analysis(view_id, node_id, doi)
    if stored.get("advisor"):
        out = dict(stored["advisor"])
        out["cached"] = True
        return out
    if _live_enabled():
        try:
            res = live_analysis(view_id, node_id, doi, store=True)
            out = dict(res["advisor"])
            out["cached"] = False
            return out
        except Exception as e:
            path = ltree.get_path(view_id, node_id)
            branch = " > ".join(p["name"] for p in path) or node_id
            return _missing_advisor(view_id, node_id, doi, branch, str(e))
    path = ltree.get_path(view_id, node_id)
    branch = " > ".join(p["name"] for p in path) or node_id
    return _missing_advisor(view_id, node_id, doi, branch, None)


def _served_part(view_id, node_id, doi, part):
    """critique/contribution: stored if precomputed, else live (and store)."""
    stored = ltree.get_analysis(view_id, node_id, doi)
    if stored.get(part):
        return {"view_id": view_id, "node_id": node_id, "doi": doi,
                part: stored[part], "cached": True, "error": None}
    if _live_enabled():
        try:
            res = live_analysis(view_id, node_id, doi, store=True)
            return {"view_id": view_id, "node_id": node_id, "doi": doi,
                    part: res[part], "cached": False, "error": None}
        except Exception as e:
            return {"view_id": view_id, "node_id": node_id, "doi": doi,
                    part: None, "cached": False, "error": str(e)}
    return {"view_id": view_id, "node_id": node_id, "doi": doi,
            part: None, "cached": False, "error": None}


def critique(view_id: str, node_id: str, doi: str) -> dict:
    return _served_part(view_id, node_id, doi, "critique")


def contribution(view_id: str, node_id: str, doi: str) -> dict:
    return _served_part(view_id, node_id, doi, "contribution")
