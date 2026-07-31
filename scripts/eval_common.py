"""Shared helpers for the Phase 0 retrieval-eval pipeline.

The five eval scripts (``build_eval_candidates``, ``llm_judge_eval``,
``spot_check_labels``, ``eval_metrics``, plus the future encoder
bake-off driver) all need to:

  - load a probes file
  - load a candidates file
  - look up a claim by id
  - render a (query, claim) pair into the *same* text the user sees
  - read/write JSONL atomically

Centralising those bits here keeps the per-script files small and
ensures the judge sees exactly the same claim text the spot-checker
sees, which is what makes Cohen's κ a meaningful sanity check.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import get_db_path  # noqa: E402

EVAL_DIR = REPO_ROOT / "data" / "eval"
PROBES_PATH = EVAL_DIR / "probes_v1.jsonl"
CANDIDATES_PATH = EVAL_DIR / "candidates_v1.jsonl"
LABELS_PATH = EVAL_DIR / "labels_v1.jsonl"
SPOT_CHECK_PATH = EVAL_DIR / "spot_check_v1.json"


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class Probe:
    id: str
    q: str
    family: str
    notes: str = ""
    view: str | None = None
    claim_type: str | None = None
    mode: str = "auto"
    sort: str = "relevance"


def load_probes(path: Path = PROBES_PATH) -> list[Probe]:
    out: list[Probe] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        d = json.loads(raw)
        out.append(Probe(
            id=d["id"], q=d["q"], family=d["family"],
            notes=d.get("notes", ""),
            view=d.get("view"),
            claim_type=d.get("claim_type"),
            mode=d.get("mode", "auto"),
            sort=d.get("sort", "relevance"),
        ))
    return out


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if raw:
            yield json.loads(raw)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Claim loading ───────────────────────────────────────────────────────────


def open_claims_db() -> sqlite3.Connection:
    """Read-only connection to chemtree.db. Caller must close."""
    db_path = get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_claims(claim_ids: list[str], conn: sqlite3.Connection) -> dict[str, dict]:
    """Bulk-load claim JSON for the given ids. Returns dict{cid: claim}.

    Joins ``claim_contextualized`` and ``paper_summary`` onto each claim
    so the judge / spot-checker / metrics see the post-Sprint-1 text the
    user sees.
    """
    out: dict[str, dict] = {}
    if not claim_ids:
        return out
    BATCH = 900  # SQLite IN-list limit safety
    for i in range(0, len(claim_ids), BATCH):
        chunk = claim_ids[i:i + BATCH]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT c.claim_id, c.data, c.claim_contextualized, "
            f"s.paper_summary "
            f"FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
            f"WHERE c.claim_id IN ({ph})",
            chunk,
        ).fetchall()
        for r in rows:
            try:
                claim = json.loads(r["data"])
            except Exception:
                continue
            if r["claim_contextualized"]:
                claim["claim_contextualized"] = r["claim_contextualized"]
            if r["paper_summary"]:
                claim["paper_summary"] = r["paper_summary"]
            out[r["claim_id"]] = claim
    return out


# ── Renderer for the judge / spot-checker ───────────────────────────────────


def render_claim_for_judge(c: dict, max_verbatim: int = 320) -> str:
    """Plain-text render of a claim that mirrors ``renderClaim`` in
    web/index.html. Used by the LLM judge AND the human spot-checker,
    so they both see the same payload.

    Post Sprint 1: when ``claim_contextualized`` is present we lead with
    the LLM-rewritten standalone sentence, then follow with the typed
    fields the user also sees in the data card. This keeps the judge
    aligned with the UI.
    """
    parts: list[str] = []
    ctx = (c.get("claim_contextualized") or "").strip()
    if ctx:
        parts.append(f"CLAIM: {ctx}")
    title = (c.get("source_paper_title") or "").strip()
    if title:
        parts.append(f"PAPER: {title}")
    venue = (c.get("source_venue") or c.get("venue") or "").strip()
    if venue:
        parts.append(f"VENUE: {venue}")
    ct = (c.get("claim_type") or "").strip()
    if ct:
        parts.append(f"TYPE: {ct}")

    # Type-specific primary fields, in the same priority the renderer uses.
    rxn = (c.get("reaction_type") or "").strip()
    if rxn:
        parts.append(f"REACTION: {rxn}")
    reactants = c.get("reactants") or []
    products = c.get("products") or []

    def _names(items):
        names = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict):
                n = it.get("name") or it.get("smiles") or it.get("formula") or ""
            else:
                n = str(it)
            if n:
                names.append(n)
        return names

    rn = _names(reactants)
    pn = _names(products)
    if rn or pn:
        arrow = " + ".join(rn) + " -> " + (" + ".join(pn) if pn else "?")
        parts.append(f"ARROW: {arrow}")
    cond = c.get("conditions")
    if isinstance(cond, dict):
        cond_parts = []
        for k in ("catalyst", "ligand", "solvent", "temperature",
                  "pressure", "atmosphere"):
            v = cond.get(k)
            if v and str(v).lower() != "null":
                if isinstance(v, list):
                    cond_parts.append(f"{k}={', '.join(str(x) for x in v)}")
                else:
                    cond_parts.append(f"{k}={v}")
        if cond_parts:
            parts.append("CONDITIONS: " + " | ".join(cond_parts))

    for k_label, k in [
        ("SUBJECT", "subject"),
        ("PROPERTY", "property_name"),
        ("PROPERTY_CATEGORY", "property_category"),
        ("MEASUREMENT_METHOD", "measurement_method"),
        ("TECHNIQUE", "technique_name"),
        ("WHAT_IT_ACHIEVES", "what_it_achieves"),
        ("KEY_INNOVATION", "key_innovation"),
        ("PROCESS", "process_described"),
        ("HYPOTHESIS", "hypothesis_text"),
        ("LIMITATION", "limitation_text"),
        ("DIRECTION", "direction_text"),
        ("FINDING", "finding_text"),
        ("COMPARISON_RESULT", "comparison_result"),
    ]:
        v = c.get(k)
        if v and str(v).strip():
            parts.append(f"{k_label}: {v}")
    val = c.get("value")
    unit = c.get("unit") or ""
    if val not in (None, "", "null"):
        parts.append(f"VALUE: {val} {unit}".strip())

    compared = c.get("compared_items")
    if isinstance(compared, list) and compared:
        ci_names = []
        for it in compared:
            if isinstance(it, dict):
                ci_names.append(it.get("name") or it.get("label") or "")
            else:
                ci_names.append(str(it))
        ci_names = [x for x in ci_names if x]
        if ci_names:
            parts.append("COMPARED_ITEMS: " + " vs ".join(ci_names))
    metric = (c.get("metric") or "").strip()
    if metric:
        parts.append(f"METRIC: {metric}")

    quote = (c.get("verbatim_quote") or "").strip()
    if quote:
        if len(quote) > max_verbatim:
            quote = quote[:max_verbatim].rstrip() + "..."
        parts.append(f"VERBATIM: {quote}")

    return "\n".join(parts)
