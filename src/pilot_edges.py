"""Pilot: build a typed claim-graph for ~10 papers around a single search query.

Subcommands:
    intra   Run a synchronous Gemini call per pilot paper to extract
            intra-paper directed edges (supports / assumes / bounded_by /
            interprets / derives_from / sub_step_of) between its claims.
    cross   For each pilot paper, run a Gemini call asking for cross-paper
            edges from its claims to claims in any of the other pilot papers
            (uses_method_of / uses_assumption_of / extends / supersedes /
            contradicts / cites_as_evidence).
    show    Print the edges currently stored for the pilot set.
    purge   Delete all edges with extractor LIKE '%_pilot_v0' (clean rollback).

Usage:
    PYTHONPATH=src python3 src/pilot_edges.py intra
    PYTHONPATH=src python3 src/pilot_edges.py cross
    PYTHONPATH=src python3 src/pilot_edges.py show
    PYTHONPATH=src python3 src/pilot_edges.py purge
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"

sys.path.insert(0, str(REPO_ROOT / "src"))
from askchem.models import (  # noqa: E402
    ClaimEdge,
    EDGE_TYPES,
    INTRA_PAPER_EDGE_TYPES,
    CROSS_PAPER_EDGE_TYPES,
)

PILOT_DOIS = [
    "10.1021/acs.joc.9b01692",
    "10.1021/acs.orglett.8b02911",
    "10.1021/acscatal.4c03531",
    "10.1021/acs.joc.2c00665",
    "10.1021/ACSCATAL.8B03979",
    "10.3762/bjoc.6.70",
    "10.1021/acs.orglett.0c00945",
    "10.1021/ACS.ORGANOMET.1C00085",
    "10.1039/C6RA24769E",
    "10.1021/acsomega.2c01360",
]

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

INTRA_EXTRACTOR = "intra_llm_gemini_pilot_v0"
CROSS_EXTRACTOR = "cross_llm_gemini_pilot_v0"

# ── Prompts ───────────────────────────────────────────────────────────────────

INTRA_PROMPT = """You are analyzing the relationships between scientific claims extracted from a single chemistry paper.

Below are all of the claims from one paper, each with a stable claim_id and structured fields. Identify directed typed edges between these claims that capture the paper's internal scientific reasoning.

Edge types (intra-paper only):
- supports        : claim X provides evidence for, or directly justifies, claim Y
- assumes         : claim X presupposes claim Y as a premise
- bounded_by      : claim X is restricted by the limitation/scope/condition stated in claim Y
- interprets      : claim X is an interpretation of an observation reported in claim Y
- derives_from    : claim X is computed/derived from inputs/results in claim Y
- sub_step_of     : claim X is a sub-component of the larger procedure in claim Y

Return JSON of the form:
{{"edges": [{{"from": "<claim_id>", "to": "<claim_id>", "type": "<edge_type>", "confidence": "high|medium|low", "evidence": "<one sentence rationale or quote from the paper>"}}]}}

Rules:
- Both endpoints must be claim_ids from the input list.
- Skip self-loops (from == to).
- Only emit edges you have clear textual grounds for; skip speculative ones.
- Do not duplicate edges (same from/to/type appears at most once).
- A typical paper has 3-15 internal edges; if you would emit zero, return {{"edges": []}}.

Paper title: {title}
Paper DOI:   {doi}

Claims (JSON):
{claims_json}
"""

CROSS_PROMPT = """You are analyzing whether claims in one chemistry paper depend on claims in other papers.

You are given (a) the source paper's claims and (b) a candidate pool of claims from up to 9 other chemistry papers in the same topic area. Identify directed cross-paper edges from source-paper claims to candidate-pool claims.

Edge types (cross-paper only):
- uses_method_of      : source claim uses an experimental/computational method established in another claim
- uses_assumption_of  : source claim relies on an assumption/result established in another claim
- extends             : source claim builds on or generalizes another claim
- supersedes          : source claim is a clear improvement over another claim under comparable conditions
- contradicts         : source claim disagrees with another claim under comparable conditions
- cites_as_evidence   : source claim invokes another claim as supporting evidence

Return JSON of the form:
{{"edges": [{{"from": "<source_claim_id>", "to": "<candidate_claim_id>", "type": "<edge_type>", "confidence": "high|medium|low", "evidence": "<one sentence rationale>"}}]}}

Rules:
- "from" MUST be a claim_id from the source paper.
- "to" MUST be a claim_id from the candidate pool (NOT the source paper).
- Only emit edges with strong textual evidence; do not speculate.
- The source paper may legitimately have zero cross-paper edges to this small candidate pool — return {{"edges": []}} in that case.

Source paper title: {source_title}
Source paper DOI:   {source_doi}

Source paper claims (JSON):
{source_claims_json}

Candidate pool from other papers (JSON):
{other_claims_json}
"""


# ── DB helpers ────────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def fetch_pilot_claims(con: sqlite3.Connection) -> dict[str, list[dict]]:
    """Return {doi: [claim_dict, ...]} for the pilot DOIs, sorted by claim_id."""
    out: dict[str, list[dict]] = {}
    for doi in PILOT_DOIS:
        rows = con.execute(
            "SELECT claim_id, data FROM claims WHERE source_doi = ? ORDER BY claim_id",
            (doi,),
        ).fetchall()
        out[doi] = [{**json.loads(d), "claim_id": cid} for cid, d in rows]
    return out


def claim_for_prompt(c: dict) -> dict:
    """Compact claim representation for the LLM (drops noisy/empty fields)."""
    keep = {
        "claim_id": c.get("claim_id"),
        "claim_type": c.get("claim_type"),
        "verbatim_quote": c.get("verbatim_quote", ""),
    }
    for k in (
        "reaction_type", "subject", "subject_smiles", "property_name",
        "value", "unit", "measurement_method",
        "process_described", "steps", "key_intermediates",
        "technique_name", "what_it_achieves", "key_innovation", "limitations",
        "compared_items", "metric", "comparison_result",
        "hypothesis_text", "limitation_text", "direction_text",
        "finding_text", "why_surprising",
    ):
        v = c.get(k)
        if v:
            keep[k] = v
    if c.get("reactants"):
        keep["reactants"] = [r.get("name") for r in c["reactants"][:5] if isinstance(r, dict)]
    if c.get("products"):
        keep["products"] = [r.get("name") for r in c["products"][:5] if isinstance(r, dict)]
    if c.get("conditions"):
        keep["conditions"] = {k: v for k, v in c["conditions"].items() if v}
    if c.get("outcomes"):
        keep["outcomes"] = {k: v for k, v in c["outcomes"].items() if v not in (None, "", "null")}
    return keep


def insert_edges(con: sqlite3.Connection, edges: list[ClaimEdge]) -> int:
    if not edges:
        return 0
    rows = [
        (
            e.from_claim_id, e.edge_type,
            e.to_claim_id or "", e.to_doi or "",
            e.confidence, e.evidence,
            e.extractor, e.extracted_at,
        )
        for e in edges
    ]
    cur = con.executemany(
        """INSERT OR IGNORE INTO claim_edges
           (from_claim_id, edge_type, to_claim_id, to_doi,
            confidence, evidence, extractor, extracted_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()
    return cur.rowcount or 0


# ── PortKey / Gemini call ─────────────────────────────────────────────────────


def call_gemini(prompt: str, *, max_tokens: int = 32768, retries: int = 3) -> dict:
    """Synchronous Gemini chat completion via the NYU PortKey gateway."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    cmd = [
        "curl", "-s", "--max-time", "300", "-X", "POST",
        "-H", "x-portkey-api-key: " + api_key,
        "-H", "x-portkey-provider: " + PROVIDER,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(body),
        GATEWAY + "/chat/completions",
    ]
    last_err = None
    for attempt in range(retries):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=320)
        if result.returncode == 0 and result.stdout.strip():
            try:
                resp = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                last_err = f"json decode: {e}: {result.stdout[:200]}"
            else:
                choices = resp.get("choices") or []
                if choices:
                    msg = choices[0].get("message", {})
                    content = (msg.get("content") or "").strip()
                    finish = choices[0].get("finish_reason")
                    if not content:
                        last_err = f"empty content (finish_reason={finish}, usage={resp.get('usage')})"
                    else:
                        try:
                            return {
                                "parsed": json.loads(content),
                                "usage": resp.get("usage", {}),
                                "finish_reason": finish,
                            }
                        except json.JSONDecodeError as e:
                            last_err = (
                                f"content not JSON (finish_reason={finish}): {e}: "
                                f"{content[:300]}"
                            )
                else:
                    last_err = f"no choices: {json.dumps(resp)[:300]}"
        else:
            last_err = f"curl rc={result.returncode}: {result.stderr[:200]}"
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_err}")


# ── Edge parsing ──────────────────────────────────────────────────────────────


def parse_edges(
    raw: list[dict],
    *,
    valid_from_ids: set[str],
    valid_to_ids: set[str],
    allowed_types: set[str],
    extractor: str,
    now: str,
) -> tuple[list[ClaimEdge], list[str]]:
    """Validate the LLM output and return (edges, problems)."""
    edges: list[ClaimEdge] = []
    problems: list[str] = []
    seen: set[tuple] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"#{i}: not a dict")
            continue
        f = (item.get("from") or "").strip()
        t = (item.get("to") or "").strip()
        et = (item.get("type") or "").strip()
        if et not in allowed_types:
            problems.append(f"#{i}: bad type {et!r}")
            continue
        if f == t:
            problems.append(f"#{i}: self-loop {f}")
            continue
        if f not in valid_from_ids:
            problems.append(f"#{i}: from {f!r} not in source-paper claims")
            continue
        if t not in valid_to_ids:
            problems.append(f"#{i}: to {t!r} not in valid target pool")
            continue
        key = (f, et, t)
        if key in seen:
            continue
        seen.add(key)
        conf = item.get("confidence", "medium")
        if conf not in {"high", "medium", "low"}:
            conf = "medium"
        edges.append(ClaimEdge(
            from_claim_id=f,
            edge_type=et,
            to_claim_id=t,
            confidence=conf,
            evidence=(item.get("evidence") or "")[:500],
            extractor=extractor,
            extracted_at=now,
        ))
    return edges, problems


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_intra(_args):
    con = open_db()
    by_doi = fetch_pilot_claims(con)

    total_edges = 0
    total_inserted = 0
    total_tokens = 0
    now = datetime.utcnow().isoformat() + "Z"

    for doi in PILOT_DOIS:
        claims = by_doi.get(doi, [])
        if len(claims) < 2:
            print(f"[skip] {doi}: only {len(claims)} claim(s)")
            continue

        title = claims[0].get("source_paper_title", "")
        compact = [claim_for_prompt(c) for c in claims]
        prompt = INTRA_PROMPT.format(
            title=title,
            doi=doi,
            claims_json=json.dumps(compact, indent=2),
        )

        print(f"[intra] {doi} ({len(claims)} claims) ... ", end="", flush=True)
        try:
            resp = call_gemini(prompt, max_tokens=32768)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        raw_edges = resp["parsed"].get("edges", [])
        valid_ids = {c["claim_id"] for c in claims}
        edges, problems = parse_edges(
            raw_edges,
            valid_from_ids=valid_ids,
            valid_to_ids=valid_ids,
            allowed_types=INTRA_PAPER_EDGE_TYPES,
            extractor=INTRA_EXTRACTOR,
            now=now,
        )
        inserted = insert_edges(con, edges)
        total_edges += len(edges)
        total_inserted += inserted
        usage = resp.get("usage") or {}
        total_tokens += usage.get("total_tokens", 0) or 0
        print(
            f"raw={len(raw_edges)} valid={len(edges)} new={inserted} "
            f"tokens={usage.get('total_tokens', '?')}"
            + (f" problems={len(problems)}" if problems else "")
        )
        for p in problems[:3]:
            print(f"    ! {p}")
        time.sleep(1)

    print(f"\n[intra summary] edges_valid={total_edges} inserted={total_inserted} total_tokens={total_tokens}")


def cmd_cross(_args):
    con = open_db()
    by_doi = fetch_pilot_claims(con)
    all_compact = {doi: [claim_for_prompt(c) for c in cs] for doi, cs in by_doi.items()}

    total_edges = 0
    total_inserted = 0
    total_tokens = 0
    now = datetime.utcnow().isoformat() + "Z"

    for src_doi in PILOT_DOIS:
        src_claims = by_doi.get(src_doi, [])
        if not src_claims:
            print(f"[skip] {src_doi}: no claims")
            continue

        # Candidate pool = every other pilot paper's claims, tagged with their DOI.
        other_pool = []
        for other_doi in PILOT_DOIS:
            if other_doi == src_doi:
                continue
            for c in all_compact[other_doi]:
                other_pool.append({**c, "_paper_doi": other_doi})
        if not other_pool:
            continue

        title = src_claims[0].get("source_paper_title", "")
        prompt = CROSS_PROMPT.format(
            source_title=title,
            source_doi=src_doi,
            source_claims_json=json.dumps(all_compact[src_doi], indent=2),
            other_claims_json=json.dumps(other_pool, indent=2),
        )

        print(f"[cross] {src_doi} ({len(src_claims)} src vs {len(other_pool)} cand) ... ", end="", flush=True)
        try:
            resp = call_gemini(prompt, max_tokens=32768)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        raw_edges = resp["parsed"].get("edges", [])
        valid_from = {c["claim_id"] for c in src_claims}
        valid_to = {c["claim_id"] for c in other_pool}
        edges, problems = parse_edges(
            raw_edges,
            valid_from_ids=valid_from,
            valid_to_ids=valid_to,
            allowed_types=CROSS_PAPER_EDGE_TYPES,
            extractor=CROSS_EXTRACTOR,
            now=now,
        )
        inserted = insert_edges(con, edges)
        total_edges += len(edges)
        total_inserted += inserted
        usage = resp.get("usage") or {}
        total_tokens += usage.get("total_tokens", 0) or 0
        print(
            f"raw={len(raw_edges)} valid={len(edges)} new={inserted} "
            f"tokens={usage.get('total_tokens', '?')}"
            + (f" problems={len(problems)}" if problems else "")
        )
        for p in problems[:3]:
            print(f"    ! {p}")
        time.sleep(1)

    print(f"\n[cross summary] edges_valid={total_edges} inserted={total_inserted} total_tokens={total_tokens}")


def cmd_show(_args):
    con = open_db()
    pilot_ids = set()
    for doi in PILOT_DOIS:
        pilot_ids.update(
            r[0] for r in con.execute(
                "SELECT claim_id FROM claims WHERE source_doi = ?", (doi,)
            ).fetchall()
        )

    if not pilot_ids:
        print("(no pilot claims found)")
        return

    placeholders = ",".join("?" * len(pilot_ids))
    rows = con.execute(
        f"""SELECT from_claim_id, edge_type, to_claim_id, to_doi,
                   confidence, extractor, substr(evidence, 1, 80)
              FROM claim_edges
             WHERE from_claim_id IN ({placeholders})
                OR to_claim_id   IN ({placeholders})
             ORDER BY extractor, edge_type, from_claim_id""",
        list(pilot_ids) + list(pilot_ids),
    ).fetchall()

    if not rows:
        print("(no edges yet for the pilot set)")
        return

    print(f"{'from':>10}  {'type':<20}  {'to':>10}  {'conf':<6}  {'extractor':<32}  evidence")
    print("-" * 130)
    for f, et, tc, td, conf, ex, ev in rows:
        target = tc[:8] if tc else f"DOI:{td[:30]}"
        print(f"{f[:8]:>10}  {et:<20}  {target:>10}  {conf:<6}  {ex:<32}  {ev or ''}")
    print(f"\n{len(rows)} edges total")


def cmd_purge(_args):
    con = open_db()
    cur = con.execute(
        "DELETE FROM claim_edges WHERE extractor LIKE '%_pilot_v0'"
    )
    con.commit()
    print(f"deleted {cur.rowcount} pilot edges")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("intra", help="Run intra-paper LLM relational pass on the pilot set")
    sub.add_parser("cross", help="Run cross-paper LLM relational pass on the pilot set")
    sub.add_parser("show",  help="Show pilot edges currently stored")
    sub.add_parser("purge", help="Delete all *_pilot_v0 edges")
    args = parser.parse_args()
    {"intra": cmd_intra, "cross": cmd_cross, "show": cmd_show, "purge": cmd_purge}[args.cmd](args)


if __name__ == "__main__":
    main()
