"""Sprint 0 — generate `paper_summary` for every paper that has at least one
deep_v1 (full-paper extraction) claim.

The summary is grounded in the paper's already-extracted claim list, NOT in the
full paper text. The summary is then used as a paper-level slot in the Sprint 1
contextualization prompt, and as a separate field in the future multi-field
FTS index.

Pipeline (mirrors src/batch_extract_arxiv.py):

    PYTHONPATH=src python3 scripts/summarize_papers.py prepare [--limit N]
        Build JSONL chunks for the Vertex Batch API.

    PYTHONPATH=src python3 scripts/summarize_papers.py submit
        Upload chunks, create batches, record IDs in tracker.json.

    PYTHONPATH=src python3 scripts/summarize_papers.py status
        Poll Vertex for batch progress.

    PYTHONPATH=src python3 scripts/summarize_papers.py collect
        Download finished batch outputs to outputs/.

    PYTHONPATH=src python3 scripts/summarize_papers.py apply
        Parse outputs and UPDATE sources.paper_summary in chemtree.db.

    PYTHONPATH=src python3 scripts/summarize_papers.py dryrun [--limit 10]
        Synchronously summarize N papers and print results — costs ~2x batch
        but lets us hand-rate the prompt before paying for the full run.

Cost reference (Gemini 3.1 Pro Preview Batch, 41k papers):
    ~21M input tokens × $1/M  +  6M output tokens × $6/M  ≈  $58.

Resumability:
    `apply` skips papers whose `sources.paper_summary` is already non-NULL
    with the same `paper_summary_version`. Re-running prepare/submit on a
    finished pipeline is a no-op (existing tracker entries stay intact).
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.gemini_batch import (  # noqa: E402
    MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M,
    make_request_line, write_chunks, write_manifest,
    submit_all, poll_all, collect_all, iter_output_rows, call_sync,
)

PROMPT_VERSION = "v1"
PROMPT_PATH = REPO_ROOT / "src" / "askchem" / "prompts" / f"summarize_paper_{PROMPT_VERSION}.txt"
PIPELINE_DIR = REPO_ROOT / "data" / "batch_jobs" / f"summarize_papers_{PROMPT_VERSION}"

MAX_CLAIMS_PER_PAPER = 60
MAX_QUOTE_CHARS = 280
# Gemini 3.1 Pro Preview is a reasoning model — its reasoning tokens count as
# output. Real visible response is tiny (~120 tokens for an 80-word paragraph
# wrapped in JSON), but the reasoning trace can chew thousands. Match the
# generous ceiling other AskChem batch scripts use.
MAX_OUTPUT_TOKENS = 16384


# ── DB helpers ───────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=60.0)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA busy_timeout = 30000")
    con.row_factory = sqlite3.Row
    return con


def list_target_papers(con: sqlite3.Connection,
                       *, limit: int | None = None,
                       only_missing: bool = True) -> list[str]:
    """All DOIs that have ≥1 deep_v1 claim, optionally filtered to those
    without an existing `paper_summary` for this prompt version.
    """
    rows = con.execute("""
        SELECT DISTINCT c.source_doi
          FROM claims c
         WHERE c.extraction_version IN ('deep_v1','deep_v1_abstract')
    """).fetchall()
    dois = [r["source_doi"] for r in rows if r["source_doi"]]

    if only_missing:
        done = {r["doi"] for r in con.execute(
            "SELECT doi FROM sources "
            "WHERE paper_summary IS NOT NULL "
            "  AND paper_summary_version = ?",
            (PROMPT_VERSION,)
        ).fetchall()}
        dois = [d for d in dois if d not in done]

    dois.sort()
    if limit is not None:
        dois = dois[:limit]
    return dois


def fetch_paper_inputs(con: sqlite3.Connection, doi: str) -> dict:
    """Return dict ready for prompt formatting: title, venue, year, claims_block."""
    src = con.execute(
        "SELECT title, venue, year FROM sources WHERE doi = ?", (doi,)
    ).fetchone()
    title = (src["title"] if src else "") or ""
    venue = (src["venue"] if src else "") or ""
    year = (src["year"] if src else "") or ""

    claim_rows = con.execute("""
        SELECT claim_id, claim_type, verbatim_quote, location_in_paper, data
          FROM claims
         WHERE source_doi = ?
           AND extraction_version IN ('deep_v1','deep_v1_abstract')
         ORDER BY (CASE confidence WHEN 'high' THEN 0
                                  WHEN 'medium' THEN 1
                                  ELSE 2 END), claim_id
         LIMIT ?
    """, (doi, MAX_CLAIMS_PER_PAPER)).fetchall()

    lines: list[str] = []
    for i, r in enumerate(claim_rows, 1):
        try:
            d = json.loads(r["data"]) or {}
        except json.JSONDecodeError:
            d = {}
        quote = (r["verbatim_quote"] or "").strip().replace("\n", " ")
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS - 1] + "…"
        ctype = r["claim_type"] or ""
        loc = (r["location_in_paper"] or "").strip().replace("\n", " ")
        # Pull a few high-signal typed fields if they exist.
        bits = []
        for k in ("reaction_type", "subject", "property_name", "value", "unit",
                  "measurement_method", "process_described", "technique_name",
                  "what_it_achieves", "key_innovation", "comparison_result",
                  "metric"):
            v = d.get(k)
            if v:
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v if x)[:120]
                else:
                    v = str(v)[:160]
                if v:
                    bits.append(f"{k}={v}")
        bits_str = "  ".join(bits)
        lines.append(f"  {i}. [{ctype}]  loc={loc}  {bits_str}\n     quote=\"{quote}\"")

    return {
        "doi": doi,
        "paper_title": title,
        "venue": venue,
        "year": year,
        "n_claims": len(claim_rows),
        "claims_block": "\n".join(lines) or "(no high-confidence claims)",
    }


def build_prompt(template: str, slots: dict) -> str:
    return template.format(**{k: (v if v is not None else "") for k, v in slots.items()})


# ── Validators ───────────────────────────────────────────────────────────────


def validate_summary(parsed, expected_doi: str, claims_block: str) -> tuple[bool, str]:
    """Return (ok, reason). Sanity checks aimed at catching the most common
    failure modes — wrong doi, unbounded length, hallucinated facts.

    Some Gemini responses come back wrapped as `[ {...} ]` instead of `{...}`.
    Unwrap a single-element list to keep the validator happy.
    """
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return False, "not_a_dict"
    summary = (parsed.get("paper_summary") or "").strip()
    if not summary:
        return False, "empty_summary"
    if len(summary) > 900:
        return False, f"too_long ({len(summary)} chars)"
    bad_starts = ("this paper", "the authors", "we ", "in this study", "in this paper")
    if summary.lower().lstrip().startswith(bad_starts):
        return False, "bad_opening"
    doi_out = (parsed.get("doi") or "").strip().lower()
    # Normalize trailing punctuation/whitespace and accept duplicate-tokenized
    # DOIs (the model occasionally collapses or echoes the input verbatim).
    def _norm_doi(d: str) -> str:
        return d.strip().rstrip(". ").lower()
    if doi_out and _norm_doi(doi_out) != _norm_doi(expected_doi):
        # also tolerate "<doi> <doi> <doi>" repetitions
        toks = doi_out.split()
        if not (toks and all(_norm_doi(t) == _norm_doi(expected_doi) for t in toks)):
            return False, f"doi_mismatch ({doi_out!r} vs {expected_doi!r})"
    return True, "ok"


# ── Subcommands ──────────────────────────────────────────────────────────────


def cmd_prepare(args):
    template = PROMPT_PATH.read_text()
    con = open_db()
    dois = list_target_papers(con, limit=args.limit, only_missing=not args.refresh)
    print(f"target papers: {len(dois)}")
    if not dois:
        print("  nothing to prepare")
        return

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    custom_id_to_doi: dict[str, str] = {}

    def _gen():
        in_tok_total = 0
        for i, doi in enumerate(dois, 1):
            slots = fetch_paper_inputs(con, doi)
            prompt = build_prompt(template, slots)
            cid = f"sum_{i:07d}"
            custom_id_to_doi[cid] = doi
            in_tok_total += len(prompt) // 4
            yield make_request_line(cid, prompt, max_tokens=MAX_OUTPUT_TOKENS)
            if i % 5000 == 0:
                print(f"  prepared {i:,}/{len(dois):,}  approx_in_tokens={in_tok_total:,}", flush=True)

    files = write_chunks(PIPELINE_DIR, _gen(), chunk_prefix="summarize")
    write_manifest(PIPELINE_DIR, {
        "kind": "summarize_papers",
        "prompt_version": PROMPT_VERSION,
        "n_papers": len(dois),
        "custom_id_to_doi": custom_id_to_doi,
        "files": files,
    })
    total_mb = sum(f["size_mb"] for f in files)
    print(f"\nprepared {len(files)} chunks, total {total_mb:.1f} MB")
    print(f"manifest: {PIPELINE_DIR / 'manifest.json'}")


def cmd_submit(args):
    res = submit_all(PIPELINE_DIR)
    print(json.dumps(res, indent=2))


def cmd_status(args):
    tally = poll_all(PIPELINE_DIR)
    print("\nbatch tally:", json.dumps(tally, indent=2))


def cmd_collect(args):
    res = collect_all(PIPELINE_DIR)
    print(json.dumps(res, indent=2))


def cmd_apply(args):
    """Parse outputs and write `paper_summary` into sources."""
    manifest_path = PIPELINE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("no manifest.json")
        return
    manifest = json.loads(manifest_path.read_text())
    cid_to_doi = manifest.get("custom_id_to_doi") or {}

    rejects_path = PIPELINE_DIR / "rejects.jsonl"
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_fh = rejects_path.open("w")

    con = open_db()
    cur = con.cursor()
    now = datetime.utcnow().isoformat() + "Z"

    parsed_ok = 0
    rejected = 0
    parse_fail = 0
    for cid, parsed, raw_item in iter_output_rows(PIPELINE_DIR):
        doi = cid_to_doi.get(cid, "")
        if parsed is None:
            parse_fail += 1
            rejects_fh.write(json.dumps({"cid": cid, "doi": doi, "reason": "parse_fail"}) + "\n")
            continue
        ok, reason = validate_summary(parsed, doi, "")
        # Unwrap single-element list responses for the UPDATE step too.
        unwrapped = parsed[0] if (isinstance(parsed, list) and len(parsed) == 1
                                  and isinstance(parsed[0], dict)) else parsed
        if not ok:
            rejected += 1
            summary_for_log = unwrapped.get("paper_summary") if isinstance(unwrapped, dict) else None
            rejects_fh.write(json.dumps({"cid": cid, "doi": doi, "reason": reason,
                                         "summary": summary_for_log}) + "\n")
            continue
        cur.execute("""
            UPDATE sources
               SET paper_summary = ?,
                   paper_summary_model = ?,
                   paper_summary_version = ?,
                   paper_summary_extracted_at = ?
             WHERE doi = ?
        """, (unwrapped["paper_summary"].strip(), MODEL, PROMPT_VERSION, now, doi))
        parsed_ok += 1
        if parsed_ok % 5000 == 0:
            con.commit()
            print(f"  applied {parsed_ok:,} so far", flush=True)
    con.commit()
    rejects_fh.close()

    print(f"\napply complete:")
    print(f"  ok:          {parsed_ok:,}")
    print(f"  rejected:    {rejected:,}  (see {rejects_path})")
    print(f"  parse_fail:  {parse_fail:,}")


def cmd_dryrun(args):
    """Synchronously summarize N papers and pretty-print results."""
    template = PROMPT_PATH.read_text()
    con = open_db()
    dois = list_target_papers(con, limit=args.limit, only_missing=not args.refresh)
    if args.seed is not None:
        random.Random(args.seed).shuffle(dois)
    dois = dois[:args.limit]
    print(f"dry-running {len(dois)} papers via SYNC Gemini ({MODEL})")

    in_tok = 0
    out_tok = 0
    for i, doi in enumerate(dois, 1):
        slots = fetch_paper_inputs(con, doi)
        prompt = build_prompt(template, slots)
        try:
            t0 = time.time()
            res = call_sync(prompt, max_tokens=MAX_OUTPUT_TOKENS)
            dt = time.time() - t0
        except Exception as e:
            print(f"  [{i}] {doi}  ERROR  {e}")
            continue
        u = res.get("usage") or {}
        pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        in_tok += pin
        out_tok += pout
        ok, reason = validate_summary(res["parsed"], doi, slots["claims_block"])
        summary = (res["parsed"].get("paper_summary") or "").strip()
        print(f"\n[{i}/{len(dois)}] {doi}  ({dt:.1f}s, in={pin}, out={pout}, validate={reason})")
        print(f"  title:   {slots['paper_title'][:100]}")
        print(f"  summary: {summary}")

    sync_cost = in_tok / 1e6 * 2.0 + out_tok / 1e6 * 12.0
    batch_cost = in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M
    print(f"\ntoken totals: in={in_tok:,}  out={out_tok:,}")
    print(f"  sync cost  (this dry-run actually paid): ${sync_cost:.4f}")
    print(f"  batch cost (for the same volume in a real submit):    ${batch_cost:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Sprint 0 — paper_summary backfill")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Build JSONL chunks for Vertex Batch")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--refresh", action="store_true",
                   help="Re-prepare even for papers that already have paper_summary")

    sub.add_parser("submit", help="Upload + create batches")
    sub.add_parser("status", help="Poll Vertex for batch progress")
    sub.add_parser("collect", help="Download completed outputs")
    sub.add_parser("apply", help="Parse outputs and UPDATE sources.paper_summary")

    p = sub.add_parser("dryrun", help="Synchronously summarize N papers (no DB writeback)")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--refresh", action="store_true")

    args = ap.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status,
     "collect": cmd_collect, "apply": cmd_apply, "dryrun": cmd_dryrun}[args.cmd](args)


if __name__ == "__main__":
    main()
