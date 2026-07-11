"""Sprint 1 — generate `claim_contextualized` for every deep_v1 (full-paper)
claim. The rewrite makes a claim independently understandable outside the
paper, using only the claim's typed fields, verbatim quote, location, and the
paper-level `paper_summary` produced by Sprint 0.

Scope (per Cho + Bing decision): only `extraction_version='deep_v1'` claims.
Abstract-only claims are NOT contextualized — there is nothing extra to add to
a claim that itself was distilled from a 250-word abstract.

Pipeline (mirrors scripts/summarize_papers.py):

    PYTHONPATH=src python3 scripts/contextualize_claims.py prepare \\
        --min-citations 1000 --limit 25000
        Build JSONL chunks for the Vertex Batch API.

    PYTHONPATH=src python3 scripts/contextualize_claims.py submit
    PYTHONPATH=src python3 scripts/contextualize_claims.py status
    PYTHONPATH=src python3 scripts/contextualize_claims.py collect
    PYTHONPATH=src python3 scripts/contextualize_claims.py apply

    PYTHONPATH=src python3 scripts/contextualize_claims.py dryrun \\
        --limit 20 --seed 42
        Synchronously rewrite N claims and print results.

Cost reference (Gemini 3.1 Pro Preview Batch, calibrated for reasoning tokens):
    Per request (8 claims): ~3,000 input + ~1,800 output (incl. reasoning)
    Per claim:              ~375 in + ~200 out  →  ~$0.0016 / claim batch
    Sprint 1d (1.52M): ~$2,400.

Resumability:
    `apply` skips claims whose `claim_contextualized` is already non-NULL with
    the matching `context_version`. Re-running prepare with --resume picks up
    where a prior run left off (the prepare step already filters out claims
    that already have a value).
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
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
PROMPT_PATH = REPO_ROOT / "src" / "askchem" / "prompts" / f"contextualize_{PROMPT_VERSION}.txt"

# We package multiple claims per batch request to amortize the prompt header.
# 8 is the sweet spot in the plan: prompt header (~150 tok) shared across 8
# claims drops effective input cost by ~25%, and the model is happy with up to
# ~10 short claims per request before quality starts to degrade.
CLAIMS_PER_REQUEST = 8   # 8 claims per request: prompt header is amortized
                         # AND (more importantly) the model burns one
                         # reasoning trace per request, not per claim.
                         # Drops effective cost ~5–7x vs 1 claim/request.
                         # Override at the CLI with --claims-per-request
                         # for retry runs where Pro was dropping claims.

MAX_QUOTE_CHARS = 600
MAX_OUTPUT_TOKENS = 8192     # generous — Gemini 3.1 burns reasoning tokens
PIPELINE_DIR_TPL = REPO_ROOT / "data" / "batch_jobs" / "contextualize_{tag}_{version}"

# typed-field keys we keep in the LLM input (drop noise that's already in
# other slots or unhelpful for rewriting).
TYPED_FIELD_KEYS = (
    "reaction_type", "reactants", "products", "conditions", "outcomes",
    "subject", "subject_smiles", "property_name", "property_category",
    "value", "unit", "measurement_method",
    "process_described", "steps", "key_intermediates",
    "technique_name", "what_it_achieves", "key_innovation", "limitations",
    "compared_items", "metric", "comparison_result",
    "hypothesis_text", "limitation_text", "direction_text",
    "finding_text", "why_surprising",
    "rationale", "evidence", "assumption", "epistemic_role",
    "is_key_result", "confidence",
)


# ── DB helpers ───────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=60.0)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA busy_timeout = 30000")
    con.row_factory = sqlite3.Row
    return con


def list_target_claims(con: sqlite3.Connection, *,
                       min_citations: int = 0,
                       limit: int | None = None,
                       order_by_citations: bool = True) -> list[dict]:
    """Return list of {claim_id, ...} dicts that need contextualization.

    For small smoke runs (e.g. Sprint 1a, top-cited only) we ORDER BY
    citation_count DESC. For the full 1.5M-claim residual, sorting the whole
    result set is prohibitively expensive on a 11GB SQLite DB, so we skip
    the ORDER BY (set `order_by_citations=False`).
    """
    if order_by_citations:
        sql = f"""
            SELECT c.claim_id,
                   c.claim_type,
                   c.verbatim_quote,
                   c.source_paper_title,
                   c.location_in_paper,
                   c.source_doi,
                   c.data,
                   COALESCE(s.citation_count, 0) AS citation_count,
                   s.paper_summary
              FROM claims c
              JOIN sources s ON c.source_doi = s.doi
             WHERE c.extraction_version IN ('deep_v1','deep_v1_abstract')
               AND c.claim_contextualized IS NULL
               AND COALESCE(s.citation_count, 0) >= ?
             ORDER BY COALESCE(s.citation_count, 0) DESC, c.claim_id
             LIMIT ?
        """
    else:
        sql = f"""
            SELECT c.claim_id,
                   c.claim_type,
                   c.verbatim_quote,
                   c.source_paper_title,
                   c.location_in_paper,
                   c.source_doi,
                   c.data,
                   COALESCE(s.citation_count, 0) AS citation_count,
                   s.paper_summary
              FROM claims c
              JOIN sources s ON c.source_doi = s.doi
             WHERE c.extraction_version IN ('deep_v1','deep_v1_abstract')
               AND c.claim_contextualized IS NULL
               AND COALESCE(s.citation_count, 0) >= ?
             LIMIT ?
        """
    rows = con.execute(sql, (min_citations, limit if limit is not None else -1)).fetchall()
    return [dict(r) for r in rows]


def iter_target_claims(con: sqlite3.Connection, *,
                       min_citations: int = 0,
                       limit: int | None = None,
                       order_by_citations: bool = False):
    """Streaming version of `list_target_claims` — yields one dict at a time.

    Uses a forward cursor (no fetchall) so the 1.5M-row residual fits in
    constant memory and `prepare` can start emitting JSONL chunks
    immediately without waiting for the full sort.
    """
    if order_by_citations:
        sql = """
            SELECT c.claim_id, c.claim_type, c.verbatim_quote,
                   c.source_paper_title, c.location_in_paper, c.source_doi,
                   c.data, COALESCE(s.citation_count, 0) AS citation_count,
                   s.paper_summary
              FROM claims c
              JOIN sources s ON c.source_doi = s.doi
             WHERE c.extraction_version IN ('deep_v1','deep_v1_abstract')
               AND c.claim_contextualized IS NULL
               AND COALESCE(s.citation_count, 0) >= ?
             ORDER BY COALESCE(s.citation_count, 0) DESC, c.claim_id
             LIMIT ?
        """
    else:
        sql = """
            SELECT c.claim_id, c.claim_type, c.verbatim_quote,
                   c.source_paper_title, c.location_in_paper, c.source_doi,
                   c.data, COALESCE(s.citation_count, 0) AS citation_count,
                   s.paper_summary
              FROM claims c
              JOIN sources s ON c.source_doi = s.doi
             WHERE c.extraction_version IN ('deep_v1','deep_v1_abstract')
               AND c.claim_contextualized IS NULL
               AND COALESCE(s.citation_count, 0) >= ?
             LIMIT ?
        """
    cursor = con.execute(sql, (min_citations, limit if limit is not None else -1))
    n = 0
    for r in cursor:
        yield dict(r)
        n += 1
    cursor.close()


def compact_typed_fields(data_blob: str) -> dict:
    try:
        d = json.loads(data_blob) or {}
    except json.JSONDecodeError:
        return {}
    out = {}
    for k in TYPED_FIELD_KEYS:
        v = d.get(k)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, str) and len(v) > 400:
            v = v[:400]
        out[k] = v
    return out


def _claim_payload(row: dict) -> dict:
    """Compact dict the LLM sees for a single claim."""
    quote = (row.get("verbatim_quote") or "").strip().replace("\n", " ")
    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[:MAX_QUOTE_CHARS - 1] + "…"
    typed = compact_typed_fields(row.get("data") or "")
    summary = (row.get("paper_summary") or "").strip()
    if not summary:
        summary = ""
    return {
        "claim_id":           row["claim_id"],
        "claim_type":         row["claim_type"] or "",
        "paper_title":        (row.get("source_paper_title") or "")[:240],
        "paper_summary":      summary[:800],
        "location_in_paper":  (row.get("location_in_paper") or "")[:160],
        "verbatim_quote":     quote,
        "typed_fields":       typed,
    }


def build_prompt_for_batch(template: str, rows: list[dict]) -> str:
    """Build one prompt covering up to CLAIMS_PER_REQUEST claims."""
    payloads = [_claim_payload(r) for r in rows]
    return template.format(claims_json=json.dumps(payloads, indent=2))


# ── Validation ───────────────────────────────────────────────────────────────


_NUM_RE = None  # lazy-compiled
_COLLAPSED_EXP_RE = None
_REWRITE_EXP_RE = None

# Common exponent / reference / count literals that Pro routinely
# inserts into rewrites without them appearing as standalone tokens in
# the verbatim quote. Examples include the base "10" of scientific
# notation ("10^18"), small index numbers ("Step 3", "Figure 5"), and
# a handful of decade markers ("100", "1000") that appear in chemistry
# language but might be inside a longer collapsed string in the source.
_NUMBER_INVENTION_ALLOWLIST: frozenset[str] = frozenset({
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25",
    "100", "1000",
})


def _numbers_in(text: str) -> set[str]:
    import re
    global _NUM_RE, _COLLAPSED_EXP_RE
    if _NUM_RE is None:
        _NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
    if _COLLAPSED_EXP_RE is None:
        # Pull "10**18" out of the collapsed "1018" the source often has
        # when the original PDF dropped the superscript. We match the
        # 10-prefix + 1-3 digit suffix and add both halves so a rewrite
        # that explicitly writes "10^18" doesn't trip the validator.
        _COLLAPSED_EXP_RE = re.compile(r"(?<!\d)10(\d{1,3})(?!\d)")
    nums = {m.group(0) for m in _NUM_RE.finditer(text or "")}
    for m in _COLLAPSED_EXP_RE.finditer(text or ""):
        nums.add("10")
        nums.add(m.group(1))
    return nums


def _exponent_split_numbers_in_rewrite(text: str) -> set[str]:
    """Numbers extracted as ``10`` and ``N`` from rewrite forms like
    ``10^18`` / ``10^{18}`` / ``×10^18`` / ``10**18``. We use this to
    map back to a haystack that has the same number written collapsed
    (``1018``).
    """
    import re
    global _REWRITE_EXP_RE
    if _REWRITE_EXP_RE is None:
        _REWRITE_EXP_RE = re.compile(r"10\s*(?:\^|\*\*)\s*\{?(\d{1,3})\}?")
    out: set[str] = set()
    for m in _REWRITE_EXP_RE.finditer(text or ""):
        out.add("10" + m.group(1))
    return out


def validate_one_rewrite(rewrite: str, *,
                         verbatim_quote: str,
                         typed_fields_json: str,
                         paper_title: str = "",
                         paper_summary: str = "",
                         location_in_paper: str = "",
                         claim_type: str = "") -> tuple[bool, str]:
    """Per-rewrite validation. Caller is responsible for matching claim_id.

    Number-invention check considers ALL inputs the model saw — verbatim
    quote, typed fields, paper title, paper summary, location, and claim
    type — because chemical formula subscripts like "Cr2Ge2Te6" frequently
    appear in the title/summary, not the verbatim quote.
    """
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return False, "empty_rewrite"
    if len(rewrite) > 400:
        return False, f"too_long ({len(rewrite)} chars)"
    bad_starts = ("the paper", "this study", "this paper", "we ", "the authors",
                  "it is reported", "in this study")
    if rewrite.lower().lstrip().startswith(bad_starts):
        return False, "bad_opening"
    rewrite_nums = _numbers_in(rewrite)
    if rewrite_nums:
        haystack = "\n".join([
            verbatim_quote, typed_fields_json,
            paper_title, paper_summary, location_in_paper, claim_type,
        ])
        haystack_nums = _numbers_in(haystack)
        # When the rewrite spells out a scientific-notation exponent
        # ("10^18"), allow the collapsed form ("1018") in the haystack
        # to satisfy the check.
        haystack_nums = haystack_nums | _exponent_split_numbers_in_rewrite(rewrite)
        invented = rewrite_nums - haystack_nums - _NUMBER_INVENTION_ALLOWLIST
        if invented:
            return False, f"invented_numbers ({sorted(invented)[:3]})"
    return True, "ok"


def parse_batch_response(parsed: dict) -> dict[str, str]:
    """Pull {"results": [{claim_id, claim_contextualized}, ...]} into a map."""
    if not isinstance(parsed, dict):
        return {}
    results = parsed.get("results") or []
    if not isinstance(results, list):
        return {}
    out: dict[str, str] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        cid = (item.get("claim_id") or "").strip()
        rewrite = (item.get("claim_contextualized") or "").strip()
        if cid:
            out[cid] = rewrite
    return out


# ── Subcommands ──────────────────────────────────────────────────────────────


def _pipeline_dir(tag: str) -> Path:
    return Path(str(PIPELINE_DIR_TPL).format(tag=tag, version=PROMPT_VERSION))


def _chunked(seq: list[dict], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def cmd_prepare(args):
    template = PROMPT_PATH.read_text()
    con = open_db()
    pdir = _pipeline_dir(args.tag)
    pdir.mkdir(parents=True, exist_ok=True)

    claims_per_request = args.claims_per_request or CLAIMS_PER_REQUEST

    cid_to_claim_ids: dict[str, list[str]] = {}
    n_claims = [0]
    n_batches = [0]

    # For full-corpus residual runs (no min_citations filter) we stream
    # without ORDER BY — sorting 1.5M rows on a 11GB DB is prohibitively
    # expensive and order doesn't matter for prepare.
    use_order_by = args.order_by_citations
    print(f"streaming target claims  (min_citations={args.min_citations}, "
          f"limit={args.limit}, order_by_citations={use_order_by}, "
          f"claims_per_request={claims_per_request})", flush=True)

    def _gen():
        batch: list[dict] = []
        for r in iter_target_claims(con, min_citations=args.min_citations,
                                     limit=args.limit,
                                     order_by_citations=use_order_by):
            batch.append(r)
            n_claims[0] += 1
            if len(batch) >= claims_per_request:
                n_batches[0] += 1
                cid = f"ctx_{n_batches[0]:07d}"
                cid_to_claim_ids[cid] = [b["claim_id"] for b in batch]
                yield make_request_line(cid, build_prompt_for_batch(template, batch),
                                        max_tokens=MAX_OUTPUT_TOKENS)
                if n_batches[0] % 5000 == 0:
                    print(f"  prepared {n_batches[0]:,} requests "
                          f"({n_claims[0]:,} claims so far)", flush=True)
                batch = []
        if batch:
            n_batches[0] += 1
            cid = f"ctx_{n_batches[0]:07d}"
            cid_to_claim_ids[cid] = [b["claim_id"] for b in batch]
            yield make_request_line(cid, build_prompt_for_batch(template, batch),
                                    max_tokens=MAX_OUTPUT_TOKENS)

    files = write_chunks(pdir, _gen(), chunk_prefix="contextualize")
    if n_claims[0] == 0:
        print("  nothing to prepare")
        return
    write_manifest(pdir, {
        "kind": "contextualize_claims",
        "prompt_version": PROMPT_VERSION,
        "tag": args.tag,
        "min_citations": args.min_citations,
        "limit": args.limit,
        "order_by_citations": use_order_by,
        "n_claims": n_claims[0],
        "n_requests": n_batches[0],
        "claims_per_request": claims_per_request,
        "custom_id_to_claim_ids": cid_to_claim_ids,
        "files": files,
    })
    total_mb = sum(f["size_mb"] for f in files)
    print(f"\nprepared {len(files)} chunks ({n_batches[0]:,} requests, "
          f"{n_claims[0]:,} claims), total {total_mb:.1f} MB")
    print(f"manifest: {pdir / 'manifest.json'}")


def cmd_submit(args):
    pdir = _pipeline_dir(args.tag)
    res = submit_all(pdir)
    print(json.dumps(res, indent=2))


def cmd_status(args):
    pdir = _pipeline_dir(args.tag)
    tally = poll_all(pdir)
    print("\nbatch tally:", json.dumps(tally, indent=2))


def cmd_collect(args):
    pdir = _pipeline_dir(args.tag)
    res = collect_all(pdir)
    print(json.dumps(res, indent=2))


def cmd_apply(args):
    pdir = _pipeline_dir(args.tag)
    manifest_path = pdir / "manifest.json"
    if not manifest_path.exists():
        print("no manifest.json")
        return
    manifest = json.loads(manifest_path.read_text())
    cid_to_claim_ids: dict[str, list[str]] = manifest.get("custom_id_to_claim_ids") or {}

    rejects_path = pdir / "rejects.jsonl"
    rejects_fh = rejects_path.open("w")

    con = open_db()
    cur = con.cursor()
    now = datetime.utcnow().isoformat() + "Z"

    parsed_ok = 0
    rejected = 0
    parse_fail = 0
    request_parse_fail = 0
    for cid, parsed, raw_item in iter_output_rows(pdir):
        expected_claim_ids = cid_to_claim_ids.get(cid) or []
        if parsed is None:
            request_parse_fail += 1
            for claim_id in expected_claim_ids:
                parse_fail += 1
                rejects_fh.write(json.dumps({"cid": cid, "claim_id": claim_id,
                                             "reason": "request_parse_fail"}) + "\n")
            continue
        rewrites = parse_batch_response(parsed)
        for claim_id in expected_claim_ids:
            rewrite = rewrites.get(claim_id, "").strip()
            if not rewrite:
                rejected += 1
                rejects_fh.write(json.dumps({"cid": cid, "claim_id": claim_id,
                                             "reason": "missing_in_response"}) + "\n")
                continue
            row = cur.execute(
                """SELECT c.verbatim_quote, c.data, c.source_paper_title,
                          c.location_in_paper, c.claim_type, s.paper_summary
                     FROM claims c
                     LEFT JOIN sources s ON c.source_doi = s.doi
                    WHERE c.claim_id=?""",
                (claim_id,)
            ).fetchone()
            if not row:
                rejected += 1
                rejects_fh.write(json.dumps({"cid": cid, "claim_id": claim_id,
                                             "reason": "claim_not_found"}) + "\n")
                continue
            typed_json = json.dumps(compact_typed_fields(row[1] or ""), separators=(",", ":"))
            ok, reason = validate_one_rewrite(
                rewrite,
                verbatim_quote=row[0] or "",
                typed_fields_json=typed_json,
                paper_title=row[2] or "",
                location_in_paper=row[3] or "",
                claim_type=row[4] or "",
                paper_summary=row[5] or "",
            )
            if not ok:
                rejected += 1
                rejects_fh.write(json.dumps({"cid": cid, "claim_id": claim_id,
                                             "reason": reason,
                                             "rewrite": rewrite}) + "\n")
                continue
            cur.execute("""
                UPDATE claims
                   SET claim_contextualized = ?,
                       context_model = ?,
                       context_version = ?,
                       context_extracted_at = ?
                 WHERE claim_id = ?
            """, (rewrite, MODEL, PROMPT_VERSION, now, claim_id))
            parsed_ok += 1
            if parsed_ok % 5000 == 0:
                con.commit()
                print(f"  applied {parsed_ok:,} so far", flush=True)
    con.commit()
    rejects_fh.close()
    print(f"\napply complete:")
    print(f"  ok:                  {parsed_ok:,}")
    print(f"  rejected:            {rejected:,}")
    print(f"  parse_fail (claims): {parse_fail:,}  (from {request_parse_fail} unparseable requests)")
    print(f"  rejects log: {rejects_path}")


def cmd_dryrun(args):
    template = PROMPT_PATH.read_text()
    con = open_db()
    rows = list_target_claims(con, min_citations=args.min_citations, limit=400)
    if args.seed is not None:
        random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    print(f"dry-running {len(rows)} claims via SYNC Gemini ({MODEL}); "
          f"batched {CLAIMS_PER_REQUEST}/request")

    in_tok, out_tok = 0, 0
    n_ok, n_bad = 0, 0
    n_requests = 0
    for batch_idx, batch in enumerate(_chunked(rows, CLAIMS_PER_REQUEST), 1):
        prompt = build_prompt_for_batch(template, batch)
        try:
            t0 = time.time()
            res = call_sync(prompt, max_tokens=MAX_OUTPUT_TOKENS)
            dt = time.time() - t0
        except Exception as e:
            print(f"  [batch {batch_idx}] ERROR  {e}")
            continue
        u = res.get("usage") or {}
        pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        in_tok += pin
        out_tok += pout
        n_requests += 1
        rewrites = parse_batch_response(res["parsed"])
        print(f"\n--- batch {batch_idx} ({len(batch)} claims, {dt:.1f}s, in={pin}, out={pout}) ---")
        for r in batch:
            cid = r["claim_id"]
            rewrite = rewrites.get(cid, "")
            typed = compact_typed_fields(r.get("data") or "")
            ok, reason = validate_one_rewrite(
                rewrite,
                verbatim_quote=r.get("verbatim_quote") or "",
                typed_fields_json=json.dumps(typed, separators=(",", ":")),
                paper_title=r.get("source_paper_title") or "",
                paper_summary=r.get("paper_summary") or "",
                location_in_paper=r.get("location_in_paper") or "",
                claim_type=r.get("claim_type") or "",
            )
            n_ok += int(ok)
            n_bad += int(not ok)
            print(f"  [cit={r['citation_count']} type={r['claim_type']} validate={reason}]")
            print(f"    quote:   {(r.get('verbatim_quote') or '')[:160]}")
            print(f"    rewrite: {rewrite}")

    sync_cost = in_tok / 1e6 * 2.0 + out_tok / 1e6 * 12.0
    batch_cost = in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M
    n_total = len(rows)
    print(f"\ntotals: {n_requests} requests, {n_total} claims, "
          f"in={in_tok:,}  out={out_tok:,}  validate {n_ok}/{n_total} ok")
    print(f"  per-claim batch cost: ${batch_cost / max(n_total, 1):.5f}")
    print(f"  extrapolated to 1.52M claims (Sprint 1d): "
          f"${batch_cost / max(n_total, 1) * 1_515_538:,.0f}")
    print(f"  sync cost  (this dry-run actually paid): ${sync_cost:.4f}")
    print(f"  batch cost (same volume, batched submit): ${batch_cost:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Sprint 1 — claim_contextualized backfill")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Build JSONL chunks for Vertex Batch")
    p.add_argument("--tag", required=True, help="Sprint sub-stage tag, e.g. '1a'")
    p.add_argument("--min-citations", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--order-by-citations", action="store_true",
                   help="Order claims by citation_count DESC (skip for residual "
                        "1.5M-claim runs — full-table sort is too expensive)")
    p.add_argument("--claims-per-request", type=int, default=None,
                   help="Override CLAIMS_PER_REQUEST (default 8). Use 4 for "
                        "retry runs where Pro silently dropped claims from "
                        "8-claim batches.")

    p = sub.add_parser("submit", help="Upload + create batches")
    p.add_argument("--tag", required=True)

    p = sub.add_parser("status", help="Poll Vertex for batch progress")
    p.add_argument("--tag", required=True)

    p = sub.add_parser("collect", help="Download completed outputs")
    p.add_argument("--tag", required=True)

    p = sub.add_parser("apply", help="Parse outputs and UPDATE claims.claim_contextualized")
    p.add_argument("--tag", required=True)

    p = sub.add_parser("dryrun", help="Synchronously rewrite N claims (no DB writeback)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--min-citations", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status,
     "collect": cmd_collect, "apply": cmd_apply, "dryrun": cmd_dryrun}[args.cmd](args)


if __name__ == "__main__":
    main()
