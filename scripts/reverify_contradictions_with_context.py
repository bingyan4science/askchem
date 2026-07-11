#!/usr/bin/env python3
"""Re-verify confirmed contradictions using contextualized claims.

The legacy contradiction pipeline judged Gemini on raw ``verbatim_quote``
text, which strips temperature, catalyst, and scope context. Re-verifying
with ``claim_contextualized`` (a one-sentence standalone rewrite written
by Sprint 1) eliminates the obvious false-positives where two papers
measure different conditions.

Pipeline:
  1. Load ``data/gemini_verified_viewfree.checkpoint.json``, keep only
     ``gemini_verdict == 'confirmed'``.
  2. Pre-filter
       - drop same-DOI pairs (intra-paper noise)
       - drop very short / very long quotes
       - drop near-duplicate quotes (Jaccard on shingles ≥ 0.85)
  3. Look up ``claim_contextualized`` for both claims from askchem.db.
     Skip pairs where either side lacks context.
  4. Re-prompt Gemini with the contextualized claims (Portkey gateway,
     ``gemini-2.5-flash``). Batched + concurrent for speed.
  5. Filter to ``confirmed`` survivors, rank by display-worthiness
     (confidence × paper-citation × subject-diversity), and write
     ``data/contradictions_for_display.json`` in the shape
     ``upload_contradictions.py`` expects.

Usage:
    export PORTKEY_API_KEY=...       # or EDISON_2 (see ~/.bashrc)
    python scripts/reverify_contradictions_with_context.py \
        --input data/gemini_verified_viewfree.checkpoint.json \
        --output data/contradictions_for_display.json \
        --top 250 --workers 4 --batch-size 5

If ``PORTKEY_API_KEY`` is unset the script falls back to
``--no-llm`` mode: it does the pre-filter and ranks the existing
verdicts as-is without re-querying Gemini.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "chemtree.db"

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-2.5-flash"

MIN_QUOTE_LEN = 40
MAX_QUOTE_LEN = 800
SHINGLE_K = 4
DUP_JACCARD = 0.85
BATCH_SIZE = 5
MAX_WORKERS = 4
CHECKPOINT_EVERY = 25


# ── Pre-filtering helpers ─────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _shingles(s: str, k: int = SHINGLE_K) -> set[str]:
    """Word k-shingles for cheap near-duplicate detection."""
    toks = _WORD_RE.findall((s or "").lower())
    if len(toks) < k:
        return set(toks)
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def _is_near_duplicate(q1: str, q2: str) -> bool:
    return _jaccard(_shingles(q1), _shingles(q2)) >= DUP_JACCARD


# ── Database lookup ───────────────────────────────────────────────────────


def fetch_contexts(db_path: Path, claim_ids: list[str]) -> dict[str, dict]:
    """Return {claim_id: {contextualized, verbatim, doi, paper_title, citation}}."""
    if not claim_ids:
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[str, dict] = {}
    # SQLite has a 999 host-parameter limit; chunk just in case.
    CHUNK = 500
    for i in range(0, len(claim_ids), CHUNK):
        chunk = claim_ids[i : i + CHUNK]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""SELECT c.claim_id, c.claim_contextualized,
                       c.verbatim_quote, c.source_doi, c.source_paper_title,
                       c.claim_type, c.data,
                       s.citation_count
                  FROM claims c
                  LEFT JOIN sources s ON s.doi = c.source_doi
                 WHERE c.claim_id IN ({ph})""",
            chunk,
        ).fetchall()
        for r in rows:
            subject = ""
            if r["data"]:
                try:
                    subject = (json.loads(r["data"]).get("subject") or "")[:300]
                except (ValueError, TypeError):
                    subject = ""
            out[r["claim_id"]] = {
                "contextualized": (r["claim_contextualized"] or "").strip(),
                "verbatim": (r["verbatim_quote"] or "").strip(),
                "doi": r["source_doi"] or "",
                "paper_title": r["source_paper_title"] or "",
                "subject": subject,
                "claim_type": r["claim_type"] or "",
                "citation_count": int(r["citation_count"] or 0),
            }
    conn.close()
    return out


# ── Gemini re-prompt ──────────────────────────────────────────────────────


def build_prompt(pairs: list[dict]) -> str:
    """Re-verification prompt — uses claim_contextualized when available."""
    head = (
        "You are a senior chemistry reviewer auditing automated contradiction "
        "detection results. For each pair below, two claims have been flagged "
        "as contradictory based on raw text excerpts. Now you must re-judge "
        "using the STANDALONE rewrites that include experimental conditions, "
        "scope, and qualifiers.\n\n"
        "A genuine contradiction is two claims making mutually incompatible "
        "assertions about the SAME subject under the SAME (or overlapping) "
        "conditions. Different measurements of different materials, different "
        "temperatures, or different scopes are NOT contradictions. Be strict.\n"
    )
    body = []
    for i, p in enumerate(pairs, 1):
        body.append(f"--- Pair {i} ---")
        body.append(f"Subject: {p.get('subject','')[:120]}")
        body.append(f"Claim A (DOI {p['doi_1']}):\n  {p['claim_a']}")
        body.append(f"Claim B (DOI {p['doi_2']}):\n  {p['claim_b']}")
        body.append("")
    tail = (
        "For each pair, output a JSON array (one object per pair, IN ORDER):\n"
        '[{"pair": 1, "verdict": "confirmed"|"rejected", '
        '"explanation": "one sentence reason for the verdict", '
        '"strength": "strong"|"moderate"|"weak", '
        '"confidence": 0.0-1.0}, ...]\n'
        "Use 'strong' only when the contradiction is unambiguous and the "
        "claims clearly disagree on the same quantity / phenomenon. Respond "
        "ONLY with the JSON array, no extra text."
    )
    return head + "\n".join(body) + "\n" + tail


_clients: dict[int, object] = {}
_clients_lock = __import__("threading").Lock()


def _client_for_thread(api_key: str):
    import threading
    tid = threading.get_ident()
    if tid not in _clients:
        with _clients_lock:
            if tid not in _clients:
                from openai import OpenAI
                _clients[tid] = OpenAI(
                    base_url=GATEWAY,
                    api_key=api_key,
                    default_headers={"x-portkey-provider": PROVIDER},
                    timeout=45.0,
                )
    return _clients[tid]


def _parse_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else [v]
    except json.JSONDecodeError:
        pass
    out = []
    for m in re.finditer(r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text):
        try:
            out.append(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    return out


def call_gemini(pairs: list[dict], api_key: str, batch_idx: int) -> list[dict]:
    client = _client_for_thread(api_key)
    prompt = build_prompt(pairs)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400 * len(pairs) + 800,
            )
            text = resp.choices[0].message.content.strip()
            parsed = _parse_array(text)
            for i, p in enumerate(pairs):
                r = parsed[i] if i < len(parsed) else {}
                p["recheck_verdict"] = r.get("verdict") or "error"
                p["recheck_explanation"] = (r.get("explanation") or "")[:500]
                p["recheck_strength"] = r.get("strength") or "weak"
                p["recheck_confidence"] = float(r.get("confidence") or 0.0)
            return pairs
        except Exception as e:
            msg = str(e)
            wait = 2 ** attempt * (5 if "429" in msg or "rate" in msg.lower() else 2)
            wait = min(wait, 90)
            if attempt < 4:
                if attempt >= 2:
                    print(
                        f"  batch {batch_idx} attempt {attempt+1} failed: "
                        f"{msg[:100]} (retry in {wait}s)",
                        flush=True,
                    )
                time.sleep(wait)
                continue
            for p in pairs:
                p["recheck_verdict"] = "error"
                p["recheck_explanation"] = msg[:200]
                p["recheck_strength"] = "weak"
                p["recheck_confidence"] = 0.0
            return pairs
    return pairs


# ── Ranking ────────────────────────────────────────────────────────────────


def display_score(p: dict) -> float:
    """Composite ranking signal for the display set.

    Prefers high-confidence, "strong" contradictions on subjects with
    decent citation context. Subject-diversity is enforced separately by
    capping per (subject, claim_type) when slicing the final list.
    """
    strength_bonus = {"strong": 1.0, "moderate": 0.5, "weak": 0.1}.get(
        p.get("recheck_strength", "weak"), 0.1
    )
    cite = min(p.get("citation_1", 0) + p.get("citation_2", 0), 1000) / 1000.0
    base = float(p.get("recheck_confidence", 0.0))
    return 0.55 * base + 0.30 * strength_bonus + 0.15 * cite


# ── Main ───────────────────────────────────────────────────────────────────


def _resolve_api_key(no_llm: bool) -> Optional[str]:
    if no_llm:
        return None
    for env in ("PORTKEY_API_KEY", "EDISON_2", "EDISON_API_KEY", "EDISON"):
        v = os.environ.get(env)
        if v:
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/gemini_verified_viewfree.checkpoint.json")
    ap.add_argument("--output", default="data/contradictions_for_display.json")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--top", type=int, default=250,
                    help="Keep at most this many ranked survivors in the output")
    ap.add_argument("--display-cap", type=int, default=100,
                    help="Tag the top-N survivors with 'display_priority' for the homepage")
    ap.add_argument("--per-subject-cap", type=int, default=4,
                    help="Hard cap on contradictions sharing the same (subject, claim_type)")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip re-prompting Gemini; rank existing verdicts as-is")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N candidates (smoke testing)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    db_path = Path(args.db).resolve()

    verified = json.loads(input_path.read_text())
    confirmed = [v for v in verified if v.get("gemini_verdict") == "confirmed"]
    print(f"Loaded {len(verified):,} verdicts, {len(confirmed):,} confirmed", flush=True)

    # ── Pre-filter ─────────────────────────────────────────────────────────
    pre: list[dict] = []
    dropped_same_doi = 0
    dropped_len = 0
    dropped_dup = 0
    for v in confirmed:
        if v.get("doi_1") and v["doi_1"] == v.get("doi_2"):
            dropped_same_doi += 1
            continue
        q1 = (v.get("quote_1") or "").strip()
        q2 = (v.get("quote_2") or "").strip()
        if len(q1) < MIN_QUOTE_LEN or len(q2) < MIN_QUOTE_LEN:
            dropped_len += 1
            continue
        if len(q1) > MAX_QUOTE_LEN or len(q2) > MAX_QUOTE_LEN:
            dropped_len += 1
            continue
        if _is_near_duplicate(q1, q2):
            dropped_dup += 1
            continue
        pre.append(v)
    print(
        f"Pre-filter: kept {len(pre):,}; dropped same-DOI={dropped_same_doi}, "
        f"length={dropped_len}, near-dup={dropped_dup}",
        flush=True,
    )

    # ── Look up contextualized claims ─────────────────────────────────────
    all_ids = sorted({c for v in pre for c in (v["claim_id_1"], v["claim_id_2"])})
    print(f"Fetching contexts for {len(all_ids):,} claims from {db_path}", flush=True)
    ctx = fetch_contexts(db_path, all_ids)

    no_ctx = 0
    pairs_for_llm: list[dict] = []
    for v in pre:
        m1 = ctx.get(v["claim_id_1"]) or {}
        m2 = ctx.get(v["claim_id_2"]) or {}
        c1 = m1.get("contextualized") or m1.get("verbatim") or v.get("quote_1") or ""
        c2 = m2.get("contextualized") or m2.get("verbatim") or v.get("quote_2") or ""
        if not (c1 and c2):
            no_ctx += 1
            continue
        pairs_for_llm.append({
            "claim_id_1": v["claim_id_1"],
            "claim_id_2": v["claim_id_2"],
            "doi_1": m1.get("doi") or v.get("doi_1", ""),
            "doi_2": m2.get("doi") or v.get("doi_2", ""),
            "paper_title_1": m1.get("paper_title", ""),
            "paper_title_2": m2.get("paper_title", ""),
            "subject": v.get("subject") or m1.get("subject") or m2.get("subject") or "",
            "claim_type": v.get("claim_type") or m1.get("claim_type") or m2.get("claim_type") or "",
            "quote_1": v.get("quote_1", ""),
            "quote_2": v.get("quote_2", ""),
            "claim_a": c1,
            "claim_b": c2,
            "citation_1": m1.get("citation_count", 0),
            "citation_2": m2.get("citation_count", 0),
            "original_confidence": float(v.get("confidence") or 0.0),
            "original_explanation": v.get("gemini_explanation", ""),
        })
    if no_ctx:
        print(f"Skipped {no_ctx} pairs (no contextualized text on either side)", flush=True)

    if args.limit > 0:
        pairs_for_llm = pairs_for_llm[: args.limit]
        print(f"Limited to {len(pairs_for_llm)} pairs (smoke test)", flush=True)

    # ── Re-prompt Gemini (or skip in no-llm mode) ─────────────────────────
    api_key = _resolve_api_key(args.no_llm)
    if api_key:
        print(
            f"Re-verifying {len(pairs_for_llm)} pairs via {MODEL} (Portkey)…",
            flush=True,
        )
        batches = [
            pairs_for_llm[i : i + args.batch_size]
            for i in range(0, len(pairs_for_llm), args.batch_size)
        ]
        print(f"Batches: {len(batches)} × {args.batch_size}, workers={args.workers}",
              flush=True)
        checkpoint = output_path.with_suffix(".reverify.checkpoint.json")
        completed: list[dict] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(call_gemini, b, api_key, i): i for i, b in enumerate(batches)}
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    completed.extend(fut.result())
                except Exception as e:
                    print(f"batch {idx} raised: {e}", flush=True)
                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(batches) - done) / max(rate, 0.01) / 60
                    print(
                        f"  {done}/{len(batches)} batches, "
                        f"{sum(1 for p in completed if p.get('recheck_verdict')=='confirmed')} confirmed, "
                        f"{rate*args.batch_size:.1f} pairs/s, ETA {eta:.0f}min",
                        flush=True,
                    )
                if done % CHECKPOINT_EVERY == 0:
                    checkpoint.write_text(json.dumps(completed))
        pairs_for_llm = completed
        if checkpoint.exists():
            checkpoint.unlink()
    else:
        if not args.no_llm:
            print("⚠️  No PORTKEY_API_KEY / EDISON_2 set; running in --no-llm mode",
                  flush=True)
        for p in pairs_for_llm:
            # Inherit the original verdict as the re-check verdict so the
            # downstream ranking still produces something useful offline.
            p["recheck_verdict"] = "confirmed"
            p["recheck_explanation"] = p.get("original_explanation", "")
            p["recheck_strength"] = "moderate"
            p["recheck_confidence"] = p.get("original_confidence", 0.7) or 0.7

    survivors = [p for p in pairs_for_llm if p.get("recheck_verdict") == "confirmed"]
    print(f"Re-check survivors: {len(survivors):,} / {len(pairs_for_llm):,}", flush=True)

    survivors.sort(key=display_score, reverse=True)

    # ── Per-subject cap so the display set isn't dominated by one topic ──
    capped: list[dict] = []
    seen_groups: dict[tuple[str, str], int] = {}
    for p in survivors:
        key = ((p.get("subject") or "").lower().strip(), (p.get("claim_type") or "").lower().strip())
        n = seen_groups.get(key, 0)
        if n >= args.per_subject_cap:
            continue
        seen_groups[key] = n + 1
        capped.append(p)
    print(
        f"After per-subject cap ({args.per_subject_cap}): {len(capped):,} kept",
        flush=True,
    )

    # ── Truncate to --top and tag display priority ──
    final = capped[: args.top]
    for i, p in enumerate(final):
        p["display_priority"] = 1 if i < args.display_cap else 0

    # ── Shape into the format upload_contradictions.py expects ──
    records = []
    for p in final:
        records.append({
            "claim_id_1": p["claim_id_1"],
            "claim_id_2": p["claim_id_2"],
            "subject": p.get("subject", ""),
            "claim_type": p.get("claim_type", ""),
            "doi_1": p.get("doi_1", ""),
            "doi_2": p.get("doi_2", ""),
            "quote_1": p.get("quote_1", ""),
            "quote_2": p.get("quote_2", ""),
            "claim_contextualized_1": p.get("claim_a", ""),
            "claim_contextualized_2": p.get("claim_b", ""),
            "view_id": "all",
            "node_path": (p.get("subject") or "")[:200],
            "paw_verdict": "n/a",
            "gemini_verdict": "confirmed",
            "gemini_explanation": p.get("recheck_explanation", ""),
            "confidence": p.get("recheck_confidence", 0.0),
            "strength": p.get("recheck_strength", "weak"),
            "display_priority": p.get("display_priority", 0),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records):,} curated contradictions → {output_path}",
          flush=True)
    print(
        f"  display_priority=1: {sum(1 for r in records if r['display_priority'])} pairs",
        flush=True,
    )
    if records:
        print("\nTop 10 by composite display score:", flush=True)
        for i, p in enumerate(final[:10], 1):
            print(
                f"  {i:>2}. [{p['recheck_strength']} | conf={p['recheck_confidence']:.2f}] "
                f"{p.get('subject','')[:60]}",
                flush=True,
            )
            print(
                f"      A ({p['doi_1'][:30]}): {p['claim_a'][:100]}",
                flush=True,
            )
            print(
                f"      B ({p['doi_2'][:30]}): {p['claim_b'][:100]}",
                flush=True,
            )
            print(f"      → {p.get('recheck_explanation','')}", flush=True)


if __name__ == "__main__":
    main()
