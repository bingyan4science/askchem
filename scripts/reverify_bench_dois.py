#!/usr/bin/env python3
"""Re-extract and re-verify every cited DOI in benchmark_results_gpt-5.5.json.

Use this after touching `extract_dois` / `verify_doi_crossref` /
`compute_metrics` in `benchmark_chemtree.py`. For each question and each
of the four base modes (`llm_alone`, `unified`, `paperclip_unified`,
`edison_scientific`), this:

  1. Re-runs `extract_dois()` over the answer text with the current
     extractor. Picks up DOIs the old regex truncated (legacy Wiley
     SICI, Elsevier S-series with balanced parens, etc.).
  2. Re-verifies each extracted DOI via `verify_doi_crossref()` —
     which now falls back to doi.org for CrossRef coverage gaps
     (arXiv DataCite, mEDRA-registered DOIs, pre-2003 SICI forms).
  3. Preserves any existing `llm_relevance` score for DOIs that
     survive re-extraction (those Gemini judge calls are expensive;
     no point re-doing them). For *newly extracted* DOIs (that the
     old regex truncated and so weren't judged), the new
     `dois_verified` cell has no `llm_relevance` and the metric
     filter (`isinstance(..., (int, float))`) correctly excludes it
     from the relevance mean. Optionally re-run the judge with
     `--judge-new`.
  4. Recomputes `metrics` for each cell using `compute_metrics()`.
  5. Refreshes the per-task aggregate blocks (`aggregate.CA`, `.TC`,
     `.CS`). The overall + notebooklm aggregates are owned by other
     scripts and stay untouched here.

A snapshot is dropped to `.pre_reverify_<ts>.bak` before the rewrite.

Usage::

    python scripts/reverify_bench_dois.py
    python scripts/reverify_bench_dois.py --dry-run
    python scripts/reverify_bench_dois.py --judge-new
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics as _stats
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_askchem as bc  # noqa: E402

BENCH_PATH = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json"
BASE_MODES = ("llm_alone", "unified", "paperclip_unified", "edison_scientific")


def _is_metric_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _existing_relevance(old_verified: dict, doi: str) -> dict:
    """Return the old llm_relevance fields if present, else {}."""
    old = old_verified.get(doi) or old_verified.get(doi.lower()) or {}
    out = {}
    for k in ("llm_relevance", "llm_relevance_rationale", "llm_relevance_judged_with_claim"):
        if k in old:
            out[k] = old[k]
    return out


def _reverify_cell(
    mode_cell: dict,
    question_text: str,
    task: str,
    edison_dois: set,
    judge_new: bool,
    cache_qid: str,
) -> tuple[dict, int, int]:
    """Reprocess one mode cell. Returns (updated_cell, n_extracted, n_new).

    Uses `verify_dois_in_text` rather than calling `verify_doi_crossref`
    directly because the bench's per-DOI cells carry a legacy
    `relevance` field (Jaccard title-relevance against the answer text)
    that `compute_metrics` still reads. The wrapper computes that for
    us alongside the CrossRef enrichment and doi.org fallback.
    """
    answer = mode_cell.get("answer") or mode_cell.get("response") or ""
    if not answer:
        return mode_cell, 0, 0
    old_verified = mode_cell.get("dois_verified") or {}
    # Wrapper produces a full per-DOI dict (exists, crossref_title,
    # relevance, citation_count, year, abstract, type) keyed by the
    # exact DOI string extract_dois returns.
    fresh = bc.verify_dois_in_text(
        answer, question=question_text, sleep_between=0.05,
    )
    n_new = 0
    new_verified: dict = {}
    for d, info in fresh.items():
        rel = _existing_relevance(old_verified, d)
        merged = {**info, **rel}
        key_l = d.lower()
        was_known = any(k.lower() == key_l for k in old_verified)
        if not was_known:
            n_new += 1
            if judge_new and merged.get("exists") and merged.get("crossref_title"):
                rec = bc.score_paper_relevance(
                    cache_qid, d, question_text,
                    merged.get("crossref_title", ""), merged.get("abstract", ""),
                    use_cache=True,
                )
                if rec.get("score") is not None:
                    merged["llm_relevance"] = int(rec["score"])
                    merged["llm_relevance_rationale"] = rec.get("rationale", "")
                    merged["llm_relevance_judged_with_claim"] = False
        new_verified[d] = merged

    new_metrics = bc.compute_metrics(
        answer, new_verified, task, edison_dois=edison_dois or None,
    )

    out = dict(mode_cell)
    out["dois_verified"] = new_verified
    out["metrics"] = new_metrics
    return out, len(fresh), n_new


def _aggregate_modes(questions_for_task: list[dict]) -> dict:
    """Recompute `aggregate.<task>` for one task from its question cells."""
    out: dict[str, dict] = {}
    # Map per-question mode key → per-task aggregate key.
    mode_key_map = {
        "llm_alone": "alone",
        "unified": "unified",
        "paperclip_unified": "paperclip_unified",
        "edison_scientific": "edison_scientific",
    }
    metric_keys = [
        "doi_existence_rate", "doi_relevance_rate", "citation_density",
        "specificity_score", "grounded_specificity",
        "citation_count_mean", "citation_count_median",
        "high_impact_rate", "recent_high_impact_rate",
        "paper_relevance_mean", "paper_relevance_high_rate",
        "edison_overlap_rate",
    ]
    for src, dst in mode_key_map.items():
        cells = [
            q.get(src, {}).get("metrics") or {}
            for q in questions_for_task
            if isinstance(q.get(src), dict) and q[src].get("metrics")
        ]
        if not cells:
            continue
        block: dict[str, object] = {"n_questions": len(cells)}
        for k in metric_keys:
            vals = [c.get(k) for c in cells]
            vals = [v for v in vals if _is_metric_num(v)]
            if vals:
                m = _stats.fmean(vals)
                s = _stats.pstdev(vals) if len(vals) > 1 else 0.0
                block[k] = {"mean": round(m, 3), "std": round(s, 3)}
            else:
                block[k] = {"mean": 0.0, "std": 0.0}
        out[dst] = block
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without rewriting the bench JSON.")
    ap.add_argument("--judge-new", action="store_true",
                    help="Call the Gemini judge for newly extracted DOIs (uses llm_relevance_cache.json).")
    args = ap.parse_args(argv)

    doc = json.loads(BENCH_PATH.read_text())
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = BENCH_PATH.with_suffix(f".pre_reverify_{ts}.bak")
    if not args.dry_run:
        bak.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    questions = doc.get("questions", [])

    # Build Edison DOI sets per question for the overlap-rate metric.
    edison_dois_by_qid: dict[str, set] = {}
    for q in questions:
        ed = (q.get("edison_scientific") or {}).get("dois_verified") or {}
        edison_dois_by_qid[q["id"]] = {
            d.lower() for d, v in ed.items() if v.get("exists")
        }

    n_questions = len(questions)
    changes = 0
    new_dois_total = 0
    flips_to_exists = 0
    started = time.monotonic()
    for q in questions:
        qid = q["id"]
        task = q.get("task") or qid[:2].upper()
        # Edison overlap is omitted for the Edison row itself (the bench's
        # convention is to record None / not report). compute_metrics
        # already handles the "edison_dois=None" case correctly.
        for mode in BASE_MODES:
            cell = q.get(mode)
            if not isinstance(cell, dict):
                continue
            old_metrics = cell.get("metrics") or {}
            old_exist = old_metrics.get("dois_exist") or 0
            edison_set = (
                None if mode == "edison_scientific"
                else edison_dois_by_qid.get(qid, set())
            )
            cache_qid = f"reverify|{qid}|{mode}"
            new_cell, n_extracted, n_new = _reverify_cell(
                cell,
                question_text=q.get("question", ""),
                task=task,
                edison_dois=edison_set,
                judge_new=args.judge_new,
                cache_qid=cache_qid,
            )
            new_exist = (new_cell.get("metrics") or {}).get("dois_exist") or 0
            delta = new_exist - old_exist
            if delta != 0 or n_new != 0:
                changes += 1
                new_dois_total += n_new
                flips_to_exists += max(0, delta)
                print(
                    f"  {qid} {mode:<22s}  "
                    f"cited {old_metrics.get('dois_cited',0):>2d}->{(new_cell.get('metrics') or {}).get('dois_cited',0):>2d}  "
                    f"exist {old_exist:>2d}->{new_exist:>2d}  "
                    f"(new from regex: {n_new})"
                )
            q[mode] = new_cell

    # Persist any new CrossRef cache entries from the re-verification.
    bc._save_crossref_cache()

    # Refresh per-task aggregate blocks. Group by task first.
    questions_by_task: dict[str, list[dict]] = {}
    for q in questions:
        t = q.get("task") or q["id"][:2].upper()
        questions_by_task.setdefault(t, []).append(q)
    for t, qs in questions_by_task.items():
        doc.setdefault("aggregate", {})[t] = {
            **doc.get("aggregate", {}).get(t, {}),
            **_aggregate_modes(qs),
        }
        # Keep any existing notebooklm aggregate block (owned by
        # rollup_notebooklm_scores.py); just don't clobber it.

    elapsed = time.monotonic() - started
    print()
    print(f"questions scanned: {n_questions}")
    print(f"cells with changes: {changes}")
    print(f"newly extracted DOIs (from regex extension): {new_dois_total}")
    print(f"DOIs that flipped to exists=True: {flips_to_exists}")
    print(f"elapsed: {elapsed:.1f}s")

    if args.dry_run:
        print("\n(dry-run: bench JSON not modified)")
        return 0

    BENCH_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"\nwrote {BENCH_PATH.relative_to(REPO_ROOT)}")
    print(f"backup at {bak.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
