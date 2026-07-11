#!/usr/bin/env python3
"""Compute `aggregate.overall.<mode>` blocks in benchmark_results_gpt-5.5.json.

The bench JSON ships per-task aggregates (`aggregate.CA`, `.TC`, `.CS`)
that summarise each mode (`alone`, `unified`, `paperclip_unified`,
`edison_scientific`, `notebooklm`) on that task. The leaderboard on
askchem.org now pools all five systems together, so the per-task split
yields a 15-row table that's hard to scan.

This script pools the per-question metric cells in `questions[]`
straight across all task types, computes `{mean, std}` per metric per
mode, and writes `aggregate.overall.<mode>` blocks back into the bench
JSON in place (with a `.pre_overall_<ts>.bak` snapshot).

Usage::

    python scripts/compute_overall_bench_aggregate.py
"""
from __future__ import annotations

import datetime as _dt
import json
import statistics as _stats
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_PATH = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json"

# (per_question_key, aggregate_key). Per-question cells use `llm_alone`
# but the per-task aggregate uses `alone`; we mirror the aggregate key
# so the renderer can read `aggregate.overall.alone` consistently with
# `aggregate.CA.alone` / `.TC.alone` / `.CS.alone`.
MODE_KEYS = (
    ("llm_alone",         "alone"),
    ("unified",           "unified"),
    ("paperclip_unified", "paperclip_unified"),
    ("edison_scientific", "edison_scientific"),
    ("notebooklm",        "notebooklm"),
)

# Mirror the per-task aggregate shape exactly so the renderer can read
# `aggregate.overall.<mode>.<metric>.mean` with the same lookup it uses
# for task-specific cells.
METRIC_KEYS = (
    "doi_existence_rate",
    "doi_relevance_rate",
    "citation_density",
    "specificity_score",
    "grounded_specificity",
    "citation_count_mean",
    "citation_count_median",
    "high_impact_rate",
    "recent_high_impact_rate",
    "paper_relevance_mean",
    "paper_relevance_high_rate",
    "edison_overlap_rate",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": round(_stats.fmean(values), 3),
        "std": round(_stats.pstdev(values), 3) if len(values) > 1 else 0.0,
    }


def main() -> int:
    doc = json.loads(BENCH_PATH.read_text())
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = BENCH_PATH.with_suffix(f".pre_overall_{ts}.bak")
    bak.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    questions = doc.get("questions", [])
    overall: dict[str, dict] = {}
    for src_key, dst_key in MODE_KEYS:
        per_question = [
            q.get(src_key, {}).get("metrics", {})
            for q in questions
            if isinstance(q.get(src_key), dict) and q[src_key].get("metrics")
        ]
        if not per_question:
            continue
        block: dict[str, object] = {"n_questions": len(per_question)}
        for key in METRIC_KEYS:
            vals = [m.get(key) for m in per_question]
            vals = [v for v in vals if isinstance(v, (int, float))]
            block[key] = _mean_std(vals)
        overall[dst_key] = block

    doc.setdefault("aggregate", {})["overall"] = overall
    BENCH_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    print(f"wrote aggregate.overall with {len(overall)} modes:")
    for _src, dst in MODE_KEYS:
        block = overall.get(dst)
        if not block:
            continue
        rel = block.get("paper_relevance_mean", {}).get("mean")
        doi = block.get("doi_existence_rate", {}).get("mean")
        n = block.get("n_questions")
        print(
            f"  {dst:<22s}  n={n:>2d}  "
            f"DOI exist={doi*100:5.1f}%  relevance={rel:.2f}"
        )
    print(f"\nbackup at {bak.relative_to(REPO_ROOT)}")
    print("restart askchem.service to invalidate the 1h /api/benchmark cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
