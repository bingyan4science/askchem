#!/usr/bin/env python3
"""Backfill Phase 1 metrics into an existing benchmark_results JSON.

Reads the cited DOIs out of every (question, mode) cell, hits CrossRef
for the new ``citation_count`` / ``year`` / ``abstract`` enrichment
(cached to ``scripts/crossref_cache.json``), recomputes the metrics
block via the updated ``compute_metrics`` (which now includes
``grounded_specificity`` and the paper-quality aggregates), and writes
the file back in-place.

LLM-judge backfill (``score_paper_relevance``) is NOT performed by this
script -- run ``scripts/backfill_paper_relevance.py`` for that step
separately. When ``llm_relevance`` is missing the paper-quality
aggregates degrade gracefully (mean == 0, n_papers_with_relevance == 0)
so the website renderer can hide the Relevance column until the judge
run completes.

Usage::

    python scripts/backfill_metrics_phase1.py [path/to/results.json ...]

If no paths are given, defaults to both
``benchmark_results_gpt-5.5.json`` and
``benchmark_results_gpt-5.5_v2-prod_may11.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_askchem import (  # noqa: E402
    compute_metrics,
    verify_doi_crossref,
    _save_crossref_cache,
    aggregate,
    EDISON_SUBSET_IDS,
)


MODE_KEYS = ("llm_alone", "strict_grounded", "retrieval_assisted", "edison_scientific", "llm_plus_askchem")


def enrich_doi_dict(verified_dois: dict, sleep_between: float = 0.05) -> None:
    """In-place enrichment of a verified_dois dict with CrossRef metadata."""
    for doi, info in verified_dois.items():
        if not info.get("exists"):
            # Still annotate with the new (None) fields so downstream
            # consumers don't KeyError on shape inspection.
            info.setdefault("citation_count", None)
            info.setdefault("year", None)
            info.setdefault("abstract", "")
            info.setdefault("type", "")
            continue
        if "citation_count" in info and info.get("citation_count") is not None:
            # Already enriched (idempotent re-runs).
            continue
        fresh = verify_doi_crossref(doi)
        # Preserve any existing fields (especially the old `relevance`
        # Jaccard) and merge in the new ones.
        for k in ("citation_count", "year", "abstract", "type"):
            info[k] = fresh.get(k)
        time.sleep(sleep_between)


def backfill_file(path: Path) -> dict:
    print(f"\n== {path.name} ==")
    if not path.exists():
        print(f"  SKIP: not found")
        return {"path": str(path), "skipped": True}
    data = json.loads(path.read_text())
    questions = data.get("questions") or []
    stats = {"questions": len(questions), "dois_enriched": 0, "metrics_recomputed": 0}

    for q in questions:
        for mode in MODE_KEYS:
            cell = q.get(mode)
            if not isinstance(cell, dict):
                continue
            verified = cell.get("dois_verified")
            if not isinstance(verified, dict):
                continue

            # Enrich
            before = sum(1 for v in verified.values()
                         if isinstance(v, dict) and v.get("citation_count") is None and v.get("exists"))
            enrich_doi_dict(verified)
            after = sum(1 for v in verified.values()
                        if isinstance(v, dict) and v.get("citation_count") is None and v.get("exists"))
            stats["dois_enriched"] += max(0, before - after)

            # Recompute metrics on the same answer text
            answer = cell.get("answer", "")
            cell["metrics"] = compute_metrics(answer, verified, q.get("task", ""))
            stats["metrics_recomputed"] += 1

        print(f"  {q.get('id', '?'):<6} done")

    # Recompute aggregate (per-task, per-mode). The helper expects the
    # per-question list shape; pass `questions` directly.
    data["aggregate"] = aggregate(questions)
    edison_subset = set(EDISON_SUBSET_IDS)
    data["aggregate_subsets"] = {
        "balanced_18_edison": aggregate(questions, allowed_ids=edison_subset),
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    _save_crossref_cache()
    print(f"  -> {stats['dois_enriched']} DOIs enriched, {stats['metrics_recomputed']} metric blocks recomputed")
    return {"path": str(path), **stats}


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = [
            REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json",
            REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5_v2-prod_may11.json",
        ]
    for p in paths:
        backfill_file(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
