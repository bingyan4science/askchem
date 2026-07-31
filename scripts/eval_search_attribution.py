"""Per-probe attribution harness for the May-29 Phase 1 diagnosis.

Decomposes the 80-probe ``db.search_claims`` result into stage-by-stage
placement of the *judged-relevant* claims from
``data/eval/labels_v1.jsonl``. The output JSON answers the central
question for the PAW search roadmap: *for each probe, is the right
answer never recalled (recall-bounded), recalled but not promoted by
the rerank (rerank-bounded), or already in the top-10 (unaffected)?*

The harness is driven by environment variables that match what
``scripts/run_paw_ft_ab.sh`` set in the May-23 A/B, plus the new
Phase 1 wiring knob ``CHEMTREE_PAW_REWRITES_RERANK``:

  - ``W0`` baseline (PAW off):
        ``CHEMTREE_DISABLE_PAW=1 CHEMTREE_PAW_REWRITES=0
         CHEMTREE_PAW_REWRITES_RERANK=0``
  - ``W1`` current PAW-on (FTS-side only):
        ``CHEMTREE_DISABLE_PAW=0 CHEMTREE_PAW_REWRITES=1
         CHEMTREE_PAW_REWRITES_RERANK=0
         CHEMTREE_PAW_FT_IDS=data/paw_ft_program_ids.json``
  - ``W2`` PAW on FTS *and* rerank input:
        ``CHEMTREE_DISABLE_PAW=0 CHEMTREE_PAW_REWRITES=1
         CHEMTREE_PAW_REWRITES_RERANK=1
         CHEMTREE_PAW_FT_IDS=data/paw_ft_program_ids.json``
  - ``W3`` same as W2 but rerank window 50:
        ``CHEMTREE_RERANK_WINDOW=50`` plus W2 env

Each run writes ``data/eval/runs/attribution_<label>.jsonl`` with one
line per probe; the companion
[scripts/eval_attribution_summary.py](scripts/eval_attribution_summary.py)
aggregates across configs.

Usage::

    PYTHONPATH=src CHEMTREE_DISABLE_PAW=1 ... \\
        .venv-benchmark/bin/python scripts/eval_search_attribution.py \\
            --label W0_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem import db, retrieval  # noqa: E402
from eval_common import LABELS_PATH, PROBES_PATH, iter_jsonl, load_probes  # noqa: E402

RUNS_DIR = REPO_ROOT / "data" / "eval" / "runs"


def load_relevance() -> dict[str, dict[str, int]]:
    """Return {probe_id -> {claim_id -> graded_score}} from labels_v1."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for r in iter_jsonl(LABELS_PATH):
        out[r["probe_id"]][r["claim_id"]] = int(r["score"])
    return out


def rank_of(claim_id: str, ranking: list[str]) -> int | None:
    """0-indexed rank, or None if absent."""
    try:
        return ranking.index(claim_id)
    except ValueError:
        return None


def attribute_probe(probe_id: str, query: str, family: str,
                    judgments: dict[str, int],
                    top_k: int = 20, view: str | None = None,
                    claim_type: str | None = None, mode: str = "auto",
                    sort: str = "relevance") -> dict:
    """Run one probe through the instrumented pipeline.

    Returns the attribution dict. Stages are recorded by claim-id list
    (truncated to a reasonable head) so the file stays small.
    """
    trace: dict = {}
    t0 = time.monotonic()
    try:
        result = db.search_claims(
            query, claim_type=claim_type, view=view, limit=top_k,
            mode=mode, sort=sort, _trace_into=trace,
        )
    except Exception as exc:
        return {
            "probe_id": probe_id, "family": family, "query": query,
            "error": repr(exc),
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
    elapsed_ms = round((time.monotonic() - t0) * 1000)

    judged_pos = {cid for cid, s in judgments.items() if s >= 1}
    judged_high = {cid for cid, s in judgments.items() if s >= 2}

    fts_pool = trace.get("fts_pool", [])
    vec_pool = trace.get("vector_pool", [])
    tree_pool = trace.get("tree_pool", [])
    paper_pool = trace.get("paper_pool", [])
    author_pool = trace.get("author_pool", [])
    rrf_pool = trace.get("rrf_pool", [])
    rerank_input = trace.get("rerank_input", [])
    rerank_output = trace.get("rerank_output", [])
    final_top = trace.get("final_top", [])

    # For each judged-relevant claim, log its rank at each stage
    # (None = absent). Use the highly-relevant set as the primary
    # signal so the "right answer rank" is a single number per probe
    # (the best high-rel claim).
    def _ranks(pool: list[str]) -> dict:
        return {
            cid: rank_of(cid, pool) for cid in judged_pos
        }

    fts_ranks = _ranks(fts_pool)
    vec_ranks = _ranks(vec_pool)
    tree_ranks = _ranks(tree_pool)
    paper_ranks = _ranks(paper_pool)
    author_ranks = _ranks(author_pool)
    rrf_ranks = _ranks(rrf_pool)
    rerank_ranks = _ranks(rerank_output) if rerank_output else _ranks(rerank_input)
    final_ranks = _ranks(final_top)

    # Classify the probe. A probe is:
    #   - "unaffected"      = best judged-positive landed in top-10 already
    #   - "rerank_bounded"  = best in rrf_pool but not top-10 after rerank
    #   - "recall_bounded"  = best not in rrf_pool at all
    #   - "no_relevant"     = no judged-positive (shouldn't happen on the 80-probe set)
    def _best(ranks: dict) -> int | None:
        valid = [r for r in ranks.values() if r is not None]
        return min(valid) if valid else None

    best_final = _best(final_ranks)
    best_rrf = _best(rrf_ranks)
    best_recall = _best({
        **fts_ranks, **vec_ranks, **tree_ranks, **paper_ranks, **author_ranks,
    })

    if not judged_pos:
        category = "no_relevant"
    elif best_final is not None and best_final < 10:
        category = "unaffected"
    elif best_rrf is None and best_recall is None:
        category = "recall_bounded"
    elif best_final is None or best_final >= 10:
        if best_rrf is not None and best_rrf < 50:
            category = "rerank_bounded"
        else:
            category = "recall_bounded"
    else:
        category = "unaffected"

    return {
        "probe_id": probe_id,
        "family": family,
        "query": query,
        "view": view,
        "claim_type": claim_type,
        "mode": mode,
        "sort": sort,
        "latency_ms": elapsed_ms,
        "n_judged_pos": len(judged_pos),
        "n_judged_high": len(judged_high),
        "pool_sizes": {
            "fts": len(fts_pool),
            "vector": len(vec_pool),
            "tree": len(tree_pool),
            "paper": len(paper_pool),
            "author": len(author_pool),
            "rrf": len(rrf_pool),
            "rerank_in": len(rerank_input),
            "rerank_out": len(rerank_output),
            "final": len(final_top),
        },
        "best_rank": {
            "fts": _best(fts_ranks),
            "vector": _best(vec_ranks),
            "tree": _best(tree_ranks),
            "paper": _best(paper_ranks),
            "author": _best(author_ranks),
            "rrf": _best(rrf_ranks),
            "rerank": _best(rerank_ranks),
            "final": best_final,
        },
        "category": category,
        "query_variants": trace.get("query_variants", []),
        "paper_doi_count": trace.get("paper_doi_count", 0),
        "paper_claims_loaded": trace.get("paper_claims_loaded", 0),
        "timings": trace.get("timings", []),
        "counts": trace.get("counts", {}),
        "experiment_config": trace.get("experiment_config", {}),
        "rerank_query": trace.get("rerank_query"),
        # Full top-K so scripts/eval_metrics.py can score nDCG@10
        # directly from this same JSONL.
        "ranked_claim_ids": list(final_top),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True,
                    help="Rankings file label (e.g. W0_baseline).")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--probes", type=Path, default=PROBES_PATH)
    args = ap.parse_args()

    knobs = {
        k: os.environ.get(k, "")
        for k in (
            "CHEMTREE_DISABLE_PAW", "CHEMTREE_PAW_REWRITES",
            "CHEMTREE_PAW_REWRITES_RERANK", "CHEMTREE_PAW_FT_IDS",
            "CHEMTREE_RERANK_WINDOW", "CHEMTREE_RETRIEVER_VERSION",
            "CHEMTREE_V2_DIM", "CHEMTREE_RERANK_ENABLED",
            "CHEMTREE_DISABLE_PRF", "CHEMTREE_DISABLE_TREE_RERANK",
            "CHEMTREE_DISABLE_TREE_RECALL",
            "CHEMTREE_DISABLE_AUTHOR_RECALL",
            "CHEMTREE_DISABLE_SOURCE_PAPER_RECALL",
            "CHEMTREE_DISABLE_CLAIM_GUIDED_PAPER_RECALL",
            "CHEMTREE_DISABLE_FTS", "CHEMTREE_DISABLE_DENSE",
            "CHEMTREE_DISABLE_CITATION_BOOST",
            "CHEMTREE_DISABLE_RERANK",
            "CHEMTREE_MAX_QUERY_VARIANTS",
        )
    }
    print(f"=== eval_search_attribution: {args.label} ===")
    for k, v in knobs.items():
        if v:
            print(f"  {k} = {v}")

    print("\nwarming retriever + SQLite...")
    retrieval.load_embeddings()
    retrieval.embed_query("warmup")
    with db.get_conn() as conn:
        conn.execute("SELECT COUNT(*) FROM claims").fetchone()
        conn.execute(
            "SELECT claim_id FROM claims_fts WHERE claims_fts MATCH ? LIMIT 5",
            ["coupling"],
        ).fetchall()
    db._load_tree_node_index()
    print("ready.\n")

    probes = load_probes(args.probes)
    relevance = load_relevance()

    out_path = RUNS_DIR / f"attribution_{args.label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cat_counts: dict[str, int] = defaultdict(int)
    rows: list[dict] = []

    with out_path.open("w") as fh:
        for i, pr in enumerate(probes, 1):
            judg = relevance.get(pr.id, {})
            row = attribute_probe(
                pr.id, pr.q, pr.family, judg, top_k=args.top,
                view=pr.view, claim_type=pr.claim_type,
                mode=pr.mode, sort=pr.sort,
            )
            cat_counts[row.get("category", "error")] += 1
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if i <= 3 or i % 10 == 0 or i == len(probes):
                br = row.get("best_rank", {}).get("final")
                print(f"  [{i:>3}/{len(probes)}] {pr.id:<10} {pr.family:<10} "
                      f"cat={row['category']:<16} best_final={br}  "
                      f"{row['latency_ms']:>5} ms")

    print(f"\nwrote {len(rows)} rows to {out_path}")
    print("category counts:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<16} {n:>3}  ({n/len(rows):.0%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
