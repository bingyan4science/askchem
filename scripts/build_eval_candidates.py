"""Build the pooled candidate set for the Phase 0 eval.

For each probe in ``data/eval/probes_v1.jsonl``, run all three of our
current candidate retrievers (FTS5, dense vectors, tree recall) and
union the top-K of each. The judge labels this *union* once; any
encoder we test later that surfaces an unlabelled doc gets either
labelled incrementally (preferred) or scored 0 (cheap).

Usage::

    python scripts/build_eval_candidates.py                # default: top 20 per retriever
    python scripts/build_eval_candidates.py --per 30       # broader pool

Outputs ``data/eval/candidates_v1.jsonl`` (one line per probe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_common import (  # noqa: E402
    PROBES_PATH, CANDIDATES_PATH, load_probes, write_jsonl,
)
from askchem.db import (  # noqa: E402
    get_conn, _build_fts_queries, _run_fts_cascade, _tree_recall,
)
from askchem import embeddings  # noqa: E402


def build_for_probe(q: str, per_retriever: int,
                    use_vector: bool) -> dict:
    fts_ids: list[str] = []
    vec_ids: list[str] = []
    tree_ids: list[str] = []
    timings: dict[str, int] = {}

    with get_conn() as conn:
        t0 = time.monotonic()
        rows, _ = _run_fts_cascade(
            _build_fts_queries(q), None, per_retriever, conn,
        )
        fts_ids = [r["claim_id"] for r in rows]
        timings["fts_ms"] = int((time.monotonic() - t0) * 1000)

        t0 = time.monotonic()
        tree_ids = _tree_recall(q, conn, top_k=per_retriever) or []
        timings["tree_ms"] = int((time.monotonic() - t0) * 1000)

    if use_vector:
        t0 = time.monotonic()
        vec_hits = embeddings.vector_search(
            q, top_k=per_retriever, min_score=0.0,
        )
        vec_ids = [cid for cid, _ in vec_hits]
        timings["vec_ms"] = int((time.monotonic() - t0) * 1000)

    pool: list[str] = []
    seen: set[str] = set()
    sources: dict[str, list[str]] = {}  # cid → which retriever surfaced it
    for label, ids in (("fts", fts_ids), ("vec", vec_ids), ("tree", tree_ids)):
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                pool.append(cid)
            sources.setdefault(cid, []).append(label)

    return {
        "q": q,
        "candidate_ids": pool,
        "sources": sources,
        "counts": {
            "fts": len(fts_ids),
            "vec": len(vec_ids),
            "tree": len(tree_ids),
            "union": len(pool),
        },
        "timings_ms": timings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per", type=int, default=20,
                   help="top-K per retriever (default: 20)")
    p.add_argument("--probes", default=str(PROBES_PATH))
    p.add_argument("--out", default=str(CANDIDATES_PATH))
    p.add_argument("--no-vector", action="store_true",
                   help="skip dense vector recall (useful when "
                        "embeddings aren't loaded)")
    args = p.parse_args()

    probes = load_probes(Path(args.probes))
    print(f"Loaded {len(probes)} probes from {args.probes}")

    if not args.no_vector:
        print("Loading embeddings...")
        embeddings.load_embeddings()
        ready = embeddings.is_loaded()
        if not ready:
            print("  embeddings not loaded; falling back to FTS+tree only",
                  flush=True)
        use_vector = ready
    else:
        use_vector = False

    rows: list[dict] = []
    pool_sizes: list[int] = []
    t_total = time.monotonic()
    for i, probe in enumerate(probes, 1):
        t0 = time.monotonic()
        rec = build_for_probe(probe.q, args.per, use_vector)
        rec["id"] = probe.id
        rec["family"] = probe.family
        elapsed = int((time.monotonic() - t0) * 1000)
        rows.append(rec)
        pool_sizes.append(len(rec["candidate_ids"]))
        print(
            f"  [{i:>2}/{len(probes)}] {probe.id:<8} "
            f"family={probe.family:<10} "
            f"fts={rec['counts']['fts']:>2} vec={rec['counts']['vec']:>2} "
            f"tree={rec['counts']['tree']:>2} -> "
            f"union={rec['counts']['union']:>2}  ({elapsed} ms)",
            flush=True,
        )

    write_jsonl(Path(args.out), rows)
    total_judgments = sum(pool_sizes)
    avg = total_judgments / max(1, len(rows))
    print()
    print(f"Wrote {args.out}")
    print(f"  probes:                {len(rows)}")
    print(f"  total judgments:       {total_judgments}")
    print(f"  avg pool/probe:        {avg:.1f}")
    print(f"  min/max pool/probe:    {min(pool_sizes)} / {max(pool_sizes)}")
    print(f"  total wall:            {int((time.monotonic() - t_total))}s")


if __name__ == "__main__":
    main()
