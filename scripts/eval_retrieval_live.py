"""End-to-end live test of ``askchem.retrieval`` (v2 + cross-encoder).

Drives the production dispatcher exactly the way ``chemtree.db`` will
in v2 mode, against the 10 K pilot corpus we built for the encoder
bake-off (which already contains every labelled claim).  This gives us
an *early signal* — before the full γ2 re-embed of 2.34 M claims
finishes — that the wiring (mxbai dense ANN → FAISS HNSW → cross-encoder
rerank) actually delivers the +0.110 nDCG@10 lift the bake-off promised.

Pipeline per probe::

    askchem.retrieval.vector_search(q, top_k=100)
        → top-100 claim_ids (mxbai dense, FAISS HNSW)
    hydrate_text_for(top_100)
        → {cid: _claim_to_text(...)}     # same renderer as indexing side
    askchem.retrieval.cross_rerank(q, pairs, top_k=20)
        → reordered top-20

Output rankings file feeds into ``scripts/eval_metrics.py`` like every
other run in this harness.

Usage::

    CHEMTREE_RETRIEVER_VERSION=v2 \
    CHEMTREE_EMBEDDINGS_V2_NPZ=$(pwd)/data/eval/vecs/pilot10-mxbai-large.npz \
    CHEMTREE_EMBEDDINGS_V2_FAISS=$(pwd)/data/eval/vecs/pilot10-mxbai-large.faiss \
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/eval_retrieval_live.py \
            --label live-v2-pilot \
            --top-dense 100 --top-rerank 20

    PYTHONPATH=src python3 scripts/eval_metrics.py \
        --run live-v2-pilot \
        --rankings data/eval/runs/live-v2-pilot.rankings.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem import retrieval  # noqa: E402
from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402
from eval_common import PROBES_PATH, load_probes  # noqa: E402

EVAL_DIR = REPO_ROOT / "data" / "eval"
RUNS_DIR = EVAL_DIR / "runs"


def hydrate_text_for(claim_ids: list[str]) -> dict[str, str]:
    """Build the indexed text for every requested claim id.

    Mirrors the indexing-side text in ``embeddings._claim_to_text``
    (claim_contextualized + paper_summary + typed fields + verbatim).
    """
    if not claim_ids:
        return {}
    db_path = get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        BATCH = 900
        out: dict[str, str] = {}
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
                txt = _claim_to_text(
                    claim,
                    claim_contextualized=r["claim_contextualized"],
                    paper_summary=r["paper_summary"],
                )
                if txt:
                    out[r["claim_id"]] = txt
        return out
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True,
                   help="Output rankings label (e.g. 'live-v2-pilot').")
    p.add_argument("--top-dense", type=int, default=100,
                   help="Top-N candidates from the dense ANN stage.")
    p.add_argument("--top-rerank", type=int, default=20,
                   help="Top-N kept after cross-encoder rerank.")
    p.add_argument("--no-rerank", action="store_true",
                   help="Skip the cross-encoder stage (dense-only baseline).")
    args = p.parse_args()

    print("=== eval_retrieval_live ===")
    print(f"  active version       : {retrieval.active_version()}")
    print(f"  cross_rerank_enabled : {retrieval.cross_rerank_enabled()}")
    print(f"  top-dense, top-rerank: {args.top_dense}, {args.top_rerank}")
    print()

    print("loading retriever (npz + FAISS, encoder)…")
    retrieval.load_embeddings()
    if not retrieval.is_loaded():
        raise SystemExit(
            "retrieval.is_loaded() == False — check "
            "CHEMTREE_EMBEDDINGS_V2_NPZ / CHEMTREE_EMBEDDINGS_V2_FAISS."
        )
    # Warm the query encoder + cross-encoder so the first probe isn't slow.
    retrieval.embed_query("warmup")
    if not args.no_rerank and retrieval.cross_rerank_enabled():
        retrieval.warmup_cross_encoder()
    print("ready.\n")

    probes = load_probes(PROBES_PATH)
    print(f"loaded {len(probes)} probes")

    rankings: list[dict] = []
    dense_ms: list[float] = []
    rerank_ms: list[float] = []
    for i, pr in enumerate(probes, 1):
        t0 = time.monotonic()
        dense_hits = retrieval.vector_search(
            pr.q, top_k=args.top_dense, min_score=0.0,
        )
        d_ms = (time.monotonic() - t0) * 1000
        dense_ms.append(d_ms)

        cids = [cid for cid, _ in dense_hits]
        if not cids:
            rankings.append({"probe_id": pr.id, "ranked_claim_ids": []})
            continue

        if args.no_rerank or not retrieval.cross_rerank_enabled():
            rankings.append({
                "probe_id": pr.id,
                "ranked_claim_ids": cids[: args.top_rerank],
            })
            if i <= 3 or i % 20 == 0:
                print(f"  [{i:>2}/{len(probes)}] {pr.id:<8} "
                      f"family={pr.family:<10} dense={d_ms:.0f}ms "
                      f"-> {len(cids[:args.top_rerank])}")
            continue

        text_map = hydrate_text_for(cids)
        pairs = [(cid, text_map[cid]) for cid in cids if cid in text_map]
        if not pairs:
            rankings.append({"probe_id": pr.id, "ranked_claim_ids": []})
            continue

        t0 = time.monotonic()
        reranked = retrieval.cross_rerank(
            pr.q, pairs, top_k=args.top_rerank,
        )
        r_ms = (time.monotonic() - t0) * 1000
        rerank_ms.append(r_ms)

        rankings.append({
            "probe_id": pr.id,
            "ranked_claim_ids": [cid for cid, _ in reranked],
        })
        if i <= 3 or i % 20 == 0:
            print(f"  [{i:>2}/{len(probes)}] {pr.id:<8} "
                  f"family={pr.family:<10} dense={d_ms:.0f}ms "
                  f"rerank={r_ms:.0f}ms ({len(pairs)})")

    def _summary(name: str, xs: list[float]) -> str:
        if not xs:
            return f"  {name}: (no samples)"
        xs_sorted = sorted(xs)
        p50 = xs_sorted[len(xs_sorted) // 2]
        p95 = xs_sorted[min(len(xs_sorted) - 1, int(len(xs_sorted) * 0.95))]
        return (f"  {name}: p50={p50:.0f} ms  p95={p95:.0f} ms  "
                f"max={max(xs_sorted):.0f} ms  n={len(xs_sorted)}")

    print("\nlatency:")
    print(_summary("dense ", dense_ms))
    if rerank_ms:
        print(_summary("rerank", rerank_ms))
        total = [d + r for d, r in zip(dense_ms[:len(rerank_ms)], rerank_ms)]
        print(_summary("total ", total))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{args.label}.rankings.jsonl"
    with out_path.open("w") as fh:
        for row in rankings:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote rankings to {out_path}")
    print(
        f"\nNext: PYTHONPATH=src python3 scripts/eval_metrics.py "
        f"--run {args.label} --rankings {out_path}"
    )


if __name__ == "__main__":
    main()
