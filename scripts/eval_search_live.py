"""Drive the full ``db.search_claims`` pipeline against the eval probes.

The companion ``eval_retrieval_live.py`` only times ``retrieval.vector_search``
(dense ANN) + ``cross_rerank``; that's the right shape for δ0/γ1/γ2 where
we wanted apples-to-apples encoder comparisons. **δ2** needs the full
13-stage hybrid pipeline (FTS + tree + RRF + injection + view filter +
bandaids) because the questions are about whether the bandaids hurt or
help end-to-end retrieval.

This script:

  1. Loads the 80-probe set from ``data/eval/probes_v1.jsonl``.
  2. For each probe, calls ``db.search_claims(q, limit=N)`` exactly the
     way ``/api/search`` does.
  3. Records the top-N ``claim_id`` ordering plus a per-stage latency
     breakdown (only emitted when ``CHEMTREE_SEARCH_PROFILE=1``).
  4. Writes the result to ``data/eval/runs/<label>.rankings.jsonl`` so
     ``scripts/eval_metrics.py --run <label>`` can score it against the
     7 483-judgement label pool from δ0/γ2.

Usage::

    CHEMTREE_RETRIEVER_VERSION=v2 CHEMTREE_RERANK_ENABLED=0 \
    CHEMTREE_V2_DIM=256 PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
        python3 scripts/eval_search_live.py \
            --label live-v2-search-256-baseline \
            --top 20

Add the relevant ``CHEMTREE_DISABLE_*`` knob to toggle a bandaid::

    CHEMTREE_DISABLE_TECHNIQUE_STRIPPER=1 \
        python3 scripts/eval_search_live.py \
            --label live-v2-search-256-no-tech-stripper
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem import db, retrieval  # noqa: E402
from eval_common import PROBES_PATH, load_probes  # noqa: E402

RUNS_DIR = REPO_ROOT / "data" / "eval" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True,
                    help="Rankings file label (e.g. live-v2-search-baseline).")
    ap.add_argument("--top", type=int, default=20,
                    help="search_claims limit (and the number of ids "
                    "written to the rankings file).")
    ap.add_argument("--use-semantic", action="store_true", default=True,
                    help="(default) leave dense channel on.")
    ap.add_argument("--no-semantic", dest="use_semantic",
                    action="store_false",
                    help="Disable dense channel — useful for sanity FTS-only "
                    "comparisons.")
    args = ap.parse_args()

    print("=== eval_search_live ===")
    print(f"  active version   : {retrieval.active_version()}")
    print(f"  rerank enabled   : {retrieval.cross_rerank_enabled()}")
    print(f"  use semantic     : {args.use_semantic}")
    print(f"  top              : {args.top}")
    knobs = [k for k in os.environ if k.startswith("CHEMTREE_DISABLE_")
             and os.environ[k] == "1"]
    if knobs:
        print(f"  disabled bandaids: {sorted(knobs)}")
    extras = [k for k in (
        "CHEMTREE_DENSE_MIN_SCORE", "CHEMTREE_TREE_MIN_SCORE",
        "CHEMTREE_V2_DIM",
    ) if os.environ.get(k)]
    if extras:
        print(f"  knobs            : "
              + ", ".join(f"{k}={os.environ[k]}" for k in extras))
    print()

    print("warming retriever + SQLite…")
    retrieval.load_embeddings()
    retrieval.embed_query("warmup")
    # Touch tables once so first probe doesn't pay page-in cost.
    with db.get_conn() as conn:
        conn.execute("SELECT COUNT(*) FROM claims").fetchone()
        conn.execute("SELECT COUNT(*) FROM claims_fts").fetchone()
        conn.execute(
            "SELECT claim_id FROM claims_fts WHERE claims_fts MATCH ? LIMIT 5",
            ["coupling"],
        ).fetchall()
    db._load_tree_node_index()
    print("ready.\n")

    probes = load_probes(PROBES_PATH)
    print(f"loaded {len(probes)} probes")

    rankings: list[dict] = []
    timings: list[float] = []
    for i, pr in enumerate(probes, 1):
        t0 = time.monotonic()
        try:
            result = db.search_claims(
                pr.q, limit=args.top, use_semantic=args.use_semantic,
            )
            cids = [r.get("claim_id") for r in result.get("results", [])
                    if r.get("claim_id")]
        except Exception as exc:
            print(f"  [{i:>2}/{len(probes)}] {pr.id} FAILED: {exc!r}",
                  file=sys.stderr)
            cids = []
        ms = (time.monotonic() - t0) * 1000
        timings.append(ms)
        rankings.append({"probe_id": pr.id, "ranked_claim_ids": cids[: args.top]})
        if i <= 3 or i % 10 == 0:
            print(f"  [{i:>3}/{len(probes)}] {pr.id:<8} "
                  f"family={pr.family:<10} "
                  f"{ms:>6.0f} ms  n={len(cids)}")

    if timings:
        ts = sorted(timings)
        p50 = ts[len(ts) // 2]
        p95 = ts[min(len(ts) - 1, int(len(ts) * 0.95))]
        print(f"\nlatency: p50={p50:.0f} ms  p95={p95:.0f} ms  "
              f"max={ts[-1]:.0f} ms  n={len(ts)}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{args.label}.rankings.jsonl"
    with out_path.open("w") as fh:
        for row in rankings:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote rankings to {out_path}")

    # May-15 ablation: record peak-ish RSS so the ablation report can
    # answer "how much memory does Matryoshka 256-d free?" without a
    # separate harness. Opt-in dependency; skipped silently if missing.
    try:
        import psutil
        rss_gib = psutil.Process().memory_info().rss / 1024**3
        print(f"[rss_end] {rss_gib:.2f} GiB")
    except Exception:
        pass

    print(
        f"\nNext: PYTHONPATH=src python3 scripts/eval_metrics.py "
        f"--run {args.label} --rankings {out_path}"
    )


if __name__ == "__main__":
    main()
