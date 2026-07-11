"""A6 ablation: search_claims result LRU.

Runs the 80-probe set twice in a single Python process:
  Pass 1 (cold): cache is empty, latency ≈ baseline
  Pass 2 (warm): every probe is a cache hit, latency ≈ cache lookup cost

Writes Pass 2 rankings (which must be byte-identical to Pass 1) to
data/eval/runs/ablation-a6-cache.rankings.jsonl so eval_metrics can
confirm zero quality regression.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Cache enabled before import so module-level constants are read correctly.
os.environ.setdefault("CHEMTREE_SEARCH_CACHE", "1")
os.environ.setdefault("CHEMTREE_SEARCH_CACHE_SIZE", "512")
os.environ.setdefault("CHEMTREE_SEARCH_CACHE_TTL_S", "3600")

from askchem import db, retrieval  # noqa: E402
from eval_common import PROBES_PATH, load_probes  # noqa: E402

RUNS_DIR = REPO_ROOT / "data" / "eval" / "runs"
TOP = 20
LABEL = "ablation-a6-cache"


def percentile(values, p):
    if not values:
        return float("nan")
    sv = sorted(values)
    idx = min(len(sv) - 1, int(len(sv) * p))
    return sv[idx]


def run_pass(probes, label):
    timings = []
    rankings = []
    for i, pr in enumerate(probes, 1):
        t0 = time.monotonic()
        try:
            result = db.search_claims(pr.q, limit=TOP)
            cids = [r.get("claim_id") for r in result.get("results", [])
                    if r.get("claim_id")]
        except Exception as exc:
            print(f"  [{i}/{len(probes)}] {pr.id} FAILED: {exc!r}",
                  file=sys.stderr)
            cids = []
        ms = (time.monotonic() - t0) * 1000
        timings.append(ms)
        rankings.append({"probe_id": pr.id, "ranked_claim_ids": cids[:TOP]})
    print(f"\n[{label}] p50={percentile(timings, 0.5):.0f} ms  "
          f"p95={percentile(timings, 0.95):.0f} ms  "
          f"max={max(timings):.0f} ms  n={len(timings)}")
    return rankings, timings


def main():
    print(f"=== {LABEL} ===")
    print(f"  CHEMTREE_SEARCH_CACHE = {os.environ.get('CHEMTREE_SEARCH_CACHE')}")
    print(f"  CHEMTREE_SEARCH_CACHE_SIZE = {os.environ.get('CHEMTREE_SEARCH_CACHE_SIZE')}")
    print(f"  CHEMTREE_SEARCH_CACHE_TTL_S = {os.environ.get('CHEMTREE_SEARCH_CACHE_TTL_S')}")
    print(f"  active version = {retrieval.active_version()}")
    print(f"  rerank enabled = {retrieval.cross_rerank_enabled()}")

    print("\nwarming retriever + SQLite...")
    retrieval.load_embeddings()
    retrieval.embed_query("warmup")
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
    print(f"loaded {len(probes)} probes\n")

    print("PASS 1 (cold, cache populating)")
    r1, t1 = run_pass(probes, "pass1-cold")

    print("\nPASS 2 (warm, all cache hits)")
    r2, t2 = run_pass(probes, "pass2-warm")

    # Confirm rankings match
    mismatched = sum(1 for a, b in zip(r1, r2)
                     if a["ranked_claim_ids"] != b["ranked_claim_ids"])
    print(f"\nRanking mismatch between cold and warm passes: {mismatched}/{len(r1)}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{LABEL}.rankings.jsonl"
    with out_path.open("w") as fh:
        for row in r2:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote warm-pass rankings to {out_path}")

    try:
        import psutil
        rss_gib = psutil.Process().memory_info().rss / 1024**3
        print(f"[rss_end] {rss_gib:.2f} GiB")
    except Exception:
        pass

    # Compact summary for the report
    print(f"\n[A6 SUMMARY]")
    print(f"  pass1-cold  p50={percentile(t1, 0.5):.0f} ms  p95={percentile(t1, 0.95):.0f} ms  max={max(t1):.0f} ms")
    print(f"  pass2-warm  p50={percentile(t2, 0.5):.0f} ms  p95={percentile(t2, 0.95):.0f} ms  max={max(t2):.0f} ms")


if __name__ == "__main__":
    main()
