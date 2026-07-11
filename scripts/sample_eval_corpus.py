"""Stratified pilot-corpus sampler for the Phase γ encoder bake-off.

Single-pass design:
  1. Scan ``claims`` once for rows with ``claim_contextualized`` populated
     (or ``extraction_version='deep_v1'`` as a fallback). Bucket
     ``claim_id``s by ``claim_type`` into in-memory lists.
  2. Force-include every claim_id that appears in
     ``data/eval/labels_v1.jsonl``.
  3. For each claim_type, shuffle the bucket with a seeded PRNG and
     take a number proportional to its share of the contextualized pool.
  4. Output one JSONL row per sampled claim: ``{claim_id, claim_type}``.

The full pass takes ~30 s on the current 1.5 M-claim contextualised pool
because we read only the indexed ``claim_id`` + ``claim_type`` columns.

Usage::

    PYTHONPATH=src python3 scripts/sample_eval_corpus.py \
        --target 200000 --out data/eval/sample_200k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem.db import get_db_path  # noqa: E402

EVAL_DIR = REPO_ROOT / "data" / "eval"
LABELS_PATH = EVAL_DIR / "labels_v1.jsonl"


def _labelled_claim_ids() -> set[str]:
    if not LABELS_PATH.exists():
        return set()
    out: set[str] = set()
    for raw in LABELS_PATH.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.add(json.loads(raw)["claim_id"])
        except Exception:
            continue
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=200_000)
    p.add_argument("--out", type=Path,
                   default=EVAL_DIR / "sample_200k.jsonl")
    p.add_argument("--seed", type=int, default=20260504)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    db_path = get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        labelled = _labelled_claim_ids()
        print(f"labelled claim_ids forced into sample: {len(labelled):,}")

        t0 = time.monotonic()
        print("scanning contextualized pool…")
        buckets: dict[str, list[str]] = defaultdict(list)
        forced_types: dict[str, str] = {}
        n_seen = 0
        cur = conn.execute(
            """
            SELECT claim_id, COALESCE(claim_type, 'other') AS ct
            FROM claims
            WHERE (claim_contextualized IS NOT NULL AND claim_contextualized != '')
               OR extraction_version = 'deep_v1'
            """
        )
        for cid, ct in cur:
            buckets[ct].append(cid)
            if cid in labelled:
                forced_types[cid] = ct
            n_seen += 1
            if n_seen % 250_000 == 0:
                print(f"  scanned {n_seen:,} ({time.monotonic()-t0:.1f}s)")
        print(f"scanned {n_seen:,} contextualized claims "
              f"in {time.monotonic()-t0:.1f}s")
        print(f"  forced labelled-ids resolved (in ctx pool): "
              f"{len(forced_types):,}/{len(labelled):,}")

        # Some labelled claims (e.g. abstract-only Tier-A) are NOT in the
        # contextualized pool. We still want them in the pilot so the
        # judge's positives can be retrieved by every encoder. Look them
        # up directly.
        missing = labelled - set(forced_types)
        if missing:
            t1 = time.monotonic()
            BATCH = 900
            id_list = sorted(missing)
            n_recovered = 0
            for i in range(0, len(id_list), BATCH):
                chunk = id_list[i:i + BATCH]
                ph = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT claim_id, COALESCE(claim_type,'other') "
                    f"FROM claims WHERE claim_id IN ({ph})",
                    chunk,
                ).fetchall()
                for cid_, ct_ in rows:
                    forced_types[cid_] = ct_
                    n_recovered += 1
            print(f"  recovered {n_recovered:,} additional labelled "
                  f"claims outside the ctx pool "
                  f"(in {time.monotonic()-t1:.1f}s)")

        forced_by_type: dict[str, int] = defaultdict(int)
        for ct in forced_types.values():
            forced_by_type[ct] += 1

        remaining = max(0, args.target - len(forced_types))
        print(f"  remaining slots (stratified): {remaining:,}")

        bucket_sizes = {ct: len(ids) for ct, ids in buckets.items()}
        total = sum(bucket_sizes.values())
        target_by_type: dict[str, int] = {}
        for ct, n_pool in bucket_sizes.items():
            share = n_pool / total
            target_by_type[ct] = int(round(remaining * share))
        diff = remaining - sum(target_by_type.values())
        if diff != 0 and target_by_type:
            ct = max(target_by_type, key=target_by_type.get)
            target_by_type[ct] = max(0, target_by_type[ct] + diff)

        print("\nstratification plan (top 10):")
        ranked = sorted(target_by_type.items(), key=lambda kv: -kv[1])
        for ct, n in ranked[:10]:
            print(f"  {ct:<24} target={n:>7,}  pool={bucket_sizes.get(ct,0):>8,}  "
                  f"forced={forced_by_type.get(ct,0)}")

        rng = random.Random(args.seed)
        all_ids: list[tuple[str, str]] = list(forced_types.items())
        excluded = set(forced_types)
        for ct, n in ranked:
            if n <= 0:
                continue
            pool = buckets.get(ct) or []
            if not pool:
                continue
            rng.shuffle(pool)
            picked = 0
            for cid in pool:
                if cid in excluded:
                    continue
                all_ids.append((cid, ct))
                excluded.add(cid)
                picked += 1
                if picked >= n:
                    break

        print(f"\ntotal sampled: {len(all_ids):,} (target {args.target:,})")

        with args.out.open("w") as fh:
            for cid, ct in all_ids:
                fh.write(json.dumps({"claim_id": cid, "claim_type": ct}) + "\n")
        size_mb = args.out.stat().st_size / 1e6
        print(f"wrote {args.out} ({size_mb:.1f} MB)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
