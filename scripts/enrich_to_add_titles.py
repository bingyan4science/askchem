"""Fetch titles for entries in data/audits/to_add.jsonl via S2 batch API.

The original ``data/s2_audit/missing_dois.jsonl`` only stored doi/year/citations/
oa/fos/venue, so ``to_add.jsonl`` has empty titles.  This script back-fills
the title field in-place using the S2 batch endpoint.

Usage:
  S2_API_KEY=... python scripts/enrich_to_add_titles.py
  S2_API_KEY=... python scripts/enrich_to_add_titles.py --top 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
TO_ADD = REPO_ROOT / "data" / "audits" / "to_add.jsonl"
OUT = REPO_ROOT / "data" / "audits" / "to_add.jsonl"  # in-place rewrite
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=2000,
                   help="Only enrich the top-N highest-citation rows (default 2000)")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    if not TO_ADD.exists():
        print(f"missing {TO_ADD}; run audit_journal_coverage.py first", file=sys.stderr)
        return 1

    rows: list[dict] = [json.loads(l) for l in TO_ADD.read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows):,} candidate rows")

    api_key = os.environ.get("S2_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
        print("using S2 API key")
    else:
        print("no S2_API_KEY -- using anonymous (slower)")

    target = rows[:args.top]
    n_filled = 0
    t0 = time.time()
    for start in range(0, len(target), args.batch_size):
        chunk = target[start:start + args.batch_size]
        ids = [f"DOI:{r['doi']}" for r in chunk if r.get("doi")]
        backoff = 1.0
        for attempt in range(5):
            try:
                resp = requests.post(
                    S2_BATCH,
                    params={"fields": "title"},
                    json={"ids": ids},
                    headers=headers,
                    timeout=45,
                )
            except requests.RequestException as e:
                print(f"  net err: {e}, sleeping {backoff:.1f}s")
                time.sleep(backoff); backoff *= 2; continue
            if resp.status_code == 429:
                print(f"  rate-limited, sleeping {backoff:.1f}s")
                time.sleep(backoff); backoff *= 2; continue
            if resp.status_code != 200:
                print(f"  http {resp.status_code}: {resp.text[:120]}")
                time.sleep(backoff); backoff *= 2; continue
            payload = resp.json()
            for row, item in zip(chunk, payload):
                if item and item.get("title"):
                    row["title"] = item["title"]
                    n_filled += 1
            break
        elapsed = time.time() - t0
        print(f"  batch {start//args.batch_size+1}: filled {n_filled:,}/{start+len(chunk):,} "
              f"in {elapsed:.0f}s")
        time.sleep(0.4 if api_key else 1.1)

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nrewrote {OUT.relative_to(REPO_ROOT)} (filled {n_filled:,} titles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
