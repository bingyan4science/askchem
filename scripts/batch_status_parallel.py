#!/usr/bin/env python3
"""Parallel Vertex Batch status checker.

The built-in ``src/batch_extract_arxiv.py status --tier N`` polls each
batch serially with a 0.5 s sleep — at 134 batches that's ~15 min.
This script reads the same ``tracker.json``, fans out status calls
across 16 workers, and writes the updated tracker back.

Usage::

    GCS_BKT=... PORTKEY_API_KEY=... python3 scripts/batch_status_parallel.py --tier 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"


def fetch_status(batch_id: str) -> dict:
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = [
        "curl", "-s", "--max-time", "60", "-X", "GET",
        "-H", f"x-portkey-api-key: {api_key}",
        "-H", f"x-portkey-provider: {PROVIDER}",
        f"{GATEWAY}/batches/{batch_id}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=70)
    try:
        return json.loads(proc.stdout) if proc.stdout.strip() else {"error": "empty"}
    except json.JSONDecodeError:
        return {"error": "parse", "raw": proc.stdout[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pdir = REPO_ROOT / "data" / f"arxiv_batch_tier{args.tier}"
    tracker_path = pdir / "tracker.json"
    if not tracker_path.exists():
        print(f"no tracker: {tracker_path}")
        return 1

    tracker: dict = json.loads(tracker_path.read_text())
    to_check = [(fname, info["batch_id"]) for fname, info in tracker.items()
                if info.get("batch_id")]
    print(f"checking {len(to_check)} batches with {args.workers} workers")

    summary: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_status, bid): (fname, bid) for fname, bid in to_check}
        for i, fut in enumerate(as_completed(futs)):
            fname, bid = futs[fut]
            try:
                resp = fut.result()
            except Exception as exc:
                resp = {"error": str(exc)[:120]}
            new_status = resp.get("status", "unknown")
            counts = resp.get("request_counts") or {}
            tracker[fname]["status"] = new_status
            tracker[fname]["request_counts"] = counts
            summary[new_status] = summary.get(new_status, 0) + 1
            completed = counts.get("completed") or 0
            total = counts.get("total") or 0
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(to_check)}] last: {fname} {new_status} ({completed}/{total})")

    tracker_path.write_text(json.dumps(tracker, indent=2))
    print(f"\nsummary: {json.dumps(summary)}")
    print(f"tracker: {tracker_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
