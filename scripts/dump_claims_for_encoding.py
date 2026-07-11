"""Dump (claim_id, text) JSONL for the cluster encoder.

Mirrors ``embeddings._claim_to_text`` exactly so the cluster's
``mxbai-embed-large-v1`` sees the same input as the local pipeline. The
output is a single ``claims.jsonl`` (one JSON object per line) suitable
for streaming into the cluster job — no SQLite required on cluster
side.

Usage::

    PYTHONPATH=src python3 scripts/dump_claims_for_encoding.py \\
        --out data/eval/cluster/claims.jsonl
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

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402

CHUNK = 5000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "data" / "eval" / "cluster" /
                           "claims.jsonl")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N rows (debug). 0 = full dump.")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    db_path = get_db_path()
    print(f"db   : {db_path}")
    print(f"out  : {args.out}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"rows : {total:,}\n")

    cur = conn.cursor()
    cur.execute(
        "SELECT c.claim_id, c.data, c.claim_contextualized, "
        "       s.paper_summary "
        "FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi"
    )

    written = 0
    empty = 0
    bad = 0
    t0 = time.monotonic()
    with args.out.open("w") as fh:
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            buf = []
            for cid, data, ctx, ps in rows:
                try:
                    claim = json.loads(data)
                except Exception:
                    bad += 1
                    continue
                txt = _claim_to_text(
                    claim,
                    claim_contextualized=ctx,
                    paper_summary=ps,
                )
                if not txt:
                    empty += 1
                    continue
                buf.append(json.dumps(
                    {"claim_id": cid, "text": txt},
                    ensure_ascii=False,
                ))
            if buf:
                fh.write("\n".join(buf) + "\n")
                written += len(buf)
            if written and written % (CHUNK * 10) == 0:
                rate = written / (time.monotonic() - t0)
                print(f"  {written:>9,}/{total:,}  ({rate:,.0f} rows/s)",
                      flush=True)
            if args.limit and written >= args.limit:
                break

    elapsed = time.monotonic() - t0
    size_mb = args.out.stat().st_size / 1e6
    print(f"\ndone in {elapsed/60:.1f} min")
    print(f"  written : {written:,}")
    print(f"  empty   : {empty:,}")
    print(f"  bad     : {bad:,}")
    print(f"  size    : {size_mb:,.1f} MB")
    print(f"  out     : {args.out}")


if __name__ == "__main__":
    main()
