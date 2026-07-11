#!/usr/bin/env python3
"""Multi-source discovery wrapper for the May-2026 incremental ingestion.

Calls ``src/update_index.discover_all(...)`` to run arXiv OAI-PMH +
ChemRxiv + journal RSS + Semantic Scholar bulk, deduplicates against
DOIs already in ``chemtree.db.sources``, and dumps the result to
``data/ingestion_2026_05/discovered_papers.jsonl``. Downstream stages
(tier, OA scan, PDF download, extract) read this file.

Usage::

    PORTKEY_API_KEY=... S2_API_KEY=... python3 scripts/harvest_new_papers.py --days 31

The script is idempotent: it always overwrites the JSONL with the
latest discovery snapshot. The harvesters themselves resume from their
own per-source checkpoints under ``data/arxiv_harvest/`` etc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from update_index import discover_all  # noqa: E402
from askchem.db import get_conn  # noqa: E402


OUT_DIR = REPO_ROOT / "data" / "ingestion_2026_05"
OUT_JSONL = OUT_DIR / "discovered_papers.jsonl"
OUT_MANIFEST = OUT_DIR / "harvest_manifest.json"


def get_existing_dois() -> set[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT doi FROM sources").fetchall()
    return {(r["doi"] or "").lower() for r in rows if r["doi"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=31,
                    help="Look back N days (default: 31, matches May-2026 ingestion plan).")
    ap.add_argument("--sources", nargs="*",
                    default=["arxiv", "chemrxiv", "rss", "s2"],
                    choices=["arxiv", "chemrxiv", "rss", "s2", "all"],
                    help="Discovery sources to run.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== Harvest — multi-source discovery ==")
    print(f"  days   : {args.days}")
    print(f"  sources: {args.sources}")
    print(f"  output : {OUT_JSONL}")

    existing = get_existing_dois()
    print(f"  existing DOIs in chemtree.db: {len(existing):,}")

    started = time.time()
    papers = discover_all(args.sources, args.days, existing)
    elapsed = time.time() - started

    by_source: dict[str, int] = {}
    for p in papers:
        src = p.get("_source") or p.get("source") or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

    print(f"\n  discovered: {len(papers):,} new papers (deduped) in {elapsed:.0f}s")
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src:12s} {n:>5d}")

    with OUT_JSONL.open("w") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest = {
        "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "days": args.days,
        "sources": args.sources,
        "n_papers": len(papers),
        "by_source": by_source,
        "elapsed_s": round(elapsed, 1),
        "existing_dois_in_db": len(existing),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {OUT_JSONL}")
    print(f"wrote {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
