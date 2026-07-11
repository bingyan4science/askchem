"""Reclaim disk by deleting PDFs whose papers are already deep-extracted.

A PDF in ``data/papers_full/`` is "no longer useful" once its DOI has at
least one ``deep_v1`` claim in ``chemtree.db`` AND a corresponding
result file in ``data/deep_results/``.  In that case the PDF can be
removed safely — re-extracting it later would only require re-downloading.

Default mode is DRY-RUN (prints what would be deleted, freed bytes).
Pass ``--apply`` to actually delete.

Usage:
  # report what's safe to delete
  python scripts/cleanup_extracted_pdfs.py
  # also restrict to PDFs older than N days
  python scripts/cleanup_extracted_pdfs.py --older-than-days 30
  # actually delete
  python scripts/cleanup_extracted_pdfs.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers_full"
RESULTS_DIR = REPO_ROOT / "data" / "deep_results"
DB_PATH = REPO_ROOT / "chemtree.db"


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Actually delete (default: dry-run)")
    p.add_argument("--older-than-days", type=float, default=0,
                   help="Only delete PDFs older than N days (mtime). 0 = no age filter")
    p.add_argument("--require-db-and-results", action="store_true", default=True,
                   help="Require BOTH chemtree.db deep_v1 row AND deep_results JSON file (safest)")
    p.add_argument("--require-results-only", action="store_true",
                   help="Looser: delete if a deep_results JSON exists (don't check DB)")
    args = p.parse_args()

    if not PAPERS_DIR.exists():
        print(f"no PAPERS_DIR ({PAPERS_DIR})", file=sys.stderr); return 1

    # Source 1: deep_v1 DOIs in the DB
    db_dois: set[str] = set()
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            for (doi,) in conn.execute(
                "SELECT DISTINCT source_doi FROM claims "
                "WHERE extraction_version = 'deep_v1' AND source_doi IS NOT NULL"
            ):
                db_dois.add(doi)
        finally:
            conn.close()
    print(f"deep_v1 dois in DB              : {len(db_dois):,}")

    # Source 2: deep_results files (custom_id stems)
    result_stems: set[str] = set()
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.json"):
            result_stems.add(f.stem)
    print(f"deep_results json files         : {len(result_stems):,}")

    # Hashes corresponding to db_dois
    db_stems = {doi_to_filename(d) for d in db_dois}

    if args.require_results_only:
        keep_set = result_stems
        print("policy: delete PDFs that have a deep_results JSON")
    else:
        keep_set = db_stems & result_stems
        print("policy: delete PDFs that have BOTH a DB deep_v1 row AND a deep_results JSON (safest)")

    print(f"PDFs eligible for deletion (by stem match): {len(keep_set):,}")

    cutoff = time.time() - args.older_than_days * 86400 if args.older_than_days else None

    candidates: list[Path] = []
    total_bytes = 0
    for f in PAPERS_DIR.iterdir():
        if f.suffix != ".pdf":
            continue
        if f.stem not in keep_set:
            continue
        if cutoff is not None and f.stat().st_mtime > cutoff:
            continue
        candidates.append(f)
        total_bytes += f.stat().st_size

    print()
    print(f"PDFs to delete             : {len(candidates):,}")
    print(f"Disk to reclaim            : {total_bytes / 1e9:.2f} GB")
    print(f"Mode                       : {'APPLY' if args.apply else 'DRY-RUN'}")

    if not candidates:
        return 0

    print("\nfirst 5 candidates:")
    for f in candidates[:5]:
        print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")

    if not args.apply:
        print("\n(dry-run; pass --apply to delete)")
        return 0

    deleted = 0
    freed = 0
    for f in candidates:
        sz = f.stat().st_size
        try:
            f.unlink()
            deleted += 1; freed += sz
        except OSError as e:
            print(f"  could not delete {f.name}: {e}", file=sys.stderr)

    print(f"\ndeleted {deleted:,} files, freed {freed/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
