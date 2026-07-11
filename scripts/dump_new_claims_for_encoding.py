#!/usr/bin/env python3
"""Dump the new claims (not yet in the v2 FAISS index) as
``{claim_id, text}`` JSONL for the cluster encoder.

Mirrors the text recipe used by the existing FAISS index:
``_claim_to_text(claim, claim_contextualized, paper_summary)``.

Usage::

    PYTHONPATH=src python3 scripts/dump_new_claims_for_encoding.py \\
        --out /tmp/claims_2026_05.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402

SRC_IDS = REPO_ROOT / "data" / "claim_embeddings.v2_256.claim_ids.npy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"loading existing claim_ids from {SRC_IDS} ...")
    existing = set(np.load(str(SRC_IDS), allow_pickle=False).tolist())
    print(f"  existing: {len(existing):,}")

    db_path = get_db_path()
    print(f"scanning {db_path} ...")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    n_scanned = 0
    n_written = 0
    n_skip_empty = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        cur = conn.execute(
            "SELECT c.claim_id, c.data, c.claim_contextualized, s.paper_summary "
            "FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
            "ORDER BY c.claim_id"
        )
        for r in cur:
            n_scanned += 1
            cid = r["claim_id"]
            if not cid or cid in existing:
                continue
            try:
                claim = json.loads(r["data"]) if r["data"] else {}
            except Exception:
                continue
            text = _claim_to_text(
                claim,
                claim_contextualized=r["claim_contextualized"],
                paper_summary=r["paper_summary"],
            )
            if not text:
                n_skip_empty += 1
                continue
            out.write(json.dumps({"claim_id": cid, "text": text},
                                 ensure_ascii=False) + "\n")
            n_written += 1
    conn.close()
    print(f"scanned: {n_scanned:,}")
    print(f"empty/skip: {n_skip_empty:,}")
    print(f"written: {n_written:,} -> {args.out}")
    print(f"size: {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
