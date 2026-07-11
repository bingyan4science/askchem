#!/usr/bin/env python3
"""Reseed the prod ``contradictions`` table from the in-repo seed JSON.

The HF-distributed ``chemtree.db`` carries an empty ``contradictions``
table (the contradiction detection pipeline runs separately, off-corpus,
in Gemini batch). The source of truth for what shows up on the homepage
``/api/contradictions`` panel lives at
``deploy/contradictions_seed_v1.json`` — a slim list of Gemini-confirmed
contradiction pairs that gets re-inserted on every ``deploy_to_vps.sh``
run, after the fresh DB lands.

Idempotent: ``DELETE FROM contradictions`` then bulk insert, so the
prod table is always exactly the contents of the seed file. Hand-edits
on prod will be overwritten on the next deploy — that is the point.

Usage::

    python3 deploy/reseed_contradictions.py \
        --db /opt/askchem/chemtree.db \
        --seed /opt/askchem/deploy/contradictions_seed_v1.json

Without flags, defaults to ``/opt/askchem/chemtree.db`` and the seed
next to this script.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = "/opt/askchem/chemtree.db"
DEFAULT_SEED = Path(__file__).resolve().parent / "contradictions_seed_v1.json"


def reseed(db_path: str, seed_path: str) -> int:
    seed = json.loads(Path(seed_path).read_text())
    if not isinstance(seed, list):
        raise SystemExit(f"seed file is not a list: {seed_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DELETE FROM contradictions")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        inserted = 0
        for r in seed:
            conn.execute(
                "INSERT INTO contradictions "
                "(claim_id_1, claim_id_2, view_id, node_path, "
                "paw_verdict, gemini_verdict, gemini_explanation, "
                "confidence, detected_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    r["claim_id_1"], r["claim_id_2"],
                    r.get("view_id") or "all",
                    r.get("node_path") or "",
                    r.get("paw_verdict") or "none",
                    r.get("gemini_verdict") or "confirmed",
                    r.get("gemini_explanation") or "",
                    float(r.get("confidence") or 0.0),
                    now,
                ),
            )
            inserted += 1
        conn.commit()
        confirmed = conn.execute(
            "SELECT COUNT(*) FROM contradictions "
            "WHERE gemini_verdict = 'confirmed'"
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM contradictions"
        ).fetchone()[0]
    finally:
        conn.close()
    print(
        f"contradictions reseeded: inserted={inserted:,}  "
        f"total={total:,}  confirmed={confirmed:,}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB,
                    help=f"sqlite DB path (default: {DEFAULT_DB})")
    ap.add_argument("--seed", default=str(DEFAULT_SEED),
                    help=f"seed JSON path (default: {DEFAULT_SEED})")
    args = ap.parse_args()
    if not Path(args.db).exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 1
    if not Path(args.seed).exists():
        print(f"ERROR: seed not found: {args.seed}", file=sys.stderr)
        return 1
    return reseed(args.db, args.seed)


if __name__ == "__main__":
    sys.exit(main())
