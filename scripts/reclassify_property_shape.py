#!/usr/bin/env python3
"""Sprint B: reclassify property-shape claims that were mis-typed.

Some 580 K claims in chemtree.db have empty type-specific primary fields
(``comparison_result`` for comparison, ``technique_name`` /
``what_it_achieves`` for computational_result / experimental_design,
``reaction_type`` for scope_entry, ``value`` / ``property_name`` for
structure) but DO carry a populated property-shape envelope:
``subject`` / ``property_name`` / ``value`` / ``unit`` /
``measurement_method``. Those rows are simply property measurements that
the upstream extractor labelled with the wrong type.

This script:

  1. Backs up the database to ``chemtree.db.pre_reclass_<ts>.bak``.
  2. Audits candidates per source ``claim_type`` (dry-run).
  3. With ``--apply``, updates each candidate row:
        - claims.claim_type = 'property'
        - data.claim_type = 'property'
        - data._reclassified_from = <orig>
        - data._reclassified_at = <ts>
        - view_paths['by_claim_type'] = ['properties']
  4. Writes a row-id log to ``data/reclassified_<ts>.txt``.

The other taxonomy views (``by_reaction_type``, ``by_substance_class``,
``by_technique``, …) are intentionally untouched: they organise claims
by content, not by the ``claim_type`` field, so reclassifying does not
change their contents.

Usage::

    python scripts/reclassify_property_shape.py            # dry-run
    python scripts/reclassify_property_shape.py --apply    # commit
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.db import get_db_path  # noqa: E402

# Source types we will consider migrating, plus the set of fields we
# require to be empty before relabelling. Order matters: the first match
# wins per row.
CANDIDATES = [
    ("comparison",
     ["comparison_result", "compared_items"]),
    ("computational_result",
     ["technique_name", "what_it_achieves"]),
    ("experimental_design",
     ["technique_name", "what_it_achieves", "key_innovation"]),
    ("scope_entry",
     ["reaction_type"]),
    ("structure",
     ["value", "property_name", "subject"]),
]

# The property-shape envelope we use as evidence the claim is a property.
PROPERTY_SHAPE_KEYS = (
    "subject", "property_name", "value", "unit",
    "measurement_method", "property_category",
)


def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip()) and v.strip().lower() != "null"
    if isinstance(v, (list, dict)):
        return bool(v)
    return bool(v)


def _has_any(d: dict, keys: Iterable[str]) -> bool:
    return any(_truthy(d.get(k)) for k in keys)


def audit(conn: sqlite3.Connection) -> dict[str, int]:
    """Return a per-source-type count of reclassification candidates."""
    counts: dict[str, int] = {}
    for src_type, primary_keys in CANDIDATES:
        n = 0
        cursor = conn.execute(
            "SELECT data FROM claims WHERE claim_type = ?",
            (src_type,),
        )
        for (data_str,) in cursor:
            try:
                d = json.loads(data_str) if data_str else {}
            except Exception:
                continue
            if _has_any(d, primary_keys):
                continue
            if not _has_any(d, PROPERTY_SHAPE_KEYS):
                continue
            n += 1
        counts[src_type] = n
    return counts


def apply_migration(conn: sqlite3.Connection,
                    audit_path: Path,
                    batch_size: int = 5000) -> int:
    """Reclassify candidates in place. Returns the number of rows changed."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total = 0
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = audit_path.open("w")
    log_f.write(f"# Reclassification log written {ts}\n")
    log_f.write("# claim_id\told_type\tsource_doi\n")

    for src_type, primary_keys in CANDIDATES:
        rows = conn.execute(
            "SELECT claim_id, data FROM claims WHERE claim_type = ?",
            (src_type,),
        ).fetchall()
        batch: list[tuple[str, str, str]] = []
        for claim_id, data_str in rows:
            try:
                d = json.loads(data_str) if data_str else {}
            except Exception:
                continue
            if _has_any(d, primary_keys):
                continue
            if not _has_any(d, PROPERTY_SHAPE_KEYS):
                continue
            d["claim_type"] = "property"
            d["_reclassified_from"] = src_type
            d["_reclassified_at"] = ts
            vp = d.get("view_paths") or {}
            if isinstance(vp, dict):
                vp["by_claim_type"] = ["properties"]
                d["view_paths"] = vp
            new_data = json.dumps(d, ensure_ascii=False)
            batch.append((new_data, claim_id))
            log_f.write(f"{claim_id}\t{src_type}\t{d.get('source_doi','')}\n")
            if len(batch) >= batch_size:
                _flush(conn, batch)
                total += len(batch)
                print(f"  {src_type}: committed {total:,} rows so far", flush=True)
                batch.clear()
        if batch:
            _flush(conn, batch)
            total += len(batch)
            batch.clear()
        print(f"  {src_type}: done", flush=True)
    log_f.close()
    return total


def _flush(conn: sqlite3.Connection,
           batch: list[tuple[str, str]]) -> None:
    """Apply a batch of (new_data, claim_id) updates."""
    conn.executemany(
        "UPDATE claims "
        "SET data = ?, "
        "    claim_type = 'property', "
        "    view_paths = json_set(coalesce(view_paths,'{}'), "
        "                          '$.by_claim_type', json_array('properties')) "
        "WHERE claim_id = ?",
        batch,
    )
    conn.commit()


def backup_db(db_path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_suffix(db_path.suffix + f".pre_reclass_{ts}.bak")
    print(f"Backing up {db_path} → {backup} ...", flush=True)
    shutil.copy2(db_path, backup)
    return backup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="Commit the migration. Without this flag, runs a "
                        "dry-run audit only.")
    p.add_argument("--db", default=None, help="Path to chemtree.db")
    args = p.parse_args()

    db_path = Path(args.db) if args.db else get_db_path()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {db_path} ({db_path.stat().st_size/1024/1024/1024:.1f} GB)")

    if not args.apply:
        ro_uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True)
        conn.row_factory = sqlite3.Row
        print("\n--- DRY-RUN audit ---")
        t0 = time.time()
        counts = audit(conn)
        total = sum(counts.values())
        for k, v in counts.items():
            print(f"  {k:<22} {v:>10,}")
        print(f"  {'TOTAL':<22} {total:>10,}")
        print(f"\nelapsed: {time.time()-t0:.1f}s")
        print("\n(Use --apply to commit.)")
        conn.close()
        return

    backup = backup_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    audit_path = ROOT / "data" / f"reclassified_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"\nApplying migration; audit log → {audit_path}")
    print("(this commits in batches; safe to interrupt — last committed "
          "batch persists)")
    t0 = time.time()
    n = apply_migration(conn, audit_path)
    print(f"\nReclassified {n:,} claims in {time.time()-t0:.1f}s")
    print(f"Backup retained at: {backup}")
    conn.close()


if __name__ == "__main__":
    main()
