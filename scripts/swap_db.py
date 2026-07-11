"""Hot-swap chemtree.db on prod, preserving user-generated tables.

Run on the prod server (askchem). Assumes:
  - The current live DB is at /opt/askchem/chemtree.db
  - The new (freshly built) DB has been rsynced to /opt/askchem/chemtree.db.new
  - The systemd service is named `askchem.service`

Workflow:
  1. Stop askchem.service (brief downtime begins)
  2. Backup current live DB to /opt/askchem/chemtree.db.bak.<timestamp>
  3. Open new DB, ATTACH old DB, copy each user-generated table.
     For tables whose schema differs we copy only the intersection of columns.
  4. Atomically `mv` new DB on top of live DB.
  5. Start askchem.service (brief downtime ends).
  6. Run a sanity query and exit non-zero on any failure (so we don't restart
     into a broken state).

Usage:
  python3 /opt/askchem/scripts/swap_db.py             # do the swap
  python3 /opt/askchem/scripts/swap_db.py --dry-run   # show what it would do
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LIVE = Path("/opt/askchem/chemtree.db")
NEW = Path("/opt/askchem/chemtree.db.new")
SERVICE = "askchem.service"

# Tables we want to PRESERVE from the live (prod) DB. Order matters: parents
# before children for FK-style dependencies.
USER_TABLES = [
    "users",
    "user_sessions",
    "api_keys",
    "key_usage",
    "authors",
    "paper_authors",
    "paper_validations",
    "submissions",
    "subscriptions",
    "bookmarks",
    "feedback",
    "community_flags",
    "query_log",
    "click_log",
    "security_log",
    "notification_log",
    "reading_lists",
    "reading_list_items",
    "saved_searches",
    "surprise_scores",
    "contradictions",
]


def _columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _table_exists(con: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    row = con.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def merge_user_tables(new_db_path: Path, old_db_path: Path) -> dict[str, int]:
    """Copy USER_TABLES from old_db into new_db. Returns rows-copied per table."""
    counts: dict[str, int] = {}
    con = sqlite3.connect(str(new_db_path))
    con.execute(f"ATTACH DATABASE '{old_db_path}' AS olddb")
    con.execute("PRAGMA foreign_keys=OFF")

    for table in USER_TABLES:
        if not _table_exists(con, table, "olddb"):
            counts[table] = -1  # didn't exist in old DB
            continue
        if not _table_exists(con, table, "main"):
            # New DB doesn't have this table at all — create it from old schema
            sql = con.execute(
                "SELECT sql FROM olddb.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()[0]
            con.execute(sql)

        old_cols = set(_columns(con, table))  # main resolves to attached when missing
        # Force fetch columns from each schema explicitly:
        new_cols = [r[1] for r in con.execute(f"PRAGMA main.table_info({table})")]
        old_cols_list = [r[1] for r in con.execute(f"PRAGMA olddb.table_info({table})")]
        common = [c for c in old_cols_list if c in new_cols]
        if not common:
            counts[table] = -2
            continue

        col_csv = ", ".join(f'"{c}"' for c in common)
        con.execute(f"DELETE FROM main.{table}")
        cur = con.execute(
            f"INSERT INTO main.{table} ({col_csv}) "
            f"SELECT {col_csv} FROM olddb.{table}"
        )
        counts[table] = cur.rowcount
        if set(old_cols_list) != set(new_cols):
            print(f"  schema diff on {table}: "
                  f"old_only={sorted(set(old_cols_list)-set(new_cols))} "
                  f"new_only={sorted(set(new_cols)-set(old_cols_list))} "
                  f"(copied common cols)")
    con.commit()
    con.execute("DETACH DATABASE olddb")
    con.close()
    return counts


def _systemctl(action: str) -> None:
    subprocess.run(["systemctl", action, SERVICE], check=True)


def _quick_sanity(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    n_claims = con.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    n_sources = con.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    n_nodes = con.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()[0]
    n_validations = con.execute("SELECT COUNT(*) FROM paper_validations").fetchone()[0]
    n_contras = con.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0]
    con.close()
    print(f"  sanity: claims={n_claims:,} sources={n_sources:,} "
          f"nodes={n_nodes:,} paper_validations={n_validations:,} "
          f"contradictions={n_contras:,}")
    assert n_claims > 1_000_000, "claims count too low — aborting"
    assert n_sources > 100_000, "sources count too low — aborting"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-restart", action="store_true",
                    help="Don't stop/start the service (manual swap)")
    args = ap.parse_args()

    if not NEW.exists():
        print(f"ERR: {NEW} does not exist — rsync the new DB there first.",
              file=sys.stderr)
        return 1
    if not LIVE.exists():
        print(f"ERR: {LIVE} does not exist", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = LIVE.with_suffix(f".db.bak.{ts}")
    print(f"[plan]")
    print(f"  live db:   {LIVE} ({LIVE.stat().st_size/1e9:.2f} GB)")
    print(f"  new db:    {NEW}  ({NEW.stat().st_size/1e9:.2f} GB)")
    print(f"  backup to: {backup}")
    print(f"  user tables to preserve: {len(USER_TABLES)}")
    if args.dry_run:
        print("dry-run — exiting before any side effects")
        return 0

    if not args.skip_restart:
        print("[1/6] stopping service")
        _systemctl("stop")

    try:
        print(f"[2/6] backing up live DB -> {backup}")
        shutil.copy2(str(LIVE), str(backup))

        print("[3/6] merging user tables from live -> new")
        t0 = time.time()
        counts = merge_user_tables(NEW, LIVE)
        for table in USER_TABLES:
            n = counts.get(table, -3)
            tag = "" if n >= 0 else "  (skipped)"
            print(f"    {table:30s} {n:>8d}{tag}")
        print(f"  merge took {time.time()-t0:.1f}s")

        print("[4/6] sanity-checking new DB")
        _quick_sanity(NEW)

        print(f"[5/6] swapping {NEW.name} -> {LIVE.name}")
        os.replace(str(NEW), str(LIVE))

    finally:
        if not args.skip_restart:
            print("[6/6] starting service")
            _systemctl("start")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
