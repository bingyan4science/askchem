"""
Apply Phase-3 L3 whitelist expansion.

For each new (view, l1, l2, new_l3) addition in
data/audits/l3/proposed_l3_additions_cleaned.json:

  1. Insert new_l3 into CANONICAL_L3 (in src/askchem/canonical_l3.py),
     placed just before the trailing "other" entry. (Idempotent: skipped if
     already present.)

  2. Re-snap claims whose ORIGINAL (pre-Phase-2) L3 was either new_l3 itself
     or one of its alias_merges → set live view_paths[view][2] = new_l3.

  3. Optionally rebuild tree_nodes.

Usage:
    PYTHONPATH=src python3 scripts/apply_l3_whitelist_expansion.py --dry-run
    PYTHONPATH=src python3 scripts/apply_l3_whitelist_expansion.py --commit --rebuild-tree
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.canonical_l3 import CANONICAL_L3  # noqa: E402

LIVE_DB = ROOT / "chemtree.db"
BACKUP_DB = ROOT / "chemtree.db.pre_phase2_20260424_1425.bak"
PLAN_PATH = ROOT / "data/audits/l3/proposed_l3_additions_cleaned.json"
SRC_PATH = ROOT / "src/askchem/canonical_l3.py"


# ── 1. Patch canonical_l3.py source ───────────────────────────────────────────

def _python_literal_for_tuple(l1: str, l2: str) -> str:
    return f'        ("{l1}", "{l2}"):'


def patch_canonical_l3_source(additions: list[dict]) -> int:
    """Insert each new_l3 into the right parent's list, just before 'other'."""
    text = SRC_PATH.read_text()
    n_inserted = 0

    # Group additions by parent (view, l1, l2) → ordered list of new_l3s
    by_parent: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for a in additions:
        by_parent[(a["view"], a["l1"], a["l2"])].append(a["new_l3"])

    for (view, l1, l2), new_l3s in by_parent.items():
        # Skip ones already in CANONICAL_L3
        existing = CANONICAL_L3.get(view, {}).get((l1, l2), [])
        to_insert = [x for x in new_l3s if x not in existing]
        if not to_insert:
            continue

        # Find the parent block in source. Anchor: '        ("L1", "L2"):'
        anchor = _python_literal_for_tuple(l1, l2)
        idx = text.find(anchor)
        if idx == -1:
            print(f"  [warn] could not locate source anchor for {view} {l1}/{l2}")
            continue

        # Find the trailing '"other",' for this list (the next occurrence after anchor).
        # The block ends with `"other",\n        ],` so we look for the next `, "other",`
        # or `\n            "other",`.
        # We scan forward until we hit the closing `],`.
        block_close = text.find("        ],", idx)
        if block_close == -1:
            print(f"  [warn] could not locate block close for {view} {l1}/{l2}")
            continue

        # Find the last 'other' inside the block
        block_text = text[idx:block_close]
        m = re.search(r'"other"\s*,\s*$', block_text.rstrip("\n").rstrip())
        # Simpler: find 'other' just before the closing
        other_pos = block_text.rfind('"other"')
        if other_pos == -1:
            print(f"  [warn] could not locate 'other' in block for {view} {l1}/{l2}")
            continue

        absolute_other_pos = idx + other_pos
        # Insert each new entry as `"new_l3", ` immediately before `"other"`.
        new_chunk = "".join(f'"{x}", ' for x in to_insert)
        text = text[:absolute_other_pos] + new_chunk + text[absolute_other_pos:]
        n_inserted += len(to_insert)

    return n_inserted, text


# ── 2. Re-snap claims in DB ───────────────────────────────────────────────────

def build_remap(additions: list[dict], aliases: list[dict]) -> dict[tuple, str]:
    """
    Build (view, l1, l2, ORIGINAL_l3) -> new_l3 remap.

    ORIGINAL_l3 may be:
      - the new_l3 itself (the L3 the LLM emitted matches the new canonical)
      - one of its alias_merges (a near-synonym that we're collapsing into new_l3)
    """
    remap: dict[tuple, str] = {}
    for a in additions:
        key = (a["view"], a["l1"], a["l2"], a["new_l3"])
        remap[key] = a["new_l3"]
    for a in aliases:
        key = (a["view"], a["l1"], a["l2"], a["alias_from"])
        remap[key] = a["alias_to"]
    return remap


def resnap_claims(conn_live: sqlite3.Connection, conn_backup: sqlite3.Connection,
                  remap: dict[tuple, str], commit: bool) -> tuple[int, int]:
    """
    Walk the BACKUP claims table to read original view_paths; for each claim
    where any view's original L3 maps via `remap`, update LIVE view_paths to
    use the new canonical L3.

    Returns (n_claims_updated, n_view_paths_changed).
    """
    # First, fetch live view_paths for all candidate claims into memory.
    # Strategy: stream both DBs ordered by claim_id and join in code.
    # Both DBs share the same claim_id space.

    cur_b = conn_backup.execute(
        "SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL ORDER BY claim_id"
    )
    cur_l = conn_live.execute(
        "SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL ORDER BY claim_id"
    )

    n_claims = 0
    n_paths = 0
    pending = []  # batch of (new_view_paths_json, claim_id)
    BATCH = 1000

    def flush():
        nonlocal pending
        if not pending:
            return
        if commit:
            conn_live.executemany(
                "UPDATE claims SET view_paths = ? WHERE claim_id = ?",
                pending,
            )
        pending = []

    # Walk both cursors in lock-step
    row_b = next(cur_b, None)
    row_l = next(cur_l, None)
    while row_b is not None and row_l is not None:
        bid, bjson = row_b
        lid, ljson = row_l
        if bid < lid:
            row_b = next(cur_b, None); continue
        if lid < bid:
            row_l = next(cur_l, None); continue

        # bid == lid
        try:
            backup_vp = json.loads(bjson) if bjson else {}
            live_vp = json.loads(ljson) if ljson else {}
        except Exception:
            row_b = next(cur_b, None); row_l = next(cur_l, None); continue

        changed = False
        for view, b_path in backup_vp.items():
            if not isinstance(b_path, list) or len(b_path) < 3:
                continue
            l1, l2, b_l3 = b_path[0], b_path[1], b_path[2]
            new_l3 = remap.get((view, l1, l2, b_l3))
            if not new_l3:
                continue
            l_path = live_vp.get(view)
            if not isinstance(l_path, list) or len(l_path) < 3:
                continue
            # Only override if live currently points at 'other' (Phase-2 result)
            # or at the legacy off-whitelist value (in case Phase 2 left it as-is).
            if l_path[0] != l1 or l_path[1] != l2:
                continue
            if l_path[2] == new_l3:
                continue
            l_path[2] = new_l3
            live_vp[view] = l_path
            changed = True
            n_paths += 1

        if changed:
            n_claims += 1
            pending.append((json.dumps(live_vp, ensure_ascii=False), lid))
            if len(pending) >= BATCH:
                flush()

        row_b = next(cur_b, None)
        row_l = next(cur_l, None)

    flush()
    if commit:
        conn_live.commit()
    return n_claims, n_paths


# ── 3. Driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="Actually write canonical_l3.py and update DB.")
    ap.add_argument("--rebuild-tree", action="store_true",
                    help="After --commit, regenerate tree_nodes.")
    args = ap.parse_args()

    if not PLAN_PATH.exists():
        sys.exit(f"missing plan: {PLAN_PATH}")
    if not BACKUP_DB.exists():
        sys.exit(f"missing pre-Phase-2 backup: {BACKUP_DB}")

    plan = json.load(PLAN_PATH.open())
    additions = plan["additions"]
    aliases = plan["alias_merges"]
    print(f"Plan: {len(additions)} new L3 buckets, {len(aliases)} alias merges")

    # 1. Patch source
    n_inserted, new_src = patch_canonical_l3_source(additions)
    print(f"\n[source] would insert {n_inserted} new entries into canonical_l3.py")
    if args.commit:
        SRC_PATH.write_text(new_src)
        print(f"[source] wrote {SRC_PATH}")

    # 2. Build remap and resnap
    remap = build_remap(additions, aliases)
    print(f"\n[remap] {len(remap)} (view, l1, l2, original_l3) → new_l3 mappings")

    print("\n[db] connecting…")
    conn_live = sqlite3.connect(LIVE_DB)
    conn_backup = sqlite3.connect(BACKUP_DB)
    n_claims, n_paths = resnap_claims(conn_live, conn_backup, remap, commit=args.commit)
    print(f"[db] {n_claims:,} claims would be updated  ({n_paths:,} view_path entries)")

    if args.commit and args.rebuild_tree:
        print("\n[tree] importing reclassify_l3_batch.rebuild_tree …")
        # rebuild_tree() reads CANONICAL_L3 from the freshly-written module;
        # we must reload the module since we modified the file on disk.
        import importlib
        import askchem.canonical_l3 as _c
        importlib.reload(_c)
        from reclassify_l3_batch import rebuild_tree
        rebuild_tree()
        print("[tree] rebuild_tree() done")

    if not args.commit:
        print("\n(dry-run — re-run with --commit --rebuild-tree to apply)")


if __name__ == "__main__":
    main()
