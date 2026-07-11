"""
Apply L2 merges from the Gemini-validated cache.

For every cached pair where decision == "merge" (synonyms only — demotes deferred):
  In each claim's view_paths[view]:
    - if path[0] == l1 and path[1] == small: set path[1] = big
    - LEAVE path[2] unchanged. The L3 may not be in CANONICAL_L3 for the new
      parent, but tree_nodes will still materialize it. A follow-up Phase-2-style
      L3-snap pass can collapse any introduced fragmentation.

Then rebuild tree_nodes.

Usage:
  PYTHONPATH=src python3 scripts/apply_l2_merges.py             # dry-run
  PYTHONPATH=src python3 scripts/apply_l2_merges.py --commit    # write
  PYTHONPATH=src python3 scripts/apply_l2_merges.py --commit --rebuild-tree
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.canonical_l3 import CANONICAL_L3  # noqa: E402

DB = ROOT / "chemtree.db"
CACHE = ROOT / "data/audits/l2/gemini_validation_cache.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--rebuild-tree", action="store_true")
    args = ap.parse_args()

    cache = json.load(CACHE.open())
    # Build remap: (view, l1, small_l2) -> big_l2
    # ONLY apply 'merge' decisions in this phase. 'demote_small_to_l3' deferred
    # to a separate phase that promotes small as a new L3 of big (otherwise we'd
    # silently lose 218k claim L3 details).
    remap = {}
    decisions = Counter()
    for v in cache.values():
        d = v.get("decision")
        decisions[d] += 1
        if d == "merge":
            remap[(v["view"], v["l1"], v["small"])] = v["big"]

    print(f"Decision tally: {dict(decisions)}")
    print(f"Active L2 remappings: {len(remap)}")
    print()

    # Iterate claims, apply remaps to view_paths
    con = sqlite3.connect(DB)
    cur = con.execute("SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL")

    n_claims_changed = 0
    n_paths_changed = 0
    pending = []
    BATCH = 1000

    def flush():
        nonlocal pending
        if not pending: return
        if args.commit:
            con.executemany("UPDATE claims SET view_paths = ? WHERE claim_id = ?", pending)
        pending = []

    for claim_id, vp_json in cur:
        try:
            vp = json.loads(vp_json) if vp_json else {}
        except Exception:
            continue
        changed = False
        for vid, p in list(vp.items()):
            if not isinstance(p, list) or len(p) < 2:
                continue
            l1 = p[0]
            l2 = p[1]
            new_l2 = remap.get((vid, l1, l2))
            if not new_l2 or new_l2 == l2:
                continue
            # apply remap (L3 left intact — may be off-whitelist for new parent
            # but tree_nodes will still materialize it)
            p[1] = new_l2
            n_paths_changed += 1
            vp[vid] = p
            changed = True
        if changed:
            n_claims_changed += 1
            pending.append((json.dumps(vp, ensure_ascii=False), claim_id))
            if len(pending) >= BATCH:
                flush()
    flush()
    if args.commit:
        con.commit()
    con.close()

    print(f"Claims changed:            {n_claims_changed:,}")
    print(f"View_path entries changed: {n_paths_changed:,}")

    if args.commit and args.rebuild_tree:
        print("\n[tree] rebuilding tree_nodes …")
        import importlib, askchem.canonical_l3 as _c
        importlib.reload(_c)
        from reclassify_l3_batch import rebuild_tree
        rebuild_tree()
        print("[tree] done")

    if not args.commit:
        print("\n(dry-run)")


if __name__ == "__main__":
    main()
