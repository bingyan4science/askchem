"""
Phase 5: Apply L2-to-L3 demotions for the 797 'demote_small_to_l3' decisions
(after the recheck pass moved 16 synonyms out to 'merge').

Transformation per claim view_path:
   [l1, small_l2, X, ...]  ->  [l1, big_l2, small_l2, X, ...]
                                    (L2)    (new L3)  (now L4)

Result: a 4-level taxonomy where original L3 detail is preserved as L4.

Also patches src/askchem/canonical_l3.py to:
  - Append small_l2 to CANONICAL_L3[view][(l1, big_l2)] (so the new L3 is
    recognized for the new parent).
  - Inserts a marker block at the end so the regenerator can find/update it.

Then rebuilds tree_nodes.

Usage:
  PYTHONPATH=src python3 scripts/apply_l2_demotions.py             # dry-run
  PYTHONPATH=src python3 scripts/apply_l2_demotions.py --commit    # write claims + canonical_l3
  PYTHONPATH=src python3 scripts/apply_l2_demotions.py --commit --rebuild-tree
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DB = ROOT / "chemtree.db"
PRIMARY = ROOT / "data/audits/l2/gemini_validation_cache.json"
CANON_PY = ROOT / "src/askchem/canonical_l3.py"

# Hard cap on path depth to keep tree tractable. We were 3-deep; demotions
# add 1 → max 4. Anything deeper is truncated.
MAX_DEPTH = 4

DEMOTION_BLOCK_BEGIN = "# ── PHASE 5 DEMOTED L2→L3 (auto-generated, do not hand-edit below) ──"
DEMOTION_BLOCK_END = "# ── END PHASE 5 DEMOTED L2→L3 ──"


def load_demotions():
    """Returns dict (view, l1, small) -> big."""
    cache = json.load(PRIMARY.open())
    remap = {}
    for v in cache.values():
        if v.get("decision") == "demote_small_to_l3":
            remap[(v["view"], v["l1"], v["small"])] = v["big"]
    return remap


def patch_canonical_l3(remap: dict, commit: bool):
    """Append a Phase-5 block listing demoted small_l2 → registered as L3 of big_l2."""
    text = CANON_PY.read_text()
    if DEMOTION_BLOCK_BEGIN in text:
        # strip prior block so we can rewrite cleanly
        pat = re.compile(
            re.escape(DEMOTION_BLOCK_BEGIN) + r".*?" + re.escape(DEMOTION_BLOCK_END) + r"\n?",
            re.DOTALL,
        )
        text = pat.sub("", text)

    by_parent = defaultdict(list)  # (view, l1, big) -> [small,...]
    for (view, l1, small), big in remap.items():
        by_parent[(view, l1, big)].append(small)

    lines = ["", DEMOTION_BLOCK_BEGIN, "# These small L2 buckets were demoted to L3 children of the bigger L2",
             "# in Phase 5 of the taxonomy cleanup. Original L3 of the affected claims",
             "# becomes L4 (and is allowed off-whitelist).", ""]
    lines.append("PHASE5_DEMOTED_L3 = {")
    for (view, l1, big), smalls in sorted(by_parent.items()):
        smalls = sorted(set(smalls))
        smalls_lit = ", ".join(repr(s) for s in smalls)
        lines.append(f"    ({view!r}, {l1!r}, {big!r}): [{smalls_lit}],")
    lines.append("}")
    lines.append("")
    lines.append("# Merge the demoted L3 names into the main CANONICAL_L3 whitelist so")
    lines.append("# downstream classification recognises them under the new parent.")
    lines.append("for (_view, _l1, _l2), _smalls in PHASE5_DEMOTED_L3.items():")
    lines.append("    bucket = CANONICAL_L3.setdefault(_view, {}).setdefault((_l1, _l2), [])")
    lines.append("    for _s in _smalls:")
    lines.append("        if _s not in bucket:")
    lines.append("            bucket.insert(-1 if 'other' in bucket else len(bucket), _s)")
    lines.append("")
    lines.append(DEMOTION_BLOCK_END)
    lines.append("")

    new_text = text.rstrip() + "\n" + "\n".join(lines)
    if commit:
        CANON_PY.write_text(new_text)
        print(f"[canonical_l3] patched {CANON_PY} (+{len(by_parent)} (l1,big_l2) groups, {sum(len(s) for s in by_parent.values())} new L3 entries)")
    else:
        print(f"[canonical_l3] would patch {CANON_PY} (+{len(by_parent)} (l1,big_l2) groups, {sum(len(s) for s in by_parent.values())} new L3 entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--rebuild-tree", action="store_true")
    args = ap.parse_args()

    remap = load_demotions()
    print(f"Active L2→L3 demotions: {len(remap)}")

    con = sqlite3.connect(DB)
    cur = con.execute("SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL")

    n_claims_changed = 0
    n_paths_changed = 0
    n_paths_truncated = 0
    pending = []
    BATCH = 1000
    by_demotion = Counter()

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
            # Insert big_l2 in front of small_l2 → small_l2 becomes L3
            new_path = [l1, new_l2, l2] + p[2:]
            if len(new_path) > MAX_DEPTH:
                new_path = new_path[:MAX_DEPTH]
                n_paths_truncated += 1
            vp[vid] = new_path
            n_paths_changed += 1
            by_demotion[(vid, l1, l2, new_l2)] += 1
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
    print(f"View_path entries truncated to depth {MAX_DEPTH}: {n_paths_truncated:,}")
    print()
    print("Top 12 demotions (by claim count):")
    for (vid, l1, small, big), n in by_demotion.most_common(12):
        print(f"   [{vid:18s}] {l1:24s} {small:30s} -> child of {big}  (n={n:,})")

    patch_canonical_l3(remap, args.commit)

    if args.commit and args.rebuild_tree:
        print("\n[tree] rebuilding tree_nodes (will materialise 4-level paths) …")
        import importlib, askchem.canonical_l3 as _c
        importlib.reload(_c)
        from reclassify_l3_batch import rebuild_tree
        rebuild_tree()
        print("[tree] done")

    if not args.commit:
        print("\n(dry-run)")


if __name__ == "__main__":
    main()
