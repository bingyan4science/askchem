"""Merge a freshly-placed "full-text-only" grown tree into the existing v16 tree.

`batch_place.py collect` rebuilds `output/grown_views.json` from the raw scaffold
using ONLY the current batch's placements. For the incremental full-text run we
therefore collect a "full-text-only" tree and then graft it onto the cleaned v16
tree with this script, matching nodes by the SAME identity `apply_to_db.py` uses
(`__root__` for open_root, else `scaffold_builder._norm(name)`).

Because the incremental batch used `--exclude-placed`, the incoming claims are
disjoint from v16, so no cross-tree leaf de-dup is needed. Shared scaffold nodes
union their leaf children; internal nodes present only in the incoming tree are
attached under their matched parent (nearest matched ancestor, else root). The
cleanup chain (semantic_dedup -> combine_nodes) then reconciles any duplicate
proposed concepts the merge introduces (e.g. a scaffold node v16 cleanup renamed).

Usage:
    python3 living_taxonomy/merge_grown.py \
        --base output/grown_views.v16.json \
        --incoming output/grown_views.json \
        --out output/grown_views.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
import scaffold_builder as sb  # noqa: E402

ROOT_ID = "__root__"


def node_id(n: dict) -> str:
    if n.get("kind") == "open_root":
        return ROOT_ID
    return sb._norm(n.get("name", "")) or ("n_" + str(abs(hash(n.get("name", ""))) % 10 ** 8))


def index_internal(root: dict) -> dict:
    """node_id -> node, for every internal (non-leaf) node in the tree."""
    idx: dict = {}

    def walk(n):
        idx.setdefault(node_id(n), n)
        for c in n.get("children", []) or []:
            if c.get("kind") != "leaf":
                walk(c)
    walk(root)
    return idx


def merge_view(a_root: dict, b_root: dict) -> dict:
    a_idx = index_internal(a_root)
    stats = {"added_nodes": 0, "added_leaves": 0, "matched_nodes": 0}

    def ensure(bnode: dict, bparent: dict | None) -> dict:
        """Return the A-node matching bnode, creating a shallow copy under the
        A-match of bparent (or root) when absent."""
        nid = node_id(bnode)
        hit = a_idx.get(nid)
        if hit is not None:
            stats["matched_nodes"] += 1
            return hit
        aparent = a_root if bparent is None else a_idx.get(node_id(bparent), a_root)
        newnode = {k: v for k, v in bnode.items() if k != "children"}
        newnode["children"] = []
        aparent.setdefault("children", []).append(newnode)
        a_idx[nid] = newnode
        stats["added_nodes"] += 1
        return newnode

    def walk(bnode: dict, bparent: dict | None):
        anode = a_root if bparent is None else ensure(bnode, bparent)
        existing = {c.get("claim_id") for c in (anode.get("children", []) or [])
                    if c.get("kind") == "leaf"}
        for c in bnode.get("children", []) or []:
            if c.get("kind") == "leaf":
                cid = c.get("claim_id")
                if cid and cid not in existing:
                    anode.setdefault("children", []).append(c)
                    existing.add(cid)
                    stats["added_leaves"] += 1
            else:
                walk(c, bnode)

    walk(b_root, None)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(_HERE / "output" / "grown_views.v16.json"))
    ap.add_argument("--incoming", default=str(_HERE / "output" / "grown_views.json"))
    ap.add_argument("--out", default=str(_HERE / "output" / "grown_views.json"))
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text())
    inc = json.loads(Path(args.incoming).read_text())
    A, B = base["views"], inc["views"]
    for vid, broot in B.items():
        if vid not in A:
            A[vid] = broot
            print(f"[merge] {vid}: view only in incoming -> added whole")
            continue
        st = merge_view(A[vid], broot)
        print(f"[merge] {vid}: +{st['added_leaves']:,} leaves, +{st['added_nodes']} new nodes "
              f"({st['matched_nodes']:,} matched hosts)")
    base["subtitle"] = (str(base.get("subtitle", "")) + " + full-text incremental")[:200]
    Path(args.out).write_text(json.dumps(base))
    print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
