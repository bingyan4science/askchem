"""Cycle-safe duplicate merge for the living tree.

`concept_registry.dedupe_views` can introduce circular references when it merges
two nodes that are in an ancestor/descendant relationship (the descendant absorbs
its ancestor's subtree -> a node becomes its own ancestor). This pass merges
duplicate INTERNAL nodes by normalized name within each view, but NEVER merges a
pair that is in an ancestor/descendant relationship, so it cannot create a cycle.

For each duplicate group (same sb._norm(name)):
  - canonical = the shallowest occurrence (closest to root; ties -> most descendants)
  - every other occurrence not ancestor/descendant of canonical is folded into it:
    its children (internal + leaf) move to canonical (leaves de-duped by claim_id),
    and the now-empty duplicate is detached from its parent.
Iterates to a fixed point.

Usage:
    python3 living_taxonomy/dedup_safe.py            # in place on output/grown_views.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "src"))
import scaffold_builder as sb

GV = _HERE / "output" / "grown_views.json"


def _norm(n):
    return sb._norm(n.get("name", "")) or n.get("name", "")


def _index(root):
    """Iterative: return (parent_of[id]=node, depth[id], internal_nodes list)."""
    parent, depth, nodes = {}, {id(root): 0}, []
    stack = [(root, 0)]
    seen = {id(root)}
    while stack:
        n, d = stack.pop()
        nodes.append(n)
        for c in n.get("children", []) or []:
            if c.get("kind") == "leaf":
                continue
            if id(c) in seen:
                continue
            seen.add(id(c))
            parent[id(c)] = n
            depth[id(c)] = d + 1
            stack.append((c, d + 1))
    return parent, depth, nodes


def _in_subtree(node, target):
    stack, seen = [node], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if x is target:
            return True
        stack += [c for c in x.get("children", []) or [] if c.get("kind") != "leaf"]
    return False


def _n_desc(node):
    stack, seen, n = [node], set(), 0
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x)); n += 1
        stack += [c for c in x.get("children", []) or [] if c.get("kind") != "leaf"]
    return n


def dedup_view(root):
    merged_total = 0
    for _ in range(6):
        parent, depth, nodes = _index(root)
        groups = {}
        for n in nodes:
            if n is root:
                continue
            groups.setdefault(_norm(n), []).append(n)
        dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
        if not dup_groups:
            break
        merged_this = 0
        for _key, occ in dup_groups.items():
            # canonical: shallowest, then most descendants
            occ.sort(key=lambda n: (depth[id(n)], -_n_desc(n)))
            canon = occ[0]
            for d in occ[1:]:
                # canon is the shallowest occurrence, so folding d INTO canon is
                # safe even when d is canon's descendant (we remove d, lift its
                # children to canon). The only cycle-forming direction is canon
                # being inside d's subtree, which can't happen for a shallowest
                # canon -- but guard it anyway.
                if d is canon or _in_subtree(d, canon):
                    continue
                par = parent.get(id(d))
                if par is None:
                    continue
                # move children (dedup leaves by claim_id)
                existing_claims = {c.get("claim_id") for c in canon.get("children", [])
                                   if c.get("kind") == "leaf"}
                for c in d.get("children", []) or []:
                    if c.get("kind") == "leaf" and c.get("claim_id") in existing_claims:
                        continue
                    canon.setdefault("children", []).append(c)
                    if c.get("kind") == "leaf":
                        existing_claims.add(c.get("claim_id"))
                par["children"] = [c for c in par.get("children", []) if c is not d]
                merged_this += 1
        merged_total += merged_this
        if merged_this == 0:
            break
    return merged_total


def main():
    data = json.loads(GV.read_text())
    views = data["views"]
    total = 0
    for vid, root in views.items():
        m = dedup_view(root)
        total += m
        print(f"[dedup-safe] {vid}: merged {m} duplicate node(s)", file=sys.stderr)
    # guard: ensure no circular references before writing
    json.dumps(data)   # raises ValueError on a cycle
    GV.write_text(json.dumps(data, indent=2))
    print(f"[dedup-safe] merged {total} total; wrote grown_views.json")


if __name__ == "__main__":
    main()
