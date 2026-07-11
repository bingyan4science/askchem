"""Explicit abstraction ladder + generality constraints for the living tree.

The tree must read general -> specific from root to leaf. Every node has a ROLE
with a rank; an edge parent->child is only valid when the parent is strictly more
general (lower rank) than the child, AND the child attaches under the *most
specific* governing ancestor available (no large rank "gaps" that leave a
specific phenomenon dangling directly under a framework next to a theory).

    0 open_root
    1 framework / law      (QM, thermodynamics, kinetics, conservation, Coulomb)
    2 theory / principle   (transition-state theory, Marcus theory, mass action)
    3 model                (VSEPR, band theory, Jablonski diagram)
    4 mechanism            (elementary motifs: oxidative addition, proton transfer)
    5 phenomenon / class   (named reactivity / material families: photocatalysis,
                            cross-coupling, nanomaterials)
    6 leaf                 (paper-grounded instance)

This module is pure logic (no LLM, no I/O). Repair takes a `choose_parent`
callback so the semantic decision (which governing node) can be supplied by an
LLM or a heuristic by the caller.
"""

from __future__ import annotations

ROLE_RANK = {
    "open_root": 0,
    "law": 1, "framework": 1,
    "theory": 2, "principle": 2,
    "model": 3,
    "mechanism": 4,
    "phenomenon": 5, "class": 5,
    "leaf": 6, "paper": 6,
}
RANK_LABEL = {0: "open_root", 1: "framework/law", 2: "theory", 3: "model",
              4: "mechanism", 5: "phenomenon/class", 6: "leaf"}

# Roles that may host paper leaves directly (the most-specific concept layer).
HOST_ROLES = {"mechanism", "phenomenon", "class"}
DEFAULT_RANK = 4


def role_of(node):
    return node.get("role") or node.get("kind") or ""


def rank_of(node):
    return ROLE_RANK.get(role_of(node), DEFAULT_RANK)


def is_leaf(node):
    return role_of(node) in ("leaf", "paper")


def valid_edge(parent, child):
    """An edge is valid when the parent is strictly more general than the child,
    OR they are at the same host level (a phenomenon/mechanism/class subtype nested
    under its broader sibling, e.g. photoredox catalysis under photocatalysis). The
    upper trunk (framework/theory/model) stays strictly ordered."""
    pr, cr = rank_of(parent), rank_of(child)
    if pr < cr:
        return True
    if pr == cr and role_of(parent) in HOST_ROLES and role_of(child) in HOST_ROLES:
        return True
    return False


def walk(root, fn, parent=None, _seen=None):
    """Pre-order walk calling fn(node, parent). Cycle-safe: a node is visited at
    most once, so an accidental parent/child cycle can't cause infinite recursion."""
    if _seen is None:
        _seen = set()
    if id(root) in _seen:
        return
    _seen.add(id(root))
    fn(root, parent)
    for c in root.get("children", []) or []:
        walk(c, fn, root, _seen)


def parent_map(root):
    pm = {}
    walk(root, lambda n, p: pm.__setitem__(id(n), p))
    return pm


def find_violations(root):
    """Return edge violations between internal (non-leaf) nodes.

    type = 'inversion'  -> parent is not strictly more general (rank>=child)
    type = 'gap'        -> child sits >=2 ranks below parent (too high; a more
                            specific governing node should sit in between)
    """
    out = []

    def visit(node, parent):
        if parent is None or is_leaf(node) or is_leaf(parent):
            return
        pr, cr = rank_of(parent), rank_of(node)
        if not valid_edge(parent, node) and pr >= cr:
            out.append({"type": "inversion", "parent": parent, "child": node,
                        "parent_rank": pr, "child_rank": cr})
        elif cr - pr >= 2:
            out.append({"type": "gap", "parent": parent, "child": node,
                        "parent_rank": pr, "child_rank": cr})
    walk(root, visit)
    return out


def candidate_parents(root, child, cur_parent):
    """Internal nodes under `cur_parent` that could govern `child`: rank strictly
    between the current parent's rank and the child's rank (more specific than the
    current parent, still more general than the child), excluding the child's own
    subtree. Restricting to cur_parent's subtree keeps the child within the same
    top-level framework - we nest it under a governing sibling, not relocate it."""
    cr = rank_of(child)
    pr = rank_of(cur_parent) if cur_parent is not None else 0
    scope = cur_parent if cur_parent is not None else root
    banned = set()
    walk(child, lambda n, p: banned.add(id(n)))
    cands = []

    def visit(node, _p):
        if node is child or id(node) in banned or is_leaf(node):
            return
        r = rank_of(node)
        if pr < r < cr:
            cands.append(node)
    walk(scope, visit)
    return cands


def detach(root, node, pmap=None):
    pmap = pmap or parent_map(root)
    p = pmap.get(id(node))
    if p is not None:
        p["children"] = [c for c in p.get("children", []) if c is not node]
    return p


def repair(root, choose_parent):
    """Re-attach violating children under a more-specific governing parent.

    choose_parent(child, cur_parent, candidates) -> a node from candidates, or
    None to leave the edge as-is. Only 'gap' (and resolvable 'inversion')
    violations are moved; unresolved ones are returned for reporting.
    """
    stats = {"moved": 0, "unresolved": 0}
    # snapshot violations first (tree mutates during repair)
    for v in find_violations(root):
        child, cur = v["child"], v["parent"]
        pmap = parent_map(root)
        if pmap.get(id(child)) is not cur:
            continue  # already moved
        cands = candidate_parents(root, child, cur)
        target = choose_parent(child, cur, cands) if cands else None
        if target is not None and target is not cur:
            detach(root, child, pmap)
            target.setdefault("children", []).append(child)
            stats["moved"] += 1
        else:
            stats["unresolved"] += 1
    return stats
