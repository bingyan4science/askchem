"""Canonical concept registry: one node per concept, at scale.

The taxonomy duplicated concepts because nodes were keyed by normalized name only
("Marcus Theory" != "Marcus theory of electron transfer"). This module canonicalizes
concepts by (normalized name | synonym | embedding similarity) AND role rank, so the
same concept reuses one node instead of spawning duplicates.

Two uses:
  * at node-creation time (placement): `ConceptRegistry.canonical(name, rank, vec)`
    returns an existing concept to reuse, or None.
  * backfill: `dedupe_views(views)` merges existing same-rank near-duplicates in
    output/grown_views.json (e.g. the two Marcus-theory nodes -> one), moving their
    children/leaves onto the canonical and recording the rename map.

Only nodes of the SAME role rank are merged, so a theory and its application (a
phenomenon that merely shares a name) are never collapsed - that is the leveling
job (levels.repair), not de-duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import levels
from scaffold_builder import _norm


def _text(node):
    d = (node.get("definition") or "").strip()
    return f"{node.get('name','')}. {d}".strip()


def _dnorm(name):
    """Dedup normalization that KEEPS head words like 'theory'/'law' (unlike
    scaffold _norm), so 'Marcus theory' -> ['marcus','theory'] and the prefix test
    can require a >=2-token head. Only true stopwords are dropped."""
    import re as _re
    import unicodedata as _u
    s = _u.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    s = _re.sub(r"[^a-z0-9 ]", " ", s)
    stop = {"the", "of", "a", "an", "and", "to", "for", "in", "on"}
    return [t for t in s.split() if t and t not in stop]


class ConceptRegistry:
    def __init__(self, tau=0.9):
        self.tau = tau
        self.concepts = []   # {name, norm, rank, vec, node}

    def canonical(self, name, rank, vec=None):
        nm = _norm(name)
        for c in self.concepts:
            if c["rank"] == rank and nm and c["norm"] == nm:
                return c
        if vec is not None:
            best, bestsim = None, self.tau
            for c in self.concepts:
                if c["rank"] == rank and c["vec"] is not None:
                    sim = float(np.dot(vec, c["vec"]))
                    if sim >= bestsim:
                        best, bestsim = c, sim
            return best
        return None

    def register(self, name, rank, vec=None, node=None):
        c = {"name": name, "norm": _norm(name), "rank": rank, "vec": vec, "node": node}
        self.concepts.append(c)
        return c


# ── backfill: merge existing same-rank near-duplicates within each view ──────────

def _internal_nodes(root):
    out = []
    levels.walk(root, lambda n, p: (out.append((n, p)) if p is not None
                                    and not levels.is_leaf(n) else None))
    return out


def _n_leaves(node):
    c = {"n": 0}
    levels.walk(node, lambda n, p: c.__setitem__("n", c["n"] + (1 if levels.is_leaf(n) else 0)))
    return c["n"]


def dedupe_view(root, vecs, tau=0.9):
    """Merge same-rank near-duplicate internal nodes within one view tree.
    `vecs` maps id(node)->vector. Returns {dupe_name: canonical_name} remap."""
    nodes = [n for n, _ in _internal_nodes(root)]
    n = len(nodes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    def _mergeable(a, b):
        if levels.rank_of(a) != levels.rank_of(b):
            return False
        na, nb = _norm(a["name"]), _norm(b["name"])
        if na and na == nb:                       # identical concept (exact)
            return True
        # merge when the shorter name is a >=2-token PREFIX (head) of the longer one
        # ("marcus theory" -> "marcus theory of electron transfer"). A LEADING
        # qualifier marks a distinct subtype ("associative ligand substitution"),
        # and token-sharing siblings ("lewis acid base" vs "bronsted lowry acid
        # base") are not prefixes - both stay distinct. >=2 tokens avoids a bare
        # generic head ("catalysis") swallowing specifics.
        da, dbb = _dnorm(a["name"]), _dnorm(b["name"])
        short, lng = (da, dbb) if len(da) <= len(dbb) else (dbb, da)
        return len(short) >= 2 and lng[:len(short)] == short

    # First-token BLOCKING: any mergeable pair (exact-norm or shared-head prefix)
    # shares its first _dnorm token, so only compare within same-first-token blocks.
    # Cuts O(n^2) across all nodes to O(sum of block^2).
    blocks = {}
    for i in range(n):
        dn = _dnorm(nodes[i]["name"])
        blocks.setdefault(dn[0] if dn else f"_{i}", []).append(i)
    for idxs in blocks.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                i, j = idxs[x], idxs[y]
                if _mergeable(nodes[i], nodes[j]):
                    union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(nodes[i])

    pmap = levels.parent_map(root)
    remap = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = max(members, key=lambda nd: (_n_leaves(nd), len(nd.get("children", []))))
        for nd in members:
            if nd is canonical:
                continue
            remap[nd["name"]] = canonical["name"]
            # move children onto the canonical, then detach the duplicate
            for ch in nd.get("children", []) or []:
                canonical.setdefault("children", []).append(ch)
            p = pmap.get(id(nd))
            if p is not None:
                p["children"] = [c for c in p.get("children", []) if c is not nd]
    return remap


def dedupe_views(views, tau=0.9):
    """Embed every internal node once, then dedupe each view tree."""
    import placement as pm
    allnodes = []
    for root in views.values():
        allnodes += [n for n, _ in _internal_nodes(root)]
    # embed NAMES only - definitions dilute the synonym signal ("Marcus Theory" vs
    # "Marcus theory of electron transfer" are near-identical names, different defs).
    texts = [n.get("name", "") for n in allnodes]
    vmap = {}
    if texts:
        vecs = pm._embed(texts, is_query=False)
        for n, v in zip(allnodes, vecs):
            vmap[id(n)] = v
    total = {}
    for vid, root in views.items():
        r = dedupe_view(root, vmap, tau=tau)
        if r:
            print(f"[registry] {vid}: merged {len(r)} duplicate(s)", file=sys.stderr)
        total.update(r)
    return total
