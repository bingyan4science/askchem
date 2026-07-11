"""Incrementally GROW a living reaction tree from papers, one at a time.

This is the real "living taxonomy" construction (no hand-built seed):

  1. Randomly sample a paper; skip it if it is not reaction-related.
  2. Paper 1 -> ask Gemini to infer the underlying PRINCIPLES and MECHANISMS
     and build an initial tree (principles -> mechanisms -> reaction leaves).
  3. Paper N -> show Gemini the current tree skeleton + the new paper's
     reactions; it attaches leaves to existing nodes or ADDS new
     principle/mechanism branches (variable depth) when nothing fits.
  4. Repeat until 30 papers are processed.

Tree depth is NOT fixed: branches deepen only where the chemistry warrants.
Reads chemtree.db strictly read-only. LLM = Gemini via the NYU gateway.

Usage:
    export PORTKEY_API_KEY=...        # NYU AI gateway key
    python3 living_taxonomy/incremental_build.py --papers 30 --seed 7
    open living_taxonomy/output/tree_incremental.html
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_viz
import pilot_data
import placement as pm

OUT_DIR = _HERE / "output"
INC_DIR = OUT_DIR / "incremental"
DB = "file:" + str(_HERE.parent / "chemtree.db") + "?immutable=1"


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_reaction_papers(n, seed, min_reactions=3):
    """Randomly walk papers; keep only reaction-related ones until we have n."""
    conn = sqlite3.connect(DB, uri=True)
    all_dois = [r[0] for r in conn.execute("SELECT doi FROM sources")]
    random.Random(seed).shuffle(all_dois)
    kept = []
    for doi in all_dois:
        n_rx = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE source_doi=? AND claim_type='reaction'",
            (doi,),
        ).fetchone()[0]
        if n_rx >= min_reactions:
            kept.append(doi)
            if len(kept) >= n:
                break
    conn.close()
    return kept


def load_reactions(doi, max_rx=10):
    """Return (title, year, [reaction_text, ...]) for a paper."""
    conn = sqlite3.connect(DB, uri=True)
    row = conn.execute("SELECT title, year FROM sources WHERE doi=?", (doi,)).fetchone()
    title, year = (row or ("", 0))
    rxs = []
    for (data_json,) in conn.execute(
        "SELECT data FROM claims WHERE source_doi=? AND claim_type='reaction'", (doi,)
    ):
        try:
            d = json.loads(data_json)
        except (TypeError, json.JSONDecodeError):
            continue
        t = pilot_data._reaction_leaf_text(d)
        if t:
            rxs.append((bool(d.get("is_key_result")), t))
    rxs.sort(key=lambda x: not x[0])  # key results first
    return title or "", year or 0, [t for _, t in rxs[:max_rx]]


# ── tree ───────────────────────────────────────────────────────────────────────

class Tree:
    """Forest of principle/mechanism internal nodes with reaction leaves."""

    def __init__(self):
        self.nodes = {}      # id -> {id,name,kind,desc,parent,children:[ids],leaves:[]}
        self.roots = []      # top-level principle ids
        self._n = 0

    def new_id(self):
        self._n += 1
        return f"n{self._n}"

    def add_node(self, name, kind, desc, parent_id):
        nid = self.new_id()
        self.nodes[nid] = {"id": nid, "name": name, "kind": kind, "desc": desc,
                           "parent": parent_id, "children": [], "leaves": []}
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id]["children"].append(nid)
        else:
            self.roots.append(nid)
        return nid

    def attach_leaf(self, node_id, leaf):
        if node_id in self.nodes:
            self.nodes[node_id]["leaves"].append(leaf)
            return True
        return False

    def skeleton(self):
        """Indented text of internal nodes (no leaves) for the LLM prompt."""
        lines = []

        def walk(nid, depth):
            n = self.nodes[nid]
            nleaf = self._count_leaves(nid)
            lines.append("  " * depth +
                         f"{nid} [{n['kind']}] {n['name']} — {n['desc']}"
                         f" ({nleaf} leaves)")
            for c in n["children"]:
                walk(c, depth + 1)

        for r in self.roots:
            walk(r, 0)
        return "\n".join(lines) if lines else "(empty — no nodes yet)"

    def _count_leaves(self, nid):
        n = self.nodes[nid]
        c = len(n["leaves"])
        for ch in n["children"]:
            c += self._count_leaves(ch)
        return c

    def to_d3(self):
        def conv(nid):
            n = self.nodes[nid]
            children = [conv(c) for c in n["children"]]
            for lf in n["leaves"]:
                children.append({"name": lf["text"][:48], "kind": "leaf",
                                 "full": lf["text"], "doi": lf["doi"],
                                 "year": lf["year"], "score": 0})
            out = {"name": n["name"], "kind": n["kind"], "count": self._count_leaves(nid)}
            if children:
                out["children"] = children
            return out
        return {"name": "(unifying principle unknown)", "kind": "open_root",
                "count": sum(self._count_leaves(r) for r in self.roots),
                "children": [conv(r) for r in self.roots]}


# ── LLM step ─────────────────────────────────────────────────────────────────

_SYS = (
    "You are building a LIVING TAXONOMY of chemical reactivity from papers. "
    "Internal nodes are PRINCIPLES (fundamental governing concepts, e.g. "
    "'single-electron transfer', 'organometallic catalytic cycle') and "
    "MECHANISMS (elementary-step motifs under a principle). Leaves are "
    "specific reactions from papers. The tree has VARIABLE depth and NO single "
    "root — top-level nodes are fundamental principles. "
    "Given the current tree and a new paper's reactions, attach each reaction "
    "under the most appropriate EXISTING node, or ADD new principle/mechanism "
    "nodes when the reaction reflects a principle/mechanism not yet present "
    "(that is new knowledge / an exception). Prefer reusing existing nodes; "
    "only add nodes when genuinely warranted, and you may nest a new mechanism "
    "under a new principle. Respond with JSON only."
)

_USER_TMPL = """CURRENT TREE:
{skeleton}

NEW PAPER: {title} ({doi})
REACTIONS:
{reactions}

Return ONLY this JSON (no prose, no code fence):
{{
  "new_nodes": [
    {{"tmp_id": "t1", "parent_id": "<existing node id | a tmp_id above | null>",
      "kind": "principle|mechanism", "name": "...", "desc": "one sentence"}}
  ],
  "placements": [
    {{"reaction": 0, "target_id": "<existing id | tmp_id>",
      "is_exception": false, "reason": "short"}}
  ]
}}
- parent_id null = a NEW top-level principle.
- is_exception true when the reaction required creating new branch(es).
- every reaction index must appear exactly once in placements."""


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    return json.loads(text)


def llm_step(tree, title, doi, reactions):
    user = _USER_TMPL.format(
        skeleton=tree.skeleton(), title=title, doi=doi,
        reactions="\n".join(f"[{i}] {r}" for i, r in enumerate(reactions)),
    )
    raw = pm._gemini_chat(_SYS, user, max_time=120)
    if not raw:
        raise RuntimeError("empty LLM response (check PORTKEY_API_KEY / gateway)")
    return _parse_json(raw)


def apply_step(tree, parsed, reactions, doi, year):
    # 1) create new nodes, resolving tmp parents (iterate to topo-resolve).
    pending = list(parsed.get("new_nodes", []))
    tmp_to_real = {}
    safety = 0
    while pending and safety < 50:
        safety += 1
        progressed = False
        still = []
        for nd in pending:
            p = nd.get("parent_id")
            if p in (None, "null", ""):
                real_parent = None
            elif p in tree.nodes:
                real_parent = p
            elif p in tmp_to_real:
                real_parent = tmp_to_real[p]
            else:
                still.append(nd); continue          # parent tmp not ready yet
            rid = tree.add_node(nd.get("name", "?"), nd.get("kind", "mechanism"),
                                nd.get("desc", ""), real_parent)
            tmp_to_real[nd.get("tmp_id")] = rid
            progressed = True
        pending = still
        if not progressed:
            break
    # orphans: attach remaining as top-level principles
    for nd in pending:
        rid = tree.add_node(nd.get("name", "?"), nd.get("kind", "principle"),
                            nd.get("desc", ""), None)
        tmp_to_real[nd.get("tmp_id")] = rid

    # 2) attach leaves
    added_names = [tree.nodes[r]["name"] for r in tmp_to_real.values()]
    n_exc = 0
    placed = 0
    for pl in parsed.get("placements", []):
        idx = pl.get("reaction")
        if not isinstance(idx, int) or idx < 0 or idx >= len(reactions):
            continue
        tid = pl.get("target_id")
        tid = tmp_to_real.get(tid, tid)
        leaf = {"text": reactions[idx], "doi": doi, "year": year}
        if not tree.attach_leaf(tid, leaf):
            # fallback: attach to first root principle
            if tree.roots:
                tree.attach_leaf(tree.roots[0], leaf)
        placed += 1
        if pl.get("is_exception"):
            n_exc += 1
    return {"nodes_added": added_names, "n_exceptions": n_exc, "n_placed": placed}


# ── orchestrate ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-reactions", type=int, default=3)
    ap.add_argument("--max-rx-per-paper", type=int, default=10)
    args = ap.parse_args()

    INC_DIR.mkdir(parents=True, exist_ok=True)
    dois = sample_reaction_papers(args.papers, args.seed, args.min_reactions)
    print(f"[grow] sampled {len(dois)} reaction papers (seed={args.seed})",
          file=sys.stderr)

    tree = Tree()
    growth = []
    for i, doi in enumerate(dois, 1):
        title, year, rxs = load_reactions(doi, args.max_rx_per_paper)
        if not rxs:
            continue
        try:
            parsed = llm_step(tree, title, doi, rxs)
            info = apply_step(tree, parsed, rxs, doi, year)
        except Exception as e:
            print(f"[grow] paper {i} FAILED: {e}", file=sys.stderr)
            growth.append({"step": i, "doi": doi, "error": str(e)})
            continue
        n_nodes = len(tree.nodes)
        print(f"[grow] {i:2d}/{len(dois)} +{len(info['nodes_added'])} nodes "
              f"(total {n_nodes}), {info['n_placed']} leaves, "
              f"{info['n_exceptions']} exc | {title[:50]}", file=sys.stderr)
        growth.append({"step": i, "doi": doi, "title": title,
                       "n_reactions": len(rxs), "total_nodes": n_nodes,
                       **info})
        (INC_DIR / f"step_{i:02d}.json").write_text(json.dumps(tree.to_d3(), indent=2))

    (OUT_DIR / "tree_incremental.json").write_text(json.dumps(tree.to_d3(), indent=2))
    (INC_DIR / "growth_log.json").write_text(json.dumps(growth, indent=2))

    n_leaves = sum(tree._count_leaves(r) for r in tree.roots)
    n_principle = sum(1 for n in tree.nodes.values() if n["kind"] == "principle")
    n_mech = sum(1 for n in tree.nodes.values() if n["kind"] == "mechanism")
    subtitle = (f"{n_leaves} reactions &middot; {len(dois)} papers &middot; "
                f"{n_principle} principles, {n_mech} mechanisms (grown incrementally)")
    out_html = OUT_DIR / "tree_incremental.html"
    build_viz.render_html(tree.to_d3(), "grown from papers", subtitle, out_html)
    print(f"\n[grow] {n_principle} principles, {n_mech} mechanisms, {n_leaves} leaves")
    print(f"[grow] wrote {out_html}\n[grow] open {out_html}", file=sys.stderr)


if __name__ == "__main__":
    main()
