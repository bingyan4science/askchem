"""Tier-2 consolidation/commit pass for the living taxonomy.

Insertion (grow_onto_scaffold / batch_place) only records PROPOSED branches.
This pass governs taxonomy evolution: it merges near-duplicate proposals, folds
proposals that match an existing host into it, promotes the genuinely-new ones to
committed branches, and bumps the version. One Gemini call per view.

Usage:
    python3 living_taxonomy/consolidate.py            # uses output/grown_views.json
    python3 living_taxonomy/apply_to_db.py --version v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_viz
import grow_onto_scaffold as g
import incremental_build as ib
import placement as pm
import view_layers as vl

OUT = g.OUT

_SYS = ("You curate a chemistry taxonomy. You are given the COMMITTED host "
        "categories and a list of PROPOSED new branches (with example members). "
        "For each proposed branch decide: MERGE it into the single best-matching "
        "COMMITTED host or another proposed branch (same mechanism/family); "
        "PROMOTE it (genuinely new family, keep as its own branch); or DROP it "
        "(too vague/out-of-scope - its members fold into its parent). Eliminate "
        "near-duplicates aggressively but never merge distinct mechanisms.")


def _index(top):
    name_node, parent = {}, {}
    def w(n):
        name_node[n["name"]] = n
        for c in n.get("children", []):
            parent[c["name"]] = n
            w(c)
    w(top)
    return name_node, parent


def _collect(top):
    committed, proposed = [], []
    def w(n):
        for c in n.get("children", []):
            if c.get("kind") == "leaf":
                continue
            if c.get("proposed"):
                ex = [x["name"] for x in c.get("children", []) if x.get("kind") == "leaf"][:5]
                proposed.append({"name": c["name"], "examples": ex,
                                 "n": sum(1 for x in c.get("children", []) if x.get("kind") == "leaf")})
            elif c.get("shared") is False:
                committed.append(c["name"])
            w(c)
    w(top)
    return committed, proposed


_PROP_CHUNK = 60      # proposed branches per LLM call (keeps the prompt bounded)
_MIN_PROP_MEMBERS = 3  # proposed branches smaller than this are folded into their
                       # parent WITHOUT an LLM call. At full scale placement spawns
                       # tens of thousands of 1-2 member "proposed" branches (noise);
                       # LLM-adjudicating all of them is intractable, so only branches
                       # with >= this many members go to the LLM.


def consolidate_view(view, top):
    committed, proposed = _collect(top)
    if not proposed:
        return {"merged": 0, "promoted": 0, "dropped": 0}

    # ── pre-prune: fold tiny proposed branches into their parent (no LLM) ──
    name_node, parent = _index(top)
    folded = 0
    big = []
    for p in proposed:
        node = name_node.get(p["name"])
        if node is None:
            continue
        if p["n"] < _MIN_PROP_MEMBERS:
            par = parent.get(p["name"])
            if par is not None:
                leaves = [c for c in node.get("children", []) if c.get("kind") == "leaf"]
                par.setdefault("children", []).extend(leaves)
                par["children"] = [c for c in par.get("children", []) if c is not node]
            folded += 1
        else:
            big.append(p)
    if folded:
        print(f"[consolidate] {view}: folded {folded} tiny proposed branches "
              f"(<{_MIN_PROP_MEMBERS} members); {len(big)} sizable remain", file=sys.stderr)

    # re-index after the structural prune
    name_node, parent = _index(top)
    host_list = "\n".join(f"- {h}" for h in committed)
    # Chunk the sizable proposed branches; committed hosts stay as full context.
    decisions = []
    for i in range(0, len(big), _PROP_CHUNK):
        chunk = big[i:i + _PROP_CHUNK]
        prop_list = "\n".join(
            f'- "{p["name"]}" ({p["n"]} members; e.g. {", ".join(p["examples"][:3])})'
            for p in chunk)
        user = (f"COMMITTED hosts in the {view} tree:\n{host_list}\n\n"
                f"PROPOSED branches to curate:\n{prop_list}\n\n"
                'Return ONLY JSON {"decisions":[{"branch":"<proposed name>",'
                '"action":"merge|promote|drop","target":"<exact committed host or '
                'other proposed name; required for merge>"}]}')
        try:
            decisions += ib._parse_json(pm._gemini_chat(_SYS, user, max_time=120)).get("decisions", [])
        except Exception as e:
            print(f"[consolidate] {view} chunk {i} ERROR {e}", file=sys.stderr)

    dmap = {d.get("branch"): d for d in decisions}
    merged = promoted = dropped = 0
    for p in big:
        d = dmap.get(p["name"], {"action": "promote"})
        node = name_node.get(p["name"])
        if node is None:
            continue
        action = d.get("action", "promote")
        if action == "promote":
            node["proposed"] = False
            promoted += 1
            continue
        # merge or drop -> move this branch's leaves elsewhere, detach the branch
        tgt_name = d.get("target") if action == "merge" else None
        target = name_node.get(tgt_name) if tgt_name else None
        if target is None:                       # drop / unresolved -> fold into parent
            target = parent.get(p["name"])
            dropped += 1
        else:
            merged += 1
        leaves = [c for c in node.get("children", []) if c.get("kind") == "leaf"]
        if target is not None:
            target.setdefault("children", []).extend(leaves)
        par = parent.get(p["name"])
        if par is not None:
            par["children"] = [c for c in par.get("children", []) if c is not node]
    return {"merged": merged, "promoted": promoted, "dropped": dropped,
            "folded": folded, "before": len(proposed)}


_TRUNK_KINDS = {"law", "framework", "theory", "model"}
_HOST_KINDS = {"mechanism", "class"}

_REPARENT_SYS = (
    "You curate a chemistry taxonomy hierarchy. Internal nodes are principles/"
    "frameworks/theories/models (the trunk) and mechanism/technique hosts. A host "
    "must hang under the THEORY/PRINCIPLE that governs it - not under an unrelated "
    "host. You are given hosts that are currently nested under another host; for "
    "each, KEEP it only if it is genuinely a sub-mechanism/sub-method of its "
    "current parent, otherwise REPARENT it under the best trunk node.")


def reparent_view(view, top):
    """Re-attach hosts mis-nested under another host to the accurate trunk node."""
    name_node, parent = _index(top)
    trunk = [n for n, nd in name_node.items() if nd.get("kind") in _TRUNK_KINDS]
    pairs = []

    def w(n):
        for c in n.get("children", []):
            if c.get("kind") in _HOST_KINDS and n.get("kind") in _HOST_KINDS:
                ex = [x["name"] for x in c.get("children", []) if x.get("kind") == "leaf"][:3]
                pairs.append({"child": c["name"], "parent": n["name"], "examples": ex})
            w(c)
    w(top)
    if not pairs:
        return {"reparented": 0, "kept": 0}

    user = (f"TRUNK theories/principles (valid new parents):\n"
            + "\n".join(f"- {t}" for t in trunk) + "\n\nHOSTS nested under another host:\n"
            + "\n".join(f'- "{p["child"]}" (under "{p["parent"]}"'
                        + (f"; e.g. {', '.join(p['examples'][:2])}" if p["examples"] else "")
                        + ")" for p in pairs)
            + '\n\nReturn ONLY JSON {"decisions":[{"child":"...","action":"keep|reparent",'
              '"parent":"<exact trunk name; required for reparent>"}]}')
    try:
        resp = pm._gemini_chat(_REPARENT_SYS, user, max_time=120)
        decisions = {d.get("child"): d for d in ib._parse_json(resp).get("decisions", [])}
    except Exception as e:
        print(f"[reparent] {view} ERROR {e}", file=sys.stderr)
        return {"reparented": 0, "kept": 0, "error": str(e)}

    reparented = kept = 0
    for p in pairs:
        d = decisions.get(p["child"], {"action": "keep"})
        if d.get("action") != "reparent":
            kept += 1
            continue
        child = name_node.get(p["child"])
        newp = name_node.get(d.get("parent"))
        oldp = parent.get(p["child"])
        if child is None or newp is None or oldp is None or newp is child:
            kept += 1
            continue
        oldp["children"] = [c for c in oldp.get("children", []) if c is not child]
        newp.setdefault("children", []).append(child)
        parent[p["child"]] = newp
        reparented += 1
    return {"reparented": reparented, "kept": kept, "before": len(pairs)}


def main():
    data = json.loads((OUT / "grown_views.json").read_text())
    views = data["views"]
    summary = {}
    for view, top in views.items():
        summary[view] = consolidate_view(view, top)
        rp = reparent_view(view, top)
        summary[view]["reparent"] = rp
        vl._count(top)
        print(f"[consolidate] {view}: {summary[view]}", file=sys.stderr)
    sub = data.get("subtitle", "") + " · consolidated"
    (OUT / "grown_views.json").write_text(json.dumps({"views": views, "subtitle": sub}, indent=2))
    build_viz.render_html(views["by_reaction_type"], "chemistry living tree (consolidated)",
                          sub, OUT / "scaffold_multiview.html", views=views)
    print(f"[consolidate] wrote grown_views.json + scaffold_multiview.html")
    print(f"[consolidate] next: python3 living_taxonomy/apply_to_db.py --version v2")


if __name__ == "__main__":
    main()
