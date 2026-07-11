"""Semantic de-duplication of overlapping specific-layer nodes (all views).

The name-based dedup (concept_registry) only merges exact-normalized or
shared-head-prefix names. It misses same-concept nodes that are PARAPHRASES
("Photoinduced charge transfer" vs "Photogeneration of charge carriers"),
ACRONYMS ("PICT" = Plasmon-induced charge transfer), or scattered under
different parents. This pass:

  1. embeds each specific-layer node (mechanism/phenomenon/class) by name+definition,
  2. clusters embedding-similar nodes per view (cosine >= TAU),
  3. asks Gemini to keep only genuinely same-concept groups (guarding against
     merging distinct-but-similar mechanisms), and
  4. merges each group into a canonical node (children/leaves moved up), recording
     the rename map.

Runs on output/grown_views.json; re-apply with apply_to_db.py.

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/semantic_dedup.py [--tau 0.86]
    python3 living_taxonomy/apply_to_db.py --version v14
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import levels
import placement as pm
from incremental_build import _parse_json

OUT = _HERE / "output"
GV = OUT / "grown_views.json"
HOST_ROLES = {"mechanism", "phenomenon", "class"}

_SYS = (
    "You de-duplicate a chemistry mechanism/phenomenon taxonomy. You are given a "
    "cluster of nodes (name: definition) that are embedding-similar. Group together "
    "ONLY nodes that denote the SAME underlying concept - including acronyms and "
    "paraphrases (e.g. 'PICT' == 'Plasmon-induced charge transfer'; 'Photoinduced "
    "charge generation' == 'Photogeneration of charge carriers'). Keep genuinely "
    "DISTINCT mechanisms separate (e.g. 'Photoionization', 'Photoisomerization', and "
    "'Photoinduced charge transfer' are different concepts). Be conservative: when in "
    "doubt, keep separate.")


def _collect_specific(root):
    out = []
    levels.walk(root, lambda n, p: out.append(n)
                if p is not None and levels.role_of(n) in HOST_ROLES else None)
    return out


def _cluster(nodes, vecs, tau):
    n = len(nodes)
    if n < 2:
        return []
    S = vecs @ vecs.T
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    iu = np.triu_indices(n, 1)
    for i, j in zip(iu[0][S[iu] >= tau], iu[1][S[iu] >= tau]):
        parent[find(int(i))] = find(int(j))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(nodes[i])
    return [g for g in groups.values() if len(g) > 1]


def _prompt(cluster_nodes):
    lines = "\n".join(f"- {n['name']}: {(n.get('definition') or '')[:130]}"
                      for n in cluster_nodes)
    return (f"Cluster of similar nodes:\n{lines}\n\nReturn ONLY JSON: "
            '{"groups":[{"canonical":"<clearest full name>","members":["<exact name>",'
            '"<exact name>", ...]}]}. Include a group ONLY when >=2 of these are the '
            "same concept; omit singletons and distinct concepts.")


def _adjudicate(cluster_nodes):
    try:
        resp = pm._gemini_chat(_SYS, _prompt(cluster_nodes), max_time=120)
        return _parse_json(resp).get("groups", [])
    except Exception as e:
        print(f"[semdedup] cluster error: {e}", file=sys.stderr)
        return []


def _n_leaves(node):
    c = [0]
    levels.walk(node, lambda n, p: c.__setitem__(0, c[0] + (1 if levels.is_leaf(n) else 0)))
    return c[0]


def _in_subtree(anc, node):
    found = [False]
    levels.walk(anc, lambda n, p: found.__setitem__(0, found[0] or n is node))
    return found[0]


def dedupe_view(vid, root, tau):
    nodes = _collect_specific(root)
    if len(nodes) < 2:
        return {}
    vecs = pm._embed([f"{n['name']}. {n.get('definition','')}" for n in nodes], is_query=False)
    clusters = _cluster(nodes, vecs, tau)
    if not clusters:
        return {}
    print(f"[semdedup] {vid}: {len(clusters)} candidate cluster(s) from {len(nodes)} nodes",
          file=sys.stderr)

    # adjudicate clusters in parallel (I/O-bound LLM calls)
    groups = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(_adjudicate, c) for c in clusters]):
            groups += f.result()

    by_name = {}
    levels.walk(root, lambda n, p: by_name.__setitem__(n["name"], n)
                if p is not None else None)
    remap = {}
    for grp in groups:
        members = [by_name[m] for m in grp.get("members", []) if m in by_name]
        members = [m for m in members if levels.role_of(m) in HOST_ROLES]
        if len(members) < 2:
            continue
        # canonical = most general (lowest rank) then most leaves -> avoids inversions
        canonical = min(members, key=lambda m: (levels.rank_of(m), -_n_leaves(m)))
        cname = (grp.get("canonical") or canonical["name"]).strip()
        pmap = levels.parent_map(root)
        for m in members:
            if m is canonical or _in_subtree(m, canonical) or _in_subtree(canonical, m):
                continue
            for ch in list(m.get("children", []) or []):
                canonical.setdefault("children", []).append(ch)
            par = pmap.get(id(m))
            if par is not None:
                par["children"] = [c for c in par.get("children", []) if c is not m]
            remap[m["name"]] = cname
        # adopt the clearer canonical name
        if cname and cname != canonical["name"] and cname not in by_name:
            canonical["name"] = cname
    return remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.86)
    args = ap.parse_args()

    data = json.loads(GV.read_text())
    views = data["views"]
    shutil.copy(GV, GV.with_suffix(".json.pre_semdedup.bak"))

    total = {}
    for vid, root in views.items():
        r = dedupe_view(vid, root, args.tau)
        total.update(r)
        print(f"[semdedup] {vid}: merged {len(r)} node(s)", file=sys.stderr)

    GV.write_text(json.dumps(data))
    (OUT / "semantic_dedup.json").write_text(json.dumps(total, indent=2))
    # report residual generality violations introduced (should be ~0 since canonical
    # is the most-general member)
    inv = sum(sum(1 for v in levels.find_violations(r) if v["type"] == "inversion")
              for r in views.values())
    print(f"[semdedup] total merged: {len(total)}; inversions now: {inv}")
    print("[semdedup] next: python3 living_taxonomy/apply_to_db.py --version v14")


if __name__ == "__main__":
    main()
