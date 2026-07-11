"""Tidy flat sibling lists: aggressive merge + family grouping (all views).

After semantic_dedup, big parents (e.g. "Chemical Kinetics") still hold long flat
lists of specific mechanisms/phenomena. This pass, PER PARENT:

  1. embed-clusters the parent's specific-layer children into thematic families,
  2. for each candidate family asks Gemini to (a) MERGE synonyms and narrower
     special-cases into the broader node (e.g. 'radical beta-scission' -> 'radical
     fragmentation'; 'PICT' -> its full name), and (b) if the remaining members form
     a coherent FAMILY, propose one family node to group them under (e.g.
     photoexcitation / photodissociation / photoionization -> 'Photochemical primary
     processes'),
  3. applies the merges and inserts the family node (children re-homed under it).

Family nodes keep the most-general member role so the ladder holds (host-layer
same-rank nesting is allowed by levels.valid_edge). Runs on output/grown_views.json.

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/combine_nodes.py [--tau 0.70 --min-children 6]
    python3 living_taxonomy/apply_to_db.py --version v15
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
_RANK_ROLE = {4: "mechanism", 5: "phenomenon"}

_SYS = (
    "You tidy a flat list of sibling chemistry mechanisms/phenomena under one parent. "
    "Do TWO things:\n"
    "(1) MERGE entries that are the same concept OR where one is a narrower special-"
    "case/mode of another - merge into the BROADER name (e.g. 'radical beta-scission' "
    "is a mode of 'radical fragmentation' -> merge to 'radical fragmentation'; 'PICT' "
    "== 'plasmon-induced charge transfer').\n"
    "(2) After merging, if several remaining entries form a coherent FAMILY, propose "
    "ONE family node to group them under (e.g. photoexcitation, photodissociation, "
    "photoionization -> 'Photochemical primary processes'; various radical H-abstraction/"
    "addition steps -> 'Radical chain propagation'). Use a real chemical family name. "
    "Only group entries that genuinely belong together; leave truly unrelated ones out.")


def _children_hosts(node):
    return [c for c in node.get("children", []) or [] if levels.role_of(c) in HOST_ROLES]


def _n_leaves(node):
    c = [0]
    levels.walk(node, lambda n, p: c.__setitem__(0, c[0] + (1 if levels.is_leaf(n) else 0)))
    return c[0]


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


def _prompt(cluster):
    lines = "\n".join(f"- {n['name']}: {(n.get('definition') or '')[:120]}" for n in cluster)
    return (f"Sibling entries:\n{lines}\n\nReturn ONLY JSON: "
            '{"merge":[{"canonical":"<broader name>","members":["<exact>","<exact>"]}],'
            '"families":[{"name":"<family node name>","members":["<exact>","<exact>"]}]}. '
            "Omit empty arrays' contents; only include merges of >=2 and families of >=2.")


def _adjudicate(cluster):
    try:
        return _parse_json(pm._gemini_chat(_SYS, _prompt(cluster), max_time=120))
    except Exception as e:
        print(f"[combine] cluster error: {e}", file=sys.stderr)
        return {}


def _family_role(members):
    ranks = [levels.rank_of(m) for m in members]
    return _RANK_ROLE.get(min(ranks), "mechanism")


def combine_view(vid, root, tau, min_children):
    # collect parents worth tidying (many host children)
    targets = []
    levels.walk(root, lambda n, p: targets.append(n)
                if len(_children_hosts(n)) >= min_children else None)
    if not targets:
        return {"merged": 0, "families": 0}

    # build candidate clusters across all target parents, adjudicate in parallel
    jobs = []          # (parent, cluster_nodes)
    for parent in targets:
        kids = _children_hosts(parent)
        vecs = pm._embed([f"{k['name']}. {k.get('definition','')}" for k in kids], is_query=False)
        for cl in _cluster(kids, vecs, tau):
            jobs.append((parent, cl))
    if not jobs:
        return {"merged": 0, "families": 0}
    print(f"[combine] {vid}: {len(jobs)} candidate clusters under {len(targets)} parents",
          file=sys.stderr)

    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_adjudicate, cl): i for i, (_, cl) in enumerate(jobs)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()

    merged = families = 0
    for (parent, cluster), res in zip(jobs, results):
        if not res:
            continue
        by_name = {c["name"]: c for c in _children_hosts(parent)}   # refresh each cluster
        # --- (1) merges ---
        for grp in res.get("merge", []):
            members = [by_name[m] for m in grp.get("members", []) if m in by_name]
            if len(members) < 2:
                continue
            canonical = min(members, key=lambda m: (levels.rank_of(m), -_n_leaves(m)))
            cname = (grp.get("canonical") or canonical["name"]).strip()
            for m in members:
                if m is canonical:
                    continue
                for ch in list(m.get("children", []) or []):
                    canonical.setdefault("children", []).append(ch)
                parent["children"] = [c for c in parent.get("children", []) if c is not m]
                merged += 1
            if cname and cname != canonical["name"]:
                canonical["name"] = cname
            by_name = {c["name"]: c for c in _children_hosts(parent)}
        # --- (2) family grouping ---
        for fam in res.get("families", []):
            members = [by_name[m] for m in fam.get("members", []) if m in by_name]
            fname = (fam.get("name") or "").strip()
            if len(members) < 2 or not fname or fname in by_name:
                continue
            famnode = {"name": fname, "kind": _family_role(members), "shared": False,
                       "children": []}
            for m in members:
                famnode["children"].append(m)
                parent["children"] = [c for c in parent.get("children", []) if c is not m]
            parent.setdefault("children", []).append(famnode)
            families += 1
            by_name = {c["name"]: c for c in _children_hosts(parent)}
    return {"merged": merged, "families": families}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--min-children", type=int, default=6)
    args = ap.parse_args()

    data = json.loads(GV.read_text())
    views = data["views"]
    shutil.copy(GV, GV.with_suffix(".json.pre_combine.bak"))

    tot_m = tot_f = 0
    for vid, root in views.items():
        r = combine_view(vid, root, args.tau, args.min_children)
        tot_m += r["merged"]; tot_f += r["families"]
        print(f"[combine] {vid}: merged {r['merged']}, +{r['families']} family nodes",
              file=sys.stderr)

    GV.write_text(json.dumps(data))
    inv = sum(sum(1 for v in levels.find_violations(r) if v["type"] == "inversion")
              for r in views.values())
    print(f"[combine] total: merged {tot_m}, +{tot_f} families; inversions now: {inv}")
    print("[combine] next: python3 living_taxonomy/apply_to_db.py --version v15")


if __name__ == "__main__":
    main()
