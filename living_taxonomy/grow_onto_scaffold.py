"""Insert test-paper leaves onto the multi-view scaffold.

Populates the curated scaffold with real paper-grounded leaves so the tree has
searchable content. For each test paper and each view, it extracts the view's
leaf entities (reactions / substances) and attaches each under the most similar
HOST node (embedding nearest-host; below a threshold -> an Exceptions branch).

Read-only on chemtree.db. Embedding placement is LLM-free (fast); pass
``--use-llm`` later for adjudication if desired.

Usage:
    python3 living_taxonomy/grow_onto_scaffold.py --papers 15
    open living_taxonomy/output/scaffold_multiview.html
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_viz
import incremental_build as ib
import pilot_data
import placement as pm
import view_layers as vl

OUT = _HERE / "output"
# All four meaningful views go through the same placement path at scale. (The
# 300-paper pilot batch-placed only reaction+substance and live-placed the rest;
# for 3k/full we place all four uniformly.)
GROW_VIEWS = ["by_reaction_type", "by_substance_class", "by_mechanism", "by_technique"]
_CODE = re.compile(r"^[0-9]+[a-z]{0,3}$", re.I)   # filter compound codes like "3aa"


def sample_fulltext_papers(n=300, seed=11, min_claims=25, min_reaction=0,
                           exclude_placed=False):
    """Pick papers (reuse existing claims): >= min_claims total claims.

    Full-PDF extractions return a few dozen claims vs <10 for abstract-only, so
    claim count is the practical proxy: raise `min_claims` for full-PDF-only,
    lower it (e.g. 3) to fold in abstract-only papers. `min_reaction` defaults to
    0 now: the old rx>=3 gate biased the sample toward reaction papers, starving
    the substance / technique / mechanism views. Per-view leaf availability is
    handled downstream by pilot_data.load_leaves (a paper only contributes leaves
    to views where it has relevant claims).

    `exclude_placed=True` drops DOIs already present in `taxonomy_leaves`, so an
    expansion run only places *new* papers (incremental grow) instead of
    reprocessing the corpus already in the current tree."""
    conn = pilot_data._connect()
    rows = conn.execute(
        "SELECT source_doi, COUNT(*) c, "
        "SUM(CASE WHEN claim_type='reaction' THEN 1 ELSE 0 END) rx "
        "FROM claims GROUP BY source_doi HAVING c >= ? AND rx >= ?",
        (min_claims, min_reaction)).fetchall()
    placed = set()
    if exclude_placed:
        try:
            placed = {r[0] for r in conn.execute(
                "SELECT DISTINCT doi FROM taxonomy_leaves").fetchall()}
        except Exception:
            placed = set()
    conn.close()
    dois = [r[0] for r in rows]
    if placed:
        before = len(dois)
        dois = [d for d in dois if d not in placed]
        print(f"[sample] exclude_placed: {before} -> {len(dois)} "
              f"(dropped {before - len(dois)} already in tree)", file=sys.stderr)
    random.Random(seed).shuffle(dois)
    return dois[:n]


def host_descs(view, nodes):
    if view in ("by_reaction_type", "by_mechanism"):
        return {n["name"]: n["desc"] for n in nodes.values()
                if n["kind"] == "mechanism"}
    raw = json.loads((vl.VL / f"{view}_raw.json").read_text())
    return {h["name"]: h.get("definition", "") for h in raw}


def short_label(view, lf):
    t = lf["text"]
    m = re.match(r"(?:Reaction|Substance|Mechanism|Technique):\s*([^.]+)", t)
    return (m.group(1) if m else t)[:42]


def clean_leaves(view, leaves):
    out, seen = [], set()
    for lf in leaves:
        lab = short_label(view, lf).strip()
        if not lab or lab.lower() in seen:
            continue
        if view == "by_substance_class" and _CODE.match(lab.replace(" ", "")):
            continue                      # skip opaque product codes
        seen.add(lab.lower())
        lf["label"] = lab
        out.append(lf)
    return out


def grow_view(view, nodes, host_node_map, dois, attach, exception, max_leaves):
    names = list(host_node_map)
    if not names:
        return 0, 0, None
    descs = host_descs(view, nodes)
    hv = pm._embed([f"{n}. {descs.get(n,'')}" for n in names], is_query=False)

    leaves = clean_leaves(view, pilot_data.load_leaves(view, dois, max_leaves))
    if not leaves:
        return 0, 0, None
    lv = pm._embed([lf["text"] for lf in leaves], is_query=True)
    sims = lv @ hv.T

    exc_node = {"name": "Exceptions (proposed new branches)", "kind": "exception",
                "count": 0, "shared": False, "children": []}
    n_attach = n_exc = 0
    for i, lf in enumerate(leaves):
        j = int(np.argmax(sims[i]))
        s = float(sims[i, j])
        leafnode = {"name": lf["label"], "kind": "leaf", "claim_id": lf["claim_id"],
                    "full": lf["text"], "doi": lf["doi"], "year": lf["year"],
                    "score": round(s, 3)}
        if s <= exception:
            exc_node["children"].append(leafnode)
            n_exc += 1
        else:
            host_node_map[names[j]].setdefault("children", []).append(leafnode)
            n_attach += 1
    return n_attach, n_exc, exc_node if exc_node["children"] else None


_LLM_SYS = (
    "You file a specific paper finding under the MOST SPECIFIC host category that "
    "governs it in a chemistry knowledge tree. Prefer the candidate host whose "
    "mechanism/phenomenon governs the finding - a different example of the same "
    "mechanism still belongs there. Propose a NEW branch ONLY when the finding's "
    "governing family is genuinely absent (then give the new branch's ROLE on the "
    "ladder framework>theory>model>mechanism>phenomenon, and name the most-specific "
    "EXISTING host/theory that governs it as its parent - never a general framework "
    "when a governing mechanism/theory exists). Be accurate, not over-strict.")


def _index_by_name(top):
    """name -> node over an entire view tree (for resolving proposal parents)."""
    out = {}

    def walk(n):
        out[n["name"]] = n
        for c in n.get("children", []):
            walk(c)
    walk(top)
    return out


def _build_prompt(view, host_block, lvs):
    findings = "\n".join(f"[{i}] {lf['text'][:200]}" for i, lf in enumerate(lvs))
    return (f"HOSTS for the {view} tree (choose by EXACT name):\n{host_block}\n\n"
            f"For each finding, pick the best-fitting host by EXACT name, or "
            f"propose a new branch ONLY if its governing family is genuinely "
            f"absent from the hosts.\nFINDINGS:\n{findings}\n\nReturn ONLY JSON: "
            '{"assign":[{"i":0,"host":"<exact host name>"} OR '
            '{"i":0,"propose":{"branch":"short concept name",'
            '"role":"phenomenon|mechanism","parent":"<exact most-specific governing '
            'host/theory name>"}}]}')


def _gemini_assign(user):
    """One Gemini call -> {leaf_index: assignment}. Thread-safe (curl subprocess)."""
    try:
        resp = pm._gemini_chat(_LLM_SYS, user, max_time=90)
        return {x.get("i"): x for x in ib._parse_json(resp).get("assign", [])}
    except Exception:
        return {}


def llm_grow_view(view, nodes, host_node_map, view_top, dois, per_paper, workers=6):
    """Stateless per-paper decisions with the Gemini calls run in PARALLEL
    (insertion has no shared state), then a GLOBAL apply with proposal dedup.
    Local embeddings run in the main thread (model is not thread-safe); only the
    I/O-bound LLM calls are parallelized."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    descs = host_descs(view, nodes)
    names = list(host_node_map)
    host_block = "\n".join(f"- {n}: {descs.get(n,'')}" for n in names)
    host_vecs = pm._embed([f"{n}. {descs.get(n,'')}" for n in names], is_query=False)

    leaves = clean_leaves(view, pilot_data.load_leaves(
        view, dois, max_leaves=per_paper * len(dois) * 4, per_paper=per_paper * 4))
    by_doi = defaultdict(list)
    for lf in leaves:
        by_doi[lf["doi"]].append(lf)

    # Stage 1a - main thread: local embeddings -> nearest-host (top1) + prompt
    prepared = []
    for doi, lvs in by_doi.items():
        lvs = lvs[:per_paper]
        lvecs = pm._embed([lf["text"] for lf in lvs], is_query=True)
        sims = lvecs @ host_vecs.T
        top1s = [names[int(np.argmax(sims[i]))] for i in range(len(lvs))]
        prepared.append({"lvs": lvs, "top1s": top1s,
                         "user": _build_prompt(view, host_block, lvs)})
    print(f"[grow] {view}: {len(prepared)} papers prepared, calling Gemini "
          f"({workers} parallel)…", file=sys.stderr)

    # Stage 1b - parallel Gemini calls (I/O-bound, stateless)
    amaps = [None] * len(prepared)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_gemini_assign, pr["user"]): i
                for i, pr in enumerate(prepared)}
        for f in as_completed(futs):
            amaps[futs[f]] = f.result()
            done += 1
            if done % 25 == 0:
                print(f"[grow] {view} decided {done}/{len(prepared)}", file=sys.stderr)

    # Stage 1c - assemble flat decisions
    decisions = []
    for pr, amap in zip(prepared, amaps):
        for i, lf in enumerate(pr["lvs"]):
            a = (amap or {}).get(i, {})
            decisions.append({"leaf": lf, "host": a.get("host", ""),
                              "propose": a.get("propose") or {}, "top1": pr["top1s"][i]})

    # Stage 2 - global apply (shared with the batch path)
    return apply_decisions(view, host_node_map, view_top, decisions)


def apply_decisions(view, host_node_map, view_top, decisions):
    """GLOBAL apply: attach leaf placements + globally-deduped proposed branches.
    Shared by the parallel-sync and Gemini-batch placement paths. Each decision:
    {leaf:{claim_id,label,text,doi,year}, host:str, propose:{branch,parent}, top1:str}."""
    import levels
    name_node = _index_by_name(view_top)
    norm_index = {vl.sb._norm(k): v for k, v in name_node.items()}
    norm_host = {vl.sb._norm(n): n for n in host_node_map}
    pmap = levels.parent_map(view_top)

    def governing_parent(pnode, new_rank):
        """Climb until the parent is strictly more general than the new node."""
        guard = 0
        while pnode is not None and levels.rank_of(pnode) >= new_rank and guard < 12:
            pnode = pmap.get(id(pnode)); guard += 1
        return pnode

    def resolve_host(label):
        if not label:
            return None
        if label in host_node_map:
            return host_node_map[label]
        return host_node_map.get(norm_host.get(vl.sb._norm(label), ""))

    proposed, prop_log = {}, []
    na = ne = 0
    for dec in decisions:
        lf = dec["leaf"]
        leafnode = {"name": lf["label"], "kind": "leaf", "claim_id": lf["claim_id"],
                    "full": lf["text"], "doi": lf["doi"], "year": lf.get("year", 0),
                    "score": 0}
        node = resolve_host(dec.get("host", ""))
        if node is not None:
            node.setdefault("children", []).append(leafnode); na += 1; continue
        prop = dec.get("propose") or {}
        top1 = dec.get("top1") or (next(iter(host_node_map)))
        if prop:
            hb = resolve_host(prop.get("branch", ""))   # proposed == existing host
            if hb is not None:
                hb.setdefault("children", []).append(leafnode); na += 1; continue
            bname = (prop.get("branch") or "Uncategorized phenomenon").strip()
            newrole = (prop.get("role") or "").strip().lower()
            if newrole not in ("phenomenon", "mechanism", "model", "class"):
                newrole = "phenomenon"
            parent = (prop.get("parent") or top1).strip()
            pnode = (resolve_host(parent) or name_node.get(parent)
                     or norm_index.get(vl.sb._norm(parent)) or host_node_map[top1])
            # enforce generality: the parent must be strictly more general
            new_rank = levels.ROLE_RANK.get(newrole, 5)
            gp = governing_parent(pnode, new_rank)
            pnode = gp if gp is not None else host_node_map[top1]
            key = vl.sb._norm(bname)
            if key in proposed:
                proposed[key].setdefault("children", []).append(leafnode)
            else:
                newnode = {"name": bname, "kind": newrole, "proposed": True,
                           "shared": False, "children": [leafnode]}
                proposed[key] = newnode
                pnode.setdefault("children", []).append(newnode)
                prop_log.append({"branch": bname, "role": newrole,
                                 "parent": pnode["name"], "example": lf["label"]})
            ne += 1
            continue
        host_node_map[top1].setdefault(
            "children", []).append(leafnode); na += 1     # fallback: nearest host
    (vl.VL / f"proposed_{view}.json").write_text(json.dumps(prop_log, indent=2))
    print(f"[grow] {view}: {na} placed, {ne} proposed, "
          f"{len(proposed)} proposed branches", file=sys.stderr)
    return na, ne, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-leaves", type=int, default=180)
    ap.add_argument("--per-paper", type=int, default=40)
    ap.add_argument("--min-claims", type=int, default=25,
                    help="full-PDF proxy: min total claims per paper")
    ap.add_argument("--attach", type=float, default=0.50)
    ap.add_argument("--exception", type=float, default=0.42)
    ap.add_argument("--fulltext", action="store_true", default=True,
                    help="select full-PDF papers by claim count (default)")
    ap.add_argument("--use-llm", action="store_true",
                    help="accurate Gemini placement (host-by-name) instead of embeddings")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel Gemini calls for LLM placement")
    ap.add_argument("--only-views", default="",
                    help="comma-separated views to place (default: all GROW_VIEWS)")
    ap.add_argument("--merge", action="store_true",
                    help="preserve other views from existing grown_views.json")
    args = ap.parse_args()

    to_place = args.only_views.split(",") if args.only_views else GROW_VIEWS
    nodes, views, host_nodes_by_view, manifest = vl.build_all_views(use_cache=True)
    dois = sample_fulltext_papers(args.papers, args.seed, args.min_claims)
    print(f"[grow] {len(dois)} full-text papers ({'LLM' if args.use_llm else 'embedding'} "
          f"placement, >= {args.min_claims} claims); views={to_place}", file=sys.stderr)

    for view in to_place:
        if args.use_llm:
            na, ne, exc = llm_grow_view(view, nodes, host_nodes_by_view[view],
                                        views[view], dois, args.per_paper,
                                        workers=args.workers)
        else:
            na, ne, exc = grow_view(view, nodes, host_nodes_by_view[view], dois,
                                    args.attach, args.exception, args.max_leaves)
        if exc:
            views[view]["children"].append(exc)
        vl._count(views[view])
        print(f"[grow] {view}: {na} leaves attached, {ne} exceptions",
              file=sys.stderr)

    if args.merge and (OUT / "grown_views.json").exists():
        prev = json.loads((OUT / "grown_views.json").read_text()).get("views", {})
        for v, top in prev.items():        # keep previously-placed views untouched
            if v not in to_place:
                views[v] = top
    sub = (f"scaffold + {len(dois)} test papers &middot; leaves under accurate "
           f"hosts &middot; {len(views)} views")
    (OUT / "grown_views.json").write_text(json.dumps(
        {"views": views, "subtitle": sub}, indent=2))   # persist for LLM-free re-render
    build_viz.render_html(views["by_reaction_type"], "chemistry living tree (with leaves)",
                          sub, OUT / "scaffold_multiview.html", views=views)
    print(f"[grow] wrote {OUT/'scaffold_multiview.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
