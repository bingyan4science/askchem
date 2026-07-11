"""Validate (and fix) the positioning of the living tree on the 300-paper set.

`metrics`  - report generality inversions, rank gaps, duplicate candidates, depth
             balance, host/leaf counts per view (the freeze criteria).
`fix`      - restructure output/grown_views.json in place:
               1. role pass  (LLM): assign each internal node a ladder role
                  (framework>theory>model>mechanism>phenomenon), introducing the
                  phenomenon level for named reactivity/material families.
               2. dedupe     (registry): merge same-rank near-duplicate concepts.
               3. repair      (levels + LLM): nest rank-gap/inversion children under
                  the most-specific governing node in their own branch.
             Backs up grown_views.json first, then re-runs metrics.

Tune prompts/thresholds here on the 300 (overfit) set, FREEZE, then the same code
runs once at full scale.

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/audit_positioning.py metrics
    python3 living_taxonomy/audit_positioning.py fix
    python3 living_taxonomy/apply_to_db.py --version v8
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import concept_registry as reg
import levels
import placement as pm
from incremental_build import _parse_json

OUT = _HERE / "output"
GV = OUT / "grown_views.json"


# ── metrics ──────────────────────────────────────────────────────────────────

def _depths(root):
    out = {}
    def visit(n, p):
        out[id(n)] = 0 if p is None else out[id(p)] + 1
    levels.walk(root, visit)
    return out


def metrics(views):
    print("=== positioning metrics ===")
    for vid, root in views.items():
        viol = levels.find_violations(root)
        inv = sum(1 for v in viol if v["type"] == "inversion")
        gap = sum(1 for v in viol if v["type"] == "gap")
        fwgap = sum(1 for v in viol if v["type"] == "gap" and v["parent_rank"] <= 1)
        # duplicate candidates: same-rank internal nodes sharing normalized name
        byname = defaultdict(list)
        levels.walk(root, lambda n, p: byname[(levels.rank_of(n), reg._norm(n["name"]))].append(n)
                    if p is not None and not levels.is_leaf(n) and reg._norm(n["name"]) else None)
        dups = sum(1 for k, v in byname.items() if len(v) > 1)
        d = _depths(root)
        internal = [n for n, _ in reg._internal_nodes(root)]
        dh = Counter(d[id(n)] for n in internal)
        print(f"\n{vid}: inversions={inv} framework-gaps={fwgap} (total gaps={gap}) "
              f"name-dup groups={dups} internal={len(internal)}")
        print(f"   depth hist (internal): {dict(sorted(dh.items()))}")
        # show a few worst gaps (specific node directly under a general one)
        worst = sorted([v for v in viol if v["type"] == "gap"],
                       key=lambda v: v["child_rank"] - v["parent_rank"], reverse=True)[:4]
        for v in worst:
            print(f"   gap: {v['child']['name'][:34]!r} ({levels.RANK_LABEL[v['child_rank']]}) "
                  f"directly under {v['parent']['name'][:30]!r} ({levels.RANK_LABEL[v['parent_rank']]})")


# ── role pass (LLM) ───────────────────────────────────────────────────────────

_ROLE_SYS = (
    "You assign each chemistry taxonomy node its role on a strict generality "
    "ladder, so the tree reads general -> specific:\n"
    "  framework = foundational pillar (quantum mechanics, thermodynamics, kinetics)\n"
    "  theory    = explanatory theory/law (transition-state theory, Marcus theory, mass action)\n"
    "  model     = a concrete model under a theory (VSEPR, band theory, Jablonski)\n"
    "  mechanism = an elementary-step reactivity motif (oxidative addition, proton transfer)\n"
    "  phenomenon= a NAMED reactivity / material family or process (photocatalysis, "
    "cross-coupling, ATRP, nanomaterials, biological electron transfer) - the most "
    "specific concept layer above papers.\n"
    "Pick the single best role. Named processes/families/applications are 'phenomenon', "
    "NOT 'theory' - even if their name cites a theory in parentheses (e.g. 'Biological "
    "Electron Transfer (Marcus Theory)' is a phenomenon, while 'Marcus theory' itself is "
    "a theory). Reserve 'theory' for the explanatory theory itself.")


def _role_prompt(chunk):
    lines = "\n".join(f"- {n} [{k}] :: {(d or '')[:140]}" for n, k, d in chunk)
    return (f"Nodes (name [current kind] :: definition):\n{lines}\n\nReturn ONLY JSON: "
            '{"nodes":[{"name":"<exact input name>","role":"framework|theory|model|'
            'mechanism|phenomenon"}]}')


def _role_call(chunk):
    try:
        return _parse_json(pm._gemini_chat(_ROLE_SYS, _role_prompt(chunk), max_time=120)).get("nodes", [])
    except Exception as e:
        print(f"[audit] role chunk error: {e}", file=sys.stderr); return []


def assign_roles(views):
    nodes = {}
    for root in views.values():
        levels.walk(root, lambda n, p: nodes.setdefault(n["name"], (n.get("kind"), n.get("definition", "")))
                    if p is not None and not levels.is_leaf(n) else None)
    items = [(nm, k, d) for nm, (k, d) in nodes.items()]
    chunks = [items[i:i + 45] for i in range(0, len(items), 45)]
    print(f"[audit] role pass: {len(items)} nodes in {len(chunks)} chunks", file=sys.stderr)
    import re
    roles = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(_role_call, ch) for ch in chunks]):
            for r in f.result():
                nm = re.sub(r"\s*\[[^\]]*\].*$", "", r.get("name", "")).strip()
                role = (r.get("role") or "").strip().lower()
                if nm and role in ("framework", "theory", "model", "mechanism", "phenomenon"):
                    roles[nm] = role
    n = 0
    def apply(nd, p):
        nonlocal n
        if p is not None and not levels.is_leaf(nd) and nd["name"] in roles:
            # never demote a foundational anchor below framework
            if nd.get("kind") not in ("open_root", "law"):
                nd["kind"] = roles[nd["name"]]; n += 1
    for root in views.values():
        levels.walk(root, apply)
    print(f"[audit] roles assigned: {n}", file=sys.stderr)


# ── repair (levels + batched LLM choose-parent) ───────────────────────────────

_REPARENT_SYS = (
    "You fix the nesting of a chemistry knowledge tree so it reads general -> "
    "specific. A specific node (a phenomenon or mechanism) must hang under the MOST "
    "SPECIFIC concept that genuinely governs it, never as a sibling of a general "
    "theory. Given a too-general parent, its candidate governing sub-nodes, and the "
    "specific children currently mis-placed directly under it, choose for each child "
    "the candidate that governs it - or 'none' if truly none does.")


def _repair_group_prompt(parent, cands, children):
    cl = "\n".join(f"- {c['name']}: {(c.get('definition') or '')[:90]}" for c in cands)
    kids = "\n".join(f"- {c['name']}: {(c.get('definition') or '')[:90]}" for c in children)
    return (f"Too-general parent: {parent['name']}\n\nCANDIDATE governing nodes "
            f"(choose by EXACT name):\n{cl}\n\nMIS-PLACED children to re-home:\n{kids}\n\n"
            'Return ONLY JSON: {"moves":[{"child":"<exact child name>",'
            '"parent":"<exact candidate name or none>"}]}')


def _in_subtree(node, target):
    """True if `target` lies within `node`'s subtree (iterative, cycle-safe)."""
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


def _fix_inversions(root):
    """Deterministic: an inversion means the parent is MORE specific than the
    child, so promote the child up to the nearest ancestor that is more general."""
    moved = 0
    pmap = levels.parent_map(root)
    for v in [x for x in levels.find_violations(root) if x["type"] == "inversion"]:
        child, parent = v["child"], v["parent"]
        if pmap.get(id(child)) is not parent:
            continue
        cr = levels.rank_of(child)
        anc = pmap.get(id(parent))
        while anc is not None and levels.rank_of(anc) >= cr:
            anc = pmap.get(id(anc))
        if anc is not None and anc is not parent:
            parent["children"] = [c for c in parent.get("children", []) if c is not child]
            anc.setdefault("children", []).append(child)
            pmap[id(child)] = anc
            moved += 1
    return moved


def repair_view(root, gemini, rounds=3):
    """Fix two real errors: (1) inversions (promote up, deterministic); (2) a
    specific node sitting DIRECTLY under a top framework/law while a governing
    theory/mechanism exists among its siblings (LLM nest-down). Legitimate
    theory->phenomenon/class edges (no intermediate available) are left alone."""
    moved = 0
    for _ in range(rounds):
        moved += _fix_inversions(root)
        viol = levels.find_violations(root)
        groups = defaultdict(list)
        for v in viol:
            if v["type"] == "gap" and v["parent_rank"] <= 1:   # under a top framework/law
                groups[id(v["parent"])].append(v)
        any_move = False
        for items in groups.values():
            parent = items[0]["parent"]
            children = [it["child"] for it in items]
            cands, seen = [], set()
            for ch in children:
                for cand in levels.candidate_parents(root, ch, parent):
                    if id(cand) not in seen:
                        seen.add(id(cand)); cands.append(cand)
            if not cands:
                continue
            try:
                resp = _parse_json(gemini(_REPARENT_SYS, _repair_group_prompt(parent, cands, children)))
                moves = resp.get("moves", [])
            except Exception:
                moves = []
            cand_by_name = {c["name"]: c for c in cands}
            child_by_name = {c["name"]: c for c in children}
            pmap = levels.parent_map(root)
            for mv in moves:
                ch = child_by_name.get(mv.get("child", ""))
                tgt = cand_by_name.get(mv.get("parent", ""))
                if ch is None or tgt is None or tgt is ch:
                    continue
                if levels.rank_of(tgt) >= levels.rank_of(ch):
                    continue
                if _in_subtree(ch, tgt):   # never reparent a node under its own descendant (cycle)
                    continue
                if pmap.get(id(ch)) is parent:
                    parent["children"] = [c for c in parent.get("children", []) if c is not ch]
                    tgt.setdefault("children", []).append(ch)
                    moved += 1; any_move = True
        if not any_move:
            break
    return moved


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["metrics", "fix"])
    ap.add_argument("--tau", type=float, default=0.92)
    args = ap.parse_args()

    data = json.loads(GV.read_text())
    views = data["views"]

    if args.cmd == "metrics":
        metrics(views)
        return

    print("[audit] BEFORE:")
    metrics(views)
    shutil.copy(GV, GV.with_suffix(".json.pre_positioning.bak"))

    assign_roles(views)
    remap = reg.dedupe_views(views, tau=args.tau)
    print(f"[audit] deduped {len(remap)} node(s): "
          + "; ".join(f"{k}->{v}" for k, v in list(remap.items())[:6]))
    total_moved = 0
    for vid, root in views.items():
        m = repair_view(root, pm._gemini_chat)
        total_moved += m
        print(f"[audit] {vid}: repaired {m} edge(s)", file=sys.stderr)

    data["views"] = views
    GV.write_text(json.dumps(data, indent=2))
    print(f"\n[audit] AFTER (moved {total_moved}, deduped {len(remap)}):")
    metrics(views)
    print("\n[audit] next: python3 living_taxonomy/apply_to_db.py --version v8")


if __name__ == "__main__":
    main()
