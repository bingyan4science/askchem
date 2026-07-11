"""Scaffold enrichment: fill MISSING governing theories so specifics stop dangling.

Many specific nodes (mechanisms/phenomena) sit directly under a top framework
because the governing mid-level theory is absent from the scaffold (e.g. there is
no "Photochemistry" node, so photochemical excitation, photocatalysis and
photoredox catalysis scatter under kinetics/QM and loosely-related theories).

For each (view, framework) this pass asks Gemini to (1) propose the missing
governing theory/model node(s), (2) re-home the dangling specifics (and any
related specifics elsewhere in the same framework) under them, and (3) nest
subtypes under their type (photoredox under photocatalysis). Generality is then
re-validated. Writes output/grown_views.json in place.

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/enrich_gaps.py
    python3 living_taxonomy/apply_to_db.py --version v9
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import levels
import placement as pm
from incremental_build import _parse_json

OUT = _HERE / "output"
GV = OUT / "grown_views.json"

_SYS = (
    "You repair a chemistry knowledge tree so it reads general -> specific. A "
    "framework (e.g. quantum mechanics, kinetics) is too general to host specific "
    "mechanisms/phenomena directly; between them sit governing THEORIES. When a "
    "cluster of specifics shares a governing theory that is MISSING, you name that "
    "theory and group the specifics under it, nesting subtypes under their parent "
    "type. Only group things that genuinely share a governing principle; never "
    "invent spurious groupings. Use canonical theory names (e.g. 'Photochemistry', "
    "'Electron transfer theory').")


def _index(root):
    idx = {}
    levels.walk(root, lambda n, p: idx.setdefault(n["name"], n))
    return idx


def _in_subtree(anc, node):
    found = {"v": False}
    levels.walk(anc, lambda n, p: found.__setitem__("v", found["v"] or n is node))
    return found["v"]


def _prompt(fw, mids, danglers, others):
    midblk = "\n".join(f"- {n['name']} [{levels.role_of(n)}]: {(n.get('definition') or '')[:80]}"
                       for n in mids) or "(none)"
    dblk = "\n".join(f"- {n['name']}: {(n.get('definition') or '')[:90]}" for n in danglers)
    oblk = ", ".join(n["name"] for n in others) or "(none)"
    return (
        f"FRAMEWORK: {fw['name']}\n\n"
        f"EXISTING governing theories/models under it (you may reuse these as parents "
        f"by EXACT name):\n{midblk}\n\n"
        f"SPECIFICS dangling directly under the framework (must be re-homed):\n{dblk}\n\n"
        f"OTHER specifics already under this framework (re-home ONLY if they clearly "
        f"belong with a new group):\n{oblk}\n\n"
        "Return ONLY JSON:\n"
        '{"new_theories":[{"name":"...","definition":"one sentence","equation":"<LaTeX or empty>"}],'
        '"assign":[{"node":"<exact specific name>","parent":"<exact existing or new '
        'theory/model name>"}],'
        '"subtypes":[{"child":"<exact specific name>","parent":"<exact specific name it '
        'is a subtype of>"}]}\n'
        "Every dangling specific MUST appear in 'assign'. Prefer an existing theory; "
        "propose a new one only when none governs the cluster.")


def enrich_view(vid, root):
    idx = _index(root)
    pmap = levels.parent_map(root)
    frameworks = [n for n in root.get("children", []) if levels.rank_of(n) <= 1]
    created = moved = 0
    for fw in frameworks:
        # gather framework-subtree specifics
        mids, danglers, others = [], [], []
        def visit(n, p):
            if n is fw or p is None:
                return
            role = levels.role_of(n)
            if role in ("theory", "model"):
                mids.append(n)
            elif role in ("mechanism", "phenomenon"):
                (danglers if p is fw else others).append(n)
        levels.walk(fw, visit)
        if not danglers:
            continue
        try:
            resp = _parse_json(pm._gemini_chat(_SYS, _prompt(fw, mids, danglers, others), max_time=120))
        except Exception as e:
            print(f"[enrich] {vid}/{fw['name']}: error {e}", file=sys.stderr); continue

        # 1) create new theory nodes under the framework
        for t in resp.get("new_theories", []):
            nm = (t.get("name") or "").strip()
            if not nm or nm in idx:
                continue
            node = {"name": nm, "kind": "theory", "definition": t.get("definition", ""),
                    "equation": (t.get("equation") or ""), "shared": False, "children": []}
            fw.setdefault("children", []).append(node)
            idx[nm] = node; created += 1

        pmap = levels.parent_map(root)

        def reattach(child, parent):
            if child is None or parent is None or child is parent:
                return False
            if _in_subtree(child, parent):   # would create a cycle
                return False
            if not levels.valid_edge(parent, child):   # allows host subtype nesting
                return False
            op = pmap.get(id(child))
            if op is not None:
                op["children"] = [c for c in op.get("children", []) if c is not child]
            parent.setdefault("children", []).append(child)
            pmap[id(child)] = parent
            return True

        # 2) assign danglers (and claimed others) under their governing node
        for a in resp.get("assign", []):
            ch = idx.get(a.get("node", "")); par = idx.get(a.get("parent", ""))
            if reattach(ch, par):
                moved += 1
        # 3) subtype nesting among specifics
        for s in resp.get("subtypes", []):
            ch = idx.get(s.get("child", "")); par = idx.get(s.get("parent", ""))
            if reattach(ch, par):
                moved += 1
    return created, moved


def main():
    data = json.loads(GV.read_text())
    views = data["views"]
    import shutil
    shutil.copy(GV, GV.with_suffix(".json.pre_enrich.bak"))
    tot_c = tot_m = 0
    for vid, root in views.items():
        c, m = enrich_view(vid, root)
        tot_c += c; tot_m += m
        print(f"[enrich] {vid}: +{c} governing theories, {m} specifics re-homed", file=sys.stderr)
    # final generality check
    for vid, root in views.items():
        viol = levels.find_violations(root)
        fw = sum(1 for v in viol if v["type"] == "gap" and v["parent_rank"] <= 1)
        print(f"[enrich] {vid}: framework-gaps now {fw}", file=sys.stderr)
    data["views"] = views
    GV.write_text(json.dumps(data, indent=2))
    print(f"[enrich] total: +{tot_c} theories, {tot_m} re-homed")
    print("[enrich] next: python3 living_taxonomy/apply_to_db.py --version v9")


if __name__ == "__main__":
    main()
