"""Compile a comprehensive first-principles scaffold of chemistry.

Method (the "digest the textbooks" approach): for each classic subfield
(general / physical / organic / inorganic / analytical / biochemistry) ask
Gemini (NYU) to enumerate the canonical LAWS, THEORIES, MODELS and elementary
MECHANISMS that subfield teaches, each anchored under a fixed physics trunk.
Then MERGE all subfields deterministically (dedupe by normalized name, keep the
most-fundamental kind, resolve parents by name) into a single scaffold tree.

This produces the curated upper trunk that papers later grow leaves onto. No
paper data is touched. Output: output/scaffold_full.{json,html} + counts.

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/scaffold_builder.py
    open living_taxonomy/output/scaffold_full.html
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_viz
import placement as pm
from incremental_build import _parse_json

OUT = _HERE / "output"

SUBFIELDS = ["general", "physical", "organic", "inorganic",
             "analytical", "biochemistry"]

# Fixed top-level anchors (the physics trunk). (name, kind)
ANCHORS = [
    ("Conservation laws", "law"),
    ("Electromagnetic (Coulomb) interaction", "law"),
    ("Quantum mechanics", "framework"),
    ("Thermodynamics & statistical mechanics", "framework"),
    ("Chemical kinetics & dynamics", "framework"),
]

KIND_RANK = {"open_root": 0, "law": 1, "framework": 2, "theory": 3,
             "model": 4, "principle": 3, "mechanism": 5, "phenomenon": 6,
             "class": 6, "leaf": 7}

_SYS = ("You are compiling the conceptual scaffold of chemistry from classic "
        "textbooks. You output the fundamental LAWS, THEORIES, MODELS and "
        "elementary MECHANISMS a subfield teaches, organized from most "
        "fundamental to most specialized under fundamental physics. You never "
        "output techniques, instruments, named reactions, or specific compounds.")


def _user(subfield):
    anchors = "\n".join(f"- {n} ({k})" for n, k in ANCHORS)
    return f"""Fixed top-level anchors (use these EXACT names as ultimate parents):
{anchors}

For "{subfield} chemistry", list the canonical concepts as JSON nodes. Each node:
{{"name": "...",
  "kind": "law|theory|model|mechanism",
  "parent": "name of the more fundamental concept it sits under (an anchor above, or another concept you list)",
  "framework": "which anchor it ultimately belongs to (exact name from the list)",
  "definition": "one concise sentence"}}

Rules:
- Order STRICTLY from general to specific: a parent must be MORE FUNDAMENTAL (more
  general) than its child. The generality ladder is:
    framework/law (1) > theory (2) > model (3) > mechanism (4).
  A child's kind must be one rank more specific than its parent's, and attach under
  the MOST SPECIFIC concept that still governs it (never skip straight to an anchor
  when a governing theory/model exists).
- "theory" = an explanatory theory (e.g. molecular orbital theory, transition-state theory).
- "model" = a concrete model under a theory (e.g. VSEPR, band theory).
- "mechanism" = an elementary-step reactivity motif where specific reactions attach
  (e.g. nucleophilic substitution, oxidative addition, proton transfer).
- Do NOT include techniques, instruments, named reactions, specific compounds, or
  named reactivity FAMILIES (e.g. "photocatalysis", "cross-coupling") - those are
  phenomena added later as hosts UNDER the governing mechanism/theory, not here.
- Aim for the canonical ~25-40 concepts of this subfield.
Return ONLY JSON: {{"nodes": [ ... ]}}"""


def _norm(name):
    s = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    drop = {"the", "of", "a", "an", "and", "to",
            "theory", "theories", "principle", "principles", "law", "laws",
            "model", "models", "effect", "rule", "rules", "mechanism", "mechanisms"}
    toks = [t for t in s.split() if t not in drop]
    return " ".join(toks)


def enumerate_subfields():
    raw = {}
    for sf in SUBFIELDS:
        try:
            resp = pm._gemini_chat(_SYS, _user(sf), max_time=120)
            nodes = _parse_json(resp).get("nodes", [])
        except Exception as e:
            print(f"[scaffold] {sf}: ERROR {e}", file=sys.stderr)
            nodes = []
        raw[sf] = nodes
        print(f"[scaffold] {sf}: {len(nodes)} nodes", file=sys.stderr)
    (OUT / "scaffold_raw.json").write_text(json.dumps(raw, indent=2))
    return raw


# Keyword -> anchor index, for resolving fuzzy framework labels to the trunk.
_ANCHOR_KEYWORDS = [
    (("conserv",), 0),
    (("coulomb", "electrostat", "electromag", "ionic", "dipole"), 1),
    (("quantum", "schrod", "wavefunction", "wave function", "orbital",
      "pauli", "spectroscop", "electronic structure", "dft", "density functional",
      "ab initio", "hartree"), 2),
    (("thermodynam", "entrop", "enthalp", "equilibri", "free energy",
      "statistical mech", "phase", "ensemble", "boltzmann", "colligative"), 3),
    (("kinetic", "rate", "reaction dynamic", "cataly", "transition state",
      "marcus", "diffusion", "transport", "electrode", "collision",
      "mechanism", "photochem"), 4),
]


def _anchor_for(text, anchor_norms):
    low = (text or "").lower()
    for keys, idx in _ANCHOR_KEYWORDS:
        if any(k in low for k in keys):
            return anchor_norms[idx]
    return None


def merge(raw):
    """Deterministic merge of all subfield nodes into one tree."""
    nodes = {}            # norm_name -> {name,kind,desc,parent_norm,framework_norm,subfields}
    anchor_norm = {}      # norm -> canonical name
    anchor_norms = []     # ordered, indexable by anchor position
    for name, kind in ANCHORS:
        nm = _norm(name)
        anchor_norm[nm] = name
        anchor_norms.append(nm)
        nodes[nm] = {"name": name, "kind": kind, "desc": "", "parent": None,
                     "framework": None, "subfields": set()}

    for sf, lst in raw.items():
        for nd in lst:
            name = (nd.get("name") or "").strip()
            if not name:
                continue
            nm = _norm(name)
            if not nm:
                continue
            kind = nd.get("kind", "theory")
            if nm in nodes:
                # keep the more fundamental kind, prefer existing canonical name
                if KIND_RANK.get(kind, 3) < KIND_RANK.get(nodes[nm]["kind"], 3):
                    nodes[nm]["kind"] = kind
                if not nodes[nm]["desc"]:
                    nodes[nm]["desc"] = nd.get("definition", "")
                nodes[nm]["subfields"].add(sf)
                continue
            nodes[nm] = {
                "name": name, "kind": kind, "desc": nd.get("definition", ""),
                "parent": _norm(nd.get("parent", "")) or None,
                "framework": _norm(nd.get("framework", "")) or None,
                "subfields": {sf},
            }

    # build a token index for fuzzy parent matching
    tok_index = {nm: set(nm.split()) for nm in nodes}

    def fuzzy_parent(p_norm):
        """Find an existing node whose tokens best match p_norm (>=2 shared)."""
        if not p_norm:
            return None
        ptoks = set(p_norm.split())
        best, best_overlap = None, 0
        for nm, toks in tok_index.items():
            ov = len(ptoks & toks)
            if ov >= 2 and ov > best_overlap and (toks <= ptoks or ptoks <= toks
                                                  or ov >= 2):
                best, best_overlap = nm, ov
        return best

    # resolve parents -> children, breaking unknowns/cycles
    children = {nm: [] for nm in nodes}
    roots = []
    for nm, n in nodes.items():
        if nm in anchor_norm:
            roots.append(nm)
            continue
        p = n["parent"]
        if p not in nodes or p == nm:
            p = None
        if p is None and n["framework"] in nodes:        # exact framework
            p = n["framework"]
        if p is None:                                    # fuzzy parent label
            p = fuzzy_parent(n["parent"])
        if p is None:                                    # keyword -> anchor
            p = _anchor_for(n["framework"] or "", anchor_norms) \
                or _anchor_for(n["name"], anchor_norms)
        n["_parent"] = p
    # cycle-safe attach
    def is_ancestor(a, b):
        seen = set()
        cur = b
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = nodes[cur].get("_parent") if cur in nodes else None
            if cur == a:
                return True
        return False
    for nm, n in nodes.items():
        if nm in anchor_norm:
            continue
        p = n.get("_parent")
        if p is None or p == nm or is_ancestor(nm, p):
            roots.append(nm)            # orphan -> top level (under open_root)
        else:
            children[p].append(nm)
    return nodes, children, roots, anchor_norm


def to_d3(nm, nodes, children):
    n = nodes[nm]
    kids = [to_d3(c, nodes, children) for c in children.get(nm, [])]
    out = {"name": n["name"], "kind": n["kind"], "count": 0}
    if kids:
        out["children"] = kids
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true",
                    help="re-merge from scaffold_raw.json without calling the LLM")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.cache and (OUT / "scaffold_raw.json").exists():
        raw = json.loads((OUT / "scaffold_raw.json").read_text())
        print("[scaffold] re-merging from cache", file=sys.stderr)
    else:
        raw = enumerate_subfields()
    nodes, children, roots, anchor_norm = merge(raw)

    # top-level: anchors first, then orphans
    anchors = [a for a in roots if a in anchor_norm]
    orphans = [r for r in roots if r not in anchor_norm]
    top = {"name": "(unifying principle unknown)", "kind": "open_root", "count": 0,
           "children": [to_d3(a, nodes, children) for a in anchors]
                       + [to_d3(o, nodes, children) for o in orphans]}

    # generality self-check (general -> specific must hold on every edge)
    import levels
    viol = levels.find_violations(top)
    if viol:
        inv = sum(1 for v in viol if v["type"] == "inversion")
        gap = sum(1 for v in viol if v["type"] == "gap")
        print(f"[scaffold] generality violations: {inv} inversion(s), {gap} gap(s)",
              file=sys.stderr)
        (OUT / "scaffold_violations.json").write_text(json.dumps(
            [{"type": v["type"], "parent": v["parent"]["name"],
              "child": v["child"]["name"]} for v in viol], indent=2))

    from collections import Counter
    kinds = Counter(n["kind"] for n in nodes.values())
    (OUT / "scaffold_full.json").write_text(json.dumps(top, indent=2))
    sub = (f"chemistry scaffold from {len(SUBFIELDS)} subfields &middot; "
           + ", ".join(f"{k}:{v}" for k, v in kinds.most_common()))
    build_viz.render_html(top, "chemistry first-principles scaffold", sub,
                          OUT / "scaffold_full.html")
    print("\n=== SCAFFOLD COUNTS ===")
    for k, v in kinds.most_common():
        print(f"  {k:10s}: {v}")
    print(f"  total internal nodes: {sum(kinds.values())}")
    print(f"  orphans (no resolved parent): {len(orphans)}")
    print(f"[scaffold] wrote {OUT/'scaffold_full.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
