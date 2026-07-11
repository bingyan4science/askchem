"""Node-link redesign: add a terse display label + LaTeX equation per node.

The node-link tree shows a SHORT glyph label (e.g. "Marcus Theory") instead of the
full name, with the full name/definition/equation revealed on hover. This pass asks
Gemini, for every internal node (using its name + definition for context), for:
  * short_label : 1-3 words, the core law/theory/model (no qualifiers)
  * equation    : a concise LaTeX equation/relationship if one exists, else ""

Writes short_label + equation into output/grown_views.json; re-apply with
apply_to_db.py (which now persists both columns).

Usage:
    export PORTKEY_API_KEY=...   # source ~/.bashrc
    python3 living_taxonomy/enrich_nodes.py
    python3 living_taxonomy/apply_to_db.py --version v7
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import placement as pm
from incremental_build import _parse_json

OUT = _HERE / "output"
LEAF = "leaf"

_SYS = (
    "You label chemistry taxonomy nodes for a compact tree diagram. For each node "
    "you are given its full name and definition. Return:\n"
    "- short_label: the core law/theory/model/principle in 1-3 words, dropping "
    "qualifiers and context (e.g. 'Biological Electron Transfer (Marcus Theory)' -> "
    "'Marcus Theory'; 'Step-growth and chain-growth polymerization' -> "
    "'Polymerization'; 'Law of mass action' -> 'Mass Action').\n"
    "- equation: a concise, correct LaTeX expression of the governing relationship "
    "if one cleanly exists (e.g. 'k \\\\propto e^{-\\\\Delta G^{\\\\ddagger}/RT}', "
    "'H\\\\Psi = E\\\\Psi', 'PV = nRT'); use \"\" when there is no clean equation. "
    "LaTeX only, no surrounding $."
)


def _prompt(nodes):
    lines = "\n".join(f"- {n} [{k}] :: {(d or '')[:160]}" for n, k, d in nodes)
    return (f"Nodes (name [kind] :: definition):\n{lines}\n\nReturn ONLY JSON: "
            '{"nodes":[{"name":"<exact input name>","short_label":"<1-3 words>",'
            '"equation":"<LaTeX or empty>"}]}')


def _call(chunk):
    try:
        resp = pm._gemini_chat(_SYS, _prompt(chunk), max_time=120)
        return _parse_json(resp).get("nodes", [])
    except Exception as e:
        print(f"[enrich] chunk error: {e}", file=sys.stderr)
        return []


def main():
    data = json.loads((OUT / "grown_views.json").read_text())
    views = data["views"]
    nodes = {}

    def collect(n):
        for c in n.get("children", []):
            if c.get("kind") != LEAF:
                # incremental: only enrich nodes that lack a short_label yet
                # (v16 nodes already have one; this targets the new promoted branches)
                if not (c.get("short_label") or "").strip():
                    nodes.setdefault(c["name"], (c["kind"], c.get("definition", "")))
                collect(c)
    for top in views.values():
        collect(top)

    items = [(nm, k, d) for nm, (k, d) in nodes.items()]
    chunks = [items[i:i + 45] for i in range(0, len(items), 45)]
    print(f"[enrich] {len(items)} internal nodes in {len(chunks)} chunks", file=sys.stderr)

    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(_call, ch) for ch in chunks]):
            for r in f.result():
                nm = re.sub(r"\s*\[[^\]]*\].*$", "", r.get("name", "")).strip()
                if nm:
                    results[nm] = r

    n_lab = n_eq = 0

    def apply(n):
        nonlocal n_lab, n_eq
        for c in n.get("children", []):
            if c.get("kind") != LEAF:
                r = results.get(c["name"])
                if r:
                    sl = (r.get("short_label") or "").strip()
                    eq = (r.get("equation") or "").strip()
                    if sl:
                        c["short_label"] = sl
                        n_lab += 1
                    if eq:
                        c["equation"] = eq
                        n_eq += 1
                apply(c)
    for top in views.values():
        apply(top)

    (OUT / "grown_views.json").write_text(json.dumps({"views": views,
        "subtitle": data.get("subtitle", "")}, indent=2))
    (OUT / "node_enrich.json").write_text(json.dumps(list(results.values()), indent=2))
    print(f"[enrich] short_label set: {n_lab}; equations set: {n_eq}")
    print("[enrich] next: python3 living_taxonomy/apply_to_db.py --version v7")


if __name__ == "__main__":
    main()
