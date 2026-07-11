"""Workstream A: make internal nodes concrete, content-bearing laws/theories.

Today nodes are bare labels with no statement. This pass asks Gemini, for every
internal node, for a one-sentence STATEMENT of the actual law/principle/theory it
denotes (with the key equation/relationship when one exists), and an optional
sharper name when the label is vague. Writes `definition` (+ optional rename)
into output/grown_views.json; re-apply with apply_to_db.py.

Usage:
    export PORTKEY_API_KEY=...   # source ~/.bashrc
    python3 living_taxonomy/audit_nodes.py
    python3 living_taxonomy/apply_to_db.py --version v6
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
    "You make chemistry taxonomy nodes concrete. For each concept node, give a "
    "one-sentence STATEMENT of the actual law/principle/theory/model it denotes, "
    "including the key relationship or equation when one exists (e.g. Newton's "
    "second law -> 'force equals mass times acceleration, F=ma'; transition-state "
    "theory -> 'reaction rate is proportional to exp(-dG/RT)'; mass conservation -> "
    "'total mass of an isolated system is constant'). If the node name is vague and "
    "a specific named governing law/theory is the real concept, also give a sharper "
    "'rename'. Be accurate and concise; no marketing language.")


def _prompt(nodes):
    lines = "\n".join(f"- {n} [{k}]" for n, k in nodes)
    return (f"Nodes:\n{lines}\n\nReturn ONLY JSON: "
            '{"nodes":[{"name":"<exact input name>","statement":"<one sentence>",'
            '"rename":"<optional sharper name; omit if the name is already specific>"}]}')


def _call(chunk):
    try:
        resp = pm._gemini_chat(_SYS, _prompt(chunk), max_time=120)
        return _parse_json(resp).get("nodes", [])
    except Exception as e:
        print(f"[audit] chunk error: {e}", file=sys.stderr)
        return []


def main():
    data = json.loads((OUT / "grown_views.json").read_text())
    views = data["views"]
    names = {}

    def collect(n):
        for c in n.get("children", []):
            if c.get("kind") != LEAF:
                # incremental: only define nodes that lack a definition (the new
                # promoted branches); v16 nodes are already defined.
                if not (c.get("definition") or "").strip():
                    names.setdefault(c["name"], c["kind"])
                collect(c)
    for top in views.values():
        collect(top)

    items = list(names.items())
    chunks = [items[i:i + 45] for i in range(0, len(items), 45)]
    print(f"[audit] {len(items)} internal nodes in {len(chunks)} chunks", file=sys.stderr)

    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(_call, ch) for ch in chunks]):
            for r in f.result():
                nm = re.sub(r"\s*\[[^\]]*\]\s*$", "", r.get("name", "")).strip()
                if nm:
                    r["name"] = nm
                    results[nm] = r

    existing = set(names)
    renamed = {}
    n_def = 0

    def apply(n):
        nonlocal n_def
        for c in n.get("children", []):
            if c.get("kind") != LEAF:
                r = results.get(c["name"])
                if r:
                    if r.get("statement"):
                        c["definition"] = r["statement"]
                        n_def += 1
                    nr = (r.get("rename") or "").strip()
                    # rename only if distinct and not colliding with another node
                    if nr and nr != c["name"] and nr not in existing:
                        renamed[c["name"]] = nr
                        c["name"] = nr
                apply(c)
    for top in views.values():
        apply(top)

    (OUT / "grown_views.json").write_text(json.dumps({"views": views,
        "subtitle": data.get("subtitle", "")}, indent=2))
    (OUT / "node_audit.json").write_text(json.dumps(
        {"results": list(results.values()), "renamed": renamed}, indent=2))
    print(f"[audit] statements set: {n_def} occurrences; renamed: {len(renamed)} nodes")
    print("[audit] next: python3 living_taxonomy/apply_to_db.py --version v6")


if __name__ == "__main__":
    main()
