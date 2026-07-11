"""Build a small embedding index over living-tree NODES for semantic search.

Embeds each node's `name + ". " + definition` (per view) with the same mxbai
encoder the corpus uses, so a query can be routed to the right branch by meaning
(not just a name substring). Writes:
  * output/node_index.npz       - arrays: view_ids, node_ids, vecs (n, d) float32
  * output/node_index_meta.json - per-row {view_id,node_id,name,short_label,kind,
                                   definition,equation}

Cheap (~1.5k short texts). Re-run after every apply_to_db.

Usage:
    python3 living_taxonomy/build_node_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
import placement as pm
from askchem import db

OUT = _HERE / "output"


def main():
    rows = []
    with db.get_conn() as c:
        views = [r[0] for r in c.execute(
            "SELECT DISTINCT view_id FROM taxonomy_edges").fetchall()]
        for view_id in views:
            nodes = c.execute(
                "SELECT DISTINCT n.node_id, n.name, n.definition, n.short_label, "
                "n.kind, n.equation FROM taxonomy_edges e "
                "JOIN taxonomy_nodes n ON n.node_id=e.child_id "
                "WHERE e.view_id=? AND n.kind NOT IN ('leaf','paper')",
                (view_id,)).fetchall()
            for n in nodes:
                rows.append({
                    "view_id": view_id, "node_id": n["node_id"],
                    "name": n["name"] or "", "definition": n["definition"] or "",
                    "short_label": n["short_label"] or "", "kind": n["kind"] or "",
                    "equation": n["equation"] or "",
                })

    if not rows:
        print("[node-index] no nodes found; run apply_to_db first", file=sys.stderr)
        return

    texts = [f'{r["name"]}. {r["definition"]}'.strip() for r in rows]
    print(f"[node-index] embedding {len(texts)} nodes across "
          f"{len(set(r['view_id'] for r in rows))} views…", file=sys.stderr)
    vecs = pm._embed(texts, is_query=False).astype(np.float32)

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "node_index.npz",
             view_ids=np.array([r["view_id"] for r in rows]),
             node_ids=np.array([r["node_id"] for r in rows]),
             vecs=vecs)
    (OUT / "node_index_meta.json").write_text(json.dumps(rows))
    print(f"[node-index] wrote {OUT/'node_index.npz'} ({vecs.shape}) + meta")


if __name__ == "__main__":
    main()
