"""Load the grown living-taxonomy views into askchem.db tables.

Reads ``output/grown_views.json`` (produced by grow_onto_scaffold.py) and writes
into ``taxonomy_nodes`` / ``taxonomy_edges`` / ``taxonomy_leaves`` / ``taxonomy_meta``
so the API can serve the living tree from the DB. Additive + idempotent: it
clears and rewrites only the living-taxonomy tables, never touching claims.

Internal-node ids are the normalized concept name (shared across views, so the
trunk is deduped); the view root is ``__root__``; leaves reference claim_id.

Usage:
    python3 living_taxonomy/apply_to_db.py [--db askchem.db] [--version v1]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
import scaffold_builder as sb   # for _norm

OUT = _HERE / "output"


def _default_db():
    """Canonical DB path (askchem.db, with chemtree.db fallback)."""
    try:
        from askchem import db
        return db.get_db_path()
    except Exception:
        c = _HERE.parent / "askchem.db"
        return c if c.exists() else _HERE.parent / "chemtree.db"


DEFAULT_DB = _default_db()
ROOT_ID = "__root__"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS taxonomy_nodes (
    node_id TEXT PRIMARY KEY, kind TEXT, name TEXT, definition TEXT,
    short_label TEXT, equation TEXT,
    proposed INTEGER DEFAULT 0, data TEXT);
CREATE TABLE IF NOT EXISTS taxonomy_edges (
    view_id TEXT NOT NULL, parent_id TEXT, child_id TEXT NOT NULL,
    PRIMARY KEY (view_id, parent_id, child_id));
CREATE TABLE IF NOT EXISTS taxonomy_leaves (
    view_id TEXT NOT NULL, node_id TEXT NOT NULL, claim_id TEXT NOT NULL,
    doi TEXT, score REAL, label TEXT, PRIMARY KEY (view_id, node_id, claim_id));
CREATE TABLE IF NOT EXISTS taxonomy_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_ltax_edges_parent ON taxonomy_edges(view_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_ltax_edges_child ON taxonomy_edges(view_id, child_id);
CREATE INDEX IF NOT EXISTS idx_ltax_leaves_node ON taxonomy_leaves(view_id, node_id);
CREATE INDEX IF NOT EXISTS idx_ltax_leaves_claim ON taxonomy_leaves(claim_id);
CREATE INDEX IF NOT EXISTS idx_ltax_leaves_doi ON taxonomy_leaves(doi);
"""


def _node_id(node):
    if node.get("kind") == "open_root":
        return ROOT_ID
    return sb._norm(node["name"]) or ("n_" + str(abs(hash(node["name"])) % 10**8))


def load(conn, views, version):
    cur = conn.cursor()
    # drop + recreate so schema changes (e.g. new columns) always apply
    cur.executescript("DROP TABLE IF EXISTS taxonomy_nodes;"
                      "DROP TABLE IF EXISTS taxonomy_edges;"
                      "DROP TABLE IF EXISTS taxonomy_leaves;")
    cur.executescript(_SCHEMA)

    # Accumulate rows then bulk-insert (executemany in one transaction) - per-row
    # execute is far too slow at ~1M leaves.
    node_rows = {}   # nid -> tuple (dedupe: shared trunk across views)
    edge_rows = []
    leaf_rows = []

    def walk(view_id, node, parent_id):
        nid = _node_id(node)
        if nid not in node_rows:
            node_rows[nid] = (nid, node.get("kind"), node.get("name"),
                              node.get("definition", ""), node.get("short_label", ""),
                              node.get("equation", ""),
                              1 if node.get("proposed") else 0, "{}")
        edge_rows.append((view_id, parent_id, nid))
        for c in node.get("children", []):
            if c.get("kind") == "leaf":
                if not c.get("claim_id"):
                    continue
                leaf_rows.append((view_id, nid, c["claim_id"], c.get("doi"),
                                  c.get("score", 0), c.get("name")))
            else:
                walk(view_id, c, nid)

    for view_id, top in views.items():
        walk(view_id, top, None)

    cur.executemany(
        "INSERT OR REPLACE INTO taxonomy_nodes "
        "(node_id,kind,name,definition,short_label,equation,proposed,data) "
        "VALUES (?,?,?,?,?,?,?,?)", list(node_rows.values()))
    cur.executemany(
        "INSERT OR IGNORE INTO taxonomy_edges (view_id,parent_id,child_id) "
        "VALUES (?,?,?)", edge_rows)
    cur.executemany(
        "INSERT OR IGNORE INTO taxonomy_leaves "
        "(view_id,node_id,claim_id,doi,score,label) VALUES (?,?,?,?,?,?)", leaf_rows)
    n_nodes, n_edges, n_leaves = len(node_rows), len(edge_rows), len(leaf_rows)

    cur.execute("INSERT OR REPLACE INTO taxonomy_meta (key,value) VALUES ('version',?)",
                (version,))
    cur.execute("INSERT OR REPLACE INTO taxonomy_meta (key,value) VALUES ('updated_at',?)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"),))
    cur.execute("INSERT OR REPLACE INTO taxonomy_meta (key,value) VALUES ('views',?)",
                (json.dumps(list(views.keys())),))
    conn.commit()
    return n_nodes, n_edges, n_leaves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--grown", default=str(OUT / "grown_views.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.grown).read_text())
    views = data["views"]
    conn = sqlite3.connect(args.db)
    nn, ne, nl = load(conn, views, args.version)
    conn.close()
    print(f"[apply] db={args.db} version={args.version}")
    print(f"[apply] nodes={nn} edges={ne} leaves={nl} views={list(views)}")


if __name__ == "__main__":
    main()
