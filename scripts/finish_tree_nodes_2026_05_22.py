#!/usr/bin/env python3
"""Finish stages 3-4 of apply_harvest_2026_05_21.py after the DB-lock crash.

The crash happened in the tree-nodes pass because the outer ``get_conn``
held a write lock while ``upsert_tree_node`` opened its own connection
for new nodes. This script re-reads the new claims directly from the DB
and applies all tree-node updates + by_paper nodes through a single
shared connection.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import (  # noqa: E402
    DB_PATH, get_conn, update_metadata_counts, index_authors_for_doi,
)

NEW_CUTOFF = "2026-05-22T00:00:00"


def main() -> int:
    started = time.time()

    # 1. Find all new claims and their view_paths via a single SELECT
    print("loading new claims (extracted_at > 2026-05-22)...")
    with get_conn(readonly=True) as conn:
        rows = conn.execute(
            "SELECT claim_id, source_doi, source_paper_title, view_paths "
            "FROM claims WHERE extracted_at > ?",
            (NEW_CUTOFF,)
        ).fetchall()
    print(f"  {len(rows):,} new claims")

    node_claims: dict[tuple[str, str], list[str]] = defaultdict(list)
    paper_claim_ids: dict[str, list[str]] = defaultdict(list)
    paper_titles: dict[str, str] = {}

    for r in rows:
        claim_id = r["claim_id"]
        doi = r["source_doi"] or ""
        try:
            view_paths = json.loads(r["view_paths"]) if r["view_paths"] else {}
        except Exception:
            view_paths = {}

        if doi:
            paper_claim_ids[doi].append(claim_id)
            if doi not in paper_titles and r["source_paper_title"]:
                paper_titles[doi] = r["source_paper_title"]

        for vid, path in view_paths.items():
            if not isinstance(path, list):
                continue
            for depth in range(len(path)):
                partial = "/".join(str(s) for s in path[: depth + 1])
                node_claims[(vid, partial)].append(claim_id)

    print(f"  {len(node_claims):,} affected tree_nodes")
    print(f"  {len(paper_claim_ids):,} papers for by_paper nodes")

    # 2. Apply tree_node updates inside ONE connection
    print(f"\n[1/3] tree_node updates (single connection)...")
    with get_conn(readonly=False) as conn:
        cur = conn.cursor()
        n_new = 0
        n_updated = 0
        for (vid, path), cids in node_claims.items():
            row = cur.execute(
                "SELECT claim_ids FROM tree_nodes WHERE view_id=? AND path=?",
                [vid, path],
            ).fetchone()
            if row is None:
                segs = path.split("/")
                node_data = {
                    "view_id": vid, "path": path,
                    "name": segs[-1], "level": len(segs),
                    "claim_count": len(cids), "children": [],
                    "claim_ids": cids,
                }
                cur.execute(
                    "INSERT OR REPLACE INTO tree_nodes "
                    "(view_id, path, name, level, claim_count, "
                    "children, claim_ids, data) VALUES (?,?,?,?,?,?,?,?)",
                    (vid, path,
                     segs[-1].replace("_", " ").title(),
                     len(segs),
                     len(cids),
                     "[]",
                     json.dumps(cids[:100]),
                     json.dumps(node_data)),
                )
                n_new += 1
            else:
                existing_cids = (json.loads(row["claim_ids"])
                                 if row["claim_ids"] else [])
                merged = list(dict.fromkeys(existing_cids + cids))
                cur.execute(
                    "UPDATE tree_nodes SET claim_ids = ?, claim_count = ? "
                    "WHERE view_id = ? AND path = ?",
                    (json.dumps(merged[:100]), len(merged), vid, path),
                )
                n_updated += 1
        conn.commit()
        print(f"  new nodes: {n_new}, updated: {n_updated}")

    # 3. by_paper nodes
    print(f"\n[2/3] by_paper nodes for {len(paper_claim_ids):,} papers...")
    with get_conn(readonly=False) as conn:
        cur = conn.cursor()
        n_paper = 0
        for doi, cids in paper_claim_ids.items():
            title = paper_titles.get(doi) or doi
            doi_path = doi.replace("/", "__")
            node_data = {
                "view_id": "by_paper", "path": doi_path,
                "name": title, "level": 1,
                "claim_count": len(cids), "children": [],
                "claim_ids": cids, "doi": doi,
            }
            cur.execute(
                "INSERT OR REPLACE INTO tree_nodes "
                "(view_id, path, name, level, claim_count, "
                "children, claim_ids, data) VALUES (?,?,?,?,?,?,?,?)",
                ("by_paper", doi_path, title, 1, len(cids),
                 "[]", json.dumps(cids[:100]), json.dumps(node_data)),
            )
            n_paper += 1
        conn.commit()
        print(f"  upserted {n_paper} by_paper nodes")

    # 4. Author indexing (each opens own conn; outside the with block)
    print(f"\n[3/3] indexing authors for {len(paper_claim_ids):,} papers...")
    n_authors = 0
    for doi in paper_claim_ids:
        try:
            index_authors_for_doi(doi)
            n_authors += 1
        except Exception:
            pass
        if n_authors % 500 == 0:
            print(f"  [{n_authors}/{len(paper_claim_ids)}]")
    print(f"  done: {n_authors}")

    update_metadata_counts()

    print(f"\ndone in {(time.time() - started):.1f}s")
    with get_conn(readonly=True) as conn:
        n_src = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
        n_clm = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
        n_node = conn.execute("SELECT COUNT(*) AS n FROM tree_nodes").fetchone()["n"]
    print(f"final DB: {n_src:,} sources, {n_clm:,} claims, {n_node:,} tree_nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
