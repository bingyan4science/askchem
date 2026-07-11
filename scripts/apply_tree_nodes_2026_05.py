#!/usr/bin/env python3
"""Update tree_nodes for the new claims inserted by apply_incremental_2026_05.py.

The first apply crashed in stage 3 after the claims were inserted but
before tree_nodes were updated. This script reads the claims back from
the DB, scoped to this-ingest DOIs, and rebuilds the tree_node entries.

Usage::

    python3 scripts/apply_tree_nodes_2026_05.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import (  # noqa: E402
    DB_PATH, upsert_tree_node, update_metadata_counts, index_authors_for_doi,
)

TIER_1_JSONL = REPO_ROOT / "data" / "arxiv_harvest" / "tier_1.jsonl"


def main() -> int:
    started = time.time()

    # Load this-ingest DOIs
    dois: list[str] = []
    with TIER_1_JSONL.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("doi"):
                dois.append(d["doi"])
    print(f"this-ingest DOIs: {len(dois):,}")

    node_claims: dict[tuple[str, str], list[str]] = defaultdict(list)
    paper_claims: dict[str, list[str]] = defaultdict(list)
    paper_title: dict[str, str] = {}
    n_claims = 0

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        # Fetch claims for our DOIs in chunks
        BATCH = 500
        for i in range(0, len(dois), BATCH):
            chunk = dois[i:i + BATCH]
            ph = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"SELECT claim_id, source_doi, source_paper_title, view_paths "
                f"FROM claims WHERE source_doi IN ({ph})",
                chunk,
            ).fetchall()
            for r in rows:
                cid = r["claim_id"]
                doi = r["source_doi"]
                vp = json.loads(r["view_paths"]) if r["view_paths"] else {}
                paper_claims[doi].append(cid)
                paper_title[doi] = r["source_paper_title"] or doi
                n_claims += 1
                for vid, path in vp.items():
                    if not path:
                        continue
                    for depth in range(len(path)):
                        partial = "/".join(str(s) for s in path[: depth + 1])
                        node_claims[(vid, partial)].append(cid)

    print(f"collected: {n_claims:,} claims across {len(paper_claims):,} papers, "
          f"{len(node_claims):,} unique (view, path) keys")

    # Update tree_nodes
    print("\n[1/2] updating tree_nodes...")
    with sqlite3.connect(str(DB_PATH), timeout=60) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        n_updated = 0
        n_new = 0
        for i, ((vid, path), cids) in enumerate(node_claims.items()):
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
                    "claim_ids": cids[:100],
                }
                cur.execute(
                    "INSERT OR REPLACE INTO tree_nodes "
                    "(view_id, path, name, level, claim_count, "
                    "children, claim_ids, data) VALUES (?,?,?,?,?,?,?,?)",
                    (vid, path, segs[-1], len(segs),
                     len(cids), json.dumps([]), json.dumps(cids[:100]),
                     json.dumps(node_data)),
                )
                n_new += 1
            else:
                existing = json.loads(row["claim_ids"]) if row["claim_ids"] else []
                merged = list(dict.fromkeys(existing + cids))
                cur.execute(
                    "UPDATE tree_nodes SET claim_ids = ?, claim_count = ? "
                    "WHERE view_id = ? AND path = ?",
                    (json.dumps(merged[:100]), len(merged), vid, path),
                )
                n_updated += 1
            if (i + 1) % 1000 == 0:
                conn.commit()
                print(f"  [{i+1:,}/{len(node_claims):,}] new={n_new} updated={n_updated}")
        conn.commit()
        print(f"  tree_nodes: {n_new} new, {n_updated} updated")

        # by_paper nodes + author indexing
        print("\n[2/2] by_paper nodes + author indexing...")
        for j, (doi, cids) in enumerate(paper_claims.items()):
            title = paper_title.get(doi, doi)
            doi_path = doi.replace("/", "__")
            cur.execute(
                "INSERT OR REPLACE INTO tree_nodes "
                "(view_id, path, name, level, claim_count, "
                "children, claim_ids, data) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "by_paper", doi_path, title, 1,
                    len(cids), json.dumps([]), json.dumps(cids[:100]),
                    json.dumps({
                        "view_id": "by_paper", "path": doi_path,
                        "name": title, "level": 1,
                        "claim_count": len(cids), "children": [],
                        "claim_ids": cids[:100], "doi": doi,
                    }),
                ),
            )
            if (j + 1) % 200 == 0:
                conn.commit()
                print(f"  [{j+1:,}/{len(paper_claims):,}] by_paper nodes")
        conn.commit()
        print("  by_paper nodes done")

    print("\nindexing authors...")
    for j, doi in enumerate(paper_claims):
        try:
            index_authors_for_doi(doi)
        except Exception as exc:
            if (j + 1) % 200 == 0:
                print(f"  authors: {j+1}/{len(paper_claims)} (err: {exc!r:80})")
            continue
        if (j + 1) % 500 == 0:
            print(f"  authors: {j+1}/{len(paper_claims)}")
    print("  authors done")

    update_metadata_counts()
    print(f"\ndone in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
