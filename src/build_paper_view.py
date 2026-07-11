"""
Build the by_paper view: for each paper (DOI), list all extracted claims.

Tree structure:
  root  →  paper_doi  →  (claims stored as claim_ids on the paper node)

Each paper node stores the paper title as its display name and all
claim_ids extracted from that paper.

Usage:
    python src/build_paper_view.py
"""

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH = Path(__file__).parent.parent / "chemtree.db"
VIEW_ID = "by_paper"


def main():
    db_path = DB_PATH
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.1f} GB)")
    t0 = time.time()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print("Phase 1: Reading claims and grouping by paper...", flush=True)

    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"  Total claims: {total:,}")

    papers: dict[str, dict] = {}  # doi -> {title, claim_ids}

    batch_size = 50000
    offset = 0
    while offset < total:
        rows = conn.execute(
            "SELECT claim_id, data FROM claims LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break

        for row in rows:
            claim_id = row[0]
            data = json.loads(row[1])
            doi = data.get('source_doi', '').strip()
            title = data.get('source_paper_title', '').strip()

            if not doi:
                continue

            if doi not in papers:
                papers[doi] = {'title': title, 'claim_ids': []}
            papers[doi]['claim_ids'].append(claim_id)

        offset += batch_size
        elapsed = time.time() - t0
        print(f"  Processed {offset:,}/{total:,} ({elapsed:.0f}s)", flush=True)

    print(f"\n  Unique papers: {len(papers):,}")
    total_claims_in_papers = sum(len(p['claim_ids']) for p in papers.values())
    print(f"  Claims with DOI: {total_claims_in_papers:,}")

    print("\nPhase 2: Building tree nodes...", flush=True)
    t1 = time.time()

    c = conn.cursor()
    c.execute("DELETE FROM tree_nodes WHERE view_id = ?", (VIEW_ID,))

    batch = []
    for doi, pdata in papers.items():
        doi_path = doi.replace('/', '__')
        name = pdata['title'] if pdata['title'] else doi
        claim_count = len(pdata['claim_ids'])
        claim_ids_json = json.dumps(pdata['claim_ids'])
        children_json = json.dumps([])

        node_data = {
            'view_id': VIEW_ID,
            'path': doi_path,
            'name': name,
            'level': 1,
            'claim_count': claim_count,
            'children': [],
            'claim_ids': pdata['claim_ids'],
            'doi': doi,
        }

        batch.append((
            VIEW_ID, doi_path, name, 1, claim_count,
            children_json, claim_ids_json, json.dumps(node_data),
        ))

    root_children = sorted(
        [doi.replace('/', '__') for doi in papers.keys()]
    )
    root_data = {
        'view_id': VIEW_ID,
        'path': '',
        'name': 'By Paper',
        'level': 0,
        'claim_count': total_claims_in_papers,
        'children': root_children,
        'claim_ids': [],
    }
    batch.append((
        VIEW_ID, '', 'By Paper', 0, total_claims_in_papers,
        json.dumps(root_children), json.dumps([]), json.dumps(root_data),
    ))

    chunk_size = 10000
    for i in range(0, len(batch), chunk_size):
        c.executemany(
            "INSERT OR REPLACE INTO tree_nodes "
            "(view_id, path, name, level, claim_count, children, claim_ids, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            batch[i:i + chunk_size]
        )
    conn.commit()

    elapsed2 = time.time() - t1
    print(f"  Tree nodes created: {len(batch):,} in {elapsed2:.0f}s")

    # Register the view
    view_data = {
        'view_id': VIEW_ID,
        'name': 'By Paper',
        'description': 'Browse claims organized by their source paper (DOI)',
        'organizing_principle': 'source_paper',
        'root_node_id': '',
        'node_count': len(batch),
        'claim_count': total_claims_in_papers,
        'max_depth': 1,
        'created_at': '',
        'updated_at': '',
    }
    c.execute(
        "INSERT OR REPLACE INTO views (view_id, name, description, data) VALUES (?, ?, ?, ?)",
        (VIEW_ID, 'By Paper', view_data['description'], json.dumps(view_data))
    )
    conn.commit()

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE in {total_elapsed:.0f}s")
    print(f"  Papers: {len(papers):,}")
    print(f"  Claims covered: {total_claims_in_papers:,} / {total:,}")
    print(f"  Tree nodes: {len(batch):,}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
