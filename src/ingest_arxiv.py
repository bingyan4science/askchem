"""
Ingest harvested arXiv papers into chemtree.db.

Reads JSONL files from data/arxiv_harvest/ and inserts papers into the
sources table, with full deduplication against existing DOIs and arXiv IDs.

Usage:
    python src/ingest_arxiv.py                # Ingest all harvested files
    python src/ingest_arxiv.py --dry-run      # Count without inserting
"""

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
HARVEST_DIR = DATA_DIR / "arxiv_harvest"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"

BATCH_SIZE = 500


def get_db_path():
    return Path(os.environ.get("CHEMTREE_DB", str(DB_PATH)))


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arXiv.{arxiv_id}"


def make_source_row(paper: dict):
    aid = paper["arxiv_id"]
    doi = paper.get("doi") or arxiv_doi(aid)
    authors_list = [{"name": a} for a in paper.get("authors", [])]
    authors_json = json.dumps(authors_list)

    data_blob = {
        "arxiv_id": aid,
        "doi": doi,
        "title": paper.get("title", ""),
        "authors": authors_list,
        "year": paper.get("year"),
        "venue": f"arXiv:{aid}",
        "abstract": paper.get("abstract", ""),
        "categories": paper.get("categories", []),
        "published": paper.get("published", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "arxiv_url": paper.get("arxiv_url", ""),
        "source_type": "arxiv",
        "externalIds": {"ArXiv": aid, "DOI": doi},
        "openAccessPdf": {"url": paper.get("pdf_url", "")},
        "citationCount": 0,
    }

    return (
        doi,
        paper.get("title", ""),
        authors_json,
        paper.get("year") or 0,
        f"arXiv:{aid}",
        paper.get("abstract", ""),
        0,
        paper.get("pdf_url", ""),
        json.dumps(data_blob),
    )


def load_existing_keys(conn):
    """Load all DOIs and arXiv IDs already in the DB for final dedup."""
    rows = conn.execute("SELECT doi, data FROM sources").fetchall()
    dois = set()
    arxiv_ids = set()
    for doi, data_str in rows:
        if doi:
            dois.add(doi.lower().strip())
        if data_str:
            try:
                ext = json.loads(data_str).get("externalIds") or {}
                aid = ext.get("ArXiv", "")
                if aid:
                    arxiv_ids.add(aid.lower())
            except (json.JSONDecodeError, AttributeError):
                pass
            if doi and "10.48550/arxiv" in doi.lower():
                m = re.search(r"10\.48550/arXiv\.(.+)", doi, re.IGNORECASE)
                if m:
                    arxiv_ids.add(m.group(1).lower())
    return dois, arxiv_ids


def ingest(dry_run=False):
    if not HARVEST_DIR.exists():
        print("No harvest directory found. Run harvest_arxiv.py first.")
        return

    jsonl_files = sorted(HARVEST_DIR.glob("harvest_*.jsonl"))
    if not jsonl_files:
        print("No harvest files found.")
        return

    print(f"{'='*60}")
    print(f"arXiv Ingestion — {datetime.now().isoformat()}")
    print(f"Files: {len(jsonl_files)}")
    print(f"{'='*60}")

    all_papers = []
    for jf in jsonl_files:
        count = 0
        with open(jf) as f:
            for line in f:
                try:
                    all_papers.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError:
                    pass
        print(f"  {jf.name}: {count:,} papers")

    print(f"\nTotal harvested papers: {len(all_papers):,}")

    if dry_run:
        print("DRY RUN — no database changes.")
        return

    db_path = get_db_path()
    if not db_path.exists():
        print(f"ERROR: {db_path} not found.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()

    print("\nLoading existing keys for final dedup check...", flush=True)
    existing_dois, existing_arxiv_ids = load_existing_keys(conn)
    print(f"  {len(existing_dois):,} DOIs, {len(existing_arxiv_ids):,} arXiv IDs in DB")

    inserted = 0
    skipped = 0
    batch = []

    for paper in all_papers:
        aid = paper["arxiv_id"].lower()
        doi = paper.get("doi") or arxiv_doi(paper["arxiv_id"])
        doi_low = doi.lower().strip()
        adoi_low = arxiv_doi(aid).lower()

        if doi_low in existing_dois or adoi_low in existing_dois or aid in existing_arxiv_ids:
            skipped += 1
            continue

        existing_dois.add(doi_low)
        existing_arxiv_ids.add(aid)
        batch.append(make_source_row(paper))

        if len(batch) >= BATCH_SIZE:
            c.executemany(
                "INSERT OR IGNORE INTO sources "
                "(doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)
            batch = []
            print(f"  Inserted: {inserted:,}", flush=True)

    if batch:
        c.executemany(
            "INSERT OR IGNORE INTO sources "
            "(doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()
        inserted += len(batch)

    conn.close()

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Inserted: {inserted:,}")
    print(f"  Skipped (duplicate): {skipped:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into chemtree.db")
    parser.add_argument("--dry-run", action="store_true", help="Count without inserting")
    args = parser.parse_args()
    ingest(dry_run=args.dry_run)
