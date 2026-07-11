"""
Refresh citation counts from Semantic Scholar and recompute author citation totals.

Usage:
    python -m src.refresh_citations          # refresh all
    python -m src.refresh_citations --limit 1000  # refresh top 1000 by current citations

Source: Semantic Scholar Academic Graph API
  POST https://api.semanticscholar.org/graph/v1/paper/batch
  Up to 500 DOIs per request, 1 request/sec with API key.
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH_SIZE = 500
REQUEST_DELAY = 1.1  # seconds between requests (S2 rate limit: 1/sec)

DB_PATH = Path(__file__).parent.parent / "chemtree.db"


def get_db_path() -> Path:
    return Path(os.environ.get("CHEMTREE_DB", str(DB_PATH)))


def _s2_headers():
    key = os.environ.get("S2_API_KEY", "")
    if not key:
        print("WARNING: S2_API_KEY not set. Requests may be rate-limited.")
    return {"x-api-key": key} if key else {}


def refresh_citations(limit: int | None = None):
    """Fetch fresh citation counts from S2 and update the database."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get all DOIs
    if limit:
        rows = c.execute(
            "SELECT doi FROM sources WHERE doi != '' ORDER BY citation_count DESC LIMIT ?",
            [limit],
        ).fetchall()
    else:
        rows = c.execute("SELECT doi FROM sources WHERE doi != ''").fetchall()

    dois = [r["doi"] for r in rows]
    print(f"Refreshing citations for {len(dois):,} papers...", flush=True)

    headers = _s2_headers()
    updated = 0
    failed_batches = 0
    total_batches = (len(dois) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(dois), BATCH_SIZE):
        batch_dois = dois[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        ids = [f"DOI:{doi}" for doi in batch_dois]

        try:
            resp = requests.post(
                S2_BATCH,
                params={"fields": "externalIds,citationCount"},
                json={"ids": ids},
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                print(f"  Rate limited at batch {batch_num}, waiting 60s...", flush=True)
                time.sleep(60)
                resp = requests.post(
                    S2_BATCH,
                    params={"fields": "externalIds,citationCount"},
                    json={"ids": ids},
                    headers=headers,
                    timeout=30,
                )

            if resp.status_code != 200:
                print(f"  Batch {batch_num}/{total_batches}: HTTP {resp.status_code}", flush=True)
                failed_batches += 1
                time.sleep(REQUEST_DELAY)
                continue

            results = resp.json()
            batch_updated = 0
            for paper in results:
                if paper is None:
                    continue
                doi = (paper.get("externalIds") or {}).get("DOI")
                cite_count = paper.get("citationCount")
                if doi and cite_count is not None:
                    # Update sources table
                    old_data_row = c.execute("SELECT data FROM sources WHERE doi = ?", [doi]).fetchone()
                    if old_data_row:
                        data = json.loads(old_data_row["data"])
                        data["citation_count"] = cite_count
                        c.execute(
                            "UPDATE sources SET citation_count = ?, data = ? WHERE doi = ?",
                            [cite_count, json.dumps(data), doi],
                        )
                        batch_updated += 1

            conn.commit()
            updated += batch_updated

            if batch_num % 10 == 0 or batch_num == total_batches:
                print(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"{updated:,} updated so far",
                    flush=True,
                )

        except requests.RequestException as e:
            print(f"  Batch {batch_num}/{total_batches}: request error: {e}", flush=True)
            failed_batches += 1

        time.sleep(REQUEST_DELAY)

    print(f"\nCitation refresh complete: {updated:,} papers updated, {failed_batches} failed batches")
    return updated


def recompute_author_citations(conn=None):
    """Recompute cited_by_count for all authors from current source citation_counts."""
    close_conn = False
    if conn is None:
        db_path = get_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        close_conn = True
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("Recomputing author citation totals...", flush=True)
    rows = c.execute("""
        SELECT pa.author_id,
               SUM(s.citation_count) as total_cites,
               COUNT(DISTINCT pa.doi) as paper_count
        FROM paper_authors pa
        JOIN sources s ON pa.doi = s.doi
        GROUP BY pa.author_id
    """).fetchall()

    batch = []
    for r in rows:
        cites = int(r["total_cites"] or 0)
        wcount = int(r["paper_count"] or 0)
        aid = r["author_id"]

        old_row = c.execute("SELECT data FROM authors WHERE author_id = ?", [aid]).fetchone()
        if old_row:
            data = json.loads(old_row["data"])
            data["cited_by_count"] = cites
            data["works_count"] = wcount
            batch.append((cites, wcount, json.dumps(data), aid))

    for i in range(0, len(batch), 10000):
        chunk = batch[i : i + 10000]
        c.executemany(
            "UPDATE authors SET cited_by_count = ?, works_count = ?, data = ? WHERE author_id = ?",
            chunk,
        )
        conn.commit()
        print(f"  Authors updated: {min(i + 10000, len(batch)):,}/{len(batch):,}", flush=True)

    print(f"Author citations recomputed for {len(batch):,} authors", flush=True)

    if close_conn:
        conn.close()


def update_citation_metadata():
    """Store citation source and refresh timestamp in the metadata table."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for k, v in [
        ("citation_source", "Semantic Scholar Academic Graph API"),
        ("citation_source_url", "https://api.semanticscholar.org/"),
        ("citations_updated_at", now),
    ]:
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()
    print(f"Metadata updated: citations_updated_at = {now}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Refresh citation counts from Semantic Scholar")
    parser.add_argument("--limit", type=int, default=None, help="Only refresh top N papers by current citations")
    args = parser.parse_args()

    refresh_citations(limit=args.limit)
    recompute_author_citations()
    update_citation_metadata()
    print("\nDone!")


if __name__ == "__main__":
    main()
