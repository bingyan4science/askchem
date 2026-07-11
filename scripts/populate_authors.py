"""Populate authors, paper_authors, and coauthor_edges from sources table."""
import sqlite3
import json
import hashlib
import re
import sys
from collections import defaultdict

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"

def normalize_name(name: str) -> str:
    """Normalize author name for deduplication."""
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    # Remove diacritics-ish: keep as-is for now, just normalize whitespace/case for ID
    return name

def make_author_id(name: str) -> str:
    """Create a stable author ID from a normalized name."""
    key = normalize_name(name).lower()
    return "N" + hashlib.md5(key.encode("utf-8")).hexdigest()[:15]

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Ensure coauthor_edges table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coauthor_edges (
            author_id_1 TEXT NOT NULL,
            author_id_2 TEXT NOT NULL,
            paper_count INTEGER DEFAULT 1,
            PRIMARY KEY (author_id_1, author_id_2)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coauthor_1 ON coauthor_edges(author_id_1)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coauthor_2 ON coauthor_edges(author_id_2)")
    conn.commit()

    # Clear existing data
    print("Clearing old author data...")
    conn.execute("DELETE FROM authors")
    conn.execute("DELETE FROM paper_authors")
    conn.execute("DELETE FROM coauthor_edges")
    conn.commit()

    # Step 1: Read all sources with authors
    print("Reading sources...")
    rows = conn.execute(
        "SELECT doi, authors FROM sources WHERE authors IS NOT NULL AND authors != '' AND authors != '[]'"
    ).fetchall()
    print(f"  {len(rows)} sources with author data")

    # Step 2: Build author -> papers mapping and paper -> authors mapping
    author_names = {}  # author_id -> best display name
    author_papers = defaultdict(set)  # author_id -> set of dois
    paper_author_ids = defaultdict(list)  # doi -> [(author_id, position)]

    for i, row in enumerate(rows):
        doi = row["doi"]
        try:
            names = json.loads(row["authors"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(names, list):
            continue

        for idx, name in enumerate(names):
            if not name or not isinstance(name, str) or len(name.strip()) < 2:
                continue
            name = normalize_name(name)
            aid = make_author_id(name)
            
            # Keep the longest version of the name as display name
            if aid not in author_names or len(name) > len(author_names[aid]):
                author_names[aid] = name
            
            author_papers[aid].add(doi)
            position = "first" if idx == 0 else ("last" if idx == len(names) - 1 else "middle")
            paper_author_ids[doi].append((aid, position))

        if (i + 1) % 20000 == 0:
            print(f"  Processed {i+1}/{len(rows)} sources...")

    print(f"  {len(author_names)} unique authors found")

    # Step 3: Insert authors
    print("Inserting authors...")
    batch = []
    for aid, name in author_names.items():
        paper_count = len(author_papers[aid])
        data = json.dumps({
            "author_id": aid, "name": name,
            "papers_in_index": paper_count,
        })
        batch.append((aid, name, "", "", "", "", 0, 0, 0, "[]", data))
    
    conn.executemany(
        "INSERT OR REPLACE INTO authors "
        "(author_id,name,openalex_id,orcid,institution,institution_country,"
        "h_index,works_count,cited_by_count,concepts,data) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        batch,
    )
    conn.commit()
    print(f"  Inserted {len(batch)} authors")

    # Step 4: Insert paper_authors
    print("Inserting paper_authors...")
    pa_batch = []
    for doi, authors in paper_author_ids.items():
        for aid, pos in authors:
            pa_batch.append((doi, aid, pos))
    
    conn.executemany(
        "INSERT OR IGNORE INTO paper_authors (doi, author_id, position) VALUES (?,?,?)",
        pa_batch,
    )
    conn.commit()
    print(f"  Inserted {len(pa_batch)} paper-author links")

    # Step 5: Build coauthor edges
    print("Building coauthor edges...")
    coauthor_counts = defaultdict(int)  # (aid1, aid2) -> count
    
    for doi, authors in paper_author_ids.items():
        aids = list(set(a[0] for a in authors))
        if len(aids) < 2 or len(aids) > 50:  # skip papers with too many authors
            continue
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a1, a2 = min(aids[i], aids[j]), max(aids[i], aids[j])
                coauthor_counts[(a1, a2)] += 1

    print(f"  {len(coauthor_counts)} coauthor pairs found")
    
    # Only keep edges with >= 2 shared papers to reduce noise
    strong_edges = [(a1, a2, cnt) for (a1, a2), cnt in coauthor_counts.items() if cnt >= 2]
    print(f"  {len(strong_edges)} edges with >= 2 shared papers")
    
    conn.executemany(
        "INSERT OR REPLACE INTO coauthor_edges (author_id_1, author_id_2, paper_count) VALUES (?,?,?)",
        strong_edges,
    )
    conn.commit()

    # Update author data with paper counts
    print("Updating author paper counts in data field...")
    for aid in author_names:
        pc = len(author_papers[aid])
        data = json.dumps({
            "author_id": aid, "name": author_names[aid],
            "papers_in_index": pc,
        })
        conn.execute("UPDATE authors SET data = ? WHERE author_id = ?", (data, aid))
    conn.commit()

    # Summary
    a_count = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    pa_count = conn.execute("SELECT COUNT(*) FROM paper_authors").fetchone()[0]
    ce_count = conn.execute("SELECT COUNT(*) FROM coauthor_edges").fetchone()[0]
    print(f"\nDone! authors={a_count}, paper_authors={pa_count}, coauthor_edges={ce_count}")
    conn.close()

if __name__ == "__main__":
    main()
