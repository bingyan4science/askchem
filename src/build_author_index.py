"""
Build the author index by fetching disambiguated author data from OpenAlex.

Strategy: Query OpenAlex Works API by DOI (batch 50 per request) to get
per-paper disambiguated authorships. This resolves name ambiguity — each
"Wei Zhang" on a specific paper maps to a unique OpenAlex author ID.

Populates (these tables PERSIST in the DB; the search / coauthor-network
endpoints just read them -- the index is NOT rebuilt per request):
  - authors table          (unique authors + metadata)
  - paper_authors table    (doi, author_id, position)
  - coauthor_edges table   (co-authorship pairs with paper counts)
  - author_index_progress  (DOIs already attempted -> lets a run RESUME)

The crawl writes to the DB INCREMENTALLY (every --flush-every batches) and
records progress, so it is fully resumable and throttle-proof: if OpenAlex
rate-limits us it stops cleanly and you just re-run to continue where it left
off. This also makes future updates cheap -- after ingesting new papers, just
re-run: only DOIs not already in author_index_progress are crawled.

Usage:
    python src/build_author_index.py                 # crawl / resume (default)
    python src/build_author_index.py --reset          # wipe + start over
    python src/build_author_index.py --no-enrich       # skip h-index enrichment
    python src/build_author_index.py --max-batches 10  # smoke test
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from askchem import db as _db_mod
    DB_PATH = _db_mod.get_db_path()
except Exception:
    DB_PATH = Path(__file__).parent.parent / "askchem.db"
OPENALEX_BASE = "https://api.openalex.org"
EMAIL = "bing.yan@nyu.edu"
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.2       # polite pacing between successful requests
FLUSH_EVERY = 40             # persist to DB every N batches (~2,000 papers)
MAX_CONSECUTIVE_FAILS = 12   # circuit-breaker: stop & let the user resume later


def _http_get(url: str):
    """GET -> parsed JSON, or None on failure (429 / 5xx / transport).

    Honors Retry-After on 429 and backs off; returns None after exhausting
    retries so the caller can decide to stop-and-resume rather than lose data.
    """
    req = urllib.request.Request(url, headers={"User-Agent": f"AskChem/1.0 (mailto:{EMAIL})"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                wait = int(ra) if (ra and ra.isdigit()) else 2 ** (attempt + 1)
                time.sleep(min(wait, 30))
                continue
            if e.code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1 + attempt)
    return None


def fetch_works_batch(dois: list[str]):
    """Return list of OpenAlex works for these DOIs, or None on request failure."""
    doi_filter = "|".join(dois)
    url = (f"{OPENALEX_BASE}/works?filter=doi:{doi_filter}"
           f"&per-page={BATCH_SIZE}&select=doi,authorships&mailto={EMAIL}")
    data = _http_get(url)
    return None if data is None else data.get("results", [])


def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS authors (
            author_id TEXT PRIMARY KEY, name TEXT, openalex_id TEXT, orcid TEXT,
            institution TEXT, institution_country TEXT, h_index INTEGER DEFAULT 0,
            works_count INTEGER DEFAULT 0, cited_by_count INTEGER DEFAULT 0,
            concepts TEXT, data TEXT)""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_authors (
            doi TEXT NOT NULL, author_id TEXT NOT NULL, position TEXT,
            PRIMARY KEY (doi, author_id))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_authors_doi ON paper_authors(doi)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coauthor_edges (
            author_id_1 TEXT NOT NULL, author_id_2 TEXT NOT NULL,
            paper_count INTEGER DEFAULT 1, PRIMARY KEY (author_id_1, author_id_2))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coauthor_1 ON coauthor_edges(author_id_1)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coauthor_2 ON coauthor_edges(author_id_2)")
    conn.execute("CREATE TABLE IF NOT EXISTS author_index_progress (doi TEXT PRIMARY KEY)")
    conn.commit()


def _flush(conn, authors_data, pa_rows, edge_counts, progress_dois):
    """Persist one chunk. Idempotent: safe to re-run / resume."""
    if authors_data:
        rows = []
        for aid, ad in authors_data.items():
            rows.append((aid, ad.get("name", ""), ad.get("openalex_id", ""),
                         ad.get("orcid", ""), ad.get("institution", ""),
                         ad.get("institution_country", ""), 0, 0, 0, "[]",
                         json.dumps(ad)))
        # Preserve any existing metadata (e.g. enrichment) — only insert new authors;
        # existing rows keep their richer data. papers_in_index is recomputed below.
        conn.executemany(
            "INSERT OR IGNORE INTO authors (author_id,name,openalex_id,orcid,"
            "institution,institution_country,h_index,works_count,cited_by_count,"
            "concepts,data) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    if pa_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO paper_authors (doi,author_id,position) VALUES (?,?,?)",
            pa_rows)
    if edge_counts:
        conn.executemany(
            "INSERT INTO coauthor_edges (author_id_1,author_id_2,paper_count) "
            "VALUES (?,?,?) ON CONFLICT(author_id_1,author_id_2) DO UPDATE SET "
            "paper_count = paper_count + excluded.paper_count",
            [(a1, a2, c) for (a1, a2), c in edge_counts.items()])
    if progress_dois:
        conn.executemany("INSERT OR IGNORE INTO author_index_progress (doi) VALUES (?)",
                         [(d,) for d in progress_dois])
    # Refresh papers_in_index (stored in authors.data) for the authors touched here,
    # from the authoritative paper_authors counts -- correct across resumes.
    if authors_data:
        touched = list(authors_data.keys())
        for i in range(0, len(touched), 400):
            ids = touched[i:i + 400]
            ph = ",".join("?" for _ in ids)
            counts = dict(conn.execute(
                f"SELECT author_id, COUNT(*) FROM paper_authors "
                f"WHERE author_id IN ({ph}) GROUP BY author_id", ids).fetchall())
            for aid, cnt in counts.items():
                r = conn.execute("SELECT data FROM authors WHERE author_id=?", [aid]).fetchone()
                if r:
                    d = json.loads(r[0]); d["papers_in_index"] = cnt
                    conn.execute("UPDATE authors SET data=? WHERE author_id=?",
                                 [json.dumps(d), aid])
    conn.commit()


def build_from_sources(conn):
    """Build the FULL author index directly from ``sources.authors`` (author-name
    lists that already cover ~96% of the corpus), without OpenAlex.

    Uses the exact same name-keyed id scheme as the server fallback
    (``db._local_author_id`` -> ``local:<sha1>``) so ids resolve consistently
    across the indexed path, the JSON fallback, and per-DOI lookups. OpenAlex
    enrichment (institution / h-index / concepts / disambiguation) can be layered
    on top later via ``_enrich_top_authors`` once daily credits reset.

    Author position (first/middle/last) is derived from list order. Coauthor
    edges are generated pairwise per paper, skipping pathological consortium
    papers (> MAX_AUTHORS_FOR_EDGES authors) to keep the edge table meaningful;
    those papers still contribute to ``paper_authors`` and author paper counts.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from askchem import db as _db

    MAX_AUTHORS_FOR_EDGES = 50

    print("Wiping author tables for a fresh name-keyed build from sources...")
    for t in ("authors", "paper_authors", "coauthor_edges", "author_index_progress"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()

    rows = conn.execute(
        "SELECT doi, authors, citation_count FROM sources "
        "WHERE doi IS NOT NULL AND authors IS NOT NULL "
        "AND authors != '' AND authors != '[]'"
    ).fetchall()
    print(f"Papers with author lists: {len(rows):,}\n")

    # aid -> {name, cites, papers}
    authors_meta: dict[str, dict] = {}
    pa_rows: list[tuple] = []
    edge_counts: Counter = Counter()
    prog: list[str] = []
    t0 = time.time()

    for n, (doi, authors_raw, cites) in enumerate(rows):
        names = _db._parse_source_authors(authors_raw)
        prog.append(doi)
        if not names:
            continue
        cites = int(cites or 0)
        ids_in_paper: list[str] = []
        for name in names:
            aid = _db._local_author_id(name)
            m = authors_meta.get(aid)
            if m is None:
                authors_meta[aid] = {"name": name, "cites": cites, "papers": 1}
            else:
                m["papers"] += 1
                m["cites"] += cites
            pos = _db._position_in_authors(name, names)
            pa_rows.append((doi, aid, pos))
            ids_in_paper.append(aid)
        if 2 <= len(ids_in_paper) <= MAX_AUTHORS_FOR_EDGES:
            for a1, a2 in combinations(sorted(set(ids_in_paper)), 2):
                edge_counts[(a1, a2)] += 1

        if (n + 1) % 20000 == 0:
            print(f"  scanned {n+1:,}/{len(rows):,} papers "
                  f"| {len(authors_meta):,} authors, {len(edge_counts):,} edges "
                  f"| {time.time()-t0:.0f}s", flush=True)

    print(f"\nWriting {len(authors_meta):,} authors...", flush=True)
    author_rows = []
    for aid, m in authors_meta.items():
        data = {
            "author_id": aid, "name": m["name"], "openalex_id": "", "orcid": "",
            "institution": "", "institution_country": "", "h_index": 0,
            "works_count": m["papers"], "cited_by_count": m["cites"],
            "papers_in_index": m["papers"], "concepts": [],
        }
        author_rows.append((aid, m["name"], "", "", "", "", 0, m["papers"],
                            m["cites"], "[]", json.dumps(data)))
    conn.executemany(
        "INSERT OR REPLACE INTO authors (author_id,name,openalex_id,orcid,"
        "institution,institution_country,h_index,works_count,cited_by_count,"
        "concepts,data) VALUES (?,?,?,?,?,?,?,?,?,?,?)", author_rows)

    print(f"Writing {len(pa_rows):,} paper-author links...", flush=True)
    for i in range(0, len(pa_rows), 50000):
        conn.executemany(
            "INSERT OR IGNORE INTO paper_authors (doi,author_id,position) "
            "VALUES (?,?,?)", pa_rows[i:i + 50000])

    print(f"Writing {len(edge_counts):,} coauthor edges...", flush=True)
    edge_rows = [(a1, a2, c) for (a1, a2), c in edge_counts.items()]
    for i in range(0, len(edge_rows), 50000):
        conn.executemany(
            "INSERT OR IGNORE INTO coauthor_edges (author_id_1,author_id_2,paper_count) "
            "VALUES (?,?,?)", edge_rows[i:i + 50000])

    conn.executemany("INSERT OR IGNORE INTO author_index_progress (doi) VALUES (?)",
                     [(d,) for d in prog])
    conn.commit()

    covered = conn.execute("SELECT COUNT(DISTINCT doi) FROM paper_authors").fetchone()[0]
    n_auth = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    n_edge = conn.execute("SELECT COUNT(*) FROM coauthor_edges").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"  FROM-SOURCES BUILD COMPLETE in {time.time()-t0:.0f}s")
    print(f"  Papers covered: {covered:,}")
    print(f"  Authors: {n_auth:,}  |  Co-author edges: {n_edge:,}")
    print(f"{'='*60}")


def _enrich_top_authors(conn, max_enrich=10000):
    """Best-effort: add h-index / institution / concepts for the most-indexed
    authors. Skips silently on throttle (the coauthor network works without it)."""
    top = [r[0] for r in conn.execute(
        "SELECT author_id FROM paper_authors GROUP BY author_id "
        "ORDER BY COUNT(*) DESC LIMIT ?", [max_enrich]).fetchall()]
    print(f"\nEnriching top {len(top):,} authors...", flush=True)
    enriched = 0
    for i in range(0, len(top), 50):
        batch = top[i:i + 50]
        ids_filter = "|".join(f"https://openalex.org/{aid}" for aid in batch)
        url = (f"{OPENALEX_BASE}/authors?filter=openalex:{ids_filter}&per-page=50"
               f"&select=id,display_name,orcid,last_known_institutions,summary_stats,"
               f"works_count,cited_by_count,x_concepts&mailto={EMAIL}")
        data = _http_get(url)
        if data is None:
            print("  enrichment throttled; stopping (re-run to finish enrichment)")
            break
        for a in data.get("results", []):
            aid = a.get("id", "").split("/")[-1]
            row = conn.execute("SELECT data FROM authors WHERE author_id=?", [aid]).fetchone()
            if not row:
                continue
            d = json.loads(row[0])
            insts = a.get("last_known_institutions") or []
            inst = insts[0].get("display_name", "") if insts else d.get("institution", "")
            country = insts[0].get("country_code", "") if insts else d.get("institution_country", "")
            stats = a.get("summary_stats", {})
            concepts = [{"name": c.get("display_name", ""), "score": round(c.get("score", 0), 3)}
                        for c in (a.get("x_concepts") or [])[:10]]
            d.update({"name": a.get("display_name", d.get("name", "")),
                      "institution": inst, "institution_country": country,
                      "h_index": stats.get("h_index", 0),
                      "works_count": a.get("works_count", 0),
                      "cited_by_count": a.get("cited_by_count", 0), "concepts": concepts})
            conn.execute(
                "UPDATE authors SET name=?, institution=?, institution_country=?, "
                "h_index=?, works_count=?, cited_by_count=?, concepts=?, data=? "
                "WHERE author_id=?",
                [d["name"], inst, country, d["h_index"], d["works_count"],
                 d["cited_by_count"], json.dumps(concepts), json.dumps(d), aid])
            enriched += 1
        conn.commit()
        if (i // 50 + 1) % 20 == 0:
            print(f"  enriched {enriched:,}...", flush=True)
        time.sleep(RATE_LIMIT_DELAY)
    print(f"  Enriched {enriched:,} authors")


def main():
    ap = argparse.ArgumentParser(description="Build/resume the author index from OpenAlex")
    ap.add_argument("--db", type=str, default=str(DB_PATH))
    ap.add_argument("--reset", action="store_true", help="wipe tables + progress, start over")
    ap.add_argument("--from-sources", action="store_true",
                    help="build the full name-keyed index from sources.authors (no OpenAlex)")
    ap.add_argument("--no-enrich", action="store_true", help="skip author-profile enrichment")
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches (smoke test)")
    ap.add_argument("--flush-every", type=int, default=FLUSH_EVERY)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}"); sys.exit(1)
    print(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.1f} GB)")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_tables(conn)

    if args.from_sources:
        build_from_sources(conn)
        conn.close()
        return

    if args.reset:
        for t in ("authors", "paper_authors", "coauthor_edges", "author_index_progress"):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
        print("Reset: cleared authors / paper_authors / coauthor_edges / progress")

    # On first run, seed progress from already-indexed papers so we don't recrawl
    # DOIs that are already in paper_authors.
    if conn.execute("SELECT COUNT(*) FROM author_index_progress").fetchone()[0] == 0:
        conn.execute("INSERT OR IGNORE INTO author_index_progress (doi) "
                     "SELECT DISTINCT doi FROM paper_authors WHERE doi IS NOT NULL")
        conn.commit()

    all_dois = [r[0] for r in conn.execute(
        "SELECT doi FROM sources WHERE doi IS NOT NULL").fetchall()]
    done = {r[0] for r in conn.execute("SELECT doi FROM author_index_progress").fetchall()}
    remaining = [d for d in all_dois if d not in done]
    print(f"Papers with DOI: {len(all_dois):,}  |  already attempted: {len(done):,}  "
          f"|  remaining: {len(remaining):,}")
    if not remaining:
        print("Nothing to crawl. Author index is up to date.")
        if not args.no_enrich:
            _enrich_top_authors(conn)
        conn.close()
        return

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    if args.max_batches:
        batches = batches[:args.max_batches]
    print(f"Batches to crawl: {len(batches):,}\n")

    authors_data, pa_rows, edge_counts, prog = {}, [], Counter(), []
    t0 = time.time()
    works_found = authorships = fails = consecutive = 0
    completed = True

    for bi, batch in enumerate(batches):
        works = fetch_works_batch(batch)
        if works is None:
            fails += 1; consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILS:
                print(f"\n  OpenAlex throttling us ({consecutive} consecutive failures). "
                      f"Stopping cleanly -- progress is saved; re-run to resume.")
                completed = False
                break
            time.sleep(min(2 ** consecutive, 30))
            continue
        consecutive = 0
        works_found += len(works)
        for work in works:
            doi = (work.get("doi") or "").replace("https://doi.org/", "")
            ids_in_paper = []
            for a in work.get("authorships", []):
                au = a.get("author", {})
                oa = au.get("id", "")
                if not oa:
                    continue
                aid = oa.split("/")[-1]
                pa_rows.append((doi, aid, a.get("author_position", "middle")))
                ids_in_paper.append(aid)
                authorships += 1
                if aid not in authors_data:
                    insts = a.get("institutions", [])
                    authors_data[aid] = {
                        "author_id": aid, "name": au.get("display_name", ""),
                        "openalex_id": oa,
                        "orcid": (au.get("orcid") or "").replace("https://orcid.org/", ""),
                        "institution": insts[0].get("display_name", "") if insts else "",
                        "institution_country": insts[0].get("country_code", "") if insts else "",
                    }
            for a1, a2 in combinations(sorted(set(ids_in_paper)), 2):
                edge_counts[(a1, a2)] += 1
        prog.extend(batch)   # got a response -> these DOIs are attempted

        if (bi + 1) % args.flush_every == 0 or bi == len(batches) - 1:
            _flush(conn, authors_data, pa_rows, edge_counts, prog)
            authors_data, pa_rows, edge_counts, prog = {}, [], Counter(), []
            elapsed = time.time() - t0
            rate = (bi + 1) / max(elapsed, 1)
            eta = (len(batches) - bi - 1) / max(rate, 0.01)
            total_pa = conn.execute("SELECT COUNT(*) FROM paper_authors").fetchone()[0]
            total_au = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
            print(f"  [{bi+1:,}/{len(batches):,}] flushed | DB: {total_au:,} authors, "
                  f"{total_pa:,} links | {rate:.1f} req/s, ETA {eta:.0f}s, fails {fails}",
                  flush=True)
        time.sleep(RATE_LIMIT_DELAY)

    # final flush of any tail
    _flush(conn, authors_data, pa_rows, edge_counts, prog)

    covered = conn.execute("SELECT COUNT(DISTINCT doi) FROM paper_authors").fetchone()[0]
    n_auth = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    n_edge = conn.execute("SELECT COUNT(*) FROM coauthor_edges").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"  {'COMPLETE' if completed else 'PARTIAL (resume with a re-run)'} in "
          f"{time.time()-t0:.0f}s")
    print(f"  Papers covered: {covered:,} / {len(all_dois):,}")
    print(f"  Authors: {n_auth:,}  |  Co-author edges: {n_edge:,}")
    print(f"{'='*60}")

    if completed and not args.no_enrich:
        _enrich_top_authors(conn)
    conn.close()


if __name__ == "__main__":
    main()
