"""Backfill missing citation counts from Semantic Scholar API."""
import sqlite3
import json
import sys
import time
import urllib.request
import urllib.error

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=citationCount"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT doi, data FROM sources WHERE citation_count IS NULL OR citation_count = 0"
    ).fetchall()
    print(f"Sources missing citations: {len(rows)}")

    updated = 0
    failed = 0

    for r in rows:
        doi = r["doi"]
        if not doi:
            continue

        try:
            url = S2_API.format(doi=urllib.request.quote(doi, safe=''))
            req = urllib.request.Request(url, headers={"User-Agent": "AskChem/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            cite_count = data.get("citationCount", 0) or 0

            if cite_count > 0:
                src_data = json.loads(r["data"])
                src_data["citation_count"] = cite_count
                conn.execute(
                    "UPDATE sources SET citation_count = ?, data = ? WHERE doi = ?",
                    [cite_count, json.dumps(src_data), doi],
                )
                updated += 1
                print(f"  {doi}: {cite_count} citations")
            else:
                print(f"  {doi}: 0 citations (S2)")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  {doi}: not found on S2")
            elif e.code == 429:
                print(f"  Rate limited, waiting 5s...")
                time.sleep(5)
                failed += 1
            else:
                print(f"  {doi}: HTTP {e.code}")
                failed += 1
        except Exception as e:
            print(f"  {doi}: error {e}")
            failed += 1

        time.sleep(0.5)

    conn.commit()
    print(f"\nDone: updated {updated}, failed {failed}")

    remaining = conn.execute(
        "SELECT COUNT(*) FROM sources WHERE citation_count IS NULL OR citation_count = 0"
    ).fetchone()[0]
    print(f"Still missing: {remaining}")
    conn.close()


if __name__ == "__main__":
    main()
