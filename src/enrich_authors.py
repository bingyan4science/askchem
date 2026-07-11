"""
AskChem Author Enrichment — OpenAlex integration.

Enriches author data from OpenAlex (free, no key needed):
  - ORCID, institution, h-index, works count, cited-by count
  - Research concepts (topics)
  - Co-author network

Usage:
    python src/enrich_authors.py                    # Enrich all authors
    python src/enrich_authors.py --status           # Show progress
    python src/enrich_authors.py --max 1000         # Limit to 1000 authors
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import requests

sys.path.insert(0, str(Path(__file__).parent))
from askchem import db

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_FILE = DATA_DIR / "author_enrichment_checkpoint.json"

OPENALEX_BASE = "https://api.openalex.org"
RATE_LIMIT_DELAY = 0.1  # OpenAlex allows 10 req/s for polite pool
EMAIL = "askchem@mit.edu"


def get_unique_authors() -> list[dict]:
    """Extract unique authors from the sources table."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT doi, authors, year, citation_count FROM sources WHERE authors IS NOT NULL"
        ).fetchall()

    author_papers = defaultdict(list)
    for row in rows:
        try:
            authors = json.loads(row["authors"])
        except (json.JSONDecodeError, TypeError):
            continue
        for i, name in enumerate(authors):
            if not name or not isinstance(name, str):
                continue
            author_papers[name.strip()].append({
                "doi": row["doi"],
                "year": row["year"],
                "citations": row["citation_count"],
                "position": "first" if i == 0 else ("last" if i == len(authors) - 1 else "middle"),
            })

    authors = []
    for name, papers in author_papers.items():
        total_citations = sum(p["citations"] or 0 for p in papers)
        authors.append({
            "name": name,
            "paper_count": len(papers),
            "total_citations": total_citations,
            "first_author_count": sum(1 for p in papers if p["position"] == "first"),
            "last_author_count": sum(1 for p in papers if p["position"] == "last"),
        })

    authors.sort(key=lambda a: a["total_citations"], reverse=True)
    return authors


def search_openalex_author(name: str) -> dict | None:
    """Search OpenAlex for an author by name."""
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/authors",
            params={
                "search": name,
                "per_page": 1,
                "mailto": EMAIL,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None

        author = results[0]
        return {
            "openalex_id": author.get("id", ""),
            "display_name": author.get("display_name", ""),
            "orcid": (author.get("orcid") or "").replace("https://orcid.org/", ""),
            "institution": (author.get("last_known_institutions") or [{}])[0].get("display_name", ""),
            "institution_country": (author.get("last_known_institutions") or [{}])[0].get("country_code", ""),
            "h_index": author.get("summary_stats", {}).get("h_index", 0),
            "i10_index": author.get("summary_stats", {}).get("i10_index", 0),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "concepts": [
                {"name": c.get("display_name", ""), "score": c.get("score", 0)}
                for c in (author.get("x_concepts") or [])[:10]
            ],
        }
    except Exception:
        return None


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"enriched": {}, "failed": [], "started_at": datetime.now().isoformat()}


def save_checkpoint(checkpoint: dict):
    checkpoint["updated_at"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Enrich AskChem authors from OpenAlex")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        cp = load_checkpoint()
        print(f"Enriched: {len(cp.get('enriched', {})):,}")
        print(f"Failed: {len(cp.get('failed', []))}")
        return

    print("Loading unique authors from index...", flush=True)
    authors = get_unique_authors()
    print(f"Found {len(authors):,} unique authors", flush=True)

    if args.max:
        authors = authors[:args.max]
        print(f"Processing top {len(authors):,} by citations", flush=True)

    checkpoint = load_checkpoint()
    enriched = checkpoint.get("enriched", {})
    failed = checkpoint.get("failed", [])

    remaining = [a for a in authors if a["name"] not in enriched and a["name"] not in failed]
    print(f"Already enriched: {len(enriched):,}, remaining: {len(remaining):,}\n", flush=True)

    success_count = 0
    fail_count = 0
    t0 = time.time()

    for i, author in enumerate(remaining):
        result = search_openalex_author(author["name"])

        if result:
            enriched[author["name"]] = {
                **result,
                "askchem_papers": author["paper_count"],
                "askchem_citations": author["total_citations"],
                "enriched_at": datetime.now().isoformat(),
            }
            success_count += 1
        else:
            failed.append(author["name"])
            fail_count += 1

        if (i + 1) % 100 == 0:
            checkpoint["enriched"] = enriched
            checkpoint["failed"] = failed
            save_checkpoint(checkpoint)
            elapsed = time.time() - t0
            rate = (success_count + fail_count) / elapsed
            print(f"  [{i+1:,}/{len(remaining):,}] "
                  f"OK: {success_count:,} | Fail: {fail_count:,} | "
                  f"{rate:.1f} authors/s",
                  flush=True)

        time.sleep(RATE_LIMIT_DELAY)

    checkpoint["enriched"] = enriched
    checkpoint["failed"] = failed
    save_checkpoint(checkpoint)

    elapsed = time.time() - t0
    print(f"\nDone: {success_count:,} enriched, {fail_count:,} failed in {elapsed:.0f}s")
    print(f"Total enriched: {len(enriched):,}")


if __name__ == "__main__":
    main()
