"""
Scan Semantic Scholar for open-access PDF availability across the corpus.

Queries the S2 batch endpoint for all DOIs in chemtree.db that lack deep_v1
claims, records which have downloadable OA PDFs, and writes a summary.

Output: data/oa_scan.json
"""

import json
import os
import sqlite3
import time
import sys
from pathlib import Path
from collections import Counter, defaultdict

import requests

DB_PATH = Path(__file__).parent.parent / "chemtree.db"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "oa_scan.json"
BATCH_SIZE = 500
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "externalIds,openAccessPdf,citationCount,year,title,fieldsOfStudy"


def load_papers_from_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    deep_dois = set()
    for row in conn.execute(
        "SELECT DISTINCT source_doi FROM claims WHERE extraction_version = 'deep_v1'"
    ):
        deep_dois.add(row["source_doi"].lower())

    papers = []
    for row in conn.execute(
        "SELECT doi, title, year, citation_count, open_access_url FROM sources"
    ):
        doi = row["doi"]
        if doi.lower() in deep_dois:
            continue
        papers.append({
            "doi": doi,
            "title": row["title"] or "",
            "year": row["year"] or 0,
            "citation_count": row["citation_count"] or 0,
            "existing_oa_url": row["open_access_url"] or "",
        })
    conn.close()
    return papers, deep_dois


def batch_query_s2(dois: list[str], api_key: str = "") -> dict:
    """Query S2 batch endpoint. Returns {doi_lower: paper_data}."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    ids = [f"DOI:{d}" for d in dois]
    try:
        resp = requests.post(
            S2_BATCH_URL,
            params={"fields": FIELDS},
            json={"ids": ids},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 429:
            return None  # rate limited
        resp.raise_for_status()
        results = resp.json()
        out = {}
        for paper, doi in zip(results, dois):
            if paper is not None:
                out[doi.lower()] = paper
        return out
    except Exception as e:
        print(f"  Batch error: {e}", flush=True)
        return {}


def main():
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        print("Using S2 API key", flush=True)
    else:
        print("No S2_API_KEY -- using unauthenticated (slower rate limits)", flush=True)

    print("Loading papers from DB...", flush=True)
    papers, deep_dois = load_papers_from_db()
    papers.sort(key=lambda p: p["citation_count"], reverse=True)
    print(f"  {len(papers)} non-deep papers to scan", flush=True)
    print(f"  {len(deep_dois)} already-deep papers excluded", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_PATH.parent / "oa_scan_checkpoint.json"
    scanned = {}
    start_idx = 0
    if checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text())
        scanned = cp.get("scanned", {})
        start_idx = cp.get("next_idx", 0)
        print(f"  Resuming from checkpoint: {len(scanned)} already scanned, idx={start_idx}", flush=True)

    all_dois = [p["doi"] for p in papers]
    total_batches = (len(all_dois) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(start_idx // BATCH_SIZE, total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_dois))
        batch_dois = all_dois[start:end]

        unscanned = [d for d in batch_dois if d.lower() not in scanned]
        if not unscanned:
            continue

        result = batch_query_s2(unscanned, api_key)
        if result is None:
            print(f"  Rate limited at batch {batch_num+1}/{total_batches}, waiting 10s...", flush=True)
            time.sleep(10)
            result = batch_query_s2(unscanned, api_key)
            if result is None:
                print(f"  Still rate limited, waiting 30s...", flush=True)
                time.sleep(30)
                result = batch_query_s2(unscanned, api_key)
                if result is None:
                    print(f"  Giving up on batch {batch_num+1}, saving checkpoint", flush=True)
                    break

        for doi in unscanned:
            paper_data = (result or {}).get(doi.lower())
            if paper_data:
                oa = paper_data.get("openAccessPdf") or {}
                scanned[doi.lower()] = {
                    "url": oa.get("url", ""),
                    "status": oa.get("status", ""),
                    "year": paper_data.get("year"),
                    "cites": paper_data.get("citationCount"),
                }
            else:
                scanned[doi.lower()] = {"url": "", "status": "", "year": None, "cites": None}

        if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
            checkpoint_path.write_text(json.dumps({
                "scanned": scanned,
                "next_idx": end,
            }))
            oa_count = sum(1 for v in scanned.values() if v.get("url"))
            print(f"  Batch {batch_num+1}/{total_batches}: "
                  f"scanned {len(scanned)}, OA found: {oa_count}", flush=True)

        delay = 0.3 if api_key else 1.1
        time.sleep(delay)

    # Build summary
    oa_papers = []
    paper_lookup = {p["doi"].lower(): p for p in papers}
    for doi_lower, info in scanned.items():
        if not info.get("url"):
            continue
        db_paper = paper_lookup.get(doi_lower, {})
        oa_papers.append({
            "doi": db_paper.get("doi", doi_lower),
            "title": db_paper.get("title", ""),
            "year": info.get("year") or db_paper.get("year", 0),
            "citation_count": info.get("cites") or db_paper.get("citation_count", 0),
            "pdf_url": info["url"],
            "oa_status": info.get("status", ""),
        })

    oa_papers.sort(key=lambda p: p["citation_count"], reverse=True)

    # Distribution stats
    year_dist = Counter()
    cite_buckets = Counter()
    for p in oa_papers:
        year_dist[p["year"]] += 1
        c = p["citation_count"]
        if c >= 1000:
            cite_buckets["1000+"] += 1
        elif c >= 500:
            cite_buckets["500-999"] += 1
        elif c >= 200:
            cite_buckets["200-499"] += 1
        elif c >= 100:
            cite_buckets["100-199"] += 1
        elif c >= 50:
            cite_buckets["50-99"] += 1
        elif c >= 20:
            cite_buckets["20-49"] += 1
        elif c >= 10:
            cite_buckets["10-19"] += 1
        else:
            cite_buckets["0-9"] += 1

    output = {
        "scan_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_scanned": len(scanned),
        "total_with_oa": len(oa_papers),
        "total_without_oa": len(scanned) - len(oa_papers),
        "already_deep": len(deep_dois),
        "citation_distribution": dict(sorted(cite_buckets.items())),
        "year_distribution": dict(sorted(year_dist.items())),
        "papers": oa_papers,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nScan complete!", flush=True)
    print(f"  Total scanned: {len(scanned)}", flush=True)
    print(f"  With OA PDF: {len(oa_papers)} ({100*len(oa_papers)/max(1,len(scanned)):.1f}%)", flush=True)
    print(f"  Without OA: {len(scanned) - len(oa_papers)}", flush=True)
    print(f"\nCitation distribution of OA papers:", flush=True)
    for bucket in ["1000+", "500-999", "200-499", "100-199", "50-99", "20-49", "10-19", "0-9"]:
        print(f"  {bucket}: {cite_buckets.get(bucket, 0)}", flush=True)
    print(f"\nYear distribution (top years):", flush=True)
    for year, count in sorted(year_dist.items(), key=lambda x: -x[1])[:15]:
        print(f"  {year}: {count}", flush=True)
    print(f"\nResults written to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
