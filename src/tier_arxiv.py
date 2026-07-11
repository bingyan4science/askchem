"""
Tier harvested arXiv papers by citation count + chemistry relevance.

Pipeline:
  1. Read OAI-PMH harvest JSONL files
  2. Batch-enrich with Semantic Scholar for citation counts
  3. Score chemistry relevance (categories + keyword density)
  4. Assign tiers 1-4
  5. Write per-tier JSONL files
  6. Ingest all into chemtree.db with tier tag

Tier definitions:
  Tier 1: High citations (≥50) AND chemistry-relevant (score ≥5)
  Tier 2: Moderate citations (≥10) AND relevant (≥5), OR high citations (≥50)
  Tier 3: All remaining chemistry-relevant papers (score ≥4)
  Tier 4: Everything else (low relevance / no citations)

Usage:
    python src/tier_arxiv.py                    # Full pipeline
    python src/tier_arxiv.py --skip-s2          # Skip S2 enrichment (use cached)
    python src/tier_arxiv.py --dry-run          # Score and tier without ingesting
    python src/tier_arxiv.py --stats            # Show tier distribution
"""

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
HARVEST_DIR = DATA_DIR / "arxiv_harvest"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "citationCount,externalIds"
S2_BATCH_SIZE = 400
S2_DELAY = 1.1

CATEGORY_SCORES = {
    "chem-ph": 10,
    "mtrl-sci": 7,
    "comp-ph": 5,
    "atom-ph": 5,
    "soft": 4,
    "bio-ph": 4,
    "quant-ph": 2,
}

CHEM_KEYWORDS = re.compile(
    r"\b(?:molecul|chemical|chemistry|reaction|catalyst|catalysis|"
    r"DFT|density.functional|ab.initio|quantum.chemistry|"
    r"molecular.dynamics|force.field|potential.energy.surface|"
    r"spectroscop|vibrational|rotational|electronic.structure|"
    r"bond(?:ing)?|orbital|electron.density|wave.?function|"
    r"photochem|electrochemist|thermochem|"
    r"polymer|protein|enzyme|DNA|RNA|amino.acid|"
    r"nanoparticle|nanomaterial|crystal.struct|lattice.dynamic|"
    r"solvent|solvation|aqueous|ionic.liquid|"
    r"synthesis|compound|reagent|substrate|ligand|"
    r"adsorption|desorption|surface.reaction|"
    r"SMILES|InChI|IUPAC)\b",
    re.IGNORECASE,
)


def get_db_path():
    return Path(os.environ.get("CHEMTREE_DB", str(DB_PATH)))


def get_s2_headers():
    key = os.environ.get("S2_API_KEY", "")
    if key:
        return {"x-api-key": key}
    return {}


# ── Load papers ───────────────────────────────────────────────────────────────

def load_oai_papers():
    """Load all OAI-PMH harvested papers."""
    papers = []
    for jf in sorted(HARVEST_DIR.glob("oai_*.jsonl")):
        with open(jf) as f:
            for line in f:
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return papers


# ── S2 enrichment ────────────────────────────────────────────────────────────

def enrich_with_citations(papers, skip_cached=False):
    """Batch-enrich papers with Semantic Scholar citation counts."""
    cache_file = HARVEST_DIR / "s2_citations_cache.json"
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"  Citation cache: {len(cache):,} entries", flush=True)

    if skip_cached and cache:
        for p in papers:
            p["citation_count"] = cache.get(p["arxiv_id"], 0)
        print(f"  Using cached citations for {len(papers):,} papers", flush=True)
        return

    headers = get_s2_headers()
    to_lookup = [p for p in papers if p["arxiv_id"] not in cache]
    print(f"  Papers needing S2 lookup: {len(to_lookup):,}", flush=True)

    batches_done = 0
    for i in range(0, len(to_lookup), S2_BATCH_SIZE):
        batch = to_lookup[i:i + S2_BATCH_SIZE]
        ids = [f"ArXiv:{p['arxiv_id']}" for p in batch]

        for attempt in range(5):
            try:
                time.sleep(S2_DELAY)
                resp = requests.post(
                    S2_BATCH_URL,
                    headers=headers,
                    json={"ids": ids},
                    params={"fields": S2_FIELDS},
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"    429 rate limit, waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait = 5 * (attempt + 1)
                print(f"    S2 error ({attempt+1}/5): {e}, retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"    Exhausted retries at batch {i//S2_BATCH_SIZE}, filling with 0", flush=True)
            for p in batch:
                cache[p["arxiv_id"]] = 0
            continue

        results = resp.json()
        for p, r in zip(batch, results):
            cites = 0
            if r and isinstance(r, dict):
                cites = r.get("citationCount", 0) or 0
            cache[p["arxiv_id"]] = cites

        batches_done += 1
        if batches_done % 50 == 0:
            print(f"    S2 batch {batches_done}: {i + len(batch):,}/{len(to_lookup):,}", flush=True)
            with open(cache_file, "w") as f:
                json.dump(cache, f)

    # Final cache save
    with open(cache_file, "w") as f:
        json.dump(cache, f)
    print(f"  S2 enrichment done. Cache: {len(cache):,} entries", flush=True)

    for p in papers:
        p["citation_count"] = cache.get(p["arxiv_id"], 0)


# ── Chemistry relevance scoring ──────────────────────────────────────────────

def compute_relevance(paper: dict) -> int:
    """Score 0-13 based on primary category + keyword density."""
    cats = paper.get("categories", [])
    primary = cats[0] if cats else ""

    cat_score = 0
    for suffix, score in CATEGORY_SCORES.items():
        if primary.endswith(suffix):
            cat_score = score
            break
    if cat_score == 0:
        for c in cats[1:]:
            for suffix, score in CATEGORY_SCORES.items():
                if c.endswith(suffix):
                    cat_score = max(cat_score, score - 1)
                    break

    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    keyword_hits = len(CHEM_KEYWORDS.findall(text))
    keyword_score = min(keyword_hits, 3)

    return cat_score + keyword_score


def assign_tier(relevance: int, citations: int) -> int:
    if citations >= 50 and relevance >= 5:
        return 1
    if (citations >= 10 and relevance >= 5) or (citations >= 50 and relevance >= 2):
        return 2
    if relevance >= 4 or (citations >= 10 and relevance >= 2):
        return 3
    return 4


# ── Ingest ────────────────────────────────────────────────────────────────────

def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arXiv.{arxiv_id}"


def make_source_row(paper: dict):
    aid = paper["arxiv_id"]
    doi = paper.get("doi") or arxiv_doi(aid)
    authors_list = [{"name": a} for a in paper.get("authors", [])]

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
        "source_type": "arxiv",
        "tier": paper.get("tier", 4),
        "relevance_score": paper.get("relevance", 0),
        "externalIds": {"ArXiv": aid, "DOI": doi},
        "openAccessPdf": {"url": paper.get("pdf_url", "")},
        "citationCount": paper.get("citation_count", 0),
    }

    return (
        doi,
        paper.get("title", ""),
        json.dumps(authors_list),
        paper.get("year") or 0,
        f"arXiv:{aid}",
        paper.get("abstract", ""),
        paper.get("citation_count", 0),
        paper.get("pdf_url", ""),
        json.dumps(data_blob),
    )


def ingest_papers(papers):
    db_path = get_db_path()
    if not db_path.exists():
        print(f"ERROR: {db_path} not found.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()

    existing = set()
    for (doi,) in conn.execute("SELECT doi FROM sources").fetchall():
        if doi:
            existing.add(doi.lower().strip())

    inserted = 0
    skipped = 0
    batch = []

    for p in papers:
        doi = (p.get("doi") or arxiv_doi(p["arxiv_id"])).lower().strip()
        if doi in existing:
            skipped += 1
            continue
        existing.add(doi)
        batch.append(make_source_row(p))

        if len(batch) >= 500:
            c.executemany(
                "INSERT OR IGNORE INTO sources "
                "(doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) "
                "VALUES (?,?,?,?,?,?,?,?,?)", batch)
            conn.commit()
            inserted += len(batch)
            batch = []
            if inserted % 5000 == 0:
                print(f"    Inserted: {inserted:,}", flush=True)

    if batch:
        c.executemany(
            "INSERT OR IGNORE INTO sources "
            "(doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) "
            "VALUES (?,?,?,?,?,?,?,?,?)", batch)
        conn.commit()
        inserted += len(batch)

    conn.close()
    print(f"  Inserted: {inserted:,}  Skipped: {skipped:,}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(skip_s2=False, dry_run=False):
    print(f"{'='*60}", flush=True)
    print(f"arXiv Tiering — {datetime.now().isoformat()}", flush=True)
    print(f"{'='*60}", flush=True)

    print("\n1. Loading OAI-PMH harvested papers...", flush=True)
    papers = load_oai_papers()
    print(f"   {len(papers):,} papers loaded", flush=True)

    if not papers:
        print("No papers to process.")
        return

    print("\n2. Enriching with Semantic Scholar citations...", flush=True)
    enrich_with_citations(papers, skip_cached=skip_s2)

    print("\n3. Computing chemistry relevance scores...", flush=True)
    for p in papers:
        p["relevance"] = compute_relevance(p)

    print("\n4. Assigning tiers...", flush=True)
    for p in papers:
        p["tier"] = assign_tier(p["relevance"], p.get("citation_count", 0))

    tier_counts = Counter(p["tier"] for p in papers)
    tier_papers = {t: [] for t in range(1, 5)}
    for p in papers:
        tier_papers[p["tier"]].append(p)

    for t in range(1, 5):
        tp = tier_papers[t]
        if tp:
            avg_cite = sum(p.get("citation_count", 0) for p in tp) / len(tp)
            avg_rel = sum(p["relevance"] for p in tp) / len(tp)
        else:
            avg_cite = avg_rel = 0
        print(
            f"   Tier {t}: {tier_counts.get(t, 0):>7,} papers  "
            f"(avg citations: {avg_cite:.0f}, avg relevance: {avg_rel:.1f})",
            flush=True,
        )

    print("\n5. Writing per-tier JSONL files...", flush=True)
    for t in range(1, 5):
        path = HARVEST_DIR / f"tier_{t}.jsonl"
        with open(path, "w") as f:
            for p in sorted(tier_papers[t], key=lambda x: -(x.get("citation_count", 0))):
                f.write(json.dumps(p) + "\n")
        print(f"   {path.name}: {len(tier_papers[t]):,} papers", flush=True)

    if dry_run:
        print("\nDRY RUN — no database changes.")
        return

    print("\n6. Ingesting into chemtree.db...", flush=True)
    for t in range(1, 5):
        print(f"   Tier {t} ({len(tier_papers[t]):,} papers)...", flush=True)
        ingest_papers(tier_papers[t])

    print(f"\n{'='*60}", flush=True)
    print("TIERING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)


def show_stats():
    """Show tier distribution from existing tier files."""
    for t in range(1, 5):
        path = HARVEST_DIR / f"tier_{t}.jsonl"
        if not path.exists():
            print(f"  Tier {t}: (not found)")
            continue
        papers = []
        with open(path) as f:
            for line in f:
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if papers:
            avg_cite = sum(p.get("citation_count", 0) for p in papers) / len(papers)
            max_cite = max(p.get("citation_count", 0) for p in papers)
            avg_rel = sum(p.get("relevance", 0) for p in papers) / len(papers)
        else:
            avg_cite = max_cite = avg_rel = 0
        print(
            f"  Tier {t}: {len(papers):>7,} papers  "
            f"avg_cit={avg_cite:.0f}  max_cit={max_cite}  avg_rel={avg_rel:.1f}",
        )
        # Top 5 most cited
        top = sorted(papers, key=lambda x: -(x.get("citation_count", 0)))[:5]
        for p in top:
            print(f"    {p.get('citation_count',0):>6} cit | {p['arxiv_id']:>15} | {p['title'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier arXiv papers by citations + relevance")
    parser.add_argument("--skip-s2", action="store_true", help="Use cached S2 data")
    parser.add_argument("--dry-run", action="store_true", help="Don't ingest into DB")
    parser.add_argument("--stats", action="store_true", help="Show tier stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        run(skip_s2=args.skip_s2, dry_run=args.dry_run)
