"""
Harvest papers from arXiv categories relevant to chemistry.

Uses the arXiv Atom API to fetch papers from 7 categories, with chemistry
keyword filtering for broad categories (quant-ph) and deduplication against
the existing chemtree.db corpus.

Usage:
    python src/harvest_arxiv.py                     # Full harvest, all categories
    python src/harvest_arxiv.py --resume            # Resume from checkpoint
    python src/harvest_arxiv.py --status            # Show progress
    python src/harvest_arxiv.py --categories physics.chem-ph cond-mat.mtrl-sci
    python src/harvest_arxiv.py --max-per-cat 1000  # Limit papers per category
"""

import argparse
import json
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
HARVEST_DIR = DATA_DIR / "arxiv_harvest"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"

ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 500
REQUEST_DELAY = 3.0  # arXiv requires >= 3 s between requests

CATEGORIES = [
    "physics.chem-ph",
    "cond-mat.mtrl-sci",
    "physics.comp-ph",
    "physics.atom-ph",
    "cond-mat.soft",
    "physics.bio-ph",
    "quant-ph",
]

DEFAULT_MAX_PER_CAT = {
    "physics.chem-ph": 10_000,
    "cond-mat.mtrl-sci": 10_000,
    "physics.comp-ph": 5_000,
    "physics.atom-ph": 5_000,
    "cond-mat.soft": 5_000,
    "physics.bio-ph": 5_000,
    "quant-ph": 5_000,
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

FILTER_CATEGORIES = {"quant-ph"}

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


def get_db_path():
    return Path(os.environ.get("CHEMTREE_DB", str(DB_PATH)))


def load_existing_dois():
    db_path = get_db_path()
    if not db_path.exists():
        print("  Warning: chemtree.db not found, skipping DB dedup", flush=True)
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT doi FROM sources").fetchall()
        dois = {r[0].lower().strip() for r in rows if r[0]}
        print(f"  Loaded {len(dois):,} existing DOIs from DB", flush=True)
        return dois
    finally:
        conn.close()


def load_existing_arxiv_ids():
    db_path = get_db_path()
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    arxiv_ids: set[str] = set()
    try:
        rows = conn.execute("SELECT doi, data FROM sources").fetchall()
        for doi, data_str in rows:
            if doi and "10.48550/arxiv" in doi.lower():
                m = re.search(r"10\.48550/arXiv\.(.+)", doi, re.IGNORECASE)
                if m:
                    arxiv_ids.add(m.group(1).lower())
            if data_str:
                try:
                    ext = json.loads(data_str).get("externalIds") or {}
                    aid = ext.get("ArXiv", "")
                    if aid:
                        arxiv_ids.add(aid.lower())
                except (json.JSONDecodeError, AttributeError):
                    pass
        print(f"  Loaded {len(arxiv_ids):,} existing arXiv IDs from DB", flush=True)
        return arxiv_ids
    finally:
        conn.close()


def _strip_version(raw_id: str) -> str:
    return re.sub(r"v\d+$", "", raw_id)


def parse_arxiv_id(url_or_id: str) -> str:
    if "/" in url_or_id:
        raw = url_or_id.rstrip("/").split("/")[-1]
    else:
        raw = url_or_id
    return _strip_version(raw)


def is_chemistry_relevant(title: str, abstract: str) -> bool:
    return bool(CHEM_KEYWORDS.search(f"{title or ''} {abstract or ''}"))


def parse_entries(xml_text: str):
    root = ET.fromstring(xml_text)

    total_el = root.find("opensearch:totalResults", ATOM_NS)
    total = int(total_el.text) if total_el is not None and total_el.text else 0

    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        id_el = entry.find("atom:id", ATOM_NS)
        if id_el is None or not id_el.text:
            continue
        arxiv_url = id_el.text.strip()
        arxiv_id = parse_arxiv_id(arxiv_url)

        title_el = entry.find("atom:title", ATOM_NS)
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""

        summary_el = entry.find("atom:summary", ATOM_NS)
        abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""

        pub_el = entry.find("atom:published", ATOM_NS)
        published = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        year = int(published[:4]) if len(published) >= 4 else None

        authors = []
        for author in entry.findall("atom:author", ATOM_NS):
            name_el = author.find("atom:name", ATOM_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        doi_el = entry.find("arxiv:doi", ATOM_NS)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url:
            pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

        categories = []
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        if primary is not None and primary.get("term"):
            categories.append(primary.get("term"))
        for cat in entry.findall("atom:category", ATOM_NS):
            t = cat.get("term")
            if t and t not in categories:
                categories.append(t)

        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "published": published,
            "doi": doi,
            "pdf_url": pdf_url,
            "categories": categories,
            "arxiv_url": arxiv_url,
        })

    return entries, total


def _cp_path(category: str) -> Path:
    return HARVEST_DIR / f"checkpoint_{category.replace('.', '_')}.json"


def _out_path(category: str) -> Path:
    return HARVEST_DIR / f"harvest_{category.replace('.', '_')}.jsonl"


def load_checkpoint(category):
    p = _cp_path(category)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"start": 0, "harvested": 0, "filtered": 0, "duplicated": 0}


def save_checkpoint(category, state):
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cp_path(category), "w") as f:
        json.dump(state, f, indent=2)


def harvest_category(category, max_papers, existing_dois, existing_arxiv_ids, global_seen):
    out = _out_path(category)
    needs_filter = category in FILTER_CATEGORIES

    state = load_checkpoint(category)
    start = state["start"]
    harvested = state["harvested"]
    filtered = state["filtered"]
    duplicated = state["duplicated"]

    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    global_seen.add(json.loads(line)["arxiv_id"].lower())
                except (json.JSONDecodeError, KeyError):
                    pass

    print(f"\n{'─'*60}", flush=True)
    print(f"Category: {category}  (target {max_papers:,})", flush=True)
    if needs_filter:
        print("  Chemistry keyword filter: ON", flush=True)
    if start > 0:
        print(f"  Resuming from start={start}, harvested={harvested}", flush=True)

    consecutive_empty = 0

    while harvested < max_papers:
        time.sleep(REQUEST_DELAY)

        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": BATCH_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        for attempt in range(5):
            try:
                resp = requests.get(ARXIV_API, params=params, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait = 10 * (attempt + 1)
                print(f"    Request error ({attempt+1}/5): {e}, retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"    Exhausted retries at start={start}, stopping", flush=True)
            break

        entries, total_available = parse_entries(resp.text)

        if not entries:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("  No more results (3 empty pages)", flush=True)
                break
            start += BATCH_SIZE
            save_checkpoint(category, {"start": start, "harvested": harvested,
                                       "filtered": filtered, "duplicated": duplicated})
            continue

        consecutive_empty = 0
        batch_new = 0

        with open(out, "a") as fout:
            for paper in entries:
                aid = paper["arxiv_id"].lower()
                doi_low = (paper["doi"] or "").lower().strip()
                arxiv_doi = f"10.48550/arxiv.{aid}"

                if doi_low and doi_low in existing_dois:
                    duplicated += 1
                    continue
                if arxiv_doi in existing_dois:
                    duplicated += 1
                    continue
                if aid in existing_arxiv_ids:
                    duplicated += 1
                    continue
                if aid in global_seen:
                    duplicated += 1
                    continue

                if needs_filter and not is_chemistry_relevant(paper["title"], paper["abstract"]):
                    filtered += 1
                    continue

                global_seen.add(aid)
                fout.write(json.dumps(paper) + "\n")
                batch_new += 1
                harvested += 1

                if harvested >= max_papers:
                    break

        start += len(entries)
        save_checkpoint(category, {"start": start, "harvested": harvested,
                                   "filtered": filtered, "duplicated": duplicated})

        print(
            f"  start={start - len(entries):>6} | batch={len(entries):>4} | "
            f"new={batch_new:>4} | total={harvested:>6}/{max_papers} | "
            f"avail={total_available:,}",
            flush=True,
        )

        if start >= total_available:
            print(f"  Reached end ({total_available:,} total)", flush=True)
            break

    print(
        f"  DONE: {harvested:,} harvested, "
        f"{filtered:,} filtered, {duplicated:,} dupes",
        flush=True,
    )
    return harvested, filtered, duplicated


def run_harvest(categories=None, max_per_cat=None, resume=False):
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    cats = categories or CATEGORIES

    print(f"{'='*60}", flush=True)
    print(f"arXiv Harvest — {datetime.now().isoformat()}", flush=True)
    print(f"Categories: {len(cats)}", flush=True)
    print(f"{'='*60}", flush=True)

    print("\nLoading existing corpus for deduplication...", flush=True)
    existing_dois = load_existing_dois()
    existing_arxiv_ids = load_existing_arxiv_ids()
    global_seen: set[str] = set()

    if not resume:
        for cat in cats:
            for p in (_cp_path(cat), _out_path(cat)):
                if p.exists():
                    p.unlink()

    totals = {"harvested": 0, "filtered": 0, "duplicated": 0}
    cat_stats = {}

    for cat in cats:
        mx = max_per_cat or DEFAULT_MAX_PER_CAT.get(cat, 5000)
        h, f, d = harvest_category(cat, mx, existing_dois, existing_arxiv_ids, global_seen)
        cat_stats[cat] = {"harvested": h, "filtered": f, "duplicated": d}
        totals["harvested"] += h
        totals["filtered"] += f
        totals["duplicated"] += d

    summary = {"timestamp": datetime.now().isoformat(), "categories": cat_stats, "totals": totals}
    with open(HARVEST_DIR / "harvest_summary.json", "w") as fout:
        json.dump(summary, fout, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("HARVEST COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    for cat, s in cat_stats.items():
        print(
            f"  {cat:<25} {s['harvested']:>6,} harvested  "
            f"{s['filtered']:>5,} filtered  {s['duplicated']:>5,} dupes",
            flush=True,
        )
    print(
        f"  {'TOTAL':<25} {totals['harvested']:>6,} harvested  "
        f"{totals['filtered']:>5,} filtered  {totals['duplicated']:>5,} dupes",
        flush=True,
    )


def show_status():
    if not HARVEST_DIR.exists():
        print("No harvest data found.")
        return
    total = 0
    for cat in CATEGORIES:
        out = _out_path(cat)
        if out.exists():
            count = sum(1 for _ in open(out))
            print(f"  {cat:<25} {count:>6,} papers")
            total += count
        else:
            print(f"  {cat:<25}      0 papers")
    print(f"  {'TOTAL':<25} {total:>6,} papers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest arXiv papers for AskChem")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument("--categories", nargs="+", help="Specific categories to harvest")
    parser.add_argument("--max-per-cat", type=int, help="Max papers per category")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        run_harvest(
            categories=args.categories,
            max_per_cat=args.max_per_cat,
            resume=args.resume,
        )
