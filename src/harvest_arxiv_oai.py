"""
Bulk harvest arXiv papers via OAI-PMH (no pagination limit).

Uses resumption tokens to iterate through entire category sets.
Much faster and more complete than the Atom API approach.

Usage:
    python src/harvest_arxiv_oai.py                          # Harvest all 7 categories
    python src/harvest_arxiv_oai.py --resume                 # Resume from checkpoint
    python src/harvest_arxiv_oai.py --status                 # Show progress
    python src/harvest_arxiv_oai.py --categories physics:physics:chem-ph
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

OAI_ENDPOINT = "http://export.arxiv.org/oai2"
REQUEST_DELAY = 3.0

# OAI-PMH set names (from ListSets)
OAI_SETS = {
    "physics.chem-ph":   "physics:physics:chem-ph",
    "cond-mat.mtrl-sci": "physics:cond-mat:mtrl-sci",
    "physics.comp-ph":   "physics:physics:comp-ph",
    "physics.atom-ph":   "physics:physics:atom-ph",
    "cond-mat.soft":     "physics:cond-mat:soft",
    "physics.bio-ph":    "physics:physics:bio-ph",
    "quant-ph":          "physics:quant-ph",
}

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"

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


def get_db_path():
    return Path(os.environ.get("CHEMTREE_DB", str(DB_PATH)))


def load_existing_keys():
    """Load DOIs and arXiv IDs already in the DB."""
    db_path = get_db_path()
    if not db_path.exists():
        print("  Warning: chemtree.db not found, no dedup", flush=True)
        return set(), set()
    conn = sqlite3.connect(str(db_path))
    dois = set()
    arxiv_ids = set()
    for doi, data_str in conn.execute("SELECT doi, data FROM sources").fetchall():
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
    conn.close()
    print(f"  {len(dois):,} DOIs, {len(arxiv_ids):,} arXiv IDs in DB", flush=True)
    return dois, arxiv_ids


def _text(el, tag, ns=None):
    child = el.find(f"{{{ns}}}{tag}" if ns else tag)
    return child.text.strip() if child is not None and child.text else ""


def parse_oai_response(xml_text):
    """Parse OAI-PMH ListRecords response."""
    root = ET.fromstring(xml_text)

    error = root.find(f"{{{OAI_NS}}}error")
    if error is not None:
        return [], None, error.get("code", ""), error.text or ""

    lr = root.find(f"{{{OAI_NS}}}ListRecords")
    if lr is None:
        return [], None, None, None

    records = []
    for rec in lr.findall(f"{{{OAI_NS}}}record"):
        header = rec.find(f"{{{OAI_NS}}}header")
        if header is not None and header.get("status") == "deleted":
            continue

        meta_wrap = rec.find(f"{{{OAI_NS}}}metadata")
        if meta_wrap is None:
            continue
        arxiv_el = meta_wrap.find(f"{{{ARXIV_NS}}}arXiv")
        if arxiv_el is None:
            continue

        arxiv_id = _text(arxiv_el, "id", ARXIV_NS)
        if not arxiv_id:
            continue

        title = " ".join(_text(arxiv_el, "title", ARXIV_NS).split())
        abstract = " ".join(_text(arxiv_el, "abstract", ARXIV_NS).split())
        created = _text(arxiv_el, "created", ARXIV_NS)
        year = int(created[:4]) if len(created) >= 4 else None
        doi = _text(arxiv_el, "doi", ARXIV_NS) or None
        categories = _text(arxiv_el, "categories", ARXIV_NS).split()

        authors = []
        authors_el = arxiv_el.find(f"{{{ARXIV_NS}}}authors")
        if authors_el is not None:
            for a in authors_el.findall(f"{{{ARXIV_NS}}}author"):
                fore = _text(a, "forenames", ARXIV_NS)
                key = _text(a, "keyname", ARXIV_NS)
                name = f"{fore} {key}".strip() if fore else key
                if name:
                    authors.append(name)

        pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

        records.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "published": created,
            "doi": doi,
            "pdf_url": pdf_url,
            "categories": categories,
        })

    token_el = lr.find(f"{{{OAI_NS}}}resumptionToken")
    token = None
    if token_el is not None and token_el.text:
        token = token_el.text.strip()
        if not token:
            token = None

    return records, token, None, None


def is_chemistry_relevant(title, abstract):
    return bool(CHEM_KEYWORDS.search(f"{title or ''} {abstract or ''}"))


def _cp_file(short_name):
    return HARVEST_DIR / f"oai_checkpoint_{short_name.replace('.', '_')}.json"


def harvest_category(short_name, oai_set, existing_dois, existing_arxiv_ids, global_seen, resume):
    out_path = HARVEST_DIR / f"oai_{short_name.replace('.', '_')}.jsonl"
    needs_filter = short_name in FILTER_CATEGORIES
    cp = _cp_file(short_name)

    token = None
    harvested = 0
    filtered = 0
    duplicated = 0
    pages = 0

    if resume and cp.exists():
        with open(cp) as f:
            state = json.load(f)
        token = state.get("token")
        harvested = state.get("harvested", 0)
        filtered = state.get("filtered", 0)
        duplicated = state.get("duplicated", 0)
        pages = state.get("pages", 0)
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        global_seen.add(json.loads(line)["arxiv_id"].lower())
                    except (json.JSONDecodeError, KeyError):
                        pass
        if token is None:
            print(f"  {short_name}: already complete ({harvested:,}), skipping", flush=True)
            return harvested, filtered, duplicated
        print(f"  Resuming {short_name} from page {pages}, {harvested:,} harvested", flush=True)

    print(f"\n{'─'*60}", flush=True)
    print(f"Category: {short_name}  (OAI set: {oai_set})", flush=True)
    if needs_filter:
        print("  Chemistry keyword filter: ON", flush=True)

    first_request = True

    while True:
        time.sleep(REQUEST_DELAY)

        if first_request and token is None:
            params = {"verb": "ListRecords", "set": oai_set, "metadataPrefix": "arXiv"}
        else:
            params = {"verb": "ListRecords", "resumptionToken": token}

        first_request = False

        for attempt in range(5):
            try:
                resp = requests.get(OAI_ENDPOINT, params=params, timeout=120)
                if resp.status_code == 503:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    print(f"    503, retry after {retry_after}s", flush=True)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait = 10 * (attempt + 1)
                print(f"    Error ({attempt+1}/5): {e}, retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print("    Exhausted retries, stopping category", flush=True)
            break

        records, token, err_code, err_msg = parse_oai_response(resp.text)

        if err_code:
            if err_code == "noRecordsMatch":
                print(f"  No records for {short_name}", flush=True)
            else:
                print(f"  OAI error: {err_code}: {err_msg}", flush=True)
            break

        pages += 1
        batch_new = 0

        with open(out_path, "a") as fout:
            for paper in records:
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

        # Checkpoint
        with open(cp, "w") as f:
            json.dump({"token": token, "harvested": harvested, "filtered": filtered,
                        "duplicated": duplicated, "pages": pages}, f)

        print(
            f"  page {pages:>4} | batch={len(records):>5} | "
            f"new={batch_new:>5} | total={harvested:>7,} | "
            f"dupes={duplicated:>6,}" + (" | filtered=" + str(filtered) if needs_filter else ""),
            flush=True,
        )

        if token is None:
            print(f"  Reached end of {short_name}", flush=True)
            break

    # Mark complete
    with open(cp, "w") as f:
        json.dump({"token": None, "harvested": harvested, "filtered": filtered,
                    "duplicated": duplicated, "pages": pages, "complete": True}, f)

    print(
        f"  DONE: {harvested:,} harvested, {filtered:,} filtered, {duplicated:,} dupes",
        flush=True,
    )
    return harvested, filtered, duplicated


def run_harvest(categories=None, resume=False):
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)

    if categories:
        cats = {c: OAI_SETS[c] for c in categories if c in OAI_SETS}
    else:
        cats = OAI_SETS

    print(f"{'='*60}", flush=True)
    print(f"arXiv OAI-PMH Harvest — {datetime.now().isoformat()}", flush=True)
    print(f"Categories: {len(cats)}", flush=True)
    print(f"{'='*60}", flush=True)

    print("\nLoading existing corpus for deduplication...", flush=True)
    existing_dois, existing_arxiv_ids = load_existing_keys()
    global_seen: set[str] = set()

    totals = {"harvested": 0, "filtered": 0, "duplicated": 0}
    cat_stats = {}

    for short_name, oai_set in cats.items():
        h, f, d = harvest_category(short_name, oai_set, existing_dois, existing_arxiv_ids,
                                   global_seen, resume)
        cat_stats[short_name] = {"harvested": h, "filtered": f, "duplicated": d}
        totals["harvested"] += h
        totals["filtered"] += f
        totals["duplicated"] += d

    summary = {"timestamp": datetime.now().isoformat(), "categories": cat_stats, "totals": totals}
    with open(HARVEST_DIR / "oai_harvest_summary.json", "w") as fout:
        json.dump(summary, fout, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("OAI-PMH HARVEST COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    for cat, s in cat_stats.items():
        print(
            f"  {cat:<25} {s['harvested']:>7,} harvested  "
            f"{s['filtered']:>6,} filtered  {s['duplicated']:>7,} dupes",
            flush=True,
        )
    print(
        f"  {'TOTAL':<25} {totals['harvested']:>7,} harvested  "
        f"{totals['filtered']:>6,} filtered  {totals['duplicated']:>7,} dupes",
        flush=True,
    )


def show_status():
    if not HARVEST_DIR.exists():
        print("No harvest data found.")
        return
    total = 0
    for short_name in OAI_SETS:
        out = HARVEST_DIR / f"oai_{short_name.replace('.', '_')}.jsonl"
        cp = _cp_file(short_name)
        if out.exists():
            count = sum(1 for _ in open(out))
            complete = ""
            if cp.exists():
                with open(cp) as f:
                    state = json.load(f)
                if state.get("complete"):
                    complete = " (complete)"
                else:
                    complete = f" (page {state.get('pages', '?')})"
            print(f"  {short_name:<25} {count:>7,} papers{complete}")
            total += count
        else:
            print(f"  {short_name:<25}       0 papers")
    print(f"  {'TOTAL':<25} {total:>7,} papers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk harvest arXiv via OAI-PMH")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument("--categories", nargs="+", help="Category short names")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        run_harvest(categories=args.categories, resume=args.resume)
