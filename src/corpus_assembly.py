"""
Corpus Assembly for AskChem — 100K paper target.

Uses Semantic Scholar bulk search with personal API key (1 req/s limit).
Broad query coverage across all chemistry subfields, with incremental
checkpoint saves so the run can be resumed if interrupted.

Usage:
    python src/corpus_assembly.py                    # Full run
    python src/corpus_assembly.py --resume           # Resume from checkpoint
    python src/corpus_assembly.py --status            # Show progress
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"
CHECKPOINT_DIR = DATA_DIR / "corpus_checkpoints"

S2_BULK_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = (
    "paperId,title,abstract,year,citationCount,venue,"
    "openAccessPdf,authors,fieldsOfStudy,externalIds"
)

TARGET_PAPERS = 110_000  # overshoot slightly to account for duplicates/no-abstract
MIN_DELAY = 1.1  # seconds between requests (S2 limit: 1 req/s)

# ---------------------------------------------------------------------------
# Queries designed for broad chemistry coverage.
# Each tuple: (subfield, query, min_citations, year_range, max_pages)
#   max_pages: how many 1000-result pages to fetch (None = all available)
# ---------------------------------------------------------------------------
QUERIES = [
    # ── Organic Chemistry ──────────────────────────────────────────────
    ("organic_chemistry", "organic synthesis methodology", 5, None, 10),
    ("organic_chemistry", "total synthesis natural product", 5, None, 10),
    ("organic_chemistry", "palladium catalyzed cross coupling", 3, None, 10),
    ("organic_chemistry", "asymmetric catalysis enantioselective synthesis", 3, None, 10),
    ("organic_chemistry", "C-H activation functionalization", 3, None, 10),
    ("organic_chemistry", "radical chemistry organic reaction", 3, None, 5),
    ("organic_chemistry", "organocatalysis metal-free", 3, None, 5),
    ("organic_chemistry", "click chemistry bioorthogonal", 3, None, 5),
    ("organic_chemistry", "flow chemistry continuous synthesis", 3, None, 5),
    ("organic_chemistry", "photochemistry organic synthesis visible light", 3, None, 5),

    # ── Inorganic & Materials Chemistry ────────────────────────────────
    ("inorganic_materials", "metal-organic framework MOF synthesis", 3, None, 10),
    ("inorganic_materials", "coordination chemistry transition metal complex", 3, None, 10),
    ("inorganic_materials", "perovskite materials synthesis properties", 3, None, 10),
    ("inorganic_materials", "nanoparticle synthesis characterization", 3, None, 10),
    ("inorganic_materials", "two-dimensional materials chemistry", 3, None, 5),
    ("inorganic_materials", "covalent organic framework COF", 3, None, 5),
    ("inorganic_materials", "zeolite synthesis catalysis", 3, None, 5),
    ("inorganic_materials", "quantum dot semiconductor nanocrystal", 3, None, 5),
    ("inorganic_materials", "supramolecular chemistry self-assembly", 3, None, 5),

    # ── Physical Chemistry ─────────────────────────────────────────────
    ("physical_chemistry", "reaction kinetics mechanism rate", 3, None, 10),
    ("physical_chemistry", "spectroscopy molecular characterization", 3, None, 10),
    ("physical_chemistry", "surface chemistry adsorption", 3, None, 5),
    ("physical_chemistry", "thermodynamics chemical equilibrium", 3, None, 5),
    ("physical_chemistry", "photophysics excited state dynamics", 3, None, 5),
    ("physical_chemistry", "single molecule spectroscopy imaging", 3, None, 5),
    ("physical_chemistry", "ultrafast spectroscopy femtosecond", 3, None, 5),

    # ── Analytical Chemistry ───────────────────────────────────────────
    ("analytical_chemistry", "mass spectrometry proteomics metabolomics", 3, None, 10),
    ("analytical_chemistry", "NMR spectroscopy structure elucidation", 3, None, 5),
    ("analytical_chemistry", "chromatography separation analytical", 3, None, 5),
    ("analytical_chemistry", "electrochemical sensor biosensor", 3, None, 10),
    ("analytical_chemistry", "fluorescence imaging probe detection", 3, None, 5),
    ("analytical_chemistry", "Raman spectroscopy SERS", 3, None, 5),
    ("analytical_chemistry", "X-ray crystallography structure determination", 3, None, 5),

    # ── Biochemistry & Chemical Biology ────────────────────────────────
    ("biochemistry", "enzyme catalysis mechanism kinetics", 3, None, 10),
    ("biochemistry", "protein structure function folding", 3, None, 10),
    ("biochemistry", "chemical biology molecular probe", 3, None, 5),
    ("biochemistry", "drug discovery medicinal chemistry", 3, None, 10),
    ("biochemistry", "CRISPR chemical biology genome editing", 3, None, 5),
    ("biochemistry", "metabolic pathway engineering synthetic biology", 3, None, 5),
    ("biochemistry", "nucleic acid chemistry aptamer", 3, None, 5),
    ("biochemistry", "lipid membrane chemistry", 3, None, 5),

    # ── Catalysis ──────────────────────────────────────────────────────
    ("catalysis", "heterogeneous catalysis surface reaction", 3, None, 10),
    ("catalysis", "homogeneous catalysis organometallic", 3, None, 10),
    ("catalysis", "photocatalysis water splitting hydrogen", 3, None, 10),
    ("catalysis", "electrocatalysis oxygen reduction CO2 reduction", 3, None, 10),
    ("catalysis", "biocatalysis enzyme engineering", 3, None, 5),
    ("catalysis", "single atom catalyst", 3, None, 5),
    ("catalysis", "catalyst design high throughput screening", 3, None, 5),

    # ── Computational Chemistry ────────────────────────────────────────
    ("computational", "density functional theory DFT calculation", 3, None, 10),
    ("computational", "molecular dynamics simulation chemistry", 3, None, 10),
    ("computational", "machine learning molecular property prediction", 3, None, 10),
    ("computational", "ab initio quantum chemistry", 3, None, 5),
    ("computational", "force field molecular mechanics", 3, None, 5),
    ("computational", "reaction path transition state", 3, None, 5),
    ("computational", "neural network potential molecular", 3, None, 5),

    # ── Energy Chemistry ───────────────────────────────────────────────
    ("energy", "lithium ion battery electrode electrolyte", 3, None, 10),
    ("energy", "solar cell photovoltaic perovskite organic", 3, None, 10),
    ("energy", "fuel cell proton exchange membrane", 3, None, 5),
    ("energy", "hydrogen storage materials", 3, None, 5),
    ("energy", "supercapacitor energy storage", 3, None, 5),
    ("energy", "CO2 capture utilization conversion", 3, None, 5),

    # ── Environmental Chemistry ────────────────────────────────────────
    ("environmental", "water treatment pollutant degradation", 3, None, 5),
    ("environmental", "atmospheric chemistry aerosol", 3, None, 5),
    ("environmental", "green chemistry sustainable synthesis", 3, None, 5),
    ("environmental", "microplastics environmental chemistry", 3, None, 5),
    ("environmental", "soil chemistry remediation", 3, None, 5),

    # ── Polymer Chemistry ──────────────────────────────────────────────
    ("polymer", "polymer synthesis controlled polymerization", 3, None, 10),
    ("polymer", "block copolymer self-assembly", 3, None, 5),
    ("polymer", "biodegradable polymer sustainable materials", 3, None, 5),
    ("polymer", "conjugated polymer organic electronics", 3, None, 5),
    ("polymer", "hydrogel smart responsive material", 3, None, 5),

    # ── Electrochemistry ───────────────────────────────────────────────
    ("electrochemistry", "electrochemistry electrode interface", 3, None, 5),
    ("electrochemistry", "electrodeposition electroplating", 3, None, 5),
    ("electrochemistry", "electrochemical synthesis organic", 3, None, 5),

    # ── Food & Agricultural Chemistry ──────────────────────────────────
    ("food_agri", "food chemistry antioxidant bioactive", 3, None, 5),
    ("food_agri", "pesticide agrochemical crop protection", 3, None, 5),

    # ── Nuclear & Radiochemistry ───────────────────────────────────────
    ("nuclear", "radiochemistry nuclear medicine isotope", 3, None, 5),

    # ── Geochemistry ───────────────────────────────────────────────────
    ("geochemistry", "geochemistry isotope mineral", 3, None, 5),
]


def get_s2_headers():
    key = os.environ.get("S2_API_KEY", "")
    if not key:
        raise EnvironmentError("S2_API_KEY not set. Add it to ~/.bashrc")
    return {"x-api-key": key}


def load_checkpoint():
    """Load collection state from checkpoint."""
    cp_file = CHECKPOINT_DIR / "state.json"
    if cp_file.exists():
        with open(cp_file) as f:
            return json.load(f)
    return {
        "seen_paper_ids": [],
        "completed_queries": [],
        "papers_collected": 0,
        "papers_with_abstract": 0,
        "started_at": datetime.now().isoformat(),
    }


def save_checkpoint(state):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "state.json", "w") as f:
        json.dump(state, f)


def load_seen_ids():
    """Load the set of already-seen paper IDs from the shard files."""
    seen = set()
    if not CHECKPOINT_DIR.exists():
        return seen
    for f in sorted(CHECKPOINT_DIR.glob("shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    p = json.loads(line)
                    pid = p.get("paperId")
                    if pid:
                        seen.add(pid)
                except json.JSONDecodeError:
                    pass
    return seen


def count_collected():
    """Count papers already collected across all shards."""
    total = 0
    with_abstract = 0
    if not CHECKPOINT_DIR.exists():
        return 0, 0
    for f in sorted(CHECKPOINT_DIR.glob("shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    p = json.loads(line)
                    total += 1
                    if p.get("abstract"):
                        with_abstract += 1
                except json.JSONDecodeError:
                    pass
    return total, with_abstract


def get_current_shard_path():
    """Get the path for the current shard file (rotate every 10K papers)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CHECKPOINT_DIR.glob("shard_*.jsonl"))
    if not existing:
        return CHECKPOINT_DIR / "shard_000.jsonl"
    last = existing[-1]
    line_count = sum(1 for _ in open(last))
    if line_count >= 10_000:
        idx = int(last.stem.split("_")[1]) + 1
        return CHECKPOINT_DIR / f"shard_{idx:03d}.jsonl"
    return last


def bulk_search_paginated(query, headers, min_citations=3, year=None,
                          max_pages=10, seen_ids=None):
    """
    Paginate through S2 bulk search results.
    Yields individual paper dicts. Respects 1 req/s rate limit.
    """
    if seen_ids is None:
        seen_ids = set()

    params = {
        "query": query,
        "fields": S2_FIELDS,
    }
    if min_citations:
        params["minCitationCount"] = min_citations
    if year:
        params["year"] = year

    token = None
    pages_fetched = 0

    while pages_fetched < (max_pages or 999):
        if token:
            params["token"] = token

        for attempt in range(8):
            try:
                time.sleep(MIN_DELAY)
                resp = requests.get(S2_BULK_SEARCH, params=params,
                                    headers=headers, timeout=60)
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"      429 rate limited, waiting {wait}s "
                          f"(attempt {attempt+1}/8)...", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                wait = 10 * (attempt + 1)
                print(f"      Request error (attempt {attempt+1}/8): {e}, "
                      f"retrying in {wait}s...", flush=True)
                time.sleep(wait)
        else:
            print("      Exhausted retries, skipping page.", flush=True)
            break

        batch = data.get("data", [])
        token = data.get("token")
        pages_fetched += 1

        new_in_batch = 0
        for p in batch:
            pid = p.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                new_in_batch += 1
                yield p

        print(f"      Page {pages_fetched}: {len(batch)} results, "
              f"{new_in_batch} new (token: {'yes' if token else 'done'})",
              flush=True)

        if not token:
            break


def run_assembly(resume=False):
    """Run the full 100K paper corpus assembly."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = get_s2_headers()

    state = load_checkpoint() if resume else {
        "seen_paper_ids": [],
        "completed_queries": [],
        "papers_collected": 0,
        "papers_with_abstract": 0,
        "started_at": datetime.now().isoformat(),
    }

    if resume:
        seen_ids = load_seen_ids()
        completed = set(state.get("completed_queries", []))
        total, with_abs = count_collected()
        print(f"Resuming: {total} papers collected ({with_abs} with abstract), "
              f"{len(completed)} queries done", flush=True)
    else:
        seen_ids = set()
        completed = set()
        total, with_abs = 0, 0

    print(f"\n{'='*60}", flush=True)
    print(f"AskChem Corpus Assembly — Target: {TARGET_PAPERS:,} papers", flush=True)
    print(f"Queries: {len(QUERIES)} | Rate limit: {MIN_DELAY}s between requests", flush=True)
    print(f"{'='*60}\n", flush=True)

    query_log = []

    for qi, (subfield, query, min_cit, year, max_pages) in enumerate(QUERIES):
        query_key = f"{subfield}::{query}"
        if query_key in completed:
            print(f"[{qi+1}/{len(QUERIES)}] SKIP (done): [{subfield}] {query}", flush=True)
            continue

        if with_abs >= TARGET_PAPERS:
            print(f"\nReached target of {TARGET_PAPERS:,} papers with abstracts. "
                  f"Stopping.", flush=True)
            break

        print(f"\n[{qi+1}/{len(QUERIES)}] [{subfield}] \"{query}\"", flush=True)
        print(f"    min_citations={min_cit}, year={year or 'all'}, "
              f"max_pages={max_pages}", flush=True)

        shard_path = get_current_shard_path()
        batch_new = 0
        batch_with_abs = 0

        for paper in bulk_search_paginated(
            query, headers, min_citations=min_cit, year=year,
            max_pages=max_pages, seen_ids=seen_ids,
        ):
            paper["_subfield"] = subfield
            paper["_query"] = query

            with open(shard_path, "a") as f:
                f.write(json.dumps(paper) + "\n")

            total += 1
            batch_new += 1
            if paper.get("abstract"):
                with_abs += 1
                batch_with_abs += 1

            if total % 5000 == 0:
                shard_path = get_current_shard_path()
                print(f"    ... {total:,} total, {with_abs:,} with abstract",
                      flush=True)

        completed.add(query_key)
        state["completed_queries"] = list(completed)
        state["papers_collected"] = total
        state["papers_with_abstract"] = with_abs
        save_checkpoint(state)

        print(f"    -> {batch_new} new papers ({batch_with_abs} with abstract) | "
              f"Running total: {total:,} ({with_abs:,} with abstract)", flush=True)

        query_log.append({
            "subfield": subfield,
            "query": query,
            "new_papers": batch_new,
            "new_with_abstract": batch_with_abs,
            "running_total": total,
            "running_with_abstract": with_abs,
            "timestamp": datetime.now().isoformat(),
        })

    # ── Finalize: merge shards into all_papers.json ──────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Merging shards into final metadata...", flush=True)

    all_papers = []
    seen_final = set()
    for f in sorted(CHECKPOINT_DIR.glob("shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    p = json.loads(line)
                    pid = p.get("paperId")
                    if pid and pid not in seen_final:
                        seen_final.add(pid)
                        all_papers.append(p)
                except json.JSONDecodeError:
                    pass

    with_abstract = [p for p in all_papers if p.get("abstract")]
    with_doi = [p for p in all_papers
                if (p.get("externalIds") or {}).get("DOI")]
    processable = [p for p in all_papers
                   if p.get("abstract") and (p.get("externalIds") or {}).get("DOI")]

    citations = [(p.get("citationCount") or 0) for p in all_papers]
    years = [p["year"] for p in all_papers if p.get("year")]

    subfield_dist = Counter(p.get("_subfield", "unknown") for p in all_papers)
    year_dist = Counter(str(y) for y in years)

    stats = {
        "total_papers": len(all_papers),
        "with_abstract": len(with_abstract),
        "with_doi": len(with_doi),
        "processable": len(processable),
        "timestamp": datetime.now().isoformat(),
        "citation_stats": {
            "mean": sum(citations) / max(len(citations), 1),
            "median": sorted(citations)[len(citations) // 2] if citations else 0,
            "max": max(citations) if citations else 0,
        },
        "year_distribution": dict(sorted(year_dist.items())),
        "subfield_distribution": dict(sorted(subfield_dist.items(), key=lambda x: -x[1])),
    }

    with open(METADATA_DIR / "all_papers.json", "w") as f:
        json.dump(all_papers, f)
    with open(METADATA_DIR / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(CHECKPOINT_DIR / "query_log.json", "w") as f:
        json.dump(query_log, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("CORPUS ASSEMBLY COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total unique papers:    {len(all_papers):>8,}", flush=True)
    print(f"  With abstract:          {len(with_abstract):>8,}", flush=True)
    print(f"  With DOI:               {len(with_doi):>8,}", flush=True)
    print(f"  Processable (abs+DOI):  {len(processable):>8,}", flush=True)
    if citations:
        print(f"  Citations — mean: {stats['citation_stats']['mean']:.1f}, "
              f"median: {stats['citation_stats']['median']}, "
              f"max: {stats['citation_stats']['max']}", flush=True)
    if years:
        print(f"  Year range: {min(years)} – {max(years)}", flush=True)
    print(f"\n  Subfield distribution:", flush=True)
    for sf, count in sorted(subfield_dist.items(), key=lambda x: -x[1]):
        print(f"    {sf}: {count:,}", flush=True)

    return stats


def show_status():
    """Show current collection progress."""
    total, with_abs = count_collected()
    state = load_checkpoint()
    completed = len(state.get("completed_queries", []))
    print(f"Papers collected: {total:,} ({with_abs:,} with abstract)")
    print(f"Queries completed: {completed}/{len(QUERIES)}")
    print(f"Target: {TARGET_PAPERS:,}")
    pct = with_abs / TARGET_PAPERS * 100 if TARGET_PAPERS else 0
    print(f"Progress: {pct:.1f}%")
    if state.get("started_at"):
        print(f"Started: {state['started_at']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AskChem corpus assembly")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true",
                        help="Show collection progress")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        print(f"AskChem Corpus Assembly — {datetime.now().isoformat()}", flush=True)
        print(f"Target: {TARGET_PAPERS:,} papers across {len(QUERIES)} queries",
              flush=True)
        run_assembly(resume=args.resume)
