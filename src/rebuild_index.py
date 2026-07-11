"""
Rebuild the AskChem index from all extraction results.

Combines:
1. Deep PDF extractions (v2, 10 papers, 130 claims)
2. Scaled abstract extractions (500+ papers)

Then classifies all claims into the 5-view hierarchy.
"""

import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from askchem.models import Claim, Source
from askchem.store import AskChemStore
from askchem.indexer import build_index, load_extracted_claims, classify_claims_batch

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
SCALE_DIR = EXPERIMENTS_DIR / "005_scale_extraction" / "checkpoints"


def load_scaled_claims() -> tuple[list[Claim], list[Source]]:
    """Load claims from scaled abstract extraction."""
    claims = []
    sources_seen = {}

    if not SCALE_DIR.exists():
        return [], []

    for batch_file in sorted(SCALE_DIR.glob("batch_*.json")):
        with open(batch_file) as f:
            batch = json.load(f)

        for paper in batch:
            doi = paper.get("doi", "")
            title = paper.get("title", "")
            if not doi and not title:
                continue

            # Create source
            if doi and doi not in sources_seen:
                source = Source(
                    doi=doi,
                    title=title,
                    authors=paper.get("authors", []),
                    year=paper.get("year") or 0,
                    venue=paper.get("venue", ""),
                    citation_count=paper.get("citation_count", 0),
                )
                sources_seen[doi] = source

            # Create claims
            for raw_claim in paper.get("claims", []):
                claim_type = raw_claim.get("claim_type", "unknown")
                content_hash = str(hash(json.dumps(raw_claim, sort_keys=True)))[:12]
                claim_id = Claim.generate_id(doi or title, claim_type, content_hash)

                claim = Claim(
                    claim_id=claim_id,
                    claim_type=claim_type,
                    source_doi=doi,
                    source_paper_title=title,
                    confidence=raw_claim.get("confidence", "medium"),
                    location_in_paper=raw_claim.get("location_in_paper", "abstract"),
                    verbatim_quote=raw_claim.get("verbatim_quote", ""),
                    extraction_model="gpt-5-mini",
                    extraction_version="v3-abstract",
                    extracted_at=datetime.now().isoformat(),
                    reaction_type=raw_claim.get("reaction_type", ""),
                    reactants=raw_claim.get("reactants", []),
                    products=raw_claim.get("products", []),
                    conditions=raw_claim.get("conditions", {}),
                    outcomes=raw_claim.get("outcomes", {}),
                    subject=raw_claim.get("subject", ""),
                    subject_smiles=raw_claim.get("subject_smiles", ""),
                    property_name=raw_claim.get("property_name", ""),
                    value=str(raw_claim.get("value", "")),
                    unit=raw_claim.get("unit", ""),
                    measurement_method=raw_claim.get("measurement_method", ""),
                    technique_name=raw_claim.get("technique_name", ""),
                    what_it_achieves=raw_claim.get("what_it_achieves", ""),
                    process_described=raw_claim.get("process_described", ""),
                    steps=raw_claim.get("steps", []),
                    compared_items=raw_claim.get("compared_items", []),
                    metric=raw_claim.get("metric", ""),
                    comparison_result=raw_claim.get("comparison_result", ""),
                )
                claims.append(claim)

    return claims, list(sources_seen.values())


def main():
    print(f"AskChem Index Rebuild - {datetime.now().isoformat()}", flush=True)

    # Back up existing index
    if INDEX_DIR.exists():
        backup_dir = INDEX_DIR.parent / f"chemtree_index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(INDEX_DIR, backup_dir)
        print(f"Backed up existing index to {backup_dir.name}", flush=True)
        shutil.rmtree(INDEX_DIR)

    # Initialize fresh store
    store = AskChemStore(INDEX_DIR)
    store.initialize()

    # Load deep PDF extractions
    print("\nLoading deep PDF extractions (v2)...", flush=True)
    pdf_claims, pdf_sources = load_extracted_claims(EXPERIMENTS_DIR)
    print(f"  {len(pdf_claims)} claims from {len(pdf_sources)} sources", flush=True)

    # Load scaled abstract extractions
    print("Loading scaled abstract extractions...", flush=True)
    abstract_claims, abstract_sources = load_scaled_claims()
    print(f"  {len(abstract_claims)} claims from {len(abstract_sources)} sources", flush=True)

    # Combine
    all_claims = pdf_claims + abstract_claims
    all_sources = pdf_sources + abstract_sources

    # Deduplicate sources by DOI
    seen_dois = set()
    unique_sources = []
    for s in all_sources:
        if s.doi not in seen_dois:
            seen_dois.add(s.doi)
            unique_sources.append(s)

    print(f"\nTotal: {len(all_claims)} claims from {len(unique_sources)} sources", flush=True)

    # Build index
    result = build_index(store, all_claims, unique_sources)

    print(f"\n{'='*60}", flush=True)
    print("INDEX REBUILD COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Claims: {result['claims_indexed']}", flush=True)
    print(f"Nodes: {result['nodes_created']}", flush=True)


if __name__ == "__main__":
    main()
