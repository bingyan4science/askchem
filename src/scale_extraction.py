"""
Scaled Extraction Pipeline for AskChem.

Processes open-access papers in batches using abstracts (since we can't download
all PDFs at scale). Uses GPT-4o-mini for cost efficiency at scale.

Strategy:
- Use abstracts for broad extraction (available for all papers)
- Use full PDFs for deep extraction (available for ~40% of papers)
- Process in batches with checkpointing
"""

import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from askchem.models import Claim, Source
from askchem.store import AskChemStore
from askchem.indexer import classify_claim

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
CHECKPOINT_DIR = Path(__file__).parent.parent / "experiments" / "005_scale_extraction"

client = OpenAI()

ABSTRACT_EXTRACTION_PROMPT = """You are a chemistry expert. Extract structured knowledge claims from this paper's abstract and metadata.

Paper metadata:
Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Abstract: {abstract}

Extract ALL factual claims. Return a JSON object with:
{{
  "claims": [
    {{
      "claim_id": sequential number,
      "claim_type": "reaction|property|method|mechanism|comparison|computational_result",
      "confidence": "high|medium|low",

      // For reactions:
      "reaction_type": "e.g., Suzuki coupling",
      "reactants": [{{"name": "...", "smiles": "or null", "role": "substrate|reagent|catalyst"}}],
      "products": [{{"name": "...", "smiles": "or null"}}],
      "conditions": {{"catalyst": "...", "solvent": "...", "temperature": "...", "other": "..."}},
      "outcomes": {{"yield_percent": null, "selectivity": "...", "other": "..."}},

      // For properties:
      "subject": "molecule/material",
      "property_name": "e.g., BET surface area",
      "value": "value with units",
      "measurement_method": "technique",

      // For methods:
      "technique_name": "name",
      "what_it_achieves": "description",

      // For mechanisms:
      "process_described": "what process",
      "steps": ["step1", "step2"],

      // For all:
      "verbatim_quote": "exact text from abstract"
    }}
  ]
}}

Extract 3-10 claims from the abstract. Focus on the main findings."""


def extract_from_abstract(paper: dict) -> list[dict]:
    """Extract claims from a paper's abstract using GPT-4o-mini."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if not abstract:
        return []

    authors = [a.get("name", "") for a in (paper.get("authors") or [])[:5]]
    year = paper.get("year", "")
    venue = paper.get("venue", "")

    prompt = ABSTRACT_EXTRACTION_PROMPT.format(
        title=title, authors=", ".join(authors), year=year, venue=venue, abstract=abstract
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("claims", [])
    except Exception as e:
        return []


def process_batch(papers: list[dict], batch_id: int, checkpoint_dir: Path) -> list[dict]:
    """Process a batch of papers and save checkpoint."""
    results = []
    for i, paper in enumerate(papers):
        pid = paper.get("paperId", "unknown")
        title = (paper.get("title") or "")[:60]

        claims = extract_from_abstract(paper)

        doi = (paper.get("externalIds") or {}).get("DOI", "")
        results.append({
            "paperId": pid,
            "doi": doi,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])[:5]],
            "citation_count": paper.get("citationCount", 0),
            "claims": claims,
            "num_claims": len(claims),
        })

        if (i + 1) % 20 == 0:
            print(f"  Batch {batch_id}: {i+1}/{len(papers)} papers processed", flush=True)

    # Save checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / f"batch_{batch_id:04d}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_scaled_extraction(max_papers: int = 500, batch_size: int = 50):
    """Run extraction on open-access papers at scale."""
    checkpoint_dir = CHECKPOINT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load papers
    with open(DATA_DIR / "metadata" / "all_papers.json") as f:
        all_papers = json.load(f)

    # Filter to papers with abstracts
    papers_with_abstracts = [p for p in all_papers if p.get("abstract")]
    papers_with_abstracts.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)
    papers = papers_with_abstracts[:max_papers]

    print(f"Scaled extraction: {len(papers)} papers (of {len(papers_with_abstracts)} with abstracts)", flush=True)

    # Check for existing checkpoints
    existing = list(checkpoint_dir.glob("batch_*.json"))
    processed_ids = set()
    if existing:
        for f in existing:
            with open(f) as fh:
                batch_data = json.load(fh)
            for r in batch_data:
                processed_ids.add(r["paperId"])
        print(f"Found {len(existing)} existing checkpoints ({len(processed_ids)} papers already processed)", flush=True)

    # Filter out already processed
    papers = [p for p in papers if p.get("paperId") not in processed_ids]
    print(f"Remaining: {len(papers)} papers to process", flush=True)

    if not papers:
        print("All papers already processed!", flush=True)
        return

    # Process in batches
    all_results = []
    batch_id = len(existing)
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        print(f"\nBatch {batch_id} ({len(batch)} papers)...", flush=True)

        results = process_batch(batch, batch_id, checkpoint_dir)
        all_results.extend(results)

        total_claims = sum(r["num_claims"] for r in results)
        print(f"  -> {total_claims} claims from {len(results)} papers", flush=True)

        batch_id += 1
        time.sleep(2)

    # Summary
    total_claims = sum(r["num_claims"] for r in all_results)
    papers_with_claims = sum(1 for r in all_results if r["num_claims"] > 0)

    print(f"\n{'='*60}", flush=True)
    print("SCALED EXTRACTION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Papers processed: {len(all_results)}", flush=True)
    print(f"Papers with claims: {papers_with_claims}", flush=True)
    print(f"Total claims: {total_claims}", flush=True)
    print(f"Average claims/paper: {total_claims/max(len(all_results),1):.1f}", flush=True)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "papers_processed": len(all_results),
        "papers_with_claims": papers_with_claims,
        "total_claims": total_claims,
        "avg_claims_per_paper": total_claims / max(len(all_results), 1),
    }
    with open(CHECKPOINT_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    print(f"AskChem Scaled Extraction - {datetime.now().isoformat()}", flush=True)
    # Start with 500 papers for Round 1
    run_scaled_extraction(max_papers=500, batch_size=50)
