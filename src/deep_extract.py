"""
AskChem Deep Extraction Pipeline — full-paper claim extraction at scale.

Uses the OpenAI Batch API with gpt-5.4 and native PDF input for maximum
extraction quality. Batch API provides 50% cost reduction and 24h turnaround.

Usage:
    python src/deep_extract.py prepare --tier 1        # Build batch JSONL files
    python src/deep_extract.py submit                   # Submit to OpenAI Batch API
    python src/deep_extract.py poll                     # Check batch status
    python src/deep_extract.py collect                  # Download results
    python src/deep_extract.py status                   # Show pipeline progress
    python src/deep_extract.py results                  # Summarize extracted claims
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from askchem.llm import MODELS

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers_full"
CORPUS_DIR = DATA_DIR / "corpus_checkpoints"
PIPELINE_DIR = DATA_DIR / "deep_pipeline"
RESULTS_DIR = DATA_DIR / "deep_results"

MODEL = MODELS["strong"]  # gpt-5.4

TIER_LIMITS = {1: 5_000, 2: 25_000, 3: 80_000}
MAX_BATCH_FILE_BYTES = 90 * 1024 * 1024  # 90MB safety margin (limit is 100MB)
MAX_PDF_BYTES = 50 * 1024 * 1024  # skip PDFs larger than 50MB (base64 = ~67MB)

EXTRACTION_PROMPT = """You are a chemistry expert performing EXHAUSTIVE knowledge extraction from a research paper.

Extract EVERY piece of structured knowledge. Target 20-50 claims per paper. Do NOT summarize — extract individual data points.

Return a JSON object with:
{
  "paper_knowledge": {
    "hypothesis": "The central hypothesis or research question",
    "experimental_design": "Brief description of the experimental approach",
    "conclusions": ["Main conclusion 1", "Main conclusion 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "future_directions": ["Future direction 1", "Future direction 2"],
    "surprising_findings": ["Any unexpected or counter-intuitive results"],
    "paper_type": "research_article|review|communication|computational_study|methods_paper",
    "subfield": "organic_synthesis|inorganic|materials|catalysis|physical_chemistry|biochemistry|computational|electrochemistry|photochemistry|polymer|environmental|analytical|other"
  },
  "claims": [
    {
      "claim_id": sequential number,
      "claim_type": "reaction|property|method|mechanism|comparison|scope_entry|computational_result|structure|hypothesis|experimental_design|limitation|future_direction|surprising_finding",
      "confidence": "high|medium|low",
      "location_in_paper": "Table 1, entry 3" or "Figure 2" or "Results, paragraph 4",

      // FOR REACTIONS (including each scope entry as a separate claim):
      "reaction_type": "e.g., Suzuki coupling, C-H activation, MOF synthesis",
      "reactants": [
        {"name": "...", "smiles": "... or null if not determinable", "role": "substrate|reagent|catalyst|ligand|additive"}
      ],
      "products": [
        {"name": "...", "smiles": "... or null", "role": "major|minor|byproduct"}
      ],
      "conditions": {
        "catalyst": "...", "ligand": "...", "solvent": "...",
        "temperature": "...", "time": "...", "atmosphere": "...",
        "additives": ["..."], "concentration": "...", "other": "..."
      },
      "outcomes": {
        "yield_percent": number or null,
        "ee_percent": number or null,
        "dr": "...",
        "selectivity": "...",
        "conversion_percent": number or null,
        "turnover_number": number or null
      },
      "is_key_result": true/false,

      // FOR PROPERTIES:
      "subject": "molecule/material name",
      "subject_smiles": "...",
      "property_name": "e.g., melting point, BET surface area, IC50",
      "property_category": "physical|chemical|biological|spectroscopic|electrochemical|mechanical|optical|thermal",
      "value": "numerical value with units",
      "measurement_method": "instrument/technique",

      // FOR MECHANISMS:
      "process_described": "what reaction/process",
      "steps": ["step 1", "step 2"],
      "key_intermediates": ["..."],
      "evidence": [{"type": "...", "description": "..."}],

      // FOR METHODS:
      "technique_name": "name",
      "what_it_achieves": "description",
      "key_innovation": "what's new",

      // FOR COMPARISONS:
      "compared_items": ["item A", "item B"],
      "metric": "what's being compared",
      "comparison_result": "A is better/worse/equal to B by X",

      // FOR HYPOTHESIS:
      "hypothesis_text": "The specific hypothesis being tested",

      // FOR LIMITATION:
      "limitation_text": "The specific limitation described",

      // FOR FUTURE_DIRECTION:
      "direction_text": "The specific future direction suggested",

      // FOR SURPRISING_FINDING:
      "finding_text": "The unexpected result",
      "why_surprising": "Why this is unexpected given prior knowledge",

      // FOR ALL:
      "verbatim_quote": "exact sentence(s) from paper supporting this claim"
    }
  ]
}

CRITICAL INSTRUCTIONS:
1. Extract EVERY entry from substrate scope tables — each row is a separate claim
2. Extract EVERY entry from optimization tables — each row is a separate claim
3. Extract ALL characterization data (NMR, IR, MS, XRD, etc.)
4. Extract ALL numerical results from figures where readable
5. Include control experiments and negative results
6. Extract hypotheses from the introduction
7. Extract limitations from the discussion/conclusion
8. Extract future directions from the conclusion
9. Flag any surprising or counter-intuitive findings
10. A typical paper should yield 20-50 claims. If you have fewer than 15, you are likely missing data."""


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def load_tier_papers(tier: int, max_papers: int | None = None) -> list[dict]:
    """Load Tier N papers that have PDFs on disk, sorted by citation count."""
    on_disk = {f: (PAPERS_DIR / f).stat().st_size
               for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')}

    shards = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith('.jsonl'))
    papers = {}
    for shard in shards:
        with open(CORPUS_DIR / shard) as f:
            for line in f:
                paper = json.loads(line)
                doi = (paper.get('externalIds') or {}).get('DOI', '')
                if not doi or doi.lower() in papers:
                    continue
                papers[doi.lower()] = {
                    'doi': doi,
                    'title': paper.get('title', '')[:120],
                    'citations': paper.get('citationCount', 0) or 0,
                    'year': paper.get('year') or 0,
                }

    all_sorted = sorted(papers.values(), key=lambda p: p['citations'], reverse=True)
    tier_limit = TIER_LIMITS.get(tier, 5_000)
    tier_papers = all_sorted[:tier_limit]

    result = []
    skipped_size = 0
    for p in tier_papers:
        fname = doi_to_filename(p['doi']) + '.pdf'
        if fname in on_disk:
            size = on_disk[fname]
            if size > MAX_PDF_BYTES:
                skipped_size += 1
                continue
            if size == 0:
                continue
            p['pdf_path'] = str(PAPERS_DIR / fname)
            p['filename'] = fname
            p['pdf_size'] = size
            result.append(p)

    result.sort(key=lambda p: p['citations'], reverse=True)
    if max_papers:
        result = result[:max_papers]

    print(f"Tier {tier}: {len(tier_papers):,} total papers")
    print(f"  With PDFs on disk: {len(result) + skipped_size}")
    print(f"  Skipped (>{MAX_PDF_BYTES // 1024 // 1024}MB): {skipped_size}")
    print(f"  Ready for extraction: {len(result)}")
    return result


def build_batch_request(paper: dict) -> dict:
    """Build a single Batch API request line for a paper."""
    custom_id = doi_to_filename(paper['doi'])

    pdf_bytes = open(paper['pdf_path'], 'rb').read()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "file", "file": {
                        "filename": paper['filename'],
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    }},
                ],
            }],
            "max_completion_tokens": 16384,
            "response_format": {"type": "json_object"},
        },
    }


def cmd_prepare(args):
    """Build batch JSONL files from Tier N PDFs."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_tier_papers(args.tier, args.max)
    if not papers:
        print("No papers to extract.")
        return

    # Check which papers are already extracted
    already_done = set()
    if RESULTS_DIR.exists():
        already_done = {f.stem for f in RESULTS_DIR.glob("*.json")}

    to_extract = [p for p in papers if doi_to_filename(p['doi']) not in already_done]
    print(f"  Already extracted: {len(papers) - len(to_extract)}")
    print(f"  To extract: {len(to_extract)}")

    if not to_extract:
        print("\nAll papers already extracted!")
        return

    print(f"\nModel: {MODEL}")
    print(f"Building batch JSONL files (max {MAX_BATCH_FILE_BYTES // 1024 // 1024}MB each)...\n",
          flush=True)

    batch_idx = 0
    current_file = None
    current_size = 0
    papers_in_batch = 0
    total_requests = 0
    batch_files = []
    skipped_encode = 0

    for i, paper in enumerate(to_extract):
        try:
            request = build_batch_request(paper)
            line = json.dumps(request) + "\n"
            line_bytes = len(line.encode('utf-8'))
        except Exception as e:
            skipped_encode += 1
            if skipped_encode <= 3:
                print(f"  Skip {paper['doi']}: {str(e)[:60]}", flush=True)
            continue

        if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {papers_in_batch} papers, "
                      f"{current_size / 1e6:.1f} MB", flush=True)

            batch_idx += 1
            fname = PIPELINE_DIR / f"batch_{args.tier}_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            current_size = 0
            papers_in_batch = 0

        current_file.write(line)
        current_size += line_bytes
        papers_in_batch += 1
        total_requests += 1

        if (i + 1) % 50 == 0:
            print(f"  Encoded {i + 1}/{len(to_extract)} papers...", flush=True)

    if current_file:
        current_file.close()
        print(f"  {batch_files[-1].name}: {papers_in_batch} papers, "
              f"{current_size / 1e6:.1f} MB", flush=True)

    # Save metadata
    meta = {
        "tier": args.tier,
        "model": MODEL,
        "total_requests": total_requests,
        "batch_files": [f.name for f in batch_files],
        "skipped_encode": skipped_encode,
        "created_at": datetime.now().isoformat(),
        "paper_dois": {doi_to_filename(p['doi']): p['doi'] for p in to_extract},
    }
    with open(PIPELINE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_size = sum(f.stat().st_size for f in batch_files) / 1e9
    print(f"\n{'='*60}")
    print(f"PREPARE COMPLETE")
    print(f"{'='*60}")
    print(f"Total requests: {total_requests}")
    print(f"Batch files: {len(batch_files)} ({total_size:.2f} GB)")
    print(f"Skipped (encode error): {skipped_encode}")
    print(f"Model: {MODEL}")
    print(f"\nSubmit with: python src/deep_extract.py submit")


def cmd_submit(args):
    """Submit batch JSONL files to OpenAI Batch API."""
    client = OpenAI()

    batch_files = sorted(PIPELINE_DIR.glob(f"batch_*.jsonl"))
    if not batch_files:
        print("No batch files found. Run 'prepare' first.")
        return

    tracker_file = PIPELINE_DIR / "tracker.json"
    tracker = {}
    if tracker_file.exists():
        with open(tracker_file) as f:
            tracker = json.load(f)

    for fpath in batch_files:
        if fpath.name in tracker and tracker[fpath.name].get('status') not in ('failed', 'expired', 'cancelled'):
            print(f"  {fpath.name}: already submitted (batch {tracker[fpath.name]['batch_id']}, "
                  f"status={tracker[fpath.name].get('status')})")
            continue

        size_mb = fpath.stat().st_size / 1e6
        print(f"  Uploading {fpath.name} ({size_mb:.1f} MB)...", flush=True)

        uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
        print(f"  File uploaded: {uploaded.id}", flush=True)

        print(f"  Creating batch...", flush=True)
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        tracker[fpath.name] = {
            "batch_id": batch.id,
            "file_id": uploaded.id,
            "status": batch.status,
            "submitted_at": datetime.now().isoformat(),
        }
        with open(tracker_file, "w") as f:
            json.dump(tracker, f, indent=2)

        print(f"  Batch {batch.id} created ({batch.status})")
        time.sleep(2)

    print(f"\n{len(tracker)} batches tracked.")
    print(f"Poll with: python src/deep_extract.py poll")


def cmd_poll(args):
    """Check status of submitted batches."""
    tracker_file = PIPELINE_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No batches submitted yet.")
        return

    client = OpenAI()
    with open(tracker_file) as f:
        tracker = json.load(f)

    all_done = True
    total_completed = 0
    total_failed = 0
    total_total = 0

    for fname, info in sorted(tracker.items()):
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        if batch.output_file_id:
            info["output_file_id"] = batch.output_file_id
        if batch.error_file_id:
            info["error_file_id"] = batch.error_file_id

        status_str = batch.status
        if batch.request_counts:
            rc = batch.request_counts
            status_str += f" ({rc.completed}/{rc.total} done, {rc.failed} failed)"
            total_completed += rc.completed
            total_failed += rc.failed
            total_total += rc.total

        print(f"  {fname}: {status_str}")

        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_done = False

    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"\nOverall: {total_completed}/{total_total} completed, {total_failed} failed")

    if all_done:
        print("\nAll batches done! Collect with: python src/deep_extract.py collect")
    else:
        print("\nStill running. Poll again later.")


def cmd_collect(args):
    """Download batch results and save individual paper extractions."""
    tracker_file = PIPELINE_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No batches to collect.")
        return

    client = OpenAI()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(tracker_file) as f:
        tracker = json.load(f)

    # Load DOI mapping
    meta_file = PIPELINE_DIR / "meta.json"
    doi_map = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        doi_map = meta.get("paper_dois", {})

    total_collected = 0
    total_claims = 0
    total_errors = 0

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            print(f"  {fname}: no output (status={info.get('status')})")
            continue

        raw_path = PIPELINE_DIR / "raw_results" / fname
        raw_path.parent.mkdir(exist_ok=True)

        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            with open(raw_path, "wb") as f:
                f.write(content.read())
            print(f"  Saved raw results ({raw_path.stat().st_size / 1e6:.1f} MB)")

        # Parse results into individual paper files
        with open(raw_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    response = result.get("response", {})
                    body = response.get("body", {})

                    if response.get("status_code") != 200:
                        total_errors += 1
                        error = body.get("error", {})
                        if total_errors <= 5:
                            print(f"    Error for {custom_id}: {error.get('message', 'unknown')[:80]}")
                        continue

                    choices = body.get("choices", [])
                    if not choices:
                        total_errors += 1
                        continue

                    text = choices[0].get("message", {}).get("content", "")
                    usage = body.get("usage", {})

                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        total_errors += 1
                        continue

                    claims = parsed.get("claims", [])
                    doi = doi_map.get(custom_id, custom_id)

                    output = {
                        "doi": doi,
                        "custom_id": custom_id,
                        "model": MODEL,
                        "num_claims": len(claims),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "collected_at": datetime.now().isoformat(),
                        "data": parsed,
                    }

                    result_path = RESULTS_DIR / f"{custom_id}.json"
                    with open(result_path, "w") as rf:
                        json.dump(output, rf, indent=2)

                    total_collected += 1
                    total_claims += len(claims)

                except Exception as e:
                    total_errors += 1
                    if total_errors <= 5:
                        print(f"    Parse error: {str(e)[:80]}")

        # Also download error file if present
        error_id = info.get("error_file_id")
        if error_id:
            err_path = PIPELINE_DIR / "raw_results" / f"errors_{fname}"
            if not err_path.exists():
                try:
                    content = client.files.content(error_id)
                    with open(err_path, "wb") as f:
                        f.write(content.read())
                except Exception:
                    pass

    print(f"\n{'='*60}")
    print(f"COLLECT COMPLETE")
    print(f"{'='*60}")
    print(f"Papers collected: {total_collected}")
    print(f"Total claims: {total_claims}")
    print(f"Errors: {total_errors}")
    print(f"Results in: {RESULTS_DIR}")


def cmd_status(args):
    """Show pipeline progress."""
    print("=== Deep Extraction Pipeline ===\n")

    # PDFs
    if PAPERS_DIR.exists():
        n_pdfs = len([f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')])
        print(f"PDFs on disk: {n_pdfs:,}")

    # Prepared
    meta_file = PIPELINE_DIR / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"Prepared: {meta['total_requests']:,} requests "
              f"(Tier {meta['tier']}, model: {meta['model']})")
        print(f"  Batch files: {len(meta.get('batch_files', []))}")
    else:
        print("Prepared: none")

    # Batches
    tracker_file = PIPELINE_DIR / "tracker.json"
    if tracker_file.exists():
        with open(tracker_file) as f:
            tracker = json.load(f)
        statuses = Counter(info["status"] for info in tracker.values())
        print(f"Batches: {dict(statuses)}")
    else:
        print("Batches: none submitted")

    # Results
    if RESULTS_DIR.exists():
        result_files = list(RESULTS_DIR.glob("*.json"))
        total_claims = 0
        total_prompt = 0
        total_completion = 0
        for rf in result_files:
            try:
                data = json.loads(rf.read_text())
                total_claims += data.get('num_claims', 0)
                total_prompt += data.get('prompt_tokens', 0)
                total_completion += data.get('completion_tokens', 0)
            except Exception:
                pass
        print(f"Extracted papers: {len(result_files):,}")
        print(f"Total claims: {total_claims:,}")
        if result_files:
            print(f"Avg claims/paper: {total_claims / len(result_files):.1f}")
            print(f"Tokens: {total_prompt:,} prompt + {total_completion:,} completion")
    else:
        print("Extracted: 0")


def cmd_results(args):
    """Detailed summary of extraction results."""
    if not RESULTS_DIR.exists():
        print("No results yet.")
        return

    result_files = list(RESULTS_DIR.glob("*.json"))
    print(f"Total extracted papers: {len(result_files)}\n")

    claim_counts = []
    claim_types = Counter()
    subfields = Counter()
    paper_types = Counter()
    total_prompt = 0
    total_completion = 0

    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
            n = data.get('num_claims', 0)
            claim_counts.append(n)
            total_prompt += data.get('prompt_tokens', 0)
            total_completion += data.get('completion_tokens', 0)

            parsed = data.get('data', {})
            pk = parsed.get('paper_knowledge', {})
            subfields[pk.get('subfield', 'unknown')] += 1
            paper_types[pk.get('paper_type', 'unknown')] += 1

            for c in parsed.get('claims', []):
                claim_types[c.get('claim_type', 'unknown')] += 1
        except Exception:
            pass

    if claim_counts:
        claim_counts.sort()
        n = len(claim_counts)
        print(f"Claims per paper:")
        print(f"  Min: {claim_counts[0]}, P25: {claim_counts[n//4]}, "
              f"Median: {claim_counts[n//2]}, P75: {claim_counts[3*n//4]}, "
              f"Max: {claim_counts[-1]}")
        print(f"  Total claims: {sum(claim_counts):,}")

    if claim_types:
        print(f"\nClaim types:")
        for ct, count in claim_types.most_common():
            print(f"  {ct}: {count:,}")

    if subfields:
        print(f"\nSubfields:")
        for sf, count in subfields.most_common(10):
            print(f"  {sf}: {count}")

    if paper_types:
        print(f"\nPaper types:")
        for pt, count in paper_types.most_common():
            print(f"  {pt}: {count}")

    print(f"\nTokens: {total_prompt:,} prompt + {total_completion:,} completion")
    cost_prompt = total_prompt / 1_000_000 * 2.50 * 0.5
    cost_completion = total_completion / 1_000_000 * 10.00 * 0.5
    print(f"Estimated cost (Batch 50% discount): ${cost_prompt + cost_completion:.2f}")


def main():
    parser = argparse.ArgumentParser(description="AskChem Deep Extraction Pipeline")
    sub = parser.add_subparsers(dest="command")

    prep_p = sub.add_parser("prepare", help="Build batch JSONL files")
    prep_p.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    prep_p.add_argument("--max", type=int, default=None, help="Limit number of papers")

    sub.add_parser("submit", help="Submit batches to OpenAI")
    sub.add_parser("poll", help="Check batch status")
    sub.add_parser("collect", help="Download and parse results")
    sub.add_parser("status", help="Show pipeline progress")
    sub.add_parser("results", help="Detailed results summary")

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "poll":
        cmd_poll(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "results":
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
