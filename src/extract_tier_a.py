"""
Deep extraction for Tier A papers using OpenAI Batch API.

Builds batch JSONL files from downloaded PDFs in data/papers_full/,
submits to Batch API, polls, and collects results.

Uses the same extraction prompt and model as deep_extract.py.

Usage:
    python src/extract_tier_a.py prepare          # Build batch JSONL files
    python src/extract_tier_a.py submit           # Submit to OpenAI Batch API
    python src/extract_tier_a.py poll             # Check batch status
    python src/extract_tier_a.py collect          # Download and parse results
    python src/extract_tier_a.py status           # Show pipeline progress
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

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers_full"
PIPELINE_DIR = DATA_DIR / "tier_a_pipeline"
RESULTS_DIR = DATA_DIR / "deep_results"
OA_SCAN = DATA_DIR / "oa_scan.json"

MODEL = "gpt-5.4"
MAX_BATCH_FILE_BYTES = 90 * 1024 * 1024
MAX_PDF_BYTES = 50 * 1024 * 1024

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


def load_tier_a_with_pdfs() -> list[dict]:
    """Load Tier A papers that have PDFs on disk."""
    if not OA_SCAN.exists():
        print(f"Error: {OA_SCAN} not found")
        sys.exit(1)

    with open(OA_SCAN) as f:
        scan = json.load(f)

    tier_a = [p for p in scan['papers']
              if p.get('year', 0) >= 2020 and p.get('citation_count', 0) >= 100]

    on_disk = {}
    if PAPERS_DIR.exists():
        for f in PAPERS_DIR.iterdir():
            if f.suffix == '.pdf':
                size = f.stat().st_size
                if 0 < size <= MAX_PDF_BYTES:
                    on_disk[f.stem] = (str(f), f.name, size)

    result = []
    for p in tier_a:
        fhash = doi_to_filename(p['doi'])
        if fhash in on_disk:
            pdf_path, filename, pdf_size = on_disk[fhash]
            p['pdf_path'] = pdf_path
            p['filename'] = filename
            p['pdf_size'] = pdf_size
            result.append(p)

    result.sort(key=lambda p: p['citation_count'], reverse=True)
    return result


def build_batch_request(paper: dict) -> dict:
    pdf_bytes = open(paper['pdf_path'], 'rb').read()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')
    custom_id = doi_to_filename(paper['doi'])

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
            "max_completion_tokens": 65536,
            "response_format": {"type": "json_object"},
        },
    }


def cmd_prepare(args):
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_tier_a_with_pdfs()
    print(f"Tier A papers with PDFs: {len(papers)}")

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
    print(f"Building batch JSONL files (max {MAX_BATCH_FILE_BYTES // 1024 // 1024}MB each)...\n")

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
            if skipped_encode <= 5:
                print(f"  Skip {paper['doi']}: {str(e)[:60]}", flush=True)
            continue

        if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {papers_in_batch} papers, "
                      f"{current_size / 1e6:.1f} MB", flush=True)

            batch_idx += 1
            fname = PIPELINE_DIR / f"batch_tier_a_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            current_size = 0
            papers_in_batch = 0

        current_file.write(line)
        current_size += line_bytes
        papers_in_batch += 1
        total_requests += 1

        if (i + 1) % 25 == 0:
            print(f"  Encoded {i + 1}/{len(to_extract)} papers...", flush=True)

    if current_file:
        current_file.close()
        print(f"  {batch_files[-1].name}: {papers_in_batch} papers, "
              f"{current_size / 1e6:.1f} MB", flush=True)

    meta = {
        "tier": "A",
        "model": MODEL,
        "total_requests": total_requests,
        "batch_files": [f.name for f in batch_files],
        "skipped_encode": skipped_encode,
        "created_at": datetime.now().isoformat(),
        "paper_dois": {doi_to_filename(p['doi']): p['doi'] for p in to_extract
                       if doi_to_filename(p['doi']) not in already_done},
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
    print(f"\nSubmit with: python src/extract_tier_a.py submit")


def cmd_submit(args):
    client = OpenAI(timeout=300.0)

    batch_files = sorted(PIPELINE_DIR.glob("batch_tier_a_*.jsonl"))
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

        for attempt in range(3):
            try:
                uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
                break
            except Exception as e:
                print(f"    Upload attempt {attempt+1} failed: {str(e)[:60]}", flush=True)
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"    Skipping {fpath.name} after 3 failures", flush=True)
                    continue
        else:
            continue

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
        time.sleep(3)

    print(f"\n{len(tracker)} batches tracked.")
    print(f"Poll with: python src/extract_tier_a.py poll")


def cmd_poll(args):
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
        print("\nAll batches done! Collect with: python src/extract_tier_a.py collect")
    else:
        print("\nStill running. Poll again later.")


def cmd_collect(args):
    tracker_file = PIPELINE_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No batches to collect.")
        return

    client = OpenAI()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(tracker_file) as f:
        tracker = json.load(f)

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

                    doi = doi_map.get(custom_id, custom_id)
                    claims = parsed.get("claims", [])
                    paper_knowledge = parsed.get("paper_knowledge", {})

                    result_data = {
                        "doi": doi,
                        "custom_id": custom_id,
                        "num_claims": len(claims),
                        "collected_at": datetime.now().isoformat(),
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                        "data": {
                            "paper_knowledge": paper_knowledge,
                            "claims": claims,
                        },
                    }

                    result_path = RESULTS_DIR / f"{custom_id}.json"
                    with open(result_path, "w") as rf:
                        json.dump(result_data, rf, indent=2)

                    total_collected += 1
                    total_claims += len(claims)

                    if total_collected % 50 == 0:
                        print(f"    Collected {total_collected} papers, "
                              f"{total_claims:,} claims...", flush=True)

                except Exception as e:
                    total_errors += 1
                    if total_errors <= 5:
                        print(f"    Parse error: {str(e)[:80]}")

    print(f"\n{'='*60}")
    print(f"COLLECT COMPLETE")
    print(f"{'='*60}")
    print(f"Papers collected: {total_collected}")
    print(f"Total claims: {total_claims:,}")
    print(f"Errors: {total_errors}")
    print(f"Avg claims/paper: {total_claims / max(1, total_collected):.1f}")
    print(f"\nResults saved to: {RESULTS_DIR}")


def cmd_status(args):
    print("=== Tier A Extraction Pipeline Status ===\n")

    pdfs = list(PAPERS_DIR.glob("*.pdf")) if PAPERS_DIR.exists() else []
    print(f"PDFs on disk: {len(pdfs)}")

    if PIPELINE_DIR.exists():
        batch_files = list(PIPELINE_DIR.glob("batch_tier_a_*.jsonl"))
        print(f"Batch files: {len(batch_files)}")
        meta_file = PIPELINE_DIR / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            print(f"Total requests: {meta.get('total_requests', '?')}")
    else:
        print("Pipeline: not started")

    results = list(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    if results:
        total_claims = 0
        for rf in results:
            try:
                d = json.loads(rf.read_text())
                total_claims += d.get("num_claims", 0)
            except:
                pass
        print(f"Results: {len(results)} papers, {total_claims:,} claims")
    else:
        print("Results: none yet")


def main():
    parser = argparse.ArgumentParser(description="Tier A deep extraction pipeline")
    parser.add_argument("command", choices=["prepare", "submit", "poll", "collect", "status"])
    parser.add_argument("--max", type=int, help="Max papers to extract")
    args = parser.parse_args()

    cmd = {
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "poll": cmd_poll,
        "collect": cmd_collect,
        "status": cmd_status,
    }[args.command]
    cmd(args)


if __name__ == "__main__":
    main()
