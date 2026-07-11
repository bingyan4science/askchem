"""
Extraction v2: Two-stage deep paper analysis.

Stage 1: Paper overview — understand structure, identify what to extract
Stage 2: Exhaustive extraction with improved prompts targeting 30-100 claims per paper

Based on learnings from v1 (see experiments/003_extraction_v2/decisions.md).
"""

import json
import base64
import os
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACTION_DIR = Path(__file__).parent.parent / "experiments" / "003_extraction_v2"

client = OpenAI()

STAGE1_PROMPT = """You are a chemistry expert performing a preliminary analysis of a research paper.

Your task is to understand the paper's structure and create an inventory of extractable knowledge. Return a JSON object with:

{
  "paper_metadata": {
    "title": "full title",
    "authors": ["first author", "..."],
    "year": 2024,
    "journal": "journal name",
    "doi": "if visible",
    "paper_type": "research_article|review|communication|computational_study|methods_paper",
    "subfield": "organic_synthesis|inorganic|materials|catalysis|physical_chemistry|biochemistry|computational|electrochemistry|photochemistry|other"
  },
  "paper_summary": "3-5 sentence summary of the paper's main contribution and findings",
  "content_inventory": {
    "num_reactions_described": N,
    "has_substrate_scope_table": true/false,
    "num_scope_entries_estimated": N,
    "has_optimization_table": true/false,
    "num_optimization_entries_estimated": N,
    "num_characterized_compounds": N,
    "num_figures": N,
    "num_tables": N,
    "has_mechanistic_study": true/false,
    "has_computational_results": true/false,
    "has_comparison_data": true/false,
    "key_molecules_mentioned": ["list of key molecule names"],
    "key_techniques_used": ["list of techniques/instruments"]
  },
  "extraction_plan": "Brief description of what should be extracted in Stage 2"
}

Be thorough in counting — look at ALL tables, figures, and supplementary information visible."""

STAGE2_PROMPT = """You are a chemistry expert performing EXHAUSTIVE knowledge extraction from a research paper.

Based on the preliminary analysis, extract EVERY piece of structured knowledge. Your goal is 30-100 claims per paper. Do NOT summarize — extract individual data points.

Return a JSON object with:

{
  "claims": [
    {
      "claim_id": 1,
      "claim_type": "reaction|property|method|mechanism|comparison|structure|scope_entry|computational_result",
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
      "parent_reaction_id": null or ID (for scope entries, reference the parent reaction),

      // FOR PROPERTIES:
      "subject": "molecule/material name",
      "subject_smiles": "...",
      "property_name": "e.g., melting point, BET surface area, IC50",
      "property_category": "physical|chemical|biological|spectroscopic|electrochemical|mechanical|optical|thermal",
      "value": "numerical value with units",
      "measurement_method": "instrument/technique",
      "is_computed": false,

      // FOR MECHANISMS:
      "process_described": "what reaction/process",
      "steps": ["step 1", "step 2", ...],
      "key_intermediates": ["..."],
      "evidence": [{"type": "...", "description": "..."}],

      // FOR COMPARISONS:
      "compared_items": ["item A", "item B"],
      "metric": "what's being compared",
      "result": "A is better/worse/equal to B by X",

      // FOR ALL:
      "verbatim_quote": "exact sentence(s) from paper supporting this claim"
    }
  ]
}

CRITICAL INSTRUCTIONS:
1. Extract EVERY entry from substrate scope tables — each row is a separate claim with claim_type "scope_entry"
2. Extract EVERY entry from optimization tables — each row is a separate claim
3. Extract ALL characterization data (NMR peaks, IR bands, MS values, XRD parameters, etc.)
4. Extract ALL numerical results from figures where readable
5. For every molecule, attempt to provide SMILES. Write "not determinable" if you cannot.
6. Do NOT skip data because it seems routine — every data point matters.
7. Include control experiments and negative results.
8. A typical research paper should yield 30-100 claims. If you have fewer than 20, you are likely missing data."""


def encode_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def call_gpt4o(pdf_path, prompt, max_completion_tokens=16000):
    pdf_b64 = encode_pdf(pdf_path)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "file", "file": {
                    "filename": pdf_path.name,
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                }},
            ],
        }],
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
    )
    return {
        "result": json.loads(response.choices[0].message.content),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def extract_paper(pdf_path):
    """Two-stage extraction on a single paper."""
    name = pdf_path.stem

    # Stage 1: Overview
    print(f"  Stage 1: Paper overview...", flush=True)
    stage1 = call_gpt4o(pdf_path, STAGE1_PROMPT, max_completion_tokens=4000)
    overview = stage1["result"]
    inv = overview.get("content_inventory", {})
    print(f"    Type: {overview.get('paper_metadata', {}).get('paper_type', '?')}", flush=True)
    print(f"    Subfield: {overview.get('paper_metadata', {}).get('subfield', '?')}", flush=True)
    print(f"    Reactions: ~{inv.get('num_reactions_described', '?')}, "
          f"Scope entries: ~{inv.get('num_scope_entries_estimated', '?')}, "
          f"Compounds: ~{inv.get('num_characterized_compounds', '?')}", flush=True)
    print(f"    Tokens: {stage1['usage']['total_tokens']}", flush=True)

    time.sleep(2)

    # Stage 2: Exhaustive extraction
    # Customize the prompt based on Stage 1 findings
    stage2_context = f"""Paper overview from preliminary analysis:
- Title: {overview.get('paper_metadata', {}).get('title', 'unknown')}
- Type: {overview.get('paper_metadata', {}).get('paper_type', 'unknown')}
- Subfield: {overview.get('paper_metadata', {}).get('subfield', 'unknown')}
- Content: {json.dumps(inv)}
- Plan: {overview.get('extraction_plan', 'Extract all claims')}

"""
    full_prompt = stage2_context + STAGE2_PROMPT

    print(f"  Stage 2: Exhaustive extraction...", flush=True)
    stage2 = call_gpt4o(pdf_path, full_prompt, max_completion_tokens=16000)
    claims = stage2["result"].get("claims", [])
    print(f"    Claims extracted: {len(claims)}", flush=True)
    print(f"    Tokens: {stage2['usage']['total_tokens']}", flush=True)

    # Count by type
    type_counts = {}
    for c in claims:
        ct = c.get("claim_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"    By type: {type_counts}", flush=True)

    total_tokens = stage1["usage"]["total_tokens"] + stage2["usage"]["total_tokens"]

    return {
        "paper_name": name,
        "pdf_path": str(pdf_path),
        "timestamp": datetime.now().isoformat(),
        "stage1": stage1,
        "stage2": stage2,
        "total_tokens": total_tokens,
        "num_claims": len(claims),
        "claim_type_distribution": type_counts,
    }


def main():
    os.makedirs(EXTRACTION_DIR / "raw", exist_ok=True)
    os.makedirs(EXTRACTION_DIR / "results", exist_ok=True)
    os.makedirs(EXTRACTION_DIR / "prompts", exist_ok=True)

    # Save prompts
    with open(EXTRACTION_DIR / "prompts" / "v2_prompts.json", "w") as f:
        json.dump({
            "stage1": STAGE1_PROMPT,
            "stage2": STAGE2_PROMPT,
        }, f, indent=2)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Extraction v2: {len(pdfs)} papers", flush=True)

    all_results = []
    for pdf_path in pdfs:
        print(f"\n{'='*60}", flush=True)
        print(f"{pdf_path.stem[:60]}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            result = extract_paper(pdf_path)
            all_results.append(result)

            # Save individual result
            with open(EXTRACTION_DIR / "raw" / f"{pdf_path.stem}.json", "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_results.append({"paper_name": pdf_path.stem, "error": str(e)})

        time.sleep(3)

    # Save combined results
    with open(EXTRACTION_DIR / "results" / "all_extractions_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("EXTRACTION V2 COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)

    total_claims = 0
    total_tokens = 0
    for r in all_results:
        if "error" in r:
            print(f"  {r['paper_name'][:40]}: ERROR", flush=True)
            continue
        nc = r["num_claims"]
        nt = r["total_tokens"]
        total_claims += nc
        total_tokens += nt
        print(f"  {r['paper_name'][:40]}: {nc} claims, {nt:,} tokens", flush=True)
        print(f"    Types: {r['claim_type_distribution']}", flush=True)

    print(f"\nTotal: {total_claims} claims, {total_tokens:,} tokens across {len(all_results)} papers", flush=True)
    print(f"Average: {total_claims/max(len(all_results),1):.1f} claims/paper", flush=True)
    if total_claims > 0:
        print(f"Improvement over v1: {total_claims/86:.1f}x more claims (v1 had 86)", flush=True)


if __name__ == "__main__":
    print(f"AskChem Extraction v2 - {datetime.now().isoformat()}", flush=True)
    main()
