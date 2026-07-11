"""
Deep Paper Analysis and Claim Extraction for AskChem.

Tests multiple extraction approaches on downloaded PDFs using GPT-4o:
1. Single-pass: one prompt extracts all structured claims
2. Multi-pass: separate prompts for reactions, properties, methods, mechanisms
3. Comparison of extraction quality across approaches

Uses OpenAI's vision API to read PDF pages as images.
"""

import json
import base64
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
EXTRACTION_DIR = EXPERIMENTS_DIR / "002_extraction_v1"

client = OpenAI()

SINGLE_PASS_PROMPT = """You are a chemistry expert analyzing a research paper. Extract ALL structured knowledge claims from this paper.

For each claim, provide:

1. **claim_type**: One of: "reaction", "property", "method", "mechanism", "observation", "computational_result"

2. **For reactions:**
   - reactants: list of molecules/reagents (use IUPAC names or common names + SMILES if possible)
   - products: list of products
   - conditions: {catalyst, solvent, temperature, pressure, time, atmosphere, other}
   - outcomes: {yield_percent, selectivity, ee_percent, conversion, turnover_number, other}
   - reaction_type: e.g., "cross-coupling", "oxidation", "reduction", "cyclization", etc.

3. **For properties:**
   - subject: what molecule/material
   - property_name: e.g., "melting point", "band gap", "IC50", "surface area"
   - value: numerical value with units
   - measurement_method: how it was measured

4. **For methods:**
   - technique_name: name of the method/approach
   - what_it_achieves: what goal it accomplishes
   - key_innovation: what's new about this approach
   - limitations: known limitations

5. **For mechanisms:**
   - reaction_or_process: what reaction/process this mechanism describes
   - proposed_pathway: step-by-step mechanism
   - evidence_type: what evidence supports this (DFT, kinetic isotope effect, trapping experiments, etc.)
   - confidence: "established", "proposed", "speculative"

6. **For all claims:**
   - confidence: "high" (directly stated with data), "medium" (inferred from data), "low" (speculative/suggested)
   - location_in_paper: where in the paper this claim appears (e.g., "Table 2", "Figure 3", "Results section paragraph 2")
   - verbatim_quote: the exact sentence(s) from the paper supporting this claim

Return a JSON object with:
{
  "paper_summary": "2-3 sentence summary of the paper's main contribution",
  "paper_type": "research_article" | "review" | "communication" | "computational_study",
  "subfield": primary chemistry subfield,
  "claims": [list of structured claims as described above]
}

Be thorough — extract EVERY factual claim, not just the main results. Include control experiments, characterization data, computational results, etc. A typical research paper should yield 10-50+ claims."""

REACTIONS_ONLY_PROMPT = """You are a chemistry expert. Extract ALL chemical reactions described in this paper.

For each reaction, provide a structured JSON entry:
{
  "reaction_id": sequential number,
  "reaction_type": classification (e.g., "Suzuki coupling", "C-H activation", "nucleophilic substitution"),
  "reactants": [{"name": "...", "smiles": "...", "role": "substrate/reagent/catalyst"}],
  "products": [{"name": "...", "smiles": "...", "role": "major/minor/byproduct"}],
  "conditions": {
    "catalyst": "...",
    "ligand": "...",
    "solvent": "...",
    "temperature": "...",
    "time": "...",
    "atmosphere": "...",
    "additives": ["..."],
    "other": "..."
  },
  "outcomes": {
    "yield_percent": number or null,
    "ee_percent": number or null,
    "dr": "..." or null,
    "selectivity": "...",
    "conversion_percent": number or null,
    "turnover_number": number or null,
    "turnover_frequency": "..." or null
  },
  "is_novel": true/false (is this reaction new or a known reaction applied here?),
  "is_key_result": true/false (is this a main result vs. optimization/control?),
  "scope_entry": true/false (is this part of a substrate scope table?),
  "verbatim_quote": "exact text from paper",
  "location_in_paper": "Table X / Figure Y / text"
}

Extract EVERY reaction mentioned — including substrate scope entries, control experiments, optimization entries, and literature-referenced reactions. Be exhaustive.

Return your answer as a JSON object with a "reactions" key containing the list."""

PROPERTIES_PROMPT = """You are a chemistry expert. Extract ALL measured/computed properties and characterization data from this paper.

For each property measurement, provide:
{
  "property_id": sequential number,
  "subject": "what molecule/material/system",
  "subject_smiles": "SMILES if applicable",
  "property_name": "e.g., melting point, BET surface area, IC50, band gap, pKa",
  "property_category": "physical" | "chemical" | "biological" | "spectroscopic" | "electrochemical" | "computational",
  "value": "numerical value",
  "unit": "unit of measurement",
  "conditions": "measurement conditions if relevant",
  "measurement_method": "instrument/technique used",
  "is_computed": true/false,
  "computation_method": "DFT/B3LYP/6-31G* etc. if computed",
  "verbatim_quote": "exact text",
  "location_in_paper": "where in paper"
}

Be exhaustive — include ALL characterization data (NMR, IR, MS, XRD, BET, TGA, DSC, UV-Vis, fluorescence, electrochemistry, etc.).

Return your answer as a JSON object with a "properties" key containing the list."""

MECHANISMS_PROMPT = """You are a chemistry expert. Extract ALL mechanistic proposals, hypotheses, and explanations from this paper.

For each mechanistic claim:
{
  "mechanism_id": sequential number,
  "process_described": "what reaction/phenomenon this mechanism explains",
  "mechanism_type": "catalytic cycle" | "reaction pathway" | "degradation" | "binding" | "electron transfer" | "other",
  "steps": ["step 1 description", "step 2 description", ...],
  "key_intermediates": ["intermediate species"],
  "rate_determining_step": "if identified",
  "evidence": [
    {"type": "kinetic study", "description": "..."},
    {"type": "DFT calculation", "description": "..."},
    {"type": "isotope labeling", "description": "..."}
  ],
  "confidence": "established" | "proposed" | "speculative",
  "alternative_mechanisms": ["if any alternatives are discussed"],
  "verbatim_quote": "exact text",
  "location_in_paper": "where in paper"
}

Return your answer as a JSON object with a "mechanisms" key containing the list."""


def encode_pdf_for_api(pdf_path):
    """Read PDF and encode as base64 for the OpenAI API."""
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def extract_single_pass(pdf_path, model="gpt-5.4"):
    """Single-pass extraction: one prompt extracts everything."""
    pdf_b64 = encode_pdf_for_api(pdf_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SINGLE_PASS_PROMPT,
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=16000,
        response_format={"type": "json_object"},
    )

    return {
        "method": "single_pass",
        "model": model,
        "result": json.loads(response.choices[0].message.content),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def extract_multi_pass(pdf_path, model="gpt-5.4"):
    """Multi-pass extraction: separate prompts for reactions, properties, mechanisms."""
    pdf_b64 = encode_pdf_for_api(pdf_path)
    results = {}

    for pass_name, prompt in [
        ("reactions", REACTIONS_ONLY_PROMPT),
        ("properties", PROPERTIES_PROMPT),
        ("mechanisms", MECHANISMS_PROMPT),
    ]:
        print(f"      Running {pass_name} pass...", flush=True)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "file",
                            "file": {
                                "filename": pdf_path.name,
                                "file_data": f"data:application/pdf;base64,{pdf_b64}",
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=16000,
            response_format={"type": "json_object"},
        )

        results[pass_name] = {
            "result": json.loads(response.choices[0].message.content),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        time.sleep(2)

    total_tokens = sum(r["usage"]["total_tokens"] for r in results.values())
    return {
        "method": "multi_pass",
        "model": model,
        "results": results,
        "total_tokens": total_tokens,
    }


def run_extraction_experiment():
    """Run extraction on all downloaded PDFs."""
    os.makedirs(EXTRACTION_DIR / "raw", exist_ok=True)
    os.makedirs(EXTRACTION_DIR / "results", exist_ok=True)
    os.makedirs(EXTRACTION_DIR / "prompts", exist_ok=True)

    # Save prompts for reproducibility
    prompts = {
        "single_pass": SINGLE_PASS_PROMPT,
        "reactions_only": REACTIONS_ONLY_PROMPT,
        "properties": PROPERTIES_PROMPT,
        "mechanisms": MECHANISMS_PROMPT,
    }
    with open(EXTRACTION_DIR / "prompts" / "v1_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs to process", flush=True)

    all_results = []

    for pdf_path in pdfs:
        paper_name = pdf_path.stem
        print(f"\n{'='*60}", flush=True)
        print(f"Processing: {paper_name}", flush=True)
        print(f"{'='*60}", flush=True)

        paper_result = {
            "paper_name": paper_name,
            "pdf_path": str(pdf_path),
            "pdf_size_kb": pdf_path.stat().st_size // 1024,
            "timestamp": datetime.now().isoformat(),
        }

        # Single-pass extraction
        print(f"  [1/2] Single-pass extraction...", flush=True)
        try:
            single = extract_single_pass(pdf_path)
            paper_result["single_pass"] = single
            n_claims = len(single["result"].get("claims", []))
            print(f"    -> {n_claims} claims extracted, {single['usage']['total_tokens']} tokens", flush=True)
        except Exception as e:
            print(f"    -> ERROR: {e}", flush=True)
            paper_result["single_pass"] = {"error": str(e)}

        time.sleep(3)

        # Multi-pass extraction
        print(f"  [2/2] Multi-pass extraction...", flush=True)
        try:
            multi = extract_multi_pass(pdf_path)
            paper_result["multi_pass"] = multi
            for pass_name, data in multi["results"].items():
                items = data["result"]
                if isinstance(items, dict):
                    count = sum(len(v) if isinstance(v, list) else 1 for v in items.values())
                else:
                    count = len(items) if isinstance(items, list) else 1
                print(f"    -> {pass_name}: ~{count} items", flush=True)
            print(f"    -> Total tokens: {multi['total_tokens']}", flush=True)
        except Exception as e:
            print(f"    -> ERROR: {e}", flush=True)
            paper_result["multi_pass"] = {"error": str(e)}

        # Save individual result
        result_path = EXTRACTION_DIR / "raw" / f"{paper_name}.json"
        with open(result_path, "w") as f:
            json.dump(paper_result, f, indent=2)
        print(f"  Saved to {result_path.name}", flush=True)

        all_results.append(paper_result)
        time.sleep(3)

    # Save combined results
    with open(EXTRACTION_DIR / "results" / "all_extractions.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("EXTRACTION EXPERIMENT COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)

    for r in all_results:
        name = r["paper_name"][:50]
        sp = r.get("single_pass", {})
        mp = r.get("multi_pass", {})

        if "error" not in sp:
            sp_claims = len(sp.get("result", {}).get("claims", []))
            sp_tokens = sp.get("usage", {}).get("total_tokens", 0)
        else:
            sp_claims = "ERR"
            sp_tokens = 0

        if "error" not in mp:
            mp_tokens = mp.get("total_tokens", 0)
        else:
            mp_tokens = 0

        print(f"  {name}")
        print(f"    Single-pass: {sp_claims} claims, {sp_tokens} tokens")
        print(f"    Multi-pass: {mp_tokens} tokens total")


if __name__ == "__main__":
    print(f"AskChem Extraction Experiment v1 - {datetime.now().isoformat()}", flush=True)
    run_extraction_experiment()
