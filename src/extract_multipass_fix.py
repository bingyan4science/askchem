"""
Re-run multi-pass extraction with fixed prompts (added 'json' keyword).
Updates existing result files with multi-pass data.
"""

import json
import base64
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

PAPERS_DIR = Path(__file__).parent.parent / "data" / "papers"
EXTRACTION_DIR = Path(__file__).parent.parent / "experiments" / "002_extraction_v1"

client = OpenAI()

REACTIONS_PROMPT = """You are a chemistry expert. Extract ALL chemical reactions described in this paper.

For each reaction, provide a structured JSON entry with these fields:
- reaction_id: sequential number
- reaction_type: classification (e.g., "Suzuki coupling", "C-H activation")
- reactants: list of {name, smiles (if possible), role}
- products: list of {name, smiles (if possible), role}
- conditions: {catalyst, ligand, solvent, temperature, time, atmosphere, additives, other}
- outcomes: {yield_percent, ee_percent, selectivity, conversion_percent, turnover_number}
- is_novel: boolean
- is_key_result: boolean
- scope_entry: boolean
- verbatim_quote: exact text from paper
- location_in_paper: where in paper

Return a JSON object with a "reactions" key containing the list. Be exhaustive."""

PROPERTIES_PROMPT = """You are a chemistry expert. Extract ALL measured/computed properties and characterization data from this paper.

For each property, provide a JSON entry with:
- property_id: sequential number
- subject: what molecule/material
- subject_smiles: SMILES if applicable
- property_name: e.g., melting point, BET surface area, IC50
- property_category: physical/chemical/biological/spectroscopic/electrochemical/computational
- value: numerical value
- unit: unit of measurement
- conditions: measurement conditions
- measurement_method: instrument/technique
- is_computed: boolean
- verbatim_quote: exact text
- location_in_paper: where in paper

Return a JSON object with a "properties" key containing the list. Be exhaustive."""

MECHANISMS_PROMPT = """You are a chemistry expert. Extract ALL mechanistic proposals and explanations from this paper.

For each mechanism, provide a JSON entry with:
- mechanism_id: sequential number
- process_described: what this mechanism explains
- mechanism_type: catalytic cycle/reaction pathway/degradation/binding/electron transfer/other
- steps: list of step descriptions
- key_intermediates: list of intermediate species
- rate_determining_step: if identified
- evidence: list of {type, description}
- confidence: established/proposed/speculative
- verbatim_quote: exact text
- location_in_paper: where in paper

Return a JSON object with a "mechanisms" key containing the list."""


def encode_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def run_pass(pdf_path, prompt, pass_name):
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
        max_completion_tokens=16000,
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


def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Re-running multi-pass on {len(pdfs)} papers", flush=True)

    for pdf_path in pdfs:
        name = pdf_path.stem
        result_file = EXTRACTION_DIR / "raw" / f"{name}.json"

        print(f"\n{'='*60}", flush=True)
        print(f"{name[:60]}", flush=True)

        # Load existing results
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
        else:
            data = {"paper_name": name, "pdf_path": str(pdf_path)}

        multi_results = {}
        total_tokens = 0

        for pass_name, prompt in [("reactions", REACTIONS_PROMPT), ("properties", PROPERTIES_PROMPT), ("mechanisms", MECHANISMS_PROMPT)]:
            print(f"  {pass_name}...", flush=True)
            try:
                result = run_pass(pdf_path, prompt, pass_name)
                multi_results[pass_name] = result
                items = result["result"].get(pass_name, [])
                count = len(items) if isinstance(items, list) else 0
                tokens = result["usage"]["total_tokens"]
                total_tokens += tokens
                print(f"    -> {count} items, {tokens} tokens", flush=True)
            except Exception as e:
                print(f"    -> ERROR: {e}", flush=True)
                multi_results[pass_name] = {"error": str(e)}
            time.sleep(2)

        data["multi_pass"] = {
            "method": "multi_pass",
            "model": "gpt-5.4",
            "results": multi_results,
            "total_tokens": total_tokens,
        }

        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Total: {total_tokens} tokens", flush=True)
        time.sleep(3)

    print(f"\n{'='*60}", flush=True)
    print("MULTI-PASS RE-RUN COMPLETE", flush=True)


if __name__ == "__main__":
    print(f"Multi-pass extraction fix - {datetime.now().isoformat()}", flush=True)
    main()
