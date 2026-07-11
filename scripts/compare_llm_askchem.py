#!/usr/bin/env python3
"""
Compare "LLM alone" vs "LLM + AskChem" on chemistry questions.

Demonstrates that the same LLM produces dramatically better answers
when it has access to AskChem's structured, citation-grounded claims.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/compare_llm_askchem.py

Output is saved to scripts/comparison_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai requests")
    sys.exit(1)

ASKCHEM_API = os.environ.get("ASKCHEM_API", "https://askchem.org/api")
OUTPUT_FILE = Path(__file__).parent / "comparison_results.json"

QUERIES = [
    {
        "id": "q1_coupling_conditions",
        "question": (
            "What catalysts and conditions have been used for C-N coupling "
            "of heteroaryl chlorides? Give specific catalysts, ligands, "
            "solvents, temperatures, and yields from the literature."
        ),
        "askchem_params": {
            "q": "C-N bond formation coupling catalyst conditions",
            "claim_type": "reaction",
            "limit": 50,
        },
        "capability": "Condition aggregation across papers",
    },
    {
        "id": "q2_perovskite_evolution",
        "question": (
            "How has the scientific understanding of perovskite degradation "
            "mechanisms evolved over time? Describe the key shifts in the "
            "field's understanding, citing specific findings and years."
        ),
        "askchem_params": {
            "q": "perovskite degradation mechanism",
            "limit": 100,
        },
        "capability": "Temporal evolution tracking",
    },
    {
        "id": "q3_water_organocatalysis",
        "question": (
            "Does water help or hurt organocatalytic reactions? What does "
            "the literature say about the role of water in organocatalysis? "
            "Are there contradictory findings?"
        ),
        "askchem_params": {
            "q": "organocatalysis water",
            "limit": 50,
        },
        "capability": "Contradiction surfacing",
    },
]


def query_askchem(params: dict) -> list[dict]:
    """Query AskChem search API and return claims."""
    url = f"{ASKCHEM_API}/search"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", []), data.get("total", 0)


def format_claims_for_llm(claims: list[dict], max_claims: int = 40) -> str:
    """Format claims into a text block for LLM consumption."""
    lines = []
    for i, c in enumerate(claims[:max_claims]):
        quote = c.get("verbatim_quote", "") or ""
        doi = c.get("source_doi", "")
        ctype = c.get("claim_type", "")
        title = c.get("source_paper_title", "")
        year = ""
        if "source_year" in c:
            year = str(c["source_year"])

        line = f"[{i+1}] [{ctype}] {quote}"
        if title:
            line += f" — Paper: {title}"
        if year:
            line += f" ({year})"
        line += f" [DOI: {doi}]"
        lines.append(line)
    return "\n\n".join(lines)


def llm_alone(client: OpenAI, question: str, model: str = "gpt-4o") -> str:
    """Ask the LLM the question directly, with no external knowledge."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a chemistry expert. Answer the question concisely "
                    "but with specific details: catalyst names, conditions, "
                    "yields, DOIs where possible. If you're unsure about a "
                    "specific detail, say so."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return resp.choices[0].message.content


def llm_with_askchem(
    client: OpenAI,
    question: str,
    claims_text: str,
    total_claims: int,
    model: str = "gpt-4o",
) -> str:
    """Ask the LLM to synthesize an answer from AskChem claims."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a chemistry expert. Synthesize a concise, "
                    "authoritative answer to the question using ONLY the "
                    "research claims provided below. Cite specific DOIs in "
                    "parentheses. Highlight any contradictions or nuances "
                    "between different papers. Do not add information beyond "
                    "what the claims support."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"The following {total_claims} research claims were retrieved "
                    f"from the AskChem index (showing top {min(total_claims, 40)}):\n\n"
                    f"{claims_text}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return resp.choices[0].message.content


def run_comparison(client: OpenAI, query: dict) -> dict:
    """Run a single comparison: LLM alone vs LLM + AskChem."""
    print(f"\n{'='*60}")
    print(f"Query: {query['id']}")
    print(f"Capability: {query['capability']}")
    print(f"Question: {query['question'][:80]}...")
    print(f"{'='*60}")

    # Step 1: LLM alone
    print("\n[1/3] Asking LLM directly...")
    t0 = time.time()
    answer_alone = llm_alone(client, query["question"])
    t_alone = time.time() - t0
    print(f"  Done ({t_alone:.1f}s, {len(answer_alone)} chars)")

    # Step 2: Query AskChem
    print("[2/3] Querying AskChem API...")
    t0 = time.time()
    claims, total = query_askchem(query["askchem_params"])
    t_askchem = time.time() - t0
    claims_text = format_claims_for_llm(claims)
    print(f"  Got {total} claims ({t_askchem:.1f}s)")

    # Step 3: LLM + AskChem
    print("[3/3] Asking LLM with AskChem claims...")
    t0 = time.time()
    answer_grounded = llm_with_askchem(
        client, query["question"], claims_text, total
    )
    t_grounded = time.time() - t0
    print(f"  Done ({t_grounded:.1f}s, {len(answer_grounded)} chars)")

    result = {
        "id": query["id"],
        "question": query["question"],
        "capability": query["capability"],
        "llm_alone": {
            "answer": answer_alone,
            "time_s": round(t_alone, 1),
        },
        "askchem": {
            "total_claims": total,
            "claims_shown": len(claims),
            "query_time_s": round(t_askchem, 1),
            "example_claims": [
                {
                    "type": c.get("claim_type", ""),
                    "quote": (c.get("verbatim_quote", "") or "")[:200],
                    "doi": c.get("source_doi", ""),
                    "paper": c.get("source_paper_title", ""),
                }
                for c in claims[:5]
            ],
        },
        "llm_plus_askchem": {
            "answer": answer_grounded,
            "time_s": round(t_grounded, 1),
        },
    }

    # Print comparison
    print(f"\n--- LLM ALONE ---")
    print(answer_alone[:500])
    print(f"\n--- LLM + ASKCHEM ({total} claims) ---")
    print(answer_grounded[:500])

    return result


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        print("Usage: OPENAI_API_KEY=sk-... python scripts/compare_llm_askchem.py")
        sys.exit(1)

    client = OpenAI()
    results = []

    for query in QUERIES:
        result = run_comparison(client, query)
        results.append(result)

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
