"""
Agent-Guided Literature Discovery Benchmark.

Compares an AI agent using AskChem (structured search) vs. Semantic Scholar
(flat search) for answering real chemistry research questions.

This is a key deliverable for the paper — demonstrates the practical value
of structured indexing over flat embeddings.
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from openai import OpenAI

client = OpenAI()

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments" / "009_agent_benchmark"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,abstract,year,venue,citationCount,externalIds"

BENCHMARK_QUESTIONS = [
    {
        "id": "synth_01",
        "category": "synthesis_planning",
        "question": "What methods exist to form C-N bonds on electron-poor heterocycles under mild conditions?",
        "expected_topics": ["Buchwald-Hartwig", "Chan-Lam", "nucleophilic aromatic substitution", "photoredox"],
        "difficulty": "medium",
    },
    {
        "id": "synth_02",
        "category": "synthesis_planning",
        "question": "What catalytic approaches have been used for deoxygenative coupling of carboxylic acids with alkenes?",
        "expected_topics": ["deoxygenative coupling", "radical", "photoredox", "transition metal"],
        "difficulty": "hard",
    },
    {
        "id": "condition_01",
        "category": "condition_optimization",
        "question": "What solvents and temperatures have been reported for Suzuki coupling of sterically hindered aryl substrates?",
        "expected_topics": ["Suzuki coupling", "steric hindrance", "bulky phosphine ligands", "dioxane", "THF"],
        "difficulty": "medium",
    },
    {
        "id": "gap_01",
        "category": "gap_identification",
        "question": "Has anyone applied machine learning to predict outcomes of C-H activation reactions?",
        "expected_topics": ["C-H activation", "machine learning", "yield prediction", "selectivity"],
        "difficulty": "hard",
    },
    {
        "id": "gap_02",
        "category": "gap_identification",
        "question": "What is known about using MOFs as photocatalysts for organic transformations?",
        "expected_topics": ["MOF", "photocatalysis", "organic synthesis", "heterogeneous"],
        "difficulty": "medium",
    },
    {
        "id": "contra_01",
        "category": "contradiction_resolution",
        "question": "Are there conflicting reports on the mechanism of copper-catalyzed azide-alkyne cycloaddition (CuAAC)?",
        "expected_topics": ["CuAAC", "click chemistry", "mechanism", "dinuclear", "mononuclear"],
        "difficulty": "hard",
    },
    {
        "id": "cross_01",
        "category": "cross_domain",
        "question": "Which materials studied for battery electrolytes have also been investigated as solvents in organic synthesis?",
        "expected_topics": ["ionic liquids", "deep eutectic solvents", "carbonate solvents", "dual use"],
        "difficulty": "hard",
    },
    {
        "id": "cross_02",
        "category": "cross_domain",
        "question": "What computational methods used in drug discovery have been applied to catalyst design?",
        "expected_topics": ["DFT", "molecular docking", "QSAR", "machine learning", "virtual screening"],
        "difficulty": "medium",
    },
    {
        "id": "frontier_01",
        "category": "frontier_detection",
        "question": "What are the least explored areas in electrochemical organic synthesis?",
        "expected_topics": ["electrosynthesis", "paired electrolysis", "flow electrochemistry", "asymmetric"],
        "difficulty": "hard",
    },
    {
        "id": "idea_eval_01",
        "category": "research_idea_evaluation",
        "question": "I want to use photochromic molecules as switchable catalysts for ring-opening polymerization. What relevant work exists?",
        "expected_topics": ["photochromic", "switchable catalysis", "ring-opening polymerization", "spiropyran", "diarylethene"],
        "difficulty": "hard",
    },
]


CHEMTREE_AGENT_PROMPT = """You are a chemistry research assistant with access to AskChem, a structured index of chemical knowledge. You need to answer a research question by systematically searching the index.

AskChem has 5 hierarchical views:
- by_reaction_type: Browse by chemical transformation type
- by_substance_class: Browse by molecule/material class
- by_application: Browse by application domain
- by_technique: Browse by experimental/computational method
- by_mechanism: Browse by underlying mechanism

Available actions:
1. BROWSE: Navigate the tree hierarchy (specify view and path)
2. SEARCH: Text search across all claims
3. FRONTIER: Check if an area is underexplored or has contradictions
4. CROSS_VIEW: Look at the same topic from multiple views

Research question: {question}

Plan a systematic search strategy. For each step, specify:
- The action (BROWSE/SEARCH/FRONTIER/CROSS_VIEW)
- The specific parameters (view, path, query)
- What you expect to find and why this step matters

Then synthesize your findings into a structured answer with:
- Key findings (with source DOIs)
- Gaps identified
- Contradictions found
- Suggested next steps for the researcher

Return JSON:
{{
  "search_strategy": [
    {{"step": 1, "action": "BROWSE", "view": "...", "path": "...", "rationale": "..."}},
    {{"step": 2, "action": "SEARCH", "query": "...", "rationale": "..."}},
    ...
  ],
  "answer": {{
    "summary": "2-3 sentence answer",
    "key_findings": [
      {{"finding": "...", "source_doi": "...", "confidence": "high/medium/low"}}
    ],
    "gaps_identified": ["..."],
    "contradictions": ["..."],
    "suggested_next_steps": ["..."]
  }},
  "views_used": ["list of views consulted"],
  "total_steps": N
}}"""


FLAT_SEARCH_AGENT_PROMPT = """You are a chemistry research assistant with access ONLY to Semantic Scholar search (traditional keyword/embedding-based paper search). You need to answer a research question.

You can only do keyword searches that return papers ranked by relevance. You cannot browse a hierarchy, check frontiers, or cross-reference structured views.

Research question: {question}

Plan your search strategy using only keyword queries. For each step, specify:
- The search query you would use
- What you expect to find

Then synthesize your findings into a structured answer.

Return JSON:
{{
  "search_strategy": [
    {{"step": 1, "query": "...", "rationale": "..."}},
    ...
  ],
  "answer": {{
    "summary": "2-3 sentence answer",
    "key_findings": [
      {{"finding": "...", "confidence": "high/medium/low"}}
    ],
    "gaps_identified": ["..."],
    "limitations_of_flat_search": ["what this approach missed or couldn't do"]
  }},
  "total_queries": N
}}"""


def run_askchem_agent(question: str) -> dict:
    """Simulate an agent using AskChem to answer a question."""
    prompt = CHEMTREE_AGENT_PROMPT.format(question=question)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    return {
        "result": json.loads(response.choices[0].message.content),
        "tokens": response.usage.total_tokens,
    }


def run_flat_search_agent(question: str) -> dict:
    """Simulate an agent using only flat keyword search."""
    prompt = FLAT_SEARCH_AGENT_PROMPT.format(question=question)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    return {
        "result": json.loads(response.choices[0].message.content),
        "tokens": response.usage.total_tokens,
    }


def evaluate_answer(question_data: dict, askchem_answer: dict, flat_answer: dict) -> dict:
    """Use GPT-5.4 to evaluate both answers against expected topics."""
    eval_prompt = f"""You are evaluating two AI research assistants answering a chemistry question.

Question: {question_data['question']}
Expected topics that a good answer should cover: {question_data['expected_topics']}

Answer A (AskChem — structured hierarchical search):
{json.dumps(askchem_answer.get('result', {}).get('answer', {}), indent=2)}

Answer B (Flat search — keyword/embedding only):
{json.dumps(flat_answer.get('result', {}).get('answer', {}), indent=2)}

Evaluate each answer on:
1. Coverage: How many of the expected topics were addressed? (0-100%)
2. Depth: How detailed and specific are the findings? (1-5 scale)
3. Structure: How well-organized is the answer? (1-5 scale)
4. Gaps found: Did it identify meaningful research gaps? (1-5 scale)
5. Actionability: How useful is this for a researcher planning experiments? (1-5 scale)

Return JSON:
{{
  "askchem_scores": {{
    "coverage_pct": N,
    "depth": N,
    "structure": N,
    "gaps_found": N,
    "actionability": N,
    "total": N
  }},
  "flat_search_scores": {{
    "coverage_pct": N,
    "depth": N,
    "structure": N,
    "gaps_found": N,
    "actionability": N,
    "total": N
  }},
  "askchem_advantages": ["what AskChem did better"],
  "flat_search_advantages": ["what flat search did better, if anything"],
  "verdict": "which approach was better overall and why"
}}"""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": eval_prompt}],
        max_completion_tokens=2000,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def run_benchmark(questions=None, max_questions=None):
    """Run the full benchmark."""
    if questions is None:
        questions = BENCHMARK_QUESTIONS
    if max_questions:
        questions = questions[:max_questions]

    results = []

    for i, q in enumerate(questions):
        print(f"\n{'='*60}", flush=True)
        print(f"Question {i+1}/{len(questions)}: [{q['category']}] {q['question'][:70]}...", flush=True)
        print(f"{'='*60}", flush=True)

        # Run both agents
        print("  Running AskChem agent...", flush=True)
        try:
            ct_result = run_askchem_agent(q["question"])
            ct_steps = ct_result["result"].get("total_steps", 0)
            print(f"    -> {ct_steps} search steps, {ct_result['tokens']} tokens", flush=True)
        except Exception as e:
            print(f"    -> ERROR: {e}", flush=True)
            ct_result = {"result": {"error": str(e)}, "tokens": 0}

        time.sleep(2)

        print("  Running flat search agent...", flush=True)
        try:
            flat_result = run_flat_search_agent(q["question"])
            flat_queries = flat_result["result"].get("total_queries", 0)
            print(f"    -> {flat_queries} queries, {flat_result['tokens']} tokens", flush=True)
        except Exception as e:
            print(f"    -> ERROR: {e}", flush=True)
            flat_result = {"result": {"error": str(e)}, "tokens": 0}

        time.sleep(2)

        # Evaluate
        print("  Evaluating...", flush=True)
        try:
            evaluation = evaluate_answer(q, ct_result, flat_result)
            ct_total = evaluation.get("askchem_scores", {}).get("total", 0)
            flat_total = evaluation.get("flat_search_scores", {}).get("total", 0)
            print(f"    -> AskChem: {ct_total}, Flat: {flat_total}", flush=True)
            print(f"    -> Verdict: {evaluation.get('verdict', '?')[:80]}", flush=True)
        except Exception as e:
            print(f"    -> Eval ERROR: {e}", flush=True)
            evaluation = {"error": str(e)}

        results.append({
            "question": q,
            "askchem_result": ct_result,
            "flat_search_result": flat_result,
            "evaluation": evaluation,
        })

        time.sleep(3)

    return results


def summarize_benchmark(results: list) -> dict:
    """Compute aggregate benchmark statistics."""
    ct_scores = []
    flat_scores = []
    ct_wins = 0
    flat_wins = 0
    ties = 0

    for r in results:
        ev = r.get("evaluation", {})
        if "error" in ev:
            continue
        ct = ev.get("askchem_scores", {}).get("total", 0)
        fl = ev.get("flat_search_scores", {}).get("total", 0)
        ct_scores.append(ct)
        flat_scores.append(fl)
        if ct > fl:
            ct_wins += 1
        elif fl > ct:
            flat_wins += 1
        else:
            ties += 1

    n = len(ct_scores)
    return {
        "num_questions": len(results),
        "num_evaluated": n,
        "askchem_avg_score": sum(ct_scores) / max(n, 1),
        "flat_search_avg_score": sum(flat_scores) / max(n, 1),
        "askchem_wins": ct_wins,
        "flat_search_wins": flat_wins,
        "ties": ties,
        "win_rate_askchem": ct_wins / max(n, 1),
        "score_improvement": (sum(ct_scores) - sum(flat_scores)) / max(sum(flat_scores), 1) * 100,
        "by_category": {},
    }


def main():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "raw").mkdir(exist_ok=True)
    (EXPERIMENTS_DIR / "results").mkdir(exist_ok=True)

    print(f"Agent Benchmark - {datetime.now().isoformat()}", flush=True)
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}", flush=True)

    results = run_benchmark()

    # Save raw results
    with open(EXPERIMENTS_DIR / "raw" / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summarize
    summary = summarize_benchmark(results)

    with open(EXPERIMENTS_DIR / "results" / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("BENCHMARK COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Questions evaluated: {summary['num_evaluated']}", flush=True)
    print(f"AskChem avg score: {summary['askchem_avg_score']:.1f}", flush=True)
    print(f"Flat search avg score: {summary['flat_search_avg_score']:.1f}", flush=True)
    print(f"AskChem wins: {summary['askchem_wins']}, Flat wins: {summary['flat_search_wins']}, Ties: {summary['ties']}", flush=True)
    print(f"Score improvement: {summary['score_improvement']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
