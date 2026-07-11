"""Test PAW functions for AskChem before integration.

Compiles three domain-specific programs and validates their outputs
against chemistry-domain test cases.
"""
import os
import sys
import time

os.environ["GGML_NO_METAL"] = "1"

import programasweights as paw

INTENT_SPEC = """Classify a chemistry research search query into exactly one category.

Output ONLY one of these words: author, substance, method, concept, paper

Rules:
- "author" if the input is a person's name (first/last name, e.g. "John Smith", "Coley")
- "substance" if the input names a chemical, material, molecule, element, or compound (e.g. "benzene", "TiO2", "palladium nanoparticles")
- "method" if the input names a technique, algorithm, tool, or experimental method (e.g. "DFT", "L-BFGS", "mass spectrometry", "Suzuki coupling")
- "concept" if the input is a general question or abstract topic (e.g. "what is catalysis", "reaction mechanisms", "thermodynamics")
- "paper" if the input references a specific publication, journal, or DOI (e.g. "Nature 2024", "doi:10.1021/...", "JACS review")

Examples:
Input: Connor Coley
Output: author

Input: Bing Yan
Output: author

Input: benzene oxidation
Output: substance

Input: TiO2 photocatalysis
Output: substance

Input: palladium nanoparticles
Output: substance

Input: L-BFGS
Output: method

Input: density functional theory
Output: method

Input: Suzuki coupling
Output: method

Input: what is catalysis
Output: concept

Input: reaction kinetics
Output: concept

Input: how do enzymes work
Output: concept

Input: Nature 2024 palladium review
Output: paper

Input: JACS 2023
Output: paper
"""

NORMALIZER_SPEC = """Extract the core chemistry search terms from a natural language query.

Remove question framing words (what is, how does, why do, can you explain, etc.), filler words, and conversational phrasing. Keep only the essential chemistry/science terms that should be searched.

If the input contains a well-known chemistry abbreviation, expand it AND keep the abbreviation.

Output ONLY the cleaned search terms, nothing else. Do not add explanation.

Examples:
Input: what is the mechanism of L-BFGS?
Output: L-BFGS mechanism

Input: how does Suzuki coupling work
Output: Suzuki coupling

Input: can you explain DFT calculations for TiO2
Output: DFT density functional theory TiO2

Input: what are the applications of palladium catalysis in organic synthesis
Output: palladium catalysis organic synthesis

Input: why is benzene stable
Output: benzene stability

Input: tell me about metal oxide nanoparticles
Output: metal oxide nanoparticles

Input: machine learning for drug discovery
Output: machine learning drug discovery

Input: L-BFGS
Output: L-BFGS
"""

RELEVANCE_SPEC = """Determine if a chemistry research claim is relevant to a search query.

The input format is:
QUERY: <search query>
CLAIM: <claim text>

Output ONLY the word "relevant" or "not_relevant".

A claim is "relevant" if it directly discusses, mentions, or is closely related to the topic in the query. It is "not_relevant" if the connection is superficial, tangential, or the claim is about a completely different topic.

Examples:
Input: QUERY: Suzuki coupling CLAIM: Pd-catalyzed Suzuki-Miyaura coupling of aryl bromides with boronic acids achieved 95% yield at room temperature.
Output: relevant

Input: QUERY: Suzuki coupling CLAIM: The thermal stability of polyethylene was measured using TGA under nitrogen atmosphere.
Output: not_relevant

Input: QUERY: machine learning CLAIM: A graph neural network was trained to predict molecular properties with 0.95 R-squared.
Output: relevant

Input: QUERY: machine learning CLAIM: The pH of the solution was adjusted to 7.4 using phosphate buffer.
Output: not_relevant

Input: QUERY: TiO2 photocatalysis CLAIM: Titanium dioxide nanoparticles showed enhanced photocatalytic degradation of methylene blue under UV light.
Output: relevant

Input: QUERY: TiO2 photocatalysis CLAIM: Gold nanoparticles were synthesized via citrate reduction method.
Output: not_relevant
"""

def compile_program(name, spec):
    print(f"\n{'='*60}")
    print(f"Compiling: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = paw.compile(spec, compiler="paw-4b-qwen3-0.6b")
    elapsed = time.time() - t0
    print(f"  Compiled in {elapsed:.1f}s")
    print(f"  Program ID: {result.id}")
    print(f"  Status: {result.status}")
    return result.id

def run_paw_program_tests(name, program_id, test_cases):
    """Run test cases and report pass/fail."""
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({program_id})")
    print(f"{'='*60}")

    fn = paw.function(program_id, n_gpu_layers=0)

    passed = 0
    failed = 0
    for inp, expected in test_cases:
        t0 = time.time()
        result = fn(inp).strip().lower()
        elapsed = time.time() - t0

        expected_lower = expected.lower()
        ok = result == expected_lower
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] Input: {inp!r}")
        print(f"         Expected: {expected!r}, Got: {result!r} ({elapsed:.2f}s)")

    print(f"\n  Results: {passed}/{passed+failed} passed")
    return passed, failed

def main():
    # --- Compile all three programs ---
    intent_id = compile_program("Query Intent Classifier", INTENT_SPEC)
    normalizer_id = compile_program("Query Normalizer", NORMALIZER_SPEC)
    relevance_id = compile_program("Relevance Classifier", RELEVANCE_SPEC)

    print(f"\n\nCompiled program IDs:")
    print(f"  Intent:     {intent_id}")
    print(f"  Normalizer: {normalizer_id}")
    print(f"  Relevance:  {relevance_id}")

    # --- Test intent classifier ---
    intent_tests = [
        ("Connor Coley", "author"),
        ("Bing Yan", "author"),
        ("benzene", "substance"),
        ("palladium nanoparticles", "substance"),
        ("L-BFGS", "method"),
        ("Suzuki coupling", "method"),
        ("mass spectrometry", "method"),
        ("what is catalysis", "concept"),
        ("how do enzymes work", "concept"),
        ("reaction kinetics", "concept"),
        ("Nature 2024 palladium review", "paper"),
        ("JACS 2023", "paper"),
    ]
    ip, if_ = run_paw_program_tests("Intent Classifier", intent_id, intent_tests)

    # --- Test normalizer ---
    normalizer_tests = [
        ("what is the mechanism of L-BFGS?", "l-bfgs mechanism"),
        ("how does Suzuki coupling work", "suzuki coupling"),
        ("tell me about metal oxide nanoparticles", "metal oxide nanoparticles"),
        ("L-BFGS", "l-bfgs"),
    ]
    np_, nf = run_paw_program_tests("Query Normalizer", normalizer_id, normalizer_tests)

    # --- Test relevance classifier ---
    relevance_tests = [
        ("QUERY: Suzuki coupling CLAIM: Pd-catalyzed Suzuki-Miyaura coupling of aryl bromides with boronic acids achieved 95% yield.", "relevant"),
        ("QUERY: Suzuki coupling CLAIM: The thermal stability of polyethylene was measured using TGA.", "not_relevant"),
        ("QUERY: machine learning CLAIM: A graph neural network predicted molecular properties with 0.95 R-squared.", "relevant"),
        ("QUERY: machine learning CLAIM: The pH was adjusted to 7.4 using phosphate buffer.", "not_relevant"),
        ("QUERY: TiO2 CLAIM: Titanium dioxide nanoparticles showed photocatalytic degradation of methylene blue.", "relevant"),
        ("QUERY: TiO2 CLAIM: Gold nanoparticles were synthesized via citrate reduction.", "not_relevant"),
    ]
    rp, rf = run_paw_program_tests("Relevance Classifier", relevance_id, relevance_tests)

    # --- Summary ---
    total_pass = ip + np_ + rp
    total_fail = if_ + nf + rf
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_pass}/{total_pass+total_fail} tests passed")
    print(f"{'='*60}")

    print(f"\nProgram IDs for integration:")
    print(f'INTENT_PROGRAM_ID = "{intent_id}"')
    print(f'NORMALIZER_PROGRAM_ID = "{normalizer_id}"')
    print(f'RELEVANCE_PROGRAM_ID = "{relevance_id}"')

    return 0 if total_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
