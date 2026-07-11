"""Compile PAW contradiction detector v4.

v4 strategy: minimal spec to maximize context budget for long real-world quotes.
The v3 spec was 2506 chars — leaving little room for two long chemistry quotes
in a 2048 token context window. v4 uses a much shorter spec.

Run: conda run -n paw python scripts/compile_paw_contradiction_v4.py
"""
import json
import sys
import time
from pathlib import Path

import programasweights as paw

SPECS = {
    "v4a": """Do these two claims about the same material CONTRADICT each other?
Output ONLY: contradicts or compatible

contradicts = they say OPPOSITE things about the SAME property (e.g. high cost vs low cost, stable vs unstable, efficient vs inefficient)
compatible = they discuss different properties, or agree

Input: CLAIM_A: X has high cost CLAIM_B: X has low cost
Output: contradicts

Input: CLAIM_A: X is thermally stable CLAIM_B: X degrades at low temperature
Output: contradicts

Input: CLAIM_A: X has excellent catalytic activity CLAIM_B: X is chemically inert
Output: contradicts

Input: CLAIM_A: X has high surface area CLAIM_B: X has tunable pore sizes
Output: compatible

Input: CLAIM_A: X is safe but has low energy density CLAIM_B: X has excellent cycle life
Output: compatible""",

    "v4b": """Two chemistry claims about the same material. Do they DISAGREE?
Output: contradicts or compatible

RULE: If claim A says a property is GOOD and claim B says the SAME property is BAD, output "contradicts". Otherwise output "compatible".

Input: CLAIM_A: High cost limits adoption CLAIM_B: Low cost makes it attractive
Output: contradicts

Input: CLAIM_A: Excellent energy storage CLAIM_B: Poor conductivity limits storage performance
Output: contradicts

Input: CLAIM_A: Efficient catalyst CLAIM_B: Chemically inert, not suitable for catalysis
Output: contradicts

Input: CLAIM_A: Long cycling lifetime CLAIM_B: Short cycling lifetime
Output: contradicts

Input: CLAIM_A: Safe and durable CLAIM_B: High rate capability
Output: compatible

Input: CLAIM_A: High surface area CLAIM_B: Poor chemical stability
Output: compatible

Input: CLAIM_A: Low toxicity CLAIM_B: High photocatalytic activity
Output: compatible""",

    "v4c": """Detect contradiction between two chemistry claims about the same subject.
Return ONLY one word: contradicts or compatible

contradicts: opposite assertions about the same property
compatible: different properties or same assessment

Input: CLAIM_A: The material has high cost CLAIM_B: The material has low cost
Output: contradicts

Input: CLAIM_A: Shows excellent storage performance CLAIM_B: Poor conductivity limits storage performance
Output: contradicts

Input: CLAIM_A: Highly efficient catalyst for the reaction CLAIM_B: Chemically inert and not promising for catalysis
Output: contradicts

Input: CLAIM_A: Selectively detects analyte X CLAIM_B: Shows high selectivity for analyte Y
Output: contradicts

Input: CLAIM_A: The material has high coercivity CLAIM_B: The material has low coercivity
Output: contradicts

Input: CLAIM_A: Efficient light emission CLAIM_B: Transitions are forbidden limiting emission
Output: contradicts

Input: CLAIM_A: Has high surface area CLAIM_B: Has tunable pore sizes
Output: compatible

Input: CLAIM_A: Doped form shows enhanced storage CLAIM_B: Undoped form has poor performance
Output: compatible

Input: CLAIM_A: Suffers from high cost CLAIM_B: Also suffers from CO poisoning
Output: compatible""",
}


def evaluate(fn, rows):
    correct = 0
    tp = fp = fn_count = tn = 0
    for row in rows:
        inp = "CLAIM_A: {} CLAIM_B: {}".format(row["quote_1"], row["quote_2"])
        raw = fn(inp).strip().strip("'\"").lower()
        pred = raw.split()[0] if raw else "compatible"
        if pred.startswith("contradict"):
            pred = "contradicts"
        elif pred.startswith("compat"):
            pred = "compatible"
        else:
            pred = "compatible"

        match = pred == row["label"]
        correct += match
        mark = "OK" if match else "MISS"

        if row["label"] == "contradicts":
            if pred == "contradicts":
                tp += 1
            else:
                fn_count += 1
        else:
            if pred == "contradicts":
                fp += 1
            else:
                tn += 1

        print("  {:<6} {:<12} {:<12} {}".format(
            mark, row["label"], pred, row["subject"][:45]))

    n = len(rows)
    recall = tp / 10 if tp + fn_count > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print("  Accuracy: {}/{} = {:.1%} | Recall: {:.1%} | Precision: {:.1%}".format(
        correct, n, correct / n, recall, precision))
    return correct, tp, fn_count, tn, fp


def main():
    benchmark_path = Path("data/paw_test_set.json")
    rows = json.loads(benchmark_path.read_text())

    best_id = None
    best_score = 0

    for name, spec in SPECS.items():
        print("\n{'='*60}")
        print("Compiling {} (spec length: {} chars)...".format(name, len(spec)))

        try:
            result = paw.compile(spec, compiler="paw-4b-qwen3-0.6b")
            program_id = result.id if hasattr(result, "id") else str(result)
            print("Program ID: {}".format(program_id))
        except Exception as e:
            print("Compile failed: {}".format(e))
            continue

        fn = None
        for load_attempt in range(10):
            try:
                fn = paw.function(program_id, n_gpu_layers=0)
                break
            except Exception as e:
                wait = 5 * (load_attempt + 1)
                print("  Load attempt {}/10 failed. Retrying in {}s...".format(
                    load_attempt + 1, wait))
                time.sleep(wait)

        if fn is None:
            print("  Could not load {}".format(name))
            continue

        fn("CLAIM_A: test CLAIM_B: test")  # warm up

        print("\nBenchmark results for {}:".format(name))
        correct, tp, fn_count, tn, fp = evaluate(fn, rows)

        if correct > best_score:
            best_score = correct
            best_id = program_id
            best_name = name

    print("\n" + "=" * 60)
    if best_id:
        print("Best: {} with {}/20".format(best_name, best_score))
        print('CONTRADICTION_PROGRAM_ID = "{}"'.format(best_id))
    else:
        print("All compilations failed.")


if __name__ == "__main__":
    main()
