#!/usr/bin/env python3
"""Compile improved PAW contradiction detector (v2).

Key improvements over v1:
- Explicit examples from real chemistry contradictions
- Clarifies that "advantages vs challenges" IS a contradiction when
  they disagree on the SAME specific property (e.g., "high cost" vs "low cost")
- Teaches the distinction between complementary claims and contradictions

Run: /opt/homebrew/bin/python3.10 scripts/compile_paw_contradiction_v2.py
"""
import sys
import time
from pathlib import Path

import programasweights as paw

SPEC = """Detect if two scientific claims about the same material DISAGREE on a specific property.

Input: "CLAIM_A: {text} CLAIM_B: {text}"
Output one word: "contradicts" or "compatible"

CONTRADICTS means they assert opposite things about the SAME property:

Examples:
Input: CLAIM_A: NCA has high cost which limits adoption CLAIM_B: NCA is attractive due to its low cost
Output: contradicts

Input: CLAIM_A: The material shows excellent thermal stability up to 500C CLAIM_B: The material degrades rapidly above 200C
Output: contradicts

Input: CLAIM_A: LMBs offer long cycling lifetime CLAIM_B: LMBs suffer from short cycling lifetime
Output: contradicts

Input: CLAIM_A: TMOs have good electrochemical activity CLAIM_B: TMOs are limited by poor electrochemical activity
Output: contradicts

Input: CLAIM_A: Ionic liquids have high conductivity CLAIM_B: RTILs suffer from poor conductivity
Output: contradicts

COMPATIBLE means they can both be true — they describe different properties or different conditions:

Examples:
Input: CLAIM_A: MgH2 shows enhanced hydrogen storage after doping CLAIM_B: Undoped MgH2 has poor dehydrogenation ability
Output: compatible

Input: CLAIM_A: LTO is safe and durable but has low energy density CLAIM_B: LTO has advantages like safety and long cycle life
Output: compatible

Input: CLAIM_A: MOFs have high surface area CLAIM_B: MOFs have poor chemical stability
Output: compatible

Input: CLAIM_A: Graphene has excellent conductivity CLAIM_B: Graphene synthesis is expensive
Output: compatible"""


def main():
    print("Compiling PAW contradiction detector v2...")
    print(f"Spec length: {len(SPEC)} chars")

    for attempt in range(5):
        try:
            result = paw.compile(SPEC, compiler="paw-4b-qwen3-0.6b",
                                 name="Chemistry Contradiction Detector v2",
                                 tags=["contradicts", "chemistry", "nli"])
            program_id = result.id if hasattr(result, "id") else str(result)
            print(f"\nSUCCESS! Program ID: {program_id}")

            # Freshly compiled artifacts can take a few seconds to become loadable.
            print("\nWaiting for PAW artifact to become loadable...")
            fn = None
            last_error = None
            for load_attempt in range(8):
                try:
                    fn = paw.function(program_id, n_gpu_layers=0)
                    break
                except Exception as e:
                    last_error = e
                    wait = 5 * (load_attempt + 1)
                    print(
                        f"  Load attempt {load_attempt+1}/8 failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)

            if fn is None:
                print(f"  Error: could not load compiled program: {last_error}")
                sys.exit(1)

            # Test
            print("\nTesting v2...")

            tests = [
                ("CLAIM_A: NCA has high cost CLAIM_B: NCA is low cost", "contradicts"),
                ("CLAIM_A: TMPs are excellent storage materials CLAIM_B: low conductivity limits TMPs", "contradicts"),
                ("CLAIM_A: LMBs have long lifetime CLAIM_B: LMBs have short cycling lifetime", "contradicts"),
                ("CLAIM_A: MOFs have high surface area CLAIM_B: MOFs have tunable pore sizes", "compatible"),
                ("CLAIM_A: Doped MgH2 is enhanced CLAIM_B: Raw MgH2 has poor performance", "compatible"),
            ]

            correct = 0
            for inp, expected in tests:
                raw = fn(inp).strip().strip("'\"").lower().split()[0]
                ok = raw == expected
                correct += ok
                print(f"  {'OK' if ok else 'WRONG':5s} expected={expected:12s} got={raw:12s} | {inp[:70]}")

            print(f"\nTest accuracy: {correct}/{len(tests)}")
            benchmark_path = Path("data/paw_test_set.json")
            if benchmark_path.exists():
                try:
                    import json

                    rows = json.loads(benchmark_path.read_text())
                    bench_correct = 0
                    for row in rows:
                        raw = fn(
                            f"CLAIM_A: {row['quote_1']} CLAIM_B: {row['quote_2']}"
                        ).strip().strip("'\"").lower()
                        pred = raw.split()[0] if raw else "compatible"
                        bench_correct += pred == row["label"]
                    print(
                        f"Benchmark on {benchmark_path}: "
                        f"{bench_correct}/{len(rows)} = {bench_correct/len(rows):.3f}"
                    )
                except Exception as e:
                    print(f"Benchmark skipped: {e}")
            print(f"\nUpdate paw_functions.py with:\n  CONTRADICTION_PROGRAM_ID = \"{program_id}\"")
            return

        except Exception as e:
            if "500" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Server error (attempt {attempt+1}/5), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                sys.exit(1)

    print("Failed after 5 attempts. PAW GPU workers may be offline.")
    print("Check: https://programasweights.com/api/v1/health")
    sys.exit(1)


if __name__ == "__main__":
    main()
