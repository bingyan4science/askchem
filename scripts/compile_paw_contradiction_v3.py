"""Compile PAW contradiction detector v3.

v3 improvements over v2:
- More examples of the "same-property disagreement" pattern that v2 missed
- Explicit examples where one claim praises and another criticizes the SAME property
- Examples of conflicting selectivity/target claims
- Kept compatible examples to maintain precision

Run: conda run -n paw python scripts/compile_paw_contradiction_v3.py
"""
import json
import sys
import time
from pathlib import Path

import programasweights as paw

SPEC = """Detect if two chemistry claims CONTRADICT each other about the same material or substance.

Input: "CLAIM_A: {text} CLAIM_B: {text}"
Output one word: "contradicts" or "compatible"

CONTRADICTS means the claims disagree on the SAME specific property or capability:

Input: CLAIM_A: NCA has high cost which limits adoption CLAIM_B: NCA is attractive due to its low cost
Output: contradicts

Input: CLAIM_A: The material shows excellent thermal stability up to 500C CLAIM_B: The material degrades rapidly above 200C
Output: contradicts

Input: CLAIM_A: LMBs offer long cycling lifetime CLAIM_B: LMBs suffer from short cycling lifetime
Output: contradicts

Input: CLAIM_A: The catalyst is highly efficient for oxidative dehydrogenation CLAIM_B: The material is chemically inert and not promising for catalysis
Output: contradicts

Input: CLAIM_A: The sensor selectively detects ONOO- in biological systems CLAIM_B: The sensor shows high selectivity for Cu2+ ions
Output: contradicts

Input: CLAIM_A: The nanoparticles exhibit high coercivity CLAIM_B: The nanoparticles have coercivity of 120 Oe which is relatively low
Output: contradicts

Input: CLAIM_A: Supercapacitors have low energy density compared to batteries CLAIM_B: Supercapacitors have superior energy density
Output: contradicts

Input: CLAIM_A: TMPs are excellent energy storage materials CLAIM_B: Low conductivity and volume expansion limit TMP performance
Output: contradicts

Input: CLAIM_A: The perovskites have very efficient light emission CLAIM_B: Parity-forbidden transitions are an intrinsic limitation of these perovskites
Output: contradicts

COMPATIBLE means claims can both be true — they discuss DIFFERENT properties or different conditions:

Input: CLAIM_A: MOFs have high surface area CLAIM_B: MOFs have tunable pore sizes
Output: compatible

Input: CLAIM_A: Doped MgH2 shows enhanced hydrogen storage CLAIM_B: Undoped MgH2 has poor dehydrogenation ability
Output: compatible

Input: CLAIM_A: LTO is safe and durable but has low energy density CLAIM_B: LTO has high rate capability and excellent cyclic performance
Output: compatible

Input: CLAIM_A: Pt catalysts suffer from high cost CLAIM_B: Pt catalysts suffer from CO poisoning
Output: compatible

Input: CLAIM_A: Aptamers have high affinity and low cost CLAIM_B: Aptamers offer specific molecular recognition and low immunogenicity
Output: compatible

Input: CLAIM_A: ZnO has low toxicity and biocompatibility CLAIM_B: ZnO has high photocatalytic activity
Output: compatible"""


def main():
    print("Compiling PAW contradiction detector v3...")
    print("Spec length: {} chars".format(len(SPEC)))

    for attempt in range(5):
        try:
            result = paw.compile(SPEC, compiler="paw-4b-qwen3-0.6b")
            program_id = result.id if hasattr(result, "id") else str(result)
            print("\nSUCCESS! Program ID: {}".format(program_id))

            print("\nWaiting for artifact to become loadable...")
            fn = None
            last_error = None
            for load_attempt in range(10):
                try:
                    fn = paw.function(program_id, n_gpu_layers=0)
                    break
                except Exception as e:
                    last_error = e
                    wait = 5 * (load_attempt + 1)
                    print("  Load attempt {}/10 failed: {}. Retrying in {}s...".format(
                        load_attempt + 1, e, wait))
                    time.sleep(wait)

            if fn is None:
                print("  Error: could not load compiled program: {}".format(last_error))
                sys.exit(1)

            # Quick sanity check
            print("\nSanity check...")
            sanity = [
                ("CLAIM_A: NCA has high cost CLAIM_B: NCA has low cost", "contradicts"),
                ("CLAIM_A: MOFs have high surface area CLAIM_B: MOFs have tunable pores", "compatible"),
            ]
            for inp, expected in sanity:
                raw = fn(inp).strip().strip("'\"").lower().split()[0]
                ok = "OK" if raw.startswith(expected[:5]) else "WRONG"
                print("  {} expected={} got={} | {}".format(ok, expected, raw, inp[:60]))

            # Full benchmark
            benchmark_path = Path("data/paw_test_set.json")
            if benchmark_path.exists():
                rows = json.loads(benchmark_path.read_text())
                correct = 0
                tp = fp = fn_count = tn = 0
                print("\nFull 20-pair benchmark:")
                print("-" * 100)
                for i, row in enumerate(rows):
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

                    print("{:<6} {:<12} {:<12} {}".format(
                        mark, row["label"], pred, row["subject"][:50]))

                print("-" * 100)
                n = len(rows)
                print("Accuracy: {}/{} = {:.1%}".format(correct, n, correct / n))
                print("Contradicts: TP={}, FN={} (recall={:.1%})".format(
                    tp, fn_count, tp / 10))
                print("Compatible:  TN={}, FP={} (precision={:.1%})".format(
                    tn, fp, tp / (tp + fp) if (tp + fp) else 0))

            print("\nProgram ID for paw_functions.py:")
            print('  CONTRADICTION_PROGRAM_ID = "{}"'.format(program_id))
            return

        except Exception as e:
            if "500" in str(e):
                wait = 30 * (attempt + 1)
                print("  Server error (attempt {}/5), retrying in {}s...".format(
                    attempt + 1, wait))
                time.sleep(wait)
            else:
                print("  Error: {}".format(e))
                sys.exit(1)

    print("Failed after 5 attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
