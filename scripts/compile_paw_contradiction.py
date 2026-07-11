#!/usr/bin/env python3
"""One-time compilation of a PAW program for contradiction detection.

Produces a program ID to store in paw_functions.py.
"""
import programasweights as paw

SPEC = """Given two scientific claims from chemistry papers about the same subject,
determine if they contradict each other.

Input format: "CLAIM_A: {text} CLAIM_B: {text}"
Output one word: "contradicts", "compatible", or "unclear"

- "contradicts": the claims make opposing assertions about the same thing
- "compatible": the claims agree or discuss different aspects
- "unclear": not enough information to determine"""

print("Compiling PAW contradiction program...")
result = paw.compile(SPEC, compiler="paw-4b-qwen3-0.6b")
program_id = result.id if hasattr(result, 'id') else str(result)
print(f"Program ID: {program_id}")

print("\nTesting...")
fn = paw.function(program_id, n_gpu_layers=0)

tests = [
    ('CLAIM_A: MOFs retain crystallinity after water exposure for 24 hours '
     'CLAIM_B: MOFs lose structural integrity when exposed to moisture',
     "contradicts"),
    ('CLAIM_A: Suzuki coupling achieves 95% yield with Pd catalyst '
     'CLAIM_B: Suzuki coupling achieves 40% yield with Pd catalyst',
     "contradicts"),
    ('CLAIM_A: TiO2 is an effective photocatalyst for water splitting '
     'CLAIM_B: TiO2 nanoparticles show strong UV absorption',
     "compatible"),
]

for inp, expected in tests:
    result = fn(inp).strip().lower()
    status = "OK" if result == expected else f"UNEXPECTED (expected {expected})"
    print(f"  {result:12s} [{status}] {inp[:80]}...")

print(f"\nAdd to paw_functions.py:\n  CONTRADICTION_PROGRAM_ID = \"{program_id}\"")
