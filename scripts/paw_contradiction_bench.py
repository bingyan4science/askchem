"""Run the 20-pair contradiction benchmark against the current PAW program."""
import json
import time
from pathlib import Path

import programasweights as paw

PROGRAM_ID = "e1ae405a57318a3fb9db"
BENCHMARK = Path("data/paw_test_set.json")

rows = json.loads(BENCHMARK.read_text())
fn = paw.function(PROGRAM_ID, n_gpu_layers=0)

# warm up
fn("CLAIM_A: test CLAIM_B: test")

correct = 0
times = []
results = []

print("{:<6} {:<12} {:<12} {:<40} {}".format(
    "#", "Expected", "Got", "Subject", "Time"))
print("-" * 110)

for i, row in enumerate(rows):
    inp = "CLAIM_A: {} CLAIM_B: {}".format(row["quote_1"], row["quote_2"])
    t0 = time.perf_counter()
    raw = fn(inp).strip().strip("'\"").lower()
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)

    pred = raw.split()[0] if raw else "compatible"
    # normalize
    if pred.startswith("contradict"):
        pred = "contradicts"
    elif pred.startswith("compat"):
        pred = "compatible"
    elif pred.startswith("unclear"):
        pred = "unclear"

    match = pred == row["label"]
    correct += match
    mark = "OK" if match else "MISS"

    subj = row["subject"][:38]
    print("{:<6} {:<12} {:<12} {:<40} {:>7.0f}ms  {}".format(
        mark, row["label"], pred, subj, elapsed,
        raw[:50] if not match else ""))
    results.append({
        "idx": i, "subject": row["subject"], "expected": row["label"],
        "predicted": pred, "raw": raw, "match": match, "ms": round(elapsed, 1)
    })

print("-" * 110)
print()
n = len(rows)
print("Accuracy: {}/{} = {:.1%}".format(correct, n, correct / n))

n_contra = sum(1 for r in rows if r["label"] == "contradicts")
n_compat = sum(1 for r in rows if r["label"] == "compatible")
tp = sum(1 for r in results if r["expected"] == "contradicts" and r["predicted"] == "contradicts")
fn_count = sum(1 for r in results if r["expected"] == "contradicts" and r["predicted"] != "contradicts")
fp = sum(1 for r in results if r["expected"] == "compatible" and r["predicted"] == "contradicts")
tn = sum(1 for r in results if r["expected"] == "compatible" and r["predicted"] != "contradicts")

print()
print("Contradicts: TP={}, FN={} (recall={:.1%})".format(tp, fn_count, tp / n_contra if n_contra else 0))
print("Compatible:  TN={}, FP={} (precision={:.1%})".format(tn, fp, tp / (tp + fp) if (tp + fp) else 0))
print()
print("Avg latency: {:.0f}ms | Min: {:.0f}ms | Max: {:.0f}ms".format(
    sum(times) / len(times), min(times), max(times)))
