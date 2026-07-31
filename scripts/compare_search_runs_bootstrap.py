#!/usr/bin/env python3
"""Paired-bootstrap comparison of two eval_metrics scored search runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("ndcg@10", "ndcg@20", "mrr@20", "recall@10", "recall@20")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def paired_rows(base: dict, challenger: dict) -> list[tuple[dict, dict]]:
    base_by_id = {row["probe_id"]: row for row in base["per_probe"]}
    new_by_id = {row["probe_id"]: row for row in challenger["per_probe"]}
    common = sorted(base_by_id.keys() & new_by_id.keys())
    return [(base_by_id[probe_id], new_by_id[probe_id]) for probe_id in common]


def summarize_pairs(
    pairs: list[tuple[dict, dict]], iterations: int, seed: int,
) -> dict:
    rng = random.Random(seed)
    report = {}
    for metric in METRICS:
        valid = [
            (float(base[metric]), float(new[metric]))
            for base, new in pairs
            if not math.isnan(float(base[metric]))
            and not math.isnan(float(new[metric]))
        ]
        if not valid:
            continue
        observed = statistics.mean(new - base for base, new in valid)
        samples = []
        for _ in range(iterations):
            draw = [valid[rng.randrange(len(valid))] for _ in valid]
            samples.append(statistics.mean(new - base for base, new in draw))
        report[metric] = {
            "n": len(valid),
            "delta": round(observed, 6),
            "ci95": [
                round(quantile(samples, 0.025), 6),
                round(quantile(samples, 0.975), 6),
            ],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("challenger", type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    challenger = json.loads(args.challenger.read_text())
    pairs = paired_rows(base, challenger)
    by_family: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        by_family[pair[0]["family"]].append(pair)
    report = {
        "base": str(args.base),
        "challenger": str(args.challenger),
        "paired_probes": len(pairs),
        "iterations": args.iterations,
        "aggregate": summarize_pairs(pairs, args.iterations, args.seed),
        "by_family": {
            family: summarize_pairs(
                family_pairs, args.iterations, args.seed + index + 1,
            )
            for index, (family, family_pairs) in enumerate(sorted(by_family.items()))
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
