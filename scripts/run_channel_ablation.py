#!/usr/bin/env python3
"""Run reproducible add-one-in and leave-one-out search channel ablations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHANNEL_FLAGS = {
    "fts": "CHEMTREE_DISABLE_FTS",
    "dense": "CHEMTREE_DISABLE_DENSE",
    "tree": "CHEMTREE_DISABLE_TREE_RECALL",
    "source_paper": "CHEMTREE_DISABLE_SOURCE_PAPER_RECALL",
    "claim_guided_paper": "CHEMTREE_DISABLE_CLAIM_GUIDED_PAPER_RECALL",
    "author": "CHEMTREE_DISABLE_AUTHOR_RECALL",
}


def configs() -> list[tuple[str, set[str]]]:
    channels = set(CHANNEL_FLAGS)
    rows: list[tuple[str, set[str]]] = [("baseline", set())]
    rows.extend((f"loo_no_{channel}", {channel}) for channel in sorted(channels))
    for enabled in (
        {"fts"},
        {"dense"},
        {"fts", "dense"},
        {"fts", "dense", "source_paper"},
        {"fts", "dense", "claim_guided_paper"},
        {"fts", "dense", "tree"},
    ):
        rows.append((
            "add_" + "_".join(sorted(enabled)),
            channels - enabled,
        ))
    return rows


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes", type=Path, default=ROOT / "data/eval/probes_v2.jsonl",
    )
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--prefix", default="channels_v2")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selected = set(args.only)
    for label, disabled in configs():
        if selected and label not in selected:
            continue
        full_label = f"{args.prefix}_{label}"
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": "src",
            "CHEMTREE_SEARCH_CACHE": "0",
            "CHEMTREE_SEARCH_PROFILE": "1",
        })
        for channel, flag in CHANNEL_FLAGS.items():
            env[flag] = "1" if channel in disabled else "0"
        run([
            sys.executable,
            "scripts/eval_search_live.py",
            "--label", full_label,
            "--probes", str(args.probes),
            "--top", str(args.top),
        ], env)
        rankings = ROOT / "data/eval/runs" / f"{full_label}.rankings.jsonl"
        run([
            sys.executable,
            "scripts/eval_metrics.py",
            "--run", full_label,
            "--rankings", str(rankings),
            "--top-k", str(args.top),
            "--probes", str(args.probes),
        ], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
