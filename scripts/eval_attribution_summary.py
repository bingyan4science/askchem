"""Aggregate the Phase 1 attribution runs from
``scripts/eval_search_attribution.py``.

For one or more attribution labels, prints:

  - Category counts (recall-bounded / rerank-bounded / unaffected / no_relevant).
  - Median best-rank at each stage (fts / vector / rrf / rerank / final).
  - Per-family category breakdown.
  - Per-probe category-transition table when comparing two labels.

Usage::

    .venv-benchmark/bin/python scripts/eval_attribution_summary.py \\
        --labels W0_baseline W1_paw_fts W2_paw_rerank W3_paw_rerank_w50
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "eval" / "runs"


def load_rows(label: str) -> list[dict]:
    path = RUNS_DIR / f"attribution_{label}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    out: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def print_category_table(labels: list[str], data: dict[str, list[dict]]) -> None:
    print()
    print("=" * 78)
    print("Category counts")
    print("=" * 78)
    cats = ("unaffected", "rerank_bounded", "recall_bounded", "no_relevant", "error")
    header = f"  {'category':<18}  " + "  ".join(f"{lbl:>14}" for lbl in labels)
    print(header)
    print("  " + "-" * (16 + 16 * len(labels)))
    for cat in cats:
        cells = []
        for lbl in labels:
            n = sum(1 for r in data[lbl] if r.get("category") == cat)
            total = len(data[lbl]) or 1
            cells.append(f"{n:>4} ({n/total:>4.0%})")
        print(f"  {cat:<18}  " + "  ".join(f"{c:>14}" for c in cells))


def print_best_rank_table(labels: list[str], data: dict[str, list[dict]]) -> None:
    print()
    print("=" * 78)
    print("Median best-rank of any judged-positive claim, per stage")
    print("(None entries = the right answer never reached that stage)")
    print("=" * 78)
    stages = ("fts", "vector", "rrf", "rerank", "final")
    header = f"  {'stage':<10}  " + "  ".join(f"{lbl:>14}" for lbl in labels)
    print(header)
    print("  " + "-" * (8 + 16 * len(labels)))
    for st in stages:
        cells = []
        for lbl in labels:
            vals = [r["best_rank"][st] for r in data[lbl]
                    if r.get("best_rank") and r["best_rank"].get(st) is not None]
            absent = sum(1 for r in data[lbl]
                         if r.get("best_rank") and r["best_rank"].get(st) is None)
            med = statistics.median(vals) if vals else None
            cells.append(
                f"{fmt(med):>5}  (n={len(vals)},absent={absent})"
            )
        print(f"  {st:<10}  " + "  ".join(f"{c:>14}" for c in cells))


def print_per_family(labels: list[str], data: dict[str, list[dict]]) -> None:
    print()
    print("=" * 78)
    print("Per-family category share (rerank_bounded + recall_bounded share)")
    print("=" * 78)
    families = sorted({r["family"] for rs in data.values() for r in rs})
    header = f"  {'family':<10}  " + "  ".join(f"{lbl:>14}" for lbl in labels)
    print(header)
    print("  " + "-" * (8 + 16 * len(labels)))
    for fam in families:
        cells = []
        for lbl in labels:
            fam_rows = [r for r in data[lbl] if r["family"] == fam]
            n = len(fam_rows) or 1
            bound = sum(1 for r in fam_rows
                        if r.get("category") in ("rerank_bounded", "recall_bounded"))
            cells.append(f"{bound}/{n} ({bound/n:>4.0%})")
        print(f"  {fam:<10}  " + "  ".join(f"{c:>14}" for c in cells))


def print_transitions(labels: list[str], data: dict[str, list[dict]]) -> None:
    if len(labels) < 2:
        return
    print()
    print("=" * 78)
    print("Per-probe category transitions (vs first label)")
    print("=" * 78)
    base = {r["probe_id"]: r for r in data[labels[0]]}
    for lbl in labels[1:]:
        new = {r["probe_id"]: r for r in data[lbl]}
        improved = worsened = same = 0
        moves: list[tuple[str, str, str, str]] = []
        for pid, br in base.items():
            nr = new.get(pid)
            if not nr:
                continue
            bc = br.get("category")
            nc = nr.get("category")
            br_rank = (br.get("best_rank") or {}).get("final")
            nr_rank = (nr.get("best_rank") or {}).get("final")
            # Improvement = moved to better category or same category but
            # better rank. Use a category-priority lookup.
            cat_rank = {"unaffected": 0, "rerank_bounded": 1,
                        "recall_bounded": 2, "no_relevant": 3, "error": 4}
            if cat_rank.get(nc, 9) < cat_rank.get(bc, 9):
                improved += 1
                moves.append(("UP", pid, bc, nc))
            elif cat_rank.get(nc, 9) > cat_rank.get(bc, 9):
                worsened += 1
                moves.append(("DN", pid, bc, nc))
            elif br_rank is not None and nr_rank is not None:
                if nr_rank < br_rank:
                    improved += 1
                    moves.append(("up", pid, f"{bc}@{br_rank}",
                                  f"{nc}@{nr_rank}"))
                elif nr_rank > br_rank:
                    worsened += 1
                    moves.append(("dn", pid, f"{bc}@{br_rank}",
                                  f"{nc}@{nr_rank}"))
                else:
                    same += 1
            else:
                same += 1
        print(f"\n  {labels[0]:>14}  →  {lbl:<20}  "
              f"UP {improved}   DN {worsened}   SAME {same}")
        print("  --- biggest moves (first 6):")
        for mark, pid, before, after in moves[:6]:
            print(f"    {mark}  {pid:<10}  {before}  →  {after}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()

    data: dict[str, list[dict]] = {}
    for lbl in args.labels:
        data[lbl] = load_rows(lbl)
        print(f"loaded {len(data[lbl])} rows from attribution_{lbl}.jsonl")

    print_category_table(args.labels, data)
    print_best_rank_table(args.labels, data)
    print_per_family(args.labels, data)
    print_transitions(args.labels, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
