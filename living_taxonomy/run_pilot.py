"""Phase-0 pilot runner for the living taxonomy.

Samples a small focused set of papers (read-only), places each candidate
leaf into a hand-built seed tree by embedding similarity, flags exceptions,
and writes inspectable output. No production data is written.

Usage:
    python3 living_taxonomy/run_pilot.py --view by_reaction_type --papers 30
    python3 living_taxonomy/run_pilot.py --view by_substance_class --papers 30
    # optional Gemini gray-zone adjudication (needs PORTKEY_API_KEY):
    python3 living_taxonomy/run_pilot.py --view by_reaction_type --use-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pilot_data
import placement as place_mod
import seed_trees

OUT_DIR = _HERE / "output"


def _summarize(placements):
    by_decision = Counter(p.decision for p in placements)
    by_branch = Counter(
        p.branch_path[-1] for p in placements if p.decision == "attach_leaf"
    )
    return by_decision, by_branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="by_reaction_type",
                    choices=list(seed_trees.PILOT_TREES.keys()))
    ap.add_argument("--papers", type=int, default=30)
    ap.add_argument("--max-leaves", type=int, default=400)
    ap.add_argument("--attach", type=float, default=place_mod.ATTACH_THRESHOLD)
    ap.add_argument("--exception", type=float, default=place_mod.EXCEPTION_THRESHOLD)
    ap.add_argument("--use-llm", action="store_true",
                    help="adjudicate gray-zone placements with Gemini (NYU)")
    args = ap.parse_args()

    print(f"[pilot] view={args.view} papers={args.papers} "
          f"attach>={args.attach} exception<={args.exception}", file=sys.stderr)

    dois = pilot_data.sample_papers(n_papers=args.papers)
    print(f"[pilot] sampled {len(dois)} papers", file=sys.stderr)

    leaves = pilot_data.load_leaves(args.view, dois, max_leaves=args.max_leaves)
    print(f"[pilot] {len(leaves)} candidate leaves", file=sys.stderr)

    branches = seed_trees.leaf_branches(args.view)
    print(f"[pilot] {len(branches)} seed leaf-branches", file=sys.stderr)

    placements = place_mod.place_leaves(
        leaves, branches, attach=args.attach, exception=args.exception
    )

    if args.use_llm:
        n_gray = sum(1 for p in placements if p.decision == "gray_zone")
        print(f"[pilot] adjudicating {n_gray} gray-zone leaves with Gemini...",
              file=sys.stderr)
        placements = place_mod.adjudicate_gray_zone(placements, branches)

    by_decision, by_branch = _summarize(placements)

    print("\n=== PLACEMENT SUMMARY ===")
    total = len(placements)
    for dec in ("attach_leaf", "gray_zone", "exception"):
        n = by_decision.get(dec, 0)
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {dec:12s}: {n:4d}  ({pct:5.1f}%)")
    print("\n  attached-by-branch:")
    for name, n in by_branch.most_common():
        print(f"    {n:4d}  {name}")

    print("\n  sample exceptions (would seed new branches):")
    shown = 0
    for p in placements:
        if p.decision == "exception":
            print(f"    [{p.score:.2f}] {p.text[:110]}")
            shown += 1
            if shown >= 8:
                break

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"placements_{args.view}.json"
    with out_path.open("w") as f:
        json.dump({
            "view": args.view,
            "thresholds": {"attach": args.attach, "exception": args.exception},
            "n_papers": len(dois),
            "n_leaves": total,
            "summary": dict(by_decision),
            "placements": [
                {
                    "claim_id": p.claim_id, "doi": p.doi, "title": p.title,
                    "year": p.year, "decision": p.decision,
                    "branch_path": p.branch_path, "score": round(p.score, 4),
                    "runner_up": list(p.runner_up),
                    "current_path": p.current_path,
                    "llm_verdict": p.llm_verdict, "text": p.text,
                }
                for p in placements
            ],
        }, f, indent=2)
    print(f"\n[pilot] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
