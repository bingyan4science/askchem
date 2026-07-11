"""Augment ``data/eval/candidates_v1.jsonl`` with the v2-CLS rankings.

Adds the top-20 from ``live-v2-full-dense-cls`` and
``live-v2-full-rerank-cls`` to each probe's candidate set. The
existing ``llm_judge_eval.py`` then judges only the *newly added*
(probe, claim) pairs (it dedupes against ``labels_v1.jsonl``).

Why we need this
----------------
The current label pool was constructed from ``baseline-mini`` and
``alpha-mini-ctx`` rankings (α3 expansion). v2 mxbai surfaces *new*
relevant claims that nobody judged yet — 43 % of v2-CLS top-10 is
unjudged → the metric scores those as 0 → nDCG@10 is artificially
deflated. To get an apples-to-apples v1 vs v2 comparison we need to
judge the v2-surfaced claims too.

Usage::

    PYTHONPATH=src python3 scripts/expand_eval_pool_for_v2cls.py \\
        --runs data/eval/runs/live-v2-full-dense-cls.rankings.jsonl \\
               data/eval/runs/live-v2-full-rerank-cls.rankings.jsonl \\
        --top 20

Then::

    PYTHONPATH=src PORTKEY_API_KEY=... \\
        python3 scripts/llm_judge_eval.py \\
            --model gemini-3.1-pro-preview --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_common import CANDIDATES_PATH, LABELS_PATH, iter_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more rankings.jsonl files to fold into "
                        "the candidate pool.")
    p.add_argument("--top", type=int, default=20,
                   help="Use top-K from each run (default 20).")
    p.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    p.add_argument("--labels", type=Path, default=LABELS_PATH,
                   help="Used only for stats — counts how many newly "
                        "added pairs are unjudged.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Defaults to overwriting "
                        "--candidates in place (idempotent).")
    args = p.parse_args()
    if args.out is None:
        args.out = args.candidates

    cands = list(iter_jsonl(args.candidates))
    cand_by_pid = {r["id"]: r for r in cands}
    print(f"loaded {len(cands)} probes from {args.candidates}")

    labels = set()
    if args.labels.exists():
        for r in iter_jsonl(args.labels):
            labels.add((r["probe_id"], r["claim_id"]))
        print(f"loaded {len(labels):,} judgments from {args.labels}")

    new_pairs_total = 0
    unjudged_pairs_total = 0
    for run_path in args.runs:
        rp = Path(run_path)
        run_new = 0
        run_unj = 0
        for r in iter_jsonl(rp):
            pid = r["probe_id"]
            ranked = r["ranked_claim_ids"][: args.top]
            cand = cand_by_pid.get(pid)
            if cand is None:
                print(f"  WARN: probe {pid} not in candidates; skipping")
                continue
            existing = set(cand["candidate_ids"])
            sources = cand.setdefault("sources", {})
            tag = rp.stem  # e.g. "live-v2-full-dense-cls.rankings"
            for cid in ranked:
                if cid not in existing:
                    cand["candidate_ids"].append(cid)
                    existing.add(cid)
                    run_new += 1
                    if (pid, cid) not in labels:
                        run_unj += 1
                sources.setdefault(cid, [])
                if tag not in sources[cid]:
                    sources[cid].append(tag)
            cand["counts"] = cand.get("counts", {})
            cand["counts"]["union"] = len(cand["candidate_ids"])
        print(f"  + {rp.name}: added {run_new:,} new candidates "
              f"({run_unj:,} unjudged → must judge)")
        new_pairs_total += run_new
        unjudged_pairs_total += run_unj

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w") as fh:
        for cand in cands:
            fh.write(json.dumps(cand, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"\nwrote {args.out}")
    print(f"  total new candidate slots : {new_pairs_total:,}")
    print(f"  of which need a judgment  : {unjudged_pairs_total:,}")
    print(f"  pool size now             : "
          f"{sum(len(c['candidate_ids']) for c in cands):,}")


if __name__ == "__main__":
    main()
