"""Hand spot-check a stratified random sample of LLM judgments.

Reads ``data/eval/labels_v1.jsonl``, samples N rows stratified by score
(over-sampling the borderline 1-class), shows each in the terminal,
and asks you for a 0/1/2 verdict. Writes
``data/eval/spot_check_v1.json`` and prints the LLM-vs-human Cohen's
quadratic-weighted κ.

Decision threshold:
  κ ≥ 0.7  → ship LLM labels as-is
  0.5–0.7 → tighten the prompt and re-judge disagreements
  κ < 0.5 → fall back to hand labelling

Usage::

    python scripts/spot_check_labels.py --n 100
    python scripts/spot_check_labels.py --n 50  --resume

Resume re-opens ``spot_check_v1.json`` and only asks about the rows
you haven't labelled yet.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_common import (  # noqa: E402
    LABELS_PATH, SPOT_CHECK_PATH, iter_jsonl, open_claims_db,
    load_claims, render_claim_for_judge,
)


def _stratified_sample(rows: list[dict], n: int, seed: int = 0xc0fee) -> list[dict]:
    """Sample n rows with class proportions tilted toward the 1-class."""
    by_score: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_score[int(r["score"])].append(r)
    rng = random.Random(seed)
    for s in by_score.values():
        rng.shuffle(s)
    weights = {0: 0.30, 1: 0.40, 2: 0.30}
    out: list[dict] = []
    for score, w in weights.items():
        take = round(n * w)
        out.extend(by_score.get(score, [])[:take])
    if len(out) < n:
        leftover = [r for s, lst in by_score.items() for r in lst[len(out):]]
        rng.shuffle(leftover)
        out.extend(leftover[: n - len(out)])
    rng.shuffle(out)
    return out[:n]


def _cohens_kappa_quadratic(y_a: list[int], y_b: list[int]) -> float:
    """Compute Cohen's κ with quadratic weights, no sklearn dependency."""
    if not y_a or len(y_a) != len(y_b):
        return float("nan")
    classes = sorted({*y_a, *y_b})
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(y_a, y_b):
        obs[idx[a]][idx[b]] += 1
    n = len(y_a)
    row_tot = [sum(obs[i]) for i in range(k)]
    col_tot = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if k <= 1:
            return 0.0
        return ((i - j) ** 2) / ((k - 1) ** 2)

    num = den = 0.0
    for i in range(k):
        for j in range(k):
            exp = row_tot[i] * col_tot[j] / max(1, n)
            num += w(i, j) * obs[i][j]
            den += w(i, j) * exp
    if den == 0:
        return 1.0
    return 1.0 - num / den


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(LABELS_PATH))
    p.add_argument("--out", default=str(SPOT_CHECK_PATH))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0xc0fee)
    p.add_argument("--resume", action="store_true",
                   help="continue from existing spot_check_v1.json")
    args = p.parse_args()

    labels_path = Path(args.labels)
    out_path = Path(args.out)

    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run llm_judge_eval.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows = list(iter_jsonl(labels_path))
    print(f"Loaded {len(rows)} judgments from {labels_path}")
    score_dist = Counter(int(r["score"]) for r in rows)
    print(f"  judge score distribution: "
          f"0={score_dist[0]}  1={score_dist[1]}  2={score_dist[2]}")

    sample = _stratified_sample(rows, args.n, seed=args.seed)
    print(f"  sampled {len(sample)} rows for spot check")

    existing: list[dict] = []
    seen: set[str] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        seen = {f"{r['probe_id']}::{r['claim_id']}" for r in existing}
        print(f"  resuming with {len(existing)} previously labelled rows")

    conn = open_claims_db()
    try:
        claim_ids = [r["claim_id"] for r in sample]
        claims = load_claims(claim_ids, conn)
    finally:
        conn.close()

    out: list[dict] = list(existing)
    try:
        for i, r in enumerate(sample, 1):
            key = f"{r['probe_id']}::{r['claim_id']}"
            if key in seen:
                continue
            cid = r["claim_id"]
            claim = claims.get(cid)
            print()
            print("=" * 78)
            print(f"[{i}/{len(sample)}] probe={r['probe_id']} family={r['family']}")
            print(f"  Q:     {r['q']}")
            print()
            if claim is None:
                print("  (CLAIM NOT FOUND IN DB)")
            else:
                print(render_claim_for_judge(claim))
            print()
            print(f"  GPT score: {r['score']}  ({r['rationale']})")
            while True:
                v = input("  Your score [0/1/2/skip/quit]: ").strip().lower()
                if v in {"0", "1", "2"}:
                    out.append({**r, "human_score": int(v)})
                    break
                if v in {"skip", "s"}:
                    out.append({**r, "human_score": None, "skipped": True})
                    break
                if v in {"quit", "q"}:
                    raise KeyboardInterrupt()
                print("  please type 0, 1, 2, skip, or quit")
            with out_path.open("w") as f:
                json.dump(out, f, indent=2)
    except KeyboardInterrupt:
        print("\n  saved partial spot-check; rerun with --resume to continue.")

    judged = [r for r in out if r.get("human_score") is not None]
    if not judged:
        print("\n  no rows labelled — nothing to score.")
        return

    y_gpt = [int(r["score"]) for r in judged]
    y_hum = [int(r["human_score"]) for r in judged]
    kappa = _cohens_kappa_quadratic(y_gpt, y_hum)
    agree = sum(1 for a, b in zip(y_gpt, y_hum) if a == b) / len(judged)
    diff_off_by_2 = sum(1 for a, b in zip(y_gpt, y_hum) if abs(a - b) >= 2)

    print()
    print("=" * 78)
    print(f"Spot-check finished — {len(judged)} rows labelled")
    print(f"  exact agreement:        {agree*100:.1f}%")
    print(f"  off-by-2 disagreements: {diff_off_by_2}")
    print(f"  quadratic-weighted κ:    {kappa:.3f}")
    if kappa >= 0.7:
        print("  → GPT-judge labels look reliable. Ship them.")
    elif kappa >= 0.5:
        print("  → moderate reliability. Tighten the rubric and re-judge "
              "disagreements before relying on these labels.")
    else:
        print("  → poor reliability. Fall back to hand labelling.")


if __name__ == "__main__":
    main()
