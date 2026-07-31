"""Compute nDCG@k, MRR and Recall@k against the labelled eval pool.

This module is the metric core for the encoder bake-off. It loads
``data/eval/labels_v1.jsonl`` (built by ``llm_judge_eval.py``), then
either:

  (a) runs ``chemtree.db.search_claims`` over each probe and reports
      the metrics — useful as a baseline/regression check; or

  (b) reads a precomputed ``runs/<label>.jsonl`` of
      ``{probe_id, ranked_claim_ids}`` lines (for offline encoder
      experiments that don't go through the full search pipeline).

Usage::

    # Baseline run against current production search_claims:
    python scripts/eval_metrics.py --run baseline --top-k 20

    # Score a precomputed retrieval run:
    python scripts/eval_metrics.py --run pilot-bge-large \\
        --rankings data/eval/runs/pilot-bge-large.jsonl

    # Compare two runs:
    python scripts/eval_metrics.py --compare baseline pilot-bge-large
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_common import (  # noqa: E402
    EVAL_DIR, LABELS_PATH, PROBES_PATH, iter_jsonl, load_probes,
)


RUNS_DIR = EVAL_DIR / "runs"


# ── Metric primitives ──────────────────────────────────────────────────────


def _dcg(scores: list[int]) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(retrieved: list[str], judg: dict[str, int], k: int) -> float:
    rels = [judg.get(cid, 0) for cid in retrieved[:k]]
    ideal = sorted(judg.values(), reverse=True)[:k]
    if sum(ideal) == 0:
        return float("nan")
    return _dcg(rels) / _dcg(ideal)


def mrr(retrieved: list[str], judg: dict[str, int],
        relevant_threshold: int = 2) -> float:
    for i, cid in enumerate(retrieved):
        if judg.get(cid, 0) >= relevant_threshold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: list[str], judg: dict[str, int], k: int,
                relevant_threshold: int = 1) -> float:
    n_rel = sum(1 for s in judg.values() if s >= relevant_threshold)
    if n_rel == 0:
        return float("nan")
    n_hit = sum(1 for cid in retrieved[:k]
                if judg.get(cid, 0) >= relevant_threshold)
    return n_hit / n_rel


# ── Loading ────────────────────────────────────────────────────────────────


def load_judgments() -> dict[str, dict[str, int]]:
    """Return {probe_id → {claim_id → score}} merged across all judges."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for r in iter_jsonl(LABELS_PATH):
        out[r["probe_id"]][r["claim_id"]] = int(r["score"])
    return out


def load_rankings(path: Path) -> dict[str, list[str]]:
    """Read a precomputed rankings file (one JSON line per probe)."""
    out: dict[str, list[str]] = {}
    for r in iter_jsonl(path):
        out[r["probe_id"]] = list(r["ranked_claim_ids"])
    return out


# ── Live retrieval (run against current search_claims) ─────────────────────


def run_live(probes, top_k: int) -> dict[str, list[str]]:
    from askchem.db import search_claims  # noqa: E402

    rankings: dict[str, list[str]] = {}
    for i, probe in enumerate(probes, 1):
        t0 = time.monotonic()
        try:
            res = search_claims(
                probe.q,
                claim_type=probe.claim_type,
                view=probe.view,
                limit=top_k,
                mode=probe.mode,
                sort=probe.sort,
            )
        except Exception as e:
            print(f"  [{i}/{len(probes)}] {probe.id}  ERROR: {e!r}",
                  file=sys.stderr)
            rankings[probe.id] = []
            continue
        items = (res or {}).get("results", []) or []
        cids = [it.get("claim_id") for it in items if it.get("claim_id")]
        rankings[probe.id] = cids
        print(f"  [{i:>2}/{len(probes)}] {probe.id:<8} family={probe.family:<10} "
              f"hits={len(cids):>2}  ({int((time.monotonic()-t0)*1000)} ms)")
    return rankings


# ── Aggregation ────────────────────────────────────────────────────────────


def score_run(rankings: dict[str, list[str]],
              judgments: dict[str, dict[str, int]],
              probes,
              top_k: int = 20) -> dict:
    per_probe: list[dict] = []
    by_family: dict[str, list[dict]] = defaultdict(list)
    for probe in probes:
        ranked = rankings.get(probe.id, [])
        judg = judgments.get(probe.id, {})
        if not judg:
            continue
        rec = {
            "probe_id": probe.id,
            "family": probe.family,
            "q": probe.q,
            "n_retrieved": len(ranked),
            "n_judged": len(judg),
            "n_relevant": sum(1 for s in judg.values() if s >= 1),
            "n_highly_relevant": sum(1 for s in judg.values() if s >= 2),
            "ndcg@10": ndcg_at_k(ranked, judg, 10),
            "ndcg@20": ndcg_at_k(ranked, judg, 20),
            "mrr@20": mrr(ranked[:20], judg),
            "recall@10": recall_at_k(ranked, judg, 10),
            "recall@20": recall_at_k(ranked, judg, 20),
        }
        per_probe.append(rec)
        by_family[probe.family].append(rec)

    def _avg(rs, key):
        vs = [r[key] for r in rs if not _isnan(r.get(key))]
        return statistics.mean(vs) if vs else float("nan")

    aggregate = {
        "n_probes_scored": len(per_probe),
        "ndcg@10": _avg(per_probe, "ndcg@10"),
        "ndcg@20": _avg(per_probe, "ndcg@20"),
        "mrr@20": _avg(per_probe, "mrr@20"),
        "recall@10": _avg(per_probe, "recall@10"),
        "recall@20": _avg(per_probe, "recall@20"),
        "by_family": {},
    }
    for fam, rs in sorted(by_family.items()):
        aggregate["by_family"][fam] = {
            "n": len(rs),
            "ndcg@10": _avg(rs, "ndcg@10"),
            "ndcg@20": _avg(rs, "ndcg@20"),
            "mrr@20": _avg(rs, "mrr@20"),
            "recall@20": _avg(rs, "recall@20"),
        }
    return {"aggregate": aggregate, "per_probe": per_probe}


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ── Pretty-print ───────────────────────────────────────────────────────────


def _fmt(x) -> str:
    if x is None or _isnan(x):
        return "  n/a"
    return f"{x:.3f}"


def print_summary(label: str, scored: dict) -> None:
    a = scored["aggregate"]
    print()
    print(f"RUN = {label}")
    print(f"  probes scored:  {a['n_probes_scored']}")
    print(f"  nDCG@10:        {_fmt(a['ndcg@10'])}")
    print(f"  nDCG@20:        {_fmt(a['ndcg@20'])}")
    print(f"  MRR@20:         {_fmt(a['mrr@20'])}")
    print(f"  Recall@10:      {_fmt(a['recall@10'])}")
    print(f"  Recall@20:      {_fmt(a['recall@20'])}")
    print()
    print(f"  {'family':<12} {'n':>4} {'nDCG@10':>10} {'nDCG@20':>10} "
          f"{'MRR@20':>10} {'R@20':>10}")
    for fam, m in scored["aggregate"]["by_family"].items():
        print(f"  {fam:<12} {m['n']:>4} "
              f"{_fmt(m['ndcg@10']):>10} {_fmt(m['ndcg@20']):>10} "
              f"{_fmt(m['mrr@20']):>10} {_fmt(m['recall@20']):>10}")


def print_diff(base_label: str, base: dict,
               new_label: str, new: dict) -> None:
    a = base["aggregate"]
    b = new["aggregate"]
    print()
    print("=" * 80)
    print(f"{'metric':<14} {base_label:>14} {new_label:>14} {'Δ':>10}")
    for k in ("ndcg@10", "ndcg@20", "mrr@20", "recall@10", "recall@20"):
        bv = a.get(k); nv = b.get(k)
        d = (nv - bv) if not (_isnan(bv) or _isnan(nv)) else float("nan")
        print(f"  {k:<12} {_fmt(bv):>14} {_fmt(nv):>14} "
              f"{('+' + _fmt(d)) if not _isnan(d) and d >= 0 else _fmt(d):>10}")
    print()
    print(f"  {'family':<12} {'base nDCG@10':>14} {'new nDCG@10':>14} {'Δ':>10}")
    fams = sorted(set(a.get("by_family", {})) | set(b.get("by_family", {})))
    for fam in fams:
        bv = a.get("by_family", {}).get(fam, {}).get("ndcg@10")
        nv = b.get("by_family", {}).get(fam, {}).get("ndcg@10")
        if bv is None or nv is None:
            continue
        d = nv - bv if not (_isnan(bv) or _isnan(nv)) else float("nan")
        print(f"  {fam:<12} {_fmt(bv):>14} {_fmt(nv):>14} "
              f"{('+' + _fmt(d)) if not _isnan(d) and d >= 0 else _fmt(d):>10}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", help="label for this run; required unless --compare")
    p.add_argument("--rankings",
                   help="precomputed rankings JSONL "
                        "(default: run search_claims live)")
    p.add_argument("--probes", type=Path, default=PROBES_PATH)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"),
                   help="diff two saved runs by label")
    args = p.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        base_label, new_label = args.compare
        bp = RUNS_DIR / f"{base_label}.scored.json"
        np_ = RUNS_DIR / f"{new_label}.scored.json"
        if not bp.exists() or not np_.exists():
            print(f"ERROR: missing {bp if not bp.exists() else np_}",
                  file=sys.stderr)
            sys.exit(1)
        base = json.loads(bp.read_text())
        new = json.loads(np_.read_text())
        print_summary(base_label, base)
        print_summary(new_label, new)
        print_diff(base_label, base, new_label, new)
        return

    if not args.run:
        print("ERROR: --run is required", file=sys.stderr)
        sys.exit(1)

    probes = load_probes(args.probes)
    judgments = load_judgments()
    if not judgments:
        print(f"ERROR: no judgments at {LABELS_PATH}", file=sys.stderr)
        sys.exit(1)

    if args.rankings:
        rankings = load_rankings(Path(args.rankings))
        print(f"Loaded {len(rankings)} rankings from {args.rankings}")
    else:
        print(f"Running search_claims live, top_k={args.top_k}...")
        rankings = run_live(probes, args.top_k)
        live_path = RUNS_DIR / f"{args.run}.rankings.jsonl"
        with live_path.open("w") as f:
            for pid, cids in rankings.items():
                f.write(json.dumps({"probe_id": pid,
                                    "ranked_claim_ids": cids}) + "\n")
        print(f"  saved live rankings to {live_path}")

    scored = score_run(rankings, judgments, probes, top_k=args.top_k)
    out_path = RUNS_DIR / f"{args.run}.scored.json"
    out_path.write_text(json.dumps(scored, indent=2))
    print_summary(args.run, scored)
    print(f"\nSaved scored run to {out_path}")


if __name__ == "__main__":
    main()
