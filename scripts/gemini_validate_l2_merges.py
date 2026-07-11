"""
Gemini-validate proposed L2 merges before applying them.

For each candidate pair (BIG L2, small L2 under same view+L1), ask Gemini:
  Should `small` be merged into `big` as an L2-level alias?

Decision schema per pair:
  {
    "decision": "merge" | "keep_separate" | "demote_small_to_l3",
    "confidence": "high" | "medium" | "low",
    "reason": "<one sentence>"
  }

Batched in groups of N pairs per call to amortize LLM overhead.
Caches results so re-running is cheap.

Usage:
  PYTHONPATH=src python3 scripts/gemini_validate_l2_merges.py
  PYTHONPATH=src python3 scripts/gemini_validate_l2_merges.py --resume
  PYTHONPATH=src python3 scripts/gemini_validate_l2_merges.py --limit 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backfill_edges import call_gemini  # noqa: E402

PLAN_DIR = ROOT / "data/audits/l2"
CACHE_PATH = PLAN_DIR / "gemini_validation_cache.json"

BATCH_SIZE = 10
MAX_WORKERS = 10

PROMPT = """You are a chemistry-taxonomy curator. For each candidate L2 (level-2) pair below, decide whether the SMALL L2 bucket should be merged into the BIG L2 bucket.

Tree shape:  view  >  L1  >  L2  >  L3 (subcategories)  >  claims
The L2 buckets here all sit under the SAME (view, L1).

Decision options:
- "merge"             : SMALL is essentially a synonym, sub-flavor, or vague variant of BIG; safe to alias-snap SMALL → BIG with no loss of information.
- "keep_separate"     : SMALL is a meaningfully distinct concept (different subfield, different physical phenomenon, different chemical class). Merging would lose specificity.
- "demote_small_to_l3": SMALL is a strict sub-class / specialization of BIG (e.g., halide_perovskites under perovskites, transition_metal_oxides under metal_oxides). It deserves to live ON as a child of BIG, not as a sibling L2.

Be a strict chemistry domain expert. Default to "keep_separate" if uncertain. Output only valid JSON.

Output schema:
{{"decisions": [
  {{"id": <int 1-based pair index>, "decision": "merge"|"keep_separate"|"demote_small_to_l3", "confidence": "high"|"medium"|"low", "reason": "<≤25-word rationale>"}}
]}}

Pairs to evaluate:
{pairs_block}
"""


def _format_pair(idx: int, p: dict) -> str:
    return (
        f"({idx}) view={p['view']} L1={p['l1']}  "
        f"BIG='{p['big']}' (claims={p['big_n']:,})  "
        f"SMALL='{p['small']}' (claims={p['small_n']:,})  jaccard={p['jaccard']:.2f}"
    )


def load_pairs() -> list[dict]:
    """Read the 3 plans, dedupe, return one canonical list."""
    seen = set()
    pairs = []
    for fname in ["plan_A_lexical.tsv", "plan_B_subtype.tsv", "plan_C_borderline.tsv"]:
        path = PLAN_DIR / fname
        if not path.exists():
            continue
        with path.open() as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                key = (r["view"], r["l1"], r["big_l2"], r["small_l2"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "view": r["view"],
                    "l1": r["l1"],
                    "big": r["big_l2"],
                    "big_n": int(r["big_count"]),
                    "small": r["small_l2"],
                    "small_n": int(r["small_count"]),
                    "jaccard": float(r["jaccard"]),
                    "kind": r["kind"],
                    "source_plan": fname,
                })
    return pairs


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.load(CACHE_PATH.open())
    return {}


def save_cache(cache: dict) -> None:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(CACHE_PATH)


def cache_key(p: dict) -> str:
    return f"{p['view']}|{p['l1']}|{p['big']}|{p['small']}"


_cache_lock = threading.Lock()


def validate_batch(batch: list[dict]) -> list[dict]:
    """Send one batch of pairs to Gemini, return parsed decisions."""
    pairs_block = "\n".join(_format_pair(i + 1, p) for i, p in enumerate(batch))
    prompt = PROMPT.format(pairs_block=pairs_block)
    try:
        result = call_gemini(prompt, max_tokens=4000, retries=3)
    except Exception as e:
        return [{"id": i + 1, "decision": "error", "reason": f"gemini_failed: {e}"} for i in range(len(batch))]
    parsed = result.get("parsed") or {}
    decisions = parsed.get("decisions") or []
    if len(decisions) != len(batch):
        return [{"id": i + 1, "decision": "error", "reason": f"size_mismatch: got {len(decisions)} for batch of {len(batch)}"} for i in range(len(batch))]
    return decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Validate only the first N (uncached) pairs (0=all)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip pairs already in the cache")
    args = ap.parse_args()

    pairs = load_pairs()
    cache = load_cache()
    print(f"Loaded {len(pairs)} unique candidate pairs; cache has {len(cache)} entries")

    todo = [p for p in pairs if cache_key(p) not in cache] if args.resume else pairs
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Nothing to do.")
        return
    print(f"Will validate {len(todo)} pairs in batches of {BATCH_SIZE} via Gemini")

    # Build batches
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    n_done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(validate_batch, b): b for b in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            try:
                decisions = fut.result()
            except Exception as e:
                decisions = [{"id": i + 1, "decision": "error", "reason": f"batch_failed: {e}"} for i in range(len(batch))]
            with _cache_lock:
                for p, dec in zip(batch, decisions):
                    cache[cache_key(p)] = {
                        "view": p["view"], "l1": p["l1"],
                        "big": p["big"], "small": p["small"],
                        "big_n": p["big_n"], "small_n": p["small_n"],
                        "jaccard": p["jaccard"],
                        "decision": dec.get("decision", "error"),
                        "confidence": dec.get("confidence", ""),
                        "reason": dec.get("reason", ""),
                    }
                n_done += len(batch)
                if n_done % 100 < BATCH_SIZE:
                    save_cache(cache)
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1)
                    eta = (len(todo) - n_done) / max(rate, 1e-3)
                    print(f"  [{n_done}/{len(todo)}]  rate={rate:.1f}/s  eta={eta:.0f}s")
    save_cache(cache)

    # Summary
    decisions_count = {"merge": 0, "keep_separate": 0, "demote_small_to_l3": 0, "error": 0}
    for v in cache.values():
        decisions_count[v.get("decision", "error")] = decisions_count.get(v.get("decision", "error"), 0) + 1
    print("\n--- decision tally ---")
    for k, v in decisions_count.items():
        print(f"  {k:25s}  {v:>5d}")
    print(f"\nFinished in {time.time() - t0:.0f}s; cache saved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
