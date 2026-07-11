"""
Second-pass Gemini check on the 813 'demote_small_to_l3' decisions.

The first pass (gemini_validate_l2_merges.py) sometimes flagged plurals,
hyphenation variants, and "X" / "X reaction" / "X process" synonyms as
'demote' when they're really just synonym renames. This pass uses a
tighter prompt focused only on distinguishing:

  reclassify_as_merge : SMALL is a true synonym / plural / spelling
                        variant / "X" vs "X reaction|process|method"
                        rewording of BIG. There's no extra info.

  keep_as_demote      : SMALL is a real chemistry sub-class of BIG.
                        Two different chemists would describe SMALL
                        and BIG differently and SMALL adds info.

Outputs a separate cache so we can apply the newly-merge entries via
apply_l2_merges.py and only the remaining true subtypes go to Phase 5.
"""
from __future__ import annotations

import argparse
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
PRIMARY_CACHE = PLAN_DIR / "gemini_validation_cache.json"
RECHECK_CACHE = PLAN_DIR / "gemini_recheck_demotes_cache.json"

BATCH_SIZE = 12
MAX_WORKERS = 10

PROMPT = """You are a senior chemistry-taxonomy curator doing a second-pass review.

A previous reviewer flagged each pair below as "demote" — meaning SMALL was
called a sub-class of BIG that should live as a child of BIG, not as its sibling.

Your job: catch the cases where SMALL is actually just a SYNONYM/RENAME of BIG,
not a real sub-class. These should be re-tagged as MERGE so they get fully
absorbed (no new sub-bucket created).

Re-tag rules:

re_merge   — SMALL is a synonym, plural, hyphenation variant, abbreviation
             expansion, or "X" vs "X reaction" / "X process" / "X method" /
             "X technique" rewording of BIG. Treating SMALL as a sub-class
             would introduce a meaningless empty sibling under BIG.
             Examples that should be re_merge:
               oxygen_evolution            → oxygen_evolution_reaction
               cell_based_assay            → cell_based_assays   (plural)
               2d_materials                → two_dimensional_materials
               metallic_nanoparticles      → metal_nanoparticles
               surface_functionalization   → functionalization (under surface_modification)
               carbon_capture_and_storage  → carbon_capture
               electrochemical_impedance_spectroscopy → impedance_spectroscopy

keep_demote — SMALL is a real chemistry sub-class of BIG. A chemist would
              still want to filter for SMALL specifically inside BIG.
              Examples that should stay keep_demote:
                halide_perovskites          ⊂ perovskites
                transition_metals           ⊂ metals
                cryo_electron_microscopy    ⊂ electron_microscopy
                magnetic_nanoparticles      ⊂ nanoparticles
                greenhouse_gases            ⊂ gases
                solar_energy_conversion     ⊂ energy_conversion

Output schema (one entry per input pair):
{{"decisions": [
  {{"id": <int 1-based>, "decision": "re_merge"|"keep_demote", "confidence": "high"|"medium"|"low", "reason": "<≤20 words>"}}
]}}

Pairs to re-evaluate (all currently flagged 'demote'):
{pairs_block}
"""


def load_demote_pairs() -> list[dict]:
    cache = json.load(PRIMARY_CACHE.open())
    out = []
    for k, v in cache.items():
        if v.get("decision") == "demote_small_to_l3":
            out.append({
                "key": k,
                "view": v["view"], "l1": v["l1"],
                "big": v["big"], "small": v["small"],
                "big_n": v["big_n"], "small_n": v["small_n"],
                "jaccard": v.get("jaccard", 0.0),
                "prev_reason": v.get("reason", ""),
            })
    return out


def load_recheck_cache() -> dict:
    if RECHECK_CACHE.exists():
        return json.load(RECHECK_CACHE.open())
    return {}


def save_recheck(cache: dict) -> None:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RECHECK_CACHE.with_suffix(".tmp")
    json.dump(cache, tmp.open("w"), indent=2)
    tmp.replace(RECHECK_CACHE)


def _format_pair(idx: int, p: dict) -> str:
    return (
        f"({idx}) view={p['view']} L1={p['l1']}  "
        f"BIG='{p['big']}' (claims={p['big_n']:,})  "
        f"SMALL='{p['small']}' (claims={p['small_n']:,})"
    )


_lock = threading.Lock()


def call_batch(batch: list[dict]) -> list[dict]:
    pairs_block = "\n".join(_format_pair(i + 1, p) for i, p in enumerate(batch))
    prompt = PROMPT.format(pairs_block=pairs_block)
    try:
        result = call_gemini(prompt, max_tokens=4000, retries=3)
    except Exception as e:
        return [{"id": i + 1, "decision": "error", "reason": f"gemini: {e}"} for i in range(len(batch))]
    parsed = result.get("parsed") or {}
    decisions = parsed.get("decisions") or []
    if len(decisions) != len(batch):
        return [{"id": i + 1, "decision": "error", "reason": f"size_mismatch {len(decisions)} vs {len(batch)}"} for i in range(len(batch))]
    return decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pairs = load_demote_pairs()
    cache = load_recheck_cache()
    print(f"Loaded {len(pairs)} demote pairs; recheck cache has {len(cache)} entries")

    todo = [p for p in pairs if p["key"] not in cache] if args.resume else pairs
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Nothing to do.")
        return
    print(f"Re-checking {len(todo)} pairs in batches of {BATCH_SIZE} via Gemini")

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    n_done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(call_batch, b): b for b in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            try:
                decisions = fut.result()
            except Exception as e:
                decisions = [{"id": i + 1, "decision": "error", "reason": f"batch: {e}"} for i in range(len(batch))]
            with _lock:
                for p, dec in zip(batch, decisions):
                    cache[p["key"]] = {
                        "view": p["view"], "l1": p["l1"],
                        "big": p["big"], "small": p["small"],
                        "big_n": p["big_n"], "small_n": p["small_n"],
                        "decision": dec.get("decision", "error"),
                        "confidence": dec.get("confidence", ""),
                        "reason": dec.get("reason", ""),
                    }
                n_done += len(batch)
                if n_done % 100 < BATCH_SIZE:
                    save_recheck(cache)
                    rate = n_done / max(time.time() - t0, 1)
                    eta = (len(todo) - n_done) / max(rate, 1e-3)
                    print(f"  [{n_done}/{len(todo)}] rate={rate:.1f}/s eta={eta:.0f}s")
    save_recheck(cache)

    tally = {"re_merge": 0, "keep_demote": 0, "error": 0}
    for v in cache.values():
        tally[v.get("decision", "error")] = tally.get(v.get("decision", "error"), 0) + 1
    print("\n--- recheck tally ---")
    for k, v in tally.items():
        print(f"  {k:15s}  {v:>4d}")
    print(f"\nFinished in {time.time() - t0:.0f}s; cache: {RECHECK_CACHE}")


if __name__ == "__main__":
    main()
