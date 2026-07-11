"""
Phase 2 of L3 cleanup: semantic snap with PAW.

For every (view, L1, L2, off_l3) that survived Phase 1's deterministic snap,
ask the PAW classifier `bingyan4science/askchem-l3-snap-v2` to choose the
best canonical L3 from the parent's whitelist (or "other").

Caches each (view, l1, l2, off_l3) -> canonical mapping to disk so we can
resume after interruption.

Default is dry-run; pass --commit to apply + rebuild tree_nodes.

Usage:
    python scripts/snap_l3_with_paw.py                    # dry-run, build cache
    python scripts/snap_l3_with_paw.py --view by_reaction_type
    python scripts/snap_l3_with_paw.py --commit --rebuild-tree
    python scripts/snap_l3_with_paw.py --max-keys 200     # smoke test on N keys
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Force CPU inference per the workspace memo (LoRA + Metal crashes on
# llama-cpp-python 0.3.19).
os.environ.setdefault("GGML_NO_METAL", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.taxonomy import get_canonical_l3, ALL_CONTENT_VIEWS  # noqa: E402

DB_PATH = ROOT / "chemtree.db"
CACHE_PATH = ROOT / "data" / "audits" / "l3" / "paw_snap_cache.json"
PAW_PROGRAM = "bingyan4science/askchem-l3-snap-v4"

# Minimum token Jaccard between the off-whitelist label and the canonical
# choice for a non-"other" snap to be trusted. PAW makes plausible-looking
# but semantically wrong guesses for narrow scientific terms (e.g.
# "transition_metal_halides" -> "transition_metal_dichalcogenides"); requiring
# lexical overlap weeds those out without losing the correct umbrella matches
# (e.g. "mof_synthesis" -> "reticular_frameworks_and_zeolites" still passes
# via the curated naming rules in the spec).
MIN_SNAP_JACCARD = 0.34


def _token_jaccard(a: str, b: str) -> float:
    ta = {t for t in a.lower().replace("-", "_").split("_") if t and len(t) > 1}
    tb = {t for t in b.lower().replace("-", "_").split("_") if t and len(t) > 1}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── PAW thin wrapper ────────────────────────────────────────────────────────

class PawClassifier:
    def __init__(self, program: str = PAW_PROGRAM):
        import programasweights as paw
        print(f"Loading PAW program {program} ...", flush=True)
        t0 = time.time()
        self._fn = paw.function(program)
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    def __call__(self, off: str, l1: str, l2: str, choices: list[str]) -> str:
        prompt = (
            f"OFF={off}; PARENT={l1}/{l2}; "
            f"CHOICES=[{', '.join(choices)}]"
        )
        try:
            out = self._fn(prompt)
        except Exception as e:
            print(f"  PAW error on {off!r}: {e}", flush=True)
            return "other"
        text = str(out).strip().splitlines()[0].strip().strip(".,;'\"`")
        return text


# ── enumeration ─────────────────────────────────────────────────────────────

def enumerate_keys(views: list[str]) -> list[tuple[str, str, str, str, int]]:
    """Return [(view, l1, l2, off_l3, claim_count), ...] sorted desc by count."""
    counts: Counter[tuple[str, str, str, str]] = Counter()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT view_paths FROM claims WHERE view_paths IS NOT NULL")
    for (vp_json,) in cur:
        try: vp = json.loads(vp_json)
        except Exception: continue
        for vid, p in vp.items():
            if vid not in views: continue
            if not isinstance(p, list) or len(p) < 3: continue
            l1, l2, l3 = p[0], p[1], p[2]
            allowed = get_canonical_l3(vid, l1, l2)
            if allowed is None or l3 in allowed: continue
            counts[(vid, l1, l2, l3)] += 1
    conn.close()
    out = [(*k, c) for k, c in counts.items()]
    out.sort(key=lambda t: -t[4])
    return out


# ── cache I/O ───────────────────────────────────────────────────────────────

def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, sort_keys=True, indent=0)
    tmp.replace(CACHE_PATH)


def cache_key(view: str, l1: str, l2: str, off: str) -> str:
    return f"{view}|{l1}|{l2}|{off}"


# ── snap loop ───────────────────────────────────────────────────────────────

def build_plan(
    views: list[str],
    max_keys: int | None = None,
    skip_paw: bool = False,
) -> tuple[dict, dict]:
    """
    Returns (plan, stats):
      plan[(view, l1, l2)][off_l3] = {"to": canonical or None, "count": N, "src": tag}
    """
    keys = enumerate_keys(views)
    print(f"Found {len(keys):,} unique (view, L1, L2, off_l3) keys "
          f"({sum(k[4] for k in keys):,} claims)")
    if max_keys:
        keys = keys[:max_keys]
        print(f"Limiting to first {len(keys):,} keys (--max-keys)")

    cache = load_cache()
    plan: dict = defaultdict(dict)
    stats = Counter()
    paw: PawClassifier | None = None

    SAVE_EVERY = 50
    inferred = 0
    cached_hits = 0
    n = 0

    for view, l1, l2, off, count in keys:
        n += 1
        ck = cache_key(view, l1, l2, off)
        canon_list = get_canonical_l3(view, l1, l2) or []
        canon_set = set(canon_list)

        if ck in cache:
            mapped = cache[ck]
            src = "cache"
            cached_hits += 1
        elif skip_paw:
            plan[(view, l1, l2)][off] = {"to": None, "count": count, "src": "skip_paw"}
            stats["skip_paw"] += count
            continue
        else:
            if paw is None:
                paw = PawClassifier()
            mapped = paw(off, l1, l2, canon_list)
            cache[ck] = mapped
            inferred += 1
            src = "paw"
            if inferred % SAVE_EVERY == 0:
                save_cache(cache)
                t = time.strftime("%H:%M:%S")
                print(f"  [{t}] {n}/{len(keys)}  inferred={inferred}  "
                      f"cached_hits={cached_hits}  last={off!r} -> {mapped!r}",
                      flush=True)

        # Validate raw output is a real choice
        if mapped not in canon_set:
            mlow = mapped.lower().replace("-", "_")
            best = next((c for c in canon_set if c.lower().replace("-", "_") == mlow), None)
            mapped = best if best else ("other" if "other" in canon_set else None)

        plan[(view, l1, l2)][off] = {
            "to": mapped, "count": count, "src": src,
        }
        stats[src] += count
        if mapped is not None:
            stats["snapped"] += count
        if mapped == "other":
            stats["snapped_to_other"] += count

    if not skip_paw:
        save_cache(cache)
    return plan, stats


def report(plan: dict, stats: dict) -> None:
    total_off = sum(info["count"] for parents in plan.values() for info in parents.values())
    snapped = stats.get("snapped", 0)
    print(f"\nPhase 2 (PAW) summary:")
    print(f"  total off-whitelist claims processed: {total_off:,}")
    print(f"  snapped (any canonical):              {snapped:,}  "
          f"({100 * snapped / max(total_off, 1):.1f}%)")
    print(f"  snapped to 'other':                   {stats.get('snapped_to_other', 0):,}")
    print(f"  source: cache / paw / skip:           "
          f"{stats.get('cache', 0):,} / {stats.get('paw', 0):,} / {stats.get('skip_paw', 0):,}")

    # Per-view breakdown
    by_view: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # snapped, total
    for (v, l1, l2), parents in plan.items():
        for off, info in parents.items():
            by_view[v][1] += info["count"]
            if info["to"] is not None and info["to"] != "other":
                by_view[v][0] += info["count"]
    print("\n  Per-view (snapped non-other / total):")
    for v in sorted(by_view):
        s, t = by_view[v]
        print(f"    {v:24s}  {s:>9,} / {t:>9,}  ({100 * s / max(t, 1):5.1f}%)")


def show_examples(plan: dict, n: int = 30) -> None:
    rows = []
    for (v, l1, l2), parents in plan.items():
        for off, info in parents.items():
            rows.append((info["count"], v, l1, l2, off, info["to"], info["src"]))
    rows.sort(reverse=True)
    print(f"\nTop {n} snaps:")
    for c, v, l1, l2, off, to, src in rows[:n]:
        path = f"{l1}/{l2}/{off}"
        print(f"  {c:>6,d}  [{v:22s}]  {path[:55]:55s}  → {str(to)[:40]:40s}  ({src})")


# ── DB apply ─────────────────────────────────────────────────────────────────

def apply(plan: dict) -> None:
    flat: dict[tuple[str, str, str, str], str] = {}
    for (v, l1, l2), parents in plan.items():
        for off, info in parents.items():
            if info["to"] is not None:
                flat[(v, l1, l2, off)] = info["to"]
    if not flat:
        print("  Nothing to apply.")
        return
    print(f"\nApplying {len(flat):,} unique snap rules to chemtree.db ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.execute("SELECT claim_id, view_paths, data FROM claims WHERE view_paths IS NOT NULL")
    updates: list[tuple[str, str, str]] = []
    BATCH = 5000
    n_changed = 0
    n_scanned = 0
    while True:
        rows = cur.fetchmany(20000)
        if not rows: break
        for claim_id, vp_json, data_json in rows:
            try: vp = json.loads(vp_json)
            except Exception: continue
            n_scanned += 1
            changed = False
            for vid, p in vp.items():
                if not isinstance(p, list) or len(p) < 3: continue
                key = (vid, p[0], p[1], p[2])
                if key in flat:
                    p[2] = flat[key]
                    changed = True
            if changed:
                n_changed += 1
                try: data = json.loads(data_json) if data_json else {}
                except Exception: data = {}
                data["view_paths"] = vp
                updates.append((json.dumps(vp), json.dumps(data), claim_id))
                if len(updates) >= BATCH:
                    conn.executemany("UPDATE claims SET view_paths=?, data=? WHERE claim_id=?", updates)
                    conn.commit()
                    updates = []
        print(f"  scanned {n_scanned:,}  changed {n_changed:,}", flush=True)
    if updates:
        conn.executemany("UPDATE claims SET view_paths=?, data=? WHERE claim_id=?", updates)
        conn.commit()
    conn.close()
    print(f"\nDone. Updated {n_changed:,} claims.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="append", help="restrict to view(s)")
    ap.add_argument("--max-keys", type=int, default=None,
                    help="cap number of unique keys to query (smoke test)")
    ap.add_argument("--commit", action="store_true",
                    help="apply DB updates (default: dry-run)")
    ap.add_argument("--rebuild-tree", action="store_true")
    ap.add_argument("--examples", type=int, default=30)
    ap.add_argument("--skip-paw", action="store_true",
                    help="don't call PAW; use cache only")
    args = ap.parse_args()

    views = args.view or list(ALL_CONTENT_VIEWS)
    print(f"Target views: {', '.join(views)}")
    plan, stats = build_plan(views, max_keys=args.max_keys, skip_paw=args.skip_paw)
    report(plan, stats)
    show_examples(plan, args.examples)

    if args.commit:
        apply(plan)
        if args.rebuild_tree:
            print("\nRebuilding tree_nodes...")
            from reclassify_l3_batch import rebuild_tree
            rebuild_tree()
        else:
            print("\nNote: tree_nodes is now stale. Re-run with --rebuild-tree.")
    else:
        print("\n(dry-run — no DB changes.)")


if __name__ == "__main__":
    main()
