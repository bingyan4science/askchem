"""Side-by-side search quality comparison: local Python vs prod HTTP.

Runs the same golden query set through:

  * local: in-process ``chemtree.db.search_claims`` (uses the FAISS
    vector index when available)
  * prod : the public HTTP API (FTS-only — no vector path)

For each query the script prints a compact comparison: total matches,
top-K overlap, first-result agreement, citation/venue tier deltas, and
latency.  An aggregate roll-up at the end answers the practical
question: "how much does losing the vector search on prod cost us?"

Usage:

  python scripts/compare_local_vs_prod.py
  python scripts/compare_local_vs_prod.py --top-k 10
  python scripts/compare_local_vs_prod.py --prod-url https://askchem.org
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import search_claims  # noqa: E402

# Reuse the same golden set + tier classifier the eval harness uses
# so numbers stay comparable across runs.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from eval_search import GOLDEN_SET, _venue_tier  # noqa: E402


# ---------------------------------------------------------------------------
# Local + prod runners
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    doi: str
    title: str
    venue: str
    tier: str
    citations: int


@dataclass
class Run:
    hits: list[Hit] = field(default_factory=list)
    total: int = 0
    latency_ms: int = 0
    error: str | None = None


def _normalise(items: list[dict]) -> list[Hit]:
    out = []
    for it in items or []:
        venue = (it.get("venue") or it.get("source_venue") or "")
        out.append(Hit(
            doi=(it.get("source_doi") or "").lower(),
            title=(it.get("source_paper_title") or "")[:100],
            venue=venue,
            tier=_venue_tier(venue),
            citations=int(it.get("citation_count")
                          or it.get("source_citation_count") or 0),
        ))
    return out


def run_local(q: str, top_k: int) -> Run:
    t0 = time.monotonic()
    try:
        res = search_claims(q, limit=top_k)
    except Exception as e:
        return Run(error=repr(e), latency_ms=int((time.monotonic() - t0) * 1000))
    return Run(
        hits=_normalise((res or {}).get("results", [])),
        total=int((res or {}).get("total", 0) or 0),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def run_prod(q: str, top_k: int, base: str) -> Run:
    url = f"{base.rstrip('/')}/api/search?q={urllib.parse.quote_plus(q)}&limit={top_k}"
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
    except Exception as e:
        return Run(error=repr(e), latency_ms=int((time.monotonic() - t0) * 1000))
    return Run(
        hits=_normalise(d.get("results", [])),
        total=int(d.get("total", 0) or 0),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Per-query comparison
# ---------------------------------------------------------------------------

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def compare_query(q: str, family: str, top_k: int, prod_url: str) -> dict:
    local = run_local(q, top_k)
    prod = run_prod(q, top_k, prod_url)
    l_dois = {h.doi for h in local.hits if h.doi}
    p_dois = {h.doi for h in prod.hits if h.doi}
    overlap = jaccard(l_dois, p_dois)
    same_first = bool(local.hits and prod.hits
                      and local.hits[0].doi == prod.hits[0].doi)
    local_only = sorted(l_dois - p_dois)
    prod_only = sorted(p_dois - l_dois)

    def med(xs):
        return int(statistics.median(xs)) if xs else 0

    def share(hits, t):
        return sum(1 for h in hits if h.tier == t) / max(1, len(hits))

    return {
        "q": q,
        "family": family,
        "local": {
            "n": len(local.hits),
            "total": local.total,
            "lat_ms": local.latency_ms,
            "med_cites": med([h.citations for h in local.hits]),
            "tier_A": sum(1 for h in local.hits if h.tier == "A"),
            "tier_B": sum(1 for h in local.hits if h.tier == "B"),
            "tier_C": sum(1 for h in local.hits if h.tier == "C"),
            "tier_unknown": sum(1 for h in local.hits if h.tier == "unknown"),
            "first_title": (local.hits[0].title if local.hits else ""),
            "error": local.error,
        },
        "prod": {
            "n": len(prod.hits),
            "total": prod.total,
            "lat_ms": prod.latency_ms,
            "med_cites": med([h.citations for h in prod.hits]),
            "tier_A": sum(1 for h in prod.hits if h.tier == "A"),
            "tier_B": sum(1 for h in prod.hits if h.tier == "B"),
            "tier_C": sum(1 for h in prod.hits if h.tier == "C"),
            "tier_unknown": sum(1 for h in prod.hits if h.tier == "unknown"),
            "first_title": (prod.hits[0].title if prod.hits else ""),
            "error": prod.error,
        },
        "overlap": overlap,
        "same_first": same_first,
        "local_only_dois": local_only[:5],
        "prod_only_dois": prod_only[:5],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_per_query_table(results: list[dict]) -> None:
    print(f"{'family':<11} {'overlap':>7} {'1st=':>4}  "
          f"{'l_n':>3} {'p_n':>3}  "
          f"{'l_medC':>6} {'p_medC':>6}  "
          f"{'l_tA':>4} {'p_tA':>4}  "
          f"{'l_ms':>5} {'p_ms':>5}   query")
    print("-" * 110)
    for r in results:
        l, p = r["local"], r["prod"]
        flag = "✓" if r["same_first"] else " "
        print(f"{r['family']:<11} {r['overlap']:>6.0%}  {flag:<3} "
              f"{l['n']:>3} {p['n']:>3}  "
              f"{l['med_cites']:>6} {p['med_cites']:>6}  "
              f"{l['tier_A']:>4} {p['tier_A']:>4}  "
              f"{l['lat_ms']:>5} {p['lat_ms']:>5}   {r['q']}")


def print_aggregates(results: list[dict]) -> None:
    n = len(results)
    overlaps = [r["overlap"] for r in results]
    same_first = sum(1 for r in results if r["same_first"])
    l_lat = [r["local"]["lat_ms"] for r in results]
    p_lat = [r["prod"]["lat_ms"] for r in results]
    l_cites = [r["local"]["med_cites"] for r in results if r["local"]["n"]]
    p_cites = [r["prod"]["med_cites"] for r in results if r["prod"]["n"]]
    l_total_A = sum(r["local"]["tier_A"] for r in results)
    p_total_A = sum(r["prod"]["tier_A"] for r in results)
    l_total_hits = sum(r["local"]["n"] for r in results)
    p_total_hits = sum(r["prod"]["n"] for r in results)
    l_zero = sum(1 for r in results if r["local"]["n"] == 0)
    p_zero = sum(1 for r in results if r["prod"]["n"] == 0)

    print()
    print("=" * 78)
    print("AGGREGATES")
    print("=" * 78)
    print(f"  queries                  : {n}")
    print(f"  zero-hit (local / prod)  : {l_zero} / {p_zero}")
    print(f"  same first result        : {same_first}/{n}  ({same_first/n*100:.0f}%)")
    print(f"  mean top-K overlap       : {statistics.mean(overlaps)*100:.1f}%")
    print(f"  median top-K overlap     : {statistics.median(overlaps)*100:.1f}%")
    print(f"  full-overlap queries     : "
          f"{sum(1 for o in overlaps if o == 1.0)}/{n}")
    print(f"  zero-overlap queries     : "
          f"{sum(1 for o in overlaps if o == 0.0)}/{n}")
    print()
    print(f"  median citations         : "
          f"local={statistics.median(l_cites or [0]):.0f}  "
          f"prod={statistics.median(p_cites or [0]):.0f}  "
          f"(Δ {statistics.median(p_cites or [0]) - statistics.median(l_cites or [0]):+.0f})")
    print(f"  tier-A share             : "
          f"local={l_total_A/max(1, l_total_hits)*100:.1f}%  "
          f"prod={p_total_A/max(1, p_total_hits)*100:.1f}%")
    print()
    print(f"  avg latency              : "
          f"local={statistics.mean(l_lat):.0f}ms  "
          f"prod={statistics.mean(p_lat):.0f}ms  "
          f"({statistics.mean(p_lat)/statistics.mean(l_lat):.1f}× slower)")
    print(f"  p95 latency              : "
          f"local={sorted(l_lat)[int(0.95*len(l_lat))]:.0f}ms  "
          f"prod={sorted(p_lat)[int(0.95*len(p_lat))]:.0f}ms")

    fams: dict[str, list[dict]] = {}
    for r in results:
        fams.setdefault(r["family"], []).append(r)
    print()
    print(f"  {'family':<14} {'n':>3} {'overlap':>9} {'1st=':>5} "
          f"{'l_medC':>7} {'p_medC':>7}  {'l_tA':>5} {'p_tA':>5}")
    for fam, items in sorted(fams.items()):
        ovl = statistics.mean(r["overlap"] for r in items)
        sf = sum(1 for r in items if r["same_first"]) / len(items)
        lc = statistics.median([r["local"]["med_cites"] for r in items
                                if r["local"]["n"]] or [0])
        pc = statistics.median([r["prod"]["med_cites"] for r in items
                                if r["prod"]["n"]] or [0])
        la = sum(r["local"]["tier_A"] for r in items) \
            / max(1, sum(r["local"]["n"] for r in items))
        pa = sum(r["prod"]["tier_A"] for r in items) \
            / max(1, sum(r["prod"]["n"] for r in items))
        print(f"  {fam:<14} {len(items):>3} {ovl*100:>8.1f}% "
              f"{sf*100:>4.0f}% {lc:>7.0f} {pc:>7.0f}  "
              f"{la*100:>4.0f}% {pa*100:>4.0f}%")


def print_first_result_diffs(results: list[dict], limit: int = 12) -> None:
    diffs = [r for r in results if not r["same_first"]
             and r["local"]["n"] and r["prod"]["n"]]
    if not diffs:
        return
    print()
    print("=" * 78)
    print(f"FIRST-RESULT DIFFERENCES ({len(diffs)} queries)")
    print("=" * 78)
    for r in diffs[:limit]:
        print(f"  • {r['q']}  ({r['family']}, overlap {r['overlap']*100:.0f}%)")
        print(f"      local: {r['local']['first_title'][:72]}")
        print(f"      prod : {r['prod']['first_title'][:72]}")
    if len(diffs) > limit:
        print(f"  ... and {len(diffs) - limit} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--prod-url", default="https://askchem.org")
    p.add_argument("--save", default=str(REPO_ROOT / "scripts" / "eval_results"
                                          / "local_vs_prod.json"))
    args = p.parse_args()

    print(f"comparing {len(GOLDEN_SET)} queries  "
          f"(local in-process vs {args.prod_url}, top_k={args.top_k})")
    print()

    results: list[dict] = []
    for item in GOLDEN_SET:
        r = compare_query(item["q"], item["family"], args.top_k, args.prod_url)
        results.append(r)
        l, p = r["local"], r["prod"]
        flag = "✓" if r["same_first"] else "·"
        print(f"  {flag} {r['family']:<11} ovl={r['overlap']*100:>3.0f}%  "
              f"l={l['n']:>2}/{l['total']:<5} p={p['n']:>2}/{p['total']:<5}  "
              f"medC l={l['med_cites']:>4} p={p['med_cites']:>4}  "
              f"{r['q']}")

    print_aggregates(results)
    print_first_result_diffs(results)

    out = Path(args.save).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    try:
        rel = out.relative_to(REPO_ROOT)
        print(f"\n[saved] {rel}")
    except ValueError:
        print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
