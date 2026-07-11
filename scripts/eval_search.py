"""Golden-set benchmark for the AskChem search pipeline.

Runs a curated query set against ``search_claims`` and records per-query
metrics (hits, unique DOIs, venue-tier distribution, citation stats,
latency) plus aggregate numbers.  Results are stored as JSON keyed by a
human-readable run label so successive tier changes can be compared
side-by-side.

Usage:

  python scripts/eval_search.py --run baseline
  python scripts/eval_search.py --run tier-a --compare baseline
  python scripts/eval_search.py --run tier-c --compare tier-a

A missing "results" directory is created automatically.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import search_claims  # noqa: E402


RESULTS_DIR = REPO_ROOT / "scripts" / "eval_results"


# ---------------------------------------------------------------------------
# Golden set — 40 queries across 6 families
# ---------------------------------------------------------------------------

GOLDEN_SET: list[dict[str, Any]] = [
    # Topical, well-covered (should always do well)
    {"q": "suzuki coupling palladium catalyst",            "family": "topical"},
    {"q": "heavy metal adsorption",                         "family": "topical"},
    {"q": "CO2 electroreduction on copper",                 "family": "topical"},
    {"q": "asymmetric hydrogenation BINAP",                 "family": "topical"},
    {"q": "photocatalytic water splitting TiO2",            "family": "topical"},
    {"q": "ZIF-8 synthesis",                                "family": "topical"},
    {"q": "Heck reaction mechanism",                        "family": "topical"},
    {"q": "graphene oxide membrane filtration",             "family": "topical"},
    {"q": "single atom catalyst oxygen evolution",          "family": "topical"},
    {"q": "MoS2 hydrogen evolution reaction",               "family": "topical"},
    # Punctuation / Unicode edge cases (Tier-A target)
    {"q": "what is Suzuki coupling?",                       "family": "punctuation"},
    {"q": "catalysts for CO2 reduction.",                   "family": "punctuation"},
    {"q": "photocatalytic H₂ evolution",                    "family": "punctuation"},
    {"q": "C–H activation palladium",                       "family": "punctuation"},
    {"q": "TiO₂ anatase rutile",                            "family": "punctuation"},
    {"q": "25 °C reaction kinetics",                        "family": "punctuation"},
    # Acronym / formula (Tier-B target)
    {"q": "NMR of MOF",                                     "family": "acronym"},
    {"q": "MXene supercapacitor",                           "family": "acronym"},
    {"q": "ROMP polymer synthesis",                         "family": "acronym"},
    {"q": "BiVO4 photocatalyst",                            "family": "acronym"},
    {"q": "g-C3N4 visible light",                           "family": "acronym"},
    {"q": "NHC catalysis",                                  "family": "acronym"},
    # Reaction names missing from current dict (Tier-B target)
    {"q": "olefin metathesis Grubbs catalyst",              "family": "reaction"},
    {"q": "aldol reaction enantioselective",                "family": "reaction"},
    {"q": "Diels-Alder cycloaddition",                      "family": "reaction"},
    {"q": "click chemistry azide alkyne",                   "family": "reaction"},
    {"q": "C-H functionalization iridium",                  "family": "reaction"},
    {"q": "Mannich reaction",                               "family": "reaction"},
    # Morphological variants (Tier-C target: porter stemmer)
    {"q": "adsorbed heavy metals on MOF",                   "family": "morphology"},
    {"q": "catalysing Suzuki coupling",                     "family": "morphology"},
    {"q": "oxidized copper surfaces",                       "family": "morphology"},
    {"q": "polymerized ethylene",                           "family": "morphology"},
    # Author lookups (already fixed, sanity check)
    {"q": "papers by John Hartwig",                         "family": "author"},
    {"q": "Robert Grubbs",                                  "family": "author"},
    {"q": "Stephen Buchwald",                               "family": "author"},
    {"q": "author: Feng Liu",                               "family": "author"},
    # Multi-concept / long
    {"q": "Pd catalyzed cross-coupling mild conditions",     "family": "multi"},
    {"q": "CO2 reduction to methanol Cu catalyst",           "family": "multi"},
    {"q": "visible-light photoredox trifluoromethylation",  "family": "multi"},
    {"q": "MOF water purification arsenic removal",         "family": "multi"},
]


# ---------------------------------------------------------------------------
# Venue tiers — crude but consistent across runs
# ---------------------------------------------------------------------------

TIER_A = {
    'nature', 'science', 'nature chemistry', 'nature materials', 'nature catalysis',
    'nature communications', 'nature synthesis', 'chemical reviews', 'chem. rev.',
    'chemical society reviews', 'chem soc rev', 'accounts of chemical research',
    'acc. chem. res.', 'journal of the american chemical society', 'j. am. chem. soc.',
    'jacs', 'angewandte chemie', 'angew. chem.', 'angewandte chemie international edition',
    'angew. chem. int. ed.', 'chem',
}
TIER_B = {
    'acs catalysis', 'acs nano', 'acs central science', 'organic letters', 'org. lett.',
    'journal of catalysis', 'j. catal.', 'advanced materials', 'adv. mater.',
    'energy & environmental science', 'green chemistry', 'chemical science',
    'chem sci', 'acs applied materials & interfaces', 'applied catalysis b',
    'the journal of physical chemistry', 'j. phys. chem.', 'inorganic chemistry',
    'inorg. chem.', 'the journal of organic chemistry', 'j. org. chem.',
}


def _venue_tier(venue: str) -> str:
    if not venue:
        return "unknown"
    v = venue.lower()
    for pat in TIER_A:
        if pat in v:
            return "A"
    for pat in TIER_B:
        if pat in v:
            return "B"
    return "C"


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    q: str
    family: str
    hits: int
    unique_dois: int
    duplicates: int                  # hits - unique_dois within top_k
    zero_hits: bool
    tier_A: int
    tier_B: int
    tier_C: int
    tier_unknown: int
    citations_max: int
    citations_median: int
    citations_mean: float
    zero_citation_count: int
    latency_ms: int
    first_title: str
    top_venues: list[str] = field(default_factory=list)


def run_query(q: str, family: str, top_k: int = 10) -> QueryResult:
    t0 = time.monotonic()
    try:
        res = search_claims(q, limit=top_k)
    except Exception as e:
        return QueryResult(
            q=q, family=family, hits=0, unique_dois=0, duplicates=0,
            zero_hits=True,
            tier_A=0, tier_B=0, tier_C=0, tier_unknown=0,
            citations_max=0, citations_median=0, citations_mean=0.0,
            zero_citation_count=0, latency_ms=int((time.monotonic() - t0) * 1000),
            first_title=f"ERROR: {e!r}",
        )
    dt = int((time.monotonic() - t0) * 1000)
    items = (res or {}).get("results", []) or []
    dois = [(it.get("source_doi") or "").lower() for it in items]
    titles = [(it.get("source_paper_title") or "") for it in items]
    venues = [(it.get("venue") or it.get("source_venue") or "") for it in items]
    citations = [it.get("citation_count") or it.get("source_citation_count") or 0
                 for it in items]
    tier_counts = Counter(_venue_tier(v) for v in venues)
    unique_dois = len({d for d in dois if d})
    return QueryResult(
        q=q, family=family,
        hits=len(items),
        unique_dois=unique_dois,
        duplicates=max(0, len(items) - unique_dois),
        zero_hits=(len(items) == 0),
        tier_A=tier_counts.get("A", 0),
        tier_B=tier_counts.get("B", 0),
        tier_C=tier_counts.get("C", 0),
        tier_unknown=tier_counts.get("unknown", 0),
        citations_max=max(citations) if citations else 0,
        citations_median=int(statistics.median(citations)) if citations else 0,
        citations_mean=float(statistics.mean(citations)) if citations else 0.0,
        zero_citation_count=sum(1 for c in citations if not c),
        latency_ms=dt,
        first_title=(titles[0][:80] if titles else ""),
        top_venues=venues[:5],
    )


# ---------------------------------------------------------------------------
# Run + aggregate
# ---------------------------------------------------------------------------

def run_suite(top_k: int = 10) -> dict[str, Any]:
    per_query: list[QueryResult] = []
    for item in GOLDEN_SET:
        qr = run_query(item["q"], item["family"], top_k=top_k)
        per_query.append(qr)
        print(f"  {qr.family:<12} hits={qr.hits:>2} uniq={qr.unique_dois:>2} "
              f"A={qr.tier_A:>2} B={qr.tier_B} C={qr.tier_C:>2} "
              f"medC={qr.citations_median:>4} lat={qr.latency_ms:>5}ms  {qr.q}")
    all_hits = sum(q.hits for q in per_query)
    agg = {
        "n_queries": len(per_query),
        "n_zero_hit": sum(1 for q in per_query if q.zero_hits),
        "total_results": all_hits,
        "avg_hits_per_query": all_hits / len(per_query) if per_query else 0,
        "n_with_duplicates": sum(1 for q in per_query if q.duplicates > 0),
        "total_duplicates": sum(q.duplicates for q in per_query),
        "avg_latency_ms": int(statistics.mean(q.latency_ms for q in per_query)),
        "p95_latency_ms": int(
            statistics.quantiles([q.latency_ms for q in per_query], n=20)[18]
        ) if len(per_query) >= 20 else 0,
        "tier_A_share": sum(q.tier_A for q in per_query) / max(1, all_hits),
        "tier_B_share": sum(q.tier_B for q in per_query) / max(1, all_hits),
        "tier_C_share": sum(q.tier_C for q in per_query) / max(1, all_hits),
        "tier_unknown_share": sum(q.tier_unknown for q in per_query) / max(1, all_hits),
        "median_citations_overall": int(statistics.median(
            [q.citations_median for q in per_query if q.hits > 0] or [0]
        )),
        "by_family": {},
    }
    fams: dict[str, list[QueryResult]] = {}
    for q in per_query:
        fams.setdefault(q.family, []).append(q)
    for fam, items in fams.items():
        fam_hits = sum(q.hits for q in items)
        agg["by_family"][fam] = {
            "n": len(items),
            "avg_hits": fam_hits / len(items),
            "n_zero_hit": sum(1 for q in items if q.zero_hits),
            "tier_A_share": sum(q.tier_A for q in items) / max(1, fam_hits),
        }
    return {
        "aggregate": agg,
        "per_query": [asdict(q) for q in per_query],
    }


def diff_runs(baseline: dict[str, Any], new: dict[str, Any]) -> None:
    b = baseline["aggregate"]
    n = new["aggregate"]
    print("\n" + "=" * 78)
    print(f"{'metric':<32} {'baseline':>14} {'new':>14} {'delta':>14}")
    print("-" * 78)
    for key in ("n_zero_hit", "total_results", "avg_hits_per_query",
                "n_with_duplicates", "total_duplicates", "avg_latency_ms",
                "tier_A_share", "tier_B_share", "tier_C_share",
                "tier_unknown_share", "median_citations_overall"):
        bv = b.get(key, 0)
        nv = n.get(key, 0)
        dv = nv - bv
        if isinstance(bv, float) or isinstance(nv, float):
            print(f"{key:<32} {bv:>14.3f} {nv:>14.3f} {dv:>+14.3f}")
        else:
            print(f"{key:<32} {bv:>14} {nv:>14} {dv:>+14}")

    print("\nBy family:")
    fams = sorted(set(b.get("by_family", {})) | set(n.get("by_family", {})))
    print(f"  {'family':<14} {'base_zero':>10} {'new_zero':>10} "
          f"{'base_tierA':>10} {'new_tierA':>10}")
    for f in fams:
        bb = b.get("by_family", {}).get(f, {})
        nn = n.get("by_family", {}).get(f, {})
        print(f"  {f:<14} {bb.get('n_zero_hit', 0):>10} "
              f"{nn.get('n_zero_hit', 0):>10} "
              f"{bb.get('tier_A_share', 0):>10.3f} "
              f"{nn.get('tier_A_share', 0):>10.3f}")

    print("\nPer-query first-result changes:")
    bmap = {q["q"]: q for q in baseline.get("per_query", [])}
    for nq in new.get("per_query", []):
        bq = bmap.get(nq["q"], {})
        if bq.get("first_title") != nq.get("first_title"):
            print(f"  • {nq['q']}")
            print(f"      was:  {bq.get('first_title', '(n/a)')[:72]}")
            print(f"      now:  {nq.get('first_title', '')[:72]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="label for this run")
    p.add_argument("--compare", help="label of a previous run to diff against")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run)
    out_path = RESULTS_DIR / f"{label}.json"

    print(f"\n[eval] running {len(GOLDEN_SET)} queries, top_k={args.top_k}")
    print(f"[eval] output: {out_path.relative_to(REPO_ROOT)}\n")

    results = run_suite(top_k=args.top_k)
    out_path.write_text(json.dumps(results, indent=2))

    agg = results["aggregate"]
    print("\n" + "=" * 78)
    print(f"RUN = {label}")
    print(f"  queries        : {agg['n_queries']}")
    print(f"  zero-hit       : {agg['n_zero_hit']}")
    print(f"  total results  : {agg['total_results']}")
    print(f"  avg hits/query : {agg['avg_hits_per_query']:.2f}")
    print(f"  dup queries    : {agg['n_with_duplicates']} "
          f"(total dup = {agg['total_duplicates']})")
    print(f"  tier-A share   : {agg['tier_A_share']*100:.1f}%")
    print(f"  tier-B share   : {agg['tier_B_share']*100:.1f}%")
    print(f"  tier-C share   : {agg['tier_C_share']*100:.1f}%")
    print(f"  unknown share  : {agg['tier_unknown_share']*100:.1f}%")
    print(f"  median cites   : {agg['median_citations_overall']}")
    print(f"  avg latency ms : {agg['avg_latency_ms']}")

    if args.compare:
        cmp_path = RESULTS_DIR / f"{args.compare}.json"
        if not cmp_path.exists():
            print(f"\n[eval] --compare: {cmp_path} not found; skipping diff.")
        else:
            diff_runs(json.loads(cmp_path.read_text()), results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
