"""Journal-list-driven coverage audit for AskChem.

Two questions, one script:

  1. **What high-impact chemistry papers are MISSING from askchem.db?**
     Scans ``data/s2_audit/missing_dois.jsonl`` (produced by
     ``audit_s2_chemistry_gap.py``) and selects rows whose ``venue`` matches
     a curated chemistry journal list AND whose citation count clears a
     tier-aware threshold AND whose year is in range.

  2. **What papers ARE in chemtree.db but only have abstract-level
     extraction (no full-PDF deep_v1 claims)?**  Same journal +
     citation + year filter applied to the local ``sources`` table.

Outputs (under ``data/audits/``):

  * ``journal_audit_summary.json``  — per-journal counts + totals
  * ``to_add.jsonl``                — missing papers, sorted by citations
  * ``to_deep_extract.jsonl``       — abstract-only papers in DB worth
                                      sending to the deep-extract pipeline

Usage:

  python scripts/audit_journal_coverage.py
  python scripts/audit_journal_coverage.py --year-min 2018
  python scripts/audit_journal_coverage.py --tier-a 20 --tier-b 50 --tier-c 100
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "chemtree.db"
MISSING_FILE = REPO_ROOT / "data" / "s2_audit" / "missing_dois.jsonl"
OUT_DIR = REPO_ROOT / "data" / "audits"


# ---------------------------------------------------------------------------
# Journal taxonomy
# ---------------------------------------------------------------------------
# Each entry is: (canonical_label, tier, [substring_patterns_to_match])
# Patterns are matched case-insensitively on the venue string.  Order
# matters: the first matching journal wins, so put more-specific patterns
# (e.g. "nature chemistry") before more-generic ones (e.g. "nature").

JOURNALS: list[tuple[str, str, list[str]]] = [
    # ---- Tier A: top generalist + flagship review/letters venues ----
    # Specific Nature children FIRST, plain "Nature" anchored to avoid
    # absorbing "Nature Communications" / "Nature Photonics" / etc.
    ("Nature Chemistry",                   "A", ["nature chemistry", "nat. chem."]),
    ("Nature Materials",                   "A", ["nature materials", "nat. mater."]),
    ("Nature Catalysis",                   "A", ["nature catalysis", "nat. catal."]),
    ("Nature Synthesis",                   "A", ["nature synthesis", "nat. synth."]),
    ("Nature Reviews Chemistry",           "A", ["nature reviews chemistry", "nature reviews. chemistry", "nat. rev. chem."]),
    ("Nature Reviews Materials",           "A", ["nature reviews materials", "nature reviews. materials", "nat. rev. mater."]),
    ("Nature Nanotechnology",              "A", ["nature nanotechnology", "nat. nanotechnol."]),
    ("Nature Energy",                      "A", ["nature energy", "nat. energy"]),
    ("Nature Photonics",                   "A", ["nature photonics", "nat. photonics"]),
    ("Nature Chemical Biology",            "A", ["nature chemical biology", "nat. chem. biol."]),
    ("Nature",                             "A", ["^nature$"]),
    ("Science Advances",                   "A", ["^science advances$", "sci. adv."]),
    ("Science",                            "A", ["^science$"]),
    ("Chemical Reviews",                   "A", ["^chemical reviews$", "chem. rev."]),
    ("Chemical Society Reviews",           "A", ["chemical society reviews", "chem. soc. rev.", "chem soc rev"]),
    ("Accounts of Chemical Research",      "A", ["accounts of chemical research", "acc. chem. res.", "acc chem res"]),
    ("J. Am. Chem. Soc. (JACS)",           "A", ["journal of the american chemical society", "j. am. chem. soc.", "^jacs$"]),
    ("Angewandte Chemie Int. Ed.",         "A", ["angewandte chemie", "angew. chem.", "angew chem"]),
    ("Chem (Cell Press)",                  "A", ["^chem$"]),
    ("PNAS",                               "A", ["proceedings of the national academy of sciences", "^pnas$", "p. natl. acad. sci."]),

    # ---- Tier B: strong specialty + applied flagships ----
    ("Nature Communications",              "B", ["nature communications", "nat. commun.", "nat commun"]),
    ("ACS Catalysis",                      "B", ["acs catalysis", "acs catal."]),
    ("ACS Nano",                           "B", ["acs nano"]),
    ("ACS Central Science",                "B", ["acs central science", "acs cent. sci."]),
    ("ACS Energy Letters",                 "B", ["acs energy letters", "acs energy lett."]),
    ("ACS Sustainable Chem. Eng.",         "B", ["acs sustainable chemistry"]),
    ("JACS Au",                            "B", ["jacs au"]),
    ("Chemical Science (RSC)",             "B", ["^chemical science$", "chem. sci.", "^chem sci$"]),
    ("Energy & Environmental Science",     "B", ["energy & environmental science", "energy environ. sci.", "energy environ sci"]),
    ("Green Chemistry",                    "B", ["^green chemistry$", "green chem."]),
    ("Joule",                              "B", ["^joule$"]),
    ("Matter (Cell Press)",                "B", ["^matter$"]),
    ("Advanced Materials",                 "B", ["^advanced materials$", "adv. mater.", "^adv mater$"]),
    ("Advanced Functional Materials",      "B", ["advanced functional materials", "adv. funct. mater."]),
    ("Advanced Energy Materials",          "B", ["advanced energy materials", "adv. energy mater."]),

    # ---- Tier C: broad-scope solid specialty venues ----
    ("Organic Letters",                    "C", ["organic letters", "org. lett."]),
    ("Inorganic Chemistry",                "C", ["^inorganic chemistry$", "inorg. chem."]),
    ("J. Org. Chem.",                      "C", ["the journal of organic chemistry", "j. org. chem.", "jorgchm"]),
    ("J. Phys. Chem. (A/B/C/Lett)",        "C", ["the journal of physical chemistry", "j. phys. chem."]),
    ("J. Mater. Chem. A/B/C",              "C", ["journal of materials chemistry", "j. mater. chem."]),
    ("J. Catal.",                          "C", ["^journal of catalysis$", "j. catal."]),
    ("Applied Catalysis B",                "C", ["applied catalysis b", "appl. catal. b"]),
    ("Catalysis Science & Technology",     "C", ["catalysis science & technology", "catal. sci. technol."]),
    ("Macromolecules",                     "C", ["^macromolecules$"]),
    ("Polymer Chemistry",                  "C", ["^polymer chemistry$", "polym. chem."]),
    ("ACS Appl. Mater. Interfaces",        "C", ["acs applied materials & interfaces", "acs appl. mater. interfaces"]),
    ("Small",                              "C", ["^small$", "^small\\."]),
    ("ChemSusChem",                        "C", ["chemsuschem", "chem. sus. chem."]),
    ("ChemCatChem",                        "C", ["chemcatchem", "chem. cat. chem."]),
    ("Carbon",                             "C", ["^carbon$"]),
    ("Biomaterials",                       "C", ["^biomaterials$"]),

    # ---- Tier D: very high volume; only consider truly high-cite papers ----
    ("RSC Advances",                       "D", ["rsc advances", "rsc adv."]),
    ("Chemical Engineering Journal",       "D", ["chemical engineering journal", "chem. eng. j."]),
    ("Scientific Reports",                 "D", ["^scientific reports$", "sci. rep."]),
    ("Molecules",                          "D", ["^molecules$"]),
    ("ACS Omega",                          "D", ["acs omega"]),
    ("Nanomaterials",                      "D", ["^nanomaterials$"]),
    ("Materials",                          "D", ["^materials$"]),
    ("PCCP",                               "D", ["physical chemistry chemical physics", "physical chemistry, chemical physics", "phys. chem. chem. phys.", "pccp"]),
]


_CANONICAL_BY_PATTERN: list[tuple[re.Pattern, str, str]] = []
for label, tier, pats in JOURNALS:
    for pat in pats:
        # Anchored regex if the pattern starts with ^ or ends with $;
        # otherwise treat as a plain substring (escape regex meta).
        if pat.startswith("^") or pat.endswith("$") or "\\" in pat:
            rx = re.compile(pat, re.IGNORECASE)
        else:
            rx = re.compile(re.escape(pat), re.IGNORECASE)
        _CANONICAL_BY_PATTERN.append((rx, label, tier))


def classify_venue(venue: str) -> tuple[str | None, str | None]:
    """Return (label, tier) for a venue string, or (None, None) if no match."""
    if not venue:
        return None, None
    for rx, label, tier in _CANONICAL_BY_PATTERN:
        if rx.search(venue):
            return label, tier
    return None, None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def passes_filter(citations: int | None, year: int | None, tier: str,
                  thresholds: dict[str, int], year_min: int, year_max: int) -> bool:
    if year is None or year < year_min or year > year_max:
        return False
    cite_min = thresholds.get(tier, 9999)
    if (citations or 0) < cite_min:
        return False
    return True


def stream_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Audit passes
# ---------------------------------------------------------------------------

def audit_missing(thresholds: dict[str, int], year_min: int, year_max: int) -> dict:
    """Scan missing_dois.jsonl for high-value papers we don't have."""
    if not MISSING_FILE.exists():
        print(f"  WARNING: {MISSING_FILE} not found — run audit_s2_chemistry_gap.py first", flush=True)
        return {"per_journal": {}, "papers": [], "year_coverage": {}}

    per_journal: dict[str, dict] = defaultdict(lambda: {
        "tier": "?", "n_total_in_chem_audit": 0, "n_high_value": 0, "n_oa": 0,
    })
    keepers: list[dict] = []
    year_counter: Counter = Counter()
    n = 0
    t0 = time.time()
    for d in stream_jsonl(MISSING_FILE):
        n += 1
        year_counter[d.get("year") or 0] += 1
        venue = (d.get("venue") or "").strip()
        label, tier = classify_venue(venue)
        if label is None:
            continue
        per_journal[label]["tier"] = tier
        per_journal[label]["n_total_in_chem_audit"] += 1

        cites = d.get("citations")
        year = d.get("year")
        if not passes_filter(cites, year, tier, thresholds, year_min, year_max):
            continue

        per_journal[label]["n_high_value"] += 1
        if d.get("oa"):
            per_journal[label]["n_oa"] += 1

        keepers.append({
            "doi": d.get("doi"),
            "title": d.get("title"),
            "year": year,
            "citations": cites or 0,
            "oa": bool(d.get("oa")),
            "venue": venue,
            "journal": label,
            "tier": tier,
        })

        if n % 500_000 == 0:
            print(f"    scanned {n:,} rows in {time.time()-t0:.0f}s, "
                  f"{len(keepers):,} keepers so far", flush=True)

    keepers.sort(key=lambda x: -(x["citations"] or 0))
    print(f"  scanned {n:,} missing rows in {time.time()-t0:.0f}s", flush=True)
    years_present = sorted(y for y in year_counter if y)
    if years_present:
        print(f"  jsonl year span: {years_present[0]}-{years_present[-1]} "
              f"(rows by year: " + ", ".join(
                  f"{y}={year_counter[y]:,}" for y in years_present[-5:]
              ) + ")", flush=True)
    return {
        "per_journal": dict(per_journal),
        "papers": keepers,
        "year_coverage": {str(k): v for k, v in year_counter.items() if k},
    }


def audit_in_db(thresholds: dict[str, int], year_min: int, year_max: int) -> dict:
    """Find DB sources that match journals + thresholds but have no deep_v1 claims."""
    if not DB_PATH.exists():
        print(f"  WARNING: {DB_PATH} not found", flush=True)
        return {"per_journal": {}, "papers": []}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    print("  loading deep_v1 DOI set ...", flush=True)
    deep_dois: set[str] = set()
    for r in conn.execute(
        "SELECT DISTINCT source_doi FROM claims WHERE extraction_version = 'deep_v1'"
    ):
        if r[0]:
            deep_dois.add(r[0].strip().lower())
    print(f"  {len(deep_dois):,} DOIs already deep-extracted", flush=True)

    per_journal: dict[str, dict] = defaultdict(lambda: {
        "tier": "?", "n_in_db_total": 0, "n_already_deep": 0,
        "n_abstract_only_high_value": 0,
    })
    keepers: list[dict] = []
    t0 = time.time()
    n_rows = 0
    for r in conn.execute(
        "SELECT doi, title, year, venue, citation_count, open_access_url "
        "FROM sources"
    ):
        n_rows += 1
        venue = (r["venue"] or "").strip()
        label, tier = classify_venue(venue)
        if label is None:
            continue
        per_journal[label]["tier"] = tier
        per_journal[label]["n_in_db_total"] += 1

        doi = (r["doi"] or "").strip()
        if doi.lower() in deep_dois:
            per_journal[label]["n_already_deep"] += 1
            continue

        cites = r["citation_count"]
        year = r["year"]
        if not passes_filter(cites, year, tier, thresholds, year_min, year_max):
            continue

        per_journal[label]["n_abstract_only_high_value"] += 1
        keepers.append({
            "doi": doi,
            "title": r["title"],
            "year": year,
            "citations": cites or 0,
            "venue": venue,
            "journal": label,
            "tier": tier,
            "open_access_url": r["open_access_url"] or "",
        })

    conn.close()
    keepers.sort(key=lambda x: -(x["citations"] or 0))
    print(f"  scanned {n_rows:,} sources rows in {time.time()-t0:.0f}s, "
          f"{len(keepers):,} abstract-only high-value", flush=True)
    return {"per_journal": dict(per_journal), "papers": keepers}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(missing: dict, in_db: dict, thresholds: dict[str, int],
                  year_min: int, year_max: int) -> None:
    print()
    print("=" * 96)
    print(f"JOURNAL COVERAGE AUDIT  ({year_min}-{year_max})")
    print("  thresholds: " + " ".join(f"{t}≥{c}c" for t, c in sorted(thresholds.items())))
    print("=" * 96)

    rows: list[tuple[str, str, int, int, int, int, int]] = []
    all_labels = sorted(set(missing["per_journal"]) | set(in_db["per_journal"]))
    for label in all_labels:
        m = missing["per_journal"].get(label, {})
        d = in_db["per_journal"].get(label, {})
        tier = m.get("tier") or d.get("tier") or "?"
        rows.append((
            tier, label,
            d.get("n_in_db_total", 0),
            d.get("n_already_deep", 0),
            d.get("n_abstract_only_high_value", 0),
            m.get("n_high_value", 0),
            m.get("n_oa", 0),
        ))
    rows.sort(key=lambda x: (x[0], -(x[5] + x[4])))

    print(f"\n{'tier':<5}{'journal':<38}{'in_db':>7}{'deep':>7}"
          f"{'abs+hi':>8}{'missing':>9}{'miss_oa':>9}")
    print("-" * 83)
    last_tier = None
    sub_in_db = sub_deep = sub_abs = sub_miss = sub_oa = 0
    g_in_db = g_deep = g_abs = g_miss = g_oa = 0
    for tier, label, in_db_n, deep, abs_hi, miss, miss_oa in rows:
        if last_tier and tier != last_tier:
            print(f"  -- tier {last_tier} subtotal{'':<23}{sub_in_db:>7}{sub_deep:>7}"
                  f"{sub_abs:>8}{sub_miss:>9}{sub_oa:>9}")
            sub_in_db = sub_deep = sub_abs = sub_miss = sub_oa = 0
        last_tier = tier
        sub_in_db += in_db_n; sub_deep += deep; sub_abs += abs_hi
        sub_miss += miss; sub_oa += miss_oa
        g_in_db += in_db_n; g_deep += deep; g_abs += abs_hi
        g_miss += miss; g_oa += miss_oa
        print(f"{tier:<5}{label:<38}{in_db_n:>7,}{deep:>7,}"
              f"{abs_hi:>8,}{miss:>9,}{miss_oa:>9,}")
    if last_tier:
        print(f"  -- tier {last_tier} subtotal{'':<23}{sub_in_db:>7}{sub_deep:>7}"
              f"{sub_abs:>8}{sub_miss:>9}{sub_oa:>9}")
    print("-" * 83)
    print(f"  {'TOTAL':<41}{g_in_db:>7,}{g_deep:>7,}"
          f"{g_abs:>8,}{g_miss:>9,}{g_oa:>9,}")

    print()
    print("LEGEND:")
    print("  in_db    = papers from this journal currently in chemtree.db")
    print("  deep     = of those, how many already have full-PDF (deep_v1) extraction")
    print("  abs+hi   = in DB but ONLY abstract claims AND clears the citation/year bar")
    print("              → consider running deep extraction")
    print("  missing  = not in DB at all but matches journal + threshold")
    print("              → consider adding")
    print("  miss_oa  = of missing, how many are open-access (easy to ingest)")


def print_top_examples(label: str, papers: list[dict], k: int = 8) -> None:
    if not papers:
        return
    print(f"\n  {label} (showing top {k}):")
    for p in papers[:k]:
        oa_flag = "OA" if p.get("oa") or p.get("open_access_url") else "  "
        title = (p.get("title") or "")[:80]
        print(f"    [{oa_flag}] {p.get('citations', 0):>5} cit  {p.get('year', '?')}  "
              f"{p.get('journal', ''):<32}  {title}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier-a", type=int, default=30,
                   help="Min citations for tier-A flagships (default 30)")
    p.add_argument("--tier-b", type=int, default=75,
                   help="Min citations for tier-B specialty (default 75)")
    p.add_argument("--tier-c", type=int, default=150,
                   help="Min citations for tier-C broad specialty (default 150)")
    p.add_argument("--tier-d", type=int, default=400,
                   help="Min citations for tier-D high-volume (default 400)")
    p.add_argument("--year-min", type=int, default=2015)
    p.add_argument("--year-max", type=int, default=2026)
    p.add_argument("--top-examples", type=int, default=8)
    args = p.parse_args()

    thresholds = {"A": args.tier_a, "B": args.tier_b,
                  "C": args.tier_c, "D": args.tier_d}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== AUDIT 1/2: papers IN chemtree.db that are abstract-only ==")
    in_db = audit_in_db(thresholds, args.year_min, args.year_max)

    print()
    print(f"== AUDIT 2/2: papers MISSING from askchem.db ==")
    missing = audit_missing(thresholds, args.year_min, args.year_max)

    print_summary(missing, in_db, thresholds, args.year_min, args.year_max)
    print_top_examples("TOP MISSING (to add)", missing["papers"], args.top_examples)
    print_top_examples("TOP ABSTRACT-ONLY IN DB (to deep-extract)",
                       in_db["papers"], args.top_examples)

    # Write outputs
    summary_path = OUT_DIR / "journal_audit_summary.json"
    add_path = OUT_DIR / "to_add.jsonl"
    deep_path = OUT_DIR / "to_deep_extract.jsonl"

    yc = missing.get("year_coverage", {})
    yc_years = sorted(int(y) for y in yc.keys()) if yc else []
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thresholds": thresholds,
        "year_range": [args.year_min, args.year_max],
        "missing_jsonl_year_span": (
            [yc_years[0], yc_years[-1]] if yc_years else None
        ),
        "missing_jsonl_caveat": (
            "to_add candidates only reflect years present in "
            "data/s2_audit/missing_dois.jsonl; re-run audit_s2_chemistry_gap.py "
            "to broaden the year coverage."
        ),
        "totals": {
            "to_add":              len(missing["papers"]),
            "to_add_open_access":  sum(1 for p in missing["papers"] if p.get("oa")),
            "to_deep_extract":     len(in_db["papers"]),
        },
        "per_journal_missing": missing["per_journal"],
        "per_journal_in_db":   in_db["per_journal"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    with add_path.open("w") as f:
        for p in missing["papers"]:
            f.write(json.dumps(p) + "\n")
    with deep_path.open("w") as f:
        for p in in_db["papers"]:
            f.write(json.dumps(p) + "\n")

    print()
    print(f"[saved] {summary_path.relative_to(REPO_ROOT)}")
    print(f"[saved] {add_path.relative_to(REPO_ROOT)}    "
          f"({len(missing['papers']):,} candidates)")
    print(f"[saved] {deep_path.relative_to(REPO_ROOT)} "
          f"({len(in_db['papers']):,} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
