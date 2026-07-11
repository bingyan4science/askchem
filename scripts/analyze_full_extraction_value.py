#!/usr/bin/env python3
"""Rank abstract-only papers by expected value of full-paper extraction."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "chemtree.db"
OUT_JSONL = Path(__file__).resolve().parents[1] / "data" / "full_extraction_value_candidates.jsonl"
OUT_SUMMARY = Path(__file__).resolve().parents[1] / "data" / "full_extraction_value_summary.json"


CHEM_KEYWORDS = {
    "catalyst", "catalysis", "reaction", "synthesis", "polymer", "electrochem",
    "battery", "electrode", "photocatal", "adsorption", "material", "molecule",
    "molecular", "organic", "inorganic", "biochem", "protein", "peptide",
    "spectroscopy", "chromatography", "mechanism", "yield", "selectivity",
    "nanoparticle", "surface", "electrolyte", "anode", "cathode", "semiconductor",
    "membrane", "sorbent", "solvent", "ligand", "complex", "metal", "alloy",
}

REVIEW_PATTERNS = [
    r"\breview\b", r"\bperspective\b", r"\boutlook\b", r"\broadmap\b",
    r"\brecent advances\b", r"\bprogress in\b", r"\bprogress toward\b",
    r"\bstate of the art\b", r"\bmini-?review\b", r"\btutorial\b",
]

NONPAPER_PATTERNS = [
    r"\bhandbook\b", r"\bbook\b", r"\bchapter\b", r"\bencyclopedia\b",
    r"\bexperience and education\b",
]

SOFTWARE_PATTERNS = [
    r"\bsoftware\b", r"\bprogram\b", r"\btool\b", r"\bpackage\b",
    r"\bframework\b", r"\bplatform\b", r"\bcode\b", r"\bsimulator\b",
    r"\blammps\b", r"\bcharmm\b", r"\bgromacs\b", r"\bamber\b",
]

HIGH_VALUE_VENUES = {
    "journal of the american chemical society",
    "angewandte chemie",
    "nature chemistry",
    "nature catalysis",
    "acs catalysis",
    "advanced materials",
    "advanced functional materials",
    "nature communications",
    "chemical science",
    "energy & environmental science",
    "journal of materials chemistry",
    "acs applied materials and interfaces",
    "small",
    "journal of biological chemistry",
}

EXPERIMENTAL_TYPES = {
    "reaction": 4,
    "scope_entry": 4,
    "method": 3,
    "experimental_design": 3,
    "mechanism": 2,
    "property": 2,
    "comparison": 1,
}


def chemistry_keyword_hits(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in CHEM_KEYWORDS if kw in t)


def matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def compute_score(row: dict) -> tuple[float, dict, list[str], str]:
    title = row["title"] or ""
    venue = row["venue"] or ""
    abstract = row["abstract"] or ""
    full_text = " ".join([title, venue, abstract]).strip()

    claim_count = row["abstract_claims"]
    distinct_types = row["distinct_claim_types"]
    experimental_weight = row["experimental_weight"]
    citations = row["citation_count"] or 0
    keyword_hits = chemistry_keyword_hits(full_text)
    has_oa = bool(row["open_access_url"])
    is_review = matches_any(REVIEW_PATTERNS, full_text)
    is_nonpaper = matches_any(NONPAPER_PATTERNS, full_text)
    is_software = matches_any(SOFTWARE_PATTERNS, full_text)
    venue_bonus = 1 if venue.lower() in HIGH_VALUE_VENUES else 0
    no_abstract = len(abstract.strip()) == 0

    components = {
        "citations": min(math.log10(citations + 1) / 4.0, 1.0) * 25,
        "claim_richness": min(claim_count / 12.0, 1.0) * 18,
        "type_diversity": min(distinct_types / 6.0, 1.0) * 12,
        "experimental_signal": min(experimental_weight / 16.0, 1.0) * 22,
        "chemistry_signal": min(keyword_hits / 5.0, 1.0) * 15,
        "oa_feasibility": 8 if has_oa else 0,
        "venue_bonus": 5 if venue_bonus else 0,
        "review_penalty": -28 if is_review else 0,
        "software_penalty": -10 if is_software else 0,
        "nonpaper_penalty": -35 if is_nonpaper else 0,
        "no_abstract_penalty": -6 if no_abstract else 0,
    }
    score = round(max(0.0, sum(components.values())), 2)

    reasons = []
    if citations >= 200:
        reasons.append(f"{citations} citations")
    if claim_count >= 8:
        reasons.append(f"{claim_count} abstract claims")
    if distinct_types >= 4:
        reasons.append(f"{distinct_types} claim types")
    if experimental_weight >= 8:
        reasons.append("strong experimental/table-heavy signal")
    if keyword_hits >= 3:
        reasons.append("strong chemistry keyword signal")
    if has_oa:
        reasons.append("open-access link available")
    if venue_bonus:
        reasons.append("high-value chemistry venue")
    if is_review:
        reasons.append("review/perspective title")
    if is_software:
        reasons.append("software/tooling paper")
    if is_nonpaper:
        reasons.append("likely non-paper/book-chapter item")
    if no_abstract:
        reasons.append("missing source abstract metadata")

    if is_nonpaper:
        paper_style = "nonpaper_or_book"
    elif is_review:
        paper_style = "review_or_perspective"
    elif is_software:
        paper_style = "software_or_reference"
    else:
        paper_style = "primary_research"

    if score >= 60:
        tier = "high_value"
    elif score >= 42:
        tier = "valuable"
    elif score >= 28:
        tier = "maybe"
    else:
        tier = "low_value"

    return score, components, reasons, tier, paper_style


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
    WITH abstract_only AS (
      SELECT source_doi
      FROM claims
      WHERE source_doi != ''
      GROUP BY source_doi
      HAVING SUM(CASE WHEN extraction_version LIKE '%abstract%' THEN 1 ELSE 0 END) > 0
         AND SUM(CASE WHEN extraction_version = 'deep_v1' THEN 1 ELSE 0 END) = 0
    ),
    claim_aggs AS (
      SELECT
        source_doi,
        COUNT(*) AS abstract_claims,
        COUNT(DISTINCT claim_type) AS distinct_claim_types,
        SUM(CASE
              WHEN claim_type = 'reaction' THEN 4
              WHEN claim_type = 'scope_entry' THEN 4
              WHEN claim_type = 'method' THEN 3
              WHEN claim_type = 'experimental_design' THEN 3
              WHEN claim_type = 'mechanism' THEN 2
              WHEN claim_type = 'property' THEN 2
              WHEN claim_type = 'comparison' THEN 1
              ELSE 0
            END) AS experimental_weight,
        json_group_array(DISTINCT claim_type) AS claim_types
      FROM claims
      WHERE extraction_version LIKE '%abstract%'
        AND source_doi IN (SELECT source_doi FROM abstract_only)
      GROUP BY source_doi
    )
    SELECT
      s.doi, s.title, s.year, s.venue, s.abstract, s.citation_count, s.open_access_url,
      c.abstract_claims, c.distinct_claim_types, c.experimental_weight, c.claim_types
    FROM sources s
    JOIN claim_aggs c ON c.source_doi = s.doi
    ORDER BY s.citation_count DESC, c.abstract_claims DESC
    """

    rows = [dict(r) for r in cur.execute(query)]
    conn.close()

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    tier_counts = Counter()
    style_counts = Counter()
    venue_counts = Counter()
    year_counts = Counter()
    total = 0
    valuable = 0
    high_value = 0

    with open(OUT_JSONL, "w") as f:
        for row in rows:
            score, components, reasons, tier, paper_style = compute_score(row)
            row["claim_types"] = safe_json_loads(row["claim_types"])
            row["score"] = score
            row["score_components"] = {k: round(v, 2) for k, v in components.items()}
            row["reasons"] = reasons
            row["tier"] = tier
            row["paper_style"] = paper_style
            f.write(json.dumps(row) + "\n")

            total += 1
            tier_counts[tier] += 1
            style_counts[paper_style] += 1
            if tier in {"high_value", "valuable"}:
                valuable += 1
                venue_counts[row["venue"] or "(unknown)"] += 1
                year_counts[row["year"] or 0] += 1
            if tier == "high_value":
                high_value += 1

    top_examples = []
    with open(OUT_JSONL) as f:
        for line in f:
            row = json.loads(line)
            if row["tier"] in {"high_value", "valuable"}:
                top_examples.append(row)
            if len(top_examples) >= 25:
                break

    summary = {
        "total_abstract_only_papers": total,
        "high_value_papers": high_value,
        "valuable_papers": valuable - high_value,
        "all_valuable_papers": valuable,
        "maybe_papers": tier_counts["maybe"],
        "low_value_papers": tier_counts["low_value"],
        "paper_styles": dict(style_counts),
        "top_venues_among_valuable": venue_counts.most_common(15),
        "top_years_among_valuable": year_counts.most_common(15),
        "top_examples": [
            {
                "doi": r["doi"],
                "title": r["title"],
                "year": r["year"],
                "venue": r["venue"],
                "citation_count": r["citation_count"],
                "abstract_claims": r["abstract_claims"],
                "score": r["score"],
                "tier": r["tier"],
                "paper_style": r["paper_style"],
                "reasons": r["reasons"],
            }
            for r in top_examples
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote candidates to {OUT_JSONL}")
    print(f"Wrote summary to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
