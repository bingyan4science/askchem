"""
Insight Generation for AskChem.

Analyzes the index to surface:
1. Frontier areas (sparse, underexplored)
2. Contradictions (conflicting claims)
3. Temporal trends (what's hot, what's cold)
4. Cross-view insights (well-studied in one view, unknown in another)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from askchem.store import AskChemStore

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
INSIGHTS_DIR = Path(__file__).parent.parent / "experiments" / "006_insights"


def analyze_index():
    """Comprehensive analysis of the AskChem index."""
    store = AskChemStore(INDEX_DIR)

    claims = store.list_claims(limit=10000)
    print(f"Analyzing {len(claims)} claims", flush=True)

    insights = {
        "timestamp": datetime.now().isoformat(),
        "total_claims": len(claims),
        "claim_type_distribution": {},
        "subfield_distribution": {},
        "temporal_analysis": {},
        "sparse_nodes": [],
        "dense_nodes": [],
        "potential_contradictions": [],
        "cross_view_insights": [],
    }

    # Claim type distribution
    type_counts = Counter(c.claim_type for c in claims)
    insights["claim_type_distribution"] = dict(type_counts.most_common())

    # Confidence distribution
    conf_counts = Counter(c.confidence for c in claims)
    insights["confidence_distribution"] = dict(conf_counts.most_common())

    # Source analysis
    source_counts = Counter(c.source_doi for c in claims if c.source_doi)
    insights["papers_with_most_claims"] = [
        {"doi": doi, "claim_count": count}
        for doi, count in source_counts.most_common(10)
    ]

    # Analyze each view for sparse/dense nodes
    for view_id in ["by_reaction_type", "by_substance_class", "by_application", "by_technique", "by_mechanism"]:
        view_dir = store.views_dir / view_id
        if not view_dir.exists():
            continue

        for node_dir in view_dir.rglob("_node.json"):
            with open(node_dir) as f:
                node_data = json.load(f)

            claim_count = node_data.get("claim_count", 0)
            path = node_data.get("path", [])
            name = node_data.get("name", "?")

            if claim_count > 0 and claim_count <= 2 and len(path) >= 2:
                insights["sparse_nodes"].append({
                    "view": view_id,
                    "path": path,
                    "name": name,
                    "claim_count": claim_count,
                    "interpretation": "Underexplored area — potential research opportunity",
                })

            if claim_count >= 5:
                insights["dense_nodes"].append({
                    "view": view_id,
                    "path": path,
                    "name": name,
                    "claim_count": claim_count,
                    "interpretation": "Well-studied area",
                })

    # Sort sparse nodes by depth (deeper = more specific = more interesting)
    insights["sparse_nodes"].sort(key=lambda x: len(x["path"]), reverse=True)
    insights["dense_nodes"].sort(key=lambda x: x["claim_count"], reverse=True)

    # Look for potential contradictions (reactions with very different yields)
    reaction_claims = [c for c in claims if c.claim_type in ("reaction", "scope_entry")]
    by_reaction_type = {}
    for c in reaction_claims:
        rt = c.reaction_type
        if rt:
            by_reaction_type.setdefault(rt, []).append(c)

    for rt, rc_claims in by_reaction_type.items():
        if len(rc_claims) < 2:
            continue
        yields = []
        for c in rc_claims:
            y = c.outcomes.get("yield_percent") if c.outcomes else None
            if y is not None:
                try:
                    yields.append((float(y), c.source_doi, c.claim_id))
                except (ValueError, TypeError):
                    pass
        if len(yields) >= 2:
            min_y = min(y for y, _, _ in yields)
            max_y = max(y for y, _, _ in yields)
            if max_y - min_y > 20:
                insights["potential_contradictions"].append({
                    "reaction_type": rt,
                    "yield_range": f"{min_y:.0f}%-{max_y:.0f}%",
                    "num_claims": len(rc_claims),
                    "claims": [{"yield": y, "doi": doi, "claim_id": cid} for y, doi, cid in yields],
                })

    # Cross-view analysis: claims that appear in many views
    multi_view_claims = []
    for c in claims:
        num_views = len(c.view_paths) if c.view_paths else 0
        if num_views >= 4:
            multi_view_claims.append({
                "claim_id": c.claim_id,
                "claim_type": c.claim_type,
                "num_views": num_views,
                "views": list(c.view_paths.keys()),
                "summary": c.verbatim_quote[:100] if c.verbatim_quote else "",
            })
    insights["cross_view_claims"] = sorted(multi_view_claims, key=lambda x: x["num_views"], reverse=True)[:20]

    return insights


def main():
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"AskChem Insight Generation - {datetime.now().isoformat()}", flush=True)

    insights = analyze_index()

    # Save
    with open(INSIGHTS_DIR / "insights.json", "w") as f:
        json.dump(insights, f, indent=2)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("INSIGHTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nTotal claims: {insights['total_claims']}", flush=True)
    print(f"\nClaim types: {insights['claim_type_distribution']}", flush=True)
    print(f"Confidence: {insights['confidence_distribution']}", flush=True)

    print(f"\nSparse nodes (research gaps): {len(insights['sparse_nodes'])}", flush=True)
    for n in insights["sparse_nodes"][:10]:
        print(f"  [{n['view']}] {'/'.join(n['path'])} ({n['claim_count']} claims)", flush=True)

    print(f"\nDense nodes (well-studied): {len(insights['dense_nodes'])}", flush=True)
    for n in insights["dense_nodes"][:10]:
        print(f"  [{n['view']}] {'/'.join(n['path'])} ({n['claim_count']} claims)", flush=True)

    print(f"\nPotential contradictions: {len(insights['potential_contradictions'])}", flush=True)
    for c in insights["potential_contradictions"][:5]:
        print(f"  {c['reaction_type']}: yield range {c['yield_range']} ({c['num_claims']} claims)", flush=True)

    print(f"\nCross-view claims (appear in 4+ views): {len(insights['cross_view_claims'])}", flush=True)
    for c in insights["cross_view_claims"][:5]:
        print(f"  {c['claim_type']} in {c['num_views']} views: {c['summary'][:60]}...", flush=True)


if __name__ == "__main__":
    main()
