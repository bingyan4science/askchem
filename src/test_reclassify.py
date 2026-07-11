"""Test the new constrained classification prompt on 50 diverse claims."""

import asyncio
import json
import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from askchem.taxonomy import (
    CANONICAL_L1, CANONICAL_L2, ALL_CONTENT_VIEWS,
    build_classification_prompt, normalize_path,
)
from askchem.llm import achat, MODELS


async def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "chemtree.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Pick 50 diverse claims: sample from different claim types and L1 categories
    rows = conn.execute("""
        SELECT claim_id, claim_type, source_paper_title, verbatim_quote, view_paths, data
        FROM claims
        WHERE verbatim_quote IS NOT NULL AND length(verbatim_quote) > 20
        ORDER BY RANDOM()
        LIMIT 50
    """).fetchall()
    conn.close()

    print(f"Testing {len(rows)} claims with new constrained prompt...\n")

    # Build allowed L2 sets for validation
    allowed = {}
    for view_id in ALL_CONTENT_VIEWS:
        allowed[view_id] = {}
        for l1, l2_list in CANONICAL_L2.get(view_id, {}).items():
            allowed[view_id][l1] = set(l2_list)

    results = []
    l1_violations = 0
    l2_violations = 0
    l2_other_count = 0
    l3_count = 0
    l4_plus_count = 0
    total_paths = 0

    sem = asyncio.Semaphore(10)

    async def classify_one(row):
        claim_type = row["claim_type"] or "property"
        quote = (row["verbatim_quote"] or "")[:300]
        title = row["source_paper_title"] or ""

        prompt = build_classification_prompt(claim_type, quote, title)

        async with sem:
            try:
                text = await achat(
                    [{"role": "user", "content": prompt}],
                    model=MODELS["fast"],
                    max_completion_tokens=2048,
                    json_mode=True,
                )
                return row["claim_id"], json.loads(text), None
            except Exception as e:
                return row["claim_id"], None, str(e)

    tasks = [classify_one(r) for r in rows]
    outputs = await asyncio.gather(*tasks)

    for claim_id, parsed, error in outputs:
        if error:
            print(f"  ERROR {claim_id}: {error}")
            continue
        if not parsed:
            print(f"  EMPTY {claim_id}")
            continue

        for view_id in ALL_CONTENT_VIEWS:
            path = parsed.get(view_id)
            if not path or path == ["not_applicable"]:
                continue

            total_paths += 1
            l1 = path[0] if len(path) >= 1 else None
            l2 = path[1] if len(path) >= 2 else None
            l3 = path[2] if len(path) >= 3 else None

            # Check L1
            if l1 not in set(CANONICAL_L1.get(view_id, [])):
                l1_violations += 1
                print(f"  L1 VIOLATION {claim_id} {view_id}: {l1} not in canonical")

            # Check L2
            if l2 and l1 in allowed.get(view_id, {}):
                if l2 not in allowed[view_id][l1]:
                    l2_violations += 1
                    print(f"  L2 VIOLATION {claim_id} {view_id}: {l1}/{l2} not allowed")
                if l2 == "other":
                    l2_other_count += 1

            if l3:
                l3_count += 1

            if len(path) > 3:
                l4_plus_count += 1
                print(f"  L4+ VIOLATION {claim_id} {view_id}: depth={len(path)} path={path}")

        results.append(parsed)

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} claims classified successfully")
    print(f"Total paths checked: {total_paths}")
    print(f"L1 violations: {l1_violations} ({l1_violations/max(total_paths,1)*100:.1f}%)")
    print(f"L2 violations: {l2_violations} ({l2_violations/max(total_paths,1)*100:.1f}%)")
    print(f"L2 = 'other': {l2_other_count} ({l2_other_count/max(total_paths,1)*100:.1f}%)")
    print(f"Has L3: {l3_count} ({l3_count/max(total_paths,1)*100:.1f}%)")
    print(f"Has L4+: {l4_plus_count} ({l4_plus_count/max(total_paths,1)*100:.1f}%)")

    # Show a few examples
    print(f"\n{'='*60}")
    print("SAMPLE OUTPUTS:")
    for i, res in enumerate(results[:5]):
        print(f"\n  Claim {i}:")
        for view_id in ALL_CONTENT_VIEWS:
            path = res.get(view_id, ["not_applicable"])
            print(f"    {view_id}: {path}")


if __name__ == "__main__":
    asyncio.run(main())
