"""
Generate canonical L3 definitions for large L2 nodes using LLM.

Reads data/large_l2_l3_samples.json (top existing L3 values per L2 node)
and asks the LLM to propose 5-15 canonical L3 subcategories for each.

Output: data/canonical_l3_raw.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "large_l2_l3_samples.json"
OUTPUT_PATH = DATA_DIR / "canonical_l3_raw.json"

SYSTEM_PROMPT = """You are a chemistry taxonomy expert. Given a hierarchical category path (view/L1/L2) and the top existing subcategories with their claim counts, propose 5-15 canonical L3 subcategories.

Rules:
- Each L3 must be mutually exclusive and collectively exhaustive within the L2.
- Use lowercase_with_underscores naming.
- Always include "other" as the last category.
- L3 should represent the most natural subdivision of the L2 for a chemistry researcher browsing the literature.
- Merge near-duplicates from the existing data (e.g., "lithium_ion_batteries" and "li_ion_batteries" become "lithium_ion").
- Don't be too specific (avoid single-compound or single-paper categories).
- Don't be too broad (each L3 should be meaningfully different from siblings).
- For reaction-type views, prefer named reactions or mechanistic distinctions.
- For substance views, prefer chemical classes or structural families.
- For technique views, prefer specific methods or instrument types.
- For application views, prefer application domains or target systems.
- For mechanism views, prefer mechanistic steps or physical phenomena.

Return ONLY a JSON object: {"categories": ["cat1", "cat2", ..., "other"]}"""


async def generate_l3_for_node(client: AsyncOpenAI, view_id: str, l1_l2: str,
                                data: dict, semaphore: asyncio.Semaphore) -> tuple[str, str, list]:
    top_l3 = data["top_l3"]
    claim_count = data["claim_count"]

    l3_text = "\n".join(f"  {name}: {cnt} claims" for name, cnt in top_l3[:50])

    user_msg = f"""Category: {view_id} / {l1_l2.replace('/', ' / ')}
Total claims: {claim_count:,}

Top existing subcategories:
{l3_text}

Propose 5-15 canonical L3 subcategories for this node."""

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            result = json.loads(text)
            categories = result.get("categories", [])
            if not categories:
                categories = ["other"]
            if categories[-1] != "other":
                categories.append("other")
            return view_id, l1_l2, categories
        except Exception as e:
            print(f"  ERROR {view_id}/{l1_l2}: {e}")
            return view_id, l1_l2, ["other"]


async def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    total = sum(len(v) for v in data.values())
    print(f"Generating canonical L3 for {total} L2 nodes...")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(20)

    tasks = []
    for view_id, l2_nodes in data.items():
        for l1_l2, node_data in l2_nodes.items():
            tasks.append(generate_l3_for_node(client, view_id, l1_l2, node_data, semaphore))

    results = await asyncio.gather(*tasks)

    output = {}
    for view_id, l1_l2, categories in results:
        output.setdefault(view_id, {})[l1_l2] = categories

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    total_cats = sum(len(cats) for v in output.values() for cats in v.values())
    print(f"\nDone! Generated {total_cats} total L3 categories for {total} L2 nodes.")
    print(f"Average: {total_cats / total:.1f} L3 per L2 node")
    print(f"Saved to {OUTPUT_PATH}")

    for view_id in sorted(output.keys()):
        cats = sum(len(c) for c in output[view_id].values())
        print(f"  {view_id}: {len(output[view_id])} nodes, {cats} categories")


if __name__ == "__main__":
    asyncio.run(main())
