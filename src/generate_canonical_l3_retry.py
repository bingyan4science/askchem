"""Retry failed L3 generation for nodes that only got ["other"]."""

import asyncio
import json
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


async def generate_with_retry(client, view_id, l1_l2, node_data, semaphore, max_retries=5):
    top_l3 = node_data["top_l3"]
    claim_count = node_data["claim_count"]
    l3_text = "\n".join(f"  {name}: {cnt} claims" for name, cnt in top_l3[:50])

    user_msg = f"""Category: {view_id} / {l1_l2.replace('/', ' / ')}
Total claims: {claim_count:,}

Top existing subcategories:
{l3_text}

Propose 5-15 canonical L3 subcategories for this node."""

    for attempt in range(max_retries):
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
                if categories and len(categories) > 1:
                    if categories[-1] != "other":
                        categories.append("other")
                    return view_id, l1_l2, categories
            except Exception as e:
                pass

        wait = 2 ** (attempt + 1)
        await asyncio.sleep(wait)

    return view_id, l1_l2, None


async def main():
    with open(INPUT_PATH) as f:
        samples = json.load(f)

    with open(OUTPUT_PATH) as f:
        existing = json.load(f)

    failed = []
    for vid, nodes in existing.items():
        for l1_l2, cats in nodes.items():
            if cats == ["other"]:
                failed.append((vid, l1_l2))

    print(f"Retrying {len(failed)} failed nodes with concurrency=5...")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(5)

    tasks = []
    for vid, l1_l2 in failed:
        node_data = samples[vid][l1_l2]
        tasks.append(generate_with_retry(client, vid, l1_l2, node_data, semaphore))

    results = await asyncio.gather(*tasks)

    fixed = 0
    still_failed = 0
    for vid, l1_l2, categories in results:
        if categories:
            existing[vid][l1_l2] = categories
            fixed += 1
        else:
            still_failed += 1

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"Fixed: {fixed}, Still failed: {still_failed}")

    total_cats = sum(len(cats) for v in existing.values() for cats in v.values())
    total_nodes = sum(len(v) for v in existing.values())
    print(f"Total: {total_cats} categories for {total_nodes} nodes (avg {total_cats/total_nodes:.1f})")


if __name__ == "__main__":
    asyncio.run(main())
