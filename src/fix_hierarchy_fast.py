"""
Fast parallel hierarchy fix using async concurrent API calls.

Resumes from the existing classification cache, then runs remaining
classifications with 20 concurrent requests.
"""

import json
import sys
import asyncio
import shutil
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from askchem.models import Claim, TreeNode
from askchem.store import AskChemStore
from askchem.indexer import CLASSIFICATION_PROMPT
from askchem.display import smart_title
from openai import AsyncOpenAI

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
CACHE_FILE = INDEX_DIR / "_classification_cache.json"
CONCURRENCY = 20

aclient = AsyncOpenAI()


async def classify_one(claim: Claim, semaphore: asyncio.Semaphore) -> dict:
    claim_summary = {
        "claim_type": claim.claim_type,
        "reaction_type": claim.reaction_type,
        "subject": claim.subject,
        "property_name": claim.property_name,
        "technique_name": claim.technique_name,
        "process_described": claim.process_described,
        "verbatim_quote": claim.verbatim_quote[:200],
        "reactants": claim.reactants[:3] if claim.reactants else [],
        "products": claim.products[:3] if claim.products else [],
        "conditions": claim.conditions,
    }
    claim_summary = {k: v for k, v in claim_summary.items() if v}
    prompt = CLASSIFICATION_PROMPT.format(claim_json=json.dumps(claim_summary, indent=2))

    async with semaphore:
        for attempt in range(3):
            try:
                response = await aclient.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    return {"claim_id": claim.claim_id, "paths": {}}
                return {"claim_id": claim.claim_id, "paths": json.loads(content)}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return {"claim_id": claim.claim_id, "paths": {}, "error": str(e)}
    return {"claim_id": claim.claim_id, "paths": {}}


async def classify_batch(claims: list[Claim], cache: dict) -> dict:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(claims)
    completed = 0

    async def classify_and_track(claim):
        nonlocal completed
        result = await classify_one(claim, semaphore)
        cache[result["claim_id"]] = result
        completed += 1
        if completed % 50 == 0:
            print(f"  Classified {completed}/{total} new claims", flush=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(list(cache.values()), f)
        return result

    tasks = [classify_and_track(c) for c in claims]
    await asyncio.gather(*tasks)
    return cache


def build_hierarchy(store, claims, cache):
    """Build the tree nodes from cached classifications."""
    print("Building hierarchy...", flush=True)

    views_dir = INDEX_DIR / "views"
    for view_id in ["by_reaction_type", "by_substance_class", "by_application", "by_technique", "by_mechanism"]:
        view_dir = views_dir / view_id
        if view_dir.exists():
            for item in view_dir.iterdir():
                if item.name not in ("_root.json", "_view.json"):
                    if item.is_dir():
                        shutil.rmtree(item)

    node_cache = {}
    assigned = 0

    for claim in claims:
        classification = cache.get(claim.claim_id, {})
        paths = classification.get("paths", {})

        for view_id, path in paths.items():
            if not path or path == ["not_applicable"]:
                continue

            for depth in range(len(path)):
                partial_path = path[:depth + 1]
                cache_key = (view_id, tuple(partial_path))

                if cache_key not in node_cache:
                    node_id = f"{view_id}_{'_'.join(partial_path)}"
                    node = TreeNode(
                        node_id=node_id,
                        name=smart_title(partial_path[-1]),
                        path=partial_path,
                        view=view_id,
                        level=depth + 1,
                    )
                    store.add_node(view_id, partial_path, node)
                    node_cache[cache_key] = node

            store.assign_claim_to_node(view_id, path, claim.claim_id)
            assigned += 1

    print(f"  {len(node_cache)} nodes, {assigned} assignments", flush=True)
    return len(node_cache)


def main():
    print(f"AskChem Fast Hierarchy Fix - {datetime.now().isoformat()}", flush=True)
    store = AskChemStore(INDEX_DIR)

    claims_dir = INDEX_DIR / "claims"
    claims = []
    for f in sorted(claims_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        try:
            claims.append(Claim.from_dict(data))
        except Exception as e:
            pass
    print(f"Loaded {len(claims)} claims", flush=True)

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cached_list = json.load(f)
        cache = {c["claim_id"]: c for c in cached_list}
    print(f"Cache: {len(cache)} already classified", flush=True)

    uncached = [c for c in claims if c.claim_id not in cache]
    print(f"Remaining: {len(uncached)} to classify ({CONCURRENCY} concurrent)\n", flush=True)

    if uncached:
        t0 = time.time()
        asyncio.run(classify_batch(uncached, cache))
        elapsed = time.time() - t0
        print(f"\nClassification done in {elapsed:.0f}s ({len(uncached)/elapsed:.1f} claims/s)", flush=True)

        with open(CACHE_FILE, "w") as f:
            json.dump(list(cache.values()), f)

    print(f"\nTotal classifications: {len(cache)}", flush=True)

    total_nodes = build_hierarchy(store, claims, cache)

    print(f"\n{'='*60}", flush=True)
    print("HIERARCHY FIX COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Claims: {len(claims)}", flush=True)
    print(f"Nodes: {total_nodes}", flush=True)


if __name__ == "__main__":
    main()
