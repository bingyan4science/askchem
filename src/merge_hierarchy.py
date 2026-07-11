"""
Post-hoc merge of fragmented L1 categories in the classification cache.

Strategy:
1. For each view, collect all unique L1 names with their claim counts.
2. Use one LLM call per view to define ~10-15 canonical L1 categories
   and map every existing L1 to one of them.
3. Rewrite the paths in the cache so L1 is always canonical.
4. Rebuild the hierarchy from the cleaned cache.
"""

import json
import shutil
import sys
import time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

from askchem.models import Claim, TreeNode
from askchem.store import AskChemStore
from askchem.display import smart_title

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
CACHE_FILE = INDEX_DIR / "_classification_cache.json"
MERGED_CACHE_FILE = INDEX_DIR / "_classification_cache_merged.json"
MERGE_MAP_FILE = INDEX_DIR / "_merge_map.json"

from askchem.llm import get_client, MODELS

client = get_client()

VIEW_IDS = [
    "by_reaction_type",
    "by_substance_class",
    "by_application",
    "by_technique",
    "by_mechanism",
]

VIEW_DESCRIPTIONS = {
    "by_reaction_type": "chemical reaction types and transformations (e.g. oxidation, reduction, coupling, substitution, polymerization, catalysis, electrochemistry)",
    "by_substance_class": "classes of molecules and materials (e.g. organic compounds, inorganic materials, biomolecules, nanomaterials, polymers, catalysts)",
    "by_application": "practical application domains (e.g. catalysis, energy, pharmaceutical, environmental, computational chemistry, materials science)",
    "by_technique": "experimental and computational methods/techniques (e.g. spectroscopy, computational methods, electrochemistry, synthesis, characterization)",
    "by_mechanism": "underlying physical/chemical mechanisms and phenomena (e.g. catalytic mechanisms, electronic structure, molecular interactions, transport, photophysics)",
}

MERGE_PROMPT = """You are consolidating a fragmented taxonomy of chemistry knowledge.

View: {view_id}
Description: {view_desc}

Below are all the L1 (top-level) category names that were independently generated during claim classification, along with how many claims each has. Many are near-duplicates or overlapping.

{l1_list}

Your task:
1. Define exactly 10-15 CANONICAL L1 categories for this view. These should be:
   - Mutually exclusive and collectively exhaustive for chemistry
   - Using standard chemistry terminology
   - Broad enough to absorb the fragments, specific enough to be useful
   - Named in lowercase_with_underscores

2. Map EVERY existing L1 name to exactly one canonical category.

Return a JSON object:
{{
  "canonical_categories": [
    {{"name": "category_name", "description": "Brief description of what this category covers"}},
    ...
  ],
  "mapping": {{
    "existing_l1_name": "canonical_category_name",
    "another_l1_name": "canonical_category_name",
    ...
  }}
}}

IMPORTANT: The mapping must include EVERY L1 name from the list above, with no exceptions. Use "other" as a catch-all canonical category for truly miscellaneous items."""


def get_l1_stats(cache, view_id):
    counter = Counter()
    for entry in cache:
        path = entry.get("paths", {}).get(view_id, [])
        if path and path != ["not_applicable"]:
            counter[path[0]] += 1
    return counter


BATCH_MAP_PROMPT = """You are mapping L1 category names to canonical categories.

View: {view_id}
The canonical categories are:
{canonical_list}

Map each of the following L1 names to exactly one canonical category name.
Return a JSON object: {{"mapping": {{"existing_name": "canonical_name", ...}}}}

L1 names to map:
{l1_list}
"""


def generate_merge_map(view_id, l1_counter):
    all_l1s = l1_counter.most_common()

    # Step 1: Get canonical categories using just the top L1s (keeps prompt small)
    top_l1s = all_l1s[:60]
    l1_list = "\n".join(f"  {name}: {count} claims" for name, count in top_l1s)

    prompt = MERGE_PROMPT.format(
        view_id=view_id,
        view_desc=VIEW_DESCRIPTIONS[view_id],
        l1_list=l1_list,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODELS["fast"],
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16384,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                time.sleep(2)
                continue
            result = json.loads(content)
            break
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    else:
        raise RuntimeError(f"Failed to get canonical categories for {view_id}")

    canonical = result["canonical_categories"]
    mapping = result.get("mapping", {})

    # Step 2: Map remaining L1s in batches
    mapped_names = set(mapping.keys())
    remaining = [(n, c) for n, c in all_l1s if n not in mapped_names]

    canonical_list = "\n".join(
        f"  {cat['name']}: {cat['description']}" for cat in canonical
    )

    BATCH_SIZE = 100
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        if not batch:
            break

        batch_l1_list = "\n".join(f"  {name} ({count} claims)" for name, count in batch)
        batch_prompt = BATCH_MAP_PROMPT.format(
            view_id=view_id,
            canonical_list=canonical_list,
            l1_list=batch_l1_list,
        )

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODELS["fast"],
                    messages=[{"role": "user", "content": batch_prompt}],
                    max_completion_tokens=8192,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                if not content:
                    time.sleep(2)
                    continue
                batch_result = json.loads(content)
                batch_mapping = batch_result.get("mapping", batch_result)
                if isinstance(batch_mapping, dict):
                    mapping.update(batch_mapping)
                break
            except Exception as e:
                print(f"    Batch attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(2)

        print(f"    Mapped batch {batch_start}-{batch_start+len(batch)}: "
              f"{len(mapping)} total mapped", flush=True)

    result["mapping"] = mapping
    return result


def apply_merge(cache, merge_maps):
    """Rewrite L1 in all paths according to the merge maps."""
    fixed = 0
    for entry in cache:
        paths = entry.get("paths", {})
        for view_id, mapping in merge_maps.items():
            path = paths.get(view_id, [])
            if not path or path == ["not_applicable"]:
                continue
            old_l1 = path[0]
            new_l1 = mapping.get(old_l1)
            if new_l1 and new_l1 != old_l1:
                path[0] = new_l1
                fixed += 1
    return fixed


def build_hierarchy(claims, cache):
    store = AskChemStore(INDEX_DIR)

    views_dir = INDEX_DIR / "views"
    for view_id in VIEW_IDS:
        view_dir = views_dir / view_id
        if view_dir.exists():
            for item in view_dir.iterdir():
                if item.name not in ("_root.json", "_view.json"):
                    if item.is_dir():
                        shutil.rmtree(item)

    cache_by_id = {e["claim_id"]: e for e in cache}
    node_cache = {}
    assigned = 0

    for claim in claims:
        entry = cache_by_id.get(claim.claim_id, {})
        paths = entry.get("paths", {})

        for view_id in VIEW_IDS:
            path = paths.get(view_id, [])
            if not path or path == ["not_applicable"]:
                continue

            for depth in range(len(path)):
                partial = path[: depth + 1]
                key = (view_id, tuple(partial))
                if key not in node_cache:
                    node_id = f"{view_id}_{'_'.join(partial)}"
                    node = TreeNode(
                        node_id=node_id,
                        name=smart_title(partial[-1]),
                        path=partial,
                        view=view_id,
                        level=depth + 1,
                    )
                    store.add_node(view_id, partial, node)
                    node_cache[key] = True

            store.assign_claim_to_node(view_id, path, claim.claim_id)
            assigned += 1

    return len(node_cache), assigned


def main():
    print("=" * 60)
    print("Post-hoc L1 Merge")
    print("=" * 60)

    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} cached classifications\n")

    merge_maps = {}
    canonical_info = {}

    for view_id in VIEW_IDS:
        print(f"\n--- {view_id} ---")
        l1_counter = get_l1_stats(cache, view_id)
        print(f"  Current: {len(l1_counter)} unique L1 categories")

        result = generate_merge_map(view_id, l1_counter)
        canonical = result["canonical_categories"]
        mapping = result["mapping"]

        canonical_info[view_id] = canonical
        merge_maps[view_id] = mapping

        print(f"  Canonical: {len(canonical)} categories")
        for cat in canonical:
            print(f"    {cat['name']}: {cat['description']}")

        unmapped = set(l1_counter.keys()) - set(mapping.keys())
        if unmapped:
            print(f"  WARNING: {len(unmapped)} L1s unmapped: {list(unmapped)[:5]}...")
            for um in unmapped:
                mapping[um] = "other"

        mapped_to = Counter(mapping.values())
        print(f"  Mapping distribution:")
        for cat_name, count in mapped_to.most_common():
            print(f"    {cat_name}: {count} L1s merged in")

    with open(MERGE_MAP_FILE, "w") as f:
        json.dump({"merge_maps": merge_maps, "canonical": canonical_info}, f, indent=2)
    print(f"\nSaved merge maps to {MERGE_MAP_FILE}")

    fixed = apply_merge(cache, merge_maps)
    print(f"\nRewrote {fixed} L1 paths")

    with open(MERGED_CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"Saved merged cache to {MERGED_CACHE_FILE}")

    # Verify
    print("\n--- Verification ---")
    for view_id in VIEW_IDS:
        l1_counter = get_l1_stats(cache, view_id)
        print(f"  {view_id}: {len(l1_counter)} L1 categories")
        for name, count in l1_counter.most_common():
            print(f"    {name}: {count}")

    # Also overwrite the main cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"\nOverwrote main cache at {CACHE_FILE}")

    # Rebuild hierarchy
    print("\nRebuilding hierarchy...")
    claims_dir = INDEX_DIR / "claims"
    claims = []
    for fp in sorted(claims_dir.glob("*.json")):
        with open(fp) as fh:
            data = json.load(fh)
        try:
            claims.append(Claim.from_dict(data))
        except Exception:
            pass

    nodes, assignments = build_hierarchy(claims, cache)
    print(f"  {nodes} nodes, {assignments} assignments")
    print("\nDONE")


if __name__ == "__main__":
    main()
