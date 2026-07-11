"""
Fix the index hierarchy after the rebuild crash.

The classification step completed (3200/3203 claims classified) but the
hierarchy building step crashed due to a missing source_doi default.

This script re-classifies all claims and builds the hierarchy from scratch,
with classification caching to avoid re-doing API calls if interrupted.
"""

import json
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from askchem.models import Claim, TreeNode
from askchem.store import AskChemStore
from askchem.indexer import classify_claim
from askchem.display import smart_title

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
CACHE_FILE = INDEX_DIR / "_classification_cache.json"


def main():
    print(f"AskChem Hierarchy Fix - {datetime.now().isoformat()}", flush=True)

    store = AskChemStore(INDEX_DIR)

    # Load all claims from disk
    claims_dir = INDEX_DIR / "claims"
    claims = []
    for f in sorted(claims_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        try:
            claim = Claim.from_dict(data)
            claims.append(claim)
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}", flush=True)

    print(f"Loaded {len(claims)} claims from disk", flush=True)

    # Load cached classifications
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cached_list = json.load(f)
        cache = {c["claim_id"]: c for c in cached_list}
        print(f"Loaded {len(cache)} cached classifications", flush=True)

    # Classify uncached claims
    uncached = [c for c in claims if c.claim_id not in cache]
    print(f"Need to classify {len(uncached)} claims ({len(cache)} cached)", flush=True)

    for i, claim in enumerate(uncached):
        try:
            paths = classify_claim(claim)
            cache[claim.claim_id] = {"claim_id": claim.claim_id, "paths": paths}
        except Exception as e:
            print(f"  Error classifying {claim.claim_id}: {e}", flush=True)
            cache[claim.claim_id] = {"claim_id": claim.claim_id, "paths": {}, "error": str(e)}

        if (i + 1) % 10 == 0:
            print(f"  Classified {i+1}/{len(uncached)} new claims", flush=True)
            # Save cache incrementally
            with open(CACHE_FILE, "w") as f:
                json.dump(list(cache.values()), f)
            time.sleep(1)

    # Final cache save
    with open(CACHE_FILE, "w") as f:
        json.dump(list(cache.values()), f)
    print(f"All {len(cache)} classifications cached", flush=True)

    # Clear existing view hierarchy (keep _root.json and _view.json)
    views_dir = INDEX_DIR / "views"
    for view_id in ["by_reaction_type", "by_substance_class", "by_application", "by_technique", "by_mechanism"]:
        view_dir = views_dir / view_id
        if view_dir.exists():
            for item in view_dir.iterdir():
                if item.name not in ("_root.json", "_view.json"):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        pass  # keep metadata files

    # Build hierarchy from classifications
    print("Building hierarchy...", flush=True)
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

            # Assign claim to leaf node
            store.assign_claim_to_node(view_id, path, claim.claim_id)
            assigned += 1

        if assigned % 500 == 0 and assigned > 0:
            print(f"  Assigned {assigned} claim-view pairs...", flush=True)

    total_nodes = len(node_cache)
    print(f"\n{'='*60}", flush=True)
    print("HIERARCHY FIX COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Claims: {len(claims)}", flush=True)
    print(f"Nodes: {total_nodes}", flush=True)
    print(f"Assignments: {assigned}", flush=True)


if __name__ == "__main__":
    main()
