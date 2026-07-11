"""
Build the AskChem index from extracted claims.

This script:
1. Initializes the store
2. Loads claims from extraction experiments
3. Classifies claims into 5 views using GPT-4o-mini
4. Builds the filesystem hierarchy
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from askchem.store import AskChemStore
from askchem.indexer import load_extracted_claims, build_index

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


def main():
    print(f"Building AskChem Index - {datetime.now().isoformat()}", flush=True)

    # Initialize store
    store = AskChemStore(INDEX_DIR)
    metadata = store.initialize()
    print(f"Store initialized at {INDEX_DIR}", flush=True)
    print(f"Views: {metadata['views']}", flush=True)

    # Load claims
    print(f"\nLoading extracted claims...", flush=True)
    claims, sources = load_extracted_claims(EXPERIMENTS_DIR)
    print(f"Loaded {len(claims)} claims from {len(sources)} sources", flush=True)

    # Build index
    print(f"\nBuilding index...", flush=True)
    result = build_index(store, claims, sources)

    print(f"\n{'='*60}", flush=True)
    print("INDEX BUILD COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Claims indexed: {result['claims_indexed']}", flush=True)
    print(f"Nodes created: {result['nodes_created']}", flush=True)

    # Show the tree structure
    print(f"\nIndex structure:", flush=True)
    for view_id in ["by_reaction_type", "by_substance_class", "by_application", "by_technique", "by_mechanism"]:
        view = store.get_view(view_id)
        if view:
            print(f"\n  {view.name}:", flush=True)
            tree = store.get_node_with_children(view_id, [], depth=2)
            if tree and "children_data" in tree:
                for child in tree["children_data"]:
                    name = child.get("name", "?")
                    cc = child.get("claim_count", 0)
                    print(f"    {name} ({cc} claims)", flush=True)
                    for grandchild in child.get("children_data", []):
                        gname = grandchild.get("name", "?")
                        gcc = grandchild.get("claim_count", 0)
                        print(f"      {gname} ({gcc} claims)", flush=True)


if __name__ == "__main__":
    main()
