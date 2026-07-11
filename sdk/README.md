# AskChem Python SDK

The official Python client for **AskChem** — a hierarchical, multi-view, source-grounded knowledge index for chemistry.

2.44M+ structured claims extracted from 146,000+ papers, organized into 5 browsable hierarchies.

## Installation

```bash
pip install askchem
```

## Quick Start

```python
from askchem import AskChem

ct = AskChem()

# Search for claims
results = ct.search("Suzuki coupling palladium")
for claim in results.claims:
    print(f"[{claim.claim_type}] {claim.verbatim_quote[:80]}...")
    print(f"  Source: {claim.source_paper_title}")

# Browse the knowledge tree
node = ct.browse("by_reaction_type", path="catalysis/cross_coupling", depth=2)
print(f"Found {node.total_claims} claims")

# Get all claims from a specific paper
paper = ct.sources.get("10.1021/jacs.2c12345")
for claim in paper.claims:
    print(claim.claim_type, claim.verbatim_quote[:60])

# List available views
for view in ct.views():
    print(f"{view.view_id}: {view.description}")

# Submit a paper for extraction
result = ct.submit("10.1038/s41586-024-07421-0")
print(result["status"])
```

## Configuration

```python
ct = AskChem(
    api_key="ac-...",                          # or set CHEMTREE_API_KEY
    base_url="https://askchem.org",            # or set CHEMTREE_BASE_URL
)
```

## API Reference

| Method | Description |
|--------|-------------|
| `ct.search(query, view=..., ...)` | Hybrid search (FTS + paper + taxonomy + vector) across all claims |
| `ct.browse(view_id, path=..., depth=...)` | Browse the knowledge tree |
| `ct.views()` | List all hierarchical views |
| `ct.stats()` | Get index statistics |
| `ct.claims.get(claim_id)` | Get a specific claim |
| `ct.sources.get(doi)` | Get all claims from a paper |
| `ct.submit(doi)` | Submit a paper for extraction |

## Views

AskChem organizes claims into 5 hierarchical views:

- **by_reaction_type** — Chemical transformations (coupling, oxidation, etc.)
- **by_substance_class** — Molecules and materials (MOFs, polymers, etc.)
- **by_application** — Application domains (drug discovery, energy, etc.)
- **by_technique** — Experimental methods (NMR, DFT, electrochemistry, etc.)
- **by_mechanism** — Physical/chemical mechanisms (electron transfer, etc.)

## Migration: `search_grouped` removed in v0.3

`ct.search_grouped(query, view=...)` was a presentation wrapper over `ct.search(query, view=...)` and has been removed. The grouping is now done client-side over the same response, since `search()` already exploits the taxonomy as one of four RRF recall signals (FTS + paper-level + tree-recall + vector). To reproduce the old grouped output:

```python
from collections import defaultdict

def group_by_view(claims, view_id):
    tree = {}
    for c in claims:
        segs = (getattr(c, "view_paths", {}) or {}).get(view_id) or []
        segs = [s for s in segs if s not in ("not_applicable", "none")]
        if not segs:
            continue
        node = tree.setdefault(segs[0], {"_claims": [], "_children": {}})
        for seg in segs[1:]:
            node = node["_children"].setdefault(seg, {"_claims": [], "_children": {}})
        node["_claims"].append(c)
    return tree

results = ct.search("perovskite degradation", view="by_technique", limit=500)
grouped = group_by_view(results.claims, "by_technique")
```

For static taxonomy browsing (without a query), `ct.browse(view_id, path=..., depth=...)` is unchanged.

## License

MIT
