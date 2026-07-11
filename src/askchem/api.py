"""
AskChem API: Agent-first REST API for browsing the chemical knowledge index.

All endpoints return JSON. The website is a client of this API.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from askchem.store import AskChemStore

INDEX_DIR = Path(__file__).parent.parent.parent / "chemtree_index"
store = AskChemStore(INDEX_DIR)

app = FastAPI(
    title="AskChem API",
    description=(
        "A hierarchical, multi-view, source-grounded index of chemical knowledge. "
        "Browse the tree, search claims, detect frontiers and contradictions. "
        "Designed for AI agents and human scientists."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://askchem.org",
        "https://www.askchem.org",
        "http://localhost:8080",
        "http://localhost:8420",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- Views ---

@app.get("/views", summary="List all available views")
def list_views():
    """
    List all hierarchical views available in the index.

    Each view organizes the same set of claims by a different principle
    (reaction type, substance class, application, technique, mechanism).

    **Agent usage:** Call this first to discover available views before browsing.
    """
    views = store.list_views()
    return {
        "views": [v.to_dict() for v in views],
        "count": len(views),
    }


# --- Tree Browsing ---

@app.get("/tree/{view_id}", summary="Browse the tree root of a view")
def get_tree_root(view_id: str, depth: int = Query(1, ge=0, le=5)):
    """
    Get the root of a view's hierarchy with children up to the specified depth.

    **Agent usage:** Start here to explore a view. Use depth=1 to see top-level
    categories, depth=2 to see two levels, etc.
    """
    view = store.get_view(view_id)
    if not view:
        raise HTTPException(404, f"View '{view_id}' not found")

    tree = store.get_node_with_children(view_id, [], depth=depth)
    return {
        "view": view.to_dict(),
        "tree": tree,
    }


@app.get("/tree/{view_id}/{path:path}", summary="Browse a specific node in the tree")
def get_tree_node(view_id: str, path: str, depth: int = Query(1, ge=0, le=5)):
    """
    Get a specific node in the tree hierarchy, with children up to the given depth.

    The path is slash-separated, e.g., `/tree/by_reaction_type/coupling/cross_coupling`.

    **Agent usage:** Navigate deeper into the tree by following paths from parent nodes.
    Use depth=0 to get just the node metadata, depth=1 to see immediate children.
    """
    view = store.get_view(view_id)
    if not view:
        raise HTTPException(404, f"View '{view_id}' not found")

    path_parts = [p for p in path.split("/") if p]
    node = store.get_node_with_children(view_id, path_parts, depth=depth)
    if not node:
        raise HTTPException(404, f"Node not found at path: {'/'.join(path_parts)}")

    # Also fetch claims at this node
    claim_ids = node.get("claim_ids", [])
    claims_data = []
    for cid in claim_ids[:50]:
        claim = store.get_claim(cid)
        if claim:
            claims_data.append(claim.to_dict())

    return {
        "view_id": view_id,
        "path": path_parts,
        "node": node,
        "claims": claims_data,
        "total_claims": len(claim_ids),
    }


# --- Claims ---

@app.get("/claims/{claim_id}", summary="Get a specific claim by ID")
def get_claim(claim_id: str):
    """
    Retrieve the full details of a specific claim, including all metadata,
    source provenance, and view assignments.

    **Agent usage:** Use this to get the full details of a claim found via
    tree browsing or search. The claim includes the verbatim quote from
    the source paper and the DOI for verification.
    """
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(404, f"Claim '{claim_id}' not found")
    return claim.to_dict()


@app.get("/claims", summary="List claims with optional filtering")
def list_claims(
    claim_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    List claims with optional filtering by type.

    Claim types: reaction, property, method, mechanism, comparison,
    scope_entry, computational_result, structure.

    **Agent usage:** Use this to browse claims directly without going
    through the tree hierarchy.
    """
    claims = store.list_claims(limit=limit + 100, offset=offset)
    if claim_type:
        claims = [c for c in claims if c.claim_type == claim_type]
    claims = claims[:limit]
    return {
        "claims": [c.to_dict() for c in claims],
        "count": len(claims),
        "offset": offset,
        "limit": limit,
    }


# --- Search ---

@app.get("/search", summary="Search claims by text query")
def search_claims(
    q: str = Query(..., description="Search query (text, molecule name, SMILES, etc.)"),
    view: Optional[str] = None,
    claim_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """
    Search for claims matching a text query. Searches across all claim fields
    including molecule names, reaction types, conditions, and verbatim quotes.

    Supports: molecule names, SMILES strings, reaction type names, technique names,
    property names, and natural language queries.

    **Agent usage:** Use this for targeted queries like "Suzuki coupling yield",
    "MOF surface area", or "palladium catalyst". Combine with view and claim_type
    filters for more precise results.
    """
    results = store.search_claims(q, view=view, claim_type=claim_type, limit=limit)
    return {
        "query": q,
        "results": [c.to_dict() for c in results],
        "count": len(results),
        "filters": {"view": view, "claim_type": claim_type},
    }


# --- Sources ---

@app.get("/sources/{doi:path}", summary="Get all claims from a source paper")
def get_source_claims(doi: str):
    """
    Get all claims extracted from a specific source paper, identified by DOI.

    **Agent usage:** Use this to see everything AskChem knows from a particular
    paper. Useful for verification — compare extracted claims against the original.
    """
    claims = store.get_claims_for_source(doi)
    source = store.get_source_by_doi(doi)
    return {
        "doi": doi,
        "source": source.to_dict() if source else None,
        "claims": [c.to_dict() for c in claims],
        "count": len(claims),
    }


# --- Frontier Detection ---

@app.get("/frontier/{view_id}/{path:path}", summary="Get frontier indicators for a node")
def get_frontier(view_id: str, path: str):
    """
    Analyze a node for frontier indicators: sparse regions, contradictions,
    recent surges, and temporal gaps.

    **Agent usage:** Use this to find unexplored areas of chemistry or areas
    with conflicting findings. This is one of AskChem's most powerful features
    for identifying research opportunities.
    """
    path_parts = [p for p in path.split("/") if p]
    indicators = store.get_frontier_indicators(view_id, path_parts)
    if not indicators:
        raise HTTPException(404, f"Node not found")
    return {
        "view_id": view_id,
        "path": path_parts,
        "frontier_indicators": indicators,
    }


# --- Metadata ---

@app.get("/", summary="Index metadata and quick start")
def get_index_info():
    """
    Get metadata about the AskChem index: total claims, sources, views,
    and a quick-start guide for agents.

    **Agent usage:** Call this first to understand the index scope and
    available endpoints.
    """
    metadata = store.get_metadata()
    views = store.list_views()
    return {
        "name": "AskChem",
        "description": "A hierarchical, multi-view, source-grounded index of chemical knowledge",
        "version": metadata.get("version", "0.1.0"),
        "stats": {
            "total_claims": store.count_claims(),
            "total_sources": metadata.get("source_count", 0),
            "total_views": len(views),
        },
        "views": [{"id": v.view_id, "name": v.name, "description": v.description} for v in views],
        "quick_start": {
            "1_list_views": "GET /views",
            "2_browse_tree": "GET /tree/{view_id}?depth=2",
            "3_zoom_in": "GET /tree/{view_id}/{path}?depth=1",
            "4_get_claim": "GET /claims/{claim_id}",
            "5_search": "GET /search?q=your+query",
            "6_find_frontiers": "GET /frontier/{view_id}/{path}",
        },
    }
