"""
AskChem MCP (Model Context Protocol) Server.

Exposes AskChem as a set of tools that MCP-compatible AI agents
(Cursor, Claude, etc.) can use natively.

Run with: python -m askchem.mcp_server
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from askchem.store import AskChemStore

INDEX_DIR = Path(__file__).parent.parent.parent / "chemtree_index"
PUBLIC_CONTENT_VIEWS = [
    "by_reaction_type",
    "by_substance_class",
    "by_technique",
    "by_application",
    "by_mechanism",
]


def create_server():
    store = AskChemStore(INDEX_DIR)
    server = Server("askchem")
    view_names = ", ".join(PUBLIC_CONTENT_VIEWS)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="askchem_browse",
                description=(
                    "Browse the AskChem knowledge hierarchy. "
                    f"Views: {view_names}. "
                    "Returns categories and claims organized hierarchically."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "view": {
                            "type": "string",
                            "description": f"View to browse: {view_names}",
                            "enum": PUBLIC_CONTENT_VIEWS,
                        },
                        "path": {
                            "type": "string",
                            "description": "Slash-separated path to navigate to (e.g., 'coupling/cross_coupling'). Empty for root.",
                            "default": "",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "How many levels of children to show (0-3)",
                            "default": 1,
                            "minimum": 0,
                            "maximum": 3,
                        },
                    },
                    "required": ["view"],
                },
            ),
            Tool(
                name="askchem_search",
                description=(
                    "Search AskChem for chemistry claims matching a query. "
                    "Supports molecule names, SMILES, reaction types, techniques, and natural language. "
                    "Returns structured claims with source provenance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search text (e.g., 'Suzuki coupling', 'MOF surface area', 'palladium catalyst')",
                        },
                        "claim_type": {
                            "type": "string",
                            "description": "Filter by claim type",
                            "enum": ["reaction", "property", "method", "mechanism", "comparison", "scope_entry", "computational_result"],
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="askchem_get_claim",
                description=(
                    "Get the full details of a specific claim by ID. "
                    "Includes source DOI, verbatim quote, structured data, and view assignments."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "description": "The 16-character hex claim identifier",
                        },
                    },
                    "required": ["claim_id"],
                },
            ),
            Tool(
                name="askchem_frontier",
                description=(
                    "Find research gaps and contradictions at a node in the hierarchy. "
                    "Returns frontier indicators: sparse regions, contradictions, temporal gaps."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "view": {
                            "type": "string",
                            "description": "View to analyze",
                            "enum": PUBLIC_CONTENT_VIEWS,
                        },
                        "path": {
                            "type": "string",
                            "description": "Slash-separated path to the node to analyze",
                        },
                    },
                    "required": ["view", "path"],
                },
            ),
            Tool(
                name="askchem_source",
                description=(
                    "Get all claims extracted from a specific paper by DOI. "
                    "Useful for verification and seeing everything AskChem knows from a paper."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "Paper DOI (e.g., '10.1038/s41467-018-06019-1')",
                        },
                    },
                    "required": ["doi"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "askchem_browse":
            view_id = arguments["view"]
            path_str = arguments.get("path", "")
            depth = min(arguments.get("depth", 1), 3)
            path_parts = [p for p in path_str.split("/") if p] if path_str else []

            tree = store.get_node_with_children(view_id, path_parts, depth=depth)
            if not tree:
                return [TextContent(type="text", text=f"Node not found: {view_id}/{path_str}")]

            claims_data = []
            for cid in (tree.get("claim_ids") or [])[:20]:
                claim = store.get_claim(cid)
                if claim:
                    claims_data.append(claim.to_dict())

            result = {"node": tree, "claims": claims_data}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "askchem_search":
            query = arguments["query"]
            claim_type = arguments.get("claim_type")
            limit = min(arguments.get("limit", 20), 50)

            results = store.search_claims(query, claim_type=claim_type, limit=limit)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "count": len(results),
                    "results": [c.to_dict() for c in results],
                }, indent=2),
            )]

        elif name == "askchem_get_claim":
            claim = store.get_claim(arguments["claim_id"])
            if not claim:
                return [TextContent(type="text", text=f"Claim not found: {arguments['claim_id']}")]
            return [TextContent(type="text", text=json.dumps(claim.to_dict(), indent=2))]

        elif name == "askchem_frontier":
            view_id = arguments["view"]
            path_str = arguments["path"]
            path_parts = [p for p in path_str.split("/") if p]
            indicators = store.get_frontier_indicators(view_id, path_parts)
            return [TextContent(type="text", text=json.dumps(indicators, indent=2))]

        elif name == "askchem_source":
            doi = arguments["doi"]
            claims = store.get_claims_for_source(doi)
            source = store.get_source_by_doi(doi)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "doi": doi,
                    "source": source.to_dict() if source else None,
                    "claims": [c.to_dict() for c in claims],
                    "count": len(claims),
                }, indent=2),
            )]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def main():
    if not HAS_MCP:
        print("MCP SDK not installed. Install with: pip install mcp")
        print("The AskChem MCP server requires the 'mcp' package.")
        return

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
