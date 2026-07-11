"""AskChem MCP server (standalone, public-API edition).

A self-contained Model Context Protocol server that exposes AskChem's tools to
MCP clients (Claude Code, Claude Desktop, Cursor, ...). Unlike the in-repo
`askchem.mcp_server`, this file needs NO private code and NO local database - it
simply calls the public AskChem REST API (https://askchem.org) with your API key.
It works today and stays valid after the project is open-sourced.

Setup:
    pip install mcp
    export ASKCHEM_API_KEY="ac-..."      # create one at https://askchem.org (profile -> API Keys)
    # optional: export ASKCHEM_BASE_URL="https://askchem.org"

Claude Code:
    claude mcp add askchem --env ASKCHEM_API_KEY=ac-... -- python /path/to/askchem_mcp.py

Claude Desktop / Cursor (JSON):
    {"mcpServers": {"askchem": {"command": "python",
      "args": ["/path/to/askchem_mcp.py"],
      "env": {"ASKCHEM_API_KEY": "ac-..."}}}}

Anonymous use (no key) works at a lower rate limit if ASKCHEM_API_KEY is unset.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    sys.stderr.write("The 'mcp' package is required. Install with: pip install mcp\n")
    raise

BASE_URL = os.environ.get("ASKCHEM_BASE_URL", "https://askchem.org").rstrip("/")
API_KEY = os.environ.get("ASKCHEM_API_KEY", "").strip()
VIEWS = ["by_reaction_type", "by_substance_class", "by_application",
         "by_technique", "by_mechanism"]


def _get(path: str, params: dict | None = None) -> str:
    """GET {BASE_URL}{path}?params with the Bearer key; return the raw JSON text."""
    url = f"{BASE_URL}{path}"
    if params:
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "askchem-mcp/1.0")
    if API_KEY:
        req.add_header("Authorization", f"Bearer {API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        return json.dumps({"error": f"HTTP {e.code}", "detail": body[:500], "url": url})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e), "url": url})


def create_server():
    server = Server("askchem")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="askchem_search",
                description=(
                    "Search AskChem for source-grounded chemistry claims with verified DOIs. "
                    "Hybrid search over the full corpus (molecule names, SMILES, reactions, "
                    "techniques, natural language)."),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search text (e.g. 'Suzuki coupling', 'MOF surface area')"},
                        "view": {"type": "string", "description": "Optional view to scope results", "enum": VIEWS},
                        "claim_type": {"type": "string", "description": "Optional claim-type filter",
                                       "enum": ["reaction", "property", "method", "mechanism", "comparison", "scope_entry", "computational_result"]},
                        "limit": {"type": "integer", "description": "Max results (<=50)", "default": 20},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="askchem_get_claim",
                description="Get full details of one claim by its ID (source DOI, verbatim quote, structured data).",
                inputSchema={"type": "object", "properties": {
                    "claim_id": {"type": "string", "description": "Claim identifier"}}, "required": ["claim_id"]},
            ),
            Tool(
                name="askchem_source",
                description="Get all claims AskChem extracted from a paper, by DOI (for verification).",
                inputSchema={"type": "object", "properties": {
                    "doi": {"type": "string", "description": "Paper DOI, e.g. 10.1038/s41467-018-06019-1"}},
                    "required": ["doi"]},
            ),
            Tool(
                name="askchem_views",
                description="List the hierarchical views (taxonomies) available for browsing.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="askchem_browse",
                description=("Browse a taxonomy view at a given path. Returns child categories "
                             "and claim counts. Use askchem_views to see available views."),
                inputSchema={"type": "object", "properties": {
                    "view": {"type": "string", "description": "View id", "enum": VIEWS},
                    "path": {"type": "string", "description": "Slash-separated path; empty for root", "default": ""},
                    "depth": {"type": "integer", "description": "Levels of children (0-2)", "default": 1, "minimum": 0, "maximum": 2},
                }, "required": ["view"]},
            ),
            Tool(
                name="askchem_authors",
                description="Search authors / find experts by name and/or topic.",
                inputSchema={"type": "object", "properties": {
                    "query": {"type": "string", "description": "Author name (optional)"},
                    "topic": {"type": "string", "description": "Topic to rank experts by (optional)"},
                }},
            ),
            Tool(
                name="askchem_stats",
                description="Index statistics (claim/paper counts and coverage).",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "askchem_search":
            text = _get("/api/search", {
                "q": arguments["query"],
                "view": arguments.get("view"),
                "claim_type": arguments.get("claim_type"),
                "limit": min(int(arguments.get("limit", 20)), 50),
            })
        elif name == "askchem_get_claim":
            cid = urllib.parse.quote(str(arguments["claim_id"]), safe="")
            text = _get(f"/api/claims/{cid}")
        elif name == "askchem_source":
            doi = urllib.parse.quote(str(arguments["doi"]), safe="")
            text = _get(f"/api/sources/{doi}")
        elif name == "askchem_views":
            text = _get("/api/views")
        elif name == "askchem_browse":
            view = urllib.parse.quote(str(arguments["view"]), safe="")
            path = str(arguments.get("path", "")).strip("/")
            path_seg = ("/" + urllib.parse.quote(path)) if path else ""
            text = _get(f"/api/tree/{view}{path_seg}", {"depth": min(int(arguments.get("depth", 1)), 2)})
        elif name == "askchem_authors":
            text = _get("/api/authors", {"q": arguments.get("query"), "topic": arguments.get("topic")})
        elif name == "askchem_stats":
            text = _get("/api/stats")
        else:
            text = json.dumps({"error": f"Unknown tool: {name}"})
        return [TextContent(type="text", text=text)]

    return server


async def _main():
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
