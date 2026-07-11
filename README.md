# AskChem

**A hierarchical, multi-view, source-grounded knowledge index for chemistry.**

AskChem structures the world's chemistry literature into a browsable, searchable, agent-usable knowledge base. Every claim is extracted from a real paper, classified into multiple hierarchical views, and linked back to its source.

**Live at [askchem.org](https://askchem.org)**

## What's Inside

- **2.4M+ structured claims** extracted from **147K+ papers** (1925-2026)
- **8 hierarchical views**: by reaction type, substance class, application, technique, mechanism, claim type, extracted data, and time period
- **Claim knowledge graph**: typed claim-to-claim edges (supports / contradicts / derives from)
- **Living Tree** (beta): an LLM-grown taxonomy of governing principles and mechanisms
- **Full-text + semantic search** (FTS5 + dense vectors fused via RRF)
- **Temporal evolution tracking**: see how any topic has evolved year-by-year
- **REST API** (OpenAPI docs), **Python SDK** (`pip install askchem`), and an **MCP server** for AI agents

## Quick Start

### Python SDK

```bash
pip install askchem
```

```python
from askchem import AskChem

ct = AskChem(base_url="https://askchem.org")

# Search for claims about Suzuki coupling
results = ct.search("Suzuki coupling palladium", limit=10)
for claim in results.claims:
    print(f"[{claim.claim_type}] {claim.verbatim_quote[:80]}...")

# Browse the knowledge tree
node = ct.browse("by_reaction_type", path="catalysis/cross_coupling", depth=2)

# Get all claims from a specific paper
paper = ct.sources.get("10.1021/jacs.2c12345")

# Submit a new paper for extraction
ct.submit("10.1038/s41586-024-07421-0")
```

### REST API

```bash
# Search
curl "https://askchem.org/api/search?q=MOF+CO2+reduction&limit=10"

# Browse tree
curl "https://askchem.org/api/tree/by_reaction_type?depth=2"

# Temporal evolution
curl "https://askchem.org/api/evolution/by_reaction_type/catalysis/cross_coupling"

# Browse by time period
curl "https://askchem.org/api/time?year=2024"
```

## Architecture

```
src/askchem/            # Core library
  models.py              # Data models (Claim, Source, TreeNode, View, PaperKnowledge)
  db.py                  # SQLite database layer (FTS5, temporal queries)
  server.py              # FastAPI REST API + web server
  llm.py                 # Centralized LLM client with content-addressable cache
  display.py             # Smart display names (preserves chemistry abbreviations)
  mcp_server.py          # Model Context Protocol server for AI agents
  indexer.py             # Claim classification and tree building

src/                     # Pipeline scripts
  download_pdfs.py       # PDF downloader (prioritized by citations)
  deep_extract.py        # Full-paper extraction pipeline (Batch API)
  process_corpus.py      # Bulk abstract extraction
  corpus_assembly.py     # Semantic Scholar corpus collection

sdk/                     # Python SDK (pip install askchem)
web/                     # Single-page application frontend
```

## Deployment (askchem.org)

### Prerequisites

- A VPS with Docker and Docker Compose installed
- The `askchem.org` domain DNS pointing to the server's IP (A record)
- The `chemtree.db` SQLite database file

### Deploy

```bash
# 1. Clone the repo on your server
git clone https://github.com/bing-yan/askchem.git /opt/askchem
cd /opt/askchem

# 2. Place the database file
mkdir -p data
# Download from HuggingFace or copy from your local machine:
# huggingface-cli download bing-yan/askchem chemtree.db --local-dir data/
cp /path/to/chemtree.db data/chemtree.db

# 3. Launch (Caddy handles HTTPS automatically)
docker compose up -d
```

That's it. Caddy will automatically obtain a Let's Encrypt certificate for `askchem.org` and serve the site over HTTPS.

### DNS Setup (Squarespace)

In your Squarespace Domains dashboard > `askchem.org` > DNS Settings:

1. Add an **A record**: Host `@`, Value = your server IP
2. Add a **CNAME**: Host `www`, Value = `askchem.org`
3. Remove any default Squarespace A/CNAME records

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
cd src && uvicorn askchem.server:app --host 0.0.0.0 --port 8420 --reload
# Open http://localhost:8420
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/search?q=...&view=...` | Hybrid search (FTS + paper-level + taxonomy + vector, fused via RRF). Optional `view` restricts to claims tagged in the selected hierarchy. |
| `GET /api/views` | List all hierarchical views |
| `GET /api/tree/{view_id}?depth=N` | Browse tree root |
| `GET /api/tree/{view_id}/{path}` | Browse a specific node |
| `GET /api/tree/{view_id}/{path}/temporal` | Year-by-year breakdown at a node |
| `GET /api/evolution/{view_id}/{path}` | Rich evolution timeline |
| `GET /api/time` | Browse by time period |
| `GET /api/claims/{id}` | Get claim details |
| `GET /api/sources/{doi}` | Get all claims from a paper |
| `GET /api/stats` | Index statistics |
| `POST /api/submit` | Submit a paper for extraction |
| `GET /api/docs` | Interactive API documentation |

All endpoints are also available under `/v1/` with API key authentication and rate limiting.

## License

MIT
