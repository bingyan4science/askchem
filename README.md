# AskChem

**A claim-centered index for cross-paper chemistry search.**

AskChem changes the unit of retrieval from the *paper* to the **provenance-carrying claim**: each paper is segmented into atomic, typed claims, each grounded by a verbatim source quote and a DOI. Over this shared claim store, AskChem exposes complementary structures for search and synthesis, so scientists and AI agents can retrieve specific findings, inspect their evidence, and assemble cross-paper answers without first reading and filtering whole documents.

**Live at [askchem.org](https://askchem.org)** · **Dataset on [HuggingFace](https://huggingface.co/datasets/bing-yan/askchem)**

## What's Inside

- **2.44M provenance-carrying claims** from **146.6K papers** (1925–2026) — each is a typed assertion with a verbatim quote and source DOI.
- **Three complementary structures over one shared claim store:**
  - **Stabilized faceted taxonomy** — corpus-induced, normalized L1/L2/L3 paths exposed as **10 navigational views**: reaction type, substance class, application, technique, mechanism, extracted data, claim type, and time, plus an **author** (coauthor graph) view and a **network** (evidence-graph) view. ~306.9K taxonomy nodes.
  - **Evidence graph** — ~171K typed claim-to-claim relations (`supports`, `contradicts`, `extends`, `derives_from`, `cites_as_evidence`); 97.9% edge-type precision on an expert audit.
  - **Living Taxonomy** (exploratory) — a principle-centered hierarchy (principles → theories → models → mechanisms → phenomena) that situates paper-grounded claims under the scientific ideas that govern them. Currently situates ~1M+ claims across the reaction, substance, technique, and mechanism views.
- **Hybrid search** — FTS5 full-text + paper-level recall + taxonomy-node recall + dense vectors, fused via reciprocal rank fusion (RRF), with cross-encoder reranking.
- **Temporal tracking** — see how any topic has evolved year by year.
- **Access everywhere** — Web UI, REST API (OpenAPI), Python SDK (`pip install askchem`), and an **MCP server** for AI agents.
- **AskChem-Bench** — a cross-paper chemistry search evaluation measuring citation groundedness and relevance.

## Quick Start

### Python SDK

```bash
pip install askchem
```

```python
from askchem import AskChem

ct = AskChem(base_url="https://askchem.org")  # or AskChem(api_key="ac-...") for higher limits

# Search for claims (each carries a verbatim quote + source DOI)
results = ct.search("Suzuki coupling palladium", limit=10)
for claim in results.claims:
    print(f"[{claim.claim_type}] {claim.verbatim_quote[:80]}...  ({claim.source_doi})")

# Browse the faceted taxonomy
node = ct.browse("by_reaction_type", path="coupling/cross_coupling", depth=2)

# Get all claims from a specific paper
paper = ct.sources.get("10.1021/jacs.2c12345")

# Index statistics and the list of views
print(ct.stats())
print([v.view_id for v in ct.views()])

# Submit a new paper for extraction
ct.submit("10.1038/s41586-024-07421-0")
```

### REST API

```bash
# Hybrid search
curl "https://askchem.org/api/search?q=MOF+CO2+reduction&limit=10"

# Browse a faceted view
curl "https://askchem.org/api/tree/by_reaction_type?depth=2"

# Evidence graph around a claim
curl "https://askchem.org/api/claims/{claim_id}/neighborhood"

# Living Taxonomy (principle-centered)
curl "https://askchem.org/api/ltree/by_mechanism/root?depth=1"

# Temporal evolution of a topic
curl "https://askchem.org/api/evolution/by_reaction_type/coupling/cross_coupling"
```

### AI agents (MCP)

AskChem ships an **MCP server** so coding/chat agents (Cursor, Claude, Copilot, …) can use it as a structured chemistry knowledge source. Point your agent at [`AGENTS.md`](AGENTS.md), or see the **Agents** tab on [askchem.org](https://askchem.org) and the OpenAPI docs at `/api/docs`.

## Architecture

```
src/askchem/             # Core library (FastAPI app + retrieval + serving)
  server.py              # REST API + web server
  db.py                  # SQLite layer (FTS5, taxonomy, temporal, authors)
  retrieval.py           # Hybrid retrieval (FTS + vector + tree + RRF)
  cross_encoder_rerank.py# Cross-encoder reranking
  embeddings.py / embeddings_v2.py   # Dense claim embeddings + FAISS index
  ltree.py               # Living Taxonomy serving (nodes, paths, semantic routing)
  advisor.py             # Paper intelligence (critique / contribution / advisor)
  taxonomy.py / canonical_l3.py       # Canonical faceted taxonomy (L1/L2/L3)
  models.py              # Data models (Claim, Source, TreeNode, View, ...)
  indexer.py             # Claim classification + tree building
  mcp_server.py          # Model Context Protocol server for AI agents
  validation.py display.py llm.py     # Schema validation, display names, LLM client

living_taxonomy/         # Living Taxonomy pipeline (principle-centered tree)
  seed_scaffold.py view_layers.py     # Scaffold of principles/theories/mechanisms
  batch_place.py grow_onto_scaffold.py# Embedding-shortlist + LLM claim placement
  consolidate.py semantic_dedup.py combine_nodes.py   # Cleanup / dedup / grouping
  enrich_nodes.py audit_nodes.py      # Node short-labels, equations, definitions
  merge_grown.py dedup_safe.py        # Incremental merge + cycle-safe dedup
  apply_to_db.py build_node_index.py  # Persist to DB + build node search index

src/                     # Corpus + extraction + graph pipeline scripts
  corpus_assembly.py download_pdfs.py process_corpus.py   # Corpus collection
  extract_claims.py deep_extract.py batch_extract_*.py    # Claim extraction
  classify_papers.py reclassify_*.py                      # Faceted classification
  cross_citation_extractor*.py grade_edges.py             # Evidence graph
  build_author_index.py upload_to_hf.py                   # Authors + dataset publish

sdk/                     # Python SDK (pip install askchem)
web/                     # Single-page web application
scripts/                 # Figures, benchmarks, and one-off utilities
tests/                   # Test suite
deploy/  docs/           # Deployment configs and documentation
```

## Deployment (askchem.org)

### Prerequisites

- A VPS with Docker and Docker Compose installed
- The `askchem.org` domain DNS pointing to the server's IP (A record)
- The `askchem.db` SQLite database file (hosted on HuggingFace)

### Deploy

```bash
# 1. Clone the repo on your server
git clone https://github.com/bingyan4science/askchem.git /opt/askchem
cd /opt/askchem

# 2. Fetch the database (SQLite + FTS5 + baked-in taxonomy)
huggingface-cli download bing-yan/askchem askchem.db --local-dir .

# 3. Launch (Caddy handles HTTPS automatically)
docker compose up -d
```

Caddy will automatically obtain a Let's Encrypt certificate for `askchem.org` and serve the site over HTTPS. (`deploy_to_vps.sh` automates a full server deploy; set `VPS` to your own host — the released copy uses a `YOUR_VPS_HOST` placeholder.)

### Local Development

```bash
pip install -r requirements.txt

# Place askchem.db in the repo root (see step 2 above), then:
cd src && uvicorn askchem.server:app --host 0.0.0.0 --port 8420 --reload
# Open http://localhost:8420
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/search?q=...&view=...` | Hybrid search (FTS + paper-level + taxonomy + vector, fused via RRF). Optional `view` restricts to a facet. |
| `GET /api/views` | List all hierarchical views |
| `GET /api/tree/{view_id}?depth=N` / `GET /api/tree/{view_id}/{path}` | Browse a faceted view / node |
| `GET /api/temporal/{view_id}/{path}` | Year-by-year breakdown at a node |
| `GET /api/evolution/{view_id}/{path}` | Rich evolution timeline |
| `GET /api/contradictions/{view_id}/{path}` | Contradictory claims under a node |
| `GET /api/time` | Browse by time period (decade → year) |
| `GET /api/claims/{id}` | Claim details |
| `GET /api/claims/{id}/neighborhood` | Inbound/outbound evidence-graph edges for a claim |
| `GET /api/sources/{doi}` | All claims from a paper |
| `GET /api/authors?q=...` / `GET /api/authors/{id}/network` | Author search / coauthor network |
| `GET /api/ltree/{view_id}/root` · `/node/{node_id}` · `/search` | Living Taxonomy browse + semantic routing |
| `GET /api/stats` · `GET /api/quality` · `GET /api/feed` | Index stats / quality report / discoveries feed |
| `GET /api/benchmark` | AskChem-Bench methodology and results |
| `POST /api/submit` · `POST /api/flag` | Submit a paper / flag a claim |
| `GET /api/docs` | Interactive OpenAPI documentation |

Anonymous access is rate-limited to 60 requests/min; an API key (`Authorization: Bearer ac-...`) raises it to 300/min. All endpoints are also available under `/v1/` with key authentication.

## Links

- Live system: <https://askchem.org>
- API documentation: <https://askchem.org/api/docs>
- Benchmark (AskChem-Bench): <https://askchem.org/api/benchmark>
- Dataset: <https://huggingface.co/datasets/bing-yan/askchem>

## License

Software is released under the **MIT License** (see [LICENSE](LICENSE)). The published dataset on HuggingFace is released under **CC-BY-4.0**.

> Claims are extracted by LLMs and may contain errors — every claim carries a verbatim quote and source DOI so you can verify it against the original paper. Do not present extracted claims as peer-reviewed without checking the cited source.
