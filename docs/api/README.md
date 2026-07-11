# AskChem API Documentation

AskChem is a hierarchical, multi-view, source-grounded index of chemical knowledge extracted from the research literature. This API is designed for AI agents and human scientists.

## Base URL

```
http://localhost:8420
```

## Quick Start

1. **Discover the index:** `GET /` — returns metadata, stats, and available views
2. **List views:** `GET /views` — see all 5 hierarchical views
3. **Browse a view:** `GET /tree/by_reaction_type?depth=2` — see the tree structure
4. **Zoom in:** `GET /tree/by_reaction_type/coupling/cross_coupling?depth=1` — navigate deeper
5. **Get a claim:** `GET /claims/{claim_id}` — full claim with provenance
6. **Search:** `GET /search?q=Suzuki+coupling` — text search across all claims
7. **Find frontiers:** `GET /frontier/by_reaction_type/coupling` — unexplored areas

## Core Concepts

### Claims
The atomic unit of knowledge. Each claim is a structured fact extracted from a paper:
- **reaction:** A chemical transformation with reactants, products, conditions, yield
- **property:** A measured/computed value (e.g., melting point, BET surface area)
- **method:** A technique or approach (e.g., operando Raman spectroscopy)
- **mechanism:** A proposed explanation for how a process works
- **comparison:** A direct comparison between methods/catalysts/materials
- **scope_entry:** A single entry from a substrate scope table
- **computational_result:** A computed prediction or simulation result

### Views
Five hierarchical organizations of the same claims:
- **by_reaction_type:** How the chemistry happens (coupling, oxidation, reduction...)
- **by_substance_class:** What molecules/materials are involved (organics, MOFs, nanoparticles...)
- **by_application:** What it's used for (drug synthesis, energy, materials...)
- **by_technique:** How the work was done (spectroscopy, catalysis, computation...)
- **by_mechanism:** What principles are at play (electron transfer, radical, catalytic cycle...)

### Sources
Every claim links back to its source paper via DOI, with a verbatim quote for verification.

## Endpoints

| Method | Endpoint | Description | Doc |
|--------|----------|-------------|-----|
| GET | `/` | Index metadata and quick start | [index.md](index.md) |
| GET | `/views` | List all views | [views.md](views.md) |
| GET | `/tree/{view_id}` | Browse tree root | [tree_browse.md](tree_browse.md) |
| GET | `/tree/{view_id}/{path}` | Browse specific node | [tree_browse.md](tree_browse.md) |
| GET | `/claims/{claim_id}` | Get a specific claim | [claims_get.md](claims_get.md) |
| GET | `/claims` | List claims with filters | [claims_get.md](claims_get.md) |
| GET | `/search` | Search claims | [search.md](search.md) |
| GET | `/sources/{doi}` | Get claims from a paper | [sources.md](sources.md) |
| GET | `/frontier/{view_id}/{path}` | Frontier indicators | [frontier.md](frontier.md) |

## For AI Agents

If you are an AI agent using AskChem as a tool:

1. Start with `GET /` to understand the index scope
2. Use `GET /search?q=...` for targeted queries
3. Use `GET /tree/{view_id}?depth=2` to explore the hierarchy
4. Use `GET /claims/{claim_id}` to get full details including source DOI
5. Use `GET /frontier/{view_id}/{path}` to find research gaps

All responses are JSON. All claims include `source_doi` and `verbatim_quote` for verification.
