# AskChem — Structured Chemistry Knowledge API

AskChem is a structured chemistry knowledge index with source-grounded, DOI-verified claims extracted from chemistry papers across 14 subfields.

**Base URL:** `https://askchem.org`
**OpenAPI:** `https://askchem.org/api/docs`

## Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/search?q={query}&view={view_id}&limit=50` | Hybrid search across all claims (FTS + paper-level + taxonomy + vector, fused via RRF). Optional `view` filters claims to those tagged in the selected hierarchy. |
| GET | `/api/views` | List hierarchical views |
| GET | `/api/tree/{view_id}/{path}?depth=1` | Browse taxonomy tree |
| GET | `/api/claims/{claim_id}` | Get full claim details |
| GET | `/api/sources/{doi}` | All claims from a paper |
| GET | `/api/authors?q={name}&topic={topic}` | Search authors / find experts |
| GET | `/api/authors/{id}` | Author profile |
| GET | `/api/papers?q={query}` | Browse indexed papers |
| GET | `/api/stats` | Index statistics |
| GET | `/api/temporal/{view_id}/{path}` | Temporal claim overlay |
| GET | `/api/evolution/{view_id}/{path}` | Topic evolution timeline |
| GET | `/api/contradictions/{view_id}/{path}` | Find contradictory claims |
| GET | `/api/quality` | Data quality report |
| GET | `/api/benchmark` | AskChem-Bench methodology and results |
| GET | `/api/feed` | Recent discoveries |
| POST | `/api/submit` | Submit a paper by DOI |
| POST | `/api/flag` | Flag a claim for review |

## Quick Example

```bash
curl -s "https://askchem.org/api/search?q=suzuki+coupling&limit=3" | python -m json.tool
```

## Authentication

- **Anonymous:** 60 requests/min
- **API key:** `Authorization: Bearer ac-...` — 300 requests/min
- Request a key: `POST /api/keys/request` with `{"name": "...", "email": "...", "use_case": "..."}`

## Views (hierarchical taxonomies)

- `by_reaction_type` — Reaction classes and subclasses
- `by_substance_class` — Materials and substance categories
- `by_technique` — Analytical and experimental methods
- `by_application` — Application domains
- `by_mechanism` — Phenomena and mechanisms
- `by_claim_type` — Claim types (experimental, method, property, etc.)
- `by_author` — Author expertise map

## When to use AskChem

- Literature-backed answers with verified DOI citations
- Cross-paper condition aggregation (catalysts, yields, conditions)
- Temporal tracking of how understanding evolved
- Contradiction surfacing across papers
- Structured browsing of chemistry knowledge

## Ethics

Always cite source DOIs when presenting claims. Do not fabricate or embellish data.
Claims are extracted by LLM and may contain errors — verify critical findings against original papers.
