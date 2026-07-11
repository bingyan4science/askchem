# AskChem Agent Integration Guide

This document describes how to use AskChem as a tool for AI agents. It can be included in an agent's system prompt or tool-use schema.

## What is AskChem?

AskChem is a structured index of chemical knowledge extracted from research papers. Unlike Google Scholar or Semantic Scholar which return documents, AskChem returns **structured facts** — individual reactions, properties, mechanisms, and methods — each linked to its source paper.

## When to Use AskChem

Use AskChem when you need to:
- Check if a specific reaction or experiment has been reported
- Find all known methods to synthesize a molecule
- Discover what catalysts have been tried for a reaction type
- Identify research gaps or contradictions in a field
- Get structured data about chemical properties
- Verify a claim against primary literature

## API Base URL

```
http://localhost:8420
```

## Tool Descriptions (for function-calling agents)

### askchem_browse
Browse the hierarchical knowledge tree.
```json
{
  "name": "askchem_browse",
  "description": "Browse the AskChem knowledge hierarchy. Returns categories and claims organized by view.",
  "parameters": {
    "view": "by_reaction_type | by_substance_class | by_application | by_technique | by_mechanism",
    "path": "Slash-separated path (e.g., 'coupling/cross_coupling'). Empty for root.",
    "depth": "How many levels of children to show (0-5, default 1)"
  },
  "endpoint": "GET /tree/{view}/{path}?depth={depth}"
}
```

### askchem_search
Search for specific chemistry knowledge.
```json
{
  "name": "askchem_search",
  "description": "Search AskChem for claims matching a query. Supports molecule names, SMILES, reaction types, and natural language.",
  "parameters": {
    "query": "Search text (e.g., 'Suzuki coupling yield', 'MOF surface area', 'palladium catalyst')",
    "claim_type": "Optional filter: reaction | property | method | mechanism | comparison",
    "limit": "Max results (default 50)"
  },
  "endpoint": "GET /search?q={query}&claim_type={claim_type}&limit={limit}"
}
```

### askchem_get_claim
Get full details of a specific claim.
```json
{
  "name": "askchem_get_claim",
  "description": "Get the complete details of a claim including source DOI, verbatim quote, and all structured data.",
  "parameters": {
    "claim_id": "The 16-character hex claim identifier"
  },
  "endpoint": "GET /claims/{claim_id}"
}
```

### askchem_frontier
Find research gaps and contradictions.
```json
{
  "name": "askchem_frontier",
  "description": "Analyze a node for frontier indicators: sparse regions, contradictions, recent surges.",
  "parameters": {
    "view": "View to analyze in",
    "path": "Path to the node to analyze"
  },
  "endpoint": "GET /frontier/{view}/{path}"
}
```

### askchem_source
Get all knowledge extracted from a paper.
```json
{
  "name": "askchem_source",
  "description": "Get all claims extracted from a specific paper by DOI.",
  "parameters": {
    "doi": "Paper DOI (e.g., '10.1038/s41467-018-06019-1')"
  },
  "endpoint": "GET /sources/{doi}"
}
```

## Example Agent Workflows

### "Has anyone done a Suzuki coupling with this substrate?"
1. `askchem_search(query="Suzuki coupling aryl bromide", claim_type="reaction")`
2. Review results for matching substrates
3. `askchem_get_claim(claim_id=...)` for full details including conditions and yield

### "What's the state of the art for MOF drug delivery?"
1. `askchem_browse(view="by_application", path="pharmaceutical/drug_delivery", depth=2)`
2. Review the hierarchy to see what's been explored
3. `askchem_frontier(view="by_application", path="pharmaceutical/drug_delivery")` to find gaps

### "Does this result contradict existing literature?"
1. `askchem_search(query="<reaction description>", claim_type="reaction")`
2. Compare reported yields/conditions with the new result
3. Check `askchem_frontier` for known contradictions in the area

### "What catalysts have been tried for CO2 reduction?"
1. `askchem_browse(view="by_reaction_type", path="reduction", depth=2)`
2. Navigate to CO2-related nodes
3. `askchem_search(query="CO2 reduction catalyst", claim_type="reaction")`

## Important Notes

- All claims include `source_doi` and `verbatim_quote` — always verify important findings
- The index currently covers ~3,000+ claims from 500+ papers (and growing)
- Claims are extracted by GPT-5.4 (deep/PDF) and GPT-5-mini (abstracts) — cross-reference with sources
- The 5 views provide different lenses on the same data — try multiple views for comprehensive results
