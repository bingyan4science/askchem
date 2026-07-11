# Frontier Detection

Analyze a node in the hierarchy for frontier indicators: sparse regions, contradictions, recent surges, and temporal gaps.

## Endpoint

`GET /frontier/{view_id}/{path}`

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| view_id | path | yes | View identifier |
| path | path | yes | Slash-separated path to the node to analyze |

## Example Request

```
GET /frontier/by_reaction_type/coupling/cross_coupling
```

## Example Response

```json
{
  "view_id": "by_reaction_type",
  "path": ["coupling", "cross_coupling"],
  "frontier_indicators": {
    "claim_count": 2,
    "source_count": 2,
    "is_sparse": true,
    "year_range": [2018, 2022],
    "has_contradictions": false
  }
}
```

## Frontier Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| claim_count | int | Number of claims at this node |
| source_count | int | Number of unique source papers |
| is_sparse | bool | True if fewer than 3 claims (potential research gap) |
| year_range | [int, int] | Earliest and latest publication years |
| has_contradictions | bool | True if claims report conflicting outcomes |
| contradiction_detail | string | Description of the contradiction (if any) |
| recent_surge | bool | True if publication rate increased recently |
| temporal_gap | bool | True if no new papers in 5+ years |

## Agent Usage Notes

- Use this to identify research opportunities: sparse nodes are underexplored
- Contradictions indicate areas where the science is unsettled
- Temporal gaps may indicate abandoned or solved problems
- Combine with tree browsing: first navigate to an area of interest, then check its frontier status
- A node with `is_sparse=true` and no `temporal_gap` is a prime research opportunity
