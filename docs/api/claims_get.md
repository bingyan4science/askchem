# Claims

Retrieve individual claims or list claims with filtering.

## Get a Specific Claim

**Endpoint:** `GET /claims/{claim_id}`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| claim_id | path | yes | The claim's unique identifier (16-char hex string) |

**Example:** `GET /claims/81cb49ed5c7ce926`

**Response:** Full claim object with all fields populated for its type.

## List Claims

**Endpoint:** `GET /claims`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| claim_type | query | no | Filter by type: reaction, property, method, mechanism, comparison, scope_entry, computational_result |
| limit | query | no | Max results (1-500, default 50) |
| offset | query | no | Skip first N results (default 0) |

**Example:** `GET /claims?claim_type=reaction&limit=10`

## Claim Schema

Every claim has these common fields:
```json
{
  "claim_id": "81cb49ed5c7ce926",
  "claim_type": "reaction",
  "source_doi": "10.1038/s41467-018-06019-1",
  "source_paper_title": "A general deoxygenation approach...",
  "confidence": "high",
  "location_in_paper": "Table 1, entry 1",
  "verbatim_quote": "Under the standard conditions...",
  "extraction_model": "gpt-5.4",
  "extraction_version": "v2",
  "view_paths": {
    "by_reaction_type": ["coupling", "deoxygenative_coupling"],
    "by_substance_class": ["organic", "carboxylic_acids"],
    "by_application": ["pharmaceutical", "drug_synthesis"]
  }
}
```

Additional fields depend on `claim_type` — see the main README for the full schema.

## Agent Usage Notes

- Use `claim_id` from tree browsing or search results to get full details
- The `view_paths` field shows where this claim sits in each hierarchy
- The `source_doi` and `verbatim_quote` enable verification against the original paper
- Use `claim_type` filter to narrow down to specific knowledge types
