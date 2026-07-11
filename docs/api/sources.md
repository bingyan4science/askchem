# Sources

Get all claims extracted from a specific source paper.

## Endpoint

`GET /sources/{doi}`

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| doi | path | yes | The paper's DOI (e.g., `10.1038/s41467-018-06019-1`) |

## Example Request

```
GET /sources/10.1038/s41467-018-06019-1
```

## Example Response

```json
{
  "doi": "10.1038/s41467-018-06019-1",
  "source": {
    "doi": "10.1038/s41467-018-06019-1",
    "title": "A general deoxygenation approach for synthesis of ketones...",
    "authors": ["Author A", "Author B"],
    "year": 2018,
    "venue": "Nature Communications",
    "citation_count": 193
  },
  "claims": [
    {"claim_id": "...", "claim_type": "reaction", ...},
    {"claim_id": "...", "claim_type": "scope_entry", ...},
    {"claim_id": "...", "claim_type": "mechanism", ...}
  ],
  "count": 10
}
```

## Agent Usage Notes

- Use this to see everything AskChem knows from a particular paper
- Useful for verification: compare extracted claims against the original
- One paper can contribute many claims across different types and views
- The `source` object includes citation count for assessing paper impact
