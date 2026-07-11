# Search

Search for claims matching a text query across all claim fields.

## Endpoint

`GET /search`

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| q | query | yes | Search query (text, molecule name, SMILES, reaction type, etc.) |
| view | query | no | Filter to claims assigned to this view |
| claim_type | query | no | Filter by claim type: reaction, property, method, mechanism, comparison, scope_entry, computational_result |
| limit | query | no | Max results (1-200, default 50) |

## Example Requests

**Simple search:**
```
GET /search?q=palladium+coupling
```

**Filtered search:**
```
GET /search?q=surface+area&claim_type=property&limit=10
```

**Search within a view:**
```
GET /search?q=MOF&view=by_substance_class
```

## Example Response

```json
{
  "query": "palladium coupling",
  "results": [
    {
      "claim_id": "81cb49ed5c7ce926",
      "claim_type": "reaction",
      "reaction_type": "deoxygenative C–C coupling",
      "reactants": [
        {"name": "4-methylbenzoic acid", "smiles": "CC1=CC=C(C=C1)C(=O)O", "role": "substrate"}
      ],
      "products": [{"name": "ketone", "role": "major"}],
      "conditions": {"catalyst": "[Ir(dF(CF3)ppy)2(dtbbpy)]PF6", "solvent": "DCM/H2O"},
      "outcomes": {"yield_percent": 72},
      "source_doi": "10.1038/s41467-018-06019-1",
      "verbatim_quote": "Under the standard conditions, the corresponding ketone (3a) was obtained in 72% yield."
    }
  ],
  "count": 1,
  "filters": {"view": null, "claim_type": null}
}
```

## Agent Usage Notes

- Search is case-insensitive and matches against all claim fields
- For molecule searches, try both common names and SMILES
- Combine with `claim_type` filter for precision: use `claim_type=reaction` when looking for reactions, `claim_type=property` for measured values
- Results include full claim data — no need to call `/claims/{id}` separately
- For structured browsing, prefer the tree endpoints over search
