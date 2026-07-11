# Tree Browse

Browse the hierarchical knowledge tree. The tree is zoomable: start at the root and navigate deeper.

## Browse Root

**Endpoint:** `GET /tree/{view_id}`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| view_id | path | yes | One of: `by_reaction_type`, `by_substance_class`, `by_application`, `by_technique`, `by_mechanism` |
| depth | query | no | How many levels of children to include (0-5, default 1) |

**Example Request:**
```
GET /tree/by_reaction_type?depth=2
```

**Example Response:**
```json
{
  "view": {
    "view_id": "by_reaction_type",
    "name": "By Reaction Type",
    "description": "Organizes claims by the type of chemical transformation"
  },
  "tree": {
    "node_id": "by_reaction_type_root",
    "name": "By Reaction Type",
    "path": [],
    "level": 0,
    "children": ["by_reaction_type_coupling", "by_reaction_type_reduction", ...],
    "children_data": [
      {
        "node_id": "by_reaction_type_coupling",
        "name": "Coupling",
        "path": ["coupling"],
        "level": 1,
        "claim_count": 5,
        "children_data": [
          {"name": "Cross Coupling", "claim_count": 2},
          {"name": "Decarboxylative Coupling", "claim_count": 1}
        ]
      }
    ]
  }
}
```

## Browse Specific Node

**Endpoint:** `GET /tree/{view_id}/{path}`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| view_id | path | yes | View identifier |
| path | path | yes | Slash-separated path to the node |
| depth | query | no | Levels of children to include (0-5, default 1) |

**Example Request:**
```
GET /tree/by_reaction_type/coupling/cross_coupling?depth=1
```

**Example Response:**
```json
{
  "view_id": "by_reaction_type",
  "path": ["coupling", "cross_coupling"],
  "node": {
    "node_id": "by_reaction_type_coupling_cross_coupling",
    "name": "Cross Coupling",
    "path": ["coupling", "cross_coupling"],
    "level": 2,
    "claim_count": 2,
    "claim_ids": ["81cb49ed5c7ce926", "a3f2b1c4d5e6f7a8"]
  },
  "claims": [
    {
      "claim_id": "81cb49ed5c7ce926",
      "claim_type": "reaction",
      "reaction_type": "deoxygenative C–C coupling",
      "source_doi": "10.1038/s41467-018-06019-1",
      "verbatim_quote": "Under the standard conditions, the corresponding ketone was obtained in 72% yield."
    }
  ],
  "total_claims": 2
}
```

**Agent usage notes:**
- Use `depth=0` when you only need the node metadata (fastest)
- Use `depth=1` to see immediate children (default, good for navigation)
- Use `depth=2-3` to get a broader view of a subtree
- Claims are included when browsing a specific node (up to 50)
