"""
Hierarchy Discovery for AskChem.

Takes extracted claims from v1 and v2, and experiments with:
1. Top-down: Map claims onto established chemical taxonomies
2. Bottom-up: Let GPT-4o cluster claims and propose hierarchy
3. Multiple views: Generate different hierarchical organizations

Uses GPT-4o to analyze claim collections and propose tree structures.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
HIERARCHY_DIR = EXPERIMENTS_DIR / "004_hierarchy_experiments"

client = OpenAI()


def load_all_claims():
    """Load all extracted claims from v1 and v2."""
    claims = []

    # v2 results (preferred — two-stage extraction)
    v2_path = EXPERIMENTS_DIR / "003_extraction_v2" / "results" / "all_extractions_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2_data = json.load(f)
        for paper in v2_data:
            if "error" in paper:
                continue
            paper_name = paper.get("paper_name", "unknown")
            s1 = paper.get("stage1", {}).get("result", {})
            paper_meta = s1.get("paper_metadata", {})
            for claim in paper.get("stage2", {}).get("result", {}).get("claims", []):
                claim["_source_paper"] = paper_name
                claim["_paper_type"] = paper_meta.get("paper_type", "unknown")
                claim["_paper_subfield"] = paper_meta.get("subfield", "unknown")
                claim["_extraction_version"] = "v2"
                claims.append(claim)

    # v1 single-pass results (supplement)
    v1_dir = EXPERIMENTS_DIR / "002_extraction_v1" / "raw"
    if v1_dir.exists():
        for f in sorted(v1_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            sp = data.get("single_pass", {})
            if "error" in sp:
                continue
            result = sp.get("result", {})
            paper_name = data.get("paper_name", f.stem)
            for claim in result.get("claims", []):
                claim["_source_paper"] = paper_name
                claim["_paper_subfield"] = result.get("subfield", "unknown")
                claim["_extraction_version"] = "v1"
                claims.append(claim)

    return claims


def generate_hierarchy_topdown(claims):
    """Use GPT-4o to map claims onto established chemical taxonomy."""
    claims_summary = json.dumps(claims[:80], indent=1)  # Send subset to fit context

    prompt = f"""You are a chemistry professor designing a hierarchical taxonomy for organizing chemical knowledge.

I have extracted {len(claims)} structured claims from 10 chemistry papers spanning organic synthesis, inorganic/materials, catalysis, physical chemistry, biochemistry, and computational chemistry.

Here is a representative sample of the claims (JSON):

{claims_summary}

Your task: Design a TOP-DOWN hierarchical taxonomy for organizing these claims, based on established chemical classification systems (IUPAC, textbook chapter structures, etc.).

Requirements:
1. The hierarchy should be 4-6 levels deep
2. Every claim in the sample should have a natural home in the hierarchy
3. The taxonomy should be extensible — it should accommodate claims from papers we haven't seen yet
4. Use established chemical terminology for node names

Return a JSON object with:
{{
  "taxonomy_name": "AskChem Top-Down Taxonomy v1",
  "description": "Brief description of the organizing principle",
  "tree": {{
    "chemistry": {{
      "children": {{
        "organic_chemistry": {{
          "children": {{
            "reactions": {{
              "children": {{
                "coupling_reactions": {{
                  "children": {{
                    "cross_coupling": {{}},
                    "C-H_activation": {{}}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
  }},
  "claim_assignments": [
    {{"claim_id": 1, "path": ["chemistry", "organic_chemistry", "reactions", "coupling_reactions", "cross_coupling"]}},
    ...
  ],
  "unassigned_claims": ["list of claim_ids that don't fit well"],
  "gaps": ["areas in the taxonomy with no claims — potential frontier areas"]
}}"""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=16000,
        response_format={"type": "json_object"},
    )

    return {
        "method": "top_down",
        "result": json.loads(response.choices[0].message.content),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def generate_hierarchy_bottomup(claims):
    """Use GPT-4o to cluster claims and propose emergent hierarchy."""
    claims_summary = json.dumps(claims[:80], indent=1)

    prompt = f"""You are a data scientist analyzing {len(claims)} structured knowledge claims extracted from chemistry papers.

Here is a representative sample:

{claims_summary}

Your task: Perform BOTTOM-UP clustering. Group these claims by similarity and propose a hierarchy that EMERGES from the data, rather than being imposed from a textbook.

Instructions:
1. First, identify natural clusters among the claims (what claims are most similar to each other?)
2. Then, group clusters into super-clusters
3. Build a hierarchy from the bottom up
4. Name each node descriptively based on what the claims in it share
5. The hierarchy should be 3-5 levels deep

Return a JSON object with:
{{
  "taxonomy_name": "AskChem Bottom-Up Clustering v1",
  "description": "How the hierarchy was derived from the data",
  "clusters": [
    {{
      "cluster_id": 1,
      "name": "descriptive name",
      "description": "what claims in this cluster share",
      "claim_ids": [1, 2, 3],
      "parent_cluster": null or cluster_id
    }}
  ],
  "tree": {{
    "root": {{
      "children": {{
        "cluster_name": {{
          "children": {{...}}
        }}
      }}
    }}
  }},
  "insights": ["interesting patterns observed during clustering"],
  "claims_that_bridge_clusters": ["claims that could belong to multiple clusters"]
}}"""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=16000,
        response_format={"type": "json_object"},
    )

    return {
        "method": "bottom_up",
        "result": json.loads(response.choices[0].message.content),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def generate_multiview_hierarchies(claims):
    """Use GPT-4o to propose multiple overlapping hierarchical views."""
    claims_summary = json.dumps(claims[:80], indent=1)

    prompt = f"""You are designing a MULTI-VIEW hierarchical index for chemical knowledge. The same set of claims should be organizable through MULTIPLE different hierarchies, each providing a different lens.

Here are {len(claims)} claims from 10 chemistry papers:

{claims_summary}

Design 5 different hierarchical views, each organizing the SAME claims differently:

1. **By Reaction/Transformation Type** — How the chemistry happens (coupling, oxidation, reduction, etc.)
2. **By Substance/Material Class** — What molecules/materials are involved (aromatics, MOFs, nanoparticles, etc.)
3. **By Application Domain** — What the chemistry is used for (drug synthesis, energy, materials, etc.)
4. **By Technique/Method** — How the work was done (spectroscopy, catalysis, computation, etc.)
5. **By Phenomenon/Mechanism** — What physical/chemical principles are at play (electron transfer, radical, etc.)

For each view, provide:
- A 3-5 level hierarchy tree
- How the sample claims map into it
- What insights this particular view reveals that others don't

Return a JSON object with:
{{
  "views": [
    {{
      "view_name": "by_reaction_type",
      "description": "Organizes claims by the type of chemical transformation",
      "tree": {{
        "all_transformations": {{
          "children": {{
            "bond_formation": {{
              "children": {{
                "C-C_bond": {{
                  "children": {{
                    "cross_coupling": {{}},
                    "C-H_activation": {{}}
                  }}
                }}
              }}
            }}
          }}
        }}
      }},
      "sample_assignments": [
        {{"claim_id": 1, "path": ["all_transformations", "bond_formation", "C-C_bond", "cross_coupling"]}}
      ],
      "unique_insights": "What this view reveals that others don't"
    }}
  ],
  "cross_view_insights": ["Insights from comparing how the same claim appears in different views"],
  "recommended_primary_view": "which view should be the default and why"
}}"""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=16000,
        response_format={"type": "json_object"},
    )

    return {
        "method": "multi_view",
        "result": json.loads(response.choices[0].message.content),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def main():
    os.makedirs(HIERARCHY_DIR / "raw", exist_ok=True)
    os.makedirs(HIERARCHY_DIR / "results", exist_ok=True)

    print(f"Hierarchy Discovery - {datetime.now().isoformat()}", flush=True)

    # Load claims
    claims = load_all_claims()
    print(f"Loaded {len(claims)} total claims", flush=True)

    # Count by type and subfield
    by_type = {}
    by_subfield = {}
    for c in claims:
        ct = c.get("claim_type", "unknown")
        by_type[ct] = by_type.get(ct, 0) + 1
        sf = c.get("_paper_subfield", "unknown")
        by_subfield[sf] = by_subfield.get(sf, 0) + 1

    print(f"By type: {by_type}", flush=True)
    print(f"By subfield: {by_subfield}", flush=True)

    # Save claims for reference
    with open(HIERARCHY_DIR / "raw" / "all_claims_combined.json", "w") as f:
        json.dump(claims, f, indent=2)

    results = {}

    # 1. Top-down taxonomy
    print(f"\n{'='*60}", flush=True)
    print("Experiment 1: Top-Down Taxonomy", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        topdown = generate_hierarchy_topdown(claims)
        results["top_down"] = topdown
        tree = topdown["result"].get("tree", {})
        print(f"  Tokens: {topdown['usage']['total_tokens']}", flush=True)
        print(f"  Top-level categories: {list(list(tree.values())[0].get('children', {}).keys()) if tree else 'N/A'}", flush=True)
        with open(HIERARCHY_DIR / "raw" / "topdown_result.json", "w") as f:
            json.dump(topdown, f, indent=2)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results["top_down"] = {"error": str(e)}

    import time; time.sleep(3)

    # 2. Bottom-up clustering
    print(f"\n{'='*60}", flush=True)
    print("Experiment 2: Bottom-Up Clustering", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        bottomup = generate_hierarchy_bottomup(claims)
        results["bottom_up"] = bottomup
        clusters = bottomup["result"].get("clusters", [])
        print(f"  Tokens: {bottomup['usage']['total_tokens']}", flush=True)
        print(f"  Clusters found: {len(clusters)}", flush=True)
        for c in clusters[:5]:
            print(f"    - {c.get('name', '?')}: {len(c.get('claim_ids', []))} claims", flush=True)
        with open(HIERARCHY_DIR / "raw" / "bottomup_result.json", "w") as f:
            json.dump(bottomup, f, indent=2)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results["bottom_up"] = {"error": str(e)}

    import time; time.sleep(3)

    # 3. Multi-view hierarchies
    print(f"\n{'='*60}", flush=True)
    print("Experiment 3: Multi-View Hierarchies", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        multiview = generate_multiview_hierarchies(claims)
        results["multi_view"] = multiview
        views = multiview["result"].get("views", [])
        print(f"  Tokens: {multiview['usage']['total_tokens']}", flush=True)
        print(f"  Views generated: {len(views)}", flush=True)
        for v in views:
            print(f"    - {v.get('view_name', '?')}: {v.get('description', '?')[:60]}", flush=True)
        rec = multiview["result"].get("recommended_primary_view", "?")
        print(f"  Recommended primary view: {rec}", flush=True)
        with open(HIERARCHY_DIR / "raw" / "multiview_result.json", "w") as f:
            json.dump(multiview, f, indent=2)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results["multi_view"] = {"error": str(e)}

    # Save combined results
    with open(HIERARCHY_DIR / "results" / "all_hierarchy_experiments.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("HIERARCHY DISCOVERY COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
