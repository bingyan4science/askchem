"""
AskChem Indexer: Populates the store from extracted claims.

Uses GPT-4o to classify each claim into the appropriate position
in each of the 5 views, then builds the filesystem hierarchy.
"""

import json
import time
from pathlib import Path
from datetime import datetime

from .models import Claim, Source, TreeNode, View
from .store import AskChemStore
from .llm import get_client, MODELS
from .display import smart_title

CLASSIFICATION_PROMPT = """You are classifying a chemistry knowledge claim into a hierarchical index with 5 views.

The claim:
{claim_json}

For each of the 5 views below, provide the hierarchical path where this claim belongs.
Each path should be 2-5 segments deep, using lowercase_with_underscores for node names.
Use established chemistry terminology.

Views:
1. by_reaction_type — Classify by the type of chemical transformation (e.g., coupling > cross_coupling > suzuki)
2. by_substance_class — Classify by the molecules/materials involved (e.g., organic > aromatics > aryl_halides)
3. by_application — Classify by practical application (e.g., pharmaceutical > drug_synthesis > c_n_bond_forming)
4. by_technique — Classify by experimental/computational method (e.g., spectroscopy > raman > operando_raman)
5. by_mechanism — Classify by underlying mechanism/phenomenon (e.g., catalytic_cycles > oxidative_addition_reductive_elimination)

If a view is not applicable to this claim (e.g., a computational result has no reaction type), use ["not_applicable"].

Return a JSON object:
{{
  "by_reaction_type": ["segment1", "segment2", ...],
  "by_substance_class": ["segment1", "segment2", ...],
  "by_application": ["segment1", "segment2", ...],
  "by_technique": ["segment1", "segment2", ...],
  "by_mechanism": ["segment1", "segment2", ...]
}}"""


def classify_claim(claim: Claim, max_retries: int = 3) -> dict[str, list[str]]:
    """Classify a claim into all 5 views using the configured LLM."""
    claim_summary = {
        "claim_type": claim.claim_type,
        "reaction_type": claim.reaction_type,
        "subject": claim.subject,
        "property_name": claim.property_name,
        "technique_name": claim.technique_name,
        "process_described": claim.process_described,
        "verbatim_quote": (claim.verbatim_quote or "")[:200],
        "reactants": (claim.reactants or [])[:3],
        "products": (claim.products or [])[:3],
        "conditions": claim.conditions,
    }
    claim_summary = {k: v for k, v in claim_summary.items() if v}

    prompt = CLASSIFICATION_PROMPT.format(claim_json=json.dumps(claim_summary, indent=2))

    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODELS["fast"],
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {}
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise
    return {}


def classify_claims_batch(claims: list[Claim], batch_size: int = 10) -> list[dict]:
    """Classify multiple claims, with rate limiting."""
    results = []
    for i, claim in enumerate(claims):
        try:
            paths = classify_claim(claim)
            results.append({"claim_id": claim.claim_id, "paths": paths})
        except Exception as e:
            print(f"  Error classifying claim {claim.claim_id}: {e}", flush=True)
            results.append({"claim_id": claim.claim_id, "paths": {}, "error": str(e)})

        if (i + 1) % batch_size == 0:
            print(f"  Classified {i+1}/{len(claims)} claims", flush=True)
            time.sleep(1)

    return results


def build_index(store: AskChemStore, claims: list[Claim], sources: list[Source] = None):
    """
    Build the full AskChem index from a list of claims.

    1. Add all sources to the store
    2. Add all claims to the store
    3. Classify each claim into the 5 views
    4. Build the hierarchy nodes
    """
    print(f"Building index with {len(claims)} claims", flush=True)

    # Add sources
    if sources:
        print(f"Adding {len(sources)} sources...", flush=True)
        for source in sources:
            store.add_source(source)

    # Add claims
    print(f"Adding {len(claims)} claims...", flush=True)
    for claim in claims:
        store.add_claim(claim)

    # Classify claims into views
    print(f"Classifying claims into 5 views...", flush=True)
    classifications = classify_claims_batch(claims)

    # Build hierarchy
    print(f"Building hierarchy...", flush=True)
    node_cache = {}  # (view_id, tuple(path)) -> TreeNode

    for classification in classifications:
        claim_id = classification["claim_id"]
        paths = classification.get("paths", {})

        for view_id, path in paths.items():
            if not path or path == ["not_applicable"]:
                continue

            # Ensure all intermediate nodes exist
            for depth in range(len(path)):
                partial_path = path[:depth + 1]
                cache_key = (view_id, tuple(partial_path))

                if cache_key not in node_cache:
                    node_id = f"{view_id}_{'_'.join(partial_path)}"
                    node = TreeNode(
                        node_id=node_id,
                        name=smart_title(partial_path[-1]),
                        path=partial_path,
                        view=view_id,
                        level=depth + 1,
                    )
                    store.add_node(view_id, partial_path, node)
                    node_cache[cache_key] = node

            # Assign claim to the leaf node
            store.assign_claim_to_node(view_id, path, claim_id)

    # Update view metadata
    for view_id in ["by_reaction_type", "by_substance_class", "by_application", "by_technique", "by_mechanism"]:
        view = store.get_view(view_id)
        if view:
            view_nodes = [k for k in node_cache if k[0] == view_id]
            view.node_count = len(view_nodes)
            view.updated_at = datetime.now().isoformat()

    # Count total nodes
    total_nodes = len(node_cache)
    print(f"Index built: {len(claims)} claims, {total_nodes} nodes across 5 views", flush=True)

    return {
        "claims_indexed": len(claims),
        "nodes_created": total_nodes,
        "classifications": len(classifications),
    }


def load_extracted_claims(experiments_dir: Path) -> tuple[list[Claim], list[Source]]:
    """Load claims from extraction experiment results and convert to Claim objects."""
    claims = []
    sources_seen = {}

    # Load v2 extractions
    v2_path = experiments_dir / "003_extraction_v2" / "results" / "all_extractions_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2_data = json.load(f)

        for paper in v2_data:
            if "error" in paper:
                continue

            s1 = paper.get("stage1", {}).get("result", {})
            paper_meta = s1.get("paper_metadata", {})
            paper_title = paper_meta.get("title", paper.get("paper_name", "unknown"))
            doi = paper_meta.get("doi", "")

            # Create source
            if doi and doi not in sources_seen:
                source = Source(
                    doi=doi,
                    title=paper_title,
                    authors=paper_meta.get("authors", []),
                    year=paper_meta.get("year", 0),
                    venue=paper_meta.get("journal", ""),
                )
                sources_seen[doi] = source

            # Create claims
            for raw_claim in paper.get("stage2", {}).get("result", {}).get("claims", []):
                claim_type = raw_claim.get("claim_type", "unknown")
                content_hash = str(hash(json.dumps(raw_claim, sort_keys=True)))[:12]
                claim_id = Claim.generate_id(doi or paper_title, claim_type, content_hash)

                claim = Claim(
                    claim_id=claim_id,
                    claim_type=claim_type,
                    source_doi=doi,
                    source_paper_title=paper_title,
                    confidence=raw_claim.get("confidence", "medium"),
                    location_in_paper=raw_claim.get("location_in_paper", ""),
                    verbatim_quote=raw_claim.get("verbatim_quote", ""),
                    extraction_model="gpt-5.4",
                    extraction_version="v2",
                    extracted_at=datetime.now().isoformat(),
                    reaction_type=raw_claim.get("reaction_type", ""),
                    reactants=raw_claim.get("reactants", []),
                    products=raw_claim.get("products", []),
                    conditions=raw_claim.get("conditions", {}),
                    outcomes=raw_claim.get("outcomes", {}),
                    is_key_result=raw_claim.get("is_key_result", False),
                    parent_reaction_id=raw_claim.get("parent_reaction_id"),
                    subject=raw_claim.get("subject", ""),
                    subject_smiles=raw_claim.get("subject_smiles", ""),
                    property_name=raw_claim.get("property_name", ""),
                    property_category=raw_claim.get("property_category", ""),
                    value=raw_claim.get("value", ""),
                    unit=raw_claim.get("unit", ""),
                    measurement_method=raw_claim.get("measurement_method", ""),
                    is_computed=raw_claim.get("is_computed", False),
                    process_described=raw_claim.get("process_described", ""),
                    steps=raw_claim.get("steps", []),
                    key_intermediates=raw_claim.get("key_intermediates", []),
                    evidence=raw_claim.get("evidence", []),
                    technique_name=raw_claim.get("technique_name", ""),
                    what_it_achieves=raw_claim.get("what_it_achieves", ""),
                    compared_items=raw_claim.get("compared_items", []),
                    metric=raw_claim.get("metric", ""),
                    comparison_result=raw_claim.get("comparison_result", raw_claim.get("result", "")),
                )
                claims.append(claim)

    return claims, list(sources_seen.values())
