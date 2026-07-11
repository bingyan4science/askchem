"""
AskChem Bulk Processing Pipeline — OpenAI Batch API.

Processes 100K+ papers in two phases using the OpenAI Batch API (50% cost savings):
  Phase 1: Extract claims from abstracts
  Phase 2: Classify claims into the 5-view hierarchy

Each phase:
  1. Builds JSONL request files (max 50K requests per file)
  2. Uploads and submits as OpenAI batch jobs
  3. Polls until completion
  4. Downloads results and writes to the AskChem index

All intermediate state is saved to disk so the pipeline can be resumed at any point.

Usage:
    python src/process_corpus.py extract              # Phase 1: submit extraction batches
    python src/process_corpus.py extract --poll        # Poll extraction batch status
    python src/process_corpus.py extract --collect     # Download & save extraction results
    python src/process_corpus.py classify              # Phase 2: submit classification batches
    python src/process_corpus.py classify --poll       # Poll classification batch status
    python src/process_corpus.py classify --collect    # Download & save classification results
    python src/process_corpus.py index                 # Write results to AskChem store
    python src/process_corpus.py status                # Show pipeline progress
    python src/process_corpus.py extract --max-papers 20  # Test with small subset
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from askchem.models import Claim, Source, TreeNode
from askchem.store import AskChemStore
from askchem.llm import MODELS
from askchem.display import smart_title

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
DATA_DIR = Path(__file__).parent.parent / "data"
PIPELINE_DIR = INDEX_DIR / "_pipeline"
MERGE_MAP_FILE = INDEX_DIR / "_merge_map.json"
CACHE_FILE = INDEX_DIR / "_classification_cache.json"

MODEL = MODELS["fast"]
EXTRACT_BATCH_LIMIT = 50_000  # Extraction requests are small (~2KB each)
CLASSIFY_BATCH_LIMIT = 14_000  # Classification requests are ~14KB each; OpenAI limit is 200MB per file

# ── Prompts ──────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a chemistry expert. Extract structured knowledge claims from this paper's abstract and metadata.

Paper metadata:
Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Abstract: {abstract}

Extract ALL factual claims. Return a JSON object with:
{{
  "claims": [
    {{
      "claim_id": sequential number,
      "claim_type": "reaction|property|method|mechanism|comparison|computational_result",
      "confidence": "high|medium|low",

      // For reactions:
      "reaction_type": "e.g., Suzuki coupling",
      "reactants": [{{"name": "...", "smiles": "or null", "role": "substrate|reagent|catalyst"}}],
      "products": [{{"name": "...", "smiles": "or null"}}],
      "conditions": {{"catalyst": "...", "solvent": "...", "temperature": "...", "other": "..."}},
      "outcomes": {{"yield_percent": null, "selectivity": "...", "other": "..."}},

      // For properties:
      "subject": "molecule/material",
      "property_name": "e.g., BET surface area",
      "value": "numeric or descriptive",
      "unit": "if applicable",
      "measurement_method": "technique",

      // For methods:
      "technique_name": "name",
      "what_it_achieves": "description",

      // For mechanisms:
      "process_described": "what process",
      "steps": ["step1", "step2"],

      // For all:
      "verbatim_quote": "exact text from abstract"
    }}
  ]
}}

Extract 3-10 claims from the abstract. Focus on the main findings."""


def _build_classification_prompt_template():
    """Build the classification prompt using canonical L1 categories."""
    from askchem.taxonomy import CANONICAL_L1
    categories_text = ""
    for view_id, cats in CANONICAL_L1.items():
        categories_text += f"\n  {view_id}: {', '.join(cats)}"

    return """You are classifying a chemistry knowledge claim into a hierarchical index with 5 views.

The claim:
{claim_json}

For each view, provide a hierarchical path of 2-4 segments where segment 1 (L1) MUST be
one of the canonical categories listed. Use lowercase_with_underscores.
If the claim does not fit a view, use ["not_applicable"].

Canonical L1 categories:""" + categories_text + """

Return a JSON object:
{{
  "by_reaction_type": ["canonical_l1", "l2", ...],
  "by_substance_class": ["canonical_l1", "l2", ...],
  "by_application": ["canonical_l1", "l2", ...],
  "by_technique": ["canonical_l1", "l2", ...],
  "by_mechanism": ["canonical_l1", "l2", ...]
}}"""


# ── Paper Loading ────────────────────────────────────────────────────────────

def load_processable_papers(max_papers=None):
    """Load papers that have both abstract and DOI."""
    papers_file = DATA_DIR / "metadata" / "all_papers.json"
    print(f"Loading papers from {papers_file}...", flush=True)
    with open(papers_file) as f:
        all_papers = json.load(f)

    processable = [
        p for p in all_papers
        if p.get("abstract") and (p.get("externalIds") or {}).get("DOI")
    ]
    processable.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)

    if max_papers:
        processable = processable[:max_papers]

    print(f"  {len(all_papers):,} total -> {len(processable):,} processable",
          flush=True)
    return processable


def load_existing_dois():
    """Get DOIs already in the index."""
    dois = set()
    sources_dir = INDEX_DIR / "sources"
    if not sources_dir.exists():
        return dois
    files = list(sources_dir.glob("*.json"))
    print(f"  Loading {len(files):,} existing sources...", flush=True)
    for i, f in enumerate(files):
        with open(f) as fh:
            s = json.load(fh)
        doi = s.get("doi", "")
        if doi:
            dois.add(doi.lower())
        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,}/{len(files):,} loaded", flush=True)
    print(f"  {len(dois):,} existing DOIs loaded", flush=True)
    return dois


def _doi(paper):
    return (paper.get("externalIds") or {}).get("DOI", "")


# ── Extraction Phase ─────────────────────────────────────────────────────────

def load_extracted_dois():
    """Get DOIs that already have extraction results."""
    results_dir = PIPELINE_DIR / "extraction_results"
    dois = set()
    if results_dir.exists():
        for f in results_dir.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        doi = r.get("custom_id", "").replace("extract::", "")
                        if doi:
                            dois.add(doi)
                    except json.JSONDecodeError:
                        pass
    return dois


def build_extraction_requests(papers):
    """Build JSONL request files for extraction batches."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    requests_dir = PIPELINE_DIR / "extraction_requests"
    requests_dir.mkdir(exist_ok=True)

    already_extracted = load_extracted_dois()
    existing_dois = load_existing_dois()
    skip = already_extracted | existing_dois

    seen_dois = set()
    todo = []
    for p in papers:
        doi = _doi(p).lower()
        if doi and doi not in skip and doi not in seen_dois:
            seen_dois.add(doi)
            todo.append(p)
    print(f"  {len(papers):,} processable, {len(skip):,} already done, "
          f"{len(todo):,} to extract", flush=True)

    if not todo:
        print("  Nothing to extract.", flush=True)
        return []

    file_paths = []
    for chunk_idx in range(0, len(todo), EXTRACT_BATCH_LIMIT):
        chunk = todo[chunk_idx : chunk_idx + EXTRACT_BATCH_LIMIT]
        fname = requests_dir / f"extract_{chunk_idx:06d}.jsonl"

        with open(fname, "w") as f:
            for paper in chunk:
                doi = _doi(paper)
                title = paper.get("title", "")
                authors = [a.get("name", "") for a in (paper.get("authors") or [])[:5]]
                year = paper.get("year", "")
                venue = paper.get("venue", "")
                abstract = paper.get("abstract", "")

                prompt = EXTRACTION_PROMPT.format(
                    title=title,
                    authors=", ".join(authors),
                    year=year,
                    venue=venue,
                    abstract=abstract,
                )

                request = {
                    "custom_id": f"extract::{doi}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": 4096,
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(request) + "\n")

        file_paths.append(fname)
        print(f"  Written {fname.name}: {len(chunk):,} requests", flush=True)

    return file_paths


def submit_batches(request_files, phase):
    """Upload JSONL files and submit as OpenAI batch jobs."""
    client = OpenAI()
    batch_ids = []
    tracker_file = PIPELINE_DIR / f"{phase}_batches.json"

    existing = {}
    if tracker_file.exists():
        with open(tracker_file) as f:
            existing = json.load(f)

    for fpath in request_files:
        fname = fpath.name
        if fname in existing:
            print(f"  {fname}: already submitted as {existing[fname]['batch_id']}",
                  flush=True)
            batch_ids.append(existing[fname]["batch_id"])
            continue

        print(f"  Uploading {fname}...", flush=True)
        with open(fpath, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")

        print(f"  Submitting batch for {fname} (file_id={file_obj.id})...",
              flush=True)
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"phase": phase, "source_file": fname},
        )

        existing[fname] = {
            "batch_id": batch.id,
            "file_id": file_obj.id,
            "status": batch.status,
            "submitted_at": datetime.now().isoformat(),
            "request_count": sum(1 for _ in open(fpath)),
        }
        batch_ids.append(batch.id)
        print(f"  -> Batch {batch.id} ({batch.status})", flush=True)

        time.sleep(1)

    with open(tracker_file, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n  {len(batch_ids)} batch(es) submitted for {phase}", flush=True)
    return batch_ids


def poll_batches(phase):
    """Check status of all batches for a phase."""
    tracker_file = PIPELINE_DIR / f"{phase}_batches.json"
    if not tracker_file.exists():
        print(f"  No batches found for {phase}.", flush=True)
        return

    with open(tracker_file) as f:
        tracker = json.load(f)

    client = OpenAI()
    all_done = True

    for fname, info in tracker.items():
        batch_id = info["batch_id"]
        batch = client.batches.retrieve(batch_id)
        info["status"] = batch.status
        info["output_file_id"] = getattr(batch, "output_file_id", None)
        info["error_file_id"] = getattr(batch, "error_file_id", None)

        counts = batch.request_counts
        completed = counts.completed if counts else "?"
        failed = counts.failed if counts else "?"
        total = counts.total if counts else "?"

        print(f"  {fname}: {batch.status} "
              f"({completed}/{total} done, {failed} failed)", flush=True)

        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_done = False

    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)

    if all_done:
        print(f"\n  All {phase} batches finished!", flush=True)
    else:
        print(f"\n  Some batches still running. Poll again later.", flush=True)

    return all_done


def collect_results(phase):
    """Download results from completed batches and save locally."""
    tracker_file = PIPELINE_DIR / f"{phase}_batches.json"
    if not tracker_file.exists():
        print(f"  No batches found for {phase}.", flush=True)
        return

    with open(tracker_file) as f:
        tracker = json.load(f)

    client = OpenAI()
    results_dir = PIPELINE_DIR / f"{phase}_results"
    results_dir.mkdir(exist_ok=True)

    total_success = 0
    total_failed = 0

    for fname, info in tracker.items():
        batch_id = info["batch_id"]
        batch = client.batches.retrieve(batch_id)
        info["status"] = batch.status

        out_file = results_dir / f"{fname}"
        if out_file.exists():
            n = sum(1 for _ in open(out_file))
            print(f"  {fname}: already collected ({n} results)", flush=True)
            total_success += n
            continue

        if batch.status not in ("completed", "failed", "expired"):
            print(f"  {fname}: status={batch.status}, skipping", flush=True)
            continue

        if batch.output_file_id:
            print(f"  Downloading results for {fname}...", flush=True)
            content = client.files.content(batch.output_file_id)
            with open(out_file, "wb") as f:
                f.write(content.read())
            n = sum(1 for _ in open(out_file))
            total_success += n
            print(f"    -> {n} successful results", flush=True)

        if batch.error_file_id:
            err_file = results_dir / f"errors_{fname}"
            content = client.files.content(batch.error_file_id)
            with open(err_file, "wb") as f:
                f.write(content.read())
            n_err = sum(1 for _ in open(err_file))
            total_failed += n_err
            print(f"    -> {n_err} errors (saved to {err_file.name})", flush=True)

    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"\n  Total: {total_success:,} successes, {total_failed:,} failures",
          flush=True)


# ── Classification Phase ─────────────────────────────────────────────────────

def load_extractions():
    """Load all extraction results from downloaded batch outputs."""
    results_dir = PIPELINE_DIR / "extraction_results"
    extractions = {}  # doi -> [raw_claims]

    if not results_dir.exists():
        return extractions

    for f in sorted(results_dir.glob("*.jsonl")):
        if f.name.startswith("errors_"):
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    doi = custom_id.replace("extract::", "")
                    response = result.get("response", {})
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content:
                            claims = json.loads(content).get("claims", [])
                            extractions[doi] = claims
                except (json.JSONDecodeError, KeyError):
                    pass

    return extractions


def load_classified_claim_ids():
    """Get claim_ids that already have classification results."""
    results_dir = PIPELINE_DIR / "classification_results"
    ids = set()
    if results_dir.exists():
        for f in results_dir.glob("*.jsonl"):
            if f.name.startswith("errors_"):
                continue
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        cid = r.get("custom_id", "").replace("classify::", "")
                        if cid:
                            ids.add(cid)
                    except json.JSONDecodeError:
                        pass
    return ids


def build_claim_objects(papers, extractions):
    """Convert raw extractions into Claim objects for classification."""
    def _s(val, default=""):
        return val if val is not None else default

    claims = []
    paper_by_doi = {_doi(p).lower(): p for p in papers if _doi(p)}

    for doi, raw_claims in extractions.items():
        paper = paper_by_doi.get(doi.lower(), {})
        title = paper.get("title", "")
        for raw in raw_claims:
            claim_type = _s(raw.get("claim_type"), "unknown")
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)
            claims.append((claim_id, doi, title, raw))

    return claims


def build_classification_requests(papers, extractions):
    """Build JSONL request files for classification batches."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    requests_dir = PIPELINE_DIR / "classification_requests"
    requests_dir.mkdir(exist_ok=True)

    prompt_template = _build_classification_prompt_template()
    already_classified = load_classified_claim_ids()
    claim_tuples = build_claim_objects(papers, extractions)

    seen_cids = set()
    todo = []
    for cid, doi, title, raw in claim_tuples:
        if cid not in already_classified and cid not in seen_cids:
            seen_cids.add(cid)
            todo.append((cid, doi, title, raw))
    print(f"  {len(claim_tuples):,} total claims, {len(already_classified):,} done, "
          f"{len(todo):,} to classify", flush=True)

    if not todo:
        print("  Nothing to classify.", flush=True)
        return []

    def _s(val, default=""):
        return val if val is not None else default

    file_paths = []
    for chunk_idx in range(0, len(todo), CLASSIFY_BATCH_LIMIT):
        chunk = todo[chunk_idx : chunk_idx + CLASSIFY_BATCH_LIMIT]
        fname = requests_dir / f"classify_{chunk_idx:06d}.jsonl"

        with open(fname, "w") as f:
            for claim_id, doi, title, raw in chunk:
                claim_summary = {
                    "claim_type": _s(raw.get("claim_type")),
                    "reaction_type": _s(raw.get("reaction_type")),
                    "subject": _s(raw.get("subject")),
                    "property_name": _s(raw.get("property_name")),
                    "technique_name": _s(raw.get("technique_name")),
                    "process_described": _s(raw.get("process_described")),
                    "verbatim_quote": _s(raw.get("verbatim_quote"))[:200],
                    "reactants": (raw.get("reactants") or [])[:3],
                    "products": (raw.get("products") or [])[:3],
                    "conditions": raw.get("conditions") or {},
                }
                claim_summary = {k: v for k, v in claim_summary.items() if v}
                prompt = prompt_template.format(
                    claim_json=json.dumps(claim_summary, indent=2)
                )

                request = {
                    "custom_id": f"classify::{claim_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": 2048,
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(request) + "\n")

        file_paths.append(fname)
        print(f"  Written {fname.name}: {len(chunk):,} requests", flush=True)

    return file_paths


# ── Index Writing ────────────────────────────────────────────────────────────

def load_classifications():
    """Load all classification results from downloaded batch outputs."""
    results_dir = PIPELINE_DIR / "classification_results"
    classifications = {}  # claim_id -> {paths}

    if not results_dir.exists():
        return classifications

    files = sorted(f for f in results_dir.glob("*.jsonl") if not f.name.startswith("errors_"))
    print(f"  Loading classifications from {len(files)} batch files...", flush=True)
    for fi, f in enumerate(files):
        with open(f) as fh:
            for line in fh:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    claim_id = custom_id.replace("classify::", "")
                    response = result.get("response", {})
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content:
                            paths = json.loads(content)
                            classifications[claim_id] = paths
                except (json.JSONDecodeError, KeyError):
                    pass
        if (fi + 1) % 10 == 0:
            print(f"    {fi+1}/{len(files)} files, {len(classifications):,} claims", flush=True)

    print(f"  {len(classifications):,} classifications loaded", flush=True)
    return classifications


def write_to_index(paper_by_doi, extractions, classifications):
    """Write extraction and classification results to the AskChem store."""
    store = AskChemStore(INDEX_DIR)
    if not (INDEX_DIR / "metadata.json").exists():
        store.initialize()

    def _s(v, d=""):
        return v if v is not None else d

    def _write_file(path, data):
        """Write JSON without metadata update overhead."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    existing_dois = load_existing_dois()

    sources_added = 0
    claims_added = 0
    nodes_created = 0
    assignments = 0
    skipped_existing = 0

    total_dois = len(extractions)
    for i, (doi, raw_claims) in enumerate(extractions.items()):
        if doi.lower() in existing_dois:
            skipped_existing += 1
            continue

        paper = paper_by_doi.get(doi.lower(), {})
        title = paper.get("title", "")

        source = Source(
            doi=doi,
            title=title,
            authors=[a.get("name", "") for a in (paper.get("authors") or [])[:10]],
            year=paper.get("year") or 0,
            venue=paper.get("venue", ""),
            citation_count=paper.get("citationCount", 0),
        )
        _write_file(store.sources_dir / f"{source.source_id}.json", source.to_dict())
        sources_added += 1

        for raw in raw_claims:
            claim_type = _s(raw.get("claim_type"), "unknown")
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            claim = Claim(
                claim_id=claim_id,
                claim_type=claim_type,
                source_doi=doi,
                source_paper_title=title,
                confidence=_s(raw.get("confidence"), "medium"),
                location_in_paper="abstract",
                verbatim_quote=_s(raw.get("verbatim_quote")),
                extraction_model=MODEL,
                extraction_version="v3-abstract-batch",
                extracted_at=datetime.now().isoformat(),
                reaction_type=_s(raw.get("reaction_type")),
                reactants=raw.get("reactants") or [],
                products=raw.get("products") or [],
                conditions=raw.get("conditions") or {},
                outcomes=raw.get("outcomes") or {},
                subject=_s(raw.get("subject")),
                subject_smiles=_s(raw.get("subject_smiles")),
                property_name=_s(raw.get("property_name")),
                value=str(raw.get("value") or ""),
                unit=_s(raw.get("unit")),
                measurement_method=_s(raw.get("measurement_method")),
                technique_name=_s(raw.get("technique_name")),
                what_it_achieves=_s(raw.get("what_it_achieves")),
                process_described=_s(raw.get("process_described")),
                steps=raw.get("steps") or [],
                compared_items=raw.get("compared_items") or [],
                metric=_s(raw.get("metric")),
                comparison_result=_s(raw.get("comparison_result")),
            )

            from askchem.taxonomy import normalize_path, build_claim_type_path, ALL_CONTENT_VIEWS
            raw_paths = classifications.get(claim_id, {})
            view_paths = {}
            for vid in ALL_CONTENT_VIEWS:
                normed = normalize_path(vid, raw_paths.get(vid, []))
                if normed:
                    view_paths[vid] = normed
            view_paths['by_claim_type'] = build_claim_type_path(claim_type)
            claim.view_paths = view_paths
            _write_file(store.claims_dir / f"{claim.claim_id}.json", claim.to_dict())
            claims_added += 1

            for view_id, path in view_paths.items():
                if not path:
                    continue
                for depth in range(len(path)):
                    partial = path[: depth + 1]
                    node_dir = store.views_dir / view_id
                    for seg in partial:
                        node_dir = node_dir / seg
                    node_file = node_dir / "_node.json"
                    if not node_file.exists():
                        node_id = f"{view_id}_{'_'.join(partial)}"
                        node_data = TreeNode(
                            node_id=node_id,
                            name=smart_title(partial[-1]),
                            path=list(partial),
                            view=view_id,
                            level=depth + 1,
                        ).to_dict()
                        _write_file(node_file, node_data)
                        nodes_created += 1
                        parent_path = partial[:-1]
                        parent_dir = store.views_dir / view_id
                        for seg in parent_path:
                            parent_dir = parent_dir / seg
                        parent_file = parent_dir / ("_node.json" if parent_path else "_root.json")
                        if parent_file.exists():
                            parent_data = store._read_json(parent_file)
                            if parent_data:
                                children = parent_data.get("children", [])
                                if node_id not in children:
                                    children.append(node_id)
                                    parent_data["children"] = children
                                    _write_file(parent_file, parent_data)

                leaf_dir = store.views_dir / view_id
                for seg in path:
                    leaf_dir = leaf_dir / seg
                leaf_file = leaf_dir / "_node.json"
                if leaf_file.exists():
                    data = store._read_json(leaf_file)
                    if data:
                        claim_ids = data.get("claim_ids", [])
                        if claim_id not in claim_ids:
                            claim_ids.append(claim_id)
                            data["claim_ids"] = claim_ids
                            data["claim_count"] = len(claim_ids)
                            _write_file(leaf_file, data)
                assignments += 1

        if (i + 1) % 5000 == 0:
            print(f"  Indexed {i+1:,}/{total_dois:,} papers "
                  f"({sources_added:,} sources, {claims_added:,} claims, "
                  f"{nodes_created:,} nodes, {skipped_existing:,} skipped)",
                  flush=True)

    # Update metadata once at the end with actual file counts
    print("  Counting final index files...", flush=True)
    actual_claims = len(list(store.claims_dir.glob("*.json")))
    actual_sources = len(list(store.sources_dir.glob("*.json")))
    meta = store.get_metadata()
    meta["claim_count"] = actual_claims
    meta["source_count"] = actual_sources
    meta["updated_at"] = datetime.now().isoformat()
    store._write_json(store._metadata_path, meta)
    print(f"  Metadata updated: {actual_sources:,} sources, {actual_claims:,} claims",
          flush=True)

    return {
        "sources_added": sources_added,
        "claims_added": claims_added,
        "nodes_created": nodes_created,
        "assignments": assignments,
        "skipped_existing": skipped_existing,
    }


# ── Status ───────────────────────────────────────────────────────────────────

def show_status():
    """Show pipeline progress overview."""
    print(f"{'='*60}", flush=True)
    print("AskChem Pipeline Status", flush=True)
    print(f"{'='*60}\n", flush=True)

    papers_file = DATA_DIR / "metadata" / "all_papers.json"
    if papers_file.exists():
        with open(papers_file) as f:
            all_p = json.load(f)
        processable = sum(
            1 for p in all_p
            if p.get("abstract") and (p.get("externalIds") or {}).get("DOI")
        )
        print(f"Corpus: {len(all_p):,} total, {processable:,} processable", flush=True)

    existing = load_existing_dois()
    print(f"Index: {len(existing):,} sources already indexed", flush=True)

    extracted = load_extracted_dois()
    print(f"Extraction results: {len(extracted):,} papers", flush=True)

    extractions = load_extractions()
    total_claims = sum(len(v) for v in extractions.values())
    print(f"  -> {total_claims:,} claims extracted", flush=True)

    classified = load_classified_claim_ids()
    print(f"Classification results: {len(classified):,} claims", flush=True)

    for phase in ("extraction", "classification"):
        tracker_file = PIPELINE_DIR / f"{phase}_batches.json"
        if tracker_file.exists():
            with open(tracker_file) as f:
                tracker = json.load(f)
            print(f"\n{phase.title()} batches:", flush=True)
            for fname, info in tracker.items():
                print(f"  {fname}: {info['status']} "
                      f"(batch_id={info['batch_id'][:20]}...)", flush=True)

    claims_dir = INDEX_DIR / "claims"
    if claims_dir.exists():
        n_claims = len(list(claims_dir.glob("*.json")))
        print(f"\nIndex claims on disk: {n_claims:,}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AskChem bulk processing via OpenAI Batch API"
    )
    parser.add_argument(
        "command",
        choices=["extract", "classify", "index", "status"],
        help="Pipeline phase to run",
    )
    parser.add_argument("--poll", action="store_true",
                        help="Poll batch job status")
    parser.add_argument("--collect", action="store_true",
                        help="Download completed batch results")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Limit number of papers (for testing)")
    args = parser.parse_args()

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "status":
        show_status()
        return

    if args.command == "extract":
        print(f"\n{'='*60}", flush=True)
        print("Phase 1: EXTRACTION", flush=True)
        print(f"{'='*60}\n", flush=True)

        if args.poll:
            poll_batches("extraction")
            return
        if args.collect:
            collect_results("extraction")
            return

        papers = load_processable_papers(max_papers=args.max_papers)
        request_files = build_extraction_requests(papers)
        if request_files:
            submit_batches(request_files, "extraction")
            print("\n  Batches submitted! Use --poll to check status, "
                  "--collect to download results.", flush=True)

    elif args.command == "classify":
        print(f"\n{'='*60}", flush=True)
        print("Phase 2: CLASSIFICATION", flush=True)
        print(f"{'='*60}\n", flush=True)

        if args.poll:
            poll_batches("classification")
            return
        if args.collect:
            collect_results("classification")
            return

        papers = load_processable_papers(max_papers=args.max_papers)
        extractions = load_extractions()
        if not extractions:
            print("  ERROR: No extraction results found. "
                  "Run 'extract --collect' first.", flush=True)
            return

        print(f"  Loaded {len(extractions):,} paper extractions "
              f"({sum(len(v) for v in extractions.values()):,} claims)",
              flush=True)

        request_files = build_classification_requests(papers, extractions)
        if request_files:
            submit_batches(request_files, "classification")
            print("\n  Batches submitted! Use --poll to check status, "
                  "--collect to download results.", flush=True)

    elif args.command == "index":
        print(f"\n{'='*60}", flush=True)
        print("Phase 3: INDEX WRITING", flush=True)
        print(f"{'='*60}\n", flush=True)

        papers = load_processable_papers(max_papers=args.max_papers)
        paper_by_doi = {_doi(p).lower(): p for p in papers if _doi(p)}
        del papers

        extractions = load_extractions()
        classifications = load_classifications()

        if not extractions:
            print("  ERROR: No extractions. Run extract phase first.", flush=True)
            return
        if not classifications:
            print("  WARNING: No classifications. Writing claims without "
                  "hierarchy assignments.", flush=True)

        print(f"  Extractions: {len(extractions):,} papers, "
              f"{sum(len(v) for v in extractions.values()):,} claims", flush=True)
        print(f"  Classifications: {len(classifications):,} claims", flush=True)

        result = write_to_index(paper_by_doi, extractions, classifications)
        del paper_by_doi, extractions, classifications

        print(f"\n{'='*60}", flush=True)
        print("INDEX WRITE COMPLETE", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Sources added:  {result['sources_added']:,}", flush=True)
        print(f"  Claims added:   {result['claims_added']:,}", flush=True)
        print(f"  Nodes created:  {result['nodes_created']:,}", flush=True)
        print(f"  Assignments:    {result['assignments']:,}", flush=True)
        print(f"  Skipped (existing): {result['skipped_existing']:,}", flush=True)


if __name__ == "__main__":
    main()
