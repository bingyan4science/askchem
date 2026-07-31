"""
Integrate deep extraction results into the AskChem index.

Two-track classification:
  Track A (content claims): reaction, property, method, mechanism, comparison,
           scope_entry, computational_result → full 5-view classification
  Track B (epistemic claims): hypothesis, conclusion, limitation, future_direction,
           surprising_finding, experimental_design, structure → topic-only classification
  All claims also get a by_claim_type path derived deterministically from claim_type.

Pipeline:
    python src/integrate_deep.py prepare      # Build classification batch files
    python src/integrate_deep.py submit        # Submit to Batch API
    python src/integrate_deep.py poll          # Check status
    python src/integrate_deep.py collect       # Download classification results
    python src/integrate_deep.py build         # Build SQLite index from everything
    python src/integrate_deep.py status        # Show progress
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from askchem.models import Claim, Source
from askchem.display import smart_title

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "deep_results"
CORPUS_DIR = DATA_DIR / "corpus_checkpoints"
PIPELINE_DIR = DATA_DIR / "classify_pipeline"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"


def _normalize_extraction_model(raw) -> str:
    """Normalise raw extraction_model strings to a clean public label.

    Inputs we have seen:
      - '@vertexai-gemini-kc119-2/gemini-3.1-pro-preview' (Portkey route)
      - 'gemini-3.1-pro-preview'
      - None (legacy)
    """
    if not raw:
        return "gemini-3.1-pro"
    s = str(raw).lower()
    if "gemini" in s:
        return "gemini-3.1-pro"
    if "gpt-5.4" in s:
        return "gpt-5.4"
    if "gpt-5-mini" in s or "gpt5mini" in s:
        return "gpt-5-mini"
    return raw

CONTENT_TYPES = {
    'reaction', 'property', 'method', 'mechanism', 'comparison',
    'scope_entry', 'computational_result',
}

EPISTEMIC_TYPES = {
    'hypothesis', 'conclusion', 'conclusions', 'limitation',
    'future_direction', 'surprising_finding', 'experimental_design',
    'structure', 'background', 'historical', 'definition', 'observation',
}

# Readable labels for the by_claim_type view L1 nodes
CLAIM_TYPE_LABELS = {
    'reaction': 'reactions',
    'property': 'properties',
    'method': 'methods',
    'mechanism': 'mechanisms',
    'comparison': 'comparisons',
    'scope_entry': 'scope_entries',
    'computational_result': 'computational_results',
    'hypothesis': 'hypotheses',
    'conclusion': 'conclusions',
    'conclusions': 'conclusions',
    'limitation': 'limitations',
    'future_direction': 'future_directions',
    'surprising_finding': 'surprising_findings',
    'experimental_design': 'experimental_designs',
    'structure': 'structures',
    'background': 'background',
    'historical': 'historical',
    'definition': 'definitions',
    'observation': 'observations',
}

# ── Prompts ──────────────────────────────────────────────────────────────────

CONTENT_PROMPT = """Classify this chemistry claim into 5 hierarchical views.
Each path: 2-4 segments, lowercase_with_underscores. Use ["not_applicable"] if a view doesn't apply.

Claim:
{claim_json}

Views:
1. by_reaction_type — chemical transformation type (e.g. ["coupling","cross_coupling","suzuki"])
2. by_substance_class — molecules/materials (e.g. ["organic","aromatics","polycyclic_aromatic_hydrocarbons"])
3. by_application — practical use (e.g. ["energy","batteries","lithium_ion"])
4. by_technique — method used (e.g. ["spectroscopy","nmr","solid_state_nmr"])
5. by_mechanism — underlying phenomenon (e.g. ["catalytic_cycles","oxidative_addition"])

Return JSON:
{{
  "by_reaction_type": [...],
  "by_substance_class": [...],
  "by_application": [...],
  "by_technique": [...],
  "by_mechanism": [...]
}}"""

EPISTEMIC_PROMPT = """Classify this scientific claim into the chemistry topic areas where it is relevant.
This is a {claim_type_label} — focus on the scientific subject matter, not the epistemic role.

Claim:
{claim_json}

For each view, provide a 2-4 segment path (lowercase_with_underscores) describing the chemistry topic.
Use ["not_applicable"] ONLY if the claim has no meaningful connection to that dimension.

Views:
1. by_substance_class — what molecules/materials is this about? (e.g. ["polymers","conjugated_polymers","polythiophenes"])
2. by_application — what application domain? (e.g. ["medicine","drug_delivery","nanoparticle_carriers"])
3. by_technique — what technique is mentioned or relevant? (e.g. ["computation","dft","hybrid_functionals"])

Return JSON:
{{
  "by_substance_class": [...],
  "by_application": [...],
  "by_technique": [...]
}}"""


def load_corpus_metadata() -> dict[str, dict]:
    """Load paper metadata from corpus shards."""
    papers = {}
    shards = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith('.jsonl'))
    for shard in shards:
        with open(CORPUS_DIR / shard) as f:
            for line in f:
                paper = json.loads(line)
                doi = (paper.get('externalIds') or {}).get('DOI', '')
                if doi and doi.lower() not in papers:
                    papers[doi.lower()] = paper
    return papers


def load_deep_results() -> list[dict]:
    """Load all deep extraction result files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get('num_claims', 0) > 0:
                results.append(data)
        except Exception:
            pass
    return results


def make_content_summary(raw: dict) -> dict:
    """Compact summary for content-type claims."""
    s = {}
    for key in ['claim_type', 'reaction_type', 'subject', 'property_name',
                 'property_category', 'technique_name', 'process_described',
                 'comparison_result', 'metric', 'what_it_achieves', 'key_innovation']:
        if raw.get(key):
            s[key] = raw[key]
    quote = raw.get('verbatim_quote', '')
    if quote:
        s['verbatim_quote'] = quote[:200]
    for key in ['reactants', 'products']:
        items = raw.get(key, [])
        if items:
            s[key] = [r.get('name', '') for r in items[:3] if isinstance(r, dict)]
    cond = raw.get('conditions', {})
    if cond and isinstance(cond, dict):
        s['conditions'] = {k: v for k, v in cond.items() if v and v not in ('...', 'null', 'N/A')}
    compared = raw.get('compared_items', [])
    if compared:
        s['compared_items'] = compared[:3]
    return s


def make_epistemic_summary(raw: dict) -> dict:
    """Compact summary for epistemic-type claims — focus on the scientific content."""
    s = {}
    for key in ['hypothesis_text', 'limitation_text', 'direction_text',
                 'finding_text', 'why_surprising', 'verbatim_quote',
                 'subject', 'property_name', 'technique_name',
                 'process_described', 'what_it_achieves', 'reaction_type']:
        val = raw.get(key)
        if val:
            s[key] = val[:250] if isinstance(val, str) else val
    for key in ['reactants', 'products']:
        items = raw.get(key, [])
        if items:
            s[key] = [r.get('name', '') for r in items[:3] if isinstance(r, dict)]
    return s


def get_claim_type_path(claim_type: str, raw: dict) -> list[str]:
    """Deterministic by_claim_type path: L1 = type label, L2 = subfield from paper."""
    l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
    return [l1]


def cmd_prepare(args):
    """Build Batch API JSONL files for claim classification."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    results = load_deep_results()
    print(f"Loaded {len(results)} deep extraction results", flush=True)

    # Build all claim records
    all_claims = []
    for result in results:
        doi = result.get('doi', '')
        subfield = result.get('data', {}).get('paper_knowledge', {}).get('subfield', '')
        for raw_claim in result.get('data', {}).get('claims', []):
            claim_type = raw_claim.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)
            all_claims.append({
                'claim_id': claim_id,
                'doi': doi,
                'claim_type': claim_type,
                'subfield': subfield,
                'raw_claim': raw_claim,
            })

    print(f"Total claims: {len(all_claims):,}", flush=True)

    # Split by track
    content_claims = [c for c in all_claims if c['claim_type'] in CONTENT_TYPES]
    epistemic_claims = [c for c in all_claims if c['claim_type'] in EPISTEMIC_TYPES]
    other_claims = [c for c in all_claims if c['claim_type'] not in CONTENT_TYPES and c['claim_type'] not in EPISTEMIC_TYPES]

    print(f"  Content claims (5-view): {len(content_claims):,}", flush=True)
    print(f"  Epistemic claims (3-view): {len(epistemic_claims):,}", flush=True)
    print(f"  Other/unknown: {len(other_claims):,}", flush=True)

    # Check already classified
    classified_file = PIPELINE_DIR / "classifications.json"
    already = set()
    if classified_file.exists():
        already = set(json.loads(classified_file.read_text()).keys())
    print(f"Already classified: {len(already):,}", flush=True)

    # Build batch requests
    MAX_PER_FILE = 50_000
    batch_idx = 0
    batch_files = []
    current_file = None
    count_in_file = 0
    total_requests = 0

    def write_request(custom_id, prompt):
        nonlocal current_file, count_in_file, batch_idx, total_requests
        if current_file is None or count_in_file >= MAX_PER_FILE:
            if current_file:
                current_file.close()
            batch_idx += 1
            fname = PIPELINE_DIR / f"classify_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            count_in_file = 0
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
        }
        current_file.write(json.dumps(request) + "\n")
        count_in_file += 1
        total_requests += 1

    # Track A: content claims → 5-view prompt
    for c in content_claims:
        if c['claim_id'] in already:
            continue
        summary = make_content_summary(c['raw_claim'])
        prompt = CONTENT_PROMPT.format(claim_json=json.dumps(summary))
        write_request(c['claim_id'], prompt)

    # Track B: epistemic claims → 3-view prompt
    for c in epistemic_claims:
        if c['claim_id'] in already:
            continue
        ct = c['claim_type']
        label = CLAIM_TYPE_LABELS.get(ct, ct).replace('_', ' ')
        summary = make_epistemic_summary(c['raw_claim'])
        prompt = EPISTEMIC_PROMPT.format(
            claim_type_label=label,
            claim_json=json.dumps(summary),
        )
        write_request(c['claim_id'], prompt)

    # Other claims: treat as content
    for c in other_claims:
        if c['claim_id'] in already:
            continue
        summary = make_content_summary(c['raw_claim'])
        prompt = CONTENT_PROMPT.format(claim_json=json.dumps(summary))
        write_request(c['claim_id'], prompt)

    if current_file:
        current_file.close()

    # Save metadata for the build step
    claim_meta = {}
    for c in all_claims:
        claim_meta[c['claim_id']] = {
            'doi': c['doi'],
            'claim_type': c['claim_type'],
            'subfield': c['subfield'],
        }
    json.dump(claim_meta, open(PIPELINE_DIR / "claim_meta.json", 'w'))

    meta = {
        "total_requests": total_requests,
        "content_claims": len(content_claims),
        "epistemic_claims": len(epistemic_claims),
        "other_claims": len(other_claims),
        "batch_files": [f.name for f in batch_files],
        "created_at": datetime.now().isoformat(),
    }
    json.dump(meta, open(PIPELINE_DIR / "classify_meta.json", 'w'), indent=2)

    for bf in batch_files:
        lines = sum(1 for _ in open(bf))
        print(f"  {bf.name}: {bf.stat().st_size / 1e6:.1f} MB, {lines:,} requests", flush=True)
    print(f"\nPrepared {len(batch_files)} batch files, {total_requests:,} requests")
    print(f"Submit with: python src/integrate_deep.py submit")


def cmd_submit(args):
    """Submit classification batches to OpenAI."""
    client = OpenAI()
    batch_files = sorted(PIPELINE_DIR.glob("classify_*.jsonl"))
    if not batch_files:
        print("No batch files. Run 'prepare' first.")
        return

    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    tracker = {}
    if tracker_file.exists():
        tracker = json.loads(tracker_file.read_text())

    for fpath in batch_files:
        if fpath.name in tracker and tracker[fpath.name].get('status') not in ('failed', 'expired', 'cancelled', 'cancelling'):
            print(f"  {fpath.name}: already submitted ({tracker[fpath.name].get('status')})")
            continue

        print(f"  Uploading {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB)...", flush=True)
        uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        tracker[fpath.name] = {
            "batch_id": batch.id,
            "file_id": uploaded.id,
            "status": batch.status,
        }
        json.dump(tracker, open(tracker_file, 'w'), indent=2)
        print(f"  Batch {batch.id} ({batch.status})")
        time.sleep(2)

    print(f"\n{len(tracker)} batches tracked. Poll with: python src/integrate_deep.py poll")


def cmd_poll(args):
    """Check classification batch status."""
    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No batches submitted.")
        return

    client = OpenAI()
    tracker = json.loads(tracker_file.read_text())
    completed = 0; failed = 0; total = 0

    for fname, info in sorted(tracker.items()):
        b = client.batches.retrieve(info['batch_id'])
        info['status'] = b.status
        if b.output_file_id:
            info['output_file_id'] = b.output_file_id
        rc = b.request_counts
        if rc:
            completed += rc.completed
            failed += rc.failed
            total += rc.total
        if b.status != 'completed':
            print(f"  {fname}: {b.status} ({rc.completed}/{rc.total})" if rc else f"  {fname}: {b.status}")

    json.dump(tracker, open(tracker_file, 'w'), indent=2)
    st = Counter(v['status'] for v in tracker.values())
    print(f"\nStatuses: {dict(st)}")
    print(f"Overall: {completed}/{total} done, {failed} failed")


def cmd_collect(args):
    """Download classification results and merge."""
    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No batches to collect.")
        return

    client = OpenAI()
    tracker = json.loads(tracker_file.read_text())

    classifications = {}
    classified_file = PIPELINE_DIR / "classifications.json"
    if classified_file.exists():
        classifications = json.loads(classified_file.read_text())

    errors = 0
    empty_content = 0
    new_count = 0

    for fname, info in sorted(tracker.items()):
        output_id = info.get('output_file_id')
        if not output_id:
            continue

        raw_path = PIPELINE_DIR / "raw" / fname
        raw_path.parent.mkdir(exist_ok=True)

        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            raw_path.write_bytes(content.read())

        for line in open(raw_path):
            try:
                result = json.loads(line)
                cid = result.get('custom_id', '')
                resp = result.get('response', {})
                body = resp.get('body', {})

                if resp.get('status_code') != 200:
                    errors += 1
                    continue

                text = body.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not text or not text.strip():
                    empty_content += 1
                    continue

                paths = json.loads(text)
                if cid not in classifications:
                    classifications[cid] = paths
                    new_count += 1
            except json.JSONDecodeError:
                errors += 1
            except Exception:
                errors += 1

    classified_file.write_text(json.dumps(classifications))
    print(f"Classifications: {len(classifications):,} total ({new_count} new)")
    if errors:
        print(f"  API errors: {errors}")
    if empty_content:
        print(f"  Empty content (token limit): {empty_content}")


def _load_sources_jsonl() -> dict[str, dict]:
    """Load paper metadata from askchem/sources.jsonl."""
    sources_file = Path(__file__).parent.parent / "askchem" / "sources.jsonl"
    papers = {}
    if not sources_file.exists():
        return papers
    with open(sources_file) as f:
        for line in f:
            paper = json.loads(line)
            doi = paper.get('doi', '').lower()
            if doi:
                papers[doi] = paper
    return papers


def _build_searchable_text(raw_claim: dict) -> str:
    """Build full-text search string from a claim's fields."""
    parts = []
    for key in ['claim_type', 'verbatim_quote', 'subject', 'property_name',
                 'reaction_type', 'technique_name', 'process_described',
                 'hypothesis_text', 'finding_text', 'limitation_text',
                 'direction_text', 'comparison_result', 'what_it_achieves',
                 'key_innovation']:
        v = raw_claim.get(key, '')
        if isinstance(v, list):
            parts.extend(str(x) for x in v if x)
        elif v:
            parts.append(str(v))
    for key in ['reactants', 'products']:
        items = raw_claim.get(key) or []
        for item in items:
            if isinstance(item, dict):
                parts.append(item.get('name', ''))
            elif isinstance(item, str) and item:
                parts.append(item)
    return ' '.join(p for p in parts if p)


def _normalize_view_path(p):
    """Clean a single view path list: handle nested lists, stringify segments."""
    if not p or p == ['not_applicable']:
        return None
    if p and isinstance(p[0], list):
        p = p[0]
    p = [str(s) for s in p if isinstance(s, (str, int, float))]
    if p and p != ['not_applicable']:
        return p
    return None


def _insert_source_from_meta(sm, doi, sources_seen, source_batch):
    """Build a source row tuple from sources.jsonl metadata."""
    if doi.lower() in sources_seen:
        return 0
    sources_seen.add(doi.lower())
    authors = sm.get('authors', [])
    if authors and isinstance(authors[0], dict):
        authors = [a.get('name', '') for a in authors[:20]]
    source_data = {
        'doi': doi,
        'title': sm.get('title', ''),
        'authors': authors,
        'year': sm.get('year') or 0,
        'venue': sm.get('venue', ''),
        'abstract': sm.get('abstract', ''),
        'citation_count': sm.get('citation_count', 0) or 0,
        'open_access_url': sm.get('open_access_url', ''),
    }
    source_batch.append((
        doi, source_data['title'], json.dumps(authors),
        source_data['year'], source_data['venue'], source_data['abstract'],
        source_data['citation_count'], source_data['open_access_url'],
        json.dumps(source_data),
    ))
    return 1


def _track_tree_nodes(view_paths, claim_id, node_counts, node_claims):
    """Accumulate tree node counts and claim lists from a claim's view_paths."""
    for view_id, path_segs in view_paths.items():
        if not path_segs:
            continue
        full_path = '/'.join(str(s) for s in path_segs)
        for depth in range(len(path_segs)):
            partial = '/'.join(str(s) for s in path_segs[:depth + 1])
            node_counts[view_id][partial] += 1
        node_claims[view_id][full_path].append(claim_id)


def cmd_build(args):
    """Build the SQLite index from existing claims + deep results."""
    import sqlite3

    ALL_CONTENT_VIEWS = ['by_reaction_type', 'by_substance_class', 'by_application',
                         'by_technique', 'by_mechanism']

    print("Loading paper metadata from askchem/sources.jsonl...", flush=True)
    source_meta = _load_sources_jsonl()
    print(f"  {len(source_meta):,} papers loaded", flush=True)

    results = load_deep_results()
    print(f"Loaded {len(results)} deep extraction results", flush=True)

    # ── Initialize DB (fresh) ──
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  Removed old {DB_PATH.name}", flush=True)
    from askchem import db
    db.init_db()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")

    from askchem.models import DEFAULT_VIEWS
    for view in DEFAULT_VIEWS:
        c.execute(
            "INSERT OR REPLACE INTO views (view_id, name, description, data) VALUES (?,?,?,?)",
            (view.view_id, view.name, view.description, json.dumps(view.to_dict()))
        )
    conn.commit()
    print(f"  Inserted {len(DEFAULT_VIEWS)} views", flush=True)

    source_batch = []
    claim_batch = []
    fts_batch = []
    node_counts = defaultdict(lambda: defaultdict(int))
    node_claims = defaultdict(lambda: defaultdict(list))
    total_claims = 0
    total_classified = 0
    total_sources = 0
    sources_seen = set()
    claim_ids_seen = set()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Existing claims from askchem/claims.jsonl
    # ════════════════════════════════════════════════════════════════════
    existing_claims_file = Path(__file__).parent.parent / "askchem" / "claims.jsonl"
    if existing_claims_file.exists():
        print("Phase 1: Loading existing claims from askchem/claims.jsonl...", flush=True)
        with open(existing_claims_file) as f:
            for li, line in enumerate(f):
                claim = json.loads(line)
                claim_id = claim.get('claim_id', '')
                if not claim_id:
                    continue
                claim_ids_seen.add(claim_id)

                claim_type = claim.get('claim_type', 'unknown')
                doi = claim.get('source_doi', '')
                source_title = claim.get('source_paper_title', '')

                if doi:
                    sm = source_meta.get(doi.lower(), {})
                    total_sources += _insert_source_from_meta(
                        sm if sm else {'doi': doi, 'title': source_title},
                        doi, sources_seen, source_batch,
                    )

                view_paths = claim.get('view_paths', {})
                if view_paths and any(v for v in view_paths.values() if v):
                    total_classified += 1

                _track_tree_nodes(view_paths, claim_id, node_counts, node_claims)
                searchable = _build_searchable_text(claim)

                claim_batch.append((
                    claim_id, claim_type, doi, source_title,
                    claim.get('confidence', 'high'),
                    claim.get('location_in_paper', ''),
                    claim.get('verbatim_quote', ''),
                    claim.get('extraction_model', 'gpt-5-mini'),
                    claim.get('extraction_version', 'v3-abstract-batch'),
                    claim.get('extracted_at', ''),
                    json.dumps(view_paths),
                    json.dumps(claim),
                ))
                fts_batch.append((
                    claim_id, claim_type, source_title,
                    claim.get('verbatim_quote', ''),
                    searchable,
                ))
                total_claims += 1

                if (li + 1) % 50000 == 0:
                    _flush_batches(c, conn, source_batch, claim_batch, fts_batch)
                    source_batch.clear(); claim_batch.clear(); fts_batch.clear()
                    print(f"  Phase 1: {li+1:,} claims, {total_sources:,} sources", flush=True)

        _flush_batches(c, conn, source_batch, claim_batch, fts_batch)
        source_batch.clear(); claim_batch.clear(); fts_batch.clear()
        print(f"  Phase 1 done: {total_claims:,} existing claims, {total_sources:,} sources", flush=True)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Deep extraction results with inline Gemini classifications
    # ════════════════════════════════════════════════════════════════════
    print("Phase 2: Loading deep extraction results...", flush=True)
    deep_claims = 0
    deep_classified = 0
    deep_dupes = 0

    for ri, result in enumerate(results):
        doi = result.get('doi', '')
        if not doi:
            continue

        sm = source_meta.get(doi.lower(), {'doi': doi})
        total_sources += _insert_source_from_meta(sm, doi, sources_seen, source_batch)

        paper_knowledge = result.get('data', {}).get('paper_knowledge', {})
        subfield = paper_knowledge.get('subfield', '')
        source_title = source_meta.get(doi.lower(), {}).get('title', '')

        for raw_claim in result.get('data', {}).get('claims', []):
            claim_type = raw_claim.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            if claim_id in claim_ids_seen:
                deep_dupes += 1
                continue
            claim_ids_seen.add(claim_id)

            view_paths = {}
            inline_class = raw_claim.get('classification', {})
            if inline_class and isinstance(inline_class, dict):
                for view_id in ALL_CONTENT_VIEWS:
                    p = _normalize_view_path(inline_class.get(view_id))
                    if p:
                        view_paths[view_id] = p

            if view_paths:
                deep_classified += 1

            ct_l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
            ct_path = [ct_l1]
            if subfield:
                ct_path.append(subfield.lower().replace(' ', '_'))
            view_paths['by_claim_type'] = ct_path

            _track_tree_nodes(view_paths, claim_id, node_counts, node_claims)
            searchable = _build_searchable_text(raw_claim)

            claim_data = dict(raw_claim)
            ext_model = _normalize_extraction_model(
                result.get('extraction_model')
                or raw_claim.get('extraction_model')
            )
            claim_data.update({
                'claim_id': claim_id,
                'source_doi': doi,
                'source_paper_title': source_title,
                'extraction_model': ext_model,
                'extraction_version': 'deep_v1',
                'view_paths': view_paths,
            })

            claim_batch.append((
                claim_id, claim_type, doi, source_title,
                raw_claim.get('confidence', 'high'),
                raw_claim.get('location_in_paper', ''),
                raw_claim.get('verbatim_quote', ''),
                ext_model, 'deep_v1',
                result.get('collected_at', datetime.now().isoformat()),
                json.dumps(view_paths),
                json.dumps(claim_data),
            ))
            fts_batch.append((
                claim_id, claim_type, source_title,
                raw_claim.get('verbatim_quote', ''),
                searchable,
            ))
            total_claims += 1
            deep_claims += 1

        if (ri + 1) % 200 == 0:
            _flush_batches(c, conn, source_batch, claim_batch, fts_batch)
            source_batch.clear(); claim_batch.clear(); fts_batch.clear()
            print(f"  Phase 2: {ri+1}/{len(results)} papers, {deep_claims:,} deep claims", flush=True)

    _flush_batches(c, conn, source_batch, claim_batch, fts_batch)
    print(f"\n  Total sources: {total_sources:,}", flush=True)
    print(f"  Total claims:  {total_claims:,}", flush=True)
    print(f"    Existing:    {total_claims - deep_claims:,}", flush=True)
    print(f"    Deep:        {deep_claims:,} ({deep_classified:,} classified, {deep_dupes:,} dupes skipped)", flush=True)
    print(f"  Total classified: {total_classified + deep_classified:,}", flush=True)

    # ── Build tree nodes ──
    print("Building tree nodes...", flush=True)
    node_batch = []
    for view_id, paths in node_counts.items():
        children_map = defaultdict(set)
        for path_str in paths:
            segments = path_str.split('/')
            if len(segments) > 1:
                parent = '/'.join(segments[:-1])
                children_map[parent].add(segments[-1])

        for path_str, count in paths.items():
            segments = path_str.split('/')
            level = len(segments)
            name = smart_title(segments[-1])
            children = sorted(children_map.get(path_str, set()))
            claim_ids = node_claims[view_id].get(path_str, [])

            node_data = {
                'view_id': view_id,
                'path': path_str,
                'name': name,
                'level': level,
                'claim_count': count,
                'children': children,
                'claim_ids': claim_ids[:2000],
            }
            node_batch.append((
                view_id, path_str, name, level, count,
                json.dumps(children),
                json.dumps(claim_ids[:2000]),
                json.dumps(node_data),
            ))
        print(f"    {view_id}: {len([p for p in paths]):,} nodes", flush=True)

    c.executemany(
        "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
        node_batch,
    )
    conn.commit()

    total_nodes = len(node_batch)
    nodes_per_view = Counter(n[0] for n in node_batch)
    print(f"  Tree nodes: {total_nodes:,}")
    for v, cnt in nodes_per_view.most_common():
        print(f"    {v}: {cnt:,} nodes")

    # ── Create root nodes for each view ──
    print("Creating root nodes...", flush=True)
    for view_id in nodes_per_view:
        l1_nodes = c.execute(
            "SELECT path, claim_count FROM tree_nodes WHERE view_id=? AND level=1 ORDER BY claim_count DESC",
            (view_id,)
        ).fetchall()
        children = [row[0] for row in l1_nodes]
        total = sum(row[1] for row in l1_nodes)
        root_data = {
            'view_id': view_id, 'path': '', 'name': view_id,
            'level': 0, 'claim_count': total,
            'children': children, 'claim_ids': [],
        }
        c.execute(
            "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            (view_id, '', view_id, 0, total, json.dumps(children), json.dumps([]), json.dumps(root_data))
        )
    conn.commit()

    # ── Populate metadata table ──
    view_count = len(nodes_per_view) + 1
    total_classified_all = total_classified + deep_classified
    for k, v in [
        ('total_claims', str(total_claims)),
        ('total_sources', str(total_sources)),
        ('total_nodes', str(total_nodes)),
        ('total_views', str(view_count)),
        ('version', '2.1.0'),
        ('extraction_model', 'gemini-3.1-pro'),
        ('classification_model', 'gemini-3.1-pro + gpt-5-mini'),
        ('built_at', datetime.now().isoformat()),
    ]:
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)", (k, v))
    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"INDEX BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Sources:       {total_sources:,}")
    print(f"Claims:        {total_claims:,}")
    print(f"  Existing:    {total_claims - deep_claims:,}")
    print(f"  Deep:        {deep_claims:,}")
    print(f"Classified:    {total_classified_all:,}")
    print(f"Tree nodes:    {total_nodes:,}")
    print(f"Views:         {view_count}")
    print(f"Database:      {DB_PATH}")


def _flush_batches(c, conn, source_batch, claim_batch, fts_batch):
    if source_batch:
        c.executemany(
            "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
            source_batch,
        )
    if claim_batch:
        c.executemany(
            "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            claim_batch,
        )
    if fts_batch:
        c.executemany(
            "INSERT OR REPLACE INTO claims_fts (claim_id,claim_type,source_paper_title,verbatim_quote,searchable_text) VALUES (?,?,?,?,?)",
            fts_batch,
        )
    conn.commit()


def cmd_status(args):
    """Show integration pipeline status."""
    import sqlite3

    print("=== Integration Pipeline Status ===\n")

    if RESULTS_DIR.exists():
        n = len(list(RESULTS_DIR.glob("*.json")))
        print(f"Deep extraction results: {n}")

    meta_file = PIPELINE_DIR / "classify_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        print(f"Content claims (5-view): {meta.get('content_claims', '?'):,}")
        print(f"Epistemic claims (3-view): {meta.get('epistemic_claims', '?'):,}")

    classified_file = PIPELINE_DIR / "classifications.json"
    if classified_file.exists():
        data = json.loads(classified_file.read_text())
        print(f"Classifications collected: {len(data):,}")
    else:
        print("Classifications: none")

    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if tracker_file.exists():
        tracker = json.loads(tracker_file.read_text())
        st = Counter(v['status'] for v in tracker.values())
        print(f"Batch status: {dict(st)}")

    if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
        conn = sqlite3.connect(str(DB_PATH))
        for table in ['sources', 'claims', 'tree_nodes', 'views']:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"DB {table}: {n:,}")
            except:
                pass
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Integrate deep extractions into AskChem index")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prepare", help="Build classification batch files")
    sub.add_parser("submit", help="Submit to Batch API")
    sub.add_parser("poll", help="Check batch status")
    sub.add_parser("collect", help="Download classification results")
    sub.add_parser("build", help="Build SQLite index")
    sub.add_parser("status", help="Show progress")

    args = parser.parse_args()
    cmd_map = {
        'prepare': cmd_prepare, 'submit': cmd_submit, 'poll': cmd_poll,
        'collect': cmd_collect, 'build': cmd_build, 'status': cmd_status,
    }
    if args.command in cmd_map:
        cmd_map[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
