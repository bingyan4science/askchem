"""
Classify and integrate Tier A deep extraction results into chemtree.db.

Uses the canonical L1/L2/L3 taxonomy from taxonomy.py for constrained classification.
Pipeline: prepare → submit → poll → collect → l3-prepare → l3-submit → l3-poll → l3-collect → build

Usage:
    python src/classify_tier_a.py prepare        # Build L1+L2 classification batch
    python src/classify_tier_a.py submit         # Submit to Batch API
    python src/classify_tier_a.py poll           # Check status
    python src/classify_tier_a.py collect        # Download and parse L1+L2 results
    python src/classify_tier_a.py l3-prepare     # Build L3 assignment batch
    python src/classify_tier_a.py l3-submit      # Submit L3 batch
    python src/classify_tier_a.py l3-poll        # Check L3 status
    python src/classify_tier_a.py l3-collect     # Download and merge L3 results
    python src/classify_tier_a.py build          # Insert claims + tree nodes into DB
    python src/classify_tier_a.py status         # Show pipeline progress
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from askchem.models import Claim
from askchem.display import smart_title
from askchem.taxonomy import (
    CANONICAL_L1, CANONICAL_L2, CANONICAL_L3,
    ALL_CONTENT_VIEWS,
    CLASSIFICATION_SYSTEM_PROMPT,
    L3_ASSIGNMENT_SYSTEM_PROMPT,
    build_classification_prompt,
    build_classification_messages,
    build_l3_assignment_messages,
    normalize_path,
    get_canonical_l3,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "deep_results"
PIPELINE_DIR = DATA_DIR / "classify_pipeline"
OA_SCAN = DATA_DIR / "oa_scan.json"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"

CLASSIFICATION_MODEL = "gpt-5-mini"
MAX_BATCH_FILE_BYTES = 190 * 1024 * 1024
MAX_BATCH_REQUESTS = 49_000

CONTENT_TYPES = {
    'reaction', 'property', 'method', 'mechanism', 'comparison',
    'scope_entry', 'computational_result',
}

EPISTEMIC_TYPES = {
    'hypothesis', 'conclusion', 'conclusions', 'limitation',
    'future_direction', 'surprising_finding', 'experimental_design',
    'structure', 'background', 'historical', 'definition', 'observation',
}

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

EPISTEMIC_SYSTEM_PROMPT = f"""Classify this scientific claim into the chemistry topic areas where it is relevant.
Focus on the scientific subject matter, not the epistemic role.

Rules:
- L1 MUST be one of the listed categories (exactly one per view).
- L2 MUST be one of the listed subcategories under that L1.
- Use lowercase_with_underscores.
- If the claim does not fit a view, use ["not_applicable"].

Canonical categories (L1 → allowed L2):
"""


def _build_epistemic_taxonomy_text():
    lines = []
    epistemic_views = ['by_substance_class', 'by_application', 'by_technique']
    for view_id in epistemic_views:
        lines.append(f"\n{view_id}:")
        l2_map = CANONICAL_L2.get(view_id, {})
        for l1 in CANONICAL_L1[view_id]:
            l2s = l2_map.get(l1, ["other"])
            lines.append(f"  {l1}: {', '.join(l2s)}")
    return "\n".join(lines)


_EPISTEMIC_TAXONOMY = _build_epistemic_taxonomy_text()

EPISTEMIC_CLASSIFICATION_PROMPT = EPISTEMIC_SYSTEM_PROMPT + _EPISTEMIC_TAXONOMY + """

Return JSON: {"by_substance_class": ["l1", "l2"], "by_application": ["l1", "l2"], "by_technique": ["l1", "l2"]}"""


def load_deep_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get('num_claims', 0) > 0:
                results.append(data)
        except Exception:
            pass
    return results


def load_paper_titles():
    titles = {}
    if OA_SCAN.exists():
        with open(OA_SCAN) as f:
            scan = json.load(f)
        for p in scan['papers']:
            if p.get('doi') and p.get('title'):
                titles[p['doi'].lower()] = p['title']
    return titles


def cmd_prepare(args):
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    results = load_deep_results()
    titles = load_paper_titles()
    print(f"Loaded {len(results)} deep extraction results", flush=True)

    all_claims = []
    for result in results:
        doi = result.get('doi', '')
        subfield = result.get('data', {}).get('paper_knowledge', {}).get('subfield', '')
        title = titles.get(doi.lower(), '')
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
                'title': title,
                'quote': (raw_claim.get('verbatim_quote') or '')[:300],
                'raw_claim': raw_claim,
            })

    print(f"Total claims: {len(all_claims):,}", flush=True)

    content_claims = [c for c in all_claims if c['claim_type'] in CONTENT_TYPES]
    epistemic_claims = [c for c in all_claims if c['claim_type'] in EPISTEMIC_TYPES]
    other_claims = [c for c in all_claims
                    if c['claim_type'] not in CONTENT_TYPES
                    and c['claim_type'] not in EPISTEMIC_TYPES]

    print(f"  Content claims (5-view): {len(content_claims):,}", flush=True)
    print(f"  Epistemic claims (3-view): {len(epistemic_claims):,}", flush=True)
    print(f"  Other/unknown: {len(other_claims):,}", flush=True)

    already = set()
    classified_file = PIPELINE_DIR / "classifications.json"
    if classified_file.exists():
        already = set(json.loads(classified_file.read_text()).keys())
    print(f"Already classified: {len(already):,}", flush=True)

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    total_requests = 0
    batch_files = []

    def write_request(custom_id, messages):
        nonlocal current_file, current_size, items_in_batch, batch_idx, total_requests
        if current_file is None or current_size + 4096 > MAX_BATCH_FILE_BYTES or items_in_batch >= MAX_BATCH_REQUESTS:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {items_in_batch:,} requests, "
                      f"{current_size / 1e6:.1f} MB", flush=True)
            batch_idx += 1
            fname = PIPELINE_DIR / f"classify_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            current_size = 0
            items_in_batch = 0

        request = {
            "custom_id": f"cls_{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CLASSIFICATION_MODEL,
                "messages": messages,
                "max_completion_tokens": 16384,
                "response_format": {"type": "json_object"},
            },
        }
        line = json.dumps(request) + "\n"
        line_bytes = len(line.encode('utf-8'))
        current_file.write(line)
        current_size += line_bytes
        items_in_batch += 1
        total_requests += 1

    for c in content_claims:
        if c['claim_id'] in already:
            continue
        msgs = build_classification_messages(c['claim_type'], c['quote'], c['title'])
        write_request(c['claim_id'], msgs)

    for c in epistemic_claims:
        if c['claim_id'] in already:
            continue
        user_msg = build_classification_prompt(c['claim_type'], c['quote'], c['title'])
        msgs = [
            {"role": "system", "content": EPISTEMIC_CLASSIFICATION_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        write_request(c['claim_id'], msgs)

    for c in other_claims:
        if c['claim_id'] in already:
            continue
        msgs = build_classification_messages(c['claim_type'], c['quote'], c['title'])
        write_request(c['claim_id'], msgs)

    if current_file:
        current_file.close()
        if items_in_batch > 0:
            print(f"  {batch_files[-1].name}: {items_in_batch:,} requests, "
                  f"{current_size / 1e6:.1f} MB", flush=True)

    claim_meta = {}
    for c in all_claims:
        claim_meta[c['claim_id']] = {
            'doi': c['doi'],
            'claim_type': c['claim_type'],
            'subfield': c['subfield'],
            'title': c['title'],
        }
    with open(PIPELINE_DIR / "claim_meta.json", 'w') as f:
        json.dump(claim_meta, f)

    meta = {
        "total_requests": total_requests,
        "content_claims": len(content_claims),
        "epistemic_claims": len(epistemic_claims),
        "other_claims": len(other_claims),
        "batch_files": [f.name for f in batch_files],
        "created_at": datetime.now().isoformat(),
    }
    with open(PIPELINE_DIR / "classify_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nPrepared {total_requests:,} requests in {len(batch_files)} batch files")
    print(f"Submit with: python src/classify_tier_a.py submit")


def cmd_submit(args):
    client = OpenAI(timeout=120.0)
    batch_files = sorted(PIPELINE_DIR.glob("classify_*.jsonl"))
    if not batch_files:
        print("No batch files. Run 'prepare' first.")
        return

    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    tracker = {}
    if tracker_file.exists():
        with open(tracker_file) as f:
            tracker = json.load(f)

    for fpath in batch_files:
        if fpath.name in tracker and tracker[fpath.name].get('status') not in ('failed', 'expired', 'cancelled'):
            print(f"  {fpath.name}: already submitted ({tracker[fpath.name].get('status')})")
            continue

        size_mb = fpath.stat().st_size / 1e6
        print(f"  Uploading {fpath.name} ({size_mb:.1f} MB)...", flush=True)
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
            "submitted_at": datetime.now().isoformat(),
        }
        with open(tracker_file, 'w') as f:
            json.dump(tracker, f, indent=2)
        print(f"  Batch {batch.id} ({batch.status})")
        time.sleep(2)

    print(f"\n{len(tracker)} batches tracked. Poll with: python src/classify_tier_a.py poll")


def cmd_poll(args):
    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No batches submitted.")
        return

    client = OpenAI()
    with open(tracker_file) as f:
        tracker = json.load(f)

    all_done = True
    total_completed = 0
    total_failed = 0
    total_total = 0

    for fname, info in sorted(tracker.items()):
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        if batch.output_file_id:
            info["output_file_id"] = batch.output_file_id

        status_str = batch.status
        if batch.request_counts:
            rc = batch.request_counts
            status_str += f" ({rc.completed}/{rc.total} done, {rc.failed} failed)"
            total_completed += rc.completed
            total_failed += rc.failed
            total_total += rc.total

        print(f"  {fname}: {status_str}")

        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_done = False

    with open(tracker_file, 'w') as f:
        json.dump(tracker, f, indent=2)

    print(f"\nOverall: {total_completed}/{total_total} done, {total_failed} failed")
    if all_done:
        print("All done! Collect with: python src/classify_tier_a.py collect")


def cmd_collect(args):
    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No batches to collect.")
        return

    client = OpenAI()
    with open(tracker_file) as f:
        tracker = json.load(f)

    classifications = {}
    classified_file = PIPELINE_DIR / "classifications.json"
    if classified_file.exists():
        classifications = json.loads(classified_file.read_text())

    errors = 0
    empty = 0
    new_count = 0

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            print(f"  {fname}: no output (status={info.get('status')})")
            continue

        raw_path = PIPELINE_DIR / "raw" / fname
        raw_path.parent.mkdir(exist_ok=True)

        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            raw_path.write_bytes(content.read())
            print(f"  Saved ({raw_path.stat().st_size / 1e6:.1f} MB)")

        for line in open(raw_path):
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                claim_id = custom_id.replace("cls_", "")
                response = result.get("response", {})
                body = response.get("body", {})

                if response.get("status_code") != 200:
                    errors += 1
                    continue

                text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not text:
                    empty += 1
                    continue

                parsed = json.loads(text)
                normalized = {}
                for view_id in ALL_CONTENT_VIEWS:
                    raw_path_val = parsed.get(view_id)
                    if raw_path_val and raw_path_val != ["not_applicable"]:
                        normed = normalize_path(view_id, raw_path_val)
                        if normed:
                            normalized[view_id] = normed

                if normalized and claim_id not in classifications:
                    classifications[claim_id] = normalized
                    new_count += 1

            except json.JSONDecodeError:
                errors += 1
            except Exception:
                errors += 1

    classified_file.write_text(json.dumps(classifications))
    print(f"\nClassifications: {len(classifications):,} total ({new_count} new)")
    print(f"Errors: {errors}, Empty: {empty}")
    print(f"\nNext: python src/classify_tier_a.py l3-prepare")


def cmd_l3_prepare(args):
    classified_file = PIPELINE_DIR / "classifications.json"
    if not classified_file.exists():
        print("No classifications.json. Run collect first.")
        return

    classifications = json.loads(classified_file.read_text())
    meta_file = PIPELINE_DIR / "claim_meta.json"
    claim_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    l3_done_file = PIPELINE_DIR / "l3_done.json"
    l3_done = set()
    if l3_done_file.exists():
        l3_done = set(json.loads(l3_done_file.read_text()))

    needs_l3 = []
    for claim_id, view_paths in classifications.items():
        if claim_id in l3_done:
            continue
        for vid, path in view_paths.items():
            if not isinstance(path, list) or len(path) < 2:
                continue
            l3_cats = get_canonical_l3(vid, path[0], path[1])
            if l3_cats is not None:
                meta = claim_meta.get(claim_id, {})
                needs_l3.append({
                    'claim_id': claim_id,
                    'claim_type': meta.get('claim_type', 'property'),
                    'title': meta.get('title', ''),
                    'quote': '',
                    'view_paths': view_paths,
                })
                break

    print(f"Claims needing L3: {len(needs_l3):,}")
    if not needs_l3:
        print("No L3 assignment needed!")
        return

    # We need verbatim_quote for L3. Load from deep results.
    quote_map = {}
    results = load_deep_results()
    for result in results:
        doi = result.get('doi', '')
        subfield = result.get('data', {}).get('paper_knowledge', {}).get('subfield', '')
        for raw_claim in result.get('data', {}).get('claims', []):
            claim_type = raw_claim.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            cid = Claim.generate_id(doi, claim_type, content_hash)
            q = raw_claim.get('verbatim_quote', '')
            if q:
                quote_map[cid] = q[:300]

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    total_requests = 0
    batch_files = []

    for c in needs_l3:
        quote = quote_map.get(c['claim_id'], '')
        msgs = build_l3_assignment_messages(
            c['claim_type'], quote, c['title'], c['view_paths'],
        )
        if not msgs:
            continue

        request = {
            "custom_id": f"l3_{c['claim_id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CLASSIFICATION_MODEL,
                "messages": msgs,
                "max_completion_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
        }
        line = json.dumps(request) + "\n"
        line_bytes = len(line.encode('utf-8'))

        if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES or items_in_batch >= MAX_BATCH_REQUESTS:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {items_in_batch:,} requests", flush=True)
            batch_idx += 1
            fname = PIPELINE_DIR / f"l3_batch_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            current_size = 0
            items_in_batch = 0

        current_file.write(line)
        current_size += line_bytes
        items_in_batch += 1
        total_requests += 1

    if current_file:
        current_file.close()
        print(f"  {batch_files[-1].name}: {items_in_batch:,} requests", flush=True)

    print(f"\nPrepared {total_requests:,} L3 requests in {len(batch_files)} files")
    print(f"Submit with: python src/classify_tier_a.py l3-submit")


def cmd_l3_submit(args):
    client = OpenAI(timeout=120.0)
    batch_files = sorted(PIPELINE_DIR.glob("l3_batch_*.jsonl"))
    if not batch_files:
        print("No L3 batch files. Run 'l3-prepare' first.")
        return

    tracker_file = PIPELINE_DIR / "l3_tracker.json"
    tracker = {}
    if tracker_file.exists():
        with open(tracker_file) as f:
            tracker = json.load(f)

    for fpath in batch_files:
        if fpath.name in tracker and tracker[fpath.name].get('status') not in ('failed', 'expired', 'cancelled'):
            print(f"  {fpath.name}: already submitted ({tracker[fpath.name].get('status')})")
            continue

        size_mb = fpath.stat().st_size / 1e6
        print(f"  Uploading {fpath.name} ({size_mb:.1f} MB)...", flush=True)
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
        with open(tracker_file, 'w') as f:
            json.dump(tracker, f, indent=2)
        print(f"  Batch {batch.id} ({batch.status})")
        time.sleep(2)

    print(f"\n{len(tracker)} L3 batches tracked.")


def cmd_l3_poll(args):
    tracker_file = PIPELINE_DIR / "l3_tracker.json"
    if not tracker_file.exists():
        print("No L3 batches submitted.")
        return

    client = OpenAI()
    with open(tracker_file) as f:
        tracker = json.load(f)

    all_done = True
    total_completed = 0
    total_total = 0

    for fname, info in sorted(tracker.items()):
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        if batch.output_file_id:
            info["output_file_id"] = batch.output_file_id

        status_str = batch.status
        if batch.request_counts:
            rc = batch.request_counts
            status_str += f" ({rc.completed}/{rc.total} done, {rc.failed} failed)"
            total_completed += rc.completed
            total_total += rc.total

        print(f"  {fname}: {status_str}")
        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_done = False

    with open(tracker_file, 'w') as f:
        json.dump(tracker, f, indent=2)

    print(f"\nOverall: {total_completed}/{total_total}")
    if all_done:
        print("All done! Collect with: python src/classify_tier_a.py l3-collect")


def cmd_l3_collect(args):
    tracker_file = PIPELINE_DIR / "l3_tracker.json"
    if not tracker_file.exists():
        print("No L3 batches.")
        return

    client = OpenAI()
    with open(tracker_file) as f:
        tracker = json.load(f)

    classified_file = PIPELINE_DIR / "classifications.json"
    classifications = json.loads(classified_file.read_text())

    updated = 0
    errors = 0

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            continue

        raw_path = PIPELINE_DIR / "raw_l3" / fname
        raw_path.parent.mkdir(exist_ok=True)

        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            raw_path.write_bytes(content.read())

        for line in open(raw_path):
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                claim_id = custom_id.replace("l3_", "")
                response = result.get("response", {})
                body = response.get("body", {})

                if response.get("status_code") != 200:
                    errors += 1
                    continue

                text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not text:
                    errors += 1
                    continue

                l3_result = json.loads(text)

                if claim_id in classifications:
                    existing = classifications[claim_id]
                    for vid, l3_val in l3_result.items():
                        if vid in existing and isinstance(existing[vid], list) and len(existing[vid]) >= 2:
                            l3_str = str(l3_val).strip().lower().replace('-', '_').replace(' ', '_')
                            allowed = get_canonical_l3(vid, existing[vid][0], existing[vid][1])
                            if allowed and l3_str in set(allowed):
                                existing[vid] = existing[vid][:2] + [l3_str]
                            elif allowed:
                                existing[vid] = existing[vid][:2] + ["other"]
                    updated += 1

            except Exception:
                errors += 1

    classified_file.write_text(json.dumps(classifications))
    l3_done = [cid for cid in classifications if any(
        len(p) >= 3 for p in classifications[cid].values() if isinstance(p, list)
    )]
    with open(PIPELINE_DIR / "l3_done.json", 'w') as f:
        json.dump(l3_done, f)

    print(f"L3 updated: {updated:,}, Errors: {errors}")
    print(f"Claims with L3: {len(l3_done):,}")
    print(f"\nNext: python src/classify_tier_a.py build")


def cmd_build(args):
    classified_file = PIPELINE_DIR / "classifications.json"
    if not classified_file.exists():
        print("No classifications.json. Run collect first.")
        return

    classifications = json.loads(classified_file.read_text())
    print(f"Classifications loaded: {len(classifications):,}")

    results = load_deep_results()
    titles = load_paper_titles()
    print(f"Deep results: {len(results)}")

    # Load corpus metadata for richer source info
    corpus = {}
    corpus_dir = DATA_DIR / "corpus_checkpoints"
    if corpus_dir.exists():
        for shard in sorted(f for f in os.listdir(corpus_dir) if f.endswith('.jsonl')):
            with open(corpus_dir / shard) as f:
                for line in f:
                    paper = json.loads(line)
                    doi = (paper.get('externalIds') or {}).get('DOI', '')
                    if doi and doi.lower() not in corpus:
                        corpus[doi.lower()] = paper

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()

    source_batch = []
    claim_batch = []
    fts_batch = []
    node_counts = defaultdict(lambda: defaultdict(int))
    node_claims = defaultdict(lambda: defaultdict(list))

    total_claims = 0
    total_classified = 0
    total_sources = 0
    sources_seen = set()

    # Check existing sources to avoid duplicates
    existing_sources = set()
    try:
        for row in c.execute("SELECT doi FROM sources"):
            existing_sources.add(row[0].lower())
    except Exception:
        pass

    for ri, result in enumerate(results):
        doi = result.get('doi', '')
        if not doi:
            continue

        if doi.lower() not in sources_seen and doi.lower() not in existing_sources:
            sources_seen.add(doi.lower())
            cp = corpus.get(doi.lower(), {})
            oa_title = titles.get(doi.lower(), '')
            paper_title = cp.get('title', '') or oa_title
            authors = [a.get('name', '') for a in (cp.get('authors') or [])[:20]]
            year = cp.get('year') or 0
            venue = cp.get('venue', '')
            abstract = cp.get('abstract', '')
            citation_count = cp.get('citationCount', 0) or 0
            oa_url = (cp.get('openAccessPdf') or {}).get('url', '')

            source_data = {
                'doi': doi, 'title': paper_title, 'authors': authors,
                'year': year, 'venue': venue, 'abstract': abstract,
                'citation_count': citation_count, 'open_access_url': oa_url,
            }
            source_batch.append((
                doi, paper_title, json.dumps(authors),
                year, venue, abstract, citation_count, oa_url,
                json.dumps(source_data),
            ))
            total_sources += 1

        paper_knowledge = result.get('data', {}).get('paper_knowledge', {})
        subfield = paper_knowledge.get('subfield', '')

        for raw_claim in result.get('data', {}).get('claims', []):
            claim_type = raw_claim.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            view_paths = {}

            llm_paths = classifications.get(claim_id, {})
            for view_id in ALL_CONTENT_VIEWS:
                p = llm_paths.get(view_id)
                if not p or p == ['not_applicable']:
                    continue
                if p and isinstance(p[0], list):
                    p = p[0]
                p = [str(s) for s in p if isinstance(s, (str, int, float))]
                if p and p != ['not_applicable']:
                    view_paths[view_id] = p

            if llm_paths:
                total_classified += 1

            ct_l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
            ct_path = [ct_l1]
            if subfield:
                ct_path.append(subfield.lower().replace(' ', '_'))
            view_paths['by_claim_type'] = ct_path

            for view_id, path_segs in view_paths.items():
                for depth in range(len(path_segs)):
                    partial = '/'.join(path_segs[:depth + 1])
                    node_counts[view_id][partial] += 1
                full_path = '/'.join(path_segs)
                node_claims[view_id][full_path].append(claim_id)

            parts = [
                raw_claim.get('claim_type', ''),
                raw_claim.get('verbatim_quote', ''),
                raw_claim.get('subject', ''),
                raw_claim.get('property_name', ''),
                raw_claim.get('reaction_type', ''),
                raw_claim.get('technique_name', ''),
                raw_claim.get('process_described', ''),
                raw_claim.get('hypothesis_text', ''),
                raw_claim.get('finding_text', ''),
                raw_claim.get('limitation_text', ''),
                raw_claim.get('direction_text', ''),
                raw_claim.get('comparison_result', ''),
                raw_claim.get('what_it_achieves', ''),
                raw_claim.get('key_innovation', ''),
            ]
            for key in ['reactants', 'products']:
                for item in raw_claim.get(key, []):
                    if isinstance(item, dict):
                        parts.append(item.get('name', ''))
            searchable = ' '.join(p for p in parts if p)

            source_title = titles.get(doi.lower(), '') or corpus.get(doi.lower(), {}).get('title', '')

            claim_data = dict(raw_claim)
            claim_data.update({
                'claim_id': claim_id,
                'source_doi': doi,
                'source_paper_title': source_title,
                'extraction_model': 'gpt-5.4',
                'extraction_version': 'deep_v1',
                'view_paths': view_paths,
            })

            claim_batch.append((
                claim_id, claim_type, doi, source_title,
                raw_claim.get('confidence', 'high'),
                raw_claim.get('location_in_paper', ''),
                raw_claim.get('verbatim_quote', ''),
                'gpt-5.4', 'deep_v1',
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

        if (ri + 1) % 100 == 0:
            _flush_batches(c, conn, source_batch, claim_batch, fts_batch)
            source_batch.clear()
            claim_batch.clear()
            fts_batch.clear()
            print(f"  Processed {ri+1}/{len(results)} papers, {total_claims:,} claims", flush=True)

    _flush_batches(c, conn, source_batch, claim_batch, fts_batch)
    print(f"\n  Sources inserted: {total_sources:,}")
    print(f"  Claims inserted: {total_claims:,}")
    print(f"  LLM-classified: {total_classified:,}")

    # Merge tree nodes with existing ones
    print("Merging tree nodes...", flush=True)
    total_nodes = 0
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

            existing = c.execute(
                "SELECT claim_count, children, claim_ids FROM tree_nodes WHERE view_id=? AND path=?",
                (view_id, path_str),
            ).fetchone()

            if existing:
                old_count = existing[0] or 0
                try:
                    old_children = set(json.loads(existing[1])) if existing[1] else set()
                except (json.JSONDecodeError, TypeError):
                    old_children = set()
                try:
                    old_claim_ids = json.loads(existing[2]) if existing[2] else []
                except (json.JSONDecodeError, TypeError):
                    old_claim_ids = []

                merged_count = old_count + count
                merged_children = sorted(old_children | set(children))
                merged_claim_ids = (old_claim_ids + claim_ids)[:2000]
            else:
                merged_count = count
                merged_children = children
                merged_claim_ids = claim_ids[:2000]

            node_data = {
                'view_id': view_id, 'path': path_str, 'name': name,
                'level': level, 'claim_count': merged_count,
                'children': merged_children, 'claim_ids': merged_claim_ids,
            }
            c.execute(
                "INSERT OR REPLACE INTO tree_nodes "
                "(view_id, path, name, level, claim_count, children, claim_ids, data) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (view_id, path_str, name, level, merged_count,
                 json.dumps(merged_children), json.dumps(merged_claim_ids),
                 json.dumps(node_data)),
            )
            total_nodes += 1

        conn.commit()
        print(f"    {view_id}: {len(paths):,} nodes merged", flush=True)

    # Ensure root nodes exist for views that had new data
    for view_id in node_counts:
        existing_root = c.execute(
            "SELECT 1 FROM tree_nodes WHERE view_id=? AND path=''", (view_id,)
        ).fetchone()
        if not existing_root:
            all_l1 = set()
            for p in node_counts[view_id]:
                all_l1.add(p.split('/')[0])
            root_data = {
                'view_id': view_id, 'path': '', 'name': view_id,
                'level': 0, 'claim_count': sum(node_counts[view_id].values()),
                'children': sorted(all_l1),
            }
            c.execute(
                "INSERT OR REPLACE INTO tree_nodes "
                "(view_id, path, name, level, claim_count, children, claim_ids, data) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (view_id, '', view_id, 0, root_data['claim_count'],
                 json.dumps(root_data['children']), '[]', json.dumps(root_data)),
            )

    conn.commit()
    conn.close()

    print(f"\nBuild complete!")
    print(f"  {total_sources:,} sources")
    print(f"  {total_claims:,} claims (deep_v1)")
    print(f"  {total_classified:,} LLM-classified")
    print(f"  {total_nodes:,} tree nodes merged")


def _flush_batches(cursor, conn, source_batch, claim_batch, fts_batch):
    if source_batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO sources "
            "(doi, title, authors, year, venue, abstract, citation_count, open_access_url, data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            source_batch,
        )
    if claim_batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO claims "
            "(claim_id, claim_type, source_doi, source_paper_title, confidence, "
            "location_in_paper, verbatim_quote, extraction_model, extraction_version, "
            "extracted_at, view_paths, data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            claim_batch,
        )
    if fts_batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO claims_fts "
            "(claim_id, claim_type, source_paper_title, verbatim_quote, searchable_text) "
            "VALUES (?,?,?,?,?)",
            fts_batch,
        )
    conn.commit()


def cmd_status(args):
    print("=== Classify & Integrate Pipeline Status ===\n")

    results = list(RESULTS_DIR.glob("*.json"))
    print(f"Deep results on disk: {len(results)}")

    classified_file = PIPELINE_DIR / "classifications.json"
    if classified_file.exists():
        cls = json.loads(classified_file.read_text())
        with_l3 = sum(1 for v in cls.values()
                      if any(len(p) >= 3 for p in v.values() if isinstance(p, list)))
        print(f"Classifications: {len(cls):,} ({with_l3:,} with L3)")
    else:
        print("Classifications: none yet")

    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT extraction_version, COUNT(*) FROM claims GROUP BY extraction_version")
        print("\nClaims in DB:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]:,}")
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Classify and integrate Tier A")
    parser.add_argument("command", choices=[
        "prepare", "submit", "poll", "collect",
        "l3-prepare", "l3-submit", "l3-poll", "l3-collect",
        "build", "status",
    ])
    args = parser.parse_args()

    cmd_map = {
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "poll": cmd_poll,
        "collect": cmd_collect,
        "l3-prepare": cmd_l3_prepare,
        "l3-submit": cmd_l3_submit,
        "l3-poll": cmd_l3_poll,
        "l3-collect": cmd_l3_collect,
        "build": cmd_build,
        "status": cmd_status,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
