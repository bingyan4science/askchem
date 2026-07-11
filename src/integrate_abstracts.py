"""
Integrate abstract-only extractions into the AskChem index.

Adds the 494 papers from experiments/005_scale_extraction that don't have
deep extractions. Uses the same two-track classification as integrate_deep.py.

Pipeline:
    python src/integrate_abstracts.py prepare
    python src/integrate_abstracts.py submit
    python src/integrate_abstracts.py poll
    python src/integrate_abstracts.py collect
    python src/integrate_abstracts.py build
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
from askchem.models import Claim
from askchem.display import smart_title
from integrate_deep import (
    CONTENT_TYPES, EPISTEMIC_TYPES, CLAIM_TYPE_LABELS,
    CONTENT_PROMPT, EPISTEMIC_PROMPT,
    make_content_summary, make_epistemic_summary,
    load_corpus_metadata,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "experiments" / "005_scale_extraction" / "checkpoints"
DEEP_DIR = DATA_DIR / "deep_results"
PIPELINE_DIR = DATA_DIR / "classify_abstracts"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"


def load_deep_dois() -> set[str]:
    dois = set()
    for f in sorted(DEEP_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            doi = data.get('doi', '')
            if doi:
                dois.add(doi.lower())
        except Exception:
            pass
    return dois


def load_abstract_papers() -> list[dict]:
    """Load abstract-extracted papers that don't have deep extractions."""
    deep_dois = load_deep_dois()
    papers = []
    for f in sorted(CHECKPOINT_DIR.glob("*.json")):
        batch = json.load(open(f))
        for p in batch:
            doi = p.get('doi', '')
            if doi and doi.lower() not in deep_dois and p.get('claims'):
                papers.append(p)
    return papers


def cmd_prepare(args):
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    papers = load_abstract_papers()
    print(f"Abstract-only papers: {len(papers)}", flush=True)

    all_claims = []
    for p in papers:
        doi = p.get('doi', '')
        for raw in p.get('claims', []):
            ct = raw.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, ct, content_hash)
            all_claims.append({
                'claim_id': claim_id,
                'doi': doi,
                'claim_type': ct,
                'raw_claim': raw,
            })

    print(f"Total claims: {len(all_claims):,}", flush=True)

    classified_file = PIPELINE_DIR / "classifications.json"
    already = set()
    if classified_file.exists():
        already = set(json.loads(classified_file.read_text()).keys())

    to_classify = [c for c in all_claims if c['claim_id'] not in already]
    print(f"Already classified: {len(already)}, to classify: {len(to_classify)}", flush=True)

    if not to_classify:
        print("All classified!")
        return

    fname = PIPELINE_DIR / "classify_001.jsonl"
    with open(fname, 'w') as f:
        for c in to_classify:
            ct = c['claim_type']
            if ct in CONTENT_TYPES:
                summary = make_content_summary(c['raw_claim'])
                prompt = CONTENT_PROMPT.format(claim_json=json.dumps(summary))
            elif ct in EPISTEMIC_TYPES:
                label = CLAIM_TYPE_LABELS.get(ct, ct).replace('_', ' ')
                summary = make_epistemic_summary(c['raw_claim'])
                prompt = EPISTEMIC_PROMPT.format(claim_type_label=label, claim_json=json.dumps(summary))
            else:
                summary = make_content_summary(c['raw_claim'])
                prompt = CONTENT_PROMPT.format(claim_json=json.dumps(summary))

            request = {
                "custom_id": c['claim_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(request) + "\n")

    print(f"  {fname.name}: {fname.stat().st_size / 1e6:.1f} MB, {len(to_classify):,} requests")


def cmd_submit(args):
    client = OpenAI()
    batch_files = sorted(PIPELINE_DIR.glob("classify_*.jsonl"))
    if not batch_files:
        print("No batch files.")
        return

    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    tracker = {}
    if tracker_file.exists():
        tracker = json.loads(tracker_file.read_text())

    for fpath in batch_files:
        if fpath.name in tracker and tracker[fpath.name].get('status') not in ('failed', 'expired', 'cancelled', 'cancelling'):
            print(f"  {fpath.name}: already submitted ({tracker[fpath.name].get('status')})")
            continue
        print(f"  Uploading {fpath.name}...", flush=True)
        uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        tracker[fpath.name] = {"batch_id": batch.id, "file_id": uploaded.id, "status": batch.status}
        json.dump(tracker, open(tracker_file, 'w'), indent=2)
        print(f"  Batch {batch.id} ({batch.status})")


def cmd_poll(args):
    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No batches.")
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
            completed += rc.completed; failed += rc.failed; total += rc.total
        if b.status != 'completed':
            print(f"  {fname}: {b.status} ({rc.completed}/{rc.total})" if rc else f"  {fname}: {b.status}")
    json.dump(tracker, open(tracker_file, 'w'), indent=2)
    st = Counter(v['status'] for v in tracker.values())
    print(f"Statuses: {dict(st)}")
    print(f"Overall: {completed}/{total} done, {failed} failed")


def cmd_collect(args):
    tracker_file = PIPELINE_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No batches.")
        return
    client = OpenAI()
    tracker = json.loads(tracker_file.read_text())
    classifications = {}
    classified_file = PIPELINE_DIR / "classifications.json"
    if classified_file.exists():
        classifications = json.loads(classified_file.read_text())

    errors = 0; empty = 0; new_count = 0
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
                    errors += 1; continue
                text = body.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not text or not text.strip():
                    empty += 1; continue
                paths = json.loads(text)
                if cid not in classifications:
                    classifications[cid] = paths
                    new_count += 1
            except Exception:
                errors += 1
    classified_file.write_text(json.dumps(classifications))
    print(f"Classifications: {len(classifications):,} ({new_count} new, {errors} errors, {empty} empty)")


def cmd_build(args):
    """Add abstract-only papers to the existing SQLite index."""
    import sqlite3

    classified_file = PIPELINE_DIR / "classifications.json"
    if not classified_file.exists():
        print("No classifications. Run collect first.")
        return

    classifications = json.loads(classified_file.read_text())
    print(f"Loaded {len(classifications):,} classifications", flush=True)

    corpus = load_corpus_metadata()
    papers = load_abstract_papers()
    print(f"Abstract-only papers: {len(papers)}", flush=True)

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")

    ALL_CONTENT_VIEWS = ['by_reaction_type', 'by_substance_class', 'by_application',
                         'by_technique', 'by_mechanism']

    source_batch = []
    claim_batch = []
    fts_batch = []
    new_nodes = defaultdict(lambda: defaultdict(int))
    new_node_claims = defaultdict(lambda: defaultdict(list))
    total_claims = 0
    total_sources = 0

    for pi, p in enumerate(papers):
        doi = p.get('doi', '')
        if not doi:
            continue

        # Source
        cp = corpus.get(doi.lower(), {})
        authors_raw = p.get('authors', [])
        if isinstance(authors_raw, list) and authors_raw and isinstance(authors_raw[0], str):
            authors = authors_raw
        else:
            authors = [a.get('name', '') for a in (cp.get('authors') or [])[:20]]

        source_data = {
            'doi': doi,
            'title': p.get('title', cp.get('title', '')),
            'authors': authors,
            'year': p.get('year') or cp.get('year') or 0,
            'venue': p.get('venue', cp.get('venue', '')),
            'abstract': cp.get('abstract', ''),
            'citation_count': p.get('citation_count') or cp.get('citationCount', 0) or 0,
            'open_access_url': (cp.get('openAccessPdf') or {}).get('url', ''),
        }
        source_batch.append((
            doi, source_data['title'], json.dumps(authors),
            source_data['year'], source_data['venue'], source_data['abstract'],
            source_data['citation_count'], source_data['open_access_url'],
            json.dumps(source_data),
        ))
        total_sources += 1

        for raw in p.get('claims', []):
            ct = raw.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, ct, content_hash)

            # View paths
            view_paths = {}
            llm_paths = classifications.get(claim_id, {})
            for vid in ALL_CONTENT_VIEWS:
                path = llm_paths.get(vid)
                if not path or path == ['not_applicable']:
                    continue
                if path and isinstance(path[0], list):
                    path = path[0]
                path = [str(s) for s in path if isinstance(s, (str, int, float))]
                if path and path != ['not_applicable']:
                    view_paths[vid] = path

            ct_l1 = CLAIM_TYPE_LABELS.get(ct, ct)
            view_paths['by_claim_type'] = [ct_l1]

            for vid, segs in view_paths.items():
                full_path = '/'.join(segs)
                for depth in range(len(segs)):
                    partial = '/'.join(segs[:depth + 1])
                    new_nodes[vid][partial] += 1
                new_node_claims[vid][full_path].append(claim_id)

            source_title = source_data['title']
            parts = [ct, raw.get('verbatim_quote', ''), raw.get('subject', ''),
                     raw.get('property_name', ''), raw.get('reaction_type', ''),
                     raw.get('technique_name', ''), raw.get('process_described', '')]
            searchable = ' '.join(x for x in parts if x)

            claim_data = dict(raw)
            claim_data.update({
                'claim_id': claim_id, 'source_doi': doi,
                'source_paper_title': source_title,
                'extraction_model': 'gpt-5-mini', 'extraction_version': 'abstract_v1',
                'view_paths': view_paths,
            })

            claim_batch.append((
                claim_id, ct, doi, source_title,
                raw.get('confidence', 'high'), raw.get('location_in_paper', ''),
                raw.get('verbatim_quote', ''),
                'gpt-5-mini', 'abstract_v1',
                datetime.now().isoformat(),
                json.dumps(view_paths), json.dumps(claim_data),
            ))
            fts_batch.append((claim_id, ct, source_title, raw.get('verbatim_quote', ''), searchable))
            total_claims += 1

    # Insert
    c.executemany(
        "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
        source_batch)
    c.executemany(
        "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        claim_batch)
    c.executemany(
        "INSERT OR REPLACE INTO claims_fts (claim_id,claim_type,source_paper_title,verbatim_quote,searchable_text) VALUES (?,?,?,?,?)",
        fts_batch)
    conn.commit()
    print(f"  Added {total_sources} sources, {total_claims:,} claims", flush=True)

    # Update/insert tree nodes
    print("Updating tree nodes...", flush=True)
    for vid, paths in new_nodes.items():
        children_map = defaultdict(set)
        for path_str in paths:
            segs = path_str.split('/')
            if len(segs) > 1:
                parent = '/'.join(segs[:-1])
                children_map[parent].add(segs[-1])

        for path_str, count in paths.items():
            segs = path_str.split('/')
            level = len(segs)
            name = smart_title(segs[-1])
            new_children = sorted(children_map.get(path_str, set()))
            new_claim_ids = new_node_claims[vid].get(path_str, [])

            existing = c.execute(
                "SELECT claim_count, children, claim_ids FROM tree_nodes WHERE view_id=? AND path=?",
                (vid, path_str)
            ).fetchone()

            if existing:
                old_count = existing[0]
                old_children = set(json.loads(existing[1])) if existing[1] else set()
                old_claim_ids = json.loads(existing[2]) if existing[2] else []
                merged_children = sorted(old_children | set(new_children))
                merged_claims = old_claim_ids + new_claim_ids
                node_data = {
                    'view_id': vid, 'path': path_str, 'name': name,
                    'level': level, 'claim_count': old_count + count,
                    'children': merged_children, 'claim_ids': merged_claims[:2000],
                }
                c.execute(
                    "UPDATE tree_nodes SET claim_count=?, children=?, claim_ids=?, data=? WHERE view_id=? AND path=?",
                    (old_count + count, json.dumps(merged_children), json.dumps(merged_claims[:2000]),
                     json.dumps(node_data), vid, path_str)
                )
            else:
                node_data = {
                    'view_id': vid, 'path': path_str, 'name': name,
                    'level': level, 'claim_count': count,
                    'children': new_children, 'claim_ids': new_claim_ids[:2000],
                }
                c.execute(
                    "INSERT INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
                    (vid, path_str, name, level, count, json.dumps(new_children),
                     json.dumps(new_claim_ids[:2000]), json.dumps(node_data))
                )
    conn.commit()

    # Update root nodes
    for vid in set(new_nodes.keys()):
        l1 = c.execute(
            "SELECT path, claim_count FROM tree_nodes WHERE view_id=? AND level=1 ORDER BY claim_count DESC",
            (vid,)
        ).fetchall()
        children = [r[0] for r in l1]
        total = sum(r[1] for r in l1)
        root_data = {'view_id': vid, 'path': '', 'name': vid, 'level': 0,
                     'claim_count': total, 'children': children, 'claim_ids': []}
        c.execute(
            "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            (vid, '', vid, 0, total, json.dumps(children), json.dumps([]), json.dumps(root_data))
        )
    conn.commit()

    # Update metadata
    total_claims_db = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    total_sources_db = c.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    total_nodes_db = c.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()[0]
    total_views_db = c.execute("SELECT COUNT(*) FROM views").fetchone()[0]
    for k, v in [
        ('total_claims', str(total_claims_db)),
        ('total_sources', str(total_sources_db)),
        ('total_nodes', str(total_nodes_db)),
        ('total_views', str(total_views_db)),
    ]:
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)", (k, v))
    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"ABSTRACT INTEGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Added: {total_sources} sources, {total_claims:,} claims")
    print(f"Index totals: {total_sources_db:,} sources, {total_claims_db:,} claims, {total_nodes_db:,} nodes")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prepare"); sub.add_parser("submit"); sub.add_parser("poll")
    sub.add_parser("collect"); sub.add_parser("build")
    args = parser.parse_args()
    cmd_map = {'prepare': cmd_prepare, 'submit': cmd_submit, 'poll': cmd_poll,
               'collect': cmd_collect, 'build': cmd_build}
    if args.command in cmd_map:
        cmd_map[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
