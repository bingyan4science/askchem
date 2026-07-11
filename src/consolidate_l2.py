"""
Consolidate L2 categories using LLM.

For each (view, L1) pair with >5 L2 slugs, ask the LLM to merge them
into 3-8 canonical categories. Then update paper_classifications.json
with the merged L2 slugs.

Usage:
    python src/consolidate_l2.py prepare   # Build batch requests
    python src/consolidate_l2.py submit    # Submit to OpenAI Batch API
    python src/consolidate_l2.py poll      # Check status
    python src/consolidate_l2.py collect   # Download and apply merges
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

from openai import OpenAI

PIPELINE_DIR = Path(__file__).parent.parent / "data" / "paper_classify"
CONSOLIDATE_DIR = PIPELINE_DIR / "consolidate_l2"

ALL_VIEWS = ['by_reaction_type', 'by_substance_class', 'by_application',
             'by_technique', 'by_mechanism']

CONSOLIDATE_PROMPT = """You are organizing a chemistry knowledge index.

View: {view_id}
L1 category: {l1}

Below are the current L2 subcategories under this L1, with paper counts.
Many are near-duplicates or overlapping. Merge them into 3-8 clean, non-overlapping canonical categories.

Current L2 categories:
{l2_list}

Rules:
1. Output 3-8 canonical L2 category names (lowercase_with_underscores).
2. Each canonical category should be a meaningful, distinct subcategory of "{l1}".
3. Map EVERY input L2 to exactly one canonical L2.
4. Prefer the most common/natural name for each canonical category.
5. Do NOT create overly broad catch-all categories like "general" or "other" — only use "other" if truly miscellaneous items exist.
6. For the "not_applicable" L1, merge all variants into just "not_applicable".

Return JSON:
{{
  "canonical": ["cat1", "cat2", ...],
  "mapping": {{
    "old_l2_slug": "canonical_l2_slug",
    ...
  }}
}}"""


def load_l2_data():
    """Load L2 distributions from paper classifications."""
    classified_file = PIPELINE_DIR / "paper_classifications.json"
    paper_paths = json.loads(classified_file.read_text())

    l2_counts = defaultdict(lambda: defaultdict(lambda: Counter()))
    for doi, paths in paper_paths.items():
        for vid in ALL_VIEWS:
            p = paths.get(vid, [])
            if len(p) >= 2:
                l1 = p[0].strip().lower().replace('-', '_')
                l2 = p[1].strip().lower().replace('-', '_')
                l2_counts[vid][l1][l2] += 1

    return l2_counts


def cmd_prepare(args):
    CONSOLIDATE_DIR.mkdir(parents=True, exist_ok=True)
    l2_counts = load_l2_data()

    fname = CONSOLIDATE_DIR / "consolidate_batch.jsonl"
    count = 0
    with open(fname, 'w') as f:
        for vid in ALL_VIEWS:
            for l1, slugs in sorted(l2_counts[vid].items()):
                if len(slugs) <= 5:
                    continue

                sorted_slugs = sorted(slugs.items(), key=lambda x: -x[1])
                l2_list = '\n'.join(f"  {slug} ({cnt} papers)" for slug, cnt in sorted_slugs)

                prompt = CONSOLIDATE_PROMPT.format(
                    view_id=vid, l1=l1, l2_list=l2_list
                )

                request = {
                    "custom_id": f"{vid}___{l1}",
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
                count += 1

    print(f"Prepared {count} consolidation requests → {fname.name}")


def cmd_submit(args):
    client = OpenAI()
    fpath = CONSOLIDATE_DIR / "consolidate_batch.jsonl"
    tracker_file = CONSOLIDATE_DIR / "tracker.json"

    tracker = {}
    if tracker_file.exists():
        tracker = json.loads(tracker_file.read_text())
    if 'batch_id' in tracker and tracker.get('status') not in ('failed', 'expired', 'cancelled'):
        print(f"Already submitted: {tracker['batch_id']} ({tracker.get('status')})")
        return

    print("Uploading...", flush=True)
    uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    tracker = {"batch_id": batch.id, "file_id": uploaded.id, "status": batch.status}
    json.dump(tracker, open(tracker_file, 'w'), indent=2)
    print(f"Batch {batch.id} ({batch.status})")


def cmd_poll(args):
    tracker_file = CONSOLIDATE_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No batch submitted.")
        return
    client = OpenAI()
    tracker = json.loads(tracker_file.read_text())
    b = client.batches.retrieve(tracker['batch_id'])
    tracker['status'] = b.status
    if b.output_file_id:
        tracker['output_file_id'] = b.output_file_id
    json.dump(tracker, open(tracker_file, 'w'), indent=2)
    rc = b.request_counts
    if rc:
        print(f"Status: {b.status} ({rc.completed}/{rc.total} done, {rc.failed} failed)")
    else:
        print(f"Status: {b.status}")


def cmd_collect(args):
    tracker_file = CONSOLIDATE_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No batch.")
        return
    client = OpenAI()
    tracker = json.loads(tracker_file.read_text())
    output_id = tracker.get('output_file_id')
    if not output_id:
        print("Batch not complete yet.")
        return

    raw_path = CONSOLIDATE_DIR / "raw_results.jsonl"
    if not raw_path.exists():
        print("Downloading...", flush=True)
        content = client.files.content(output_id)
        raw_path.write_bytes(content.read())

    # Parse results into a global merge map
    global_merge = {}  # {vid: {l1: {old_l2: new_l2}}}
    errors = 0

    for line in open(raw_path):
        try:
            result = json.loads(line)
            custom_id = result.get('custom_id', '')
            vid, l1 = custom_id.split('___', 1)

            resp = result.get('response', {})
            if resp.get('status_code') != 200:
                errors += 1
                continue

            text = resp['body']['choices'][0]['message']['content']
            data = json.loads(text)
            mapping = data.get('mapping', {})

            if vid not in global_merge:
                global_merge[vid] = {}
            global_merge[vid][l1] = {
                k.strip().lower().replace('-', '_'): v.strip().lower().replace('-', '_')
                for k, v in mapping.items()
            }
        except Exception as e:
            errors += 1
            print(f"Error: {e}")

    merge_file = CONSOLIDATE_DIR / "l2_merge_map.json"
    merge_file.write_text(json.dumps(global_merge, indent=2))
    print(f"L2 merge map: {sum(len(m) for ms in global_merge.values() for m in ms.values())} entries ({errors} errors)")

    # Apply to paper_classifications.json
    classified_file = PIPELINE_DIR / "paper_classifications.json"
    paper_paths = json.loads(classified_file.read_text())

    changes = 0
    for doi, paths in paper_paths.items():
        for vid in ALL_VIEWS:
            p = paths.get(vid, [])
            if len(p) >= 2:
                l1 = p[0].strip().lower().replace('-', '_')
                l2 = p[1].strip().lower().replace('-', '_')
                vid_merges = global_merge.get(vid, {})
                l1_merges = vid_merges.get(l1, {})
                if l2 in l1_merges and l1_merges[l2] != l2:
                    p[1] = l1_merges[l2]
                    changes += 1

    # Save updated classifications
    backup = PIPELINE_DIR / "paper_classifications_pre_l2_merge.json"
    import shutil
    shutil.copy2(classified_file, backup)
    classified_file.write_text(json.dumps(paper_paths, indent=2))
    print(f"Applied {changes} L2 merges to paper_classifications.json")
    print(f"Backup saved to {backup.name}")

    # Show resulting L2 distribution
    l2_counts = defaultdict(lambda: defaultdict(lambda: Counter()))
    for doi, paths in paper_paths.items():
        for vid in ALL_VIEWS:
            p = paths.get(vid, [])
            if len(p) >= 2:
                l1 = p[0].strip().lower().replace('-', '_')
                l2 = p[1].strip().lower().replace('-', '_')
                l2_counts[vid][l1][l2] += 1

    print(f"\nPost-merge L2 distribution:")
    for vid in ALL_VIEWS:
        total_l2 = sum(len(slugs) for slugs in l2_counts[vid].values())
        print(f"  {vid}: {total_l2} L2 categories")
        for l1, slugs in sorted(l2_counts[vid].items()):
            if len(slugs) > 1:
                sorted_slugs = sorted(slugs.items(), key=lambda x: -x[1])
                print(f"    {l1} ({len(slugs)} L2s):")
                for slug, cnt in sorted_slugs:
                    print(f"      {slug:50s} {cnt:3d} papers")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prepare"); sub.add_parser("submit")
    sub.add_parser("poll"); sub.add_parser("collect")
    args = parser.parse_args()
    cmd_map = {'prepare': cmd_prepare, 'submit': cmd_submit,
               'poll': cmd_poll, 'collect': cmd_collect}
    if args.command in cmd_map:
        cmd_map[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
