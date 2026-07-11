"""
Reclassify all claims using the new constrained L1+L2 taxonomy via OpenAI Batch API.

Usage:
    python src/reclassify_batch.py --prepare     # Build JSONL files and submit batches
    python src/reclassify_batch.py --poll         # Check batch status
    python src/reclassify_batch.py --collect      # Download results and update DB
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from askchem.taxonomy import (
    CANONICAL_L1, CANONICAL_L2, ALL_CONTENT_VIEWS,
    CLASSIFICATION_SYSTEM_PROMPT, build_classification_prompt,
    build_classification_messages, normalize_path,
)

BATCH_DIR = Path(__file__).parent.parent / "data" / "reclassify_batches"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"
MAX_BATCH_FILE_BYTES = 190 * 1024 * 1024  # 190 MB (OpenAI limit is 200MB)
CLASSIFICATION_MODEL = "gpt-5-mini"


def _build_batch_request(claim_id: str, claim_type: str, quote: str,
                         title: str, max_tokens: int = 8192) -> dict:
    messages = build_classification_messages(claim_type, quote[:300], title)
    return {
        "custom_id": f"rcls_{claim_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": CLASSIFICATION_MODEL,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
    }


def prepare():
    """Read all claims from DB, build JSONL batch files, and submit to OpenAI."""
    from openai import OpenAI
    client = OpenAI()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as c FROM claims").fetchone()["c"]
    print(f"Total claims to reclassify: {total:,}")

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    batch_files = []
    claim_count = 0

    cursor = conn.execute(
        "SELECT claim_id, claim_type, source_paper_title, verbatim_quote FROM claims"
    )

    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break

        for row in rows:
            claim_id = row["claim_id"]
            claim_type = row["claim_type"] or "property"
            title = row["source_paper_title"] or ""
            quote = row["verbatim_quote"] or ""

            if not quote:
                continue

            request = _build_batch_request(claim_id, claim_type, quote, title)
            line = json.dumps(request) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES:
                if current_file:
                    current_file.close()
                    print(f"  {batch_files[-1].name}: {items_in_batch:,} claims, "
                          f"{current_size / 1e6:.1f} MB", flush=True)
                batch_idx += 1
                fname = BATCH_DIR / f"reclassify_{batch_idx:03d}.jsonl"
                batch_files.append(fname)
                current_file = open(fname, "w")
                current_size = 0
                items_in_batch = 0

            current_file.write(line)
            current_size += line_bytes
            items_in_batch += 1
            claim_count += 1

            if claim_count % 100000 == 0:
                print(f"  Prepared {claim_count:,} / {total:,}...", flush=True)

    if current_file:
        current_file.close()
        if items_in_batch > 0:
            print(f"  {batch_files[-1].name}: {items_in_batch:,} claims, "
                  f"{current_size / 1e6:.1f} MB", flush=True)

    conn.close()
    print(f"\nPrepared {claim_count:,} claims in {len(batch_files)} JSONL files.")

    # Submit batches
    print(f"\nSubmitting {len(batch_files)} batches to OpenAI...", flush=True)
    tracker = {}
    for fpath in batch_files:
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
        print(f"    Batch {batch.id} ({batch.status})", flush=True)
        time.sleep(2)

    with open(BATCH_DIR / "tracker.json", "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"\n{len(tracker)} batches submitted. Poll with: python src/reclassify_batch.py --poll")


def poll():
    """Check status of all submitted batches."""
    from openai import OpenAI
    client = OpenAI()

    tracker_file = BATCH_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No tracker found. Run --prepare first.")
        return

    with open(tracker_file) as f:
        tracker = json.load(f)

    all_done = True
    for fname, info in sorted(tracker.items()):
        batch_id = info["batch_id"]
        batch = client.batches.retrieve(batch_id)
        info["status"] = batch.status
        if hasattr(batch, "output_file_id") and batch.output_file_id:
            info["output_file_id"] = batch.output_file_id
        if hasattr(batch, "error_file_id") and batch.error_file_id:
            info["error_file_id"] = batch.error_file_id
        if hasattr(batch, "request_counts") and batch.request_counts:
            rc = batch.request_counts
            info["completed"] = rc.completed
            info["failed"] = rc.failed
            info["total"] = rc.total

        status_str = f"{batch.status}"
        if "completed" in info:
            status_str += f" ({info['completed']}/{info['total']} done, {info['failed']} failed)"

        print(f"  {fname}: {status_str}")

        if batch.status not in ("completed", "failed", "cancelled", "expired"):
            all_done = False

    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)

    if all_done:
        print("\nAll batches finished! Run: python src/reclassify_batch.py --collect")
    else:
        print("\nSome batches still running. Poll again later.")


def collect():
    """Download results and update view_paths in the database."""
    from openai import OpenAI
    client = OpenAI()

    tracker_file = BATCH_DIR / "tracker.json"
    if not tracker_file.exists():
        print("No tracker found. Run --prepare first.")
        return

    with open(tracker_file) as f:
        tracker = json.load(f)

    raw_dir = BATCH_DIR / "raw_results"
    raw_dir.mkdir(exist_ok=True)

    # Download all result files
    results = {}
    errors = 0
    empty_responses = 0

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            print(f"  {fname}: no output (status={info.get('status')})")
            continue

        raw_path = raw_dir / fname
        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            with open(raw_path, "wb") as f:
                f.write(content.read())

        with open(raw_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    claim_id = custom_id.replace("rcls_", "")
                    response = result.get("response", {})
                    body = response.get("body", {})

                    if response.get("status_code") != 200:
                        errors += 1
                        continue

                    choices = body.get("choices", [])
                    if not choices:
                        errors += 1
                        continue

                    text = choices[0].get("message", {}).get("content", "")
                    if not text:
                        empty_responses += 1
                        continue

                    parsed = json.loads(text)

                    # Normalize all paths
                    normalized = {}
                    for view_id in ALL_CONTENT_VIEWS:
                        raw_path_val = parsed.get(view_id)
                        if raw_path_val and raw_path_val != ["not_applicable"]:
                            normed = normalize_path(view_id, raw_path_val)
                            if normed:
                                normalized[view_id] = normed

                    if normalized:
                        results[claim_id] = normalized

                except Exception:
                    errors += 1

    print(f"\nCollected {len(results):,} classifications, {errors} errors, "
          f"{empty_responses} empty responses")

    # Update database
    print(f"\nUpdating database...", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    updated = 0
    batch_size = 5000
    items = list(results.items())

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        for claim_id, view_paths in batch:
            vp_json = json.dumps(view_paths)
            conn.execute(
                "UPDATE claims SET view_paths = ? WHERE claim_id = ?",
                [vp_json, claim_id],
            )
            updated += 1

        conn.commit()
        if (i + batch_size) % 50000 < batch_size:
            print(f"  Updated {min(i + batch_size, len(items)):,} / {len(items):,}...",
                  flush=True)

    conn.commit()
    conn.close()

    print(f"\nDone! Updated {updated:,} claims.")
    print(f"Next step: rebuild the tree with: python src/reclassify_batch.py --rebuild-tree")


def rebuild_tree():
    """Rebuild tree_nodes from updated view_paths."""
    print("Rebuilding tree_nodes from reclassified view_paths...", flush=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row

    # Clear existing tree nodes
    conn.execute("DELETE FROM tree_nodes")
    conn.commit()
    print("  Cleared existing tree_nodes.", flush=True)

    # Load all views
    views = conn.execute("SELECT view_id, data FROM views").fetchall()
    view_ids = [r["view_id"] for r in views]
    print(f"  Views: {view_ids}", flush=True)

    # Build tree structure in memory
    trees = {}  # view_id -> {path -> {claim_ids: [], children: set(), data: {}}}
    for vid in view_ids:
        trees[vid] = {}

    total = conn.execute("SELECT COUNT(*) as c FROM claims").fetchone()["c"]
    processed = 0
    cursor = conn.execute("SELECT claim_id, view_paths, data FROM claims")

    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break

        for row in rows:
            claim_id = row["claim_id"]
            try:
                vp = json.loads(row["view_paths"]) if row["view_paths"] else {}
            except (json.JSONDecodeError, TypeError):
                continue

            claim_data = json.loads(row["data"])

            for vid in view_ids:
                path_segments = vp.get(vid)
                if not path_segments or not isinstance(path_segments, list):
                    continue
                path_segments = [s for s in path_segments
                                 if s not in ("not_applicable", "none")]
                if not path_segments:
                    continue

                # Ensure root exists
                if "" not in trees[vid]:
                    trees[vid][""] = {
                        "claim_ids": [], "children": set(),
                        "data": {"view_id": vid, "path": "", "name": vid,
                                 "level": 0, "claim_count": 0, "children": []},
                    }

                # Build each level of the path
                for depth in range(len(path_segments)):
                    path_key = "/".join(path_segments[: depth + 1])
                    parent_key = "/".join(path_segments[:depth]) if depth > 0 else ""
                    segment = path_segments[depth]

                    if path_key not in trees[vid]:
                        trees[vid][path_key] = {
                            "claim_ids": [], "children": set(),
                            "data": {
                                "view_id": vid, "path": path_key,
                                "name": segment, "level": depth + 1,
                                "claim_count": 0, "children": [],
                            },
                        }

                    # Add child reference to parent
                    trees[vid][parent_key]["children"].add(segment)

                # Assign claim to leaf node
                leaf_path = "/".join(path_segments)
                trees[vid][leaf_path]["claim_ids"].append(claim_id)

        processed += len(rows)
        if processed % 100000 == 0:
            print(f"  Processed {processed:,} / {total:,} claims...", flush=True)

    print(f"  Processed all {processed:,} claims.", flush=True)

    # Compute descendant claim counts bottom-up (O(n) per view)
    total_nodes = 0
    for vid in view_ids:
        tree = trees[vid]

        # Sort paths by depth (deepest first) for bottom-up aggregation
        desc_counts = {}
        for path_key, node_data in tree.items():
            desc_counts[path_key] = len(node_data["claim_ids"])

        for path_key in sorted(tree.keys(), key=lambda p: p.count("/"), reverse=True):
            if not path_key:
                continue
            parts = path_key.split("/")
            parent_key = "/".join(parts[:-1]) if len(parts) > 1 else ""
            if parent_key in desc_counts:
                desc_counts[parent_key] += desc_counts[path_key]

        for path_key, node_data in tree.items():
            data = node_data["data"]
            data["claim_count"] = desc_counts.get(path_key, 0)
            data["children"] = sorted(node_data["children"])

            conn.execute(
                "INSERT OR REPLACE INTO tree_nodes (view_id, path, data, children, claim_ids) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    vid,
                    path_key,
                    json.dumps(data),
                    json.dumps(sorted(node_data["children"])),
                    json.dumps(node_data["claim_ids"][:1000]),
                ],
            )
            total_nodes += 1

        conn.commit()
        node_count = len(tree)
        print(f"  {vid}: {node_count:,} nodes", flush=True)

    conn.close()
    print(f"\nDone! Wrote {total_nodes:,} tree nodes.")


def retry():
    """Re-submit failed claims (empty responses) with higher max_completion_tokens."""
    from openai import OpenAI
    client = OpenAI()

    retry_ids_file = BATCH_DIR / "retry_ids.json"
    if not retry_ids_file.exists():
        print("No retry_ids.json found. Run collect first to identify failures.")
        return

    with open(retry_ids_file) as f:
        retry_ids = set(json.load(f))

    print(f"Claims to retry: {len(retry_ids):,}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    retry_dir = BATCH_DIR / "retry"
    retry_dir.mkdir(exist_ok=True)

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    batch_files = []
    claim_count = 0

    placeholders_batch_size = 900
    retry_list = list(retry_ids)

    for chunk_start in range(0, len(retry_list), placeholders_batch_size):
        chunk = retry_list[chunk_start:chunk_start + placeholders_batch_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT claim_id, claim_type, source_paper_title, verbatim_quote "
            f"FROM claims WHERE claim_id IN ({placeholders})",
            chunk,
        ).fetchall()

        for row in rows:
            claim_id = row["claim_id"]
            claim_type = row["claim_type"] or "property"
            title = row["source_paper_title"] or ""
            quote = row["verbatim_quote"] or ""
            if not quote:
                continue

            request = _build_batch_request(claim_id, claim_type, quote, title,
                                           max_tokens=8192)
            line = json.dumps(request) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES:
                if current_file:
                    current_file.close()
                    print(f"  {batch_files[-1].name}: {items_in_batch:,} claims, "
                          f"{current_size / 1e6:.1f} MB", flush=True)
                batch_idx += 1
                fname = retry_dir / f"retry_{batch_idx:03d}.jsonl"
                batch_files.append(fname)
                current_file = open(fname, "w")
                current_size = 0
                items_in_batch = 0

            current_file.write(line)
            current_size += line_bytes
            items_in_batch += 1
            claim_count += 1

            if claim_count % 50000 == 0:
                print(f"  Prepared {claim_count:,} / {len(retry_ids):,}...", flush=True)

    if current_file:
        current_file.close()
        if items_in_batch > 0:
            print(f"  {batch_files[-1].name}: {items_in_batch:,} claims, "
                  f"{current_size / 1e6:.1f} MB", flush=True)

    conn.close()
    print(f"\nPrepared {claim_count:,} claims in {len(batch_files)} JSONL files.")

    print(f"\nSubmitting {len(batch_files)} retry batches...", flush=True)
    tracker = {}
    for fpath in batch_files:
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
        print(f"    Batch {batch.id} ({batch.status})", flush=True)
        time.sleep(2)

    with open(BATCH_DIR / "retry_tracker.json", "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"\n{len(tracker)} retry batches submitted.")
    print("Poll with: python src/reclassify_batch.py --retry-poll")


def retry_poll():
    """Check status of retry batches."""
    from openai import OpenAI
    client = OpenAI()

    tracker_file = BATCH_DIR / "retry_tracker.json"
    if not tracker_file.exists():
        print("No retry_tracker.json found.")
        return

    with open(tracker_file) as f:
        tracker = json.load(f)

    total_done = 0
    total_failed = 0
    total_total = 0
    completed = 0
    statuses = {}

    for fname, info in sorted(tracker.items()):
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        if hasattr(batch, "output_file_id") and batch.output_file_id:
            info["output_file_id"] = batch.output_file_id
        rc = batch.request_counts
        if rc:
            info["completed"] = rc.completed
            info["failed"] = rc.failed
            info["total"] = rc.total
            total_done += rc.completed
            total_failed += rc.failed
            total_total += rc.total
        if batch.status == "completed":
            completed += 1
        statuses.setdefault(batch.status, 0)
        statuses[batch.status] += 1

    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)

    pct = total_done / total_total * 100 if total_total else 0
    print(f"Progress: {total_done:,} / {total_total:,} ({pct:.1f}%)")
    print(f"Failed: {total_failed:,}")
    print(f"Batch statuses: {statuses}")

    if completed == len(tracker):
        print("\nAll retry batches complete! Run: --retry-collect")


def retry_collect():
    """Download retry results and update DB."""
    from openai import OpenAI
    client = OpenAI()

    tracker_file = BATCH_DIR / "retry_tracker.json"
    if not tracker_file.exists():
        print("No retry_tracker.json found.")
        return

    with open(tracker_file) as f:
        tracker = json.load(f)

    raw_dir = BATCH_DIR / "retry" / "raw_results"
    raw_dir.mkdir(exist_ok=True)

    results = {}
    errors = 0
    empty_responses = 0

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            print(f"  {fname}: no output (status={info.get('status')})")
            continue

        raw_path = raw_dir / fname
        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            with open(raw_path, "wb") as f:
                f.write(content.read())

        with open(raw_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    claim_id = custom_id.replace("rcls_", "")
                    response = result.get("response", {})
                    body = response.get("body", {})

                    if response.get("status_code") != 200:
                        errors += 1
                        continue

                    choices = body.get("choices", [])
                    if not choices:
                        errors += 1
                        continue

                    text = choices[0].get("message", {}).get("content", "")
                    if not text:
                        empty_responses += 1
                        continue

                    parsed = json.loads(text)
                    normalized = {}
                    for view_id in ALL_CONTENT_VIEWS:
                        raw_path_val = parsed.get(view_id)
                        if raw_path_val and raw_path_val != ["not_applicable"]:
                            normed = normalize_path(view_id, raw_path_val)
                            if normed:
                                normalized[view_id] = normed

                    if normalized:
                        results[claim_id] = normalized

                except Exception:
                    errors += 1

    print(f"\nCollected {len(results):,} classifications, {errors} errors, "
          f"{empty_responses} still empty")

    print(f"\nUpdating database...", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    updated = 0
    items = list(results.items())
    batch_size = 5000

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        for claim_id, view_paths in batch:
            vp_json = json.dumps(view_paths)
            conn.execute(
                "UPDATE claims SET view_paths = ? WHERE claim_id = ?",
                [vp_json, claim_id],
            )
            updated += 1
        conn.commit()
        if (i + batch_size) % 50000 < batch_size:
            print(f"  Updated {min(i + batch_size, len(items)):,} / {len(items):,}...",
                  flush=True)

    conn.commit()
    conn.close()
    print(f"\nDone! Updated {updated:,} claims. Still empty: {empty_responses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reclassify claims with constrained taxonomy")
    parser.add_argument("--prepare", action="store_true", help="Build JSONL and submit batches")
    parser.add_argument("--poll", action="store_true", help="Check batch status")
    parser.add_argument("--collect", action="store_true", help="Download results and update DB")
    parser.add_argument("--rebuild-tree", action="store_true", help="Rebuild tree_nodes table")
    parser.add_argument("--retry", action="store_true", help="Re-submit failed claims")
    parser.add_argument("--retry-poll", action="store_true", help="Poll retry batches")
    parser.add_argument("--retry-collect", action="store_true", help="Collect retry results")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.poll:
        poll()
    elif args.collect:
        collect()
    elif args.rebuild_tree:
        rebuild_tree()
    elif args.retry:
        retry()
    elif args.retry_poll:
        retry_poll()
    elif args.retry_collect:
        retry_collect()
    else:
        parser.print_help()
