"""
Assign canonical L3 subcategories to claims via OpenAI Batch API.

Claims already have L1/L2 paths. This script assigns L3 only for (L1, L2)
pairs that have canonical L3 defined in CANONICAL_L3.

Usage:
    python src/reclassify_l3_batch.py --prepare     # Build JSONL files and submit batches
    python src/reclassify_l3_batch.py --poll         # Check batch status
    python src/reclassify_l3_batch.py --collect      # Download results and update DB
    python src/reclassify_l3_batch.py --retry        # Retry failed claims
    python src/reclassify_l3_batch.py --retry-poll   # Check retry batch status
    python src/reclassify_l3_batch.py --retry-collect # Collect retry results
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from askchem.taxonomy import (
    CANONICAL_L3, L3_ASSIGNMENT_SYSTEM_PROMPT,
    build_l3_assignment_messages, normalize_path,
    get_canonical_l3, ALL_CONTENT_VIEWS,
)

BATCH_DIR = Path(__file__).parent.parent / "data" / "l3_batches"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"
MAX_BATCH_FILE_BYTES = 190 * 1024 * 1024
MAX_BATCH_REQUESTS = 49_000  # OpenAI limit is 50,000
CLASSIFICATION_MODEL = "gpt-5-mini"


def _needs_l3(view_paths: dict) -> bool:
    """Check if any view path needs L3 assignment."""
    for vid, path in view_paths.items():
        if not isinstance(path, list) or len(path) < 2:
            continue
        l3_map = CANONICAL_L3.get(vid, {})
        if (path[0], path[1]) in l3_map:
            return True
    return False


def _build_batch_request(claim_id: str, claim_type: str, quote: str,
                         title: str, view_paths: dict,
                         max_tokens: int = 2048) -> dict | None:
    msgs = build_l3_assignment_messages(
        claim_type or "property",
        (quote or "")[:300],
        title or "",
        view_paths,
    )
    if not msgs:
        return None

    return {
        "custom_id": f"l3_{claim_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": CLASSIFICATION_MODEL,
            "messages": msgs,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
    }


def prepare():
    """Read claims from DB, build JSONL batch files, and submit to OpenAI."""
    from openai import OpenAI
    client = OpenAI()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as c FROM claims WHERE view_paths IS NOT NULL").fetchone()["c"]
    print(f"Total claims: {total:,}")

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    batch_files = []
    claim_count = 0
    skipped = 0

    cursor = conn.execute(
        "SELECT claim_id, claim_type, source_paper_title, verbatim_quote, view_paths FROM claims WHERE view_paths IS NOT NULL"
    )

    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break

        for row in rows:
            claim_id = row["claim_id"]
            try:
                vp = json.loads(row["view_paths"])
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue

            if not _needs_l3(vp):
                skipped += 1
                continue

            req = _build_batch_request(
                claim_id, row["claim_type"],
                row["verbatim_quote"], row["source_paper_title"],
                vp,
            )
            if not req:
                skipped += 1
                continue

            line = json.dumps(req) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES or items_in_batch >= MAX_BATCH_REQUESTS:
                if current_file:
                    current_file.close()
                batch_idx += 1
                fpath = BATCH_DIR / f"l3_batch_{batch_idx:04d}.jsonl"
                batch_files.append(fpath)
                current_file = open(fpath, "w")
                current_size = 0
                items_in_batch = 0
                print(f"  Starting batch file {batch_idx}...")

            current_file.write(line)
            current_size += line_bytes
            items_in_batch += 1
            claim_count += 1

            if claim_count % 100000 == 0:
                print(f"  Processed {claim_count:,} claims...")

    if current_file:
        current_file.close()

    print(f"\nPrepared {claim_count:,} claims in {len(batch_files)} batch files (skipped {skipped:,})")

    # Submit batches
    batch_ids = []
    for fpath in batch_files:
        print(f"  Uploading {fpath.name}...")
        with open(fpath, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"L3 assignment {fpath.name}"},
        )
        batch_ids.append(batch.id)
        print(f"    Batch {batch.id} submitted (file: {uploaded.id})")

    manifest = {
        "batch_ids": batch_ids,
        "total_claims": claim_count,
        "batch_files": [str(f) for f in batch_files],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = BATCH_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSubmitted {len(batch_ids)} batches. Manifest saved to {manifest_path}")


def poll():
    """Check status of all submitted batches."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    completed = 0
    failed = 0
    in_progress = 0

    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        status = batch.status
        req_counts = batch.request_counts
        total = req_counts.total if req_counts else 0
        done = req_counts.completed if req_counts else 0
        fail = req_counts.failed if req_counts else 0

        if status == "completed":
            completed += 1
        elif status in ("failed", "expired", "cancelled"):
            failed += 1
        else:
            in_progress += 1

        print(f"  {bid}: {status} ({done}/{total} done, {fail} failed)")

    print(f"\nSummary: {completed} completed, {in_progress} in progress, {failed} failed")


def collect():
    """Download results and update DB."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    updated = 0
    errors = 0
    empty_responses = 0

    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        if batch.status not in ("completed", "cancelled"):
            print(f"  Skipping {bid} (status: {batch.status})")
            continue

        if not batch.output_file_id:
            print(f"  Skipping {bid} (no output file)")
            continue

        print(f"  Downloading results for {bid}...")
        content = client.files.content(batch.output_file_id)
        lines = content.text.strip().split("\n")

        batch_updated = 0
        batch_errors = 0

        for line in lines:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            custom_id = result.get("custom_id", "")
            claim_id = custom_id.replace("l3_", "", 1)

            response = result.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])

            if not choices:
                empty_responses += 1
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason", "")
            if finish_reason == "length":
                empty_responses += 1
                continue

            msg_content = choice.get("message", {}).get("content", "")
            if not msg_content:
                empty_responses += 1
                continue

            try:
                l3_result = json.loads(msg_content)
            except json.JSONDecodeError:
                errors += 1
                continue

            # Get current view_paths
            row = conn.execute(
                "SELECT view_paths FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            if not row:
                errors += 1
                continue

            try:
                vp = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                errors += 1
                continue

            # Apply L3 assignments
            changed = False
            for vid, l3_val in l3_result.items():
                if vid not in vp or not isinstance(vp[vid], list) or len(vp[vid]) < 2:
                    continue
                l1, l2 = vp[vid][0], vp[vid][1]
                allowed = get_canonical_l3(vid, l1, l2)
                if allowed is None:
                    continue
                if l3_val in allowed:
                    vp[vid] = [l1, l2, l3_val]
                    changed = True
                else:
                    vp[vid] = [l1, l2, "other"]
                    changed = True

            if changed:
                new_vp = json.dumps(vp)
                conn.execute(
                    "UPDATE claims SET view_paths = ? WHERE claim_id = ?",
                    (new_vp, claim_id),
                )
                batch_updated += 1

        conn.commit()
        updated += batch_updated
        batch_errors = errors
        print(f"    Updated {batch_updated:,} claims")

    # Also sync view_paths into data JSON
    print(f"\nSyncing view_paths into data JSON blobs...")
    synced = _sync_data_json(conn)

    conn.close()
    print(f"\nDone! Updated: {updated:,}, Errors: {errors:,}, Empty: {empty_responses:,}, Synced data: {synced:,}")


def _sync_data_json(conn):
    """Sync view_paths column into the data JSON blob."""
    cursor = conn.execute("SELECT claim_id, view_paths, data FROM claims WHERE view_paths IS NOT NULL")
    synced = 0
    batch = []

    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break
        for row in rows:
            try:
                vp = json.loads(row[1])
                data = json.loads(row[2]) if row[2] else {}
            except (json.JSONDecodeError, TypeError):
                continue

            old_vp = data.get("view_paths")
            if old_vp != vp:
                data["view_paths"] = vp
                batch.append((json.dumps(data), row[0]))
                synced += 1

        if len(batch) >= 10000:
            conn.executemany("UPDATE claims SET data = ? WHERE claim_id = ?", batch)
            conn.commit()
            batch = []

    if batch:
        conn.executemany("UPDATE claims SET data = ? WHERE claim_id = ?", batch)
        conn.commit()

    return synced


def retry():
    """Retry claims that got empty responses."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Collect failed claim IDs
    failed_ids = set()
    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        if batch.status != "completed" or not batch.output_file_id:
            continue

        content = client.files.content(batch.output_file_id)
        for line in content.text.strip().split("\n"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = result.get("custom_id", "")
            claim_id = custom_id.replace("l3_", "", 1)

            response = result.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])

            if not choices:
                failed_ids.add(claim_id)
                continue

            choice = choices[0]
            if choice.get("finish_reason") == "length":
                failed_ids.add(claim_id)
                continue

            msg_content = choice.get("message", {}).get("content", "")
            if not msg_content:
                failed_ids.add(claim_id)
                continue

            try:
                json.loads(msg_content)
            except json.JSONDecodeError:
                failed_ids.add(claim_id)

    if not failed_ids:
        print("No failed claims to retry!")
        return

    print(f"Found {len(failed_ids):,} failed claims. Preparing retry batches...")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    RETRY_DIR = BATCH_DIR / "retry"
    RETRY_DIR.mkdir(parents=True, exist_ok=True)

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    batch_files = []
    claim_count = 0

    placeholders = ",".join("?" for _ in failed_ids)
    cursor = conn.execute(
        f"SELECT claim_id, claim_type, source_paper_title, verbatim_quote, view_paths "
        f"FROM claims WHERE claim_id IN ({placeholders})",
        list(failed_ids),
    )

    for row in cursor:
        try:
            vp = json.loads(row["view_paths"])
        except (json.JSONDecodeError, TypeError):
            continue

        req = _build_batch_request(
            row["claim_id"], row["claim_type"],
            row["verbatim_quote"], row["source_paper_title"],
            vp, max_tokens=8192,
        )
        if not req:
            continue

        line = json.dumps(req) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES or items_in_batch >= MAX_BATCH_REQUESTS:
            if current_file:
                current_file.close()
            batch_idx += 1
            fpath = RETRY_DIR / f"l3_retry_{batch_idx:04d}.jsonl"
            batch_files.append(fpath)
            current_file = open(fpath, "w")
            current_size = 0
            items_in_batch = 0

        current_file.write(line)
        current_size += line_bytes
        items_in_batch += 1
        claim_count += 1

    if current_file:
        current_file.close()

    print(f"Prepared {claim_count:,} claims in {len(batch_files)} retry files")

    retry_batch_ids = []
    for fpath in batch_files:
        print(f"  Uploading {fpath.name}...")
        with open(fpath, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"L3 retry {fpath.name}"},
        )
        retry_batch_ids.append(batch.id)
        print(f"    Batch {batch.id} submitted")

    retry_manifest = {
        "batch_ids": retry_batch_ids,
        "total_claims": claim_count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RETRY_DIR / "manifest.json", "w") as f:
        json.dump(retry_manifest, f, indent=2)

    print(f"Submitted {len(retry_batch_ids)} retry batches")


def retry_poll():
    """Check status of retry batches."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "retry" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        req = batch.request_counts
        total = req.total if req else 0
        done = req.completed if req else 0
        fail = req.failed if req else 0
        print(f"  {bid}: {batch.status} ({done}/{total} done, {fail} failed)")


def retry_collect():
    """Collect retry results."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "retry" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    updated = 0
    errors = 0

    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        if batch.status != "completed" or not batch.output_file_id:
            print(f"  Skipping {bid} (status: {batch.status})")
            continue

        content = client.files.content(batch.output_file_id)
        for line in content.text.strip().split("\n"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            custom_id = result.get("custom_id", "")
            claim_id = custom_id.replace("l3_", "", 1)

            response = result.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                errors += 1
                continue

            choice = choices[0]
            msg_content = choice.get("message", {}).get("content", "")
            if not msg_content:
                errors += 1
                continue

            try:
                l3_result = json.loads(msg_content)
            except json.JSONDecodeError:
                errors += 1
                continue

            row = conn.execute(
                "SELECT view_paths FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            if not row:
                errors += 1
                continue

            try:
                vp = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                errors += 1
                continue

            changed = False
            for vid, l3_val in l3_result.items():
                if vid not in vp or not isinstance(vp[vid], list) or len(vp[vid]) < 2:
                    continue
                l1, l2 = vp[vid][0], vp[vid][1]
                allowed = get_canonical_l3(vid, l1, l2)
                if allowed is None:
                    continue
                if l3_val in allowed:
                    vp[vid] = [l1, l2, l3_val]
                    changed = True
                else:
                    vp[vid] = [l1, l2, "other"]
                    changed = True

            if changed:
                conn.execute(
                    "UPDATE claims SET view_paths = ? WHERE claim_id = ?",
                    (json.dumps(vp), claim_id),
                )
                updated += 1

    conn.commit()

    synced = _sync_data_json(conn)
    conn.close()

    print(f"Retry results: Updated {updated:,}, Errors {errors:,}, Synced {synced:,}")


def retry_other_prepare():
    """Find claims with L3='other' and resubmit with a stronger prompt."""
    from openai import OpenAI
    client = OpenAI()

    RETRY_OTHER_DIR = BATCH_DIR / "retry_other"
    RETRY_OTHER_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        "SELECT claim_id, claim_type, source_paper_title, verbatim_quote, view_paths "
        "FROM claims WHERE view_paths IS NOT NULL AND view_paths != '{}'"
    )

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    batch_files = []
    claim_count = 0
    skipped = 0

    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break

        for row in rows:
            try:
                vp = json.loads(row["view_paths"])
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue

            has_other = False
            for vid, path in vp.items():
                if isinstance(path, list) and len(path) >= 3 and path[2] == "other":
                    l3_cats = get_canonical_l3(vid, path[0], path[1])
                    if l3_cats and len(l3_cats) > 1:
                        has_other = True
                        break

            if not has_other:
                skipped += 1
                continue

            msgs = _build_retry_other_messages(
                row["claim_type"] or "property",
                (row["verbatim_quote"] or "")[:300],
                row["source_paper_title"] or "",
                vp,
            )
            if not msgs:
                skipped += 1
                continue

            req = {
                "custom_id": f"l3o_{row['claim_id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CLASSIFICATION_MODEL,
                    "messages": msgs,
                    "max_completion_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
            }

            line = json.dumps(req) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES or items_in_batch >= MAX_BATCH_REQUESTS:
                if current_file:
                    current_file.close()
                batch_idx += 1
                fpath = RETRY_OTHER_DIR / f"l3_other_{batch_idx:04d}.jsonl"
                batch_files.append(fpath)
                current_file = open(fpath, "w")
                current_size = 0
                items_in_batch = 0
                print(f"  Starting batch file {batch_idx}...")

            current_file.write(line)
            current_size += line_bytes
            items_in_batch += 1
            claim_count += 1

            if claim_count % 50000 == 0:
                print(f"  Processed {claim_count:,} claims...")

    if current_file:
        current_file.close()

    print(f"\nPrepared {claim_count:,} claims in {len(batch_files)} batch files (skipped {skipped:,})")

    batch_ids = []
    for fpath in batch_files:
        print(f"  Uploading {fpath.name}...")
        with open(fpath, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"L3 other retry {fpath.name}"},
        )
        batch_ids.append(batch.id)
        print(f"    Batch {batch.id} submitted")

    manifest = {
        "batch_ids": batch_ids,
        "total_claims": claim_count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RETRY_OTHER_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSubmitted {len(batch_ids)} batches. Manifest saved.")


def _build_retry_other_messages(claim_type, quote, title, view_paths):
    """Build L3 assignment messages that discourage 'other'."""
    l3_needed = {}
    for view_id, path in view_paths.items():
        if not isinstance(path, list) or len(path) < 3:
            continue
        if path[2] != "other":
            continue
        l1, l2 = path[0], path[1]
        l3_cats = get_canonical_l3(view_id, l1, l2)
        if l3_cats is None or len(l3_cats) <= 1:
            continue
        specific_cats = [c for c in l3_cats if c != "other"]
        l3_needed[view_id] = {"path": f"{l1}/{l2}", "allowed_l3": specific_cats}

    if not l3_needed:
        return None

    system = (
        "You are a chemistry taxonomy classifier. "
        "A previous classifier assigned 'other' for the L3 subcategory, "
        "but we want to try harder to find a specific match.\n\n"
        "Pick the BEST matching L3 subcategory from the allowed list for each view. "
        "Only return 'other' if the claim truly does not fit ANY of the specific categories. "
        "Be generous — if the claim is even loosely related to a category, pick it.\n\n"
        "Return JSON: {\"view_id\": \"chosen_l3\", ...}"
    )

    lines = [f"Claim type: {claim_type}", f"Claim: {quote}", f"Paper: {title}", ""]
    lines.append("Assign the best L3 for each view (avoid 'other' if possible):")
    for view_id, info in l3_needed.items():
        lines.append(f"  {view_id} ({info['path']}): {', '.join(info['allowed_l3'])}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def retry_other_poll():
    """Check status of retry-other batches."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "retry_other" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    completed = 0
    in_progress = 0
    failed = 0

    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        req = batch.request_counts
        total = req.total if req else 0
        done = req.completed if req else 0
        fail = req.failed if req else 0

        if batch.status == "completed":
            completed += 1
        elif batch.status in ("failed", "expired", "cancelled"):
            failed += 1
        else:
            in_progress += 1

        print(f"  {bid}: {batch.status} ({done}/{total} done, {fail} failed)")

    print(f"\nSummary: {completed} completed, {in_progress} in progress, {failed} failed")


def retry_other_collect():
    """Collect retry-other results and update DB."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "retry_other" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    updated = 0
    still_other = 0
    errors = 0

    for bid in manifest["batch_ids"]:
        batch = client.batches.retrieve(bid)
        if batch.status not in ("completed", "cancelled"):
            print(f"  Skipping {bid} (status: {batch.status})")
            continue

        if not batch.output_file_id:
            print(f"  Skipping {bid} (no output file)")
            continue

        print(f"  Downloading results for {bid}...")
        content = client.files.content(batch.output_file_id)
        lines = content.text.strip().split("\n")
        batch_updated = 0

        for line in lines:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            custom_id = result.get("custom_id", "")
            claim_id = custom_id.replace("l3o_", "", 1)

            response = result.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                errors += 1
                continue

            msg_content = choices[0].get("message", {}).get("content", "")
            if not msg_content:
                errors += 1
                continue

            try:
                l3_result = json.loads(msg_content)
            except json.JSONDecodeError:
                errors += 1
                continue

            row = conn.execute(
                "SELECT view_paths FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            if not row:
                errors += 1
                continue

            try:
                vp = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                errors += 1
                continue

            changed = False
            for vid, l3_val in l3_result.items():
                if vid not in vp or not isinstance(vp[vid], list) or len(vp[vid]) < 3:
                    continue
                if vp[vid][2] != "other":
                    continue
                l1, l2 = vp[vid][0], vp[vid][1]
                allowed = get_canonical_l3(vid, l1, l2)
                if allowed is None:
                    continue
                l3_clean = str(l3_val).strip().lower().replace("-", "_").replace(" ", "_")
                if l3_clean in allowed and l3_clean != "other":
                    vp[vid] = [l1, l2, l3_clean]
                    changed = True
                else:
                    still_other += 1

            if changed:
                conn.execute(
                    "UPDATE claims SET view_paths = ? WHERE claim_id = ?",
                    (json.dumps(vp), claim_id),
                )
                batch_updated += 1
                updated += 1

        print(f"    Updated {batch_updated:,} claims")

    conn.commit()
    synced = _sync_data_json(conn)
    conn.close()

    print(f"\nDone! Updated: {updated:,}, Still other: {still_other:,}, Errors: {errors:,}, Synced: {synced:,}")


def _smart_title(s: str) -> str:
    """Convert slug to title case."""
    return s.replace("_", " ").replace("-", " ").title()


def rebuild_tree():
    """Rebuild tree_nodes table from view_paths. Bottom-up O(N) aggregation."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print("Dropping old tree_nodes...")
    conn.execute("DELETE FROM tree_nodes")
    conn.commit()

    all_views = ALL_CONTENT_VIEWS + ["by_claim_type", "by_time_period"]

    for view_id in all_views:
        print(f"  Building tree for {view_id}...")
        nodes = {}  # path_str -> {"own_claims": int, "children": set, "claim_ids": list}

        cursor = conn.execute(
            "SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL"
        )
        claim_count = 0

        while True:
            rows = cursor.fetchmany(50000)
            if not rows:
                break
            for row in rows:
                try:
                    vp = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    continue
                path = vp.get(view_id)
                if not isinstance(path, list) or not path:
                    continue

                claim_count += 1
                claim_id = row[0]
                for depth in range(len(path)):
                    node_path = path[:depth + 1]
                    path_str = "/".join(node_path)
                    if path_str not in nodes:
                        nodes[path_str] = {"own_claims": 0, "children": set(), "claim_ids": []}
                    if depth == len(path) - 1:
                        nodes[path_str]["own_claims"] += 1
                        if len(nodes[path_str]["claim_ids"]) < 100:
                            nodes[path_str]["claim_ids"].append(claim_id)
                    if depth > 0:
                        parent_str = "/".join(path[:depth])
                        nodes[parent_str]["children"].add(path_str)

        def count_descendants(path_str):
            node = nodes[path_str]
            total = node["own_claims"]
            for child in node["children"]:
                total += count_descendants(child)
            return total

        inserts = []
        for path_str, node in nodes.items():
            parts = path_str.split("/")
            total_claims = count_descendants(path_str)
            child_segments = sorted(
                p.split("/")[-1] for p in node["children"]
            )
            name = _smart_title(parts[-1])
            level = len(parts)
            node_data = {
                "node_id": f"{view_id}_{path_str}",
                "name": name,
                "claim_count": total_claims,
            }
            inserts.append((
                view_id,
                path_str,
                name,
                level,
                total_claims,
                json.dumps(child_segments),
                json.dumps(node["claim_ids"][:100]),
                json.dumps(node_data),
            ))

        conn.executemany(
            "INSERT INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            inserts,
        )
        conn.commit()
        print(f"    {len(inserts):,} nodes, {claim_count:,} claims")

    conn.close()
    print("Done rebuilding tree_nodes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--retry-poll", action="store_true")
    parser.add_argument("--retry-collect", action="store_true")
    parser.add_argument("--retry-other-prepare", action="store_true")
    parser.add_argument("--retry-other-poll", action="store_true")
    parser.add_argument("--retry-other-collect", action="store_true")
    parser.add_argument("--rebuild-tree", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.poll:
        poll()
    elif args.collect:
        collect()
    elif args.retry:
        retry()
    elif args.retry_poll:
        retry_poll()
    elif args.retry_collect:
        retry_collect()
    elif args.retry_other_prepare:
        retry_other_prepare()
    elif args.retry_other_poll:
        retry_other_poll()
    elif args.retry_other_collect:
        retry_other_collect()
    elif args.rebuild_tree:
        rebuild_tree()
    else:
        parser.print_help()
