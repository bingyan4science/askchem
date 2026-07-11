"""
Batch classification of claims via Vertex AI (PortKey gateway).

Uses the two-step approach:
  Step 1: L1/L2 classification with CLASSIFICATION_SYSTEM_PROMPT (small prompt)
  Step 2: L3 assignment for views that have canonical L3 definitions

Workflow:
    python src/batch_classify.py generate   # Generate JSONL batch files
    python src/batch_classify.py submit      # Upload & submit all batch files
    python src/batch_classify.py status      # Check batch job statuses
    python src/batch_classify.py collect     # Download results and apply to claims
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from askchem.taxonomy import (
    CLASSIFICATION_SYSTEM_PROMPT,
    L3_ASSIGNMENT_SYSTEM_PROMPT,
    build_classification_prompt,
    build_l3_assignment_messages,
    normalize_path,
    get_canonical_l3,
    ALL_CONTENT_VIEWS,
    CANONICAL_L3,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "deep_results"
BATCH_DIR = DATA_DIR / "classify_batches"

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"
CLAIMS_PER_FILE = 10_000


def _curl_json(method, path, data=None, form_fields=None, file_path=None, max_time=60):
    """Call the PortKey gateway via curl and return parsed JSON."""
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-s", "--max-time", str(max_time), "-X", method]
    cmd += ["-H", f"x-portkey-api-key: {api_key}"]
    cmd += ["-H", f"x-portkey-provider: {PROVIDER}"]

    if data is not None:
        cmd += ["-H", "Content-Type: application/json"]
        cmd += ["-d", json.dumps(data)]
    elif form_fields or file_path:
        bucket = os.environ["GCS_BKT"]
        cmd += ["-H", f"x-portkey-vertex-storage-bucket-name: {bucket}"]
        if form_fields:
            for k, v in form_fields.items():
                if k == "provider_file_name":
                    cmd += ["-H", f"x-portkey-provider-file-name: {v}"]
                elif k == "provider_model":
                    cmd += ["-H", f"x-portkey-provider-model: {v}"]
                else:
                    cmd += ["--form", f'{k}="{v}"']
        if file_path:
            cmd += ["--form", f"file=@{file_path}"]

    cmd.append(f"{GATEWAY}{path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 30)
    if not result.stdout.strip():
        return {"error": "empty_response", "stderr": result.stderr[:200]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "parse_error", "raw": result.stdout[:500]}


def cmd_generate(args):
    """Generate JSONL batch files from deep_results/ claims."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    print(f"Scanning {len(result_files)} result files...")

    requests = []
    papers_seen = 0

    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
        except Exception:
            continue

        claims = data.get("data", {}).get("claims", [])
        if not claims:
            continue

        paper_id = rf.stem
        doi = data.get("doi", "")
        papers_seen += 1

        for ci, claim in enumerate(claims):
            existing = claim.get("classification", {})
            if isinstance(existing, dict) and any(
                isinstance(v, list) and len(v) >= 2 for v in existing.values()
            ):
                continue

            claim_type = claim.get("claim_type", "property")
            quote = (claim.get("verbatim_quote") or "")[:300]

            custom_id = f"{paper_id}__c{ci}"
            messages = [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": build_classification_prompt(claim_type, quote, doi)},
            ]
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": messages,
                    "max_completion_tokens": 16384,
                    "response_format": {"type": "json_object"},
                },
            }
            requests.append(request)

    print(f"Papers with claims: {papers_seen}")
    print(f"Claims needing classification: {len(requests)}")

    if not requests:
        print("Nothing to do.")
        return

    num_files = (len(requests) + CLAIMS_PER_FILE - 1) // CLAIMS_PER_FILE
    manifest = []

    for fi in range(num_files):
        start = fi * CLAIMS_PER_FILE
        end = min(start + CLAIMS_PER_FILE, len(requests))
        chunk = requests[start:end]

        fname = f"classify_l12_part{fi:03d}.jsonl"
        fpath = BATCH_DIR / fname
        with open(fpath, "w") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")

        manifest.append({
            "file": fname,
            "count": len(chunk),
            "size_mb": round(fpath.stat().st_size / 1e6, 1),
        })
        print(f"  {fname}: {len(chunk)} requests, {manifest[-1]['size_mb']} MB")

    manifest_path = BATCH_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "files": manifest}, f, indent=2)

    print(f"\nGenerated {num_files} batch files in {BATCH_DIR}/")
    print(f"Total requests: {len(requests)}")


def cmd_submit(args):
    """Upload JSONL files to GCS and submit batch jobs."""
    batch_dir = getattr(args, "_batch_dir", BATCH_DIR)
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json found. Run 'generate' first.")
        return

    manifest = json.loads(manifest_path.read_text())
    tracker_path = batch_dir / "tracker.json"
    tracker = {}
    if tracker_path.exists():
        tracker = json.loads(tracker_path.read_text())

    ok = 0
    fail = 0
    total = len(manifest["files"])
    for i, entry in enumerate(manifest["files"]):
        fname = entry["file"]
        if fname in tracker and tracker[fname].get("status") not in ("failed",):
            print(f"  [{i+1}/{total}] {fname}: already submitted (status={tracker[fname].get('status')}), skipping")
            continue

        fpath = batch_dir / fname
        size_mb = entry.get("size_mb", 0)
        upload_timeout = max(120, int(size_mb * 5))
        print(f"  [{i+1}/{total}] {fname} ({size_mb} MB)...", end=" ", flush=True)

        upload_resp = _curl_json("POST", "/files",
            form_fields={
                "purpose": "batch",
                "provider_file_name": fname,
                "provider_model": MODEL,
            },
            file_path=str(fpath),
            max_time=upload_timeout,
        )
        file_id = upload_resp.get("id")
        if not file_id:
            print(f"UPLOAD FAILED: {str(upload_resp)[:80]}")
            fail += 1
            time.sleep(10)
            continue

        batch_resp = _curl_json("POST", "/batches", data={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": MODEL,
        })
        batch_id = batch_resp.get("id")
        status = batch_resp.get("status", "unknown")

        if batch_id:
            print(f"OK batch={batch_id[:20]}")
            ok += 1
        else:
            print(f"NO BATCH: {str(batch_resp)[:80]}")
            fail += 1

        tracker[fname] = {
            "file_id": file_id,
            "batch_id": batch_id,
            "status": status,
            "submitted_at": datetime.now().isoformat(),
        }
        with open(tracker_path, "w") as f:
            json.dump(tracker, f, indent=2)

        time.sleep(5)

    print(f"\nSubmit done: {ok} ok, {fail} fail out of {total}")
    print(f"Tracker saved to {tracker_path}")


def cmd_status(args):
    """Check status of all submitted batch jobs."""
    batch_dir = getattr(args, "_batch_dir", BATCH_DIR)
    tracker_path = batch_dir / "tracker.json"
    if not tracker_path.exists():
        print("No tracker.json found. Run 'submit' first.")
        return

    tracker = json.loads(tracker_path.read_text())
    summary = {"validating": 0, "in_progress": 0, "completed": 0, "failed": 0, "other": 0}

    for fname, info in tracker.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            continue

        resp = _curl_json("GET", f"/batches/{batch_id}")
        new_status = resp.get("status", "unknown")
        counts = resp.get("request_counts", {})
        info["status"] = new_status
        info["request_counts"] = counts

        cat = new_status if new_status in summary else "other"
        summary[cat] += 1

        completed = counts.get("completed") or 0
        total = counts.get("total") or 0
        print(f"  {fname}: {new_status} ({completed}/{total})")
        time.sleep(0.5)

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"\nSummary: {json.dumps(summary)}")


def cmd_collect(args):
    """Download completed batch results and apply classifications to claims."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batch_dir = getattr(args, "_batch_dir", BATCH_DIR)
    tracker_path = batch_dir / "tracker.json"
    if not tracker_path.exists():
        print("No tracker.json found.")
        return

    tracker = json.loads(tracker_path.read_text())
    output_dir = batch_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    to_collect = [(k, v) for k, v in tracker.items()
                  if v.get("status") == "completed" and not v.get("collected") and v.get("batch_id")]
    print(f"Collecting {len(to_collect)} completed batches (8 workers)...")

    api_key = os.environ["PORTKEY_API_KEY"]
    lock = threading.Lock()
    stats = {"ok": 0, "empty": 0}

    def _dl_one(fname, info):
        batch_id = info["batch_id"]
        cmd = ["curl", "-s", "--max-time", "180", "-X", "GET",
               "-H", f"x-portkey-api-key: {api_key}",
               "-H", f"x-portkey-provider: {PROVIDER}",
               f"{GATEWAY}/batches/{batch_id}/output"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            raw = result.stdout.strip()
        except Exception:
            raw = ""

        if not raw or len(raw) < 10:
            with lock:
                stats["empty"] += 1
            return

        out_path = output_dir / fname
        with open(out_path, "w") as f:
            f.write(result.stdout)

        with lock:
            info["collected"] = True
            info["collected_at"] = datetime.now().isoformat()
            stats["ok"] += 1
            if stats["ok"] % 10 == 0:
                with open(tracker_path, "w") as tf:
                    json.dump(tracker, tf, indent=2)
                print(f"  Collected: {stats['ok']}/{len(to_collect)} (empty: {stats['empty']})", flush=True)

    workers = min(8, len(to_collect))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_dl_one, k, v) for k, v in to_collect]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                pass

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"Collect done: {stats['ok']} ok, {stats['empty']} empty/failed")
    _apply_results(output_dir)


def _apply_results(output_dir: Path):
    """Parse batch outputs and write classifications back to deep_results/ files."""
    classifications = {}

    for ofile in output_dir.glob("*.jsonl"):
        for line in ofile.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                custom_id = item.get("custom_id", "")
                response = item.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    parsed = json.loads(content)
                    if isinstance(parsed, list) and len(parsed) == 1:
                        parsed = parsed[0]
                    if isinstance(parsed, dict):
                        normalized = {}
                        for view_id in ALL_CONTENT_VIEWS:
                            raw_path = parsed.get(view_id)
                            if raw_path and raw_path != ["not_applicable"]:
                                normed = normalize_path(view_id, raw_path)
                                if normed:
                                    normalized[view_id] = normed
                        if normalized:
                            classifications[custom_id] = normalized
            except Exception:
                continue

    print(f"\n  Parsed {len(classifications)} classifications from batch outputs")

    applied = 0
    for rf in sorted(RESULTS_DIR.glob("*.json")):
        paper_id = rf.stem
        data = json.loads(rf.read_text())
        claims = data.get("data", {}).get("claims", [])
        modified = False

        for ci, claim in enumerate(claims):
            cid = f"{paper_id}__c{ci}"
            if cid in classifications:
                claim["classification"] = classifications[cid]
                applied += 1
                modified = True

        if modified:
            data["classified_at"] = datetime.now().isoformat()
            with open(rf, "w") as f:
                json.dump(data, f, indent=2)

    print(f"  Applied {applied} classifications to deep_results/")


def cmd_l3_generate(args):
    """Generate L3 assignment batch files for claims with L3='other'."""
    L3_BATCH_DIR = DATA_DIR / "classify_l3_batches"
    L3_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    print(f"Scanning {len(result_files)} result files...")

    requests = []
    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
        except Exception:
            continue

        claims = data.get("data", {}).get("claims", [])
        if not claims:
            continue

        paper_id = rf.stem
        doi = data.get("doi", "")

        for ci, claim in enumerate(claims):
            cls = claim.get("classification", {})
            if not isinstance(cls, dict) or not cls:
                continue

            # Check if any view needs L3
            l3_needed = {}
            for view_id, path in cls.items():
                if not isinstance(path, list) or len(path) < 2:
                    continue
                l1, l2 = path[0], path[1]
                l3_cats = get_canonical_l3(view_id, l1, l2)
                if l3_cats is not None and (len(path) < 3 or path[2] == "other"):
                    l3_needed[view_id] = {
                        "path": f"{l1}/{l2}",
                        "allowed_l3": l3_cats,
                    }

            if not l3_needed:
                continue

            claim_type = claim.get("claim_type", "property")
            quote = (claim.get("verbatim_quote") or "")[:300]

            lines = [f"Claim type: {claim_type}", f"Claim: {quote}", f"Paper: {doi}", ""]
            lines.append("Assign L3 for each view:")
            for view_id, info in l3_needed.items():
                lines.append(f"  {view_id} ({info['path']}): {', '.join(info['allowed_l3'])}")

            messages = [
                {"role": "system", "content": L3_ASSIGNMENT_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ]

            custom_id = f"{paper_id}__c{ci}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": messages,
                    "max_completion_tokens": 16384,
                    "response_format": {"type": "json_object"},
                },
            }
            requests.append(request)

    print(f"Claims needing L3: {len(requests)}")

    if not requests:
        print("Nothing to do.")
        return

    num_files = (len(requests) + CLAIMS_PER_FILE - 1) // CLAIMS_PER_FILE
    manifest = []

    for fi in range(num_files):
        start = fi * CLAIMS_PER_FILE
        end = min(start + CLAIMS_PER_FILE, len(requests))
        chunk = requests[start:end]

        fname = f"classify_l3_part{fi:03d}.jsonl"
        fpath = L3_BATCH_DIR / fname
        with open(fpath, "w") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")

        manifest.append({
            "file": fname,
            "count": len(chunk),
            "size_mb": round(fpath.stat().st_size / 1e6, 1),
        })
        print(f"  {fname}: {len(chunk)} requests, {manifest[-1]['size_mb']} MB")

    l3_manifest_path = L3_BATCH_DIR / "manifest.json"
    with open(l3_manifest_path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "files": manifest}, f, indent=2)

    print(f"\nGenerated {num_files} batch files in {L3_BATCH_DIR}/")
    print(f"Total requests: {len(requests)}")


def cmd_l3_submit(args):
    """Upload and submit L3 batch files (reuses cmd_submit logic)."""
    args._batch_dir = DATA_DIR / "classify_l3_batches"
    cmd_submit(args)


def cmd_l3_status(args):
    """Check status of L3 batch jobs."""
    args._batch_dir = DATA_DIR / "classify_l3_batches"
    cmd_status(args)


def cmd_l3_collect(args):
    """Download L3 results and merge into deep_results/."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    L3_BATCH_DIR = DATA_DIR / "classify_l3_batches"
    tracker_path = L3_BATCH_DIR / "tracker.json"
    if not tracker_path.exists():
        print("No tracker.json found.")
        return

    tracker = json.loads(tracker_path.read_text())
    output_dir = L3_BATCH_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    to_collect = [(k, v) for k, v in tracker.items()
                  if v.get("status") == "completed" and not v.get("collected") and v.get("batch_id")]
    print(f"Collecting {len(to_collect)} completed L3 batches (8 workers)...")

    api_key = os.environ["PORTKEY_API_KEY"]
    lock = threading.Lock()
    stats = {"ok": 0, "empty": 0}

    def _dl_one(fname, info):
        batch_id = info["batch_id"]
        cmd = ["curl", "-s", "--max-time", "180", "-X", "GET",
               "-H", f"x-portkey-api-key: {api_key}",
               "-H", f"x-portkey-provider: {PROVIDER}",
               f"{GATEWAY}/batches/{batch_id}/output"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            raw = result.stdout.strip()
        except Exception:
            raw = ""

        if not raw or len(raw) < 10:
            with lock:
                stats["empty"] += 1
            return

        out_path = output_dir / fname
        with open(out_path, "w") as f:
            f.write(result.stdout)

        with lock:
            info["collected"] = True
            info["collected_at"] = datetime.now().isoformat()
            stats["ok"] += 1
            if stats["ok"] % 10 == 0:
                with open(tracker_path, "w") as tf:
                    json.dump(tracker, tf, indent=2)
                print(f"  Collected: {stats['ok']}/{len(to_collect)} (empty: {stats['empty']})", flush=True)

    workers = min(8, len(to_collect))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_dl_one, k, v) for k, v in to_collect]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                pass

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"Collect done: {stats['ok']} ok, {stats['empty']} empty/failed")
    _apply_l3_results(output_dir)


def _apply_l3_results(output_dir: Path):
    """Parse L3 outputs and merge into deep_results/ classifications."""
    l3_assignments = {}

    for ofile in output_dir.glob("*.jsonl"):
        for line in ofile.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                custom_id = item.get("custom_id", "")
                content = item["response"]["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if isinstance(parsed, list) and len(parsed) == 1:
                    parsed = parsed[0]
                if isinstance(parsed, dict):
                    l3_assignments[custom_id] = parsed
            except Exception:
                continue

    print(f"\n  Parsed {len(l3_assignments)} L3 assignments from batch outputs")

    applied = 0
    for rf in sorted(RESULTS_DIR.glob("*.json")):
        paper_id = rf.stem
        data = json.loads(rf.read_text())
        claims = data.get("data", {}).get("claims", [])
        modified = False

        for ci, claim in enumerate(claims):
            cid = f"{paper_id}__c{ci}"
            if cid not in l3_assignments:
                continue

            cls = claim.get("classification", {})
            if not isinstance(cls, dict):
                continue

            l3_result = l3_assignments[cid]
            for view_id, l3_val in l3_result.items():
                if view_id not in cls:
                    continue
                path = cls[view_id]
                if not isinstance(path, list) or len(path) < 2:
                    continue
                l1, l2 = path[0], path[1]
                l3_cats = get_canonical_l3(view_id, l1, l2)
                if l3_cats is None:
                    continue
                l3_str = str(l3_val).strip().lower().replace("-", "_").replace(" ", "_")
                if l3_str in set(l3_cats):
                    cls[view_id] = [l1, l2, l3_str]
                else:
                    cls[view_id] = [l1, l2, "other"]
                applied += 1
                modified = True

        if modified:
            data["l3_classified_at"] = datetime.now().isoformat()
            with open(rf, "w") as f:
                json.dump(data, f, indent=2)

    print(f"  Applied {applied} L3 assignments to deep_results/")


def main():
    parser = argparse.ArgumentParser(description="Batch classification pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate L1/L2 JSONL batch files")
    sub.add_parser("submit", help="Upload & submit L1/L2 batch files")
    sub.add_parser("status", help="Check L1/L2 batch job statuses")
    sub.add_parser("collect", help="Download L1/L2 results and apply to claims")
    sub.add_parser("l3-generate", help="Generate L3 assignment batch files")
    sub.add_parser("l3-submit", help="Upload & submit L3 batch files")
    sub.add_parser("l3-status", help="Check L3 batch job statuses")
    sub.add_parser("l3-collect", help="Download L3 results and apply to claims")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "generate": cmd_generate, "submit": cmd_submit,
        "status": cmd_status, "collect": cmd_collect,
        "l3-generate": cmd_l3_generate, "l3-submit": cmd_l3_submit,
        "l3-status": cmd_l3_status, "l3-collect": cmd_l3_collect,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
