"""Shared Vertex Batch API helpers for AskChem retrieval-upgrade pipelines.

Wraps Portkey's pass-through to Vertex AI's Batch API for Gemini.
Used by:
    scripts/summarize_papers.py        (Sprint 0)
    scripts/contextualize_claims.py    (Sprint 1)

Pricing (Gemini 3.1 Pro Preview Batch, ≤200K context, 2026-04):
    input  $1.00 / 1M tokens
    output $6.00 / 1M tokens

The pipeline is split into 5 idempotent steps so a long run survives
restarts:

    prepare  -> writes JSONL chunks to <pipeline_dir>/chunks/
    submit   -> uploads each chunk, creates a batch, records id in tracker.json
    status   -> polls batches, updates tracker.json
    collect  -> downloads finished batch outputs to <pipeline_dir>/outputs/
    apply    -> parses outputs into final per-row results (DB writeback is the
                caller's responsibility; this module only produces parsed rows)

Manifest (<pipeline_dir>/manifest.json) records what's IN each chunk so the
caller can map (custom_id -> domain object) on the way out.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator

# Portkey gateway (same config the existing arxiv batch pipeline uses).
GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

# Vertex Batch API limits we honour:
#  * <=100k requests per JSONL file
#  * <=2 GB per JSONL file
# We cap well below these so retries are quick.
MAX_REQUESTS_PER_CHUNK = 25_000
MAX_CHUNK_BYTES = 80 * 1024 * 1024     # 80 MB

PRICE_IN_PER_M = 1.00     # batch tier
PRICE_OUT_PER_M = 6.00


# ── HTTP helper (curl shellout matches batch_extract_arxiv.py) ───────────────


def _curl_json(method: str, path: str, *,
               data: dict | None = None,
               form_fields: dict | None = None,
               file_path: str | None = None,
               timeout: int = 600) -> dict:
    """Hit the Portkey gateway via curl. Mirrors src/batch_extract_arxiv.py."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")

    cmd = ["curl", "-s", "--max-time", str(timeout), "-X", method,
           "-H", f"x-portkey-api-key: {api_key}",
           "-H", f"x-portkey-provider: {PROVIDER}"]

    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]
    elif form_fields or file_path:
        bucket = os.environ.get("GCS_BKT", "")
        if bucket:
            cmd += ["-H", f"x-portkey-vertex-storage-bucket-name: {bucket}"]
        for k, v in (form_fields or {}).items():
            if k == "provider_file_name":
                cmd += ["-H", f"x-portkey-provider-file-name: {v}"]
            elif k == "provider_model":
                cmd += ["-H", f"x-portkey-provider-model: {v}"]
            else:
                cmd += ["--form", f'{k}="{v}"']
        if file_path:
            cmd += ["--form", f"file=@{file_path}"]

    cmd.append(GATEWAY + path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    if not result.stdout.strip():
        return {"error": "empty_response", "stderr": result.stderr[:200]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "parse_error", "raw": result.stdout[:500]}


# ── PREPARE ──────────────────────────────────────────────────────────────────


def make_request_line(custom_id: str, prompt: str, *,
                      max_tokens: int = 2000,
                      response_format: dict | None = None) -> str:
    """Build one JSONL request line for the Vertex Batch API."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "response_format": response_format or {"type": "json_object"},
    }
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }) + "\n"


def write_chunks(pipeline_dir: Path,
                 lines: Iterable[str],
                 *,
                 chunk_prefix: str,
                 max_requests: int = MAX_REQUESTS_PER_CHUNK,
                 max_bytes: int = MAX_CHUNK_BYTES) -> list[dict]:
    """Stream lines into chunked JSONL files. Returns chunk metadata list."""
    chunks_dir = pipeline_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    cur_lines: list[str] = []
    cur_bytes = 0
    cur_idx = 0

    def _flush():
        nonlocal cur_lines, cur_bytes, cur_idx
        if not cur_lines:
            return
        fname = f"{chunk_prefix}_{cur_idx:04d}.jsonl"
        fpath = chunks_dir / fname
        with open(fpath, "w") as fh:
            fh.writelines(cur_lines)
        size_mb = round(fpath.stat().st_size / 1e6, 2)
        files.append({"file": fname, "count": len(cur_lines), "size_mb": size_mb})
        print(f"  {fname}: {len(cur_lines):,} requests, {size_mb} MB", flush=True)
        cur_lines = []
        cur_bytes = 0
        cur_idx += 1

    for line in lines:
        nbytes = len(line.encode("utf-8"))
        if cur_lines and (len(cur_lines) >= max_requests or cur_bytes + nbytes > max_bytes):
            _flush()
        cur_lines.append(line)
        cur_bytes += nbytes
    _flush()
    return files


def write_manifest(pipeline_dir: Path, payload: dict) -> Path:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / "manifest.json"
    payload = dict(payload)
    payload.setdefault("model", MODEL)
    payload.setdefault("provider", PROVIDER)
    payload.setdefault("created_at", datetime.now().isoformat())
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


# ── SUBMIT ───────────────────────────────────────────────────────────────────


def submit_chunk(pipeline_dir: Path, fname: str) -> dict:
    """Upload one JSONL file and create a batch. Returns tracker entry dict."""
    fpath = pipeline_dir / "chunks" / fname
    if not fpath.exists():
        return {"status": "missing", "error": "file not found on disk"}

    upload_resp = _curl_json("POST", "/files",
        form_fields={
            "purpose": "batch",
            "provider_file_name": fname,
            "provider_model": MODEL,
        },
        file_path=str(fpath),
    )
    file_id = upload_resp.get("id")
    if not file_id:
        return {"status": "failed", "error": str(upload_resp)[:300]}

    batch_resp = _curl_json("POST", "/batches", data={
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "model": MODEL,
    })
    batch_id = batch_resp.get("id")
    return {
        "file_id": file_id,
        "batch_id": batch_id,
        "status": batch_resp.get("status", "unknown") if batch_id else "failed",
        "error": None if batch_id else str(batch_resp)[:300],
        "submitted_at": datetime.now().isoformat(),
    }


def submit_all(pipeline_dir: Path, *, workers: int = 3) -> dict:
    """Upload + submit all chunks that aren't already tracked. Returns summary."""
    manifest_path = pipeline_dir / "manifest.json"
    tracker_path = pipeline_dir / "tracker.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {pipeline_dir}; run prepare first")

    manifest = json.loads(manifest_path.read_text())
    tracker: dict = json.loads(tracker_path.read_text()) if tracker_path.exists() else {}

    todo = []
    for entry in manifest["files"]:
        fname = entry["file"]
        prev = tracker.get(fname)
        if prev and prev.get("batch_id") and prev.get("status") not in ("failed", "missing"):
            continue
        todo.append(entry)

    print(f"  to submit: {len(todo)}/{len(manifest['files'])} chunks", flush=True)
    if not todo:
        return {"submitted": 0, "skipped": len(manifest["files"])}

    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def _go(entry: dict):
        result = submit_chunk(pipeline_dir, entry["file"])
        with lock:
            tracker[entry["file"]] = result
            if result.get("batch_id"):
                counts["ok"] += 1
            else:
                counts["fail"] += 1
            if (counts["ok"] + counts["fail"]) % 5 == 0:
                tracker_path.write_text(json.dumps(tracker, indent=2))
                print(f"    {counts['ok']} ok / {counts['fail']} fail", flush=True)

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
        futs = [ex.submit(_go, e) for e in todo]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                print(f"    worker error: {exc}", flush=True)

    tracker_path.write_text(json.dumps(tracker, indent=2))
    return {"submitted": counts["ok"], "failed": counts["fail"]}


# ── STATUS ───────────────────────────────────────────────────────────────────


def poll_all(pipeline_dir: Path) -> dict:
    """Poll Vertex for every tracked batch; update tracker.json. Returns tally."""
    tracker_path = pipeline_dir / "tracker.json"
    if not tracker_path.exists():
        return {"error": "no tracker.json"}
    tracker = json.loads(tracker_path.read_text())
    tally: dict[str, int] = {}

    for fname, info in tracker.items():
        bid = info.get("batch_id")
        if not bid:
            tally["no_batch_id"] = tally.get("no_batch_id", 0) + 1
            continue
        resp = _curl_json("GET", f"/batches/{bid}")
        new_status = resp.get("status", "unknown")
        counts = resp.get("request_counts", {}) or {}
        info["status"] = new_status
        info["request_counts"] = counts
        tally[new_status] = tally.get(new_status, 0) + 1
        completed = counts.get("completed") or 0
        total = counts.get("total") or 0
        print(f"  {fname}: {new_status} ({completed}/{total})", flush=True)
        time.sleep(0.4)

    tracker_path.write_text(json.dumps(tracker, indent=2))
    return tally


# ── COLLECT ──────────────────────────────────────────────────────────────────


def _collect_one(batch_id: str) -> str | None:
    """Download the output JSONL for one finished batch."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    cmd = ["curl", "-s", "--max-time", "300", "-X", "GET",
           "-H", f"x-portkey-api-key: {api_key}",
           "-H", f"x-portkey-provider: {PROVIDER}",
           f"{GATEWAY}/batches/{batch_id}/output"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
        if result.stdout.strip():
            return result.stdout
    except Exception:
        return None
    return None


def collect_all(pipeline_dir: Path, *, workers: int = 6) -> dict:
    """Download outputs for every completed batch that hasn't been collected yet."""
    tracker_path = pipeline_dir / "tracker.json"
    if not tracker_path.exists():
        return {"error": "no tracker.json"}
    tracker = json.loads(tracker_path.read_text())

    out_dir = pipeline_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    todo = [(fname, info) for fname, info in tracker.items()
            if info.get("status") == "completed"
            and info.get("batch_id")
            and not info.get("collected")]

    if not todo:
        return {"collected": 0, "note": "nothing new to collect"}

    print(f"  collecting {len(todo)} batches", flush=True)
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def _go(fname: str, info: dict):
        raw = _collect_one(info["batch_id"])
        if raw is None:
            with lock:
                counts["fail"] += 1
            return
        out_path = out_dir / fname
        out_path.write_text(raw)
        with lock:
            info["collected"] = True
            info["collected_at"] = datetime.now().isoformat()
            counts["ok"] += 1
            if counts["ok"] % 5 == 0:
                tracker_path.write_text(json.dumps(tracker, indent=2))

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
        futs = [ex.submit(_go, f, i) for f, i in todo]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                print(f"    worker error: {exc}", flush=True)

    tracker_path.write_text(json.dumps(tracker, indent=2))
    return {"collected": counts["ok"], "failed": counts["fail"]}


def iter_output_rows(pipeline_dir: Path) -> Iterator[tuple[str, dict | None, dict]]:
    """Iterate (custom_id, parsed_json_or_None, raw_item_dict) for every line in
    every output file. parsed_json_or_None is None iff the LLM returned
    something we couldn't parse as JSON.
    """
    out_dir = pipeline_dir / "outputs"
    if not out_dir.exists():
        return
    for ofile in sorted(out_dir.glob("*.jsonl")):
        for line in ofile.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = item.get("custom_id", "") or ""
            response = item.get("response") or {}
            body = response.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                yield cid, None, item
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                yield cid, None, item
                continue
            yield cid, parsed, item


# ── synchronous fallback (used for dry-runs) ─────────────────────────────────


def call_sync(prompt: str, *, max_tokens: int = 2000, retries: int = 3) -> dict:
    """Synchronous Gemini 3.1 Pro call via Portkey.

    Used for dry-runs and quality-spotchecks before paying for a batch. Same
    model and gateway as the batch path, but at standard pricing
    (~$2 input, $12 output per 1M tokens).
    """
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "x-portkey-api-key": api_key,
        "x-portkey-provider": PROVIDER,
        "Content-Type": "application/json",
    }
    import requests
    last_err = ""
    for attempt in range(retries):
        try:
            r = requests.post(f"{GATEWAY}/chat/completions", headers=headers,
                              json=body, timeout=300)
        except Exception as e:
            last_err = f"network: {e}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            last_err = f"http {r.status_code}: {r.text[:200]}"
            time.sleep(2 ** attempt)
            continue
        try:
            resp = r.json()
        except Exception as e:
            last_err = f"json: {e}"
            time.sleep(2 ** attempt)
            continue
        choices = resp.get("choices") or []
        if not choices:
            last_err = f"no choices: {json.dumps(resp)[:200]}"
            time.sleep(2 ** attempt)
            continue
        content = (choices[0].get("message") or {}).get("content") or ""
        usage = resp.get("usage") or {}
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            last_err = f"content not JSON: {e}: {content[:200]}"
            time.sleep(2 ** attempt)
            continue
        return {"parsed": parsed, "usage": usage,
                "finish_reason": choices[0].get("finish_reason")}
    raise RuntimeError(f"sync Gemini call failed after {retries} retries: {last_err}")


__all__ = [
    "MODEL", "PROVIDER", "GATEWAY",
    "PRICE_IN_PER_M", "PRICE_OUT_PER_M",
    "MAX_REQUESTS_PER_CHUNK", "MAX_CHUNK_BYTES",
    "make_request_line", "write_chunks", "write_manifest",
    "submit_all", "poll_all", "collect_all",
    "iter_output_rows", "call_sync",
]
