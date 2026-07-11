#!/usr/bin/env python3
"""Vertex Batch collector using the /files/{id}/content endpoint.

The ``/batches/{id}/output`` endpoint frequently 502s on large batches.
This collector instead:
  1. Calls ``/batches/{id}`` to get ``output_file_id`` (fast metadata).
  2. Streams ``/files/{output_file_id}/content`` to disk
     (Vertex prediction format; reliable, ~30 s per file).
  3. Parses the Vertex format line-by-line into per-paper JSON files
     at ``data/deep_results/{custom_id}.json``.

Vertex prediction format per line::

    {"requestId": "<custom_id>",
     "request": {...full input incl. PDF base64...},
     "response": {
         "candidates": [{"content": {"parts": [{"text": "...JSON..."}]}}],
         "usageMetadata": {...}
     }}

The JSON output text inside the response is what the original
extractor's prompt asked Gemini to return (``paper_knowledge`` +
``claims[]``). We strip the input and keep only the parsed response.

Usage::

    GCS_BKT=... PORTKEY_API_KEY=... python3 scripts/batch_collect_files.py --tier 1 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "deep_results"

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
META_TIMEOUT_S = 60
DOWNLOAD_TIMEOUT_S = 600  # for /files content; can be 50-300 MB
SUBPROCESS_TIMEOUT_S = DOWNLOAD_TIMEOUT_S + 60


def _api_curl(path: str, timeout_s: int = META_TIMEOUT_S,
              out_path: Path | None = None) -> bytes | None:
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = [
        "curl", "-sS", "--max-time", str(timeout_s), "-X", "GET",
        "-H", f"x-portkey-api-key: {api_key}",
        "-H", f"x-portkey-provider: {PROVIDER}",
        f"{GATEWAY}{path}",
    ]
    if out_path is not None:
        cmd += ["-o", str(out_path)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s + 60
        )
    except Exception:
        return None
    if out_path is not None:
        # Caller reads file separately.
        return b""
    return proc.stdout if proc.stdout else None


def _looks_html(text: str) -> bool:
    head = text[:120].lstrip().lower() if text else ""
    return head.startswith("<html") or head.startswith("<!doctype") or "bad gateway" in head


def get_output_file_id(batch_id: str) -> str | None:
    body = _api_curl(f"/batches/{batch_id}")
    if not body:
        return None
    try:
        d = json.loads(body)
    except Exception:
        return None
    return d.get("output_file_id")


def extract_paper_from_vertex_line(line: dict) -> tuple[str, dict | None]:
    """Parse one Vertex predictions.jsonl line into (custom_id, paper_result).

    Returns (cid, None) on any parse failure.
    """
    cid = line.get("requestId") or ""
    if not cid:
        return ("", None)
    resp = (line.get("response") or {})
    candidates = resp.get("candidates") or []
    if not candidates:
        return (cid, None)
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        return (cid, None)
    text = parts[0].get("text", "")
    if not text:
        return (cid, None)
    try:
        parsed = json.loads(text)
    except Exception:
        return (cid, None)
    usage = resp.get("usageMetadata") or {}
    return (cid, {
        "doi": "",  # filled by caller from paper_dois map
        "num_claims": len(parsed.get("claims") or []),
        "extraction_model": "gemini-3.1-pro-preview",
        "extraction_version": "deep_v1",
        "collected_at": datetime.now().isoformat(),
        "usage": {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        },
        "data": parsed,
    })


def collect_one(fname: str, info: dict, paper_dois: dict,
                out_dir: Path, lock: Lock, stats: dict) -> None:
    batch_id = info["batch_id"]
    ofid = info.get("output_file_id") or get_output_file_id(batch_id)
    if not ofid:
        with lock:
            stats["meta_fail"] += 1
        return
    info["output_file_id"] = ofid

    out_path = out_dir / fname
    _api_curl(f"/files/{ofid}/content",
              timeout_s=DOWNLOAD_TIMEOUT_S, out_path=out_path)
    if not out_path.exists() or out_path.stat().st_size < 1024:
        if out_path.exists():
            # Likely an HTML error page.
            head = out_path.read_bytes()[:200]
            if _looks_html(head.decode("utf-8", errors="ignore")):
                out_path.unlink()
        with lock:
            stats["download_fail"] += 1
        return

    saved = 0
    failed = 0
    with out_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except Exception:
                failed += 1
                continue
            cid, paper = extract_paper_from_vertex_line(line)
            if not cid or paper is None:
                failed += 1
                continue
            paper["doi"] = paper_dois.get(cid, "")
            result_path = RESULTS_DIR / f"{cid}.json"
            if result_path.exists():
                continue
            result_path.write_text(json.dumps(paper, ensure_ascii=False))
            saved += 1

    with lock:
        stats["saved"] += saved
        stats["parse_fail"] += failed
        stats["collected"] += 1
        info["collected"] = True
        info["collected_at"] = datetime.now().isoformat()
        if stats["collected"] % 5 == 0:
            print(f"  [{stats['collected']}/{stats['total']}] "
                  f"saved={stats['saved']} parse_fail={stats['parse_fail']} "
                  f"meta_fail={stats['meta_fail']} dl_fail={stats['download_fail']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pdir = REPO_ROOT / "data" / f"arxiv_batch_tier{args.tier}"
    tracker_path = pdir / "tracker.json"
    manifest_path = pdir / "manifest.json"
    out_dir = pdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tracker: dict = json.loads(tracker_path.read_text())
    paper_dois: dict = json.loads(manifest_path.read_text()).get("paper_dois", {}) if manifest_path.exists() else {}

    to_collect = [(fname, info) for fname, info in tracker.items()
                  if info.get("status") == "completed"
                  and not info.get("collected")
                  and info.get("batch_id")]
    print(f"to collect: {len(to_collect)} batches; workers={args.workers}")
    if not to_collect:
        print("nothing to collect")
        return 0

    lock = Lock()
    stats = {"saved": 0, "parse_fail": 0, "collected": 0,
             "meta_fail": 0, "download_fail": 0, "total": len(to_collect)}

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(collect_one, fname, info, paper_dois, out_dir, lock, stats)
                for fname, info in to_collect]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                print(f"  worker error: {exc}")

    tracker_path.write_text(json.dumps(tracker, indent=2))
    elapsed = time.time() - started
    print(f"\ndone in {elapsed/60:.1f} min:")
    print(f"  collected: {stats['collected']}/{stats['total']}")
    print(f"  papers saved: {stats['saved']}")
    print(f"  parse failures: {stats['parse_fail']}")
    print(f"  meta failures: {stats['meta_fail']}")
    print(f"  download failures: {stats['download_fail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
