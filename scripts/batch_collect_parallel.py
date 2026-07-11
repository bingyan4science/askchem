#!/usr/bin/env python3
"""Parallel Vertex Batch output collector with longer timeout.

The built-in ``src/batch_extract_arxiv.py collect`` uses a 120 s curl
``--max-time`` which times out on large output files (Vertex's /output
endpoint regularly takes 60-180 s to assemble the JSONL). This script
reuses the same tracker.json, fans out collects across 8 workers with a
300 s timeout, and reuses ``_parse_one_output`` from the original
script for the per-paper write step.

Usage::

    GCS_BKT=... PORTKEY_API_KEY=... python3 scripts/batch_collect_parallel.py --tier 1
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
sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse the proven parse function from the existing extractor.
from batch_extract_arxiv import _parse_one_output, RESULTS_DIR  # noqa: E402

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
DOWNLOAD_TIMEOUT_S = 300  # curl --max-time
SUBPROCESS_TIMEOUT_S = DOWNLOAD_TIMEOUT_S + 60


def _looks_like_html_error(text: str) -> bool:
    """Vertex /output sometimes returns a 502 / 503 Bad Gateway HTML page
    instead of the JSONL output. Treat those as transient failures."""
    head = (text or "")[:120].lstrip().lower()
    return head.startswith("<html") or head.startswith("<!doctype") or "bad gateway" in head


def download_output(batch_id: str, retries: int = 3) -> str | None:
    """GET /batches/{id}/output with a long timeout, retry on HTML/502."""
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = [
        "curl", "-sS", "--max-time", str(DOWNLOAD_TIMEOUT_S), "-X", "GET",
        "-H", f"x-portkey-api-key: {api_key}",
        "-H", f"x-portkey-provider: {PROVIDER}",
        f"{GATEWAY}/batches/{batch_id}/output",
    ]
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S
            )
        except Exception:
            time.sleep(min(2 ** attempt + 1, 30))
            continue
        body = proc.stdout
        if not body.strip():
            time.sleep(min(2 ** attempt + 1, 30))
            continue
        if _looks_like_html_error(body):
            time.sleep(min(2 ** attempt * 3 + 5, 60))
            continue
        return body
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pdir = REPO_ROOT / "data" / f"arxiv_batch_tier{args.tier}"
    tracker_path = pdir / "tracker.json"
    manifest_path = pdir / "manifest.json"
    output_dir = pdir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not tracker_path.exists():
        print(f"no tracker at {tracker_path}")
        return 1

    tracker: dict = json.loads(tracker_path.read_text())
    paper_dois: dict = {}
    if manifest_path.exists():
        paper_dois = json.loads(manifest_path.read_text()).get("paper_dois", {})

    to_collect = [
        (fname, info) for fname, info in tracker.items()
        if info.get("status") == "completed"
        and not info.get("collected")
        and info.get("batch_id")
    ]
    print(f"to collect: {len(to_collect)} batches; workers={args.workers}; "
          f"timeout={DOWNLOAD_TIMEOUT_S}s per request")
    if not to_collect:
        print("nothing to collect")
        return 0

    lock = Lock()
    stats = {"saved": 0, "failed": 0, "collected": 0, "download_fail": 0}

    def _work(fname: str, info: dict):
        raw = download_output(info["batch_id"])
        if raw is None:
            with lock:
                stats["download_fail"] += 1
            return ("fail", fname)

        (output_dir / fname).write_text(raw)
        saved, failed = _parse_one_output(raw, paper_dois)
        with lock:
            stats["saved"] += saved
            stats["failed"] += failed
            stats["collected"] += 1
            info["collected"] = True
            info["collected_at"] = datetime.now().isoformat()
            if stats["collected"] % 10 == 0:
                tracker_path.write_text(json.dumps(tracker, indent=2))
                print(f"  [{stats['collected']}/{len(to_collect)}] "
                      f"saved={stats['saved']} parse_fail={stats['failed']} "
                      f"dl_fail={stats['download_fail']}")
        return ("ok", fname)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_work, fname, info) for fname, info in to_collect]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                print(f"  worker error: {exc}")

    tracker_path.write_text(json.dumps(tracker, indent=2))
    elapsed = time.time() - started
    print(f"\ncollect complete in {elapsed/60:.1f} min:")
    print(f"  collected: {stats['collected']}/{len(to_collect)}")
    print(f"  papers saved: {stats['saved']}")
    print(f"  parse failures: {stats['failed']}")
    print(f"  download failures: {stats['download_fail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
