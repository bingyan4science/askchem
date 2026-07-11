#!/usr/bin/env python3
"""Re-submit Stage 4a batches with status='failed' serially (workers=1).

Bypasses the 3-worker submit which overloaded the Portkey gateway when each
upload is ~80-90 MB.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_extract_arxiv import _submit_one_file, pipeline_dir

TIER = 1
SLEEP_BETWEEN = 15  # seconds between sequential submits


def main() -> int:
    pdir = pipeline_dir(TIER)
    tracker_path = pdir / "tracker.json"
    manifest_path = pdir / "manifest.json"
    if not tracker_path.exists() or not manifest_path.exists():
        print("tracker/manifest missing", file=sys.stderr)
        return 1

    tracker = json.loads(tracker_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    size_by_file = {e["file"]: e.get("size_mb", 0) for e in manifest["files"]}

    to_retry = [
        (fname, info) for fname, info in tracker.items()
        if info.get("status") in ("failed", "missing")
    ]
    print(f"To retry: {len(to_retry)} files (serial, ~5s between)")

    ok = 0
    fail = 0
    for i, (fname, _info) in enumerate(to_retry, 1):
        fpath = pdir / fname
        if not fpath.exists():
            print(f"  [{i}/{len(to_retry)}] {fname}: file missing on disk, skip")
            tracker[fname] = {"status": "missing", "error": "file gone"}
            continue
        size = size_by_file.get(fname, 0)
        result = _submit_one_file(pdir, fname, size)
        tracker[fname] = result
        if result.get("batch_id"):
            ok += 1
            print(f"  [{i}/{len(to_retry)}] {fname} OK (batch_id={result['batch_id'][:20]}...)")
        else:
            fail += 1
            err = (result.get("error") or "")[:120]
            print(f"  [{i}/{len(to_retry)}] {fname} FAIL  {err}")

        if i % 5 == 0:
            tracker_path.write_text(json.dumps(tracker, indent=2))
        time.sleep(SLEEP_BETWEEN)

    tracker_path.write_text(json.dumps(tracker, indent=2))
    print(f"\nResubmit complete: {ok} ok, {fail} fail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
