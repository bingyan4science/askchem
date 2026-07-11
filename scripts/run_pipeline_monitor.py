"""Monitor the deep-extraction pipeline and auto-relaunch the Gemini
extractor so it picks up newly-downloaded PDFs.

Strategy:
  * Every CHECK_EVERY seconds, snapshot:
      - downloaders running?
      - audit running?
      - extractor running?
      - PDFs on disk for jobs file
      - deep_results count
  * If extractor is NOT running AND there are jobs whose PDF is on disk
    but no result yet, relaunch the extractor.
  * Print a single concise line each cycle, plus a one-line summary on
    state change.

Run it with nohup, e.g.:
  nohup env PORTKEY_API_KEY=... S2_API_KEY=... \
    python3 scripts/run_pipeline_monitor.py \
    --jobs data/audits/all_extract_jobs.jsonl \
    --workers 6 \
    > data/gemini_extract_logs/monitor.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers_full"
RESULTS_DIR = REPO_ROOT / "data" / "deep_results"
LOG_DIR = REPO_ROOT / "data" / "gemini_extract_logs"

CHECK_EVERY = 120  # seconds
EXTRACTOR_CMD = ["python3", "src/extract_gemini_batch.py"]


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def proc_running(needle: str) -> int:
    """Return PID if a process matching ``needle`` is running, else 0."""
    try:
        out = subprocess.check_output(["pgrep", "-f", needle], text=True)
    except subprocess.CalledProcessError:
        return 0
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return 0


def looks_like_complete_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(8)
            if not head.startswith(b"%PDF"):
                return False
            f.seek(0, 2); sz = f.tell()
            f.seek(max(0, sz - 1024)); tail = f.read()
            return b"%%EOF" in tail
    except OSError:
        return False


def count_pending(jobs_path: Path) -> tuple[int, int, int, int]:
    """Returns (total_jobs, with_pdf, with_result, ready_to_extract)."""
    done = {f.stem for f in RESULTS_DIR.glob("*.json")} if RESULTS_DIR.exists() else set()
    total = 0
    with_pdf = 0
    ready = 0
    with_result = 0
    seen = set()
    with jobs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = d.get("doi") or d.get("DOI")
            if not doi or doi in seen:
                continue
            seen.add(doi)
            total += 1
            cid = doi_to_filename(doi)
            if cid in done:
                with_result += 1
                continue
            pdf = PAPERS_DIR / f"{cid}.pdf"
            if pdf.exists() and pdf.stat().st_size >= 10_000:
                with_pdf += 1
                if looks_like_complete_pdf(pdf):
                    ready += 1
    return total, with_pdf, with_result, ready


def launch_extractor(jobs_path: Path, workers: int,
                     shard: int = 0, total: int = 1) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_s{shard}of{total}" if total > 1 else ""
    log_path = LOG_DIR / f"run_{int(time.time())}{suffix}.log"
    cmd = EXTRACTOR_CMD + ["--jobs", str(jobs_path), "--workers", str(workers)]
    if total > 1:
        cmd += ["--shard", str(shard), "--total", str(total)]
    env = dict(os.environ)
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=log_path.open("a"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"  [{datetime.now():%H:%M:%S}] launched extractor pid={proc.pid} "
          f"shard={shard}/{total} -> {log_path.name}", flush=True)
    return proc.pid


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--every", type=int, default=CHECK_EVERY)
    p.add_argument("--shards", type=int, default=1,
                   help="Number of parallel extractor shards to launch.")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="Exit after N cycles (0 = run until ready==0)")
    args = p.parse_args()

    if not os.environ.get("PORTKEY_API_KEY"):
        print("ERROR: PORTKEY_API_KEY not set", file=sys.stderr)
        return 1

    jobs_path = Path(args.jobs).resolve()
    if not jobs_path.exists():
        print(f"ERROR: jobs file not found: {jobs_path}", file=sys.stderr)
        return 1

    cycle = 0
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] monitor started, "
          f"jobs={jobs_path.name}", flush=True)

    while True:
        cycle += 1
        total, with_pdf, with_result, ready = count_pending(jobs_path)
        downloaders = []
        for needle in ("scripts/download_pdfs.py",):
            pid = proc_running(needle)
            if pid:
                downloaders.append(pid)
        audit_pid = proc_running("scripts/audit_s2_chemistry_gap.py")
        ext_count = 0
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "src/extract_gemini_batch.py"], text=True)
            ext_count = sum(1 for x in out.split() if x.strip().isdigit())
        except subprocess.CalledProcessError:
            ext_count = 0

        print(
            f"[{datetime.now():%H:%M:%S}] cyc{cycle} | jobs={total:,} "
            f"results={with_result:,} pdfs={with_pdf:,} ready={ready:,} | "
            f"audit={'yes' if audit_pid else 'no'} "
            f"dlrs={len(downloaders)} ext={ext_count}/{args.shards}",
            flush=True,
        )

        # Only relaunch when fully idle, then bring up all shards together
        # (avoids collisions with a still-running --total=1 extractor).
        if ext_count == 0 and ready > 0:
            for shard_idx in range(args.shards):
                launch_extractor(jobs_path, args.workers,
                                 shard=shard_idx, total=args.shards)

        if ext_count == 0 and ready == 0 and not downloaders:
            print(f"[{datetime.now():%H:%M:%S}] all done — exiting", flush=True)
            break
        if args.max_cycles and cycle >= args.max_cycles:
            print(f"[{datetime.now():%H:%M:%S}] reached --max-cycles, exiting",
                  flush=True)
            break

        time.sleep(args.every)

    return 0


if __name__ == "__main__":
    sys.exit(main())
