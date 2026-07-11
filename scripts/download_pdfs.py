"""Generic concurrent PDF downloader.

Reads a JSONL of ``{doi, pdf_url}`` jobs (e.g. produced by
``scripts/prepare_pdf_jobs.py``) and downloads each PDF to
``data/papers_full/<sha256(doi)[:16]>.pdf`` — same convention as
``src/download_tier_a.py``.

Resumable: rows whose target file already exists with size ≥ 10 KB are
skipped.

Usage:
  python scripts/download_pdfs.py \
      --jobs data/audits/to_deep_extract.jobs.jsonl \
      --workers 12
  python scripts/download_pdfs.py --jobs ... --status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers_full"
LOG_DIR = REPO_ROOT / "data" / "download_logs"

REQUEST_TIMEOUT = 60
MAX_ATTEMPTS = 4
MIN_PDF_SIZE = 10_000
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "AskChem/1.0 (academic; mailto:research@example.org)"
)


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


_session_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_session_local, "s", None)
    if s is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf, */*;q=0.5",
        })
        # urllib3-level connection retries (separate from our app-level retries).
        retry = Retry(
            total=3, connect=3, read=2,
            backoff_factor=2.0,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
        s.mount("http://", adapter); s.mount("https://", adapter)
        _session_local.s = s
    return s


def _looks_like_complete_pdf(path: Path) -> bool:
    """Verify a PDF starts with %PDF and ends with %%EOF (last 1024 B)."""
    try:
        with path.open("rb") as f:
            head = f.read(8)
            if not head.startswith(b"%PDF"):
                return False
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - 1024))
            tail = f.read()
            return b"%%EOF" in tail
    except OSError:
        return False


def download_one(url: str, dest: Path) -> tuple[bool, str]:
    sess = _session()
    err = "no attempt"
    backoff = 4.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
            if r.status_code == 200:
                ct = r.headers.get("content-type", "").lower()
                if "pdf" not in ct and "octet-stream" not in ct:
                    # doi.org typically returns the publisher landing page (HTML).
                    return False, f"bad content-type: {ct[:40]}"
                expected_len = int(r.headers.get("content-length") or 0)
                bytes_written = 0
                try:
                    with dest.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                            bytes_written += len(chunk)
                except (requests.RequestException, OSError) as e:
                    err = f"stream-broke: {type(e).__name__}: {str(e)[:80]}"
                    if dest.exists():
                        dest.unlink()
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(backoff); backoff = min(backoff * 2, 60.0)
                    continue
                if bytes_written < MIN_PDF_SIZE:
                    if dest.exists():
                        dest.unlink()
                    return False, "tiny file"
                if expected_len and bytes_written < expected_len * 0.95:
                    err = f"short read: {bytes_written}/{expected_len}"
                    if dest.exists():
                        dest.unlink()
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(backoff); backoff = min(backoff * 2, 60.0)
                    continue
                if not _looks_like_complete_pdf(dest):
                    err = "corrupt: missing %%EOF"
                    if dest.exists():
                        dest.unlink()
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(backoff); backoff = min(backoff * 2, 60.0)
                    continue
                return True, "ok"
            if r.status_code in (403, 404, 410, 451):
                return False, f"http {r.status_code}"
            err = f"http {r.status_code}"
        except requests.RequestException as e:
            err = type(e).__name__ + ": " + str(e)[:120]
        except OSError as e:
            return False, f"oserror: {e}"

        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    return False, err


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--status", action="store_true")
    args = p.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(args.jobs) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows):,} jobs from {args.jobs}", flush=True)

    queue: list[dict] = []
    n_already = 0
    n_no_url = 0
    for r in rows:
        doi = r.get("doi")
        url = r.get("pdf_url")
        if not doi or not url:
            n_no_url += 1
            continue
        dest = PAPERS_DIR / f"{doi_to_filename(doi)}.pdf"
        if dest.exists() and dest.stat().st_size >= MIN_PDF_SIZE:
            n_already += 1
            continue
        queue.append(r)

    print(f"  already on disk : {n_already:,}", flush=True)
    print(f"  no doi/url      : {n_no_url:,}", flush=True)
    print(f"  to download     : {len(queue):,}", flush=True)

    if args.status:
        return 0

    if args.limit and args.limit > 0:
        queue = queue[: args.limit]
        print(f"  --limit applied → {len(queue):,}", flush=True)

    if not queue:
        print("nothing to download.", flush=True)
        return 0

    log_path = LOG_DIR / f"download_{int(time.time())}.jsonl"
    log = log_path.open("w")
    log_lock = threading.Lock()

    progress = {"ok": 0, "fail": 0, "started": time.time()}
    progress_lock = threading.Lock()

    def _do(row: dict) -> None:
        doi = row["doi"]; url = row["pdf_url"]
        dest = PAPERS_DIR / f"{doi_to_filename(doi)}.pdf"
        ok, msg = download_one(url, dest)
        with log_lock:
            log.write(json.dumps({
                "doi": doi, "url": url, "ok": ok, "msg": msg,
                "ts": datetime.now().isoformat(),
            }) + "\n")
        with progress_lock:
            progress["ok" if ok else "fail"] += 1
            done = progress["ok"] + progress["fail"]
            if done % 50 == 0 or done == len(queue):
                el = time.time() - progress["started"]
                rate = done / max(el, 1)
                rem = (len(queue) - done) / max(rate, 0.01)
                print(
                    f"  [{done}/{len(queue)}] ok={progress['ok']} fail={progress['fail']} "
                    f"| {rate:.1f}/s | ETA {rem/60:.0f}m", flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_do, r) for r in queue]
        for _ in as_completed(futs):
            pass

    log.close()
    el = time.time() - progress["started"]
    print(f"\nDONE in {el/60:.1f} min", flush=True)
    print(f"  ok   : {progress['ok']:,}", flush=True)
    print(f"  fail : {progress['fail']:,}", flush=True)
    print(f"  log  : {log_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
