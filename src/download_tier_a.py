"""
Download Tier A PDFs (2020+, 100+ citations, OA available).

Reads from data/oa_scan.json, downloads PDFs to data/papers_full/,
using the same filename convention as deep_extract.py: sha256(doi)[:16].pdf

Usage:
    python src/download_tier_a.py                # Download all Tier A
    python src/download_tier_a.py --max 100      # Download top 100 only
    python src/download_tier_a.py --workers 15   # 15 concurrent workers
    python src/download_tier_a.py --status       # Show progress
"""

import argparse
import hashlib
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers_full"
OA_SCAN = DATA_DIR / "oa_scan.json"
CHECKPOINT_FILE = DATA_DIR / "tier_a_download_checkpoint.json"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 2
DEFAULT_WORKERS = 10
MIN_PDF_SIZE = 10_000


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


_session_local = threading.local()

def _get_session() -> requests.Session:
    if not hasattr(_session_local, 'session'):
        s = requests.Session()
        s.headers.update({"User-Agent": "AskChem/1.0 (academic research)"})
        _session_local.session = s
    return _session_local.session


def download_pdf(url: str, dest_path: Path) -> bool:
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get('content-type', '')
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    return False
                with open(dest_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                if dest_path.stat().st_size < MIN_PDF_SIZE:
                    dest_path.unlink()
                    return False
                return True
            elif resp.status_code in (403, 404, 410, 451):
                return False
            else:
                time.sleep(1 * (attempt + 1))
        except (requests.RequestException, OSError):
            time.sleep(1 * (attempt + 1))
    return False


def _download_one(paper: dict) -> tuple:
    doi = paper['doi']
    filename = doi_to_filename(doi) + ".pdf"
    dest = PAPERS_DIR / filename
    if dest.exists() and dest.stat().st_size >= MIN_PDF_SIZE:
        return paper, True, filename, "already"
    ok = download_pdf(paper['pdf_url'], dest)
    return paper, ok, filename, "downloaded" if ok else "failed"


def load_tier_a(max_papers: int = None) -> list[dict]:
    if not OA_SCAN.exists():
        print(f"Error: {OA_SCAN} not found. Run scan_oa.py first.")
        sys.exit(1)
    with open(OA_SCAN) as f:
        scan = json.load(f)
    tier_a = [p for p in scan['papers']
              if p.get('year', 0) >= 2020 and p.get('citation_count', 0) >= 100]
    tier_a.sort(key=lambda p: p['citation_count'], reverse=True)
    if max_papers:
        tier_a = tier_a[:max_papers]
    return tier_a


def cmd_status():
    if not CHECKPOINT_FILE.exists():
        print("No checkpoint file. No downloads started yet.")
        return
    with open(CHECKPOINT_FILE) as f:
        cp = json.load(f)
    downloaded = cp.get("downloaded", {})
    failed = cp.get("failed", {})
    print(f"Downloaded: {len(downloaded)}")
    print(f"Failed: {len(failed)}")
    on_disk = sum(1 for f in PAPERS_DIR.glob("*.pdf")
                  if f.stat().st_size >= MIN_PDF_SIZE) if PAPERS_DIR.exists() else 0
    print(f"PDFs on disk (>=10KB): {on_disk}")


def main():
    parser = argparse.ArgumentParser(description="Download Tier A PDFs")
    parser.add_argument("--max", type=int, help="Max papers to download")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_tier_a(args.max)
    print(f"Tier A papers to download: {len(papers)}")

    existing = set()
    if PAPERS_DIR.exists():
        for f in PAPERS_DIR.iterdir():
            if f.suffix == '.pdf' and f.stat().st_size >= MIN_PDF_SIZE:
                existing.add(f.stem)

    to_download = [p for p in papers if doi_to_filename(p['doi']) not in existing]
    already = len(papers) - len(to_download)
    print(f"  Already on disk: {already}")
    print(f"  To download: {len(to_download)}")

    if not to_download:
        print("All Tier A PDFs already downloaded!")
        return

    checkpoint = {"downloaded": {}, "failed": {}, "started_at": datetime.now().isoformat()}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)

    downloaded = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_one, p): p for p in to_download}
        for future in as_completed(futures):
            paper, ok, filename, status = future.result()
            if ok:
                downloaded += 1
                checkpoint["downloaded"][paper['doi']] = {
                    "filename": filename, "time": datetime.now().isoformat()
                }
            else:
                failed += 1
                checkpoint["failed"][paper['doi']] = {
                    "url": paper['pdf_url'], "time": datetime.now().isoformat()
                }

            total_done = downloaded + failed
            if total_done % 50 == 0 or total_done == len(to_download):
                elapsed = time.time() - start_time
                rate = total_done / max(elapsed, 1)
                remaining = (len(to_download) - total_done) / max(rate, 0.01)
                print(f"  Progress: {total_done}/{len(to_download)} "
                      f"({downloaded} ok, {failed} failed) "
                      f"[{rate:.1f}/s, ~{remaining/60:.0f}m remaining]", flush=True)
                checkpoint["updated_at"] = datetime.now().isoformat()
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(checkpoint, f)

    checkpoint["updated_at"] = datetime.now().isoformat()
    checkpoint["completed_at"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

    on_disk = sum(1 for f in PAPERS_DIR.glob("*.pdf")
                  if f.stat().st_size >= MIN_PDF_SIZE)
    total_size = sum(f.stat().st_size for f in PAPERS_DIR.glob("*.pdf")) / 1e9

    print(f"\nDownload complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Total PDFs on disk: {on_disk} ({total_size:.2f} GB)")
    print(f"  Success rate: {downloaded/(downloaded+failed)*100:.1f}%")


if __name__ == "__main__":
    main()
