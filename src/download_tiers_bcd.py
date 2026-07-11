"""
Download Tiers B, C, D PDFs (all non-Tier-A OA papers worth deep-extracting).

Tier B: 2020+, 20-99 citations
Tier C: pre-2020, 100-499 citations
Tier D: pre-2010, 500+ citations

Reads from data/oa_scan.json, downloads to data/papers_full/.

Usage:
    python src/download_tiers_bcd.py                # Download all B+C+D
    python src/download_tiers_bcd.py --tier B       # Tier B only
    python src/download_tiers_bcd.py --max 500      # Cap at 500 papers
    python src/download_tiers_bcd.py --workers 15
    python src/download_tiers_bcd.py --status
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

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers_full"
OA_SCAN = DATA_DIR / "oa_scan.json"
CHECKPOINT_FILE = DATA_DIR / "tiers_bcd_download_checkpoint.json"

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


def classify_tier(p: dict) -> str:
    year = p.get('year', 0) or 0
    cites = p.get('citation_count', 0) or 0
    if year >= 2020 and cites >= 100:
        return "A"
    if year >= 2020 and 20 <= cites < 100:
        return "B"
    if cites >= 500 and year < 2010:
        return "D"
    if 100 <= cites < 500 and year < 2020:
        return "C"
    return "other"


def load_papers(tiers: set, max_papers: int = None) -> list[dict]:
    if not OA_SCAN.exists():
        print(f"Error: {OA_SCAN} not found. Run scan_oa.py first.")
        sys.exit(1)
    with open(OA_SCAN) as f:
        scan = json.load(f)

    result = []
    for p in scan['papers']:
        t = classify_tier(p)
        if t in tiers:
            p['_tier'] = t
            result.append(p)

    result.sort(key=lambda p: p['citation_count'], reverse=True)
    if max_papers:
        result = result[:max_papers]
    return result


def main():
    parser = argparse.ArgumentParser(description="Download Tiers B/C/D PDFs")
    parser.add_argument("--tier", type=str, default="BCD",
                        help="Which tiers: B, C, D, BC, BCD (default: BCD)")
    parser.add_argument("--max", type=int, help="Max papers to download")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        if not CHECKPOINT_FILE.exists():
            print("No checkpoint file yet.")
            return
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        print(f"Downloaded: {len(cp.get('downloaded', {}))}")
        print(f"Failed: {len(cp.get('failed', {}))}")
        on_disk = sum(1 for f in PAPERS_DIR.glob("*.pdf")
                      if f.stat().st_size >= MIN_PDF_SIZE) if PAPERS_DIR.exists() else 0
        print(f"Total PDFs on disk: {on_disk}")
        return

    tiers = set(args.tier.upper())
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_papers(tiers, args.max)
    tier_counts = {}
    for p in papers:
        tier_counts[p['_tier']] = tier_counts.get(p['_tier'], 0) + 1
    print(f"Papers to download: {len(papers)}")
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]:,}")

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
        print("All PDFs already downloaded!")
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
                    "filename": filename, "tier": paper.get('_tier', '?'),
                    "time": datetime.now().isoformat()
                }
            else:
                failed += 1
                checkpoint["failed"][paper['doi']] = {
                    "url": paper['pdf_url'], "tier": paper.get('_tier', '?'),
                    "time": datetime.now().isoformat()
                }

            total_done = downloaded + failed
            if total_done % 200 == 0 or total_done == len(to_download):
                elapsed = time.time() - start_time
                rate = total_done / max(elapsed, 1)
                remaining = (len(to_download) - total_done) / max(rate, 0.01)
                print(f"  Progress: {total_done}/{len(to_download)} "
                      f"({downloaded} ok, {failed} failed) "
                      f"[{rate:.1f}/s, ~{remaining/60:.0f}m remaining]", flush=True)
                checkpoint["updated_at"] = datetime.now().isoformat()
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(checkpoint, f)

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
