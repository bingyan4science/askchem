"""
AskChem PDF Downloader — prioritized by citation count.

Downloads open-access PDFs for deep extraction, sorted by citation count
(highest first). Uses concurrent downloads for speed. Supports checkpoint/resume.

Usage:
    python src/download_pdfs.py                     # Download Tier 1 (top 5K)
    python src/download_pdfs.py --tier 2            # Download Tier 2 (next 20K)
    python src/download_pdfs.py --tier 3            # Download Tier 3 (all remaining)
    python src/download_pdfs.py --max 100           # Download top 100 only
    python src/download_pdfs.py --workers 20        # Use 20 concurrent workers
    python src/download_pdfs.py --status            # Show download progress
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
CHECKPOINT_FILE = DATA_DIR / "download_checkpoint.json"
CORPUS_FILE = DATA_DIR / "corpus_checkpoints"

TIER_LIMITS = {1: 5_000, 2: 25_000, 3: 80_000}
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2
DEFAULT_WORKERS = 10


def doi_to_filename(doi: str) -> str:
    """Convert DOI to a safe filename."""
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def load_papers_with_pdfs() -> list[dict]:
    """Load all papers that have open-access PDF URLs, sorted by citation count."""
    papers = []
    shard_dir = CORPUS_FILE
    if not shard_dir.exists():
        print(f"Error: corpus directory not found at {shard_dir}")
        sys.exit(1)

    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith('.jsonl'))
    print(f"Loading papers from {len(shards)} shards...", flush=True)

    seen_dois = set()
    for shard in shards:
        with open(shard_dir / shard) as f:
            for line in f:
                paper = json.loads(line)
                pdf_info = paper.get('openAccessPdf')
                if not isinstance(pdf_info, dict):
                    continue
                pdf_url = pdf_info.get('url', '')
                if not pdf_url:
                    continue

                doi = (paper.get('externalIds') or {}).get('DOI', '')
                if not doi or doi.lower() in seen_dois:
                    continue
                seen_dois.add(doi.lower())

                papers.append({
                    'doi': doi,
                    'title': paper.get('title', ''),
                    'year': paper.get('year', 0),
                    'citation_count': paper.get('citationCount', 0) or 0,
                    'pdf_url': pdf_url,
                    'venue': paper.get('venue', ''),
                })

    papers.sort(key=lambda p: p['citation_count'], reverse=True)
    print(f"Found {len(papers):,} papers with open-access PDFs", flush=True)
    return papers


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"downloaded": {}, "failed": {}, "started_at": datetime.now().isoformat()}


def save_checkpoint(checkpoint: dict):
    checkpoint["updated_at"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


_session_local = threading.local()

def _get_session() -> requests.Session:
    """Thread-local requests session for connection pooling."""
    if not hasattr(_session_local, 'session'):
        s = requests.Session()
        s.headers.update({"User-Agent": "AskChem/1.0 (academic research)"})
        _session_local.session = s
    return _session_local.session


def download_pdf(url: str, dest_path: Path) -> bool:
    """Download a PDF with retries. Uses thread-local session for connection reuse."""
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

                if dest_path.stat().st_size < 10_000:
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


def _download_one(paper: dict) -> tuple[dict, bool]:
    """Download a single paper's PDF. Returns (paper, success)."""
    doi = paper['doi']
    filename = doi_to_filename(doi) + ".pdf"
    dest = PAPERS_DIR / filename
    ok = download_pdf(paper['pdf_url'], dest)
    return paper, ok, filename


def main():
    parser = argparse.ArgumentParser(description="Download PDFs for AskChem deep extraction")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max", type=int, default=None, help="Max papers to download")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent download workers")
    parser.add_argument("--status", action="store_true", help="Show download progress")
    args = parser.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        cp = load_checkpoint()
        downloaded = len(cp.get("downloaded", {}))
        failed = len(cp.get("failed", {}))
        total_size = sum(
            os.path.getsize(PAPERS_DIR / f)
            for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')
        ) if PAPERS_DIR.exists() else 0
        print(f"Downloaded: {downloaded:,}")
        print(f"Failed: {failed:,}")
        print(f"Total size: {total_size / 1e9:.2f} GB")
        print(f"PDF directory: {PAPERS_DIR}")
        return

    papers = load_papers_with_pdfs()
    max_papers = args.max or TIER_LIMITS.get(args.tier, 5_000)
    target_papers = papers[:max_papers]

    print(f"\nTier {args.tier}: targeting {len(target_papers):,} papers "
          f"(min citations: {target_papers[-1]['citation_count'] if target_papers else 0})",
          flush=True)

    checkpoint = load_checkpoint()
    downloaded = checkpoint.get("downloaded", {})
    failed = checkpoint.get("failed", {})

    already = sum(1 for p in target_papers if p['doi'] in downloaded)
    remaining = [p for p in target_papers if p['doi'] not in downloaded and p['doi'] not in failed]
    print(f"Already downloaded: {already:,}, skipped (prev failed): {len(target_papers) - already - len(remaining):,}, "
          f"remaining: {len(remaining):,}", flush=True)
    print(f"Using {args.workers} concurrent workers\n", flush=True)

    success_count = 0
    fail_count = 0
    t0 = time.time()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_one, p): p for p in remaining}

        for future in as_completed(futures):
            paper, ok, filename = future.result()
            doi = paper['doi']

            with lock:
                if ok:
                    downloaded[doi] = {
                        "filename": filename,
                        "title": paper['title'][:100],
                        "citations": paper['citation_count'],
                        "downloaded_at": datetime.now().isoformat(),
                    }
                    success_count += 1
                else:
                    failed[doi] = {
                        "url": paper['pdf_url'],
                        "citations": paper['citation_count'],
                        "failed_at": datetime.now().isoformat(),
                    }
                    fail_count += 1

                total_done = success_count + fail_count
                if total_done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = total_done / elapsed
                    checkpoint["downloaded"] = downloaded
                    checkpoint["failed"] = failed
                    save_checkpoint(checkpoint)
                    print(f"  [{total_done:,}/{len(remaining):,}] "
                          f"OK: {success_count:,} | Fail: {fail_count:,} | "
                          f"{rate:.1f} papers/s | "
                          f"Latest: {paper['title'][:50]}...",
                          flush=True)

    checkpoint["downloaded"] = downloaded
    checkpoint["failed"] = failed
    save_checkpoint(checkpoint)

    elapsed = time.time() - t0
    total_downloaded = len(downloaded)
    total_size = sum(
        os.path.getsize(PAPERS_DIR / f)
        for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')
    )

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"This run: {success_count:,} downloaded, {fail_count:,} failed in {elapsed:.0f}s")
    print(f"Total PDFs: {total_downloaded:,}")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Success rate: {success_count / max(1, success_count + fail_count) * 100:.1f}%")
    print(f"Checkpoint: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
