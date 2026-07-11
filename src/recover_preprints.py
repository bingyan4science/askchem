"""
AskChem Preprint/PMC Recovery — find alternate PDFs for failed downloads.

Strategies:
  1. ArXiv: Download PDFs directly from arXiv using ArXiv IDs
  2. PMC OA: Query PMC OA service for downloadable tar.gz packages, extract PDFs

Usage:
    python src/recover_preprints.py scan              # Find recoverable papers
    python src/recover_preprints.py download           # Download ArXiv PDFs
    python src/recover_preprints.py download-pmc       # Download PMC OA packages
    python src/recover_preprints.py status             # Show progress
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers_full"
CHECKPOINT_FILE = DATA_DIR / "download_checkpoint.json"
RECOVERY_FILE = DATA_DIR / "preprint_recovery.json"
CORPUS_DIR = DATA_DIR / "corpus_checkpoints"

REQUEST_TIMEOUT = 30


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"downloaded": {}, "failed": {}}


def save_checkpoint(checkpoint: dict):
    checkpoint["updated_at"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_recovery() -> dict:
    if RECOVERY_FILE.exists():
        with open(RECOVERY_FILE) as f:
            return json.load(f)
    return {"alternates": {}, "pmc_oa_urls": {}, "scan_stats": {}}


def save_recovery(recovery: dict):
    recovery["updated_at"] = datetime.now().isoformat()
    with open(RECOVERY_FILE, "w") as f:
        json.dump(recovery, f, indent=2)


def get_on_disk_filenames() -> set[str]:
    if PAPERS_DIR.exists():
        return {f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')}
    return set()


def cmd_scan(args):
    """Scan corpus for papers with PMC/ArXiv IDs that we haven't downloaded yet."""
    on_disk = get_on_disk_filenames()
    print(f"PDFs already on disk: {len(on_disk)}", flush=True)

    shards = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith('.jsonl'))
    print(f"Scanning {len(shards)} corpus shards...", flush=True)

    min_citations = args.min_citations
    alternates = {}

    for shard in shards:
        with open(CORPUS_DIR / shard) as f:
            for line in f:
                paper = json.loads(line)
                doi = (paper.get("externalIds") or {}).get("DOI", "")
                if not doi:
                    continue

                citations = paper.get("citationCount", 0) or 0
                if citations < min_citations:
                    continue

                fname = doi_to_filename(doi) + ".pdf"
                if fname in on_disk:
                    continue

                ext = paper.get("externalIds") or {}
                title = paper.get("title", "")[:100]

                arxiv_id = ext.get("ArXiv")
                pmc_id = ext.get("PubMedCentral")

                if arxiv_id:
                    alternates[doi] = {
                        "source": "arxiv",
                        "id": arxiv_id,
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "title": title,
                        "citations": citations,
                    }
                elif pmc_id:
                    alternates[doi] = {
                        "source": "pmc",
                        "id": pmc_id,
                        "pdf_url": "",  # will be resolved via OA service
                        "title": title,
                        "citations": citations,
                    }

    from collections import Counter
    sources = Counter(v["source"] for v in alternates.values())

    recovery = load_recovery()
    recovery["alternates"] = alternates
    recovery["scan_stats"] = {
        "min_citations": min_citations,
        "total_found": len(alternates),
        "by_source": dict(sources),
        "scanned_at": datetime.now().isoformat(),
    }
    save_recovery(recovery)

    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Papers with alternate sources (>={min_citations} citations):")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")
    print(f"Total: {len(alternates)}")
    print(f"Saved to: {RECOVERY_FILE}")


# --------------- ArXiv download ---------------

_session_local = threading.local()

def _get_session() -> requests.Session:
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        s.headers.update({
            "User-Agent": "AskChem/1.0 (academic research; mailto:askchem@mit.edu)",
        })
        _session_local.session = s
    return _session_local.session


def download_arxiv_pdf(url: str, dest_path: Path) -> bool:
    session = _get_session()
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type or "octet-stream" in content_type:
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                    if dest_path.stat().st_size < 10_000:
                        dest_path.unlink()
                        return False
                    return True
                content = resp.content
                if content[:5] == b"%PDF-":
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    if len(content) < 10_000:
                        dest_path.unlink()
                        return False
                    return True
                return False
            elif resp.status_code in (403, 404, 410, 451):
                return False
            else:
                time.sleep(2 * (attempt + 1))
        except (requests.RequestException, OSError):
            time.sleep(2 * (attempt + 1))
    return False


def _download_arxiv_one(doi: str, info: dict) -> tuple[str, dict, bool, str]:
    filename = doi_to_filename(doi) + ".pdf"
    dest = PAPERS_DIR / filename
    ok = download_arxiv_pdf(info["pdf_url"], dest)
    return doi, info, ok, filename


def cmd_download(args):
    """Download ArXiv PDFs found during scan."""
    recovery = load_recovery()
    alternates = recovery.get("alternates", {})

    arxiv_papers = {d: i for d, i in alternates.items() if i["source"] == "arxiv"}
    if not arxiv_papers:
        print("No ArXiv papers found. Run 'scan' first.")
        return

    on_disk = get_on_disk_filenames()
    to_download = {
        doi: info for doi, info in arxiv_papers.items()
        if doi_to_filename(doi) + ".pdf" not in on_disk
    }

    if args.max:
        items = sorted(to_download.items(), key=lambda x: x[1].get("citations", 0), reverse=True)
        to_download = dict(items[:args.max])

    print(f"ArXiv papers to download: {len(to_download)}", flush=True)
    print(f"Using {args.workers} concurrent workers\n", flush=True)

    success_count = 0
    fail_count = 0
    t0 = time.time()
    lock = threading.Lock()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    downloaded = checkpoint.get("downloaded", {})

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_arxiv_one, doi, info): doi
                   for doi, info in to_download.items()}

        for future in as_completed(futures):
            doi, info, ok, filename = future.result()
            with lock:
                if ok:
                    downloaded[doi] = {
                        "filename": filename,
                        "title": info.get("title", "")[:100],
                        "citations": info.get("citations", 0),
                        "source": "arxiv",
                        "downloaded_at": datetime.now().isoformat(),
                    }
                    success_count += 1
                else:
                    fail_count += 1

                total_done = success_count + fail_count
                if total_done % 25 == 0:
                    elapsed = time.time() - t0
                    rate = total_done / max(0.1, elapsed)
                    checkpoint["downloaded"] = downloaded
                    save_checkpoint(checkpoint)
                    print(f"  [{total_done}/{len(to_download)}] "
                          f"OK: {success_count} | Fail: {fail_count} | "
                          f"{rate:.1f}/s",
                          flush=True)

    checkpoint["downloaded"] = downloaded
    save_checkpoint(checkpoint)

    on_disk_now = get_on_disk_filenames()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ARXIV DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Recovered: {success_count}, failed: {fail_count} ({elapsed:.0f}s)")
    print(f"Total PDFs on disk: {len(on_disk_now)}")


# --------------- PMC OA download ---------------

def _resolve_one_pmc(pmc_id: str, session: requests.Session) -> tuple[str, str | None]:
    """Query PMC OA service for a single ID. Returns (pmc_id, https_url or None)."""
    try:
        resp = session.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            params={"id": f"PMC{pmc_id}"},
            timeout=15,
        )
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            for record in root.findall(".//record"):
                for link in record.findall("link"):
                    fmt = link.get("format", "")
                    href = link.get("href", "")
                    if fmt == "tgz" and href:
                        return pmc_id, href.replace(
                            "ftp://ftp.ncbi.nlm.nih.gov",
                            "https://ftp.ncbi.nlm.nih.gov"
                        )
    except (requests.RequestException, ET.ParseError):
        pass
    return pmc_id, None


def resolve_pmc_oa_urls(pmc_ids: list[str], workers: int = 8) -> dict[str, str]:
    """Query PMC OA service for FTP/HTTPS download URLs. Returns {pmc_id: https_url}."""
    resolved = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "AskChem/1.0 (academic research)"})

    checked = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_resolve_one_pmc, pid, session): pid for pid in pmc_ids}

        for future in as_completed(futures):
            pmc_id, url = future.result()
            if url:
                resolved[pmc_id] = url
            checked += 1
            if checked % 500 == 0:
                elapsed = time.time() - t0
                rate = checked / max(0.1, elapsed)
                print(f"  Resolved {checked}/{len(pmc_ids)}, "
                      f"found {len(resolved)} OA packages ({rate:.1f}/s)", flush=True)

    return resolved


def download_pmc_package(url: str, dest_path: Path) -> bool:
    """Download a PMC tar.gz package and extract the PDF."""
    session = _get_session()
    try:
        resp = session.get(url, timeout=60, stream=True)
        if resp.status_code != 200:
            return False

        content = resp.content
        if len(content) < 1000:
            return False

        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            pdf_members = [m for m in tar.getmembers() if m.name.lower().endswith(".pdf")]
            if not pdf_members:
                return False
            # Pick the largest PDF (usually the main article)
            pdf_member = max(pdf_members, key=lambda m: m.size)
            pdf_data = tar.extractfile(pdf_member)
            if pdf_data is None:
                return False
            with open(dest_path, "wb") as f:
                f.write(pdf_data.read())
            if dest_path.stat().st_size < 10_000:
                dest_path.unlink()
                return False
            return True
    except (requests.RequestException, tarfile.TarError, OSError):
        if dest_path.exists():
            dest_path.unlink()
        return False


def _download_pmc_one(doi: str, info: dict, pmc_url: str) -> tuple[str, dict, bool, str]:
    filename = doi_to_filename(doi) + ".pdf"
    dest = PAPERS_DIR / filename
    ok = download_pmc_package(pmc_url, dest)
    return doi, info, ok, filename


def cmd_download_pmc(args):
    """Resolve PMC OA URLs and download packages."""
    recovery = load_recovery()
    alternates = recovery.get("alternates", {})

    pmc_papers = {d: i for d, i in alternates.items() if i["source"] == "pmc"}
    if not pmc_papers:
        print("No PMC papers found. Run 'scan' first.")
        return

    on_disk = get_on_disk_filenames()
    to_resolve = {
        doi: info for doi, info in pmc_papers.items()
        if doi_to_filename(doi) + ".pdf" not in on_disk
    }

    if args.max:
        items = sorted(to_resolve.items(), key=lambda x: x[1].get("citations", 0), reverse=True)
        to_resolve = dict(items[:args.max])

    print(f"PMC papers to check: {len(to_resolve)}", flush=True)

    # Step 1: Resolve which PMC IDs have OA packages
    pmc_ids = [info["id"] for info in to_resolve.values()]
    print(f"\nResolving PMC OA availability for {len(pmc_ids)} papers...", flush=True)
    pmc_urls = resolve_pmc_oa_urls(pmc_ids)
    print(f"Found {len(pmc_urls)} downloadable OA packages", flush=True)

    # Save resolved URLs for future use
    recovery["pmc_oa_urls"] = pmc_urls
    save_recovery(recovery)

    # Map back to DOIs
    pmc_id_to_doi = {info["id"]: doi for doi, info in to_resolve.items()}
    to_download = {}
    for pmc_id, url in pmc_urls.items():
        doi = pmc_id_to_doi.get(pmc_id)
        if doi and doi in to_resolve:
            to_download[doi] = (to_resolve[doi], url)

    print(f"\nDownloading {len(to_download)} PMC packages...", flush=True)
    print(f"Using {args.workers} concurrent workers\n", flush=True)

    success_count = 0
    fail_count = 0
    t0 = time.time()
    lock = threading.Lock()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    downloaded = checkpoint.get("downloaded", {})

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download_pmc_one, doi, info, url): doi
            for doi, (info, url) in to_download.items()
        }

        for future in as_completed(futures):
            doi, info, ok, filename = future.result()
            with lock:
                if ok:
                    downloaded[doi] = {
                        "filename": filename,
                        "title": info.get("title", "")[:100],
                        "citations": info.get("citations", 0),
                        "source": "pmc_oa",
                        "downloaded_at": datetime.now().isoformat(),
                    }
                    success_count += 1
                else:
                    fail_count += 1

                total_done = success_count + fail_count
                if total_done % 25 == 0:
                    elapsed = time.time() - t0
                    rate = total_done / max(0.1, elapsed)
                    checkpoint["downloaded"] = downloaded
                    save_checkpoint(checkpoint)
                    print(f"  [{total_done}/{len(to_download)}] "
                          f"OK: {success_count} | Fail: {fail_count} | "
                          f"{rate:.1f}/s",
                          flush=True)

    checkpoint["downloaded"] = downloaded
    save_checkpoint(checkpoint)

    on_disk_now = get_on_disk_filenames()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PMC DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Recovered: {success_count}, failed: {fail_count} ({elapsed:.0f}s)")
    print(f"Total PDFs on disk: {len(on_disk_now)}")


def cmd_status(args):
    recovery = load_recovery()
    stats = recovery.get("scan_stats", {})
    alternates = recovery.get("alternates", {})
    pmc_urls = recovery.get("pmc_oa_urls", {})

    if not stats:
        print("No scan run yet. Run: python src/recover_preprints.py scan")
        return

    on_disk = get_on_disk_filenames()
    recovered = sum(1 for doi in alternates if doi_to_filename(doi) + ".pdf" in on_disk)
    remaining = len(alternates) - recovered

    print(f"Scan ({stats.get('scanned_at', '?')}):")
    print(f"  Min citations: {stats.get('min_citations', '?')}")
    print(f"  Alternates found: {stats.get('total_found', 0)}")
    print(f"  Already on disk: {recovered}")
    print(f"  Remaining: {remaining}")
    print(f"\nBy source:")
    for src, count in sorted(stats.get("by_source", {}).items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")
    if pmc_urls:
        print(f"\nPMC OA resolved: {len(pmc_urls)} packages available")
    print(f"\nTotal PDFs on disk: {len(on_disk)}")
    if on_disk:
        total_size = sum(os.path.getsize(PAPERS_DIR / f) for f in on_disk)
        print(f"Total size: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Recover PDFs via ArXiv/PMC OA")
    sub = parser.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan", help="Scan corpus for alternate PDF sources")
    scan_p.add_argument("--min-citations", type=int, default=0)

    dl_p = sub.add_parser("download", help="Download ArXiv PDFs")
    dl_p.add_argument("--max", type=int, default=None)
    dl_p.add_argument("--workers", type=int, default=8)

    pmc_p = sub.add_parser("download-pmc", help="Download PMC OA packages")
    pmc_p.add_argument("--max", type=int, default=None)
    pmc_p.add_argument("--workers", type=int, default=5)

    sub.add_parser("status", help="Show recovery progress")

    args = parser.parse_args()
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "download-pmc":
        cmd_download_pmc(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
