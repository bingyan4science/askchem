#!/usr/bin/env python3
"""Stage 3 — OA PDF harvest.

Reads ``data/ingestion_2026_05/discovered_papers.jsonl`` (after Stage 2.5
S2 enrichment), tries to download a PDF for each paper from the most
lenient OA source, and writes a manifest:

    data/harvest_2026_05_21/pdf_manifest.jsonl

Each manifest row carries ``doi``, ``source``, ``pdf_path`` (or null),
``failure_reason``, ``host``.

Source routing:
    arXiv DOIs           -> https://arxiv.org/pdf/{arxiv_id}.pdf
    ChemRxiv DOIs        -> chemrxiv.org item-API + download URI
    All others           -> openAccessPdf.url if present

Polite client posture (Cloudflare / rate-limit avoidance, not policy):
    - browser-class User-Agent
    - max 4 concurrent per host
    - exponential backoff on 403 / 429 / 5xx
    - reject HTML payloads (publisher landing pages) by content-type
    - reject PDFs > 50 MB
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import hashlib
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "data" / "ingestion_2026_05" / "discovered_papers.jsonl"
PAPERS_DIR = REPO_ROOT / "data" / "papers_full"
MANIFEST_PATH = REPO_ROOT / "data" / "harvest_2026_05_21" / "pdf_manifest.jsonl"

MAX_PDF_BYTES = 50 * 1024 * 1024
HOST_CONCURRENCY = 4
HOST_MIN_DELAY_S = 2.0
GLOBAL_WORKERS = 24
REQUEST_TIMEOUT_S = 60
MAX_ATTEMPTS = 3
BACKOFF_S = (5, 15, 60)

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
              "AppleWebKit/605.1.15 (KHTML, like Gecko) "
              "Version/17.4 Safari/605.1.15")

CHEMRXIV_DOI_API = (
    "https://chemrxiv.org/engage/api-gateway/chemrxiv/public/publication/doi/{doi}"
)


def doi_to_hash(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


# ── per-host rate limiter ────────────────────────────────────────────────────

class HostLimiter:
    """Per-host concurrency + minimum delay limiter."""

    def __init__(self, concurrency: int, min_delay: float):
        self._concurrency = concurrency
        self._min_delay = min_delay
        self._sems: dict[str, threading.Semaphore] = {}
        self._last_call: dict[str, float] = defaultdict(float)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._meta_lock = threading.Lock()

    def _sem(self, host: str) -> threading.Semaphore:
        with self._meta_lock:
            sem = self._sems.get(host)
            if sem is None:
                sem = threading.Semaphore(self._concurrency)
                self._sems[host] = sem
            return sem

    def acquire(self, host: str):
        self._sem(host).acquire()
        with self._locks[host]:
            wait = self._min_delay - (time.time() - self._last_call[host])
            if wait > 0:
                time.sleep(wait + random.uniform(0, 0.4))
            self._last_call[host] = time.time()

    def release(self, host: str):
        self._sem(host).release()


# ── source-specific URL resolution ───────────────────────────────────────────

def _arxiv_id_from_paper(paper: dict) -> str | None:
    aid = paper.get("arxiv_id")
    if aid:
        return aid
    eids = paper.get("externalIds") or {}
    aid = eids.get("ArXiv")
    if aid:
        return aid
    doi = (paper.get("doi") or "").lower()
    if doi.startswith("arxiv:"):
        return doi[len("arxiv:"):]
    m = re.match(r"^10\.48550/arxiv\.(.+)$", doi)
    if m:
        return m.group(1)
    return None


def resolve_arxiv_url(paper: dict) -> str | None:
    aid = _arxiv_id_from_paper(paper)
    if not aid:
        return None
    aid = aid.replace("arXiv:", "").strip()
    return f"https://arxiv.org/pdf/{aid}.pdf"


def resolve_chemrxiv_url(paper: dict, session: requests.Session,
                         limiter: HostLimiter) -> tuple[str | None, str | None]:
    """Resolve a ChemRxiv item via item-API; return ``(url, failure_reason)``."""
    doi = paper.get("doi") or (paper.get("externalIds") or {}).get("DOI") or ""
    if not doi:
        return None, "no doi"
    api = CHEMRXIV_DOI_API.format(doi=doi)
    host = urlparse(api).netloc
    limiter.acquire(host)
    try:
        resp = session.get(api, timeout=REQUEST_TIMEOUT_S)
    except Exception as exc:
        return None, f"item-api error: {exc}"[:200]
    finally:
        limiter.release(host)
    if resp.status_code == 404:
        return None, "item-api 404"
    if resp.status_code != 200:
        return None, f"item-api HTTP {resp.status_code}"
    try:
        body = resp.json()
    except Exception:
        return None, "item-api non-json"
    main = body.get("mainArticle") or {}
    file_content = (main.get("fileContent") or {})
    uri = file_content.get("downloadUri")
    if not uri:
        return None, "no downloadUri"
    return uri, None


def resolve_oa_url(paper: dict) -> str | None:
    oa = paper.get("openAccessPdf") or {}
    url = oa.get("url")
    if url and url.lower().endswith((".pdf",)) or (url and "pdf" in url.lower()):
        return url
    return url or None


# ── download core ────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, session: requests.Session,
              limiter: HostLimiter) -> tuple[bool, str]:
    """Stream URL to ``dest``. Returns ``(ok, reason)``."""
    host = urlparse(url).netloc
    if not host:
        return False, "bad url"
    for attempt in range(MAX_ATTEMPTS):
        limiter.acquire(host)
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_S, stream=True,
                               allow_redirects=True)
        except Exception as exc:
            limiter.release(host)
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
                continue
            return False, f"net error: {exc}"[:200]

        try:
            if resp.status_code in (403, 429, 502, 503, 504):
                wait = int(resp.headers.get("Retry-After") or
                           BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
                resp.close()
                limiter.release(host)
                if attempt + 1 < MAX_ATTEMPTS:
                    time.sleep(wait)
                    continue
                return False, f"HTTP {resp.status_code}"
            if resp.status_code != 200:
                code = resp.status_code
                resp.close()
                return False, f"HTTP {code}"

            ct = (resp.headers.get("content-type") or "").lower()
            if "pdf" not in ct and "octet" not in ct:
                resp.close()
                return False, f"content-type {ct[:60]!r}"
            cl = resp.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > MAX_PDF_BYTES:
                resp.close()
                return False, f"too large {int(cl)}"

            dest.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_PDF_BYTES:
                        dest.unlink(missing_ok=True)
                        return False, "exceeded 50MB during stream"
                    fh.write(chunk)
            if written == 0:
                dest.unlink(missing_ok=True)
                return False, "0 bytes"
            with dest.open("rb") as fh:
                head = fh.read(8)
            if not head.startswith(b"%PDF"):
                dest.unlink(missing_ok=True)
                return False, "not a PDF (magic bytes)"
            return True, "ok"
        finally:
            try:
                resp.close()
            except Exception:
                pass
            limiter.release(host)
    return False, "exhausted retries"


# ── per-paper worker ─────────────────────────────────────────────────────────

def fetch_one(paper: dict, session: requests.Session,
              limiter: HostLimiter) -> dict:
    src = (paper.get("source") or "").lower()
    doi = (paper.get("externalIds") or {}).get("DOI") or paper.get("doi") or ""
    if not doi:
        return {"doi": doi, "source": src, "pdf_path": None,
                "failure_reason": "no DOI", "host": None}
    dest = PAPERS_DIR / (doi_to_hash(doi) + ".pdf")
    # Reuse PDF if it's already on disk from a prior run
    if dest.exists() and dest.stat().st_size > 0 and dest.stat().st_size <= MAX_PDF_BYTES:
        return {"doi": doi, "source": src, "pdf_path": str(dest),
                "failure_reason": "cached", "host": "local"}

    url = None
    host = None
    chemrxiv_fail = None
    if src.startswith("arxiv"):
        url = resolve_arxiv_url(paper)
        if not url:
            return {"doi": doi, "source": src, "pdf_path": None,
                    "failure_reason": "arxiv id missing", "host": None}
    elif src == "chemrxiv":
        url, chemrxiv_fail = resolve_chemrxiv_url(paper, session, limiter)
        if not url:
            return {"doi": doi, "source": src, "pdf_path": None,
                    "failure_reason": chemrxiv_fail or "chemrxiv resolve failed",
                    "host": "chemrxiv.org"}
    else:
        url = resolve_oa_url(paper)
        if not url:
            # Fall back to arXiv if the paper happens to be cross-listed
            # (S2-enriched journal papers sometimes carry an ArXiv id).
            arxiv_url = resolve_arxiv_url(paper)
            if arxiv_url:
                url = arxiv_url
            else:
                return {"doi": doi, "source": src, "pdf_path": None,
                        "failure_reason": "no openAccessPdf.url", "host": None}

    host = urlparse(url).netloc
    ok, reason = _download(url, dest, session, limiter)
    return {
        "doi": doi,
        "source": src,
        "pdf_path": str(dest) if ok else None,
        "failure_reason": None if ok else reason,
        "host": host,
        "url": url,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            doi = r.get("doi")
            if doi:
                done.add(doi.lower())
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_JSONL))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--workers", type=int, default=GLOBAL_WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N papers (0 = all).")
    ap.add_argument("--user-agent", default=os.environ.get("HARVEST_USER_AGENT",
                                                            DEFAULT_UA))
    args = ap.parse_args()

    in_path = Path(args.input)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"== OA PDF harvest ==")
    print(f"  input    : {in_path}")
    print(f"  manifest : {manifest_path}")
    print(f"  workers  : {args.workers}")

    if not in_path.exists():
        print(f"missing input: {in_path}", file=sys.stderr)
        return 1

    papers: list[dict] = []
    with in_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))

    if args.limit > 0:
        papers = papers[:args.limit]
    print(f"  papers   : {len(papers):,}")

    done = load_done(manifest_path)
    todo = [p for p in papers if (
        (p.get("externalIds") or {}).get("DOI") or p.get("doi") or ""
    ).lower() not in done]
    print(f"  skipping {len(papers) - len(todo)} already in manifest")
    print(f"  to fetch : {len(todo):,}")

    limiter = HostLimiter(HOST_CONCURRENCY, HOST_MIN_DELAY_S)
    session = requests.Session()
    session.headers.update({
        "User-Agent": args.user_agent,
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    })

    stats: dict[str, int] = defaultdict(int)
    by_source_ok: dict[str, int] = defaultdict(int)
    by_source_fail: dict[str, int] = defaultdict(int)

    started = time.time()
    written = 0
    cf403_by_host: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
    paused_hosts: set[str] = set()
    paused_lock = threading.Lock()

    def _on_result(row: dict):
        nonlocal written
        with manifest_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        src = row.get("source") or "?"
        ok = row.get("pdf_path") is not None
        if ok:
            stats["ok"] += 1
            by_source_ok[src] += 1
        else:
            stats["fail"] += 1
            by_source_fail[src] += 1
            host = row.get("host") or ""
            reason = row.get("failure_reason") or ""
            if "403" in reason or "429" in reason:
                cf403_by_host[host].append(1)
                if (host == "chemrxiv.org"
                        and sum(cf403_by_host[host]) > 50
                        and sum(cf403_by_host[host]) / max(len(cf403_by_host[host]), 1) > 0.5):
                    with paused_lock:
                        paused_hosts.add(host)
        written += 1
        if written % 100 == 0:
            elapsed = time.time() - started
            rate = written / max(elapsed, 1)
            print(f"  [{written}/{len(todo)}] ok={stats['ok']} fail={stats['fail']} "
                  f"rate={rate:.1f}/s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for p in todo:
            host_pref = ""
            src = (p.get("source") or "").lower()
            if src.startswith("arxiv"):
                host_pref = "arxiv.org"
            elif src == "chemrxiv":
                host_pref = "chemrxiv.org"
            else:
                url = (p.get("openAccessPdf") or {}).get("url") or ""
                host_pref = urlparse(url).netloc
            with paused_lock:
                if host_pref in paused_hosts:
                    _on_result({"doi": (p.get("externalIds") or {}).get("DOI") or p.get("doi"),
                                "source": src, "pdf_path": None,
                                "failure_reason": f"paused host {host_pref}",
                                "host": host_pref})
                    continue
            futures.append(ex.submit(fetch_one, p, session, limiter))
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:
                row = {"doi": None, "source": None, "pdf_path": None,
                       "failure_reason": f"worker error: {exc}", "host": None}
            _on_result(row)

    elapsed = time.time() - started
    print(f"\n== summary ==")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"  ok    : {stats['ok']}")
    print(f"  fail  : {stats['fail']}")
    print(f"  per-source ok:")
    for src in sorted(set(list(by_source_ok.keys()) + list(by_source_fail.keys()))):
        ok = by_source_ok.get(src, 0)
        fail = by_source_fail.get(src, 0)
        tot = ok + fail
        rate = (ok / tot * 100) if tot else 0
        print(f"    {src:24s} ok={ok:>5d} fail={fail:>5d} ({rate:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
