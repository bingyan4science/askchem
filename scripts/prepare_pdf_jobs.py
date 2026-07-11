"""Resolve PDF URLs for an audit JSONL and write a downloadable job list.

Pipeline:
  1. Read input JSONL (e.g. ``data/audits/to_deep_extract.jsonl`` or
     ``data/audits/to_add.jsonl``).
  2. Skip rows whose PDF is already on disk (``data/papers_full/<sha>.pdf``).
  3. For each remaining row, look up an open-access PDF URL — first in
     ``data/oa_scan.json`` (cheap local lookup), then via the Semantic
     Scholar batch endpoint as a fallback.
  4. Write a job JSONL of ``{doi, pdf_url, ...}`` ready for
     ``scripts/download_pdfs.py``.

Usage:
  S2_API_KEY=... python scripts/prepare_pdf_jobs.py \
      --input  data/audits/to_deep_extract.jsonl \
      --output data/audits/to_deep_extract.jobs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers_full"
OA_SCAN = REPO_ROOT / "data" / "oa_scan.json"

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "openAccessPdf,externalIds"
BATCH_SIZE = 500


def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def already_on_disk(doi: str) -> bool:
    f = PAPERS_DIR / f"{doi_to_filename(doi)}.pdf"
    return f.exists() and f.stat().st_size >= 10_000


def load_oa_scan_index() -> dict[str, str]:
    """Lower-cased DOI → pdf_url mapping from oa_scan.json (if present)."""
    if not OA_SCAN.exists():
        return {}
    scan = json.loads(OA_SCAN.read_text())
    out: dict[str, str] = {}
    for p in scan.get("papers", []):
        doi = (p.get("doi") or "").strip().lower()
        url = p.get("pdf_url") or ""
        if doi and url:
            out[doi] = url
    return out


def s2_batch_lookup(dois: list[str], api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    backoff = 1.0
    for attempt in range(5):
        try:
            resp = requests.post(
                S2_BATCH,
                params={"fields": S2_FIELDS},
                json={"ids": [f"DOI:{d}" for d in dois]},
                headers=headers,
                timeout=60,
            )
        except requests.RequestException as e:
            print(f"    net err {e}, sleep {backoff:.1f}s", flush=True)
            time.sleep(backoff); backoff *= 2; continue
        if resp.status_code == 429:
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        if resp.status_code != 200:
            print(f"    s2 http {resp.status_code}: {resp.text[:120]}", flush=True)
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        out: dict[str, str] = {}
        for doi, paper in zip(dois, resp.json()):
            if not paper:
                continue
            oa = paper.get("openAccessPdf") or {}
            url = oa.get("url")
            if url:
                out[doi.lower()] = url
        return out
    print("    giving up on this batch", flush=True)
    return {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-s2", action="store_true",
                   help="Only use oa_scan.json; do not call S2 fallback")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows):,} input rows from {args.input}", flush=True)

    # Filter out PDFs already on disk and rows missing DOI
    candidates = []
    n_have_pdf = 0
    n_no_doi = 0
    for r in rows:
        doi = r.get("doi") or r.get("DOI")
        if not doi:
            n_no_doi += 1; continue
        if already_on_disk(doi):
            n_have_pdf += 1; continue
        candidates.append(r)
    print(f"  already on disk : {n_have_pdf:,}", flush=True)
    print(f"  no doi          : {n_no_doi:,}", flush=True)
    print(f"  to resolve      : {len(candidates):,}", flush=True)

    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"  --limit applied → {len(candidates):,}", flush=True)

    # Pass 1: oa_scan.json
    print("\nlooking up URLs in data/oa_scan.json ...", flush=True)
    oa_index = load_oa_scan_index()
    print(f"  oa_scan has {len(oa_index):,} doi→url pairs", flush=True)

    resolved: list[dict] = []
    unresolved: list[dict] = []
    for r in candidates:
        doi = r["doi"]
        url = oa_index.get(doi.strip().lower())
        if url:
            r2 = dict(r); r2["pdf_url"] = url; r2["pdf_url_source"] = "oa_scan"
            resolved.append(r2)
        else:
            unresolved.append(r)
    print(f"  via oa_scan: {len(resolved):,}  /  unresolved: {len(unresolved):,}",
          flush=True)

    # Pass 2: S2 batch
    if unresolved and not args.no_s2:
        api_key = os.environ.get("S2_API_KEY", "")
        print(f"\nfalling back to S2 batch lookup "
              f"({'with' if api_key else 'without'} API key) ...", flush=True)
        for start in range(0, len(unresolved), BATCH_SIZE):
            chunk = unresolved[start:start + BATCH_SIZE]
            dois = [c["doi"] for c in chunk]
            url_map = s2_batch_lookup(dois, api_key)
            for c in chunk:
                u = url_map.get(c["doi"].strip().lower())
                if u:
                    r2 = dict(c); r2["pdf_url"] = u; r2["pdf_url_source"] = "s2"
                    resolved.append(r2)
            done = min(start + BATCH_SIZE, len(unresolved))
            print(f"  s2 batch {start//BATCH_SIZE + 1}: "
                  f"resolved {len(resolved):,} / scanned {done:,}", flush=True)
            time.sleep(0.4 if api_key else 1.1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in resolved:
            f.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(resolved):,} downloadable jobs → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
