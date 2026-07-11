#!/usr/bin/env python3
"""Stage 2.5 — enrich harvested papers with S2 openAccessPdf URLs.

Reads ``data/ingestion_2026_05/discovered_papers.jsonl``. For every paper
that was NOT sourced from arXiv or ChemRxiv (where we have a non-S2 path
to the PDF) and that does not already carry an ``openAccessPdf.url``,
call ``S2 paper/batch`` with up to 500 DOIs per request to fill in the
field. Writes the result back to the same JSONL.

This unlocks Stage 3 (PDF harvest) for hybrid-OA papers that CrossRef
returned without OA flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "data" / "ingestion_2026_05" / "discovered_papers.jsonl"

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500
S2_FIELDS = "openAccessPdf,externalIds,abstract"
S2_MIN_DELAY = 1.1
S2_RETRY_BACKOFF = (15, 30, 60, 120, 240)


def _headers() -> dict:
    key = os.environ.get("S2_API_KEY", "")
    h = {"Content-Type": "application/json"}
    if key:
        h["x-api-key"] = key
    return h


def _batch_lookup(dois: list[str], label: str) -> list[dict | None]:
    """POST ``paper/batch`` with up to S2_BATCH_SIZE DOIs.

    Returns a list of S2 responses in the same order as ``dois``; entries
    are dict or None (for not-found / errored).
    """
    if not dois:
        return []
    body = {"ids": [f"DOI:{d}" for d in dois]}
    params = {"fields": S2_FIELDS}

    attempt = 0
    while attempt < len(S2_RETRY_BACKOFF):
        try:
            time.sleep(S2_MIN_DELAY)
            resp = requests.post(
                S2_BATCH, params=params, json=body,
                headers=_headers(), timeout=60,
            )
            if resp.status_code in (429, 403, 502, 503):
                wait = int(resp.headers.get("Retry-After") or
                           S2_RETRY_BACKOFF[attempt])
                print(f"  [{label}] HTTP {resp.status_code}, waiting {wait}s "
                      f"(attempt {attempt+1})", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            if resp.status_code == 200:
                return resp.json()
            print(f"  [{label}] HTTP {resp.status_code}: {resp.text[:200]}",
                  flush=True)
            return [None] * len(dois)
        except Exception as exc:
            wait = S2_RETRY_BACKOFF[min(attempt, len(S2_RETRY_BACKOFF) - 1)]
            print(f"  [{label}] error: {exc} -- waiting {wait}s", flush=True)
            time.sleep(wait)
            attempt += 1
    return [None] * len(dois)


def _needs_enrichment(paper: dict) -> bool:
    """True when we'd benefit from calling S2 batch for this paper."""
    src = (paper.get("source") or "").lower()
    if src.startswith("arxiv") or src == "chemrxiv":
        return False
    if (paper.get("openAccessPdf") or {}).get("url"):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_JSONL))
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return 1

    print(f"== Enrich OA URLs ==")
    print(f"  input : {path}")

    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    targets: list[tuple[int, str]] = []
    for i, p in enumerate(rows):
        if not _needs_enrichment(p):
            continue
        doi = (p.get("externalIds") or {}).get("DOI") or p.get("doi") or ""
        if not doi or doi.lower().startswith("arxiv:"):
            continue
        targets.append((i, doi))

    print(f"  total rows: {len(rows):,}")
    print(f"  enrichment targets: {len(targets):,}")
    if not targets:
        print("  nothing to do")
        return 0

    enriched_count = 0
    pdf_count = 0
    abstract_filled = 0
    n_batches = (len(targets) + S2_BATCH_SIZE - 1) // S2_BATCH_SIZE

    for b in range(n_batches):
        chunk = targets[b * S2_BATCH_SIZE:(b + 1) * S2_BATCH_SIZE]
        dois = [doi for _, doi in chunk]
        idxs = [i for i, _ in chunk]
        label = f"batch {b+1}/{n_batches}"
        result = _batch_lookup(dois, label)
        if len(result) != len(dois):
            print(f"  [{label}] response size mismatch {len(result)} != {len(dois)}",
                  flush=True)
            result = (result + [None] * len(dois))[:len(dois)]

        for idx, doi, r in zip(idxs, dois, result):
            if not r or not isinstance(r, dict):
                continue
            enriched_count += 1
            oa = r.get("openAccessPdf") or {}
            url = oa.get("url")
            if url:
                rows[idx]["openAccessPdf"] = oa
                pdf_count += 1
            if not rows[idx].get("abstract") and r.get("abstract"):
                rows[idx]["abstract"] = r["abstract"]
                abstract_filled += 1
        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            print(f"  [{label}] cumulative: enriched={enriched_count} "
                  f"with_pdf={pdf_count} abstract_filled={abstract_filled}",
                  flush=True)

    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as f:
        for p in rows:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp_path.replace(path)

    print(f"\n== summary ==")
    print(f"  total rows : {len(rows):,}")
    print(f"  enrichment attempted : {len(targets):,}")
    print(f"  enrichment ok         : {enriched_count:,}")
    print(f"  openAccessPdf added   : {pdf_count:,}")
    print(f"  abstract filled       : {abstract_filled:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
