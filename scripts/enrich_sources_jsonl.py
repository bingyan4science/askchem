"""Append metadata for new papers to ``chemtree/sources.jsonl``.

When ``to_add`` papers get deep-extracted, ``src/integrate_deep.py`` will
need a row in ``chemtree/sources.jsonl`` for each one (it uses that file
as the canonical source-metadata table when rebuilding the DB).

This helper takes a JSONL of new papers (e.g. ``data/audits/to_add.jobs.jsonl``,
which already contains ``doi``, ``title``, ``year``, ``citations``,
``venue``, ``oa``), enriches each with ``authors`` + ``abstract`` +
``fields_of_study`` + ``semantic_scholar_id`` from the S2 batch endpoint,
and appends rows for any DOI not already present.

Usage:
  S2_API_KEY=... python scripts/enrich_sources_jsonl.py \
      --input data/audits/to_add.jobs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_JSONL = REPO_ROOT / "askchem" / "sources.jsonl"

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = ("paperId,title,abstract,authors.name,year,venue,"
             "openAccessPdf,citationCount,fieldsOfStudy,externalIds")
BATCH_SIZE = 500


def load_existing_dois() -> set[str]:
    out: set[str] = set()
    if not SOURCES_JSONL.exists():
        return out
    with SOURCES_JSONL.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = (d.get("doi") or "").strip().lower()
            if doi:
                out.add(doi)
    return out


def s2_batch(dois: list[str], api_key: str) -> dict[str, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    backoff = 1.0
    for attempt in range(5):
        try:
            r = requests.post(
                S2_BATCH,
                params={"fields": S2_FIELDS},
                json={"ids": [f"DOI:{d}" for d in dois]},
                headers=headers, timeout=60,
            )
        except requests.RequestException as e:
            print(f"    net err {e}, sleep {backoff:.1f}s", flush=True)
            time.sleep(backoff); backoff *= 2; continue
        if r.status_code == 429:
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        if r.status_code != 200:
            print(f"    s2 http {r.status_code}: {r.text[:120]}", flush=True)
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        out: dict[str, dict] = {}
        for doi, paper in zip(dois, r.json()):
            if paper:
                out[doi.lower()] = paper
        return out
    return {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="JSONL of new papers (must include 'doi' field)")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    api_key = os.environ.get("S2_API_KEY", "")
    if not api_key:
        print("warning: S2_API_KEY not set — calls will be rate-limited",
              file=sys.stderr)

    rows: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows):,} input rows from {args.input}", flush=True)

    existing = load_existing_dois()
    print(f"existing dois in chemtree/sources.jsonl: {len(existing):,}", flush=True)

    new_dois = []
    seen = set()
    for r in rows:
        doi = (r.get("doi") or "").strip()
        if not doi:
            continue
        key = doi.lower()
        if key in existing or key in seen:
            continue
        seen.add(key)
        new_dois.append(doi)

    print(f"need to enrich+append: {len(new_dois):,} dois", flush=True)
    if args.limit and args.limit > 0:
        new_dois = new_dois[: args.limit]
        print(f"  --limit applied → {len(new_dois):,}", flush=True)

    if not new_dois:
        print("nothing to do.")
        return 0

    SOURCES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with SOURCES_JSONL.open("a") as out:
        for start in range(0, len(new_dois), BATCH_SIZE):
            chunk = new_dois[start: start + BATCH_SIZE]
            data = s2_batch(chunk, api_key)
            for doi in chunk:
                paper = data.get(doi.lower())
                if not paper:
                    continue
                authors = [a.get("name", "") for a in (paper.get("authors") or [])][:50]
                oa = paper.get("openAccessPdf") or {}
                row = {
                    "doi": doi,
                    "semantic_scholar_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "authors": authors,
                    "year": paper.get("year") or 0,
                    "venue": paper.get("venue", ""),
                    "abstract": paper.get("abstract", "") or "",
                    "citation_count": paper.get("citationCount", 0) or 0,
                    "open_access_url": oa.get("url", "") or "",
                    "fields_of_study": paper.get("fieldsOfStudy") or [],
                }
                out.write(json.dumps(row) + "\n")
                appended += 1
            done = min(start + BATCH_SIZE, len(new_dois))
            print(f"  batch {start//BATCH_SIZE + 1}: appended {appended:,} / scanned {done:,}",
                  flush=True)
            time.sleep(0.4 if api_key else 1.1)

    print(f"\nappended {appended:,} rows → {SOURCES_JSONL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
