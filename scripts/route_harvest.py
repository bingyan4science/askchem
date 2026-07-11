#!/usr/bin/env python3
"""Stage 4 router — split harvested papers into deep / abstract / dropped buckets.

Reads two inputs:
  - data/ingestion_2026_05/discovered_papers.jsonl  (raw harvest, with abstracts)
  - data/harvest_2026_05_21/pdf_manifest.jsonl       (Stage 3 PDF results)

Writes three outputs:

1) data/arxiv_harvest/tier_1.jsonl
   Format expected by ``src/batch_extract_arxiv.py load_tier_papers``:
       {"doi": ..., "title": ..., "pdf_url": ..., "arxiv_id": ..., "citation_count": ...}
   ``pdf_url`` is *informational only* because ``batch_extract_arxiv.py
   prepare`` resolves the cached PDF at ``data/papers_full/<sha256:16>.pdf``
   first. We set it to the URL the Stage 3 harvester used so the file is
   self-describing.

2) data/abstract_jobs/no_pdf_2026_05_21.jsonl
   Format expected by ``src/batch_extract_abstracts.py cmd_prepare``:
       {"doi": ..., "title": ..., "abstract": ..., "authors": [...], "venue": ..., "year": ...}
   Includes ONLY papers where abstract length >= 200 chars.

3) data/harvest_2026_05_21/dropped_short_abstract.jsonl
   Logging file — papers with neither a PDF nor a long-enough abstract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DISC = REPO_ROOT / "data" / "ingestion_2026_05" / "discovered_papers.jsonl"
MANIFEST = REPO_ROOT / "data" / "harvest_2026_05_21" / "pdf_manifest.jsonl"
TIER1_OUT = REPO_ROOT / "data" / "arxiv_harvest" / "tier_1.jsonl"
ABS_OUT = REPO_ROOT / "data" / "abstract_jobs" / "no_pdf_2026_05_21.jsonl"
DROPPED_OUT = REPO_ROOT / "data" / "harvest_2026_05_21" / "dropped_short_abstract.jsonl"

MIN_ABSTRACT_LEN = 200


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _arxiv_id(paper: dict) -> str:
    a = paper.get("arxiv_id")
    if a:
        return a
    eids = paper.get("externalIds") or {}
    if eids.get("ArXiv"):
        return eids["ArXiv"]
    doi = (paper.get("doi") or "").lower()
    if doi.startswith("arxiv:"):
        return doi[len("arxiv:"):]
    if doi.startswith("10.48550/arxiv."):
        return doi[len("10.48550/arxiv."):]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovered", default=str(DISC))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--tier1", default=str(TIER1_OUT))
    ap.add_argument("--abstract", default=str(ABS_OUT))
    ap.add_argument("--dropped", default=str(DROPPED_OUT))
    ap.add_argument("--min-abstract", type=int, default=MIN_ABSTRACT_LEN)
    args = ap.parse_args()

    disc = Path(args.discovered)
    man = Path(args.manifest)
    if not disc.exists():
        print(f"missing {disc}", file=sys.stderr)
        return 1
    if not man.exists():
        print(f"missing {man}", file=sys.stderr)
        return 1

    pdf_by_doi: dict[str, dict] = {}
    for row in _read_jsonl(man):
        doi = (row.get("doi") or "").lower()
        if doi:
            pdf_by_doi[doi] = row

    tier1_path = Path(args.tier1)
    abs_path = Path(args.abstract)
    dropped_path = Path(args.dropped)
    for p in (tier1_path, abs_path, dropped_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    n_tier1 = 0
    n_abs = 0
    n_dropped = 0
    by_source_tier1: dict[str, int] = {}
    by_source_abs: dict[str, int] = {}

    with tier1_path.open("w") as tf, abs_path.open("w") as af, dropped_path.open("w") as df:
        for paper in _read_jsonl(disc):
            doi_raw = ((paper.get("externalIds") or {}).get("DOI")
                       or paper.get("doi") or "")
            doi = doi_raw.lower()
            if not doi:
                continue
            src = (paper.get("source") or "").lower()

            man_row = pdf_by_doi.get(doi)
            has_pdf = bool(man_row and man_row.get("pdf_path"))

            if has_pdf:
                tier_paper = {
                    "doi": paper.get("doi") or doi_raw,
                    "title": paper.get("title") or "",
                    "pdf_url": man_row.get("url") or "",
                    "arxiv_id": _arxiv_id(paper),
                    "citation_count": paper.get("citationCount") or 0,
                    "source": src,
                }
                tf.write(json.dumps(tier_paper, ensure_ascii=False) + "\n")
                n_tier1 += 1
                by_source_tier1[src] = by_source_tier1.get(src, 0) + 1
                continue

            abstract = (paper.get("abstract") or "").strip()
            if len(abstract) >= args.min_abstract:
                abs_paper = {
                    "doi": paper.get("doi") or doi_raw,
                    "title": paper.get("title") or "",
                    "abstract": abstract,
                    "authors": paper.get("authors") or [],
                    "venue": paper.get("venue") or "",
                    "year": paper.get("year") or 0,
                    "source": src,
                }
                af.write(json.dumps(abs_paper, ensure_ascii=False) + "\n")
                n_abs += 1
                by_source_abs[src] = by_source_abs.get(src, 0) + 1
                continue

            df.write(json.dumps({
                "doi": doi_raw,
                "title": paper.get("title") or "",
                "abstract_len": len(abstract),
                "source": src,
                "failure_reason": (man_row or {}).get("failure_reason"),
            }, ensure_ascii=False) + "\n")
            n_dropped += 1

    print(f"== routing summary ==")
    print(f"  tier_1 (full PDF) : {n_tier1:,}")
    for src, n in sorted(by_source_tier1.items(), key=lambda x: -x[1]):
        print(f"      {src:24s} {n:>6d}")
    print(f"  abstract-only     : {n_abs:,}")
    for src, n in sorted(by_source_abs.items(), key=lambda x: -x[1]):
        print(f"      {src:24s} {n:>6d}")
    print(f"  dropped (no PDF, abstract<{args.min_abstract}): {n_dropped:,}")
    print(f"\n  -> {tier1_path}")
    print(f"  -> {abs_path}")
    print(f"  -> {dropped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
