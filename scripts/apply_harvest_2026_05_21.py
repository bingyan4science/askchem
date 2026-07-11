#!/usr/bin/env python3
"""Apply step for the 2026-05-21 multi-source harvest.

Extends ``scripts/apply_incremental_2026_05.py`` to support:
  * Both Stage 4a (full-PDF, ``deep_v1``) AND Stage 4b (abstract-only,
    ``deep_v1_abstract``) extraction results.
  * Metadata pulled from ``data/ingestion_2026_05/discovered_papers.jsonl``
    (the harvest output) rather than ``chemtree/sources.jsonl`` — the
    new papers from CrossRef/S2/ChemRxiv haven't been added to
    sources.jsonl yet.
  * Preserves ``extraction_version`` per claim (so the apply step
    correctly tags abstract-only claims as ``deep_v1_abstract``,
    matching the result file's own value).

Inputs:
  - data/arxiv_harvest/tier_1.jsonl              (Stage 4a routing output)
  - data/abstract_jobs/no_pdf_2026_05_21.jsonl   (Stage 4b routing output)
  - data/ingestion_2026_05/discovered_papers.jsonl  (metadata source)
  - data/deep_results/<custom_id>.json           (extractions)
  - data/paper_classify/paper_classifications.json (classifier output)

Usage::

    python3 scripts/apply_harvest_2026_05_21.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import (  # noqa: E402
    DB_PATH, get_conn, upsert_source,
    upsert_tree_node, update_metadata_counts, index_authors_for_doi,
    build_searchable_text,
)
from askchem.models import Claim  # noqa: E402
from askchem.taxonomy import (  # noqa: E402
    CANONICAL_L1, CLAIM_TYPE_LABELS, ALL_CONTENT_VIEWS,
)

RESULTS_DIR = REPO_ROOT / "data" / "deep_results"
PAPER_CLASS_PATH = REPO_ROOT / "data" / "paper_classify" / "paper_classifications.json"
DISCOVERED_JSONL = REPO_ROOT / "data" / "ingestion_2026_05" / "discovered_papers.jsonl"
TIER_1_JSONL = REPO_ROOT / "data" / "arxiv_harvest" / "tier_1.jsonl"
ABSTRACT_JSONL = REPO_ROOT / "data" / "abstract_jobs" / "no_pdf_2026_05_21.jsonl"


def _normalize_view_path(p) -> list:
    if not p:
        return []
    if isinstance(p, list) and p and isinstance(p[0], list):
        p = p[0]
    if not isinstance(p, list):
        return []
    out = []
    for seg in p:
        if not seg:
            continue
        s = str(seg).strip().lower().replace("-", "_").replace(" ", "_")
        if s and s != "none":
            out.append(s)
    return out


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    started = time.time()

    if not PAPER_CLASS_PATH.exists():
        print(f"missing {PAPER_CLASS_PATH}", file=sys.stderr)
        return 1
    paper_classifications = json.loads(PAPER_CLASS_PATH.read_text())
    print(f"paper_classifications: {len(paper_classifications):,} papers")

    # ─ Load existing DOIs + claim_ids ───────────────────────────────────
    with get_conn(readonly=True) as conn:
        existing_doi_rows = conn.execute("SELECT doi FROM sources").fetchall()
        existing_dois = {(r["doi"] or "").lower() for r in existing_doi_rows}
        existing_claim_rows = conn.execute("SELECT claim_id FROM claims").fetchall()
        existing_claim_ids = {(r["claim_id"] or "") for r in existing_claim_rows}
    print(f"existing DB: {len(existing_dois):,} sources, "
          f"{len(existing_claim_ids):,} claims")

    # ─ Build map of THIS-INGEST DOIs from both stage 4a and 4b ──────────
    tier1_dois: list[str] = []
    for row in _read_jsonl(TIER_1_JSONL):
        doi = (row.get("doi") or "").strip()
        if doi:
            tier1_dois.append(doi)
    abstract_dois: list[str] = []
    for row in _read_jsonl(ABSTRACT_JSONL):
        doi = (row.get("doi") or "").strip()
        if doi:
            abstract_dois.append(doi)
    this_ingest_dois = tier1_dois + abstract_dois
    print(f"this-ingest DOIs: {len(this_ingest_dois):,} "
          f"({len(tier1_dois)} full-PDF + {len(abstract_dois)} abstract-only)")

    # ─ Load metadata from discovered_papers.jsonl ───────────────────────
    wanted = {d.lower() for d in this_ingest_dois}
    sources_meta: dict[str, dict] = {}
    for paper in _read_jsonl(DISCOVERED_JSONL):
        doi_field = (paper.get("doi") or
                     (paper.get("externalIds") or {}).get("DOI") or "")
        dlow = doi_field.lower()
        if dlow and dlow in wanted:
            # Normalize: prefer the doi field on the paper, fall back to externalIds
            sources_meta[dlow] = paper
    print(f"  metadata for {len(sources_meta):,}/{len(this_ingest_dois):,}")

    # ─ Find deep_results for these DOIs ─────────────────────────────────
    new_results: list[dict] = []
    missing_extraction = 0
    no_classification = 0
    already_in_db = 0

    for doi in this_ingest_dois:
        dlow = doi.lower()
        if dlow in existing_dois:
            already_in_db += 1
            continue
        cid = hashlib.sha256(doi.encode()).hexdigest()[:16]
        rp = RESULTS_DIR / f"{cid}.json"
        if not rp.exists():
            missing_extraction += 1
            continue
        if doi not in paper_classifications:
            no_classification += 1
            continue
        try:
            new_results.append(json.loads(rp.read_text()))
        except Exception:
            missing_extraction += 1
            continue
    print(f"this-ingest deep_results with classification: {len(new_results):,}")
    print(f"  already in DB:           {already_in_db}")
    print(f"  missing extraction:      {missing_extraction}")
    print(f"  missing classification:  {no_classification}")

    if not new_results:
        print("nothing to apply")
        return 0

    # ─ Apply: sources ───────────────────────────────────────────────────
    print(f"\n[1/4] inserting {len(new_results)} new sources...")
    n_inserted_sources = 0
    for result in new_results:
        doi = result.get("doi", "")
        if not doi:
            continue
        meta = sources_meta.get(doi.lower(), {})
        authors = meta.get("authors") or []
        if authors and isinstance(authors[0], dict):
            authors = [a.get("name", "") for a in authors]
        elif authors and isinstance(authors[0], str):
            pass  # already list of names
        source_data = {
            "doi": doi,
            "title": meta.get("title", ""),
            "authors": authors,
            "year": meta.get("year") or 0,
            "venue": meta.get("venue", ""),
            "abstract": meta.get("abstract", ""),
            "citation_count": meta.get("citationCount") or meta.get("citation_count") or 0,
            "open_access_url": (meta.get("openAccessPdf") or {}).get("url") or
                               meta.get("open_access_url", ""),
        }
        try:
            upsert_source(source_data)
            n_inserted_sources += 1
        except Exception as exc:
            print(f"  ! source insert error for {doi}: {exc}")
    print(f"  inserted {n_inserted_sources} sources")

    # ─ Apply: claims (bulk insert, no FTS delete) ───────────────────────
    print(f"\n[2/4] building + inserting claims...")
    claim_dicts: list[dict] = []
    node_claims: dict[tuple[str, str], list[str]] = defaultdict(list)
    paper_claim_ids: dict[str, list[str]] = defaultdict(list)
    n_new_claims = 0
    n_dup_claims = 0
    n_offtax = 0

    for result in new_results:
        doi = result.get("doi", "")
        if not doi:
            continue
        title = sources_meta.get(doi.lower(), {}).get("title", "")
        paper_paths = paper_classifications.get(doi, {}) or {}
        normalized_view_paths: dict[str, list] = {}
        for vid in ALL_CONTENT_VIEWS:
            p = _normalize_view_path(paper_paths.get(vid))
            if not p:
                continue
            l1 = p[0]
            allowed = {x.lower().replace("-", "_").replace(" ", "_")
                       for x in CANONICAL_L1.get(vid, [])}
            if l1 not in allowed:
                n_offtax += 1
                continue
            normalized_view_paths[vid] = p

        paper_knowledge = (result.get("data") or {}).get("paper_knowledge") or {}
        subfield = (paper_knowledge.get("subfield") or "").lower().replace(" ", "_")

        # Preserve the extraction_version that was set when the result
        # was collected (deep_v1 for full-PDF, deep_v1_abstract otherwise).
        extraction_version = result.get("extraction_version") or "deep_v1"

        claims_list = (result.get("data") or {}).get("claims", [])
        if not isinstance(claims_list, list):
            continue

        for raw_claim in claims_list:
            if not isinstance(raw_claim, dict):
                continue
            claim_type = (raw_claim.get("claim_type") or "unknown")
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            if claim_id in existing_claim_ids:
                n_dup_claims += 1
                continue
            existing_claim_ids.add(claim_id)

            view_paths = dict(normalized_view_paths)
            ct_l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
            ct_path = [ct_l1]
            if subfield:
                ct_path.append(subfield)
            view_paths["by_claim_type"] = ct_path

            claim_data = dict(raw_claim)
            claim_data.update({
                "claim_id": claim_id,
                "claim_type": claim_type,
                "source_doi": doi,
                "source_paper_title": title,
                "confidence": raw_claim.get("confidence", "medium"),
                "location_in_paper": raw_claim.get("location_in_paper", ""),
                "verbatim_quote": raw_claim.get("verbatim_quote", ""),
                "extraction_model": "gemini-3.1-pro",
                "extraction_version": extraction_version,
                "extracted_at": result.get("collected_at", datetime.now().isoformat()),
                "view_paths": view_paths,
            })
            claim_dicts.append(claim_data)
            n_new_claims += 1
            paper_claim_ids[doi].append(claim_id)

            for vid, path in view_paths.items():
                for depth in range(len(path)):
                    partial = "/".join(str(s) for s in path[: depth + 1])
                    node_claims[(vid, partial)].append(claim_id)

    print(f"  new claims: {n_new_claims:,}")
    print(f"  dup claim_ids skipped: {n_dup_claims}")
    print(f"  off-taxonomy L1 paths dropped: {n_offtax}")

    if claim_dicts:
        BATCH = 5000
        with sqlite3.connect(str(DB_PATH), timeout=60) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cur = conn.cursor()
            for i in range(0, len(claim_dicts), BATCH):
                chunk = claim_dicts[i:i + BATCH]
                claim_rows = []
                fts_rows = []
                for cd in chunk:
                    claim_rows.append((
                        cd["claim_id"], cd.get("claim_type", ""),
                        cd.get("source_doi", ""), cd.get("source_paper_title", ""),
                        cd.get("confidence", ""), cd.get("location_in_paper", ""),
                        cd.get("verbatim_quote", ""),
                        cd.get("extraction_model", ""),
                        cd.get("extraction_version", ""),
                        cd.get("extracted_at", ""),
                        json.dumps(cd.get("view_paths", {})),
                        json.dumps(cd),
                    ))
                    fts_rows.append((
                        cd["claim_id"], cd.get("claim_type", ""),
                        cd.get("source_paper_title", ""),
                        cd.get("verbatim_quote", ""),
                        build_searchable_text(cd),
                    ))
                cur.executemany(
                    "INSERT OR REPLACE INTO claims "
                    "(claim_id,claim_type,source_doi,source_paper_title,confidence,"
                    "location_in_paper,verbatim_quote,extraction_model,extraction_version,"
                    "extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    claim_rows,
                )
                cur.executemany(
                    "INSERT INTO claims_fts(claim_id,claim_type,source_paper_title,"
                    "verbatim_quote,searchable_text) VALUES (?,?,?,?,?)",
                    fts_rows,
                )
                conn.commit()
                print(f"  [{min(i+BATCH, len(claim_dicts)):,}/{len(claim_dicts):,}] flushed")
        print(f"  upserted {len(claim_dicts):,} claims into DB + FTS")

    # ─ Tree nodes ───────────────────────────────────────────────────────
    print(f"\n[3/4] updating {len(node_claims):,} tree nodes...")
    with get_conn(readonly=False) as conn:
        for (vid, path), cids in node_claims.items():
            row = conn.execute(
                "SELECT claim_ids, name, level FROM tree_nodes "
                "WHERE view_id=? AND path=?",
                [vid, path],
            ).fetchone()
            if row is None:
                segs = path.split("/")
                upsert_tree_node(
                    vid, path,
                    name=segs[-1].replace("_", " ").title(),
                    level=len(segs),
                    claim_ids=cids,
                    data={
                        "view_id": vid, "path": path,
                        "name": segs[-1], "level": len(segs),
                        "claim_count": len(cids), "children": [],
                        "claim_ids": cids,
                    },
                )
            else:
                existing_cids = (json.loads(row["claim_ids"])
                                 if row["claim_ids"] else [])
                merged = list(dict.fromkeys(existing_cids + cids))
                conn.execute(
                    "UPDATE tree_nodes SET claim_ids = ?, claim_count = ? "
                    "WHERE view_id = ? AND path = ?",
                    (json.dumps(merged), len(merged), vid, path),
                )
        conn.commit()

    # ─ Per-paper nodes + author indexing ────────────────────────────────
    print(f"\n[4/4] by_paper nodes + indexing authors for {len(paper_claim_ids)} papers...")
    for doi, cids in paper_claim_ids.items():
        title = sources_meta.get(doi.lower(), {}).get("title", doi)
        doi_path = doi.replace("/", "__")
        upsert_tree_node(
            "by_paper", doi_path,
            name=title, level=1, claim_ids=cids,
            data={
                "view_id": "by_paper", "path": doi_path,
                "name": title, "level": 1,
                "claim_count": len(cids), "children": [],
                "claim_ids": cids, "doi": doi,
            },
        )
        try:
            index_authors_for_doi(doi)
        except Exception:
            pass

    update_metadata_counts()

    print(f"\ndone in {(time.time() - started):.1f}s")
    with get_conn(readonly=True) as conn:
        n_src = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
        n_clm = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
        n_node = conn.execute("SELECT COUNT(*) AS n FROM tree_nodes").fetchone()["n"]
    print(f"final DB: {n_src:,} sources, {n_clm:,} claims, {n_node:,} tree_nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
