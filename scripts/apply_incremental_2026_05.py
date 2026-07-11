#!/usr/bin/env python3
"""Incremental apply of new deep_v1 papers into chemtree.db.

The full ``classify_papers.py rebuild`` drops and rebuilds the DB,
which would lose user-generated tables (users, api_keys, subscriptions,
etc.). For an incremental ingestion, we only need to:

  1. Insert new ``sources`` rows for papers in chemtree/sources.jsonl
     that aren't already in the DB.
  2. Insert new ``claims`` rows (and FTS index entries) from
     data/deep_results/<custom_id>.json files that aren't already in
     the DB, with ``view_paths`` filled from
     data/paper_classify/paper_classifications.json (paper-level) plus
     a deterministic ``by_claim_type`` path.
  3. Append the new claim_ids to ``tree_nodes`` and rebuild the
     ``children`` and ``claim_count`` columns for affected nodes.
  4. Update ``metadata`` counts.

Filters: only processes deep_results whose ``doi`` is NOT already in
chemtree.db.sources (i.e. genuinely new papers) AND whose ``doi`` has a
non-empty entry in paper_classifications.json.

Usage::

    python3 scripts/apply_incremental_2026_05.py
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
    DB_PATH, get_conn, upsert_source, upsert_claims_batch,
    upsert_tree_node, update_metadata_counts, index_authors_for_doi,
    build_searchable_text,
)
from askchem.models import Claim  # noqa: E402
from askchem.taxonomy import (  # noqa: E402
    CANONICAL_L1, CLAIM_TYPE_LABELS, ALL_CONTENT_VIEWS,
)

RESULTS_DIR = REPO_ROOT / "data" / "deep_results"
SOURCES_JSONL = REPO_ROOT / "askchem" / "sources.jsonl"
PAPER_CLASS_PATH = REPO_ROOT / "data" / "paper_classify" / "paper_classifications.json"
MANIFEST_PATH = REPO_ROOT / "data" / "arxiv_batch_tier1" / "manifest.json"
TIER_1_JSONL = REPO_ROOT / "data" / "arxiv_harvest" / "tier_1.jsonl"


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


def main() -> int:
    started = time.time()

    if not PAPER_CLASS_PATH.exists():
        print(f"missing {PAPER_CLASS_PATH}")
        return 1
    paper_classifications = json.loads(PAPER_CLASS_PATH.read_text())
    print(f"paper_classifications: {len(paper_classifications):,} papers")

    # Load existing source DOIs (for dedup)
    with get_conn(readonly=True) as conn:
        existing_doi_rows = conn.execute("SELECT doi FROM sources").fetchall()
        existing_dois = {(r["doi"] or "").lower() for r in existing_doi_rows}
        existing_claim_rows = conn.execute("SELECT claim_id FROM claims").fetchall()
        existing_claim_ids = {(r["claim_id"] or "") for r in existing_claim_rows}
    print(f"existing DB: {len(existing_dois):,} sources, "
          f"{len(existing_claim_ids):,} claims")

    # Build map of THIS-INGEST DOIs from tier_1.jsonl (canonical form)
    this_ingest_dois: list[str] = []
    with TIER_1_JSONL.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = (d.get("doi") or "").strip()
            if doi:
                this_ingest_dois.append(doi)
    print(f"this-ingest DOIs (tier_1.jsonl): {len(this_ingest_dois):,}")

    # Load paper metadata from sources.jsonl for those DOIs
    wanted = {d.lower() for d in this_ingest_dois}
    sources_meta: dict[str, dict] = {}
    with SOURCES_JSONL.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = (d.get("doi") or "").strip().lower()
            if doi in wanted:
                sources_meta[doi] = d
    print(f"  metadata for {len(sources_meta):,}/{len(this_ingest_dois):,} of those DOIs")

    # Find deep_results for these DOIs
    new_results: list[dict] = []
    missing_extraction = 0
    no_classification = 0
    for doi in this_ingest_dois:
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
    print(f"  missing extraction: {missing_extraction}")
    print(f"  missing classification: {no_classification}")

    if not new_results:
        print("nothing to apply")
        return 0

    # ─ Apply ──────────────────────────────────────────────────────────
    # 1. Insert sources
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
        source_data = {
            "doi": doi,
            "title": meta.get("title", ""),
            "authors": authors,
            "year": meta.get("year") or 0,
            "venue": meta.get("venue", ""),
            "abstract": meta.get("abstract", ""),
            "citation_count": meta.get("citation_count", 0) or 0,
            "open_access_url": meta.get("open_access_url", ""),
        }
        try:
            upsert_source(source_data)
            n_inserted_sources += 1
        except Exception as exc:
            print(f"  ! source insert error for {doi}: {exc}")
    print(f"  inserted {n_inserted_sources} sources")

    # 2. Build + insert claims
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
        # Normalize paper-level view_paths once, snap off-taxonomy L1s to none
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

        for raw_claim in (result.get("data") or {}).get("claims", []):
            claim_type = (raw_claim.get("claim_type") or "unknown")
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            if claim_id in existing_claim_ids:
                n_dup_claims += 1
                continue
            existing_claim_ids.add(claim_id)

            # Paper-level paths + deterministic by_claim_type
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
                "extraction_version": "deep_v1",
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
        # Bulk insert path: claims are guaranteed new (we filtered against
        # existing_claim_ids above), so we can skip the per-row DELETE
        # FROM claims_fts that ``upsert_claims_batch`` does. That delete
        # is the single biggest perf bottleneck against a 2.3M-row FTS5
        # virtual table — at ~1k claims per minute it would take ~50 min
        # for this batch alone. Without it: <30 s.
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

    # 3. Update tree_nodes
    print(f"\n[3/4] updating {len(node_claims):,} tree nodes...")
    with get_conn(readonly=False) as conn:
        for (vid, path), cids in node_claims.items():
            row = conn.execute(
                "SELECT claim_ids, name, level FROM tree_nodes WHERE view_id=? AND path=?",
                [vid, path],
            ).fetchone()
            if row is None:
                # New node
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
                existing_cids = json.loads(row["claim_ids"]) if row["claim_ids"] else []
                merged = list(dict.fromkeys(existing_cids + cids))  # dedup, preserve order
                conn.execute(
                    "UPDATE tree_nodes SET claim_ids = ?, claim_count = ? "
                    "WHERE view_id = ? AND path = ?",
                    (json.dumps(merged), len(merged), vid, path),
                )
        conn.commit()

    # 4. Per-paper tree nodes + author indexing
    print(f"\n[4/4] adding by_paper nodes + indexing authors for {len(paper_claim_ids)} papers...")
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
    # Re-query final counts
    with get_conn(readonly=True) as conn:
        n_src = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
        n_clm = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
        n_node = conn.execute("SELECT COUNT(*) AS n FROM tree_nodes").fetchone()["n"]
    print(f"final DB: {n_src:,} sources, {n_clm:,} claims, {n_node:,} tree_nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
