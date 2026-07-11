"""Backfill `paper_validations` from the existing CrossRef on-disk cache.

The citation-graph build (`src/build_citation_graph.py`) already fetched
CrossRef metadata for ~41K deep_v1 papers and cached the JSON responses
under `data/citations/crossref/<2-char-shard>/<sha1(doi)>.json`.

Each successful cache file is, by construction, a CrossRef "verified"
response (we only write the cache when CrossRef returns 200 + a `message`
payload). So we can populate `paper_validations.crossref_verified=1` for
every cache-hit DOI without making any new API calls.

Strategy:
  1. Walk every row in `sources` (the canonical DOI list).
  2. Compute the cache path for each DOI and check existence.
  3. If the cache file exists and parses to a valid CrossRef payload
     (`status==ok`, non-empty `message`), write a paper_validations row.
  4. Otherwise, leave it for the live `scripts/validate_dois.py` runner.

Idempotent: existing paper_validations rows are skipped.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"
CROSSREF_CACHE = REPO_ROOT / "data" / "citations" / "crossref"


def _normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    s = str(doi).strip()
    low = s.lower()
    if low.startswith("https://doi.org/"):
        s = s[len("https://doi.org/"):]
    elif low.startswith("http://doi.org/"):
        s = s[len("http://doi.org/"):]
    elif low.startswith("doi:"):
        s = s[4:].strip()
    return s.strip().lower()


def _cache_path(doi: str) -> Path:
    h = hashlib.sha1(doi.encode("utf-8")).hexdigest()
    return CROSSREF_CACHE / h[:2] / f"{h}.json"


def _summarize(payload: dict) -> dict | None:
    """Extract validation fields from a CrossRef `works` payload.

    Returns None if the payload doesn't look like a real CrossRef hit.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "ok":
        return None
    msg = payload.get("message")
    if not isinstance(msg, dict) or not msg:
        return None

    journal_list = msg.get("container-title") or []
    journal = (journal_list[0] if journal_list else "") or ""
    publisher = msg.get("publisher") or ""

    has_abstract = 1 if msg.get("abstract") else 0
    is_retracted = 1 if "retraction" in str(msg.get("update-to", "")).lower() else 0

    score = 0.5
    if has_abstract:
        score += 0.2
    if journal:
        score += 0.2
    if publisher:
        score += 0.1

    return {
        "crossref_verified": 1,
        "has_abstract": has_abstract,
        "is_retracted": is_retracted,
        "journal": journal[:200],
        "publisher": str(publisher)[:200],
        "is_chemistry": 1,
        "validation_score": round(score, 2),
    }


def main() -> None:
    if not DB_PATH.exists():
        print(f"db not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CROSSREF_CACHE.exists():
        print(f"cache not found: {CROSSREF_CACHE}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    already = {
        r["doi"] for r in conn.execute("SELECT doi FROM paper_validations")
    }
    print(f"existing paper_validations rows: {len(already):,}")

    pending: list[str] = []
    for r in conn.execute("SELECT doi FROM sources WHERE doi IS NOT NULL AND doi != ''"):
        d = r["doi"]
        if d not in already:
            pending.append(d)
    print(f"sources DOIs awaiting validation: {len(pending):,}")

    inserted = 0
    cache_hits = 0
    cache_invalid = 0
    cache_misses = 0
    started = time.time()
    batch: list[tuple] = []
    BATCH = 1000

    for i, doi in enumerate(pending, 1):
        norm = _normalize_doi(doi)
        if not norm:
            cache_misses += 1
            continue
        path = _cache_path(norm)
        if not path.exists():
            cache_misses += 1
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            cache_invalid += 1
            continue
        summary = _summarize(payload)
        if summary is None:
            cache_invalid += 1
            continue
        cache_hits += 1
        now = datetime.now(timezone.utc).isoformat()
        batch.append((
            doi,
            summary["crossref_verified"],
            summary["has_abstract"],
            summary["is_retracted"],
            summary["journal"],
            summary["publisher"],
            summary["is_chemistry"],
            summary["validation_score"],
            now,
            json.dumps({"source": "cache_backfill", **summary}),
        ))
        if len(batch) >= BATCH:
            conn.executemany(
                "INSERT OR REPLACE INTO paper_validations "
                "(doi, crossref_verified, has_abstract, is_retracted, journal, publisher, "
                "is_chemistry, validation_score, validated_at, validation_data) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)
            batch.clear()
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            print(f"  scanned {i:>7,}/{len(pending):,} | hits={cache_hits:,} "
                  f"misses={cache_misses:,} invalid={cache_invalid:,} "
                  f"| inserted={inserted:,} | {rate:,.0f}/s", flush=True)

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO paper_validations "
            "(doi, crossref_verified, has_abstract, is_retracted, journal, publisher, "
            "is_chemistry, validation_score, validated_at, validation_data) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()
        inserted += len(batch)

    total_v = conn.execute(
        "SELECT COUNT(*) FROM paper_validations WHERE crossref_verified=1"
    ).fetchone()[0]
    total_all = conn.execute("SELECT COUNT(*) FROM paper_validations").fetchone()[0]
    print()
    print(f"DONE in {time.time()-started:.1f}s")
    print(f"  scanned     : {len(pending):,}")
    print(f"  cache hits  : {cache_hits:,}")
    print(f"  cache miss  : {cache_misses:,}  (need live CrossRef call)")
    print(f"  invalid     : {cache_invalid:,}")
    print(f"  inserted    : {inserted:,}")
    print(f"  paper_validations now: {total_all:,} rows ({total_v:,} crossref_verified, "
          f"{(total_v/total_all*100 if total_all else 0):.1f}%)")
    conn.close()


if __name__ == "__main__":
    main()
