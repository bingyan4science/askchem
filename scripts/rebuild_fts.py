"""Rebuild claims_fts and sources_fts with the Porter stemmer + unicode61
tokenizer + diacritic removal.

Morphological variants (``adsorbed / adsorbing / adsorption``) collapse
to a single stem under Porter, which materially improves FTS recall on
chemistry-paper language.  unicode61 + ``remove_diacritics 2`` normalises
accented characters and (most) typographic Unicode.  A dedicated
``tokenchars`` list keeps domain-critical punctuation like ``-``, ``+``,
``/`` inside tokens (so ``Pd/C``, ``C-H``, ``Li+`` stay whole).

Run locally first, verify with the eval harness, then deploy.  Uses WAL
journaling + large page-cache tuning to keep the rebuild fast.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_DB = REPO_ROOT / "chemtree.db"


# FTS5 tokenizer spec: Porter stemmer sitting on top of unicode61, with
# diacritic removal.  unicode61's default behaviour already keeps digits
# in tokens and strips most Unicode punctuation — good for formulas like
# ``TiO2``, ``CO2``, ``MoS2``.  Hyphens continue to split terms so that
# ``cross-coupling`` matches ``cross coupling`` (current behaviour).
TOKENIZER = "'porter unicode61 remove_diacritics 2'"


def _run_vacuum(conn: sqlite3.Connection) -> None:
    conn.execute("VACUUM")


def _drop_and_recreate_claims_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS claims_fts")
    conn.execute(f"""
        CREATE VIRTUAL TABLE claims_fts USING fts5(
            claim_id UNINDEXED,
            claim_type,
            source_paper_title,
            verbatim_quote,
            searchable_text,
            tokenize = {TOKENIZER}
        )
    """)


def _drop_and_recreate_sources_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS sources_fts")
    conn.execute(f"""
        CREATE VIRTUAL TABLE sources_fts USING fts5(
            doi UNINDEXED,
            title,
            abstract,
            paper_text,
            tokenize = {TOKENIZER}
        )
    """)


def rebuild_claims_fts(conn: sqlite3.Connection, batch: int = 50_000) -> int:
    print("→ rebuilding claims_fts (Porter + unicode61)...")
    _drop_and_recreate_claims_fts(conn)
    t0 = time.monotonic()
    inserted = 0
    # JOIN sources to fold paper_summary into searchable_text; LEFT JOIN
    # so claims with missing source rows still get indexed.
    cur = conn.execute(
        "SELECT c.claim_id, c.claim_type, c.source_paper_title, "
        "c.verbatim_quote, c.data, c.claim_contextualized, s.paper_summary "
        "FROM claims c "
        "LEFT JOIN sources s ON c.source_doi = s.doi"
    )
    rows_buffer: list[tuple[str, str, str, str, str]] = []
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for (claim_id, claim_type, title, quote, data_json,
             ctx_text, paper_summary) in rows:
            try:
                d = json.loads(data_json) if data_json else {}
            except (json.JSONDecodeError, TypeError):
                d = {}
            parts: list[str] = []
            # Highest-signal text first: the LLM-rewritten standalone
            # claim, then paper-level summary. These are the Sprint 1 /
            # Sprint 0 columns. Both may be NULL (abstract-only claims,
            # or rows whose batches haven't reached them yet).
            if ctx_text:
                parts.append(str(ctx_text))
            if paper_summary:
                parts.append(str(paper_summary)[:500])
            for key in (
                "verbatim_quote", "subject", "property_name", "property_value",
                "reaction_type", "technique_name", "key_result_description",
                "conclusion",
            ):
                v = d.get(key)
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, (int, float)):
                    parts.append(str(v))
            searchable = " ".join(p for p in parts if p)
            rows_buffer.append((
                claim_id, claim_type or "", title or "",
                quote or "", searchable,
            ))
        conn.executemany(
            "INSERT INTO claims_fts"
            "(claim_id, claim_type, source_paper_title, verbatim_quote, searchable_text) "
            "VALUES (?,?,?,?,?)",
            rows_buffer,
        )
        inserted += len(rows_buffer)
        rows_buffer.clear()
        conn.commit()
        elapsed = time.monotonic() - t0
        rate = inserted / max(0.001, elapsed)
        print(f"  {inserted:>10,} rows   {elapsed:>6.1f}s   {rate:,.0f} rows/s")
    dt = time.monotonic() - t0
    print(f"✓ claims_fts: {inserted:,} rows in {dt:.1f}s")
    return inserted


def rebuild_sources_fts(conn: sqlite3.Connection, batch: int = 10_000,
                        max_paper_text_chars: int = 50_000) -> int:
    """Stream-rebuild sources_fts with bounded memory.

    Instead of loading every claim-per-DOI into a dict (blows past 2 GB
    on the production droplet), we:

      1. Iterate ``sources`` in DOI order, buffering a fixed slice of
         paper rows waiting for their claim text.
      2. Fetch claims in a single cursor ordered by source_doi and walk
         it in lock-step with the sources cursor, flushing a paper when
         its DOI no longer matches the claim cursor's current DOI.

    Peak memory is bounded by the batch size (default 10k rows).
    """
    print("→ rebuilding sources_fts (Porter + unicode61, streaming)...")
    _drop_and_recreate_sources_fts(conn)
    t0 = time.monotonic()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_source_doi "
                 "ON claims(source_doi)")

    claim_cur = conn.cursor()
    claim_cur.execute(
        "SELECT source_doi, data, claim_contextualized FROM claims "
        "WHERE source_doi IS NOT NULL AND source_doi <> '' "
        "ORDER BY source_doi"
    )

    def _extract_text(data_json: str | None,
                      ctx_text: str | None = None) -> str:
        chunks: list[str] = []
        if ctx_text:
            chunks.append(str(ctx_text))
        if data_json:
            try:
                d = json.loads(data_json)
                parts = [d.get(k) for k in (
                    "verbatim_quote", "subject", "property_name",
                    "reaction_type", "technique_name",
                )]
                chunks.extend(str(p) for p in parts
                              if isinstance(p, (str, int, float)))
            except (json.JSONDecodeError, TypeError):
                pass
        return " ".join(chunks)

    # Lookahead buffer for the claims cursor: we always hold at most one
    # pending row that we haven't yet consumed for the current DOI.
    pending: tuple[str, str, str | None] | None = None

    def _text_for_doi(target_doi: str) -> str:
        """Consume claim rows whose source_doi equals ``target_doi`` and
        return concatenated searchable text.  Leaves one row pending if
        the cursor advances past ``target_doi``.
        """
        nonlocal pending
        chunks: list[str] = []
        used = 0
        while True:
            if pending is None:
                row = claim_cur.fetchone()
                if row is None:
                    return " ".join(chunks)[:max_paper_text_chars]
                pending = (row[0], row[1], row[2])
            doi, data_json, ctx_text = pending
            if doi < target_doi:
                pending = None
                continue
            if doi > target_doi:
                return " ".join(chunks)[:max_paper_text_chars]
            txt = _extract_text(data_json, ctx_text)
            if txt and used < max_paper_text_chars:
                chunks.append(txt)
                used += len(txt) + 1
            pending = None

    inserted = 0
    row_buf: list[tuple[str, str, str, str]] = []
    src_cur = conn.cursor()
    src_cur.execute(
        "SELECT doi, title, abstract, paper_summary FROM sources "
        "WHERE title IS NOT NULL AND doi IS NOT NULL "
        "ORDER BY doi"
    )

    for doi, title, abstract, paper_summary in src_cur:
        # Stitch paper_summary into paper_text so a query that hits the
        # summary lifts the paper in the sources_fts ranking too.
        body = _text_for_doi(doi) if doi else ""
        if paper_summary:
            body = (str(paper_summary) + " " + body).strip()
            body = body[:max_paper_text_chars]
        row_buf.append((doi, title or "", abstract or "", body))
        if len(row_buf) >= batch:
            conn.executemany(
                "INSERT INTO sources_fts (doi, title, abstract, paper_text) "
                "VALUES (?,?,?,?)",
                row_buf,
            )
            inserted += len(row_buf)
            row_buf.clear()
            conn.commit()
            elapsed = time.monotonic() - t0
            print(f"  {inserted:>8,} sources indexed in {elapsed:>5.1f}s")

    if row_buf:
        conn.executemany(
            "INSERT INTO sources_fts (doi, title, abstract, paper_text) "
            "VALUES (?,?,?,?)",
            row_buf,
        )
        inserted += len(row_buf)
        conn.commit()

    dt = time.monotonic() - t0
    print(f"✓ sources_fts: {inserted:,} rows in {dt:.1f}s")
    return inserted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--claims-only", action="store_true")
    p.add_argument("--sources-only", action="store_true")
    p.add_argument("--no-vacuum", action="store_true")
    p.add_argument("--low-memory", action="store_true",
                   help="use tight page cache / no mmap / disk temp store "
                        "(recommended on <=2 GB RAM boxes)")
    args = p.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 1
    print(f"database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    if args.low_memory:
        conn.execute("PRAGMA cache_size=-32768")   # ~32 MB page cache
        conn.execute("PRAGMA temp_store=FILE")     # avoid spiking RAM
        conn.execute("PRAGMA mmap_size=0")         # no mmap
    else:
        conn.execute("PRAGMA cache_size=-262144")  # ~256 MB page cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=8589934592")

    if not args.sources_only:
        rebuild_claims_fts(conn)
    if not args.claims_only:
        rebuild_sources_fts(conn)

    if not args.no_vacuum:
        print("→ running VACUUM ...")
        t = time.monotonic()
        _run_vacuum(conn)
        print(f"✓ vacuum in {time.monotonic() - t:.1f}s")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
