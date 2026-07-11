"""Build the paper-to-paper citation graph used to drive cross-paper claim
edge extraction.

Two fetchers feed one shared `citations` table.

    crossref   For every deep_v1 DOI, hit api.crossref.org/works/{doi} via the
               polite pool (mailto in User-Agent, ~30 req/s).  Cached as JSON in
               data/citations/crossref/<sha1(doi)>.json so reruns are free.
    s2-gaps    For DOIs Crossref returned with empty / missing references, fall
               back to Semantic Scholar's references endpoint.  Honors
               S2_API_KEY for the 100 req/s authenticated tier (without it,
               the unauthenticated tier is ~1 req/s).
    report     Distribution of in-corpus references per paper, total
               in-corpus citation pairs, and per-subarea pair counts.

Resumability: both fetchers skip DOIs that already have a cache hit on disk.
The DB write step is idempotent (PRIMARY KEY (citing_doi, cited_doi, source)).

Cost: free — only API calls to Crossref and Semantic Scholar, no LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"
CACHE_ROOT = REPO_ROOT / "data" / "citations"
CROSSREF_CACHE = CACHE_ROOT / "crossref"
S2_CACHE = CACHE_ROOT / "s2"

sys.path.insert(0, str(REPO_ROOT / "src"))

CROSSREF_BASE = "https://api.crossref.org/works/"
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
DEFAULT_MAILTO = os.environ.get("CROSSREF_MAILTO", "askchem@askchem.org")

CROSSREF_CONC = 4    # Polite pool tolerates ~50 req/s; conc=4 + 0.1s sleep ≈ 30 req/s.
S2_CONC_AUTHED = 2   # S2 throttles /paper/{id}/references aggressively per key.
S2_CONC_ANON = 1     # ~1 req/s without API key
DEFAULT_TIMEOUT = 30
CROSSREF_PER_REQ_SLEEP = 0.1  # Per-thread polite delay; combined with conc gives ~30 req/s.

_DB_LOCK = threading.Lock()
_TLS = threading.local()


# ── Utility ─────────────────────────────────────────────────────────────────


def _doi_hash(doi: str) -> str:
    return hashlib.sha1(doi.encode("utf-8")).hexdigest()


def _cache_path(root: Path, doi: str) -> Path:
    h = _doi_hash(doi)
    # Two-char shard to keep any one directory under ~100K files.
    return root / h[:2] / f"{h}.json"


def _normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    s = str(doi).strip()
    if s.lower().startswith("https://doi.org/"):
        s = s[len("https://doi.org/"):]
    elif s.lower().startswith("http://doi.org/"):
        s = s[len("http://doi.org/"):]
    elif s.lower().startswith("doi:"):
        s = s[4:].strip()
    return s.strip().lower()


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ── DB ──────────────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=60.0)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA busy_timeout = 30000")
    con.row_factory = sqlite3.Row
    return con


def thread_db() -> sqlite3.Connection:
    con = getattr(_TLS, "con", None)
    if con is None:
        con = sqlite3.connect(str(DB_PATH), timeout=60.0)
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("PRAGMA busy_timeout = 30000")
        con.row_factory = sqlite3.Row
        _TLS.con = con
    return con


def _thread_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": f"AskChem/1.0 (https://askchem.org; mailto:{DEFAULT_MAILTO})",
        })
        _TLS.session = s
    return s


def _ensure_schema():
    from askchem.db import init_db
    init_db()


def insert_citations(con: sqlite3.Connection, citing_doi: str,
                     cited_dois: Iterable[str], source: str) -> int:
    rows = [(citing_doi, c, source, _utc_now()) for c in cited_dois if c]
    if not rows:
        return 0
    with _DB_LOCK:
        cur = con.executemany(
            "INSERT OR IGNORE INTO citations "
            "(citing_doi, cited_doi, source, fetched_at) VALUES (?,?,?,?)",
            rows,
        )
        con.commit()
    return cur.rowcount or 0


def _eligible_deep_v1_dois() -> list[str]:
    """Read the cached list produced by backfill_edges._eligible_deep_v1_dois."""
    cache = REPO_ROOT / "data" / ".eligible_deep_v1_dois.json"
    if not cache.exists():
        from backfill_edges import _eligible_deep_v1_dois as _e
        with open_db() as con:
            return _e(con)
    return [_normalize_doi(d) for d in json.loads(cache.read_text())]


# ── Crossref ────────────────────────────────────────────────────────────────


def _extract_crossref_refs(payload: dict) -> list[str]:
    msg = payload.get("message") or {}
    refs = msg.get("reference") or []
    out: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        d = _normalize_doi(ref.get("DOI") or ref.get("doi"))
        if d:
            out.append(d)
    # Dedup, preserve order
    seen: set[str] = set()
    deduped: list[str] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped


def fetch_crossref(doi: str, *, force: bool = False,
                   max_retries: int = 4) -> tuple[str, list[str], str]:
    """Returns (status, cited_dois, error).
    status ∈ {'cache_hit', 'fetched', 'empty', 'http_<code>', 'error'}

    Retries 429/5xx with exponential backoff; treats 404 as 'no such DOI' and
    caches an empty stub so we don't refetch.
    """
    cache = _cache_path(CROSSREF_CACHE, doi)
    if cache.exists() and not force:
        try:
            payload = json.loads(cache.read_text())
            return ("cache_hit", _extract_crossref_refs(payload), "")
        except Exception:
            try:
                cache.unlink()
            except Exception:
                pass

    sess = _thread_session()
    url = CROSSREF_BASE + requests.utils.quote(doi, safe="")
    last_err = ""
    last_code = ""
    for attempt in range(max_retries):
        try:
            r = sess.get(url, timeout=DEFAULT_TIMEOUT)
        except Exception as e:
            last_err = f"network: {e}"
            time.sleep(2 ** attempt)
            continue
        time.sleep(CROSSREF_PER_REQ_SLEEP)
        if r.status_code == 200:
            try:
                payload = r.json()
            except Exception as e:
                return ("error", [], f"json: {e}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
            refs = _extract_crossref_refs(payload)
            return ("fetched" if refs else "empty", refs, "")
        if r.status_code == 404:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"message": {"reference": []}, "_status": 404}))
            return ("empty", [], "")
        last_code = str(r.status_code)
        # 429 / 5xx: back off and retry.
        if r.status_code == 429 or 500 <= r.status_code < 600:
            time.sleep(min(2 ** attempt, 30))
            continue
        # Other 4xx: do not retry.
        return (f"http_{r.status_code}", [], r.text[:160])
    return (f"http_{last_code or 'fail'}", [], last_err or "max retries")


def cmd_crossref(args):
    _ensure_schema()
    CROSSREF_CACHE.mkdir(parents=True, exist_ok=True)
    dois = _eligible_deep_v1_dois()
    if args.limit:
        dois = dois[: args.limit]
    n = len(dois)
    print(f"[crossref] {n} DOIs, conc={args.concurrency}, mailto={DEFAULT_MAILTO}")

    counts = {"cache_hit": 0, "fetched": 0, "empty": 0, "error": 0}
    http_codes: dict[str, int] = {}
    inserted_total = 0
    t0 = time.monotonic()

    def work(d: str):
        d_n = _normalize_doi(d)
        status, refs, err = fetch_crossref(d_n)
        n_ins = 0
        if refs:
            con = thread_db()
            n_ins = insert_citations(con, d_n, refs, "crossref")
        return d_n, status, len(refs), n_ins, err

    first_failures_logged = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, d): d for d in dois}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                d_n, status, _nrefs, n_ins, err = fut.result()
            except Exception as e:
                counts["error"] += 1
                print(f"  ! exception: {e}")
                continue
            inserted_total += n_ins
            if status.startswith("http_"):
                code = status[5:]
                http_codes[code] = http_codes.get(code, 0) + 1
                if first_failures_logged < 3:
                    print(f"    [{status}] {d_n}: {err[:120]}")
                    first_failures_logged += 1
            elif status in counts:
                counts[status] += 1
            if i % 200 == 0 or i == n:
                el = time.monotonic() - t0
                rate = i / el if el > 0 else 0
                eta_min = (n - i) / rate / 60 if rate > 0 else 0
                http_str = ",".join(f"{c}={n_}" for c, n_ in sorted(http_codes.items())) or "-"
                print(
                    f"  [{i:>5}/{n}] hits={counts['cache_hit']} fetched={counts['fetched']} "
                    f"empty={counts['empty']} net_err={counts['error']} "
                    f"http=[{http_str}] rows+={inserted_total} "
                    f"rate={rate:.1f}/s eta={eta_min:.1f}min"
                )

    el = time.monotonic() - t0
    print(f"\n[crossref summary]  {n} DOIs in {el/60:.1f} min")
    print(f"  status: {counts}")
    print(f"  http codes: {http_codes}")
    print(f"  citation rows inserted (this run): {inserted_total}")


# ── Semantic Scholar fallback ───────────────────────────────────────────────


def _extract_s2_refs(payload: dict) -> list[str]:
    """S2 references endpoint shape: {"data": [{"citedPaper": {"externalIds":
    {"DOI": "..."}}}, ...]}."""
    data = payload.get("data") or []
    out: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cp = item.get("citedPaper") or {}
        ext = cp.get("externalIds") or {}
        d = _normalize_doi(ext.get("DOI"))
        if d:
            out.append(d)
    seen: set[str] = set()
    deduped: list[str] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped


def fetch_s2(doi: str, *, force: bool = False,
             max_retries: int = 6) -> tuple[str, list[str], str]:
    """Hit S2 references endpoint with exponential backoff on 429/5xx."""
    cache = _cache_path(S2_CACHE, doi)
    if cache.exists() and not force:
        try:
            payload = json.loads(cache.read_text())
            return ("cache_hit", _extract_s2_refs(payload), "")
        except Exception:
            try:
                cache.unlink()
            except Exception:
                pass

    sess = _thread_session()
    url = S2_BASE + requests.utils.quote(doi, safe="") + "/references"
    headers = {}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    params = {"fields": "externalIds", "limit": 200}

    last_err = ""
    last_code = ""
    for attempt in range(max_retries):
        try:
            r = sess.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        except Exception as e:
            last_err = f"network: {e}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code == 200:
            try:
                payload = r.json()
            except Exception as e:
                return ("error", [], f"json: {e}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
            refs = _extract_s2_refs(payload)
            return ("fetched" if refs else "empty", refs, "")
        if r.status_code == 404:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"data": [], "_status": 404}))
            return ("empty", [], "")
        last_code = str(r.status_code)
        if r.status_code == 429 or 500 <= r.status_code < 600:
            # Honor Retry-After if present, else exponential backoff.
            retry_after = r.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after and retry_after.isdigit() \
                else min(2 ** attempt, 30)
            time.sleep(sleep_for)
            continue
        return (f"http_{r.status_code}", [], r.text[:160])
    if last_code == "429":
        return ("rate_limited", [], last_err)
    return (f"http_{last_code or 'fail'}", [], last_err or "max retries")


def cmd_s2_gaps(args):
    """Fill in DOIs where Crossref returned zero references."""
    _ensure_schema()
    S2_CACHE.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("S2_API_KEY")
    conc = args.concurrency or (S2_CONC_AUTHED if api_key else S2_CONC_ANON)
    if not api_key:
        print("[s2-gaps] WARNING: S2_API_KEY not set — using anonymous tier (~1 req/s)")

    eligible = set(_eligible_deep_v1_dois())
    con = open_db()

    # Two kinds of "gap" DOIs:
    #  (1) Crossref tried and returned empty (we have a cache file)
    #  (2) Crossref errored / never tried
    dois_with_xref = {
        r["citing_doi"]
        for r in con.execute(
            "SELECT DISTINCT citing_doi FROM citations WHERE source='crossref'"
        ).fetchall()
    }
    dois_with_s2 = {
        r["citing_doi"]
        for r in con.execute(
            "SELECT DISTINCT citing_doi FROM citations WHERE source='s2'"
        ).fetchall()
    }

    gap_dois = sorted(eligible - dois_with_xref - dois_with_s2)
    # Also include DOIs where crossref ran but produced zero citations.
    s2_cached_empty: list[str] = []
    for d in sorted(eligible & dois_with_xref - dois_with_s2):
        # Check whether the Crossref payload was empty (i.e., no references field
        # or empty list).  We cached every Crossref response; a paper that has
        # any rows in `citations` for source='crossref' is *not* a gap.  So the
        # only way to be here is to have a cache file with empty refs.
        cache = _cache_path(CROSSREF_CACHE, d)
        try:
            if cache.exists():
                payload = json.loads(cache.read_text())
                if not _extract_crossref_refs(payload):
                    s2_cached_empty.append(d)
        except Exception:
            s2_cached_empty.append(d)

    targets = sorted(set(gap_dois + s2_cached_empty))
    if args.limit:
        targets = targets[: args.limit]
    n = len(targets)
    print(f"[s2-gaps] {n} DOIs to fetch from Semantic Scholar (conc={conc})")

    counts = {"cache_hit": 0, "fetched": 0, "empty": 0,
              "rate_limited": 0, "error": 0, "http": 0}
    inserted_total = 0
    t0 = time.monotonic()

    def work(d: str):
        status, refs, _err = fetch_s2(d)
        n_ins = 0
        if refs:
            con_t = thread_db()
            n_ins = insert_citations(con_t, d, refs, "s2")
        return d, status, len(refs), n_ins

    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(work, d): d for d in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _d, status, _nrefs, n_ins = fut.result()
            except Exception as e:
                counts["error"] += 1
                print(f"  ! exception: {e}")
                continue
            inserted_total += n_ins
            if status.startswith("http_"):
                counts["http"] += 1
            elif status in counts:
                counts[status] += 1
            if i % 100 == 0 or i == n:
                el = time.monotonic() - t0
                rate = i / el if el > 0 else 0
                eta_min = (n - i) / rate / 60 if rate > 0 else 0
                print(
                    f"  [{i:>5}/{n}] hits={counts['cache_hit']} fetched={counts['fetched']} "
                    f"empty={counts['empty']} 429s={counts['rate_limited']} "
                    f"http_err={counts['http']} net_err={counts['error']} "
                    f"rows+={inserted_total} rate={rate:.1f}/s eta={eta_min:.1f}min"
                )

    el = time.monotonic() - t0
    print(f"\n[s2-gaps summary]  {n} DOIs in {el/60:.1f} min")
    print(f"  status: {counts}")
    print(f"  citation rows inserted (this run): {inserted_total}")


# ── Report ──────────────────────────────────────────────────────────────────


def cmd_report(args):
    con = open_db()
    eligible = set(_eligible_deep_v1_dois())
    print(f"deep_v1 corpus: {len(eligible):,} DOIs\n")

    # Raw counts.
    total = con.execute("SELECT COUNT(*) c FROM citations").fetchone()["c"]
    by_src = con.execute(
        "SELECT source, COUNT(*) c FROM citations GROUP BY source"
    ).fetchall()
    print(f"Total citation rows: {total:,}")
    for r in by_src:
        print(f"  {r['source']:<10}{r['c']:,}")
    print()

    # Per-citing distribution (any cited).
    rows = con.execute("""
        SELECT citing_doi, COUNT(DISTINCT cited_doi) c
          FROM citations GROUP BY citing_doi
    """).fetchall()
    have = {r["citing_doi"] for r in rows}
    refs_per = sorted([r["c"] for r in rows])
    print(f"Papers with any references fetched: {len(have):,} / {len(eligible):,} "
          f"({len(have)/max(1,len(eligible))*100:.1f}%)")
    if refs_per:
        m = refs_per[len(refs_per)//2]
        p25 = refs_per[len(refs_per)//4]
        p75 = refs_per[len(refs_per)*3//4]
        avg = sum(refs_per) / len(refs_per)
        print(f"  refs/paper: median={m}  p25={p25}  p75={p75}  mean={avg:.1f}  max={refs_per[-1]}")
    missing = sorted(eligible - have)
    print(f"  DOIs with zero references: {len(missing):,}")
    print()

    # In-corpus subset.
    print("Building in-corpus pair index ...", flush=True)
    # Materialize once into a temp table for fast joins.
    con.execute("CREATE TEMP TABLE _eligible (doi TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO _eligible VALUES (?)", [(d,) for d in eligible])
    pair_count = con.execute("""
        SELECT COUNT(*) c FROM (
          SELECT DISTINCT c.citing_doi, c.cited_doi
            FROM citations c
            JOIN _eligible a ON a.doi = c.citing_doi
            JOIN _eligible b ON b.doi = c.cited_doi
        )
    """).fetchone()["c"]
    print(f"\nIn-corpus citation pairs (citing AND cited in deep_v1): {pair_count:,}")

    in_corpus_per_paper = con.execute("""
        SELECT citing_doi, COUNT(DISTINCT cited_doi) c
          FROM citations
         WHERE citing_doi IN (SELECT doi FROM _eligible)
           AND cited_doi  IN (SELECT doi FROM _eligible)
         GROUP BY citing_doi
    """).fetchall()
    have_in = {r["citing_doi"] for r in in_corpus_per_paper}
    print(f"Papers with at least 1 in-corpus reference: {len(have_in):,}")
    if in_corpus_per_paper:
        vals = sorted([r["c"] for r in in_corpus_per_paper])
        m = vals[len(vals)//2]
        avg = sum(vals) / len(vals)
        print(f"  in-corpus refs/paper: median={m}  mean={avg:.1f}  max={vals[-1]}")
    print()

    # Subarea (HER) restricted view.
    if args.subarea:
        view_id, path = args.subarea.split(":", 1)
        sub_dois = {
            r["source_doi"]
            for r in con.execute(
                """
                SELECT DISTINCT c.source_doi
                  FROM claim_view_map cvm
                  JOIN claims c ON c.claim_id = cvm.claim_id
                 WHERE cvm.view_id = ? AND cvm.path = ?
                   AND c.extraction_version = 'deep_v1'
                """,
                (view_id, path),
            ).fetchall()
        }
        sub_dois &= eligible
        print(f"Subarea {args.subarea}: {len(sub_dois):,} deep_v1 papers")
        if sub_dois:
            con.execute("CREATE TEMP TABLE _sub (doi TEXT PRIMARY KEY)")
            con.executemany("INSERT INTO _sub VALUES (?)",
                            [(d,) for d in sub_dois])
            sub_pairs = con.execute("""
                SELECT COUNT(*) c FROM (
                  SELECT DISTINCT c.citing_doi, c.cited_doi
                    FROM citations c
                    JOIN _sub a ON a.doi = c.citing_doi
                    JOIN _sub b ON b.doi = c.cited_doi
                )
            """).fetchone()["c"]
            print(f"  in-subarea citation pairs: {sub_pairs:,}")


# ── Entry ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    cr = sub.add_parser("crossref", help="Fetch references for all deep_v1 DOIs")
    cr.add_argument("--concurrency", type=int, default=CROSSREF_CONC)
    cr.add_argument("--limit", type=int, default=None)
    cr.set_defaults(func=cmd_crossref)

    s2 = sub.add_parser("s2-gaps", help="S2 references for DOIs with empty/missing Crossref refs")
    s2.add_argument("--concurrency", type=int, default=None)
    s2.add_argument("--limit", type=int, default=None)
    s2.set_defaults(func=cmd_s2_gaps)

    rp = sub.add_parser("report", help="Distribution and in-corpus pair counts")
    rp.add_argument("--subarea", type=str, default=None,
                    help="view_id:path (e.g. by_application:energy/water_splitting/hydrogen_evolution_reaction)")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
