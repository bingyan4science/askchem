"""Citation-graph-driven cross-paper claim edge extractor.

Replaces the random-nearest-neighbor cross extractor in backfill_edges.py with
a focused per-pair LLM call: for every (citing, cited) pair in the citations
table where both papers are in the deep_v1 corpus, ask Gemini "for each claim
in citing paper A, does it use/extend/contradict/etc any claim in cited paper
B?".

Subcommands:
    pilot   Run on N pairs restricted to a subarea (e.g. HER catalysis).
    full    Run on ALL in-corpus citation pairs (ignores subarea filter).
    status  Report progress and cost-to-date by extractor tag.
    purge   Delete edges + edge_jobs rows for one extractor tag.

Resumability: per-pair status is tracked in `edge_jobs` keyed by
(`paper_doi="{citing}|{cited}"`, mode='cross_citation', extractor=<tag>).
Re-run with --resume to skip pairs already marked 'done'.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.models import ClaimEdge, CROSS_PAPER_EDGE_TYPES  # noqa: E402
from backfill_edges import (  # noqa: E402
    GATEWAY, PROVIDER, MODEL,
    PRICE_IN_PER_M, PRICE_OUT_PER_M,
    DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_S, DEFAULT_RETRIES,
    DEFAULT_MAX_CLAIMS_PER_PAPER,
    _compact, validate_edges, estimate_cost,
)

DEFAULT_PILOT_TAG = "cross_citation_v1_pilot"
DEFAULT_FULL_TAG = "cross_citation_v1"
PAIR_MODE = "cross_citation"

_DB_LOCK = threading.Lock()
_TLS = threading.local()


CROSS_CITATION_PROMPT = """Paper A (citing) cites Paper B (cited).

For each claim in A, identify whether it specifically uses, extends, supersedes, contradicts, or cites_as_evidence any claim in B.

Edge types:
- uses_method_of      : claim in A uses an experimental/computational method established in a claim in B
- uses_assumption_of  : claim in A relies on an assumption/result from a claim in B
- extends             : claim in A builds on or generalizes a claim in B
- supersedes          : claim in A is a clear improvement over a claim in B under comparable conditions
- contradicts         : claim in A disagrees with a claim in B under comparable conditions
- cites_as_evidence   : claim in A invokes a claim in B as supporting evidence

Be conservative — only emit an edge with strong textual grounds. Many citations are formal/background and produce no claim-level edges; in that case return {{"edges": []}}.

Return JSON of the form:
{{"edges": [{{"from": "<A_claim_id>", "to": "<B_claim_id>", "type": "<edge_type>", "confidence": "high|medium|low", "evidence": "<one sentence rationale or quote>"}}]}}

Rules:
- "from" MUST be a claim_id from Paper A (the citing paper).
- "to"   MUST be a claim_id from Paper B (the cited paper).
- Skip self-loops; do not duplicate edges (same from/to/type once at most).
- Most pairs will produce 0-3 edges.

Paper A (citing): {a_title}
Paper A DOI:     {a_doi}
A's claims (JSON):
{a_claims_json}

Paper B (cited): {b_title}
Paper B DOI:     {b_doi}
B's claims (JSON):
{b_claims_json}
"""


# ── DB helpers ───────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=60.0, check_same_thread=False)
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
        _TLS.session = s
    return s


def call_gemini(prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS,
                retries: int = DEFAULT_RETRIES) -> dict:
    """Same shape as backfill_edges.call_gemini, but with a per-thread session
    local to this module (so the parent backfill_edges' TLS is not shared)."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "x-portkey-api-key": api_key,
        "x-portkey-provider": PROVIDER,
        "Content-Type": "application/json",
    }
    sess = _thread_session()
    last_err = ""
    for attempt in range(retries):
        try:
            r = sess.post(GATEWAY, headers=headers, json=body, timeout=DEFAULT_TIMEOUT_S)
        except Exception as e:
            last_err = f"network: {e}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            last_err = f"http {r.status_code}: {r.text[:200]}"
            time.sleep(2 ** attempt)
            continue
        try:
            resp = r.json()
        except Exception as e:
            last_err = f"json: {e}"
            time.sleep(2 ** attempt)
            continue
        choices = resp.get("choices") or []
        if not choices:
            last_err = f"no choices: {json.dumps(resp)[:200]}"
            time.sleep(2 ** attempt)
            continue
        msg = choices[0].get("message", {})
        content = (msg.get("content") or "").strip()
        finish = choices[0].get("finish_reason")
        usage = resp.get("usage") or {}
        if not content:
            last_err = f"empty content (finish={finish}, usage={usage})"
            time.sleep(2 ** attempt)
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            last_err = f"content not JSON: {e}: {content[:200]}"
            time.sleep(2 ** attempt)
            continue
        return {"parsed": parsed, "usage": usage, "finish_reason": finish}
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_err}")


def fetch_claims_for(con: sqlite3.Connection, doi: str, *, max_claims: int) -> list[dict]:
    """Case-insensitive lookup: citation graph stores normalized lowercase DOIs
    while the claims table preserves the source's original casing.  LOWER()
    in the predicate matches both."""
    rows = con.execute(
        """SELECT claim_id, data FROM claims
            WHERE LOWER(source_doi) = LOWER(?) AND extraction_version = 'deep_v1'
            ORDER BY claim_id LIMIT ?""",
        (doi, max_claims),
    ).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["data"])
        d["claim_id"] = r["claim_id"]
        out.append(d)
    return out


def fetch_paper_title(con: sqlite3.Connection, doi: str) -> str:
    r = con.execute(
        "SELECT title FROM sources WHERE LOWER(doi) = LOWER(?)", (doi,),
    ).fetchone()
    return (r["title"] if r else "") or ""


def insert_edges(con: sqlite3.Connection, edges: list[ClaimEdge]) -> int:
    if not edges:
        return 0
    rows = [
        (e.from_claim_id, e.edge_type,
         e.to_claim_id or "", e.to_doi or "",
         e.confidence, e.evidence,
         e.extractor, e.extracted_at)
        for e in edges
    ]
    with _DB_LOCK:
        cur = con.executemany(
            """INSERT OR IGNORE INTO claim_edges
               (from_claim_id, edge_type, to_claim_id, to_doi,
                confidence, evidence, extractor, extracted_at)
               VALUES (?,?,?,?,?,?,?,?)""", rows)
        con.commit()
    return cur.rowcount or 0


def record_job(con: sqlite3.Connection, *,
               pair_key: str, extractor: str, status: str,
               edges_inserted: int, tokens_in: int, tokens_out: int,
               error: str | None, started_at: str) -> None:
    with _DB_LOCK:
        con.execute("""
            INSERT INTO edge_jobs (
                paper_doi, mode, extractor, status, edges_inserted,
                tokens_in, tokens_out, error, started_at, finished_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(paper_doi, mode, extractor) DO UPDATE SET
              status         = excluded.status,
              edges_inserted = excluded.edges_inserted,
              tokens_in      = excluded.tokens_in,
              tokens_out     = excluded.tokens_out,
              error          = excluded.error,
              started_at     = excluded.started_at,
              finished_at    = excluded.finished_at
        """, (pair_key, PAIR_MODE, extractor, status, edges_inserted,
              tokens_in, tokens_out, error, started_at,
              datetime.utcnow().isoformat() + "Z"))
        con.commit()


# ── Pair selection ───────────────────────────────────────────────────────────


def _eligible_dois() -> set[str]:
    cache = REPO_ROOT / "data" / ".eligible_deep_v1_dois.json"
    return {d.lower().strip() for d in json.loads(cache.read_text())}


def select_pairs(
    con: sqlite3.Connection, *,
    extractor: str, resume: bool, limit: int | None,
    subarea_view: str | None = None,
    subarea_path: str | None = None,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """Return list of (citing_doi, cited_doi) where both are in deep_v1 and
    (if subarea_view+path given) at least one endpoint is in that subarea.

    NOTE: We require ONE endpoint in the subarea — not both — because HER
    papers often cite foundational work outside the strict HER subtree
    (e.g. DFT methods papers, electrochem theory).  This keeps the candidate
    pool meaningful while preventing irrelevant cross-domain citations.
    """
    eligible = _eligible_dois()

    if subarea_view and subarea_path:
        sub = {
            r["source_doi"].lower().strip()
            for r in con.execute(
                """
                SELECT DISTINCT c.source_doi
                  FROM claim_view_map cvm
                  JOIN claims c ON c.claim_id = cvm.claim_id
                 WHERE cvm.view_id = ? AND cvm.path = ?
                   AND c.extraction_version = 'deep_v1'
                """,
                (subarea_view, subarea_path),
            ).fetchall()
        }
        sub &= eligible
        print(f"  subarea {subarea_view}:{subarea_path}: {len(sub):,} papers")
    else:
        sub = eligible

    # Pull candidate pairs.  Use TEMP tables to avoid blowing through SQLite's
    # SQLITE_MAX_VARIABLE_NUMBER (32766 default) — the eligible set alone is
    # 23K DOIs.  Drop+create with IF NOT EXISTS isn't allowed for temp tables
    # without explicit DROP; use a per-call random suffix instead.
    target_set = sub if (subarea_view and subarea_path) else eligible
    suffix = str(random.randint(0, 1_000_000))
    t_target = f"_target_{suffix}"
    t_elig = f"_elig_{suffix}"
    con.execute(f"CREATE TEMP TABLE {t_target} (doi TEXT PRIMARY KEY)")
    con.execute(f"CREATE TEMP TABLE {t_elig} (doi TEXT PRIMARY KEY)")
    con.executemany(f"INSERT OR IGNORE INTO {t_target} VALUES (?)",
                    [(d,) for d in target_set])
    con.executemany(f"INSERT OR IGNORE INTO {t_elig} VALUES (?)",
                    [(d,) for d in eligible])
    rows = con.execute(f"""
        SELECT DISTINCT c.citing_doi, c.cited_doi
          FROM citations c
          JOIN {t_target} a ON a.doi = c.citing_doi
          JOIN {t_elig}   b ON b.doi = c.cited_doi
    """).fetchall()
    con.execute(f"DROP TABLE {t_target}")
    con.execute(f"DROP TABLE {t_elig}")
    pairs = [(r["citing_doi"], r["cited_doi"]) for r in rows]
    print(f"  candidate pairs (citing in target, cited in eligible): {len(pairs):,}")

    if resume:
        done = {
            r["paper_doi"]
            for r in con.execute(
                "SELECT paper_doi FROM edge_jobs "
                " WHERE mode=? AND extractor=? AND status='done'",
                (PAIR_MODE, extractor),
            ).fetchall()
        }
        pairs = [(c, t) for c, t in pairs if f"{c}|{t}" not in done]
        print(f"  pairs after --resume: {len(pairs):,}")

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    if limit is not None:
        pairs = pairs[:limit]
    return pairs


# ── Worker ───────────────────────────────────────────────────────────────────


def process_pair(
    citing: str, cited: str, *, extractor: str, max_claims: int,
) -> dict:
    con = thread_db()
    pair_key = f"{citing}|{cited}"
    started = datetime.utcnow().isoformat() + "Z"

    a_claims = fetch_claims_for(con, citing, max_claims=max_claims)
    b_claims = fetch_claims_for(con, cited, max_claims=max_claims)
    if not a_claims or not b_claims:
        record_job(con, pair_key=pair_key, extractor=extractor,
                   status="skipped", edges_inserted=0,
                   tokens_in=0, tokens_out=0,
                   error=f"empty claims (a={len(a_claims)} b={len(b_claims)})",
                   started_at=started)
        return {"pair": pair_key, "status": "skipped", "edges": 0,
                "tokens_in": 0, "tokens_out": 0}

    a_title = fetch_paper_title(con, citing)
    b_title = fetch_paper_title(con, cited)
    a_compact = [_compact(c) for c in a_claims]
    b_compact = [_compact(c) for c in b_claims]
    prompt = CROSS_CITATION_PROMPT.format(
        a_title=a_title, a_doi=citing,
        a_claims_json=json.dumps(a_compact, indent=1),
        b_title=b_title, b_doi=cited,
        b_claims_json=json.dumps(b_compact, indent=1),
    )

    try:
        resp = call_gemini(prompt)
    except Exception as e:
        record_job(con, pair_key=pair_key, extractor=extractor,
                   status="failed", edges_inserted=0,
                   tokens_in=0, tokens_out=0,
                   error=str(e)[:500], started_at=started)
        return {"pair": pair_key, "status": "failed", "edges": 0,
                "tokens_in": 0, "tokens_out": 0, "error": str(e)[:200]}

    valid_from = {c["claim_id"] for c in a_claims}
    valid_to = {c["claim_id"] for c in b_claims}
    edges, _problems = validate_edges(
        resp["parsed"].get("edges", []),
        valid_from_ids=valid_from, valid_to_ids=valid_to,
        allowed_types=CROSS_PAPER_EDGE_TYPES,
        extractor=extractor, now=started,
    )
    # validate_edges doesn't know which paper the cited claims came from;
    # for cross-paper edges every "to" claim belongs to `cited` by construction.
    for e in edges:
        e.to_doi = cited
    inserted = insert_edges(con, edges)
    usage = resp.get("usage") or {}
    t_in = usage.get("prompt_tokens") or 0
    t_out = usage.get("completion_tokens") or 0
    record_job(con, pair_key=pair_key, extractor=extractor,
               status="done", edges_inserted=inserted,
               tokens_in=t_in, tokens_out=t_out,
               error=None, started_at=started)
    return {"pair": pair_key, "status": "done", "edges": inserted,
            "tokens_in": t_in, "tokens_out": t_out}


# ── Runner ───────────────────────────────────────────────────────────────────


def run_pool(pairs: list[tuple[str, str]], *, extractor: str,
             concurrency: int, max_claims: int, log_every: int = 5):
    n = len(pairs)
    if n == 0:
        print(f"[{extractor}] no pairs to process")
        return
    print(f"[{extractor}] starting {n} pairs, concurrency={concurrency}")
    t0 = time.monotonic()
    done = failed = skipped = edges_total = 0
    t_in_total = t_out_total = 0
    pairs_with_edges = 0
    last_print = 0.0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(
            process_pair, c, t, extractor=extractor, max_claims=max_claims
        ): (c, t) for c, t in pairs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                failed += 1
                print(f"  ! unexpected: {e}")
                continue
            status = r.get("status")
            if status == "done":
                done += 1
                if r.get("edges", 0) > 0:
                    pairs_with_edges += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            edges_total += r.get("edges", 0)
            t_in_total += r.get("tokens_in", 0)
            t_out_total += r.get("tokens_out", 0)

            now = time.monotonic()
            processed = done + failed + skipped
            if processed % log_every == 0 or now - last_print > 30:
                last_print = now
                el = now - t0
                rate = processed / el if el > 0 else 0
                eta = (n - processed) / rate if rate > 0 else 0
                cost = estimate_cost(t_in_total, t_out_total)
                proj = cost * (n / processed) if processed > 0 else 0
                yield_pct = (pairs_with_edges / done * 100) if done else 0
                print(
                    f"  [{processed:>4}/{n}] done={done} skip={skipped} fail={failed} "
                    f"edges={edges_total} (yield={yield_pct:.0f}%) "
                    f"cost=${cost:.2f} (proj=${proj:.2f}) "
                    f"rate={rate*60:.1f}/min eta={eta/60:.1f}min"
                )

    el = time.monotonic() - t0
    cost = estimate_cost(t_in_total, t_out_total)
    print(f"\n[{extractor} summary]")
    print(f"  pairs:  done={done} skipped={skipped} failed={failed} of {n}")
    print(f"  edges inserted: {edges_total}")
    if done:
        print(f"  yield: {pairs_with_edges}/{done} pairs with ≥1 edge "
              f"({pairs_with_edges/done*100:.1f}%)")
        print(f"  edges/done-pair: {edges_total/done:.2f}")
    print(f"  tokens: in={t_in_total:,} out={t_out_total:,}")
    print(f"  cost:   ${cost:.2f}")
    print(f"  wall:   {el/60:.1f} min")


# ── Subcommands ──────────────────────────────────────────────────────────────


def _ensure_schema():
    from askchem.db import init_db
    init_db()


def cmd_pilot(args):
    _ensure_schema()
    extractor = args.extractor_tag or DEFAULT_PILOT_TAG
    con = open_db()
    print(f"[pilot] extractor={extractor}, subarea={args.subarea}")
    view, path = (args.subarea.split(":", 1) if args.subarea else (None, None))
    pairs = select_pairs(
        con, extractor=extractor, resume=args.resume,
        limit=args.limit, subarea_view=view, subarea_path=path,
        seed=args.seed,
    )
    if args.dry_run:
        print(f"[pilot dry-run] would process {len(pairs)} pairs")
        return
    run_pool(pairs, extractor=extractor,
             concurrency=args.concurrency, max_claims=args.max_claims)


def cmd_full(args):
    _ensure_schema()
    extractor = args.extractor_tag or DEFAULT_FULL_TAG
    con = open_db()
    print(f"[full] extractor={extractor}")
    pairs = select_pairs(
        con, extractor=extractor, resume=args.resume,
        limit=args.limit, seed=args.seed,
    )
    if args.dry_run:
        print(f"[full dry-run] would process {len(pairs)} pairs")
        return
    run_pool(pairs, extractor=extractor,
             concurrency=args.concurrency, max_claims=args.max_claims)


def cmd_status(_args):
    con = open_db()
    print(f"{'extractor':<40} {'done':>8} {'skip':>6} {'fail':>6} "
          f"{'edges':>8} {'cost':>10}")
    print("-" * 90)
    for r in con.execute("""
        SELECT extractor,
               SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skp,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS fail,
               SUM(edges_inserted) AS e,
               SUM(tokens_in) AS ti, SUM(tokens_out) AS to_
          FROM edge_jobs
         WHERE mode = ?
         GROUP BY extractor
         ORDER BY extractor
    """, (PAIR_MODE,)).fetchall():
        cost = estimate_cost(r["ti"] or 0, r["to_"] or 0)
        print(f"{r['extractor']:<40} {r['done'] or 0:>8} {r['skp'] or 0:>6} "
              f"{r['fail'] or 0:>6} {r['e'] or 0:>8} ${cost:>9.2f}")


def cmd_purge(args):
    if not args.extractor_tag:
        print("--extractor-tag required for purge", file=sys.stderr)
        sys.exit(1)
    con = open_db()
    n_e = con.execute("DELETE FROM claim_edges WHERE extractor=?",
                      (args.extractor_tag,)).rowcount
    n_j = con.execute("DELETE FROM edge_jobs WHERE extractor=? AND mode=?",
                      (args.extractor_tag, PAIR_MODE)).rowcount
    con.commit()
    print(f"deleted {n_e} edges and {n_j} job rows for extractor={args.extractor_tag}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--concurrency", type=int, default=8)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--resume", action="store_true")
        sp.add_argument("--extractor-tag", type=str, default=None)
        sp.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS_PER_PAPER)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--dry-run", action="store_true")

    sp_pi = sub.add_parser("pilot", help="Run on N pairs in a subarea")
    common(sp_pi)
    sp_pi.add_argument("--subarea", type=str, default=None,
                       help="view_id:path (e.g. by_application:energy/water_splitting/hydrogen_evolution_reaction)")
    sp_pi.set_defaults(func=cmd_pilot)

    sp_fu = sub.add_parser("full", help="Run on ALL in-corpus citation pairs")
    common(sp_fu)
    sp_fu.set_defaults(func=cmd_full)

    sp_st = sub.add_parser("status", help="Progress per extractor tag")
    sp_st.set_defaults(func=cmd_status)

    sp_pg = sub.add_parser("purge", help="Delete edges + job rows for one extractor tag")
    sp_pg.add_argument("--extractor-tag", required=True)
    sp_pg.set_defaults(func=cmd_purge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
