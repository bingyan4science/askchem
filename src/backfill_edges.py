"""Corpus-scale claim-edge backfill via Gemini (PortKey gateway).

Builds typed claim-to-claim edges across the deep_v1 portion of the corpus.

Subcommands:
    preflight   Run intra+cross on N random deep_v1 papers (default 200) and
                report the actual token cost, edge density, latency, and
                error rate. Use this to validate the budget before launching
                a full pass.
    intra       Run intra-paper LLM edge extraction on all eligible papers.
    cross       Run cross-paper LLM edge extraction on all eligible papers.
    status      Show progress and cost-to-date for each (mode, extractor).
    purge       Delete edge rows + edge_jobs rows for a given extractor tag.

Common flags:
    --concurrency N      Parallel Gemini calls (default 8)
    --limit N            Cap number of papers processed
    --resume             Skip papers already marked 'done' for this extractor
    --extractor-tag T    Override the extractor identifier
    --candidates K       Cross only: number of candidate papers per source (default 5)
    --max-claims N       Per-paper claim cap to keep prompts under control (default 80)
    --dry-run            Plan only, no API calls or DB writes

Resumability:
    Per-paper status is tracked in `edge_jobs(paper_doi, mode, extractor)`.
    Re-running with --resume picks up exactly where a previous run left off.
    `claim_edges` already has UNIQUE(from, type, to_claim, to_doi, extractor),
    so duplicate inserts are silently ignored — safe to re-process.

Pricing reference (Gemini 3.1 Pro Preview on Vertex, ≤200K context):
    input  $2 / 1M tokens
    output $12 / 1M tokens (reasoning tokens count as output)
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
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"

sys.path.insert(0, str(REPO_ROOT / "src"))
from askchem.models import (  # noqa: E402
    ClaimEdge, INTRA_PAPER_EDGE_TYPES, CROSS_PAPER_EDGE_TYPES,
)

# ── Constants ─────────────────────────────────────────────────────────────────

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

INTRA_EXTRACTOR_DEFAULT = "intra_llm_gemini_v1"
CROSS_EXTRACTOR_DEFAULT = "cross_llm_gemini_v1"

# Vertex Gemini 3.1 Pro Preview pricing (USD per 1M tokens, ≤200K context).
PRICE_IN_PER_M = 2.00
PRICE_OUT_PER_M = 12.00

DEFAULT_MAX_TOKENS = 32768
DEFAULT_TIMEOUT_S = 300
DEFAULT_RETRIES = 3
DEFAULT_MAX_CLAIMS_PER_PAPER = 80

# ── Prompts ──────────────────────────────────────────────────────────────────

INTRA_PROMPT = """You are analyzing the relationships between scientific claims extracted from a single chemistry paper.

Below are all of the claims from one paper, each with a stable claim_id and structured fields. Identify directed typed edges between these claims that capture the paper's internal scientific reasoning.

Edge types (intra-paper only):
- supports        : claim X provides evidence for, or directly justifies, claim Y
- assumes         : claim X presupposes claim Y as a premise
- bounded_by      : claim X is restricted by the limitation/scope/condition stated in claim Y
- interprets      : claim X is an interpretation of an observation reported in claim Y
- derives_from    : claim X is computed/derived from inputs/results in claim Y
- sub_step_of     : claim X is a sub-component of the larger procedure in claim Y

Return JSON of the form:
{{"edges": [{{"from": "<claim_id>", "to": "<claim_id>", "type": "<edge_type>", "confidence": "high|medium|low", "evidence": "<one sentence rationale or quote>"}}]}}

Rules:
- Both endpoints must be claim_ids from the input list.
- Skip self-loops (from == to).
- Only emit edges you have clear textual grounds for; skip speculative ones.
- Do not duplicate edges (same from/to/type appears at most once).
- A typical paper has 5-30 internal edges; if you would emit zero, return {{"edges": []}}.

Paper title: {title}
Paper DOI:   {doi}

Claims (JSON):
{claims_json}
"""

CROSS_PROMPT = """You are analyzing whether claims in one chemistry paper depend on claims in other papers.

You are given (a) the source paper's claims and (b) a candidate pool of claims from a small number of related chemistry papers selected by semantic similarity. Identify directed cross-paper edges from source-paper claims to candidate-pool claims.

Edge types (cross-paper only):
- uses_method_of      : source claim uses an experimental/computational method established in another claim
- uses_assumption_of  : source claim relies on an assumption/result established in another claim
- extends             : source claim builds on or generalizes another claim
- supersedes          : source claim is a clear improvement over another claim under comparable conditions
- contradicts         : source claim disagrees with another claim under comparable conditions
- cites_as_evidence   : source claim invokes another claim as supporting evidence

Return JSON of the form:
{{"edges": [{{"from": "<source_claim_id>", "to": "<candidate_claim_id>", "type": "<edge_type>", "confidence": "high|medium|low", "evidence": "<one sentence rationale>"}}]}}

Rules:
- "from" MUST be a claim_id from the source paper.
- "to" MUST be a claim_id from the candidate pool.
- Only emit edges with strong textual evidence; do not speculate.
- Most papers will have 0-5 cross-paper edges to this small candidate pool — return {{"edges": []}} if none apply.

Source paper title: {source_title}
Source paper DOI:   {source_doi}

Source paper claims (JSON):
{source_claims_json}

Candidate pool from related papers (JSON):
{other_claims_json}
"""

# ── Compact claim representation (stable across modes) ───────────────────────


def _compact(c: dict) -> dict:
    """Drop noise / empty fields; keep what the LLM actually needs to reason."""
    out = {
        "claim_id":       c.get("claim_id"),
        "claim_type":     c.get("claim_type"),
        "verbatim_quote": (c.get("verbatim_quote") or "")[:600],
    }
    for k in (
        "reaction_type", "subject", "subject_smiles", "property_name",
        "value", "unit", "measurement_method",
        "process_described", "steps", "key_intermediates",
        "technique_name", "what_it_achieves", "key_innovation", "limitations",
        "compared_items", "metric", "comparison_result",
        "hypothesis_text", "limitation_text", "direction_text",
        "finding_text", "why_surprising",
        "rationale", "evidence", "assumption", "epistemic_role",
    ):
        v = c.get(k)
        if v:
            if isinstance(v, str):
                v = v[:400]
            out[k] = v
    if c.get("reactants"):
        out["reactants"] = [r.get("name") for r in c["reactants"][:5] if isinstance(r, dict)]
    if c.get("products"):
        out["products"] = [r.get("name") for r in c["products"][:5] if isinstance(r, dict)]
    if c.get("conditions"):
        out["conditions"] = {k: v for k, v in c["conditions"].items() if v}
    if c.get("outcomes"):
        out["outcomes"] = {k: v for k, v in c["outcomes"].items() if v not in (None, "", "null")}
    return out


# ── DB helpers ───────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=60.0, check_same_thread=False)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA busy_timeout = 30000")
    con.row_factory = sqlite3.Row
    return con


_DB_LOCK = threading.Lock()
_TLS = threading.local()


def thread_db() -> sqlite3.Connection:
    """Per-thread connection. SQLite + macOS does not tolerate sharing one
    connection across threads (segfaults under concurrent reads/writes), so
    every worker gets its own.  WAL mode lets readers and writers coexist.
    """
    con = getattr(_TLS, "con", None)
    if con is None:
        con = sqlite3.connect(str(DB_PATH), timeout=60.0)
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("PRAGMA busy_timeout = 30000")
        con.row_factory = sqlite3.Row
        _TLS.con = con
    return con


_ELIGIBLE_CACHE = REPO_ROOT / "data" / ".eligible_deep_v1_dois.json"


def _eligible_deep_v1_dois(con: sqlite3.Connection, *, refresh: bool = False) -> list[str]:
    """Cached list of DOIs with ≥2 deep_v1 claims.

    The underlying GROUP BY over ~1.7M claim rows costs ~100s per invocation;
    cache it so subcommands launch instantly. Pass refresh=True to rebuild.
    """
    if not refresh and _ELIGIBLE_CACHE.exists():
        try:
            return json.loads(_ELIGIBLE_CACHE.read_text())
        except Exception:
            pass
    rows = con.execute("""
        SELECT source_doi
          FROM claims
         WHERE extraction_version = 'deep_v1'
         GROUP BY source_doi
         HAVING COUNT(*) >= 2
    """).fetchall()
    dois = [r["source_doi"] for r in rows]
    _ELIGIBLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _ELIGIBLE_CACHE.write_text(json.dumps(dois))
    return dois


def select_eligible_papers(
    con: sqlite3.Connection, *,
    mode: str, extractor: str,
    resume: bool, limit: int | None,
    sample_seed: int | None = None,
    refresh_cache: bool = False,
) -> list[str]:
    """Return DOIs of deep_v1 papers with ≥2 claims that need processing."""
    all_dois = _eligible_deep_v1_dois(con, refresh=refresh_cache)

    if resume:
        done = {
            r["paper_doi"]
            for r in con.execute(
                "SELECT paper_doi FROM edge_jobs "
                " WHERE mode=? AND extractor=? AND status='done'",
                (mode, extractor),
            ).fetchall()
        }
        all_dois = [d for d in all_dois if d not in done]

    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(all_dois)

    if limit is not None:
        all_dois = all_dois[:limit]
    return all_dois


def fetch_claims_for(con: sqlite3.Connection, doi: str, *, max_claims: int) -> list[dict]:
    rows = con.execute(
        """SELECT claim_id, data FROM claims
            WHERE source_doi = ? AND extraction_version = 'deep_v1'
            ORDER BY claim_id LIMIT ?""",
        (doi, max_claims),
    ).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["data"])
        d["claim_id"] = r["claim_id"]
        out.append(d)
    return out


def fetch_paper_meta(con: sqlite3.Connection, doi: str) -> tuple[str, str]:
    """Return (title, abstract) for a DOI."""
    r = con.execute(
        "SELECT title, abstract FROM sources WHERE doi = ?",
        (doi,),
    ).fetchone()
    if not r:
        return ("", "")
    return (r["title"] or "", r["abstract"] or "")


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
    # Serialize ALL writers globally; SQLite WAL only allows one writer at a time
    # and concurrent attempts on the same DB file would otherwise crash hard.
    with _DB_LOCK:
        cur = con.executemany(
            """INSERT OR IGNORE INTO claim_edges
               (from_claim_id, edge_type, to_claim_id, to_doi,
                confidence, evidence, extractor, extracted_at)
               VALUES (?,?,?,?,?,?,?,?)""", rows)
        con.commit()
    return cur.rowcount or 0


def record_job(con: sqlite3.Connection, *,
               doi: str, mode: str, extractor: str, status: str,
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
        """, (doi, mode, extractor, status, edges_inserted,
              tokens_in, tokens_out, error, started_at,
              datetime.utcnow().isoformat() + "Z"))
        con.commit()


# ── Gemini call (synchronous, per-thread) ────────────────────────────────────


def _thread_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = requests.Session()
        _TLS.session = s
    return s


def call_gemini(prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS,
                retries: int = DEFAULT_RETRIES) -> dict:
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
            last_err = f"json: {e}: {r.text[:200]}"
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
            last_err = f"empty content (finish_reason={finish}, usage={usage})"
            time.sleep(2 ** attempt)
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            last_err = f"content not JSON (finish_reason={finish}): {e}: {content[:200]}"
            time.sleep(2 ** attempt)
            continue
        return {"parsed": parsed, "usage": usage, "finish_reason": finish}
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_err}")


# ── Edge validation ──────────────────────────────────────────────────────────


def validate_edges(
    raw: list, *, valid_from_ids: set[str], valid_to_ids: set[str],
    allowed_types: set[str], extractor: str, now: str,
) -> tuple[list[ClaimEdge], list[str]]:
    edges: list[ClaimEdge] = []
    problems: list[str] = []
    seen: set[tuple] = set()
    if not isinstance(raw, list):
        return edges, [f"edges field is not a list: {type(raw).__name__}"]
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"#{i}: not dict")
            continue
        f = (item.get("from") or "").strip()
        t = (item.get("to") or "").strip()
        et = (item.get("type") or "").strip()
        if et not in allowed_types:
            problems.append(f"#{i}: bad type {et!r}")
            continue
        if f == t:
            continue
        if f not in valid_from_ids or t not in valid_to_ids:
            continue
        key = (f, et, t)
        if key in seen:
            continue
        seen.add(key)
        conf = item.get("confidence", "medium")
        if conf not in {"high", "medium", "low"}:
            conf = "medium"
        edges.append(ClaimEdge(
            from_claim_id=f, edge_type=et, to_claim_id=t,
            confidence=conf, evidence=(item.get("evidence") or "")[:500],
            extractor=extractor, extracted_at=now,
        ))
    return edges, problems


# ── Cross-paper candidate retrieval (FAISS over claim embeddings) ───────────

_EMBEDS_LOADED = False


def _ensure_embeddings():
    """Load FAISS index AND warm the sentence-transformer model on the main
    thread.  Construction of the encoder is now lock-protected inside
    embeddings._get_model, so worker threads can call .encode() concurrently;
    we still warm here to amortize the ~2s init cost out of the worker timeline.
    """
    global _EMBEDS_LOADED
    if _EMBEDS_LOADED:
        return
    from askchem.embeddings import load_embeddings, vector_search
    load_embeddings()
    _ = vector_search("warmup probe for embeddings", top_k=2, min_score=0.0)
    _EMBEDS_LOADED = True


def find_candidate_papers(
    con: sqlite3.Connection, *, source_doi: str, source_title: str,
    source_claims: list[dict], k_papers: int,
    deep_v1_doi_set: set[str],
) -> list[str]:
    """Return up to k_papers DOIs (excluding source_doi) of deep_v1 papers
    most similar to the source paper, via FAISS over claim embeddings."""
    _ensure_embeddings()
    from askchem.embeddings import vector_search

    parts = [source_title]
    for c in source_claims[:5]:
        q = c.get("verbatim_quote")
        if q:
            parts.append(q[:200])
    query = " ".join(p for p in parts if p)[:1500]

    # Pull a wide net so we have enough candidates after dedup-by-paper and
    # the deep_v1 filter; ~50 nearest claims usually maps to ~10-20 papers.
    hits = vector_search(query, top_k=120, min_score=0.0)
    seen: set[str] = set()
    out: list[str] = []
    if not hits:
        return out
    placeholders = ",".join("?" * len(hits))
    rows = con.execute(
        f"SELECT claim_id, source_doi FROM claims WHERE claim_id IN ({placeholders})",
        [cid for cid, _ in hits],
    ).fetchall()
    cid_to_doi = {r["claim_id"]: r["source_doi"] for r in rows}
    for cid, _score in hits:
        d = cid_to_doi.get(cid)
        if not d or d == source_doi:
            continue
        if d not in deep_v1_doi_set:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
        if len(out) >= k_papers:
            break
    return out


# ── Per-paper workers ────────────────────────────────────────────────────────


def process_intra(doi: str, *, extractor: str, max_claims: int) -> dict:
    con = thread_db()
    started = datetime.utcnow().isoformat() + "Z"
    claims = fetch_claims_for(con, doi, max_claims=max_claims)
    if len(claims) < 2:
        record_job(con, doi=doi, mode="intra", extractor=extractor,
                   status="skipped", edges_inserted=0, tokens_in=0, tokens_out=0,
                   error="<2 claims", started_at=started)
        return {"doi": doi, "status": "skipped", "edges": 0,
                "tokens_in": 0, "tokens_out": 0}
    title, _ = fetch_paper_meta(con, doi)
    compact = [_compact(c) for c in claims]
    prompt = INTRA_PROMPT.format(
        title=title, doi=doi,
        claims_json=json.dumps(compact, indent=1),
    )
    try:
        resp = call_gemini(prompt)
    except Exception as e:
        record_job(con, doi=doi, mode="intra", extractor=extractor,
                   status="failed", edges_inserted=0, tokens_in=0, tokens_out=0,
                   error=str(e)[:500], started_at=started)
        return {"doi": doi, "status": "failed", "edges": 0,
                "tokens_in": 0, "tokens_out": 0, "error": str(e)[:200]}

    valid_ids = {c["claim_id"] for c in claims}
    edges, _problems = validate_edges(
        resp["parsed"].get("edges", []),
        valid_from_ids=valid_ids, valid_to_ids=valid_ids,
        allowed_types=INTRA_PAPER_EDGE_TYPES,
        extractor=extractor, now=started,
    )
    inserted = insert_edges(con, edges)
    usage = resp.get("usage") or {}
    t_in = usage.get("prompt_tokens") or 0
    t_out = usage.get("completion_tokens") or 0
    record_job(con, doi=doi, mode="intra", extractor=extractor,
               status="done", edges_inserted=inserted,
               tokens_in=t_in, tokens_out=t_out,
               error=None, started_at=started)
    return {"doi": doi, "status": "done", "edges": inserted,
            "tokens_in": t_in, "tokens_out": t_out}


def process_cross(
    doi: str, *, extractor: str, max_claims: int,
    k_candidates: int, deep_v1_set: set[str],
) -> dict:
    con = thread_db()
    started = datetime.utcnow().isoformat() + "Z"
    src_claims = fetch_claims_for(con, doi, max_claims=max_claims)
    if not src_claims:
        record_job(con, doi=doi, mode="cross", extractor=extractor,
                   status="skipped", edges_inserted=0, tokens_in=0, tokens_out=0,
                   error="no claims", started_at=started)
        return {"doi": doi, "status": "skipped", "edges": 0,
                "tokens_in": 0, "tokens_out": 0}
    title, _ = fetch_paper_meta(con, doi)
    cand_dois = find_candidate_papers(
        con, source_doi=doi, source_title=title,
        source_claims=src_claims, k_papers=k_candidates,
        deep_v1_doi_set=deep_v1_set,
    )
    if not cand_dois:
        record_job(con, doi=doi, mode="cross", extractor=extractor,
                   status="skipped", edges_inserted=0, tokens_in=0, tokens_out=0,
                   error="no candidate papers", started_at=started)
        return {"doi": doi, "status": "skipped", "edges": 0,
                "tokens_in": 0, "tokens_out": 0}

    # Cap candidate-pool size: k_candidates × max_claims is the upper bound.
    other_pool = []
    for cdoi in cand_dois:
        for c in fetch_claims_for(con, cdoi, max_claims=max_claims):
            other_pool.append({**_compact(c), "_paper_doi": cdoi})

    src_compact = [_compact(c) for c in src_claims]
    prompt = CROSS_PROMPT.format(
        source_title=title, source_doi=doi,
        source_claims_json=json.dumps(src_compact, indent=1),
        other_claims_json=json.dumps(other_pool, indent=1),
    )
    try:
        resp = call_gemini(prompt)
    except Exception as e:
        record_job(con, doi=doi, mode="cross", extractor=extractor,
                   status="failed", edges_inserted=0, tokens_in=0, tokens_out=0,
                   error=str(e)[:500], started_at=started)
        return {"doi": doi, "status": "failed", "edges": 0,
                "tokens_in": 0, "tokens_out": 0, "error": str(e)[:200]}

    valid_from = {c["claim_id"] for c in src_claims}
    valid_to = {c["claim_id"] for c in other_pool}
    edges, _problems = validate_edges(
        resp["parsed"].get("edges", []),
        valid_from_ids=valid_from, valid_to_ids=valid_to,
        allowed_types=CROSS_PAPER_EDGE_TYPES,
        extractor=extractor, now=started,
    )
    inserted = insert_edges(con, edges)
    usage = resp.get("usage") or {}
    t_in = usage.get("prompt_tokens") or 0
    t_out = usage.get("completion_tokens") or 0
    record_job(con, doi=doi, mode="cross", extractor=extractor,
               status="done", edges_inserted=inserted,
               tokens_in=t_in, tokens_out=t_out,
               error=None, started_at=started)
    return {"doi": doi, "status": "done", "edges": inserted,
            "tokens_in": t_in, "tokens_out": t_out,
            "candidate_papers": len(cand_dois)}


# ── Runner with progress / cost meter ────────────────────────────────────────


def estimate_cost(t_in: int, t_out: int) -> float:
    return t_in * (PRICE_IN_PER_M / 1e6) + t_out * (PRICE_OUT_PER_M / 1e6)


def run_pool(
    fn, papers: list[str], *, mode: str, concurrency: int,
    log_every: int = 5,
):
    """Generic threaded runner with live progress + cost meter."""
    n = len(papers)
    if n == 0:
        print(f"[{mode}] no papers to process")
        return

    print(f"[{mode}] starting {n} papers, concurrency={concurrency}")
    t0 = time.monotonic()
    done = 0
    failed = 0
    skipped = 0
    edges_total = 0
    t_in_total = 0
    t_out_total = 0
    last_print = 0.0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(fn, doi): doi for doi in papers}
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
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            edges_total += r.get("edges", 0)
            t_in_total += r.get("tokens_in", 0)
            t_out_total += r.get("tokens_out", 0)

            now = time.monotonic()
            if (done + failed + skipped) % log_every == 0 or now - last_print > 30:
                last_print = now
                processed = done + failed + skipped
                elapsed = now - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta_s = (n - processed) / rate if rate > 0 else 0
                cost = estimate_cost(t_in_total, t_out_total)
                proj_cost = cost * (n / processed) if processed > 0 else 0
                print(
                    f"  [{processed:>5}/{n}] done={done} skip={skipped} fail={failed} "
                    f"edges={edges_total} cost=${cost:.2f} (proj=${proj_cost:.2f}) "
                    f"rate={rate*60:.1f}/min eta={eta_s/60:.1f}min"
                )

    elapsed = time.monotonic() - t0
    cost = estimate_cost(t_in_total, t_out_total)
    print(f"\n[{mode} summary]")
    print(f"  papers: done={done} skipped={skipped} failed={failed} of {n}")
    print(f"  edges inserted: {edges_total}")
    print(f"  tokens: in={t_in_total:,} out={t_out_total:,}")
    print(f"  cost:   ${cost:.2f} (input ${t_in_total*PRICE_IN_PER_M/1e6:.2f} + output ${t_out_total*PRICE_OUT_PER_M/1e6:.2f})")
    print(f"  wall:   {elapsed/60:.1f} min")
    if done > 0:
        print(f"  per-done-paper: ${cost/done:.4f}, edges={edges_total/done:.1f}, tokens={(t_in_total+t_out_total)/done:.0f}")


# ── Subcommands ──────────────────────────────────────────────────────────────


def _ensure_schema():
    from askchem.db import init_db
    init_db()


def _deep_v1_set(con: sqlite3.Connection) -> set[str]:
    # Re-use the eligibility cache; ≥2-claim filter is a near no-op subset.
    return set(_eligible_deep_v1_dois(con))


def cmd_intra(args):
    _ensure_schema()
    extractor = args.extractor_tag or INTRA_EXTRACTOR_DEFAULT
    con = open_db()
    papers = select_eligible_papers(
        con, mode="intra", extractor=extractor,
        resume=args.resume, limit=args.limit,
        sample_seed=args.sample_seed,
    )
    if args.dry_run:
        print(f"[intra dry-run] would process {len(papers)} papers with extractor={extractor}")
        return
    run_pool(
        lambda doi: process_intra(doi, extractor=extractor, max_claims=args.max_claims),
        papers, mode="intra", concurrency=args.concurrency,
    )


def cmd_cross(args):
    _ensure_schema()
    extractor = args.extractor_tag or CROSS_EXTRACTOR_DEFAULT
    con = open_db()
    deep_v1 = _deep_v1_set(con)
    papers = select_eligible_papers(
        con, mode="cross", extractor=extractor,
        resume=args.resume, limit=args.limit,
        sample_seed=args.sample_seed,
    )
    if args.dry_run:
        print(f"[cross dry-run] would process {len(papers)} papers with extractor={extractor}, "
              f"k_candidates={args.candidates}")
        return
    print(f"[cross] preloading FAISS embeddings (~3 GB) ...")
    _ensure_embeddings()
    run_pool(
        lambda doi: process_cross(
            doi, extractor=extractor,
            max_claims=args.max_claims,
            k_candidates=args.candidates,
            deep_v1_set=deep_v1,
        ),
        papers, mode="cross", concurrency=args.concurrency,
    )


def cmd_preflight(args):
    """Run intra+cross on N random deep_v1 papers and report cost/yield."""
    _ensure_schema()
    n = args.limit or 200
    intra_extractor = (args.extractor_tag or "intra_llm_gemini_preflight_v1")
    cross_extractor = (args.extractor_tag or "cross_llm_gemini_preflight_v1")

    con = open_db()
    deep_v1 = _deep_v1_set(con)

    # Pick the same N random papers for both passes (so densities match).
    papers = select_eligible_papers(
        con, mode="intra", extractor=intra_extractor,
        resume=args.resume, limit=n,
        sample_seed=args.sample_seed if args.sample_seed is not None else 7,
    )
    print(f"[preflight] {len(papers)} papers selected (seed={args.sample_seed or 7})")

    print(f"\n=== INTRA pre-flight (extractor={intra_extractor}) ===")
    run_pool(
        lambda doi: process_intra(doi, extractor=intra_extractor, max_claims=args.max_claims),
        papers, mode="intra-preflight", concurrency=args.concurrency,
    )

    print(f"\n[preflight] preloading FAISS embeddings for cross pass...")
    _ensure_embeddings()
    print(f"\n=== CROSS pre-flight (extractor={cross_extractor}, k={args.candidates}) ===")
    # Re-select for cross (resume flag means skip already-cross-done; first run = all)
    papers_cross = select_eligible_papers(
        con, mode="cross", extractor=cross_extractor,
        resume=args.resume, limit=n,
        sample_seed=args.sample_seed if args.sample_seed is not None else 7,
    )
    run_pool(
        lambda doi: process_cross(
            doi, extractor=cross_extractor,
            max_claims=args.max_claims, k_candidates=args.candidates,
            deep_v1_set=deep_v1,
        ),
        papers_cross, mode="cross-preflight", concurrency=args.concurrency,
    )

    # Aggregate report.
    print("\n=== Pre-flight extrapolation ===")
    for ex_tag, label in ((intra_extractor, "intra"), (cross_extractor, "cross")):
        r = con.execute("""
          SELECT COUNT(*) AS n,
                 SUM(edges_inserted) AS e,
                 SUM(tokens_in) AS ti,
                 SUM(tokens_out) AS to_,
                 SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done
            FROM edge_jobs
           WHERE mode = ? AND extractor = ?
        """, (label, ex_tag)).fetchone()
        if not r or not r["n"]:
            continue
        n_done = r["done"] or 0
        ti = r["ti"] or 0
        to = r["to_"] or 0
        edges = r["e"] or 0
        if n_done == 0:
            continue
        cost = estimate_cost(ti, to)
        per_paper_cost = cost / n_done
        per_paper_tokens = (ti + to) / n_done
        per_paper_edges = edges / n_done
        full_papers = 23213
        print(f"\n  [{label}]  done={n_done}  edges={edges}  tokens={ti+to:,}  cost=${cost:.2f}")
        print(f"    per paper: ${per_paper_cost:.4f}  tokens={per_paper_tokens:.0f}  edges={per_paper_edges:.1f}")
        print(f"    full corpus extrapolation ({full_papers:,} papers):")
        print(f"      tokens: {full_papers*per_paper_tokens/1e6:.1f} M")
        print(f"      cost:   ${full_papers*per_paper_cost:,.0f}")
        print(f"      edges:  {full_papers*per_paper_edges:,.0f}")


def cmd_status(_args):
    con = open_db()
    print(f"{'mode':<8} {'extractor':<40} {'done':>8} {'skip':>6} {'fail':>6} {'edges':>10} {'cost':>10}")
    print("-" * 100)
    for r in con.execute("""
        SELECT mode, extractor,
               SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skp,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS fail,
               SUM(edges_inserted) AS e,
               SUM(tokens_in) AS ti, SUM(tokens_out) AS to_
          FROM edge_jobs
         GROUP BY mode, extractor
         ORDER BY mode, extractor
    """).fetchall():
        cost = estimate_cost(r["ti"] or 0, r["to_"] or 0)
        print(f"{r['mode']:<8} {r['extractor']:<40} {r['done'] or 0:>8} "
              f"{r['skp'] or 0:>6} {r['fail'] or 0:>6} {r['e'] or 0:>10} ${cost:>9.2f}")


def cmd_purge(args):
    if not args.extractor_tag:
        print("--extractor-tag required for purge", file=sys.stderr)
        sys.exit(1)
    con = open_db()
    n_e = con.execute("DELETE FROM claim_edges WHERE extractor=?", (args.extractor_tag,)).rowcount
    n_j = con.execute("DELETE FROM edge_jobs WHERE extractor=?", (args.extractor_tag,)).rowcount
    con.commit()
    print(f"deleted {n_e} edges and {n_j} job rows for extractor={args.extractor_tag}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--concurrency", type=int, default=8)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--resume", action="store_true",
                        help="Skip papers already marked 'done' for this extractor")
        sp.add_argument("--extractor-tag", type=str, default=None)
        sp.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS_PER_PAPER)
        sp.add_argument("--sample-seed", type=int, default=None,
                        help="Shuffle eligible papers with this seed before --limit")
        sp.add_argument("--dry-run", action="store_true")

    sp_intra = sub.add_parser("intra", help="Intra-paper backfill")
    common(sp_intra); sp_intra.set_defaults(func=cmd_intra)

    sp_cross = sub.add_parser("cross", help="Cross-paper backfill (FAISS-driven candidates)")
    common(sp_cross)
    sp_cross.add_argument("--candidates", type=int, default=5,
                          help="Number of candidate papers per source")
    sp_cross.set_defaults(func=cmd_cross)

    sp_pre = sub.add_parser("preflight", help="Run intra+cross on N random papers and report")
    common(sp_pre)
    sp_pre.add_argument("--candidates", type=int, default=5)
    sp_pre.set_defaults(func=cmd_preflight)

    sp_st = sub.add_parser("status", help="Progress per (mode, extractor)")
    sp_st.set_defaults(func=cmd_status)

    sp_pg = sub.add_parser("purge", help="Delete edges + job rows for one extractor tag")
    sp_pg.add_argument("--extractor-tag", required=True)
    sp_pg.set_defaults(func=cmd_purge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
