"""
AskChem Web Server — Production FastAPI backend.

Serves the REST API (for agents and frontend), the static frontend,
and handles paper submissions from users.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import random
import sqlite3
import sys
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from . import db
from . import ltree
from . import advisor
from .taxonomy_aliases import resolve_tree_path

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

_EMAIL_RE = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')

MAX_QUERY_LENGTH = 500
ALLOWED_CALLBACK_SCHEMES = {"https"}
_PRIVATE_IP_RE = re.compile(
    r'^https?://(localhost|127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|169\.254\.|0\.0\.0\.0|\[::1\])',
    re.IGNORECASE,
)

# ── TTL Cache ────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, object]] = {}
CACHE_TTL = 60  # seconds


def _cached(key: str, ttl: int = CACHE_TTL):
    """Simple TTL cache decorator for expensive read-only queries."""
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < ttl:
            return val
    return None


def _set_cache(key: str, val: object):
    _cache[key] = (time.time(), val)


def _require_admin(request: Request):
    """Dependency that enforces admin authentication."""
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Admin endpoints disabled (ADMIN_TOKEN not configured)")
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {ADMIN_TOKEN}":
        ip_raw = request.client.host if request.client else ""
        ip_hash = hashlib.sha256(ip_raw.encode()).hexdigest()[:12] if ip_raw else ""
        db.log_security_event("admin_auth_failed", ip_hash=ip_hash, details="")
        raise HTTPException(401, "Invalid or missing admin token")


def _is_admin_request(request: Request) -> bool:
    return bool(ADMIN_TOKEN and request.headers.get("authorization", "") == f"Bearer {ADMIN_TOKEN}")


from . import retrieval as _retrieval

# ── App setup ────────────────────────────────────────────────────────────────


def _warmup_paw():
    # Both PAW programs are lazy-singleton via paw_functions._load_fn().
    # Warming only classify_intent left normalize_query cold; the first
    # query with zero FTS hits then paid a 30 s cold-load penalty inside
    # the request handler (db.py:3184-3197). Warm both at startup so the
    # request path never blocks on a fresh interpreter load.
    try:
        from askchem.paw_functions import classify_intent, normalize_query
        classify_intent("warmup")
        normalize_query("warmup")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app):
    try:
        db.init_db()
    except Exception as e:
        print(f"init_db failed at startup: {e}", flush=True)
    # Only warm the active retriever's stack (v1 OR v2). Loading both
    # wastes ~1 GB RAM and ~12 s of MPS init on dev laptops; the
    # dispatcher in askchem.retrieval routes everything correctly so
    # the unused side never needs to be initialised.
    active = _retrieval.active_version()
    try:
        _retrieval.load_embeddings()
        _retrieval.embed_query("warmup")
    except Exception as exc:
        print(f"{active} retriever warmup skipped: {exc}", flush=True)
    if active == "v2":
        try:
            _retrieval.warmup_cross_encoder()
        except Exception as exc:
            print(f"cross-encoder warmup skipped: {exc}", flush=True)
    try:
        with db.get_conn() as conn:
            conn.execute("SELECT COUNT(*) FROM claims").fetchone()
            conn.execute("SELECT COUNT(*) FROM claims_fts").fetchone()
            # Touch each FTS partition representative phrase pattern so
            # SQLite pages the FTS5 index leaves into the OS file cache;
            # otherwise the first real query pays for ~15 s of page-ins.
            for probe in (
                "chemistry", "coupling", "catalyst", "reaction",
                "spectroscopy", "synthesis",
            ):
                conn.execute(
                    "SELECT claim_id FROM claims_fts WHERE claims_fts MATCH ? LIMIT 10",
                    [probe],
                ).fetchall()
            conn.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()
            # Prewarm sources_fts (used by paper_recall) and claim_view_map
            # (used by tree recall) so first queries don't pay disk-read tax.
            try:
                conn.execute(
                    "SELECT doi FROM sources_fts WHERE sources_fts MATCH 'chemistry' LIMIT 10"
                ).fetchall()
            except Exception:
                pass
            try:
                conn.execute(
                    "SELECT claim_id FROM claim_view_map LIMIT 100"
                ).fetchall()
            except Exception:
                pass
        # Build the pre-stemmed tree-node cache off the request path.
        try:
            db._load_tree_node_index()
        except Exception:
            pass
        # Exercise the complete retrieval path before the process becomes
        # ready. Component warmups above do not populate query-variant,
        # paper-recall, candidate-hydration, reranker, or search-result caches.
        # Calling the DB layer directly avoids writing synthetic query_log rows.
        if os.environ.get("CHEMTREE_E2E_WARMUP", "") == "1":
            for probe in (
                "Suzuki coupling",
                "powder X-ray diffraction",
                "metal organic framework",
            ):
                try:
                    db.search_claims(probe, limit=50)
                except Exception as exc:
                    print(
                        f"end-to-end search warmup skipped for {probe!r}: {exc}",
                        flush=True,
                    )
    except Exception:
        pass
    import threading
    if os.environ.get("ASKCHEM_DISABLE_PAW_WARMUP", "0") != "1":
        threading.Thread(target=_warmup_paw, daemon=True).start()
    yield


app = FastAPI(
    title="AskChem API",
    description=(
        "A hierarchical, multi-view, source-grounded index of chemical knowledge. "
        "Claims extracted from chemistry papers across 14 subfields via deep full-paper analysis. "
        "7 views: reaction type, substance class, application, technique, mechanism, claim type, and time period. "
        "Designed for AI agents and human scientists. https://askchem.org"
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://askchem.org",
        "https://www.askchem.org",
        "http://localhost:8080",
        "http://localhost:8420",
    ],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ── Rate Limiting (OpenAI-style RPM + RPD) ───────────────────────────────────

_rate_buckets: dict[str, list[float]] = {}
_anon_rpd: dict[str, tuple[str, int]] = {}
WINDOW_SECONDS = 60

# Backwards-compat for tests that patch ANON_RATE_LIMIT
ANON_RATE_LIMIT = 100
REGISTERED_RATE_LIMIT = 1000

_RATE_LIMIT_EXEMPT = {"/api/health", "/api/docs", "/api/redoc", "/api/openapi.json", "/"}


def _utc_date_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def _rpm_rpd_for_key(key_info: dict | None) -> tuple[int, int]:
    """Return (requests_per_minute, requests_per_day) for tier."""
    if not key_info:
        return ANON_RATE_LIMIT, 5000
    t = (key_info.get("tier") or "tier_1").lower()
    if t in ("pro", "registered", "tier_3"):
        return 1000, 500_000
    if t == "tier_2":
        return 500, 100_000
    return 200, 20_000


def _anon_rpd_count(ip: str) -> int:
    day = _utc_date_str()
    if ip not in _anon_rpd or _anon_rpd[ip][0] != day:
        _anon_rpd[ip] = (day, 0)
    return _anon_rpd[ip][1]


def _anon_rpd_increment(ip: str) -> None:
    day = _utc_date_str()
    d, c = _anon_rpd.get(ip, (day, 0))
    if d != day:
        c = 0
    _anon_rpd[ip] = (day, c + 1)


def _rate_limit_http_headers(
    *,
    rpm_limit: int,
    rpm_remaining: int,
    rpm_reset_sec: int,
    rpd_limit: int,
    rpd_remaining: int,
    rpd_reset_sec: int,
) -> dict[str, str]:
    return {
        "x-ratelimit-limit-requests": str(rpm_limit),
        "x-ratelimit-remaining-requests": str(max(0, rpm_remaining)),
        "x-ratelimit-reset-requests": f"{max(1, rpm_reset_sec)}s",
        "x-ratelimit-limit-requests-day": str(rpd_limit),
        "x-ratelimit-remaining-requests-day": str(max(0, rpd_remaining)),
        "x-ratelimit-reset-requests-day": f"{rpd_reset_sec}s",
    }


def _check_rate_limit(request: Request) -> dict:
    """
    Enforce RPM (sliding window) and RPD (UTC calendar day).
    Returns dict with key_info and header values for successful requests.
    """
    auth = request.headers.get("authorization", "")
    key_info = None
    if auth.startswith("Bearer ac-") or auth.startswith("Bearer ct-"):  # ct- is legacy
        raw_key = auth.split(" ", 1)[1]
        key_info = db.validate_api_key(raw_key)

    rpm_limit, rpd_limit = _rpm_rpd_for_key(key_info)
    ip = request.client.host if request.client else "unknown"
    bucket_key = key_info["key_id"] if key_info else ip

    now = time.time()
    if random.random() < 0.01:
        cutoff = now - WINDOW_SECONDS * 2
        stale = [k for k, v in _rate_buckets.items() if k != bucket_key and (not v or max(v) < cutoff)]
        for k in stale:
            del _rate_buckets[k]

    if bucket_key not in _rate_buckets:
        _rate_buckets[bucket_key] = []
    _rate_buckets[bucket_key] = [t for t in _rate_buckets[bucket_key] if now - t < WINDOW_SECONDS]

    used_rpm = len(_rate_buckets[bucket_key])
    if used_rpm >= rpm_limit:
        reset = int(min(_rate_buckets[bucket_key]) + WINDOW_SECONDS - now) + 1
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:12]
        db.log_security_event("rate_limit_rpm", ip_hash=ip_hash, details="429 rpm")
        h = _rate_limit_http_headers(
            rpm_limit=rpm_limit,
            rpm_remaining=0,
            rpm_reset_sec=reset,
            rpd_limit=rpd_limit,
            rpd_remaining=max(0, rpd_limit - (db.get_key_rpd_today(bucket_key) if key_info else _anon_rpd_count(ip))),
            rpd_reset_sec=_seconds_until_utc_midnight(),
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({rpm_limit}/min). Use an API key for higher limits.",
            headers={**h, "Retry-After": str(reset)},
        )

    if key_info:
        rpd_used = db.get_key_rpd_today(key_info["key_id"])
    else:
        rpd_used = _anon_rpd_count(ip)

    if rpd_used >= rpd_limit:
        reset = _seconds_until_utc_midnight()
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:12]
        db.log_security_event("rate_limit_rpd", ip_hash=ip_hash, details="429 rpd")
        h = _rate_limit_http_headers(
            rpm_limit=rpm_limit,
            rpm_remaining=max(0, rpm_limit - used_rpm),
            rpm_reset_sec=int(WINDOW_SECONDS),
            rpd_limit=rpd_limit,
            rpd_remaining=0,
            rpd_reset_sec=reset,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Daily request limit exceeded ({rpd_limit}/day). Resets at UTC midnight.",
            headers={**h, "Retry-After": str(reset)},
        )

    _rate_buckets[bucket_key].append(now)
    if key_info:
        db.record_authenticated_api_request(key_info["key_id"])
        rpd_used_after = rpd_used + 1
    else:
        _anon_rpd_increment(ip)
        rpd_used_after = rpd_used + 1

    used_rpm_after = used_rpm + 1
    rpm_rem = max(0, rpm_limit - used_rpm_after)
    rpd_rem = max(0, rpd_limit - rpd_used_after)
    next_reset = int(min(_rate_buckets[bucket_key]) + WINDOW_SECONDS - now) + 1 if _rate_buckets[bucket_key] else WINDOW_SECONDS

    return {
        "key_info": key_info,
        "headers": _rate_limit_http_headers(
            rpm_limit=rpm_limit,
            rpm_remaining=rpm_rem,
            rpm_reset_sec=next_reset,
            rpd_limit=rpd_limit,
            rpd_remaining=rpd_rem,
            rpd_reset_sec=_seconds_until_utc_midnight(),
        ),
    }


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        raw = request.headers.get("x-client-request-id") or request.headers.get("X-Client-Request-Id")
        if raw and len(raw) <= 512:
            rid = raw.strip()[:512]
        else:
            rid = str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _RATE_LIMIT_EXEMPT or not (path.startswith("/api/") or path.startswith("/v1/")):
            return await call_next(request)
        try:
            rl = _check_rate_limit(request)
        except HTTPException as exc:
            return StarletteResponse(
                content=json.dumps({"detail": exc.detail}),
                status_code=exc.status_code,
                headers={**(exc.headers or {}), "Content-Type": "application/json"},
            )
        request.state.api_key_info = rl["key_info"]
        response = await call_next(request)
        for k, v in rl["headers"].items():
            response.headers[k] = v
        return response


app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIdMiddleware)


# ── API Routes ───────────────────────────────────────────────────────────────

@app.get("/api/health", summary="Health check")
def api_health():
    """Health check endpoint for monitoring and load balancers."""
    try:
        stats = db.get_stats()
        return {
            "status": "ok",
            "version": "1.0.0",
            "total_claims": int(stats.get("total_claims", 0)),
            "total_sources": int(stats.get("total_sources", 0)),
        }
    except Exception as e:
        raise HTTPException(503, detail=f"Database unavailable: {e}")


@app.get("/api/usage", summary="API key usage (authenticated)")
def api_usage(request: Request):
    """Usage and limits for the Bearer API key on this request (OpenAI-style dashboard data)."""
    info = getattr(request.state, "api_key_info", None)
    if not info:
        raise HTTPException(401, "Bearer API key required (Authorization: Bearer ac-...)")
    rpm, rpd = _rpm_rpd_for_key(info)
    summary = db.get_api_key_usage_summary(info["key_id"], days=30)
    return {
        "key_id": info["key_id"],
        "tier": info["tier"],
        "limits": {"rpm": rpm, "rpd": rpd},
        "daily_usage": summary["daily_usage"],
        "total_requests": summary["total_requests"],
    }


@app.get("/api/stats", summary="Index statistics")
def api_stats():
    """Get overview statistics about the AskChem index."""
    cached = _cached("stats")
    if cached is not None:
        return cached
    stats = db.get_stats()
    result = {
        "total_claims": int(stats.get('total_claims', 0)),
        "total_sources": int(stats.get('total_sources', 0)),
        "total_views": int(stats.get('total_views', 5)),
        "total_nodes": int(stats.get('total_nodes', 0)),
        "claim_types": stats.get('claim_types', {}),
        "year_distribution": stats.get('year_distribution', {}),
        "citation_source": stats.get('citation_source', ''),
        "citation_source_url": stats.get('citation_source_url', ''),
        "citations_updated_at": stats.get('citations_updated_at', ''),
    }
    _set_cache("stats", result)
    return result


_intent_pool = None
_intent_cache: dict[str, str] = {}


def _classify_intent_bg(query: str):
    """Run intent classification in background, caching the result for later."""
    cache_key = query.lower().strip()
    if cache_key in _intent_cache:
        return
    try:
        from askchem.paw_functions import classify_intent
        result = classify_intent(query)
        _intent_cache[cache_key] = result if result in (
            "author", "substance", "method", "concept", "paper", "reaction"
        ) else "concept"
    except Exception:
        _intent_cache[cache_key] = "concept"


def _get_intent(query: str) -> str:
    """Classify query intent, waiting briefly for the result."""
    global _intent_pool
    # Rule-based override: PAW often labels ``Suzuki coupling'' as ``method''
    # because of ``coupling'', but users mean a named reaction, not an
    # instrument technique. mxbai disambiguates these natively (homonym
    # nDCG@10 = 0.942) so this is a candidate to retire in δ2; kill switch
    # CHEMTREE_DISABLE_COUPLING_INTENT_OVERRIDE=1 bypasses it.
    if (db.query_signals_organic_cross_coupling(query)
            and os.environ.get(
                "CHEMTREE_DISABLE_COUPLING_INTENT_OVERRIDE", "0"
            ) != "1"):
        return "reaction"
    cache_key = query.lower().strip()
    if cache_key in _intent_cache:
        return _intent_cache[cache_key]
    if _intent_pool is None:
        import concurrent.futures
        _intent_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = _intent_pool.submit(_classify_intent_bg, query)
    try:
        future.result(timeout=2.0)
        return _intent_cache.get(cache_key, "concept")
    except Exception:
        return "concept"


_LTREE_VIEWS = ("by_reaction_type", "by_substance_class", "by_mechanism", "by_technique")


def _attach_ltree_paths(result: dict, view: Optional[str], query: str = "") -> None:
    """Attach a living-tree breadcrumb (``ltree_path``) to each search result.

    Placement is made QUERY-RELEVANT for consistency: among a paper's living-tree
    placements we pick the node most semantically similar to the search query
    (so e.g. all "suzuki coupling" results surface their cross-coupling branch
    rather than an arbitrary dominant one). Papers with no placement at all fall
    back to the query's nearest branch, flagged ``ltree_approx`` so the UI can
    label it. Reuses the resident query encoder + the node-vector index; additive
    metadata only, never affects ranking, silently no-ops if data is missing.
    """
    results = result.get("results") or []
    if not results:
        return
    pref = view if view in _LTREE_VIEWS else None
    dois = list({r.get("source_doi") for r in results if r.get("source_doi")})
    if not dois:
        return
    try:
        import numpy as np
        # doi -> {view_id: [node_id, ...]}
        by_doi: dict = {}
        with db.get_conn() as c:
            ph = ",".join("?" for _ in dois)
            for row in c.execute(
                f"SELECT DISTINCT view_id, node_id, doi FROM taxonomy_leaves "
                f"WHERE doi IN ({ph})", dois).fetchall():
                by_doi.setdefault(row["doi"], {}).setdefault(
                    row["view_id"], []).append(row["node_id"])

        # Query vector + node-vector index (for query-relevant node choice).
        qv = None
        node_vec: dict = {}          # (view_id, node_id) -> vec
        try:
            from askchem import retrieval as _r
            if _r.is_loaded() and query:
                qv = np.asarray(_r.embed_query(query), dtype="float32")
                idx = ltree._load_node_index().get("by_view", {})
                for vid, bv in idx.items():
                    for i, meta in enumerate(bv["meta"]):
                        node_vec[(vid, meta["node_id"])] = bv["vecs"][i]
        except Exception:
            qv = None

        def _sim(vid, nid):
            v = node_vec.get((vid, nid))
            if qv is None or v is None:
                return 0.0
            v = np.asarray(v, dtype="float32")
            d = min(qv.shape[0], v.shape[0])
            a, b = qv[:d], v[:d]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb)) if na and nb else 0.0

        # Precompute the query's globally-nearest node (approx fallback for papers
        # with no placement), preferring the requested view.
        approx = None
        if qv is not None and node_vec:
            order = ([pref] if pref else []) + [v for v in _LTREE_VIEWS if v != pref]
            best = None
            for (vid, nid) in node_vec:
                s = _sim(vid, nid) + (0.03 if vid == pref else 0.0)
                if best is None or s > best[0]:
                    best = (s, vid, nid)
            if best:
                approx = (best[1], best[2])

        order = ([pref] if pref else []) + [v for v in _LTREE_VIEWS if v != pref]
        path_cache: dict = {}

        def _path(v, nid):
            key = (v, nid)
            if key not in path_cache:
                try:
                    path_cache[key] = ltree.get_path(v, nid)
                except Exception:
                    path_cache[key] = None
            return path_cache[key]

        for r in results:
            m = by_doi.get(r.get("source_doi"))
            chosen, is_approx = None, False
            if m:
                # candidates = all this paper's placements; pick the one most
                # relevant to the query (with a small preference for the view).
                cands = [(v, nid) for v in m for nid in m[v]]
                if qv is not None:
                    v, nid = max(cands, key=lambda vn: _sim(*vn) + (0.05 if vn[0] == pref else 0.0))
                else:
                    # no query vector: prefer requested view, else first
                    v = next((vv for vv in order if vv in m), cands[0][0])
                    nid = m[v][0]
                chosen = (v, nid)
            elif approx:
                chosen, is_approx = approx, True
            if not chosen:
                continue
            p = _path(*chosen)
            if p:
                r["ltree_path"] = p
                r["ltree_view"] = chosen[0]
                if is_approx:
                    r["ltree_approx"] = True
    except Exception:
        return


@app.get("/api/search", summary="Search claims by text query")
def api_search(
    request: Request,
    q: str = Query(..., description="Search query", max_length=MAX_QUERY_LENGTH),
    claim_type: Optional[str] = Query(None, description="Filter by claim type"),
    view: Optional[str] = Query(None, description="Filter by view"),
    # `le=500` matches the capacity the deleted `/api/search/grouped`
    # endpoint exposed for months and that the client-side grouped flow
    # in web/index.html still needs to build its taxonomy tree without
    # truncation. `search_claims` already supports limit=500.
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    mode: str = Query(
        "auto",
        pattern="^(auto|phrase|all|any)$",
        description=(
            "FTS query mode. ``auto`` (default) runs the full cascade "
            "(phrase → NEAR → AND → 3-term combos → plural → synonyms). "
            "``phrase`` forces exact-phrase match. ``all`` ANDs all "
            "terms. ``any`` ORs the terms (broadest)."
        ),
    ),
    sort: str = Query(
        "relevance",
        pattern="^(relevance|date)$",
        description=(
            "Result ordering. ``relevance`` (default) uses the RRF + "
            "rerank score. ``date`` re-sorts by ``source_year`` "
            "descending, with relevance breaking ties."
        ),
    ),
):
    """
    Full-text search across all claims. Searches verbatim quotes, paper titles,
    molecule names, reaction types, techniques, and more.

    **Agent usage:** `GET /api/search?q=suzuki+coupling&claim_type=reaction&limit=20`
    """
    t0 = time.monotonic()
    result = db.search_claims(
        q, claim_type=claim_type, view=view, limit=limit, offset=offset,
        mode=mode, sort=sort,
    )
    result["intent"] = _get_intent(q)
    _attach_ltree_paths(result, view, q)
    latency = (time.monotonic() - t0) * 1000
    ip_raw = request.client.host if request.client else ""
    current_user = _get_current_user(request)
    qlid = db.log_query(
        query=q, endpoint="/api/search", view=view,
        filters=json.dumps({"claim_type": claim_type, "mode": mode, "sort": sort})
        if (claim_type or mode != "auto" or sort != "relevance")
        else None,
        result_count=result.get("total", 0), latency_ms=latency,
        user_agent=request.headers.get("user-agent", "")[:200],
        ip_hash=hashlib.sha256(ip_raw.encode()).hexdigest()[:12],
        user_id=current_user["user_id"] if current_user else None,
    )
    if qlid is not None:
        result["query_log_id"] = qlid
    return result


@app.get("/api/search/time", summary="Temporal histogram of ALL papers matching a query")
def api_search_time(
    request: Request,
    q: str = Query(..., description="Search query", max_length=MAX_QUERY_LENGTH),
):
    """Decade/year counts over **every** paper matching the query (uncapped),
    for the search "Time" view and temporal figures. Complements ``/api/search``
    (which returns a ranked, capped result list)."""
    return db.search_time_distribution(q)


class MultiSearchRequest(BaseModel):
    """Body for ``POST /api/searches`` — parallel multi-query merge."""

    queries: list[str]
    limit: int = 50
    offset: int = 0
    claim_type: Optional[str] = None
    view: Optional[str] = None
    mode: str = "auto"
    sort: str = "relevance"
    per_query_limit: int = 50
    max_per_source: int = 2


def _diversify_by_source(claims: list[dict], max_per_source: int) -> list[dict]:
    """Cap consecutive claims per source DOI, then pad with the remainder."""
    if max_per_source <= 0:
        return claims
    selected: list[dict] = []
    counts: dict[str, int] = {}
    skipped: list[dict] = []
    for c in claims:
        doi = (c.get("source_doi") or "").lower()
        if doi and counts.get(doi, 0) >= max_per_source:
            skipped.append(c)
            continue
        if doi:
            counts[doi] = counts.get(doi, 0) + 1
        selected.append(c)
    return selected + skipped


@app.post("/api/searches", summary="Run multiple queries in parallel and merge")
def api_searches(payload: MultiSearchRequest, request: Request):
    """
    Multi-query search: runs each entry in ``queries`` against
    ``/api/search`` semantics in parallel, then merges the per-query
    ranked lists via Reciprocal Rank Fusion (k=60), diversifies by paper,
    and returns the top window.

    Useful for agent flows that already rewrite a question into 3-4
    sub-queries (e.g. AskChem-Bench ``unified``) — promoting that pattern
    to a first-class endpoint avoids the N round-trips and fuses the
    ranks server-side.

    Each query inherits the request-level ``claim_type`` / ``view`` /
    ``mode`` / ``sort`` filters. ``per_query_limit`` caps each
    sub-search's window before RRF; ``max_per_source`` caps how many
    claims from a single DOI appear in the merged result.
    """
    queries = [q.strip() for q in (payload.queries or []) if q and q.strip()]
    if not queries:
        raise HTTPException(status_code=400, detail="queries must contain at least one non-empty string")
    if len(queries) > 8:
        raise HTTPException(status_code=400, detail="queries: at most 8 per request")
    mode = (payload.mode or "auto").lower()
    if mode not in ("auto", "phrase", "all", "any"):
        raise HTTPException(status_code=400, detail="mode must be one of auto/phrase/all/any")
    sort = (payload.sort or "relevance").lower()
    if sort not in ("relevance", "date"):
        raise HTTPException(status_code=400, detail="sort must be relevance or date")

    per_query_limit = max(1, min(int(payload.per_query_limit or 50), 200))
    limit = max(1, min(int(payload.limit or 50), 500))
    offset = max(0, int(payload.offset or 0))
    max_per_source = max(0, min(int(payload.max_per_source or 2), 10))

    t0 = time.monotonic()
    from concurrent.futures import ThreadPoolExecutor

    def _one(query: str) -> dict:
        sub_t0 = time.monotonic()
        try:
            r = db.search_claims(
                query,
                claim_type=payload.claim_type,
                view=payload.view,
                limit=per_query_limit,
                offset=0,
                mode=mode,
                sort=sort,
            )
        except Exception as exc:
            return {
                "query": query,
                "error": str(exc)[:200],
                "results": [],
                "total": 0,
                "elapsed_ms": int((time.monotonic() - sub_t0) * 1000),
            }
        return {
            "query": query,
            "results": r.get("results") or [],
            "total": r.get("total") or 0,
            "elapsed_ms": int((time.monotonic() - sub_t0) * 1000),
        }

    with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as pool:
        sub_results = list(pool.map(_one, queries))

    ranked_lists: list[list[str]] = []
    claim_by_id: dict[str, dict] = {}
    for sr in sub_results:
        ranks: list[str] = []
        for claim in sr.get("results") or []:
            cid = claim.get("claim_id")
            if not cid:
                continue
            if cid not in claim_by_id:
                claim_by_id[cid] = claim
            ranks.append(cid)
        if ranks:
            ranked_lists.append(ranks)

    merged_cids = db._rrf_merge(ranked_lists, k=60) if ranked_lists else []
    ordered = [claim_by_id[cid] for cid, _ in merged_cids if cid in claim_by_id]

    if sort == "date":
        ordered.sort(
            key=lambda r: (
                -(int(r.get("source_year")) if r.get("source_year") else -1),
                -(r.get("_relevance_score") or 0.0),
            )
        )

    if max_per_source > 0:
        ordered = _diversify_by_source(ordered, max_per_source=max_per_source)

    total = len(ordered)
    paged = ordered[offset:offset + limit]
    latency = (time.monotonic() - t0) * 1000

    ip_raw = request.client.host if request.client else ""
    current_user = _get_current_user(request)
    try:
        db.log_query(
            query="; ".join(queries)[:240],
            endpoint="/api/searches",
            view=payload.view,
            filters=json.dumps({
                "claim_type": payload.claim_type,
                "mode": mode,
                "sort": sort,
                "n_queries": len(queries),
            }) if (payload.claim_type or mode != "auto" or sort != "relevance" or len(queries) > 1) else None,
            result_count=total,
            latency_ms=latency,
            user_agent=request.headers.get("user-agent", "")[:200],
            ip_hash=hashlib.sha256(ip_raw.encode()).hexdigest()[:12],
            user_id=current_user["user_id"] if current_user else None,
        )
    except Exception:
        pass

    return {
        "results": paged,
        "total": total,
        "limit": limit,
        "offset": offset,
        "mode": mode,
        "sort": sort,
        "queries": [
            {"query": s["query"],
             "total": s.get("total", 0),
             "returned": len(s.get("results") or []),
             "elapsed_ms": s.get("elapsed_ms", 0),
             **({"error": s["error"]} if s.get("error") else {})}
            for s in sub_results
        ],
        "elapsed_ms": int(latency),
    }


@app.get("/api/search/structure", summary="Search by molecular structure")
def api_search_structure(
    smiles: str = Query(..., description="Query SMILES string", max_length=500),
    type: str = Query("substructure", description="substructure or similarity"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Search claims by molecular structure (SMILES).

    - **substructure**: Find molecules containing the query as a substructure
    - **similarity**: Find molecules with Tanimoto similarity >= 0.3

    Requires RDKit. Returns claims with matching subject_smiles.
    """
    if type not in ("substructure", "similarity"):
        raise HTTPException(400, "type must be 'substructure' or 'similarity'")
    result = db.search_by_structure(smiles, search_type=type, limit=limit, offset=offset)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/claims/featured", summary="Featured example claims")
def api_featured_claims(
    limit: int = Query(5, ge=1, le=20),
    refresh: bool = Query(False, description="Bypass cache and fetch fresh claims"),
):
    """Return well-classified, multi-view claims suitable for showcasing."""
    if not refresh:
        cached = _cached("featured_claims", ttl=600)
        if cached is not None:
            return cached

    featured = []
    try:
        with db.get_conn() as conn:
            max_rowid = conn.execute("SELECT MAX(rowid) FROM claims").fetchone()[0] or 1
            attempts = 0
            while len(featured) < limit and attempts < 50:
                attempts += 1
                rid = random.randint(1, max_rowid)
                row = conn.execute(
                    "SELECT data FROM claims WHERE rowid >= ? "
                    "AND verbatim_quote IS NOT NULL AND verbatim_quote != '' "
                    "LIMIT 1", [rid]
                ).fetchone()
                if not row:
                    continue
                claim = json.loads(row["data"])
                view_paths = claim.get("view_paths", {})
                vq = claim.get("verbatim_quote", "")
                subj = claim.get("subject", "")
                if not (isinstance(view_paths, dict) and len(view_paths) >= 3 and vq):
                    continue
                if not subj or len(subj) < 3 or subj.lower() in ("paper", "title", "authors", "none"):
                    continue
                if len(vq) < 40:
                    continue
                featured.append(claim)
    except Exception:
        pass

    result = {"claims": featured, "total": len(featured)}
    if featured:
        _set_cache("featured_claims", result)
    return result


@app.get("/api/claims/{claim_id}", summary="Get a specific claim")
def api_get_claim(claim_id: str):
    """
    Retrieve full details of a claim including provenance, view assignments,
    and all type-specific fields.

    **Agent usage:** Use the claim_id from search or tree browsing results.
    """
    claim = db.get_claim(claim_id)
    if not claim:
        raise HTTPException(404, f"Claim '{claim_id}' not found")
    return claim


class BulkClaimsRequest(BaseModel):
    claim_ids: list[str]


@app.post("/api/claims/bulk", summary="Get multiple claims by ID")
def api_bulk_claims(req: BulkClaimsRequest):
    """
    Fetch up to 200 claims in a single request.

    **Agent usage:** Collect claim IDs from tree browsing or search, then
    batch-fetch full details: `POST /api/claims/bulk {"claim_ids": ["id1", "id2"]}`
    """
    if len(req.claim_ids) > 200:
        raise HTTPException(400, "Maximum 200 claim IDs per request")
    claims = db.get_claims_bulk(req.claim_ids)
    return {"claims": claims, "count": len(claims), "requested": len(req.claim_ids)}


@app.get("/api/views", summary="List all hierarchical views")
def api_list_views():
    """
    List the 5 hierarchical views available for browsing.

    **Agent usage:** Call this first, then browse with `/api/tree/{view_id}`.
    """
    cached = _cached("views")
    if cached is not None:
        return cached
    views = db.list_views()
    result = {"views": views, "count": len(views)}
    _set_cache("views", result)
    return result


@app.get("/api/tree/{view_id}", summary="Browse tree root")
def api_tree_root(view_id: str, depth: int = Query(1, ge=0, le=4)):
    """
    Get the root of a view's hierarchy with children.

    **Agent usage:** Start here. Use depth=1 for top-level categories.
    """
    tree = db.get_tree_with_depth(view_id, '', depth)
    if not tree:
        raise HTTPException(404, f"View '{view_id}' not found")
    return {"view_id": view_id, "tree": tree}


@app.get("/api/tree/{view_id}/{path:path}", summary="Browse a tree node")
def api_tree_node(
    view_id: str, path: str,
    depth: int = Query(1, ge=0, le=4),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Browse a specific node in the hierarchy. Returns children and paginated claims.

    **Agent usage:** Navigate deeper by following paths from parent nodes.
    Path is slash-separated, e.g. `/api/tree/by_reaction_type/catalysis/heterogeneous`.
    Paginate claims with `limit` and `offset`.
    """
    requested_path = path
    path, aliased = resolve_tree_path(view_id, path)
    tree = db.get_tree_with_depth(view_id, path, depth)
    if not tree:
        raise HTTPException(404, f"Node not found: {view_id}/{path}")

    result = db.get_claims_at_node(view_id, path, limit=limit, offset=offset)
    resp = {
        "view_id": view_id,
        "path": path.split('/') if path else [],
        "node": tree,
        "claims": result["claims"],
        "total_claims": result["total"],
        "limit": limit,
        "offset": offset,
    }
    if aliased:
        resp["requested_path"] = requested_path
        resp["canonical_path"] = path
    if "children_summary" in result:
        resp["children_summary"] = result["children_summary"]
    return resp


# ── Living taxonomy (scaffold + paper-grounded leaves) ────────────────────────

@app.get("/api/ltree/views", summary="List living-taxonomy views")
def api_ltree_views():
    return ltree.list_views()


@app.get("/api/ltree/search", summary="Search-to-node in the living tree")
def api_ltree_search(
    view: str = Query(...), q: str = Query(...),
    limit: int = Query(30, ge=1, le=100),
    k: int = Query(8, ge=1, le=25),
):
    return ltree.search(view, q, limit=limit, k=k)


@app.get("/api/ltree/{view_id}/root", summary="Living tree root")
def api_ltree_root(view_id: str, depth: int = Query(1, ge=1, le=4)):
    node = ltree.get_node(view_id, ltree.ROOT_ID, depth=depth)
    if not node:
        raise HTTPException(404, f"Living view '{view_id}' not found")
    return node


@app.get("/api/ltree/{view_id}/node/{node_id}", summary="Living tree node + children")
def api_ltree_node(view_id: str, node_id: str, depth: int = Query(1, ge=1, le=4)):
    node = ltree.get_node(view_id, node_id, depth=depth)
    if not node:
        raise HTTPException(404, f"Node not found: {view_id}/{node_id}")
    node["path"] = ltree.get_path(view_id, node_id)
    return node


@app.get("/api/ltree/{view_id}/node/{node_id}/papers", summary="Paper leaves at a living node")
def api_ltree_papers(
    view_id: str, node_id: str,
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
):
    return ltree.get_papers(view_id, node_id, limit=limit, offset=offset)


@app.get("/api/ltree/{view_id}/node/{node_id}/paper-claims",
         summary="A paper's claims at this node (same tree path)")
def api_ltree_paper_claims(
    view_id: str, node_id: str, doi: str = Query(...),
    limit: int = Query(100, ge=1, le=300),
):
    return ltree.get_paper_claims(view_id, node_id, doi, limit=limit)


@app.get("/api/ltree/{view_id}/node/{node_id}/advise",
         summary="Advisor: grounded positioning questions (precomputed; live fallback)")
def api_ltree_advise(view_id: str, node_id: str, doi: str = Query(...)):
    return advisor.advise(view_id, node_id, doi)


@app.get("/api/ltree/{view_id}/node/{node_id}/influence",
         summary="Within-branch citation/influence ranking + seed paper detection")
def api_ltree_influence(view_id: str, node_id: str, limit: int = Query(200)):
    return ltree.influence(view_id, node_id, limit=limit)


@app.get("/api/ltree/{view_id}/node/{node_id}/critique",
         summary="Critical evaluation: are claims supported; is the reasoning sound")
def api_ltree_critique(view_id: str, node_id: str, doi: str = Query(...)):
    return advisor.critique(view_id, node_id, doi)


@app.get("/api/ltree/{view_id}/node/{node_id}/contribution",
         summary="Contribution: how it extends/challenges its host principle + neighbors")
def api_ltree_contribution(view_id: str, node_id: str, doi: str = Query(...)):
    return advisor.contribution(view_id, node_id, doi)


@app.get(
    "/api/views/{view_id}/children-search",
    summary="Search direct children of a tree node by name",
)
def api_search_tree_children(
    view_id: str,
    q: str = Query(..., min_length=1, max_length=120,
                   description="Substring matched against child name/path"),
    parent_path: str = Query(
        "", description="Parent path to scope the search; empty = root"
    ),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Find direct children of a parent node whose path or display name match `q`.

    Designed for wide views like `by_data` where an L1 category (e.g.
    `physical`) holds tens of thousands of measurement leaves and a fixed
    top-N listing isn't enough — researchers often want to find a specific
    measurement by name (e.g. "thermal conductivity").

    **Agent usage:** `GET /api/views/by_data/children-search?parent_path=physical&q=conductivity`
    """
    rows = db.search_tree_children(view_id, parent_path, q, limit=limit)
    return {
        "view_id": view_id,
        "parent_path": parent_path,
        "query": q,
        "results": rows,
        "count": len(rows),
    }


@app.get("/api/papers", summary="Browse or search papers")
def api_papers(
    q: Optional[str] = Query(None, description="Search papers by title"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str = Query("citations", description="Sort by: citations, year, claims"),
):
    """Browse papers sorted by citations, or search by title."""
    return db.search_papers(q=q, limit=limit, offset=offset, sort=sort)


@app.get("/api/sources/{doi:path}", summary="Get claims from a paper")
def api_source_claims(doi: str):
    """
    Get all claims extracted from a specific paper, identified by DOI.

    **Agent usage:** Useful for verification — compare extracted claims against the original.
    """
    source = db.get_source(doi)
    claims = db.get_claims_by_doi(doi)
    authors = db.get_authors_for_doi(doi)
    return {
        "doi": doi,
        "source": source,
        "claims": claims,
        "authors": authors,
        "count": len(claims),
    }


@app.get(
    "/api/papers/{doi:path}/snippet",
    summary="Top question-relevant claim verbatims for a paper",
)
def api_paper_snippet(
    doi: str,
    q: Optional[str] = Query(
        None,
        description=(
            "Optional question. When supplied, claims are ranked by their "
            "FTS match against the query; ties broken by claim_type "
            "informativeness. When omitted, returns the paper's top claims "
            "in extraction order."
        ),
        max_length=MAX_QUERY_LENGTH,
    ),
    limit: int = Query(3, ge=1, le=20),
):
    """
    Return a small set of verbatim claim quotes from one paper, ranked
    against an optional question. Mirrors the ergonomics of Paperclip's
    ``papers head /papers/<id>/content.lines`` but operates over the
    structured AskChem claim store.

    Useful for agent flows that have already retrieved a DOI (e.g. via
    ``/api/search`` or a citation) and want question-grounded snippets
    without an extra full-paper round trip.
    """
    source = db.get_source(doi)
    if not source:
        raise HTTPException(status_code=404, detail=f"paper not found: {doi}")

    claims = db.get_claims_by_doi(doi) or []

    if q and q.strip() and claims:
        # Rank claims by how often the query terms hit the verbatim quote
        # plus a small claim_type informativeness bonus. We don't go via
        # FTS here because the FTS index is per-claim-id and we already
        # have the candidate set in memory; a token-overlap scorer is
        # cheaper for a single paper (typically <30 claims) and still
        # produces the same top-3 in spot checks against /api/search.
        import re as _re
        words = [w.lower() for w in _re.findall(r"[A-Za-z0-9]{2,}", q)]
        words = [w for w in words if len(w) >= 2]
        _TYPE_BONUS = {
            "reaction": 0.5, "method": 0.5, "property": 0.4,
            "mechanism": 0.4, "computational_result": 0.4,
            "comparison": 0.3, "scope_entry": 0.3,
            "observation": 0.2, "hypothesis": 0.1,
            "background": 0.0, "historical": 0.0,
        }

        def _score(c: dict) -> float:
            text = " ".join([
                str(c.get("verbatim_quote") or ""),
                str(c.get("claim_contextualized") or ""),
                str(c.get("reaction_type") or ""),
            ]).lower()
            if not text or not words:
                return 0.0
            hits = sum(1 for w in words if w in text)
            return hits + _TYPE_BONUS.get(str(c.get("claim_type") or ""), 0.0)

        claims = sorted(claims, key=_score, reverse=True)

    def _shape(c: dict) -> dict:
        return {
            "claim_id": c.get("claim_id"),
            "claim_type": c.get("claim_type"),
            "verbatim_quote": c.get("verbatim_quote"),
            "claim_contextualized": c.get("claim_contextualized"),
            "location_in_paper": c.get("location_in_paper"),
            "confidence": c.get("confidence"),
        }

    snippets = [_shape(c) for c in claims[:limit] if c.get("verbatim_quote")]
    return {
        "doi": doi,
        "title": (source or {}).get("title") if isinstance(source, dict) else None,
        "year": (source or {}).get("year") if isinstance(source, dict) else None,
        "snippets": snippets,
        "n_claims_total": len(claims),
        "query": q,
    }


# ── Temporal View & Evolution ────────────────────────────────────────────────

@app.get("/api/time", summary="Browse by time period")
def api_time_browse(
    decade: Optional[str] = Query(None, description="Decade, e.g. '2020s'"),
    year: Optional[int] = Query(None, description="Year, e.g. 2024"),
    quarter: Optional[str] = Query(None, description="Quarter, e.g. 'q1'"),
):
    """
    Browse claims organized by time period.

    Hierarchy: decades -> years -> quarters.
    Call with no params to see decade overview, then drill down.

    **Agent usage:**
    - `GET /api/time` — list decades
    - `GET /api/time?decade=2020s` — years in 2020s
    - `GET /api/time?year=2024` — claim type distribution for 2024
    - `GET /api/time?year=2024&quarter=q4` — claims from 2024 Q4
    """
    return db.get_by_time_period(decade=decade, year=year, quarter=quarter)


@app.get("/api/temporal/{view_id}/{path:path}", summary="Temporal overlay for a node")
def api_temporal_overlay(view_id: str, path: str):
    """
    Get year-by-year breakdown of claims at a specific tree node.

    Returns claim counts per year with type distribution, surge/decline detection.
    Enables: "How has Suzuki coupling research evolved from 2015 to 2025?"

    **Agent usage:** `GET /api/temporal/by_reaction_type/catalysis/cross_coupling`
    """
    requested_path = path
    path, aliased = resolve_tree_path(view_id, path)
    result = db.get_temporal_overlay(view_id, path)
    if aliased and isinstance(result, dict):
        result = dict(result)
        result["requested_path"] = requested_path
        result["canonical_path"] = path
    return result


@app.get("/api/evolution/{view_id}/{path:path}", summary="Evolution timeline")
def api_evolution(view_id: str, path: str):
    """
    Rich evolution timeline for a tree node.

    Returns year-by-year data with claim counts, type distribution,
    top papers (by citation), and surge annotations.

    **Agent usage:** `GET /api/evolution/by_reaction_type/catalysis/cross_coupling`
    """
    requested_path = path
    path, aliased = resolve_tree_path(view_id, path)
    result = db.get_evolution_timeline(view_id, path)
    if aliased and isinstance(result, dict):
        result = dict(result)
        result["requested_path"] = requested_path
        result["canonical_path"] = path
    return result


# ── Reading List ─────────────────────────────────────────────────────────────

@app.get("/api/reading-list/{view_id}/{path:path}", summary="Reading list for a topic")
def api_reading_list(
    request: Request,
    view_id: str, path: str,
    limit: int = Query(15, ge=1, le=100),
):
    """
    Generate a reading list of papers for a topic/tree node.

    Papers are grouped into tiers (Foundational, Key Results, Recent Advances)
    ranked by citation count. Instant, no LLM cost.

    **Agent usage:** `GET /api/reading-list/by_reaction_type/catalysis/cross_coupling`
    """
    requested_path = path
    path, aliased = resolve_tree_path(view_id, path)
    result = db.get_reading_list(view_id, path, limit=limit)
    if result["total_papers"] == 0:
        raise HTTPException(404, f"No papers found for {view_id}/{path}")
    if aliased:
        result = dict(result)
        result["requested_path"] = requested_path
        result["canonical_path"] = path
    return result


# ── Discoveries Feed ────────────────────────────────────────────────────────

@app.get("/api/feed", summary="Discoveries feed")
def api_feed(
    limit: int = Query(20, ge=1, le=100),
    days: int = Query(7, ge=1, le=90),
):
    """
    AskChem Discoveries: highest-surprise claims from recently ingested papers.

    Like an AlphaArXiv for chemistry — curated by surprise score.

    **Agent usage:** `GET /api/feed?limit=10&days=7`
    """
    items = db.get_discoveries_feed(limit=limit, days=days)
    return {"discoveries": items, "count": len(items), "period_days": days}


# ── Subscriptions ───────────────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    email: str
    sub_type: str  # topic, author, query
    target: str    # tree path, author name, or search query
    frequency: str = "weekly"  # daily, weekly


@app.post("/api/me/subscriptions", summary="Subscribe to updates (authenticated)")
def api_me_subscribe(req: SubscribeRequest, request: Request):
    """
    Subscribe to updates for a topic, author, or search query.
    Requires login. Subscriptions are attached to the logged-in user.
    """
    user = _require_user(request)
    if req.sub_type not in ("topic", "author", "query"):
        raise HTTPException(400, "sub_type must be 'topic', 'author', or 'query'")
    if req.frequency not in ("daily", "weekly"):
        raise HTTPException(400, "frequency must be 'daily' or 'weekly'")
    if not req.target.strip():
        raise HTTPException(400, "target is required")

    email = (req.email or user.get("email") or "").strip() or None
    out = db.add_subscription(
        user_id=user["user_id"],
        sub_type=req.sub_type,
        target=req.target.strip(),
        frequency=req.frequency,
        email=email,
    )
    return {"subscription_id": out["subscription_id"], "status": "active"}


@app.get("/api/me/subscriptions", summary="My active subscriptions")
def api_me_list_subscriptions(request: Request):
    user = _require_user(request)
    subs = db.get_user_subscriptions(user["user_id"])
    return {"subscriptions": subs, "count": len(subs)}


@app.delete("/api/me/subscriptions/{sub_id}", summary="Cancel one of my subscriptions")
def api_me_cancel_subscription(sub_id: int, request: Request):
    user = _require_user(request)
    try:
        db.cancel_user_subscription(user["user_id"], sub_id)
    except ValueError as e:
        raise HTTPException(404 if "not found" in str(e) else 403, str(e))
    return {"status": "cancelled", "subscription_id": sub_id}


@app.get("/api/me/subscriptions/{sub_id}/history", summary="Notification history for one of my subs")
def api_me_subscription_history(sub_id: int, request: Request, limit: int = Query(20, ge=1, le=100)):
    user = _require_user(request)
    row = db.get_subscription_row(sub_id)
    if not row or row.get("user_id") != user["user_id"]:
        raise HTTPException(404, "Subscription not found")
    history = db.get_notification_history(sub_id, limit=limit)
    return {"subscription_id": sub_id, "history": history, "count": len(history)}


# ── Author View ─────────────────────────────────────────────────────────────

@app.get("/api/authors", summary="Search or list top authors")
def api_authors(
    q: Optional[str] = Query(None, description="Search authors by name"),
    topic: Optional[str] = Query(None, description="Find experts on a topic"),
    view: Optional[str] = Query(None, description="View to filter by"),
    path: Optional[str] = Query(None, description="Tree path to filter by"),
    limit: int = Query(200, ge=1, le=1000),
):
    """
    Search authors, find experts, or list top authors for a tree node.

    **Agent usage:**
    - `GET /api/authors?q=John+Hartwig` — search by name
    - `GET /api/authors?topic=CO2+reduction` — find experts on a topic
    - `GET /api/authors?view=by_reaction_type&path=catalysis` — top experts in a tree node
    """
    if q:
        return {"authors": db.search_authors(q, limit=limit), "query": q}
    if topic:
        return {"authors": db.find_experts(topic, limit=limit), "topic": topic}
    return {"authors": db.get_top_authors(view_id=view, path=path, limit=limit)}


@app.get("/api/authors/{author_id}/network", summary="Co-authorship network")
def api_author_network(
    author_id: str,
    depth: int = Query(1, ge=1, le=2),
    limit: int = Query(30, ge=1, le=100),
):
    """
    Co-authorship ego network for an author.

    Returns nodes (authors) and edges (co-authorship links with paper counts).

    **Agent usage:** `GET /api/authors/A5060505275/network?depth=1`
    """
    network = db.get_coauthor_network(author_id, depth=depth, limit=limit)
    return network


@app.get("/api/authors/{author_id}", summary="Get author profile")
def api_author_profile(author_id: str):
    """
    Full author profile with papers, claims breakdown, and research areas.

    **Agent usage:** `GET /api/authors/A5060505275`
    """
    profile = db.get_author_profile(author_id)
    if not profile:
        raise HTTPException(404, f"Author '{author_id}' not found")
    return profile


# ── Paper Submission ─────────────────────────────────────────────────────────

class SubmitPaperRequest(BaseModel):
    doi: str
    name: str = ""
    email: str = ""
    notes: str = ""
    callback_url: str = ""


def _validate_callback_url(url: str) -> str | None:
    """Validate callback URL: must be HTTPS and not point to private IPs."""
    if not url:
        return None
    if not url.startswith("https://"):
        return None
    if _PRIVATE_IP_RE.match(url):
        return None
    return url


_submit_buckets: dict[str, list[float]] = {}
SUBMIT_LIMIT = 10
SUBMIT_WINDOW = 3600  # 1 hour


@app.post("/api/submit", summary="Submit a paper for extraction")
async def api_submit_paper(req: SubmitPaperRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Submit a paper (by DOI) to be added to the AskChem index.

    The system will:
    1. Fetch metadata from Semantic Scholar
    2. Extract claims from the abstract using GPT
    3. Classify claims into the 5-view hierarchy
    4. Add to the live index

    Optionally provide `callback_url` (HTTPS only) to receive a POST when processing completes.
    Limited to 10 submissions per IP per hour.

    Returns a submission ID for tracking progress.
    """
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _submit_buckets.setdefault(ip, [])
    _submit_buckets[ip] = [t for t in bucket if now - t < SUBMIT_WINDOW]
    if len(_submit_buckets[ip]) >= SUBMIT_LIMIT:
        raise HTTPException(429, "Too many submissions. Please try again later.")

    existing = db.get_source(req.doi)
    if existing:
        claims = db.get_claims_by_doi(req.doi)
        return {
            "status": "already_indexed",
            "message": f"This paper is already in AskChem with {len(claims)} claims.",
            "doi": req.doi,
            "claims_count": len(claims),
        }

    safe_callback = _validate_callback_url(req.callback_url)

    _submit_buckets[ip].append(now)
    submission_id = db.add_submission(req.doi, req.name, req.email, req.notes)
    background_tasks.add_task(
        process_submission, submission_id, req.doi, safe_callback
    )

    return {
        "status": "accepted",
        "submission_id": submission_id,
        "message": "Paper submitted for processing. Claims will be extracted and indexed.",
        "track_url": f"/api/submissions/{submission_id}",
        "stream_url": f"/api/submissions/{submission_id}/stream",
    }


@app.get("/api/submissions/{submission_id}", summary="Check submission status")
def api_submission_status(submission_id: int):
    """Check the status of a paper submission."""
    sub = db.get_submission(submission_id)
    if not sub:
        raise HTTPException(404, "Submission not found")
    result = json.loads(sub['result']) if sub.get('result') else None
    return {
        "id": sub['id'],
        "doi": sub['doi'],
        "status": sub['status'],
        "submitted_at": sub['submitted_at'],
        "submitter_name": sub.get('submitter_name', ''),
        "result": result,
    }


@app.get("/api/submissions/{submission_id}/stream", summary="Stream submission status (SSE)")
async def api_submission_stream(submission_id: int):
    """
    Server-Sent Events stream for real-time submission status updates.

    **Agent usage:** Connect to this endpoint to receive status updates as the
    paper is processed. Events: `status` (processing/completed/failed).
    """
    from starlette.responses import StreamingResponse

    async def event_stream():
        last_status = None
        for _ in range(120):  # poll for up to 10 minutes
            sub = db.get_submission(submission_id)
            if not sub:
                yield f"event: error\ndata: {{\"error\": \"not_found\"}}\n\n"
                return
            status = sub["status"]
            if status != last_status:
                result = json.loads(sub["result"]) if sub.get("result") else None
                payload = json.dumps({
                    "id": sub["id"], "doi": sub["doi"],
                    "status": status, "result": result,
                })
                yield f"event: status\ndata: {payload}\n\n"
                last_status = status
            if status in ("completed", "failed"):
                return
            await asyncio.sleep(5)
        yield f"event: timeout\ndata: {{\"message\": \"stream timeout\"}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/submissions", summary="List recent submissions")
def api_list_submissions(status: Optional[str] = None, limit: int = Query(20, ge=1, le=100)):
    """List recent paper submissions and their processing status (no submitter PII)."""
    subs = db.list_submissions(status=status, limit=limit)
    public = []
    for s in subs:
        public.append({
            "id": s["id"],
            "doi": s["doi"],
            "submitted_at": s["submitted_at"],
            "status": s["status"],
            "result": json.loads(s["result"]) if s.get("result") else None,
        })
    return {"submissions": public, "count": len(public)}


# ── Quality Report ──

@app.get("/api/quality", summary="Data quality report")
def api_quality():
    """Get data quality metrics: extraction stats, validation rates, coverage."""
    cached = _cached("quality", ttl=300)
    if cached is not None:
        return cached
    stats = db.get_stats()
    total_claims = int(stats.get("total_claims", 0))
    total_sources = int(stats.get("total_sources", 0))
    claim_types = stats.get("claim_types", {})
    year_dist = stats.get("year_distribution", {})
    years = [int(y) for y in year_dist.keys() if int(y) > 1900]
    year_range = [min(years), max(years)] if years else [0, 0]

    # Compute extraction depth breakdown and validation stats from DB
    extraction_depth = {"full_paper": 0, "abstract_only": 0, "full_paper_by_model": {}, "full_paper_papers": 0}
    doi_verification = {"total_validated": 0, "crossref_verified": 0, "retracted_caught": 0}
    subfield_coverage = {}
    flag_stats = {"total": 0, "resolved": 0, "open": 0}

    try:
        with db.get_conn() as conn:
            for row in conn.execute(
                "SELECT "
                "  SUM(CASE WHEN extraction_version = 'deep_v1' THEN 1 ELSE 0 END) as full_paper, "
                "  SUM(CASE WHEN extraction_version LIKE '%abstract%' THEN 1 ELSE 0 END) as abstract_only "
                "FROM claims"
            ):
                extraction_depth["full_paper"] = int(row["full_paper"] or 0)
                extraction_depth["abstract_only"] = int(row["abstract_only"] or 0)

            for row in conn.execute(
                "SELECT extraction_model, COUNT(*) as n "
                "FROM claims "
                "WHERE extraction_version = 'deep_v1' "
                "GROUP BY extraction_model "
                "ORDER BY n DESC"
            ):
                extraction_depth["full_paper_by_model"][row["extraction_model"] or "unknown"] = int(row["n"] or 0)

            for row in conn.execute(
                "SELECT COUNT(DISTINCT source_doi) as n "
                "FROM claims "
                "WHERE extraction_version = 'deep_v1' AND source_doi != ''"
            ):
                extraction_depth["full_paper_papers"] = int(row["n"] or 0)

            for row in conn.execute(
                "SELECT COUNT(*) as total, "
                "  SUM(CASE WHEN crossref_verified = 1 THEN 1 ELSE 0 END) as verified, "
                "  SUM(CASE WHEN is_retracted = 1 THEN 1 ELSE 0 END) as retracted "
                "FROM paper_validations"
            ):
                doi_verification["total_validated"] = int(row["total"] or 0)
                doi_verification["crossref_verified"] = int(row["verified"] or 0)
                doi_verification["retracted_caught"] = int(row["retracted"] or 0)

            if doi_verification["total_validated"] > 0:
                doi_verification["verification_rate"] = round(
                    doi_verification["crossref_verified"] / doi_verification["total_validated"] * 100, 1
                )
            else:
                doi_verification["verification_rate"] = 0

            # Subfield coverage: aggregate the second segment of by_claim_type paths,
            # e.g. properties/materials -> materials.
            for row in conn.execute(
                "SELECT path, claim_count FROM tree_nodes "
                "WHERE view_id = 'by_claim_type' AND level = 2"
            ):
                try:
                    path = row["path"] or ""
                    if "/" not in path:
                        continue
                    subfield = path.split("/", 1)[1]
                    subfield_coverage[subfield] = subfield_coverage.get(subfield, 0) + int(row["claim_count"] or 0)
                except Exception:
                    pass

            for row in conn.execute(
                "SELECT COUNT(*) as total, "
                "  SUM(CASE WHEN status IN ('resolved', 'dismissed') THEN 1 ELSE 0 END) as resolved, "
                "  SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_count "
                "FROM community_flags"
            ):
                flag_stats["total"] = int(row["total"] or 0)
                flag_stats["resolved"] = int(row["resolved"] or 0)
                flag_stats["open"] = int(row["open_count"] or 0)
    except Exception:
        pass

    # SMILES validation stats
    smiles_stats = {"total": 0, "valid": 0, "invalid": 0}
    try:
        with db.get_conn() as conn:
            for row in conn.execute(
                "SELECT COUNT(*) as total, "
                "  SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as valid "
                "FROM smiles_validations"
            ):
                smiles_stats["total"] = int(row["total"] or 0)
                smiles_stats["valid"] = int(row["valid"] or 0)
                smiles_stats["invalid"] = smiles_stats["total"] - smiles_stats["valid"]
    except Exception:
        pass

    # Ingestion history: last 90 days of (date, extraction_version) groups
    # derived from claims.extracted_at. Used by the Quality tab to show
    # operators "what's been added recently" and the public a freshness
    # signal ("Index last updated <date>").
    ingestion_history: list[dict] = []
    last_updated = None
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(extracted_at) AS day, "
                "       COUNT(*) AS n_claims, "
                "       COUNT(DISTINCT source_doi) AS n_papers, "
                "       extraction_version "
                "FROM claims "
                "WHERE extracted_at >= DATE('now', '-90 days') "
                "  AND extracted_at != '' "
                "GROUP BY day, extraction_version "
                "ORDER BY day DESC, n_claims DESC "
                "LIMIT 60"
            ).fetchall()
            # Aggregate same day across extraction_versions for the table,
            # but keep the version breakdown for the tooltip.
            by_day: dict[str, dict] = {}
            for r in rows:
                day = r["day"]
                if not day:
                    continue
                entry = by_day.setdefault(day, {
                    "day": day,
                    "papers": 0,
                    "claims": 0,
                    "by_version": {},
                })
                entry["claims"] += int(r["n_claims"] or 0)
                # Papers are not strictly additive across versions for the
                # same day, but for human display the slight overcount is
                # fine; we capture the dominant version anyway.
                entry["papers"] += int(r["n_papers"] or 0)
                entry["by_version"][r["extraction_version"] or "unknown"] = {
                    "claims": int(r["n_claims"] or 0),
                    "papers": int(r["n_papers"] or 0),
                }
            ingestion_history = sorted(
                by_day.values(), key=lambda x: x["day"], reverse=True
            )[:30]
            if ingestion_history:
                last_updated = ingestion_history[0]["day"]
    except Exception:
        pass

    # Build dynamic limitations
    limitations = []
    if smiles_stats["total"] > 0:
        pct = round(smiles_stats["valid"] / smiles_stats["total"] * 100, 1) if smiles_stats["total"] > 0 else 0
        limitations.append(
            f"SMILES validation: {smiles_stats['valid']}/{smiles_stats['total']} validated SMILES are chemically valid ({pct}%); "
            f"{smiles_stats['invalid']} invalid SMILES flagged"
        )
    else:
        limitations.append("SMILES validation not yet run on all claims")
    try:
        with db.get_conn() as conn:
            ccount = conn.execute(
                "SELECT COUNT(*) FROM contradictions WHERE gemini_verdict = 'confirmed'"
            ).fetchone()[0]
        if ccount > 0:
            limitations.append(
                f"Contradiction detection: {ccount} confirmed contradictions detected "
                f"via PAW pre-filter + Gemini verification"
            )
        else:
            limitations.append("Contradiction detection pipeline deployed; batch scan pending")
    except Exception:
        limitations.append("Contradiction detection pipeline deployed; batch scan pending")
    limitations.append(
        "Abstract-only extractions capture headline findings; full-paper PDF "
        "claims were extracted with Gemini 3.1 Pro and include tables, "
        "conditions, and mechanistic detail"
    )

    quality = {
        "total_claims": total_claims,
        "total_sources": total_sources,
        "claim_type_distribution": claim_types,
        "year_range": year_range,
        "year_distribution": year_dist,
        "extraction_depth": extraction_depth,
        "doi_verification": doi_verification,
        "subfield_coverage": subfield_coverage,
        "flag_stats": flag_stats,
        "ltree_feedback": db.get_ltree_feedback_summary(),
        "smiles_validation": smiles_stats,
        "extraction_models": [
            "gemini-3.1-pro (full-paper PDF extraction)",
            "gemini-3.1-pro-preview (abstract extraction and classification)",
            "gpt-5-mini (legacy abstract extraction and classification)",
        ],
        "citation_source": stats.get("citation_source", "Semantic Scholar Academic Graph API"),
        "citation_source_url": stats.get("citation_source_url", "https://api.semanticscholar.org/"),
        "citations_updated_at": stats.get("citations_updated_at", ""),
        "known_limitations": limitations,
        "ingestion_history": ingestion_history,
        "last_updated": last_updated,
    }
    _set_cache("quality", quality)
    return quality


# ── Benchmark ────────────────────────────────────────────────────────────────

@app.get("/api/benchmark", summary="AskChem-Bench methodology and results")
def api_benchmark():
    """Serve the AskChem-Bench question bank, methodology, and aggregate results."""
    cached = _cached("benchmark", ttl=3600)
    if cached is not None:
        return cached

    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    questions = []
    aggregate = {}
    subset_results = {}
    benchmark_runs = {}

    bench_script = scripts_dir / "benchmark_chemtree.py"
    if bench_script.exists():
        import ast
        try:
            tree = ast.parse(bench_script.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "BENCHMARK_QUESTIONS":
                            questions = ast.literal_eval(node.value)
        except Exception:
            pass

    for model_name in ["gpt-5.5", "gpt-5.4", "gpt-4.1", "gpt-4o"]:
        safe = model_name.replace("/", "_")
        results_file = scripts_dir / f"benchmark_results_{safe}.json"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                aggregate[model_name] = data.get("aggregate", {})
                subset_results[model_name] = data.get("aggregate_subsets", {})
                benchmark_runs[model_name] = {
                    "generated_at": data.get("generated_at", ""),
                    "methods": data.get("methods", {}),
                    "subsets": data.get("subsets", {}),
                    "askchem_snapshot": data.get("askchem_snapshot", {}),
                }
            except Exception:
                pass

    # Clean questions for public consumption (remove askchem_params internals)
    public_questions = []
    for q in questions:
        public_questions.append({
            "id": q.get("id", ""),
            "task": q.get("task", ""),
            "domain": q.get("domain", ""),
            "question": q.get("question", ""),
        })

    result = {
        "name": "AskChem-Bench",
        "version": "1.1",
        "total_questions": len(public_questions),
        "task_types": {
            "CA": {"name": "Cross-Paper Condition Aggregation", "description": "Aggregate catalysts, conditions, and performance metrics across multiple papers"},
            "TC": {"name": "Temporal Claim Tracking", "description": "Track how scientific understanding of a topic has evolved over time"},
            "CS": {"name": "Contradiction Surfacing", "description": "Identify contradictory evidence and competing claims across papers"},
        },
        "methodology": {
            "evaluation_protocol": "GPT-5.5 answers each question with and without retrieval context. AskChem rewrites the question into 3-4 keyword sub-queries, fans them out to /api/search, diversifies the merged claim pool, and a grounded synthesiser writes the answer strictly from those claims. Paperclip unified uses the same rewriter and synthesiser but retrieves papers via Paperclip (hybrid/bm25 search over PMC+arXiv). Edison Scientific (FutureHouse paperqa3) runs on all 30 questions.",
            "doi_verification": "Every DOI in a model answer is checked via CrossRef (existence, citation count, year). Relevance is judged 0-3 by gemini-3.1-pro-preview against the AskChem-extracted claim (claim-grounded mode) when available, otherwise against the paper's title+abstract (paper-grounded mode). Rubric: 3=directly answers, 2=on topic, 1=loosely related, 0=irrelevant. Each system is judged on what it actually surfaced — AskChem on its claims, Paperclip/Edison on their papers.",
            "modes": [
                {"id": "alone",             "name": "LLM only",          "description": "GPT-5.5 with no AskChem context."},
                {"id": "unified",           "name": "+ AskChem",         "description": "LLM rewriter + /api/search + diversified pool + grounded synthesis."},
                {"id": "paperclip_unified", "name": "+ Paperclip",       "description": "Same rewriter and GPT-5.5 synthesis as AskChem, but retrieval via Paperclip search (PMC+arXiv)."},
                {"id": "edison_scientific", "name": "Edison Scientific", "description": "FutureHouse paperqa3 on all 30 questions."},
                {"id": "notebooklm",        "name": "NotebookLM",        "description": "Google NotebookLM web UI (no public API). Each question pasted by hand into the NotebookLM Deep Research interface; answers stored in data/eval/notebooklm_answers.md and scored by scripts/score_external_answer.py through the same paper-grounded judge as the other systems."},
            ],
            "metrics": [
                {"id": "doi_existence_rate",       "name": "DOI %",            "description": "Share of cited DOIs that resolve to a real CrossRef record."},
                {"id": "citation_density",         "name": "Cites/answer",     "description": "Average number of verified DOIs cited per answer."},
                {"id": "grounded_specificity",     "name": "Grounded specificity", "description": "Quantitative tokens (yields, temps, units) that share a sentence with a citation marker."},
                {"id": "citation_count_mean",      "name": "Avg cites/paper",  "description": "Mean CrossRef citation count of the papers cited per answer."},
                {"id": "recent_high_impact_rate",  "name": "Recent impact",    "description": "Fraction of cited papers with >=50 citations published in the last 5 years."},
                {"id": "paper_relevance_mean",     "name": "Relevance (mean 0-3)", "description": "Mean gemini-3.1-pro-preview judge score on the 0-3 rubric. Claim-grounded for AskChem unified; paper-grounded (title+abstract) for Paperclip and Edison — each system is judged on the evidence it actually surfaces."},
                {"id": "paper_relevance_high_rate","name": "On-topic (>=2 rate)", "description": "Fraction of cited evidence scored >=2 by the judge ('on topic' or 'directly answers'). Complements Relevance: a system that mostly scores 2s gets a high On-topic but middling Relevance."},
                {"id": "edison_overlap_rate",      "name": "Edison overlap",      "description": "Per-question fraction of Edison Scientific's cited DOIs that the system also cited (recall against Edison as a retrieval baseline). 0 means no shared papers; 1 means the system found everything Edison did. Not reported for Edison's own row (trivially 1.0)."},
            ],
        },
        "questions": public_questions,
        "results": aggregate,
        "subset_results": subset_results,
        "runs": benchmark_runs,
        "reproducibility": {
            "install": "pip install openai requests",
            "run": "OPENAI_API_KEY=sk-... python scripts/benchmark_chemtree.py",
            "env_vars": {
                "BENCH_MODEL": "LLM model to benchmark (default: gpt-5.5)",
                "ASKCHEM_API": "AskChem API base URL (default: https://askchem.org/api)",
                "EDISON": "Optional Edison Scientific API key to run the FutureHouse paperqa3 baseline",
                "PAPERCLIP": "Paperclip API key (same as PAPERCLIP_API_KEY) for the Paperclip retrieval baseline",
                "BENCH_RESUME": "Set to 0 to ignore cached partial results and rerun everything",
            },
        },
    }
    _set_cache("benchmark", result)
    return result


# ── Contradiction Detection ──────────────────────────────────────────────────

@app.get("/api/contradictions", summary="Browse stored contradictions")
def api_contradictions_list(
    view_id: Optional[str] = Query(None, description="Filter by view"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Return pre-computed contradictions from the database.
    These are detected via a PAW pre-filter + Gemini verification pipeline.
    """
    return db.get_contradictions(view_id=view_id, limit=limit, offset=offset)


@app.get("/api/contradictions/{view_id}/{path:path}", summary="Find contradictions at a tree node")
def api_contradictions_node(view_id: str, path: str, limit: int = Query(20, ge=1, le=100)):
    """
    Return pre-computed contradictions at or below a specific tree node.
    """
    requested_path = path
    path, aliased = resolve_tree_path(view_id, path)
    result = db.get_contradictions(
        view_id=view_id, node_path=path, limit=limit,
    )
    if aliased and isinstance(result, dict):
        result = dict(result)
        result["requested_path"] = requested_path
        result["canonical_path"] = path
    return result


# ── Community Flagging ────────────────────────────────────────────────────────

VALID_FLAG_TYPES = Literal[
    "wrong_claim", "wrong_classification", "not_chemistry",
    "duplicate", "low_quality", "other",
]


class FlagRequest(BaseModel):
    claim_id: str
    flag_type: VALID_FLAG_TYPES
    category: str = ""
    comment: str = ""
    suggested_fix: str = ""
    reporter_name: str = ""
    reporter_email: str = ""


_flag_buckets: dict[str, list[float]] = {}
FLAG_LIMIT = 20
FLAG_WINDOW = 3600


@app.post("/api/flag", summary="Flag a claim for review")
def api_flag_claim(req: FlagRequest, request: Request):
    """
    Community quality control: flag a claim as incorrect, misclassified,
    or low quality. Flags are reviewed by maintainers.
    Limited to 20 flags per IP per hour.

    flag_type options:
    - wrong_claim: The claim content is factually incorrect
    - wrong_classification: Claim is in the wrong category/view
    - not_chemistry: This is not a chemistry claim
    - duplicate: This claim duplicates another
    - low_quality: Claim is too vague or poorly extracted
    - other: Other issue (describe in comment)
    """
    if len(req.comment) > 2000:
        raise HTTPException(400, "Comment must be under 2000 characters")

    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _flag_buckets.setdefault(ip, [])
    _flag_buckets[ip] = [t for t in bucket if now - t < FLAG_WINDOW]
    if len(_flag_buckets[ip]) >= FLAG_LIMIT:
        raise HTTPException(429, "Too many flags submitted. Please try again later.")

    claim = db.get_claim(req.claim_id)
    if not claim:
        raise HTTPException(404, "Claim not found")
    try:
        flag_id = db.add_flag(
            claim_id=req.claim_id,
            flag_type=req.flag_type,
            category=req.category,
            comment=req.comment,
            suggested_fix=req.suggested_fix,
            reporter_name=req.reporter_name,
            reporter_email=req.reporter_email,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    _flag_buckets[ip].append(now)
    return {"flag_id": flag_id, "status": "submitted", "message": "Thank you for your feedback."}


@app.get("/api/flags", summary="List community flags")
def api_list_flags(
    request: Request,
    status: Optional[str] = None,
    flag_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List community flags. Open flags require admin auth; public sees resolved/reviewed/dismissed only."""
    admin = _is_admin_request(request)
    if not admin:
        if status == "open":
            raise HTTPException(403, "Admin authentication required to list open flags")
        flags = db.list_flags(
            status=status, flag_type=flag_type, limit=limit, offset=offset,
            public_only=(status is None),
        )
    else:
        flags = db.list_flags(status=status, flag_type=flag_type, limit=limit, offset=offset)
    summary = db.get_flag_summary()
    return {"flags": flags, "summary": summary}


@app.get("/api/flags/{claim_id}", summary="Flags for a specific claim")
def api_claim_flags(claim_id: str, request: Request):
    """Get flags for a specific claim (open flags omitted for non-admin)."""
    flags = db.get_flags_for_claim(claim_id)
    if not _is_admin_request(request):
        flags = [f for f in flags if f.get("status") != "open"]
    return {"claim_id": claim_id, "flags": flags}


# ── Living-tree feedback (node/placement quality) ────────────────────────────

VALID_LTREE_FEEDBACK_KINDS = {"mislabeled", "misplaced", "duplicate",
                              "wrong_parent", "missing", "other"}
_ltree_fb_buckets: dict[str, list[float]] = {}


class LtreeFeedbackRequest(BaseModel):
    """Feedback on a living-tree branch or a specific paper placement."""

    view_id: str
    kind: str
    node_id: Optional[str] = None
    doi: Optional[str] = None
    comment: str = ""
    reporter_name: str = ""
    reporter_email: str = ""


@app.post("/api/ltree/feedback", summary="Submit feedback on a living-tree node/placement")
def api_ltree_feedback(req: LtreeFeedbackRequest, request: Request):
    """
    Community quality control for the living tree. Report a branch that is
    mislabeled, a paper that is misplaced, duplicate branches, a wrong parent,
    a missing branch, or other issues. Limited to 20 submissions per IP per hour.

    kind: mislabeled | misplaced | duplicate | wrong_parent | missing | other
    """
    if req.kind not in VALID_LTREE_FEEDBACK_KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(VALID_LTREE_FEEDBACK_KINDS)}")
    if len(req.comment) > 2000:
        raise HTTPException(400, "Comment must be under 2000 characters")
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = [t for t in _ltree_fb_buckets.get(ip, []) if now - t < FLAG_WINDOW]
    if len(bucket) >= FLAG_LIMIT:
        raise HTTPException(429, "Too many submissions. Please try again later.")
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:12]
    try:
        fid = db.add_ltree_feedback(
            view_id=req.view_id, kind=req.kind, node_id=req.node_id, doi=req.doi,
            comment=req.comment, reporter_name=req.reporter_name,
            reporter_email=req.reporter_email, ip_hash=ip_hash)
    except ValueError as e:
        raise HTTPException(400, str(e))
    bucket.append(now)
    _ltree_fb_buckets[ip] = bucket
    return {"feedback_id": fid, "status": "submitted",
            "message": "Thank you for your feedback."}


# ── Claim graph (typed inter-claim edges) ────────────────────────────────────

EDGE_INTRA_TYPES = {
    "supports", "assumes", "bounded_by", "interprets",
    "derives_from", "sub_step_of",
}
EDGE_CROSS_TYPES = {
    "uses_method_of", "uses_assumption_of", "extends",
    "supersedes", "contradicts", "cites_as_evidence",
}
EDGE_ALL_TYPES = EDGE_INTRA_TYPES | EDGE_CROSS_TYPES


def _node_summary_rows(conn, claim_ids: list[str]) -> dict[str, dict]:
    """Compact node descriptors keyed by claim_id, for graph payloads."""
    if not claim_ids:
        return {}
    placeholders = ",".join("?" * len(claim_ids))
    rows = conn.execute(
        f"""SELECT claim_id, claim_type, source_doi, source_paper_title,
                   substr(verbatim_quote, 1, 200) AS snippet
              FROM claims
             WHERE claim_id IN ({placeholders})""",
        claim_ids,
    ).fetchall()
    out = {}
    for r in rows:
        out[r["claim_id"]] = {
            "id": r["claim_id"],
            "claim_type": r["claim_type"],
            "paper": r["source_doi"],
            "paper_title": r["source_paper_title"],
            "snippet": r["snippet"] or "",
        }
    return out


@app.get("/api/edges", summary="Browse claim-to-claim edges")
def api_edges(
    from_id: Optional[str] = Query(None, alias="from", description="Source claim_id filter"),
    to_id: Optional[str] = Query(None, alias="to", description="Target claim_id filter"),
    edge_type: Optional[str] = Query(None, description="Edge type filter"),
    extractor: Optional[str] = Query(None, description="Extractor tag filter"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List claim-to-claim edges with optional filters.

    Edges encode typed scientific relationships extracted by the claim-graph
    pipeline (supports / assumes / bounded_by / contradicts / uses_method_of …).
    """
    if edge_type and edge_type not in EDGE_ALL_TYPES:
        raise HTTPException(400, f"unknown edge_type: {edge_type}")
    where, params = ["1=1"], []
    if from_id:
        where.append("from_claim_id = ?"); params.append(from_id)
    if to_id:
        where.append("(to_claim_id = ? OR to_doi = ?)"); params.extend([to_id, to_id])
    if edge_type:
        where.append("edge_type = ?"); params.append(edge_type)
    if extractor:
        where.append("extractor = ?"); params.append(extractor)
    sql = (
        "SELECT id, from_claim_id, edge_type, to_claim_id, to_doi, "
        "       confidence, evidence, extractor, extracted_at "
        "  FROM claim_edges "
        " WHERE " + " AND ".join(where) +
        " ORDER BY id "
        " LIMIT ? OFFSET ?"
    )
    with db.get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        edges = [
            {
                "id": r["id"],
                "from": r["from_claim_id"],
                "to": r["to_claim_id"] or None,
                "to_doi": r["to_doi"] or None,
                "type": r["edge_type"],
                "confidence": r["confidence"],
                "evidence": r["evidence"],
                "extractor": r["extractor"],
                "extracted_at": r["extracted_at"],
            }
            for r in rows
        ]
    return {"edges": edges, "limit": limit, "offset": offset}


@app.get("/api/claims/{claim_id}/neighborhood", summary="Inbound and outbound edges around a claim")
def api_claim_neighborhood(
    claim_id: str,
    direction: str = Query("both", pattern="^(in|out|both)$"),
    edge_types: Optional[str] = Query(None, description="Comma-separated edge type filter"),
    depth: int = Query(1, ge=1, le=2),
    limit: int = Query(100, ge=1, le=500),
):
    """Return the claim, its 1- or 2-hop neighbors, and the edges between them.

    `direction=in` returns claims that point to this one (e.g. what supports it);
    `direction=out` returns claims this one points to (e.g. what it assumes);
    `direction=both` returns both.
    """
    types_filter: Optional[set[str]] = None
    if edge_types:
        requested = {t.strip() for t in edge_types.split(",") if t.strip()}
        bad = requested - EDGE_ALL_TYPES
        if bad:
            raise HTTPException(400, f"unknown edge_types: {sorted(bad)}")
        types_filter = requested

    visited_nodes: set[str] = {claim_id}
    visited_dois: set[str] = set()
    edges_out: list[dict] = []
    frontier = {claim_id}

    with db.get_conn() as conn:
        for _ in range(depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            placeholders = ",".join("?" * len(frontier))
            params: list = list(frontier)
            clauses = []
            if direction in ("out", "both"):
                clauses.append(f"from_claim_id IN ({placeholders})")
                params = list(frontier) + params if direction == "both" else params
            if direction in ("in", "both"):
                clauses.append(f"to_claim_id IN ({placeholders})")
            sql = (
                "SELECT from_claim_id, edge_type, to_claim_id, to_doi, "
                "       confidence, evidence, extractor "
                "  FROM claim_edges "
                " WHERE (" + " OR ".join(clauses) + ")"
            )
            if types_filter:
                tplaceholders = ",".join("?" * len(types_filter))
                sql += f" AND edge_type IN ({tplaceholders})"
                params.extend(sorted(types_filter))
            sql += " LIMIT ?"
            params.append(limit)
            for row in conn.execute(sql, params):
                f, et, tc, td, conf, ev, ex = row
                edge = {
                    "from": f,
                    "to": tc or None,
                    "to_doi": td or None,
                    "type": et,
                    "confidence": conf,
                    "evidence": ev,
                    "extractor": ex,
                }
                edges_out.append(edge)
                if tc and tc not in visited_nodes:
                    next_frontier.add(tc)
                    visited_nodes.add(tc)
                if f not in visited_nodes:
                    next_frontier.add(f)
                    visited_nodes.add(f)
                if td:
                    visited_dois.add(td)
            frontier = next_frontier

        node_summaries = _node_summary_rows(conn, list(visited_nodes))

    nodes = [
        node_summaries.get(nid, {"id": nid, "missing": True})
        for nid in visited_nodes
    ]
    return {
        "center": claim_id,
        "depth": depth,
        "direction": direction,
        "nodes": nodes,
        "external_dois": sorted(visited_dois),
        "edges": edges_out,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges_out),
        },
    }


@app.get("/api/search/graph", summary="Induced edge subgraph over the top search hits")
def api_search_graph(
    request: Request,
    q: Optional[str] = Query(None, max_length=MAX_QUERY_LENGTH),
    claim_ids: Optional[str] = Query(
        None,
        description="Comma-separated claim IDs to use directly as the hit set, "
                    "bypassing the search. Useful when the frontend already knows "
                    "which claims should anchor the graph (e.g. a tree subtree page).",
    ),
    limit: int = Query(50, ge=1, le=500, description="Top-N search hits to draw the subgraph over"),
    offset: int = Query(0, ge=0, description="Skip the first N hits (mirrors /api/search pagination)"),
    expand: str = Query("bridges", pattern="^(strict|bridges|one_hop)$"),
):
    """Return the typed-edge subgraph induced by a set of anchor claims.

    Two input modes:
      • `q` (+ optional `limit`/`offset`): runs the ranked search and uses the
        returned hits as anchors — keeps the graph in sync with the list view.
      • `claim_ids`: skips the search and uses those IDs directly. Use this
        when the client already has a specific set of claims in mind, e.g.
        the visible page of a tree subtree.

    Then queries `claim_edges` for relations between anchors (`expand=strict`),
    plus optional bridge nodes that connect two anchors (`expand=bridges`),
    or full 1-hop expansion (`expand=one_hop`, capped to keep hub claims
    under control).
    """
    if claim_ids:
        hit_ids = [s.strip() for s in claim_ids.split(",") if s.strip()][:500]
        results = []  # no relevance scores in this path
    elif q:
        search_result = db.search_claims(q, limit=limit, offset=offset)
        results = search_result.get("results") or search_result.get("claims") or []
        hit_ids = [c["claim_id"] for c in results if c.get("claim_id")]
    else:
        raise HTTPException(status_code=400, detail="Provide either q or claim_ids")
    hit_set = set(hit_ids)

    if not hit_ids:
        return {
            "query": q,
            "claim_ids": claim_ids,
            "expand": expand,
            "nodes": [],
            "edges": [],
            "stats": {"node_count": 0, "edge_count": 0, "hits": 0, "isolated_hits": 0},
        }

    with db.get_conn() as conn:
        conn.row_factory = sqlite3.Row
        ph = ",".join("?" * len(hit_ids))

        # Edges where BOTH endpoints are search hits (strict mode).
        strict_rows = conn.execute(
            f"""SELECT from_claim_id, edge_type, to_claim_id, to_doi,
                       confidence, evidence, extractor
                  FROM claim_edges
                 WHERE from_claim_id IN ({ph})
                   AND to_claim_id   IN ({ph})""",
            hit_ids + hit_ids,
        ).fetchall()

        included_node_ids = set(hit_set)
        edge_rows = list(strict_rows)
        bridges: set[str] = set()

        if expand == "bridges":
            # Find nodes outside the hit set that are linked to >=2 hits via
            # any direct edge — those are the structurally interesting bridges.
            adj_rows = conn.execute(
                f"""SELECT from_claim_id AS a, to_claim_id AS b, edge_type, confidence, evidence, extractor
                      FROM claim_edges
                     WHERE (from_claim_id IN ({ph}) OR to_claim_id IN ({ph}))
                       AND to_claim_id != ''""",
                hit_ids + hit_ids,
            ).fetchall()
            hit_neighbor_count: dict[str, set[str]] = {}
            for r in adj_rows:
                a, b = r["a"], r["b"]
                if a in hit_set and b not in hit_set:
                    hit_neighbor_count.setdefault(b, set()).add(a)
                elif b in hit_set and a not in hit_set:
                    hit_neighbor_count.setdefault(a, set()).add(b)
            bridges = {n for n, hits in hit_neighbor_count.items() if len(hits) >= 2}
            if bridges:
                included_node_ids |= bridges
                ph2 = ",".join("?" * len(included_node_ids))
                ids2 = list(included_node_ids)
                edge_rows = conn.execute(
                    f"""SELECT from_claim_id, edge_type, to_claim_id, to_doi,
                               confidence, evidence, extractor
                          FROM claim_edges
                         WHERE from_claim_id IN ({ph2})
                           AND to_claim_id   IN ({ph2})""",
                    ids2 + ids2,
                ).fetchall()

        elif expand == "one_hop":
            ONE_HOP_NODE_CAP = 200
            adj_rows = conn.execute(
                f"""SELECT from_claim_id AS a, to_claim_id AS b
                      FROM claim_edges
                     WHERE (from_claim_id IN ({ph}) OR to_claim_id IN ({ph}))
                       AND to_claim_id != ''""",
                hit_ids + hit_ids,
            ).fetchall()
            extra: set[str] = set()
            for r in adj_rows:
                if r["a"] in hit_set and r["b"] not in hit_set:
                    extra.add(r["b"])
                elif r["b"] in hit_set and r["a"] not in hit_set:
                    extra.add(r["a"])
                if len(extra) >= ONE_HOP_NODE_CAP:
                    break
            included_node_ids |= extra
            ph2 = ",".join("?" * len(included_node_ids))
            ids2 = list(included_node_ids)
            edge_rows = conn.execute(
                f"""SELECT from_claim_id, edge_type, to_claim_id, to_doi,
                           confidence, evidence, extractor
                      FROM claim_edges
                     WHERE from_claim_id IN ({ph2})
                       AND to_claim_id   IN ({ph2})""",
                ids2 + ids2,
            ).fetchall()

        score_by_id = {c["claim_id"]: c.get("_relevance_score", 0) for c in results if c.get("claim_id")}
        node_summaries = _node_summary_rows(conn, list(included_node_ids))

    nodes = []
    for nid in included_node_ids:
        meta = node_summaries.get(nid, {"id": nid})
        meta["in_search"] = nid in hit_set
        meta["bridge"] = nid in bridges
        meta["score"] = score_by_id.get(nid, 0)
        nodes.append(meta)
    nodes.sort(key=lambda n: (-n.get("in_search", False), -n.get("score", 0)))

    edges_out = [
        {
            "from": r["from_claim_id"],
            "to": r["to_claim_id"],
            "type": r["edge_type"],
            "confidence": r["confidence"],
            "evidence": r["evidence"],
            "extractor": r["extractor"],
        }
        for r in edge_rows
    ]

    edge_endpoints = {e["from"] for e in edges_out} | {e["to"] for e in edges_out}
    isolated_hits = [nid for nid in hit_ids if nid not in edge_endpoints]

    return {
        "query": q,
        "claim_ids": claim_ids,
        "expand": expand,
        "limit": limit,
        "offset": offset,
        "nodes": nodes,
        "edges": edges_out,
        "isolated_hits": isolated_hits,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges_out),
            "hits": len(hit_ids),
            "bridges": len(bridges),
            "isolated_hits": len(isolated_hits),
        },
    }


@app.get(
    "/api/network",
    summary="Full typed + co-mention subnetwork for a query",
    include_in_schema=False,
)
def api_network(
    q: str = Query(..., max_length=MAX_QUERY_LENGTH),
    limit: int = Query(2000, ge=1, le=5000,
        description="Maximum query-matching claims to consider when building "
                    "the subnetwork (drawn from claims_fts; no per-paper cap)"),
    max_edges: int = Query(2000, ge=1, le=10000, description="Hard cap on total edges returned"),
    per_paper_cap: int = Query(8, ge=1, le=40,
        description="Maximum co-mention spokes emitted from any single paper "
                    "(prevents a 50-hit paper from producing ~1.2K edges)"),
    edge_types: Optional[str] = Query(
        None,
        description="Optional comma-separated subset of edge_type values to include. "
                    "Use 'co_mention' to refer to the synthetic same-paper edges; all "
                    "other names match the typed claim_edges schema "
                    "(e.g. supports,contradicts,uses_method_of).",
    ),
):
    """Return the typed + co-mention subnetwork mentioning a query.

    Unlike ``/api/search/graph``, which is anchored on a relevance-ranked
    list and is constrained to typed ``claim_edges``, this endpoint runs
    FTS directly (no per-paper dedup) and overlays **co-mention** edges
    between claims that share a paper. Co-mention edges are denser by
    orders of magnitude and produce a useful network for popular topics
    (Suzuki, graphene, DFT) where typed edges alone are too sparse to
    render.

    Co-mention construction:
      * Group the FTS-matched hits by ``source_doi``.
      * For each paper with ``k >= 2`` hits, emit a star of edges from a
        single center claim out to the other hits (``min(k-1, per_paper_cap)``
        edges per paper). The star keeps the network sparse but well
        connected; users can mentally fan it out into a clique.
    """
    requested_types: Optional[set[str]] = None
    if edge_types:
        requested_types = {s.strip() for s in edge_types.split(",") if s.strip()}

    # ── Hit set via FTS (bypasses search_claims per-paper cap) ─────────────
    # We build several variant queries so hyphenated / multi-token requests
    # still hit the porter-stemmed index. FTS5's default tokenizer splits on
    # hyphens, so "Buchwald-Hartwig" must be rewritten to "Buchwald Hartwig"
    # (AND-of-tokens) or "Buchwald NEAR Hartwig" to match anything.
    raw = q.strip()
    fts_candidates: list[str] = []
    if " " in raw or "-" in raw:
        # Quoted phrase using whitespace (matches "Buchwald Hartwig" as adjacent tokens).
        phrase = re.sub(r"[\s\-]+", " ", raw).strip()
        if phrase:
            fts_candidates.append(f'"{phrase}"')
        # AND of bare tokens (most permissive).
        toks = [t for t in re.split(r"[\s\-]+", raw) if t and re.match(r"^\w[\w']*$", t)]
        if len(toks) >= 2:
            fts_candidates.append(" AND ".join(toks))
    fts_candidates.append(raw)

    hit_ids: list[str] = []
    seen_hits: set[str] = set()
    with db.get_conn() as conn:
        conn.row_factory = sqlite3.Row
        for fts_q in fts_candidates:
            try:
                rows = conn.execute(
                    "SELECT claim_id FROM claims_fts "
                    "WHERE claims_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_q, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for r in rows:
                cid = r["claim_id"]
                if cid not in seen_hits:
                    seen_hits.add(cid)
                    hit_ids.append(cid)
            if len(hit_ids) >= limit:
                break
        hit_ids = hit_ids[:limit]

        if not hit_ids:
            return {
                "query": q,
                "nodes": [],
                "edges": [],
                "stats": {
                    "node_count": 0, "edge_count": 0, "hits": 0,
                    "typed_edges": 0, "co_mention_edges": 0,
                    "papers": 0, "isolated_hits": 0,
                },
            }

        hit_set = set(hit_ids)
        # Stable score: FTS rank position (lower index = better).
        score_by_id = {cid: (len(hit_ids) - i) for i, cid in enumerate(hit_ids)}
        edges_out: list[dict] = []
        typed_count = 0
        co_mention_count = 0
        ph = ",".join("?" * len(hit_ids))

        # ── Typed edges where both endpoints are hits ──────────────────────
        if requested_types is None or any(t != "co_mention" for t in requested_types):
            typed_rows = conn.execute(
                f"""SELECT from_claim_id, edge_type, to_claim_id, to_doi,
                           confidence, evidence, extractor
                      FROM claim_edges
                     WHERE from_claim_id IN ({ph})
                       AND to_claim_id   IN ({ph})""",
                hit_ids + hit_ids,
            ).fetchall()
            for r in typed_rows:
                if requested_types is not None and r["edge_type"] not in requested_types:
                    continue
                edges_out.append({
                    "from": r["from_claim_id"],
                    "to": r["to_claim_id"],
                    "type": r["edge_type"],
                    "confidence": r["confidence"],
                    "evidence": r["evidence"],
                    "extractor": r["extractor"],
                })
                typed_count += 1
                if len(edges_out) >= max_edges:
                    break

        # ── Co-mention edges (same-paper hit pairs, star pattern) ──────────
        include_co_mention = (
            requested_types is None or "co_mention" in requested_types
        ) and len(edges_out) < max_edges
        if include_co_mention:
            paper_rows = conn.execute(
                f"""SELECT claim_id, source_doi
                      FROM claims
                     WHERE claim_id IN ({ph}) AND source_doi != ''""",
                hit_ids,
            ).fetchall()
            by_paper: dict[str, list[str]] = {}
            for r in paper_rows:
                by_paper.setdefault(r["source_doi"], []).append(r["claim_id"])
            # Iterate papers by descending hit-count so the densest clusters
            # always make it under the max_edges cap.
            for doi, claim_ids in sorted(by_paper.items(),
                                          key=lambda kv: -len(kv[1])):
                if len(claim_ids) < 2:
                    continue
                # Highest-scored hit becomes the star center; spokes ordered
                # by descending score so the visible spokes are the most
                # query-relevant ones when the cap clips the tail.
                claim_ids.sort(key=lambda cid: -score_by_id.get(cid, 0))
                center = claim_ids[0]
                spokes = claim_ids[1:1 + per_paper_cap]
                for spoke in spokes:
                    edges_out.append({
                        "from": center,
                        "to": spoke,
                        "type": "co_mention",
                        "confidence": "medium",
                        "evidence": doi,
                        "extractor": "same_paper",
                    })
                    co_mention_count += 1
                    if len(edges_out) >= max_edges:
                        break
                if len(edges_out) >= max_edges:
                    break

        # Render only nodes that participate in at least one emitted edge —
        # keeps payload focused on the structurally interesting subset.
        endpoint_ids: set[str] = set()
        for e in edges_out:
            endpoint_ids.add(e["from"])
            endpoint_ids.add(e["to"])

        node_summaries = _node_summary_rows(conn, list(endpoint_ids))

    nodes = []
    for nid in endpoint_ids:
        meta = node_summaries.get(nid, {"id": nid})
        meta["in_search"] = nid in hit_set
        meta["bridge"] = False
        meta["score"] = score_by_id.get(nid, 0)
        nodes.append(meta)
    nodes.sort(key=lambda n: (-n.get("in_search", False), -n.get("score", 0)))

    isolated_hits = [nid for nid in hit_ids if nid not in endpoint_ids]

    return {
        "query": q,
        "nodes": nodes,
        "edges": edges_out,
        "isolated_hits": isolated_hits,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges_out),
            "typed_edges": typed_count,
            "co_mention_edges": co_mention_count,
            "hits": len(hit_ids),
            "papers": len({n.get("paper") for n in nodes if n.get("paper")}),
            "isolated_hits": len(isolated_hits),
        },
    }


@app.get("/api/paper/validation/{doi:path}", summary="Paper validation status")
def api_paper_validation(doi: str):
    """Get the validation status and quality score for a paper."""
    val = db.get_paper_validation(doi)
    if not val:
        return {"doi": doi, "validated": False, "message": "No validation data available"}
    vdata = json.loads(val.get('validation_data', '{}')) if val.get('validation_data') else {}
    return {
        "doi": doi,
        "validated": True,
        "crossref_verified": bool(val.get('crossref_verified')),
        "is_retracted": bool(val.get('is_retracted')),
        "journal": val.get('journal', ''),
        "publisher": val.get('publisher', ''),
        "is_chemistry": bool(val.get('is_chemistry')),
        "validation_score": val.get('validation_score', 0),
        "validated_at": val.get('validated_at', ''),
        "issues": vdata.get('issues', []),
    }


# ── Bulk Data Access ─────────────────────────────────────────────────────────

@app.get("/api/export", summary="Bulk export claims")
def api_export(
    claim_type: Optional[str] = Query(None, description="Filter by claim type"),
    since: Optional[str] = Query(None, description="ISO date, e.g. 2025-01-01"),
    limit: int = Query(10000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """
    Bulk export claims as JSON for downstream consumers and AI training.

    Supports filtering by claim_type and date range. Paginate with limit/offset.
    """
    return db.export_claims(claim_type=claim_type, since=since, limit=limit, offset=offset)


@app.get("/api/changelog", summary="Recent changes")
def api_changelog(
    since: Optional[str] = Query(None, description="ISO date, e.g. 2025-03-01"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Get recently added or updated claims. Useful for tracking what's new in the index.
    """
    return db.get_changelog(since=since, limit=limit)


# ── Admin & Analytics ────────────────────────────────────────────────────────

@app.get("/api/admin/queries", summary="Query analytics", dependencies=[Depends(_require_admin)])
def api_query_stats(days: int = Query(30, ge=1, le=365)):
    """
    View search query analytics: top queries, daily counts, latency stats.
    Requires ADMIN_TOKEN authentication.
    """
    return db.get_query_stats(days=days)


@app.get("/api/admin/security-log", summary="Security and abuse log", dependencies=[Depends(_require_admin)])
def api_security_log(
    days: int = Query(7, ge=1, le=90),
    event_type: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """Recent security events (rate limits, failed admin auth, etc.). Requires ADMIN_TOKEN."""
    events = db.get_security_events(days=days, event_type=event_type, limit=limit)
    return {"events": events, "count": len(events), "period_days": days}


@app.post("/api/admin/keys", summary="Create API key", dependencies=[Depends(_require_admin)])
def api_create_key(name: str = Query(...), email: str = Query(""), tier: str = Query("tier_1")):
    """Create a new API key for programmatic access. Requires ADMIN_TOKEN authentication."""
    result = db.create_api_key(name=name, email=email, tier=tier)
    return result


class MyApiKeyRequest(BaseModel):
    name: str
    tier: str = "tier_1"  # reserved for future use; for now all self-serve keys are tier_1


_key_request_buckets: dict[str, list[float]] = {}
KEY_REQUEST_LIMIT = 5
KEY_REQUEST_WINDOW = 3600  # 1 hour
MAX_KEYS_PER_USER = 10


class KeyRequest(BaseModel):
    name: str
    email: str
    intended_use: str = ""


def _valid_email(e: str) -> bool:
    """Lightweight email sanity check (no external dependency)."""
    e = (e or "").strip()
    if not e or " " in e or e.count("@") != 1:
        return False
    local, _, domain = e.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")


@app.post("/api/keys/request", summary="Request an API key (self-service)")
def api_request_key(req: KeyRequest, request: Request):
    """Self-service API-key issuance for developers and AI agents.

    Anonymous and IP-rate-limited. Issues a tier_1 key (200 requests/min).
    Authenticate subsequent calls with ``Authorization: Bearer <api_key>``.
    The key is shown once in the response.
    """
    name = (req.name or "").strip()
    email = (req.email or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    if len(name) > 120:
        raise HTTPException(400, "name too long")
    if not _valid_email(email):
        raise HTTPException(400, "a valid email is required")

    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bkey = f"ip:{ip}"
    bucket = [t for t in _key_request_buckets.get(bkey, []) if now - t < KEY_REQUEST_WINDOW]
    if len(bucket) >= KEY_REQUEST_LIMIT:
        raise HTTPException(429, "Too many key requests from this address. Try again later.")
    bucket.append(now)
    _key_request_buckets[bkey] = bucket

    result = db.create_api_key(name=name, email=email, tier="tier_1")
    try:
        db.log_query(
            query=f"[api-key-request] {name} <{email}> use={(req.intended_use or '')[:120]}",
            endpoint="/api/keys/request", view=None, filters=None,
            result_count=1, latency_ms=0,
            user_agent=request.headers.get("user-agent", "")[:200],
            ip_hash=hashlib.sha256(ip.encode()).hexdigest()[:12],
        )
    except Exception:
        pass
    return {
        "key_id": result.get("key_id"),
        "api_key": result.get("api_key"),
        "name": result.get("name", name),
        "tier": result.get("tier", "tier_1"),
        "rate_limit": 200,
        "note": "Store this key now; it is shown only once.",
    }


@app.get("/api/me/api-keys", summary="List my API keys")
def api_me_list_keys(request: Request):
    user = _require_user(request)
    keys = db.list_user_api_keys(user["user_id"])
    return {"keys": keys, "count": len(keys)}


@app.post("/api/me/api-keys", summary="Create a new API key")
def api_me_create_key(req: MyApiKeyRequest, request: Request):
    """Create a tier_1 API key for the logged-in user. Key is shown once."""
    user = _require_user(request)
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    if len(name) > 80:
        raise HTTPException(400, "name too long")

    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _key_request_buckets.setdefault(user["user_id"], [])
    _key_request_buckets[user["user_id"]] = [t for t in bucket if now - t < KEY_REQUEST_WINDOW]
    if len(_key_request_buckets[user["user_id"]]) >= KEY_REQUEST_LIMIT:
        raise HTTPException(429, "Too many key requests. Try again later.")
    existing = db.list_user_api_keys(user["user_id"])
    active = [k for k in existing if k.get("is_active")]
    if len(active) >= MAX_KEYS_PER_USER:
        raise HTTPException(
            400, f"You already have {MAX_KEYS_PER_USER} active keys. Revoke one first."
        )
    _key_request_buckets[user["user_id"]].append(now)
    result = db.create_user_api_key(
        user_id=user["user_id"],
        name=name,
        email=(user.get("email") or ""),
        tier="tier_1",
    )
    db.log_query(
        query=f"[api-key-create] {name}",
        endpoint="/api/me/api-keys",
        view=None, filters=None, result_count=1, latency_ms=0,
        user_agent=request.headers.get("user-agent", "")[:200],
        ip_hash=hashlib.sha256(ip.encode()).hexdigest()[:12],
        user_id=user["user_id"],
    )
    return {
        "key_id": result["key_id"],
        "api_key": result["api_key"],
        "name": result["name"],
        "tier": result["tier"],
        "rate_limit": result["rate_limit"],
    }


@app.delete("/api/me/api-keys/{key_id}", summary="Revoke one of my API keys")
def api_me_revoke_key(key_id: str, request: Request):
    user = _require_user(request)
    try:
        ok = db.revoke_user_api_key(user["user_id"], key_id)
    except ValueError as e:
        raise HTTPException(403, str(e))
    if not ok:
        raise HTTPException(404, "API key not found")
    return {"status": "revoked", "key_id": key_id}


@app.post("/api/admin/notify", summary="Trigger subscription notifications", dependencies=[Depends(_require_admin)])
def api_admin_notify():
    """Manually trigger subscription notification check. Requires ADMIN_TOKEN authentication."""
    try:
        from .notify import check_subscriptions
        sent = check_subscriptions()
        return {"status": "ok", "notifications_sent": sent}
    except Exception as e:
        raise HTTPException(500, f"Notification check failed: {e}")


# ── Auth: Clerk JWT verification ───────────────────────────────────────────

import base64 as _b64
import jwt as _pyjwt
import urllib.request as _urllib_req

_CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY", "")
_CLERK_PUBLISHABLE_KEY = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
_CLERK_JWKS: dict | None = None
_CLERK_JWKS_TS: float = 0


def _clerk_frontend_api() -> str:
    """Derive the Clerk Frontend API host from the publishable key.

    A key like 'pk_test_<base64>' decodes to '<host>$' where host is e.g.
    'suitable-liger-1.clerk.accounts.dev'.
    """
    pk = _CLERK_PUBLISHABLE_KEY
    if not pk:
        raise RuntimeError("CLERK_PUBLISHABLE_KEY not set")
    if pk.startswith("pk_test_"):
        b64 = pk[len("pk_test_"):]
    elif pk.startswith("pk_live_"):
        b64 = pk[len("pk_live_"):]
    else:
        raise RuntimeError(f"Unrecognized CLERK_PUBLISHABLE_KEY format")
    pad = "=" * (-len(b64) % 4)
    decoded = _b64.b64decode(b64 + pad).decode("utf-8").rstrip("$")
    return decoded


def _get_clerk_jwks() -> dict:
    """Fetch and cache Clerk's public JWKS (refreshed every 6 hours)."""
    global _CLERK_JWKS, _CLERK_JWKS_TS
    if _CLERK_JWKS and (time.time() - _CLERK_JWKS_TS < 21600):
        return _CLERK_JWKS

    host = _clerk_frontend_api()
    url = f"https://{host}/.well-known/jwks.json"
    req = _urllib_req.Request(url, headers={"User-Agent": "askchem-server/1.0"})
    with _urllib_req.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    _CLERK_JWKS = data
    _CLERK_JWKS_TS = time.time()
    return _CLERK_JWKS


def _decode_clerk_token(token: str) -> dict | None:
    """Decode and verify a Clerk session JWT. Returns claims dict or None."""
    try:
        jwks_data = _get_clerk_jwks()
        from jwt.algorithms import RSAAlgorithm
        header = _pyjwt.get_unverified_header(token)
        kid = header.get("kid")
        key_data = None
        for k in jwks_data.get("keys", []):
            if k.get("kid") == kid:
                key_data = k
                break
        if not key_data:
            return None
        public_key = RSAAlgorithm.from_jwk(key_data)
        claims = _pyjwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return claims
    except Exception as e:
        print(f"Clerk JWT decode error: {e}", flush=True)
        return None


def _get_current_user(request: Request) -> dict | None:
    """Extract Clerk JWT from Authorization header and resolve local user."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth.startswith("Bearer ac-"):
        return None
    token = auth[7:]
    if len(token) < 20:
        return None

    claims = _decode_clerk_token(token)
    if not claims:
        return None

    clerk_id = claims.get("sub", "")
    email = ""
    if "email" in claims:
        email = claims["email"]
    elif "email_addresses" in claims:
        addrs = claims["email_addresses"]
        if addrs:
            email = addrs[0] if isinstance(addrs[0], str) else addrs[0].get("email_address", "")

    if not clerk_id:
        return None

    user = db.get_or_create_clerk_user(clerk_id, email)
    return user


def _require_user(request: Request) -> dict:
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Authentication required")
    return user


# ── Feedback ───────────────────────────────────────────────────────────────

VALID_FEEDBACK_TARGETS = {"claim", "paper", "search"}
VALID_FEEDBACK_TYPES = {"upvote", "downvote", "comment"}


class FeedbackRequest(BaseModel):
    target_type: str
    target_id: str
    feedback_type: str
    comment: str = ""


@app.post("/api/feedback", summary="Submit feedback")
def api_submit_feedback(req: FeedbackRequest, request: Request):
    user = _require_user(request)
    if req.target_type not in VALID_FEEDBACK_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_FEEDBACK_TARGETS}")
    if req.feedback_type not in VALID_FEEDBACK_TYPES:
        raise HTTPException(400, f"feedback_type must be one of: {VALID_FEEDBACK_TYPES}")
    if req.feedback_type == "comment" and not req.comment.strip():
        raise HTTPException(400, "comment cannot be empty for feedback_type=comment")
    if len(req.comment) > 2000:
        raise HTTPException(400, "Comment must be under 2000 characters")

    result = db.upsert_feedback(
        user_id=user["user_id"],
        target_type=req.target_type,
        target_id=req.target_id,
        feedback_type=req.feedback_type,
        comment=req.comment,
    )
    return result


@app.get("/api/feedback/summary", summary="Get feedback summary for a target")
def api_feedback_summary(
    request: Request,
    target_type: str = Query(...),
    target_id: str = Query(...),
):
    if target_type not in VALID_FEEDBACK_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_FEEDBACK_TARGETS}")
    user = _get_current_user(request)
    user_id = user["user_id"] if user else None
    return db.get_feedback_summary(target_type, target_id, user_id=user_id)


@app.get("/api/feedback/batch", summary="Get feedback for multiple targets")
def api_feedback_batch(
    request: Request,
    target_type: str = Query(...),
    target_ids: str = Query(..., description="Comma-separated target IDs"),
):
    if target_type not in VALID_FEEDBACK_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_FEEDBACK_TARGETS}")
    ids = [t.strip() for t in target_ids.split(",") if t.strip()][:100]
    user = _get_current_user(request)
    user_id = user["user_id"] if user else None
    return db.get_feedback_batch(target_type, ids, user_id=user_id)


@app.get("/api/feedback/mine", summary="Get my feedback history")
def api_my_feedback(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    user = _require_user(request)
    items = db.get_user_feedback(user["user_id"], limit=limit, offset=offset)
    return {"feedback": items}


# ── User query history & click tracking ────────────────────────────────────

VALID_CLICK_TARGETS = {"claim", "paper", "external_link"}


class ClickEvent(BaseModel):
    target_type: str
    target_id: str
    query: Optional[str] = None
    query_log_id: Optional[int] = None
    position: Optional[int] = None


@app.get("/api/queries/mine", summary="Get my recent search queries")
def api_my_queries(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    user = _require_user(request)
    items = db.get_user_query_history(user["user_id"], limit=limit, offset=offset)
    return {"queries": items, "count": len(items)}


@app.post("/api/clicks", summary="Record a click on a search result")
def api_log_click(req: ClickEvent, request: Request):
    if req.target_type not in VALID_CLICK_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_CLICK_TARGETS}")
    if not req.target_id or len(req.target_id) > 500:
        raise HTTPException(400, "target_id is required and must be under 500 chars")

    current_user = _get_current_user(request)
    ip_raw = request.client.host if request.client else ""
    db.log_click(
        target_type=req.target_type,
        target_id=req.target_id,
        query=(req.query or "")[:500] or None,
        query_log_id=req.query_log_id,
        position=req.position,
        user_id=current_user["user_id"] if current_user else None,
        ip_hash=hashlib.sha256(ip_raw.encode()).hexdigest()[:12],
    )
    return {"status": "ok"}


# ── Bookmarks ─────────────────────────────────────────────────────────────

VALID_BOOKMARK_TARGETS = {"claim", "paper", "author", "topic", "search"}


class BookmarkRequest(BaseModel):
    target_type: str
    target_id: str
    title: Optional[str] = None
    note: Optional[str] = None


@app.get("/api/me/bookmarks", summary="List my bookmarks")
def api_me_bookmarks(
    request: Request,
    target_type: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    user = _require_user(request)
    if target_type and target_type not in VALID_BOOKMARK_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_BOOKMARK_TARGETS}")
    items = db.list_bookmarks(user["user_id"], target_type=target_type, limit=limit)
    return {"bookmarks": items, "count": len(items)}


@app.post("/api/me/bookmarks", summary="Add a bookmark")
def api_me_bookmark_add(req: BookmarkRequest, request: Request):
    user = _require_user(request)
    if req.target_type not in VALID_BOOKMARK_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_BOOKMARK_TARGETS}")
    if not req.target_id or len(req.target_id) > 500:
        raise HTTPException(400, "target_id is required and must be under 500 chars")
    out = db.add_bookmark(
        user_id=user["user_id"],
        target_type=req.target_type,
        target_id=req.target_id,
        title=(req.title or "")[:500] or None,
        note=(req.note or "")[:2000] or None,
    )
    return {"status": "ok", "id": out["id"], "bookmarked": True}


@app.delete("/api/me/bookmarks", summary="Remove a bookmark by target")
def api_me_bookmark_remove(
    request: Request,
    target_type: str = Query(...),
    target_id: str = Query(...),
):
    user = _require_user(request)
    ok = db.remove_bookmark(user["user_id"], target_type, target_id)
    return {"status": "ok" if ok else "not_found", "bookmarked": False}


@app.get("/api/me/bookmarks/status", summary="Check if I have bookmarked a target")
def api_me_bookmark_status(
    request: Request,
    target_type: str = Query(...),
    target_id: str = Query(...),
):
    user = _require_user(request)
    return {"bookmarked": db.is_bookmarked(user["user_id"], target_type, target_id)}


# ── Saved searches ────────────────────────────────────────────────────────

class SavedSearchRequest(BaseModel):
    query: str
    view: Optional[str] = None
    filters: Optional[dict] = None
    name: Optional[str] = None


@app.get("/api/me/saved-searches", summary="List my saved searches")
def api_me_saved_searches(request: Request, limit: int = Query(200, ge=1, le=1000)):
    user = _require_user(request)
    items = db.list_saved_searches(user["user_id"], limit=limit)
    return {"saved_searches": items, "count": len(items)}


@app.post("/api/me/saved-searches", summary="Save a search")
def api_me_save_search(req: SavedSearchRequest, request: Request):
    user = _require_user(request)
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(400, "query is required")
    if len(q) > 500:
        raise HTTPException(400, "query too long")
    out = db.add_saved_search(
        user_id=user["user_id"],
        query=q,
        view=(req.view or None),
        filters=(req.filters or None),
        name=((req.name or "").strip()[:120] or None),
    )
    return {"status": "ok", "id": out["id"]}


@app.delete("/api/me/saved-searches/{saved_id}", summary="Delete a saved search")
def api_me_saved_search_delete(saved_id: int, request: Request):
    user = _require_user(request)
    ok = db.delete_saved_search(user["user_id"], saved_id)
    if not ok:
        raise HTTPException(404, "Saved search not found")
    return {"status": "deleted", "id": saved_id}


# ── Reading lists ─────────────────────────────────────────────────────────

VALID_READING_LIST_ITEM_TARGETS = {"paper", "claim", "author", "topic"}


class ReadingListCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False


class ReadingListUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None


class ReadingListItemRequest(BaseModel):
    target_type: str
    target_id: str
    title: Optional[str] = None
    note: Optional[str] = None


@app.get("/api/me/reading-lists", summary="List my reading lists")
def api_me_reading_lists(request: Request):
    user = _require_user(request)
    lists = db.list_reading_lists(user["user_id"])
    return {"reading_lists": lists, "count": len(lists)}


@app.post("/api/me/reading-lists", summary="Create a reading list")
def api_me_reading_list_create(req: ReadingListCreate, request: Request):
    user = _require_user(request)
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    if len(name) > 120:
        raise HTTPException(400, "name too long")
    description = (req.description or "").strip()[:1000] or None
    out = db.create_reading_list(
        user_id=user["user_id"],
        name=name,
        description=description,
        is_public=bool(req.is_public),
    )
    return {"status": "ok", **out}


@app.get("/api/me/reading-lists/containing", summary="Lists that contain a target")
def api_me_reading_lists_containing(
    request: Request,
    target_type: str = Query(...),
    target_id: str = Query(...),
):
    """Returns IDs of *my* reading lists that already include this target."""
    user = _require_user(request)
    if target_type not in VALID_READING_LIST_ITEM_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_READING_LIST_ITEM_TARGETS}")
    ids = db.get_lists_containing(user["user_id"], target_type, target_id)
    return {"list_ids": ids}


@app.get("/api/reading-lists/{list_id}", summary="View a reading list (public or owned)")
def api_reading_list_get(list_id: int, request: Request):
    """
    Returns a reading list with its items.
    Public lists are visible to everyone; private lists only to their owner.
    """
    user = _get_current_user(request)
    uid = user["user_id"] if user else None
    lst = db.get_user_reading_list(list_id, user_id=uid)
    if not lst:
        raise HTTPException(404, "Reading list not found")
    return lst


@app.patch("/api/me/reading-lists/{list_id}", summary="Rename/update a reading list")
def api_me_reading_list_update(list_id: int, req: ReadingListUpdate, request: Request):
    user = _require_user(request)
    name = None
    if req.name is not None:
        name = req.name.strip()
        if not name:
            raise HTTPException(400, "name cannot be empty")
        if len(name) > 120:
            raise HTTPException(400, "name too long")
    description = None
    if req.description is not None:
        description = req.description.strip()[:1000] or None
    ok = db.update_reading_list(
        list_id=list_id,
        user_id=user["user_id"],
        name=name,
        description=description,
        is_public=req.is_public,
    )
    if not ok:
        raise HTTPException(404, "Reading list not found")
    return {"status": "ok", "id": list_id}


@app.delete("/api/me/reading-lists/{list_id}", summary="Delete a reading list")
def api_me_reading_list_delete(list_id: int, request: Request):
    user = _require_user(request)
    ok = db.delete_reading_list(list_id, user["user_id"])
    if not ok:
        raise HTTPException(404, "Reading list not found")
    return {"status": "deleted", "id": list_id}


@app.post("/api/me/reading-lists/{list_id}/items", summary="Add an item to a list")
def api_me_reading_list_item_add(
    list_id: int, req: ReadingListItemRequest, request: Request,
):
    user = _require_user(request)
    if req.target_type not in VALID_READING_LIST_ITEM_TARGETS:
        raise HTTPException(400, f"target_type must be one of: {VALID_READING_LIST_ITEM_TARGETS}")
    if not req.target_id or len(req.target_id) > 500:
        raise HTTPException(400, "target_id is required and must be under 500 chars")
    try:
        out = db.add_reading_list_item(
            list_id=list_id,
            user_id=user["user_id"],
            target_type=req.target_type,
            target_id=req.target_id,
            title=(req.title or "")[:500] or None,
            note=(req.note or "")[:2000] or None,
        )
    except ValueError as e:
        raise HTTPException(404 if "not found" in str(e) else 403, str(e))
    return {"status": "ok", **out}


@app.delete("/api/me/reading-lists/{list_id}/items", summary="Remove an item by target")
def api_me_reading_list_item_remove(
    list_id: int,
    request: Request,
    target_type: str = Query(...),
    target_id: str = Query(...),
):
    user = _require_user(request)
    try:
        ok = db.remove_reading_list_item(
            list_id=list_id,
            user_id=user["user_id"],
            target_type=target_type,
            target_id=target_id,
        )
    except ValueError as e:
        raise HTTPException(404 if "not found" in str(e) else 403, str(e))
    return {"status": "ok" if ok else "not_found"}


# ── Versioned API (v1) ──────────────────────────────────────────────────────
# Mirrors /api/* endpoints under /v1/* for SDK stability guarantees.
# Rate limiting is handled globally by RateLimitMiddleware.

@app.get("/v1/search", summary="[v1] Search claims")
def v1_search(
    request: Request,
    q: str = Query(..., description="Search query"),
    claim_type: Optional[str] = Query(None),
    view: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return api_search(
        request=request,
        q=q,
        claim_type=claim_type,
        view=view,
        limit=limit,
        offset=offset,
        mode="auto",
        sort="relevance",
    )


@app.get("/v1/claims/{claim_id}", summary="[v1] Get claim")
def v1_get_claim(request: Request, claim_id: str):
    return api_get_claim(claim_id)


@app.post("/v1/claims/bulk", summary="[v1] Bulk claims")
def v1_bulk_claims(request: Request, req: BulkClaimsRequest):
    return api_bulk_claims(req)


@app.get("/v1/views", summary="[v1] List views")
def v1_list_views(request: Request):
    return api_list_views()


@app.get("/v1/tree/{view_id}", summary="[v1] Browse tree root")
def v1_tree_root(request: Request, view_id: str, depth: int = Query(1, ge=0, le=3)):
    return api_tree_root(view_id, depth)


@app.get("/v1/tree/{view_id}/{path:path}", summary="[v1] Browse tree node")
def v1_tree_node(request: Request, view_id: str, path: str,
                 depth: int = Query(1, ge=0, le=3),
                 limit: int = Query(50, ge=1, le=200),
                 offset: int = Query(0, ge=0)):
    return api_tree_node(view_id, path, depth, limit, offset)


@app.get("/v1/sources/{doi:path}", summary="[v1] Paper claims")
def v1_source_claims(request: Request, doi: str):
    return api_source_claims(doi)


@app.get("/v1/stats", summary="[v1] Index stats")
def v1_stats(request: Request):
    return api_stats()


@app.post("/v1/submit", summary="[v1] Submit paper")
async def v1_submit(request: Request, req: SubmitPaperRequest, background_tasks: BackgroundTasks):
    return await api_submit_paper(req, background_tasks)


@app.get("/v1/time", summary="[v1] Browse by time period")
def v1_time(request: Request, decade: Optional[str] = None, year: Optional[int] = None, quarter: Optional[str] = None):
    return api_time_browse(decade=decade, year=year, quarter=quarter)


@app.get("/v1/temporal/{view_id}/{path:path}", summary="[v1] Temporal overlay")
def v1_temporal(request: Request, view_id: str, path: str):
    return api_temporal_overlay(view_id, path)


@app.get("/v1/evolution/{view_id}/{path:path}", summary="[v1] Evolution timeline")
def v1_evolution(request: Request, view_id: str, path: str):
    return api_evolution(view_id, path)


@app.get("/v1/authors", summary="[v1] Authors")
def v1_authors(request: Request, q: Optional[str] = None, topic: Optional[str] = None,
               view: Optional[str] = None, path: Optional[str] = None, limit: int = 20):
    return api_authors(q=q, topic=topic, view=view, path=path, limit=limit)


@app.get("/v1/authors/{author_id}/network", summary="[v1] Author network")
def v1_author_network(request: Request, author_id: str, depth: int = 1, limit: int = 30):
    return api_author_network(author_id, depth=depth, limit=limit)


@app.get("/v1/authors/{author_id}", summary="[v1] Author profile")
def v1_author_profile(request: Request, author_id: str):
    return api_author_profile(author_id)


@app.get("/v1/feed", summary="[v1] Discoveries feed")
def v1_feed(request: Request, limit: int = 20, days: int = 7):
    return api_feed(limit=limit, days=days)


@app.get("/v1/me/subscriptions/{sub_id}/history", summary="[v1] Subscription history (auth)")
def v1_subscription_history(request: Request, sub_id: int, limit: int = Query(20, ge=1, le=100)):
    return api_me_subscription_history(sub_id, request=request, limit=limit)


@app.get("/v1/submissions/{submission_id}/stream", summary="[v1] Submission SSE stream")
async def v1_submission_stream(request: Request, submission_id: int):
    return await api_submission_stream(submission_id)


# ── Agent-friendly root ──────────────────────────────────────────────────────

@app.get("/api", summary="API overview and quick-start for agents")
def api_root():
    """
    AskChem API root. Returns index overview and endpoint guide.

    **Agent usage:** Call this first to understand the index and available endpoints.
    """
    stats = db.get_stats()
    return {
        "name": "AskChem",
        "description": "A hierarchical, multi-view, source-grounded index of chemical knowledge",
        "version": "1.0.0",
        "stats": {
            "total_claims": int(stats.get('total_claims', 0)),
            "total_sources": int(stats.get('total_sources', 0)),
            "total_views": int(stats.get('total_views', 5)),
        },
        "endpoints": {
            "search": "GET /api/search?q=your+query&view=by_reaction_type",
            "browse_views": "GET /api/views",
            "browse_tree": "GET /api/tree/{view_id}?depth=2",
            "zoom_in": "GET /api/tree/{view_id}/{path}?depth=1&limit=50&offset=0",
            "get_claim": "GET /api/claims/{claim_id}",
            "bulk_claims": "POST /api/claims/bulk {claim_ids: [...]}",
            "paper_claims": "GET /api/sources/{doi}",
            "time_browse": "GET /api/time",
            "temporal_overlay": "GET /api/temporal/{view_id}/{path}",
            "evolution": "GET /api/evolution/{view_id}/{path}",
            "reading_list": "GET /api/reading-list/{view_id}/{path}",
            "feed": "GET /api/feed?limit=20&days=7",
            "authors": "GET /api/authors?q=name or ?topic=query",
            "author_profile": "GET /api/authors/{author_id}",
            "author_network": "GET /api/authors/{author_id}/network",
            "submit_paper": "POST /api/submit {doi, name, email, callback_url}",
            "submission_stream": "GET /api/submissions/{id}/stream (SSE)",
            "subscribe": "POST /api/me/subscriptions (auth) {sub_type, target, frequency}",
            "my_bookmarks": "GET /api/me/bookmarks (auth)",
            "my_saved_searches": "GET /api/me/saved-searches (auth)",
            "my_reading_lists": "GET /api/me/reading-lists (auth)",
            "public_reading_list": "GET /api/reading-lists/{id}",
            "my_api_keys": "GET /api/me/api-keys (auth)",
            "versioned_api": "All endpoints also available under /v1/",
            "interactive_docs": "GET /api/docs",
        },
    }


# ── Background processing ────────────────────────────────────────────────────

def _validate_paper(doi: str) -> dict:
    """Validate a paper via CrossRef and Semantic Scholar.

    Returns a validation dict with scores and flags.
    Checks: DOI format, CrossRef registration, retraction status,
    journal existence, and whether the paper is chemistry-related.
    """
    import re
    import requests as http_requests

    val = {
        'doi_format_valid': False,
        'crossref_verified': False,
        'has_abstract': False,
        'is_retracted': False,
        'journal': '',
        'publisher': '',
        'is_chemistry': True,
        'validation_score': 0.0,
        'issues': [],
    }

    if not re.match(r'^10\.\d{4,}/', doi):
        val['issues'].append('Invalid DOI format')
        return val
    val['doi_format_valid'] = True

    try:
        cr_resp = http_requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={'User-Agent': 'AskChem/1.0 (mailto:admin@askchem.org)'},
            timeout=15,
        )
        if cr_resp.status_code == 200:
            cr_data = cr_resp.json().get('message', {})
            val['crossref_verified'] = True
            val['journal'] = (cr_data.get('container-title') or [''])[0]
            val['publisher'] = cr_data.get('publisher', '')

            update_policy = cr_data.get('update-to') or []
            for upd in update_policy:
                if upd.get('type') == 'retraction':
                    val['is_retracted'] = True
                    val['issues'].append('Paper has been RETRACTED')

            if cr_data.get('update-policy'):
                pass

            subjects = cr_data.get('subject') or []
            title_str = ' '.join(cr_data.get('title') or []).lower()
            chem_keywords = {'chemistry', 'chemical', 'catalysis', 'polymer',
                             'materials', 'electrochemistry', 'biochemistry',
                             'organic', 'inorganic', 'analytical', 'physical chemistry',
                             'pharmaceutical', 'medicinal', 'nanoscience', 'nanotechnology'}
            subjects_lower = ' '.join(subjects).lower()
            if subjects and not any(k in subjects_lower for k in chem_keywords):
                if not any(k in title_str for k in chem_keywords):
                    val['is_chemistry'] = False
                    val['issues'].append(
                        f'Paper may not be chemistry-related (subjects: {subjects})')
        else:
            val['issues'].append(f'CrossRef lookup failed (HTTP {cr_resp.status_code})')
    except Exception as e:
        val['issues'].append(f'CrossRef check error: {str(e)[:100]}')

    score = 0.0
    if val['doi_format_valid']:
        score += 0.2
    if val['crossref_verified']:
        score += 0.3
    if val['is_chemistry']:
        score += 0.3
    if not val['is_retracted']:
        score += 0.2
    val['validation_score'] = round(score, 2)

    return val


async def process_submission(submission_id: int, doi: str, callback_url: str = None):
    """Background task: validate paper, fetch metadata, extract claims, classify, index."""
    from askchem.taxonomy import (
        build_full_classification_messages,
        normalize_path,
        build_claim_type_path, ALL_CONTENT_VIEWS,
    )

    try:
        db.update_submission(submission_id, 'processing')

        # Step 0: Validate the paper
        validation = _validate_paper(doi)
        db.save_paper_validation(doi, validation)

        if not validation['doi_format_valid']:
            db.update_submission(submission_id, 'failed', {
                'error': 'Invalid DOI format',
                'validation': validation,
            })
            return

        if validation['is_retracted']:
            db.update_submission(submission_id, 'failed', {
                'error': 'This paper has been retracted and cannot be indexed',
                'validation': validation,
            })
            return

        if validation['validation_score'] < 0.3:
            db.update_submission(submission_id, 'failed', {
                'error': f'Paper did not pass validation (score: {validation["validation_score"]})',
                'validation': validation,
            })
            return

        # Step 1: Fetch metadata from Semantic Scholar
        import requests as http_requests
        s2_key = os.environ.get('S2_API_KEY', '')
        headers = {'x-api-key': s2_key} if s2_key else {}
        resp = http_requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={'fields': 'paperId,title,abstract,year,citationCount,venue,authors,externalIds'},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            db.update_submission(submission_id, 'failed', {'error': f'Semantic Scholar API returned {resp.status_code}'})
            return

        paper = resp.json()
        abstract = paper.get('abstract', '')
        if not abstract:
            db.update_submission(submission_id, 'failed', {'error': 'Paper has no abstract available'})
            return

        validation['has_abstract'] = True

        source_data = {
            'doi': doi,
            'title': paper.get('title', ''),
            'authors': [a.get('name', '') for a in paper.get('authors', [])],
            'year': paper.get('year', 0),
            'venue': paper.get('venue', ''),
            'abstract': abstract,
            'citation_count': paper.get('citationCount', 0),
        }
        db.insert_source(source_data)

        # Step 2: Extract claims using LLM
        from openai import OpenAI
        client = OpenAI()
        extraction_prompt = f"""Extract all distinct scientific claims from this chemistry paper abstract.
For each claim, provide:
- claim_type: one of [reaction, property, method, mechanism, comparison, computational_result, hypothesis, conclusion, limitation, future_direction, surprising_finding]
- verbatim_quote: exact text from the abstract supporting this claim
- confidence: high, medium, or low
- Type-specific fields as appropriate

Paper: {paper.get('title', '')}
Abstract: {abstract}

Return JSON: {{"claims": [...]}}"""

        extract_resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=4096,
        )
        claims_raw = json.loads(extract_resp.choices[0].message.content)
        claims_list = claims_raw.get('claims', [])

        # Step 3: Classify each claim into views using canonical taxonomy
        import hashlib
        indexed_claims = []
        for i, raw_claim in enumerate(claims_list):
            content_hash = hashlib.sha256(json.dumps(raw_claim, sort_keys=True).encode()).hexdigest()[:16]
            claim_id = hashlib.sha256(f"{doi}:{raw_claim.get('claim_type', '')}:{content_hash}".encode()).hexdigest()[:16]
            claim_type = raw_claim.get('claim_type', '')

            classify_resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=build_full_classification_messages(
                    claim_type=claim_type,
                    quote=raw_claim.get('verbatim_quote', ''),
                    title=paper.get('title', ''),
                ),
                response_format={"type": "json_object"},
                max_completion_tokens=2048,
            )
            raw_paths = json.loads(classify_resp.choices[0].message.content)

            view_paths = {}
            for view_id in ALL_CONTENT_VIEWS:
                normalized = normalize_path(view_id, raw_paths.get(view_id, []))
                if normalized:
                    view_paths[view_id] = normalized

            ct_path = build_claim_type_path(claim_type)
            view_paths['by_claim_type'] = ct_path

            claim_data = {
                'claim_id': claim_id,
                'claim_type': claim_type,
                'source_doi': doi,
                'source_paper_title': paper.get('title', ''),
                'confidence': raw_claim.get('confidence', 'medium'),
                'location_in_paper': 'abstract',
                'verbatim_quote': raw_claim.get('verbatim_quote', ''),
                'extraction_model': 'gpt-5-mini',
                'extraction_version': 'v4-normalized',
                'extracted_at': datetime.utcnow().isoformat(),
                'view_paths': view_paths,
                **{k: v for k, v in raw_claim.items() if k not in ('claim_type', 'verbatim_quote', 'confidence')},
            }
            db.insert_claim(claim_data)
            indexed_claims.append(claim_data)

        # Step 4: Update tree nodes for all views
        for claim_data in indexed_claims:
            vp = claim_data.get('view_paths', {})
            for view_id, path in vp.items():
                if path and isinstance(path, list):
                    for depth in range(len(path)):
                        partial_path = '/'.join(path[:depth + 1])
                        db.append_claim_to_node(view_id, partial_path, claim_data['claim_id'])

        # Step 5: Update by_paper tree node
        claim_ids = [c['claim_id'] for c in indexed_claims]
        if claim_ids:
            doi_path = doi.replace('/', '__')
            db.upsert_tree_node(
                'by_paper', doi_path,
                name=paper.get('title', doi),
                level=1,
                claim_ids=claim_ids,
                data={
                    'view_id': 'by_paper', 'path': doi_path,
                    'name': paper.get('title', doi), 'level': 1,
                    'claim_count': len(claim_ids), 'children': [],
                    'claim_ids': claim_ids, 'doi': doi,
                },
            )

        # Step 6: Index authors from OpenAlex
        db.index_authors_for_doi(doi)

        result_data = {
            'claims_count': len(indexed_claims),
            'paper_title': paper.get('title', ''),
            'claim_ids': claim_ids,
            'validation': {
                'score': validation.get('validation_score', 0),
                'crossref_verified': validation.get('crossref_verified', False),
                'journal': validation.get('journal', ''),
                'issues': validation.get('issues', []),
            },
        }
        db.update_submission(submission_id, 'completed', result_data)

        if callback_url:
            try:
                import requests as cb_requests
                cb_requests.post(callback_url, json={
                    "submission_id": submission_id, "doi": doi,
                    "status": "completed", "result": result_data,
                }, timeout=10)
            except Exception:
                pass

    except Exception as e:
        db.update_submission(submission_id, 'failed', {'error': str(e)})
        if callback_url:
            try:
                import requests as cb_requests
                cb_requests.post(callback_url, json={
                    "submission_id": submission_id, "doi": doi,
                    "status": "failed", "error": str(e),
                }, timeout=10)
            except Exception:
                pass


# ── Static frontend serving ──────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent.parent / "web"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>AskChem</h1><p>Frontend not found. API available at <a href='/api/docs'>/api/docs</a></p>")


# Serve static assets (CSS, JS, images)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
