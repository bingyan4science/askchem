"""
AskChem SQLite database layer.

Provides fast full-text search, filtering, and tree browsing over 800K+ claims.
Replaces the filesystem-based store for production web serving while keeping
the filesystem store as the canonical source of truth for batch operations.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import os
import sys
import time as _time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from askchem.display import smart_title
from askchem.runtime_db import get_runtime_conn as _open_runtime_conn

# Module-level caches
_stats_cache: dict | None = None
_stats_cache_time: float = 0
_STATS_TTL = 300  # 5 minutes

_source_cache: dict[str, dict] = {}
_SOURCE_CACHE_MAX = 2000

_fallback_author_cache: dict | None = None
_fallback_author_cache_time: float = 0
_FALLBACK_AUTHOR_TTL = 900  # 15 minutes

# search_claims result LRU (opt-in via CHEMTREE_SEARCH_CACHE=1).
# Added for the May-15 ablation: hot queries (Suzuki / CO2 / perovskite)
# pay the full 4-9 s hybrid pipeline on every request because nothing
# upstream of vector_search caches the full result set. This LRU keys
# by (query, claim_type, view, limit, offset, use_semantic) and stores
# the raw response dict; callers (api_search) attach `intent` /
# `query_log_id` after fetch, so we return a shallow-copied outer dict
# to avoid poisoning the cached value.
import threading as _threading
_SEARCH_CLAIMS_CACHE: dict = {}
_SEARCH_CLAIMS_CACHE_ORDER: list = []
_SEARCH_CLAIMS_CACHE_LOCK = _threading.Lock()


def _env_enabled(name: str) -> bool:
    """Return whether a CHEMTREE feature flag is explicitly enabled."""
    return os.environ.get(name, "0").strip() == "1"


def _search_experiment_config() -> tuple:
    """Cache-key-safe snapshot of all ranking-affecting experiment knobs."""
    flag_names = (
        "CHEMTREE_DISABLE_TREE_RECALL",
        "CHEMTREE_DISABLE_AUTHOR_RECALL",
        "CHEMTREE_DISABLE_SOURCE_PAPER_RECALL",
        "CHEMTREE_DISABLE_CLAIM_GUIDED_PAPER_RECALL",
        "CHEMTREE_DISABLE_FTS",
        "CHEMTREE_DISABLE_DENSE",
        "CHEMTREE_DISABLE_CITATION_BOOST",
        "CHEMTREE_DISABLE_RERANK",
    )
    flags = tuple((name, _env_enabled(name)) for name in flag_names)
    return flags + (
        ("CHEMTREE_MAX_QUERY_VARIANTS",
         os.environ.get("CHEMTREE_MAX_QUERY_VARIANTS", "").strip()),
    )


def _search_cache_enabled() -> bool:
    return os.environ.get("CHEMTREE_SEARCH_CACHE", "0").strip() == "1"


def _search_cache_size() -> int:
    try:
        return max(
            1, int(os.environ.get("CHEMTREE_SEARCH_CACHE_SIZE", "512") or "512")
        )
    except ValueError:
        return 512


def _search_cache_ttl_s() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get("CHEMTREE_SEARCH_CACHE_TTL_S", "300") or "300"),
        )
    except ValueError:
        return 300.0


def _search_cache_get(key):
    if not _search_cache_enabled():
        return None
    ttl = _search_cache_ttl_s()
    with _SEARCH_CLAIMS_CACHE_LOCK:
        rec = _SEARCH_CLAIMS_CACHE.get(key)
        if rec is None:
            return None
        ts, value = rec
        if ttl > 0 and (_time.time() - ts) > ttl:
            _SEARCH_CLAIMS_CACHE.pop(key, None)
            try:
                _SEARCH_CLAIMS_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            return None
        try:
            _SEARCH_CLAIMS_CACHE_ORDER.remove(key)
        except ValueError:
            pass
        _SEARCH_CLAIMS_CACHE_ORDER.append(key)
        return value


def _search_cache_put(key, value) -> None:
    if not _search_cache_enabled():
        return
    cap = _search_cache_size()
    with _SEARCH_CLAIMS_CACHE_LOCK:
        if key in _SEARCH_CLAIMS_CACHE:
            try:
                _SEARCH_CLAIMS_CACHE_ORDER.remove(key)
            except ValueError:
                pass
        _SEARCH_CLAIMS_CACHE[key] = (_time.time(), value)
        _SEARCH_CLAIMS_CACHE_ORDER.append(key)
        while len(_SEARCH_CLAIMS_CACHE_ORDER) > cap:
            ek = _SEARCH_CLAIMS_CACHE_ORDER.pop(0)
            _SEARCH_CLAIMS_CACHE.pop(ek, None)


_REPO_ROOT = Path(__file__).parent.parent.parent


def _resolve_db_path() -> Path:
    """Resolve the SQLite database path.

    The database file was renamed ``chemtree.db`` -> ``askchem.db`` for
    consistency with the ``askchem`` package / ``bing-yan/askchem`` HF dataset.
    Resolution order:
      1. Explicit override: ``ASKCHEM_DB`` (preferred) or legacy ``CHEMTREE_DB``.
      2. The canonical ``askchem.db`` if it exists.
      3. The legacy ``chemtree.db`` if it exists (smooth transition on hosts not
         yet renamed).
      4. Canonical ``askchem.db`` as the default for fresh installs.
    """
    env = os.environ.get("ASKCHEM_DB") or os.environ.get("CHEMTREE_DB")
    if env:
        return Path(env)
    canonical = _REPO_ROOT / "askchem.db"
    legacy = _REPO_ROOT / "chemtree.db"
    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    return canonical


# Back-compat module constant (some callers import db.DB_PATH directly).
DB_PATH = _resolve_db_path()


def get_db_path() -> Path:
    return _resolve_db_path()


@contextmanager
def get_conn(readonly=True):
    path = get_db_path()
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA mmap_size=268435456")    # 256MB memory-mapped I/O
    conn.execute("PRAGMA cache_size=-32768")      # 32MB page cache
    try:
        yield conn
    finally:
        conn.close()


def get_runtime_conn(readonly=True):
    """Open private mutable state with the corpus attached read-only."""
    return _open_runtime_conn(get_db_path(), readonly=readonly)


def init_db():
    """Create tables and FTS5 index."""
    path = get_db_path()
    conn = sqlite3.connect(str(path))
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            claim_type TEXT NOT NULL,
            source_doi TEXT NOT NULL,
            source_paper_title TEXT,
            confidence TEXT,
            location_in_paper TEXT,
            verbatim_quote TEXT,
            extraction_model TEXT,
            extraction_version TEXT,
            extracted_at TEXT,
            view_paths TEXT,  -- JSON
            data TEXT NOT NULL  -- full JSON blob
        );

        CREATE TABLE IF NOT EXISTS sources (
            doi TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,  -- JSON array
            year INTEGER,
            venue TEXT,
            abstract TEXT,
            citation_count INTEGER DEFAULT 0,
            open_access_url TEXT,
            data TEXT NOT NULL  -- full JSON blob
        );

        CREATE TABLE IF NOT EXISTS tree_nodes (
            view_id TEXT NOT NULL,
            path TEXT NOT NULL,  -- slash-separated, e.g. "catalysis/heterogeneous"
            name TEXT,
            level INTEGER,
            claim_count INTEGER DEFAULT 0,
            children TEXT,  -- JSON array of child path segments
            claim_ids TEXT,  -- JSON array (only for leaf-ish nodes)
            data TEXT NOT NULL,
            PRIMARY KEY (view_id, path)
        );

        CREATE TABLE IF NOT EXISTS views (
            view_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            data TEXT NOT NULL
        );

        -- ── Living taxonomy (scaffold + per-view hosts + paper leaves) ──
        -- Concept nodes (DAG): laws/frameworks/theories/models/mechanisms +
        -- proposed branches. Shared across views; membership via taxonomy_edges.
        CREATE TABLE IF NOT EXISTS taxonomy_nodes (
            node_id TEXT PRIMARY KEY,
            kind TEXT,            -- open_root|law|framework|theory|model|mechanism|class
            name TEXT,
            definition TEXT,
            short_label TEXT,     -- terse glyph label for the node-link tree
            equation TEXT,        -- concise LaTeX equation (optional)
            proposed INTEGER DEFAULT 0,   -- 1 = candidate (not yet committed)
            data TEXT
        );
        -- Per-view parent->child edges (multi-parent DAG + cross-links).
        CREATE TABLE IF NOT EXISTS taxonomy_edges (
            view_id TEXT NOT NULL,
            parent_id TEXT,       -- NULL = view root
            child_id TEXT NOT NULL,
            PRIMARY KEY (view_id, parent_id, child_id)
        );
        -- Paper-grounded leaf placements: a claim attached under a host node.
        CREATE TABLE IF NOT EXISTS taxonomy_leaves (
            view_id TEXT NOT NULL,
            node_id TEXT NOT NULL,   -- host this leaf hangs under
            claim_id TEXT NOT NULL,
            doi TEXT,                -- source paper (leaf = paper; claims grouped by doi)
            score REAL,
            label TEXT,
            PRIMARY KEY (view_id, node_id, claim_id)
        );
        CREATE TABLE IF NOT EXISTS taxonomy_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        -- Pre-computed paper intelligence per placed (view, host node, paper).
        -- Replaces real-time LLM calls: advisor questions, claim/logic critique,
        -- and structural contribution are generated in batch and served instantly.
        CREATE TABLE IF NOT EXISTS paper_analysis (
            view_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            doi TEXT NOT NULL,
            advisor_json TEXT,        -- grounded positioning questions
            critique_json TEXT,       -- are claims supported; is reasoning sound
            contribution_json TEXT,   -- how it extends/challenges parent + siblings
            generated_at TEXT,
            PRIMARY KEY (view_id, node_id, doi)
        );

        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doi TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
            submitter_name TEXT,
            submitter_email TEXT,
            notes TEXT,
            result TEXT  -- JSON with extraction/classification results
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
        CREATE INDEX IF NOT EXISTS idx_claims_doi ON claims(source_doi);
        CREATE INDEX IF NOT EXISTS idx_sources_year ON sources(year);
        CREATE INDEX IF NOT EXISTS idx_tree_view ON tree_nodes(view_id);
        CREATE INDEX IF NOT EXISTS idx_tree_level ON tree_nodes(view_id, level);
        CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);
        CREATE INDEX IF NOT EXISTS idx_ltax_edges_parent ON taxonomy_edges(view_id, parent_id);
        CREATE INDEX IF NOT EXISTS idx_ltax_edges_child ON taxonomy_edges(view_id, child_id);
        CREATE INDEX IF NOT EXISTS idx_ltax_leaves_node ON taxonomy_leaves(view_id, node_id);
        CREATE INDEX IF NOT EXISTS idx_ltax_leaves_claim ON taxonomy_leaves(claim_id);
        CREATE INDEX IF NOT EXISTS idx_panalysis_doi ON paper_analysis(doi);

        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            view TEXT,
            filters TEXT,
            result_count INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            user_agent TEXT,
            ip_hash TEXT,
            user_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_querylog_ts ON query_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_querylog_query ON query_log(query);

        CREATE TABLE IF NOT EXISTS click_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query_log_id INTEGER,
            query TEXT,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            position INTEGER,
            user_id TEXT,
            ip_hash TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_clicklog_ts ON click_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_clicklog_user ON click_log(user_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_clicklog_qlid ON click_log(query_log_id);

        CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            email TEXT,
            tier TEXT DEFAULT 'free',
            rate_limit INTEGER DEFAULT 100,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            is_active INTEGER DEFAULT 1,
            user_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys(key_hash);

        CREATE TABLE IF NOT EXISTS authors (
            author_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            openalex_id TEXT,
            orcid TEXT,
            institution TEXT,
            institution_country TEXT,
            h_index INTEGER DEFAULT 0,
            works_count INTEGER DEFAULT 0,
            cited_by_count INTEGER DEFAULT 0,
            concepts TEXT,
            data TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_authors (
            doi TEXT NOT NULL,
            author_id TEXT NOT NULL,
            position TEXT,
            PRIMARY KEY (doi, author_id)
        );

        CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(name);
        CREATE INDEX IF NOT EXISTS idx_authors_inst ON authors(institution);
        CREATE INDEX IF NOT EXISTS idx_paper_authors_doi ON paper_authors(doi);
        CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);

        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            sub_type TEXT NOT NULL,
            target TEXT NOT NULL,
            frequency TEXT DEFAULT 'weekly',
            created_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            last_notified_at TEXT,
            user_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_subs_email ON subscriptions(email);
        CREATE INDEX IF NOT EXISTS idx_subs_active ON subscriptions(is_active);

        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER NOT NULL,
            sent_at TEXT NOT NULL,
            claim_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'sent',
            error TEXT,
            FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_notif_sub ON notification_log(subscription_id);

        CREATE TABLE IF NOT EXISTS surprise_scores (
            claim_id TEXT PRIMARY KEY,
            total_score REAL DEFAULT 0,
            structural_score REAL DEFAULT 0,
            temporal_score REAL DEFAULT 0,
            content_score REAL DEFAULT 0,
            computed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_surprise_total ON surprise_scores(total_score);

        CREATE TABLE IF NOT EXISTS community_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT NOT NULL,
            flag_type TEXT NOT NULL,  -- wrong_claim, wrong_classification, not_chemistry, duplicate, other
            category TEXT,            -- e.g. which view/path is wrong
            comment TEXT,
            suggested_fix TEXT,
            reporter_name TEXT,
            reporter_email TEXT,
            status TEXT DEFAULT 'open',  -- open, reviewed, resolved, dismissed
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewer_notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_flags_claim ON community_flags(claim_id);
        CREATE INDEX IF NOT EXISTS idx_flags_status ON community_flags(status);
        CREATE INDEX IF NOT EXISTS idx_flags_type ON community_flags(flag_type);

        -- Community feedback on the LIVING TREE (nodes/branches and paper
        -- placements), distinct from claim-level community_flags. Never dropped.
        CREATE TABLE IF NOT EXISTS ltree_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            view_id TEXT NOT NULL,
            node_id TEXT,             -- branch the feedback is about (NULL = whole view)
            doi TEXT,                 -- optional: a specific paper placement under node
            kind TEXT NOT NULL,       -- mislabeled, misplaced, duplicate, wrong_parent, missing, other
            comment TEXT,
            reporter_name TEXT,
            reporter_email TEXT,
            ip_hash TEXT,
            status TEXT DEFAULT 'open',  -- open, reviewed, resolved, dismissed
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewer_notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ltfb_node ON ltree_feedback(view_id, node_id);
        CREATE INDEX IF NOT EXISTS idx_ltfb_status ON ltree_feedback(status);

        CREATE TABLE IF NOT EXISTS paper_validations (
            doi TEXT PRIMARY KEY,
            crossref_verified INTEGER DEFAULT 0,
            has_abstract INTEGER DEFAULT 0,
            is_retracted INTEGER DEFAULT 0,
            journal TEXT,
            publisher TEXT,
            is_chemistry INTEGER DEFAULT 1,
            validation_score REAL DEFAULT 0,  -- 0-1 composite quality score
            validated_at TEXT,
            validation_data TEXT  -- JSON with full validation details
        );

        CREATE INDEX IF NOT EXISTS idx_pv_score ON paper_validations(validation_score);

        CREATE TABLE IF NOT EXISTS key_usage (
            key_id TEXT NOT NULL,
            date TEXT NOT NULL,
            request_count INTEGER DEFAULT 0,
            PRIMARY KEY (key_id, date)
        );
        CREATE INDEX IF NOT EXISTS idx_key_usage_date ON key_usage(date);

        CREATE TABLE IF NOT EXISTS security_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            ip_hash TEXT,
            details TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_security_log_ts ON security_log(timestamp);

        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id_1 TEXT NOT NULL,
            claim_id_2 TEXT NOT NULL,
            view_id TEXT,
            node_path TEXT,
            paw_verdict TEXT,
            gemini_verdict TEXT,
            gemini_explanation TEXT,
            confidence REAL,
            detected_at TEXT,
            UNIQUE(claim_id_1, claim_id_2)
        );
        CREATE INDEX IF NOT EXISTS idx_contradictions_view
            ON contradictions(view_id, node_path);
        CREATE INDEX IF NOT EXISTS idx_contradictions_claim1
            ON contradictions(claim_id_1);
        CREATE INDEX IF NOT EXISTS idx_contradictions_claim2
            ON contradictions(claim_id_2);

        CREATE TABLE IF NOT EXISTS claim_view_map (
            claim_id TEXT NOT NULL,
            view_id  TEXT NOT NULL,
            path     TEXT NOT NULL,
            PRIMARY KEY (claim_id, view_id, path)
        );
        CREATE INDEX IF NOT EXISTS idx_cvm_view_path
            ON claim_view_map(view_id, path);

        -- Typed directed edges between claims (or claim → external DOI).
        -- Edge vocabulary lives in askchem.models.EDGE_TYPES; see docs/claim_edges.md.
        -- Exactly one of to_claim_id or to_doi must be non-empty (model layer
        -- enforces this on insert). The unused target is stored as '' (not NULL)
        -- so the UNIQUE constraint can include it directly.
        CREATE TABLE IF NOT EXISTS claim_edges (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            from_claim_id   TEXT NOT NULL,
            edge_type       TEXT NOT NULL,
            to_claim_id     TEXT NOT NULL DEFAULT '',
            to_doi          TEXT NOT NULL DEFAULT '',
            confidence      TEXT NOT NULL DEFAULT 'medium',
            evidence        TEXT NOT NULL DEFAULT '',
            extractor       TEXT NOT NULL,
            extracted_at    TEXT NOT NULL,
            UNIQUE(from_claim_id, edge_type, to_claim_id, to_doi, extractor)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_from
            ON claim_edges(from_claim_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to_claim
            ON claim_edges(to_claim_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to_doi
            ON claim_edges(to_doi);
        CREATE INDEX IF NOT EXISTS idx_edges_type
            ON claim_edges(edge_type);
        CREATE INDEX IF NOT EXISTS idx_edges_extractor
            ON claim_edges(extractor);

        -- Per-paper, per-mode bookkeeping for the edge backfill jobs.
        -- Used to make the runner resumable: a paper marked 'done' for a given
        -- (mode, extractor) pair is skipped on the next pass.
        CREATE TABLE IF NOT EXISTS edge_jobs (
            paper_doi      TEXT NOT NULL,
            mode           TEXT NOT NULL,
            extractor      TEXT NOT NULL,
            status         TEXT NOT NULL,   -- 'done' | 'failed' | 'skipped'
            edges_inserted INTEGER NOT NULL DEFAULT 0,
            tokens_in      INTEGER NOT NULL DEFAULT 0,
            tokens_out     INTEGER NOT NULL DEFAULT 0,
            error          TEXT,
            started_at     TEXT NOT NULL,
            finished_at    TEXT,
            PRIMARY KEY (paper_doi, mode, extractor)
        );
        CREATE INDEX IF NOT EXISTS idx_edge_jobs_mode_status
            ON edge_jobs(mode, extractor, status);

        -- Paper-to-paper citation graph used to drive cross-paper claim edge
        -- extraction.  One row per (citing, cited, source) so we can re-fetch
        -- a single source independently and audit provenance.  Multiple
        -- sources may agree on the same citing/cited pair; that's fine.
        CREATE TABLE IF NOT EXISTS citations (
            citing_doi  TEXT NOT NULL,
            cited_doi   TEXT NOT NULL,
            source      TEXT NOT NULL,    -- 'crossref' | 's2'
            fetched_at  TEXT NOT NULL,
            PRIMARY KEY (citing_doi, cited_doi, source)
        );
        CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_doi);
        CREATE INDEX IF NOT EXISTS idx_citations_cited  ON citations(cited_doi);

        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            display_name TEXT,
            clerk_id TEXT,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

        CREATE TABLE IF NOT EXISTS user_sessions (
            session_token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, target_type, target_id, feedback_type)
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_target
            ON feedback(target_type, target_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);

        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            title TEXT,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, target_type, target_id)
        );
        CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_bookmarks_target ON bookmarks(target_type, target_id);

        CREATE TABLE IF NOT EXISTS saved_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT,
            query TEXT NOT NULL,
            view TEXT,
            filters TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_saved_searches_user
            ON saved_searches(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS reading_lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            is_public INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_reading_lists_user
            ON reading_lists(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_reading_lists_public
            ON reading_lists(is_public, updated_at DESC);

        CREATE TABLE IF NOT EXISTS reading_list_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            list_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            title TEXT,
            note TEXT,
            position INTEGER DEFAULT 0,
            added_at TEXT NOT NULL,
            FOREIGN KEY (list_id) REFERENCES reading_lists(id),
            UNIQUE(list_id, target_type, target_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rlist_items_list
            ON reading_list_items(list_id, position, added_at);
    """)

    # FTS5 virtual table for full-text search (standalone, not content-synced)
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
            claim_id UNINDEXED,
            claim_type,
            source_paper_title,
            verbatim_quote,
            searchable_text
        )
    """)

    # Paper-level FTS for document recall
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(
            doi UNINDEXED,
            title,
            abstract,
            paper_text
        )
    """)

    # Migrations for existing databases
    try:
        c.execute("ALTER TABLE subscriptions ADD COLUMN last_notified_at TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        c.execute("ALTER TABLE subscriptions ADD COLUMN manage_token TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE api_keys ADD COLUMN total_requests INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN clerk_id TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE query_log ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("CREATE INDEX IF NOT EXISTS idx_querylog_user ON query_log(user_id, timestamp)")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE subscriptions ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("CREATE INDEX IF NOT EXISTS idx_subs_user ON subscriptions(user_id, is_active)")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE api_keys ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("CREATE INDEX IF NOT EXISTS idx_apikeys_user ON api_keys(user_id, is_active)")
    except sqlite3.OperationalError:
        pass

    # ── Cho-style contextual retrieval columns (Sprint 0 + Sprint 1) ──
    # claim_contextualized: one-sentence standalone rewrite, written by Sprint 1
    # for every deep_v1 (full-paper) claim. Abstract-only claims keep this
    # NULL — they are not contextualized (rewriting an abstract-derived claim
    # would just paraphrase, not enrich).
    for col, ddl in (
        ("claim_contextualized",     "ALTER TABLE claims ADD COLUMN claim_contextualized TEXT"),
        ("context_model",            "ALTER TABLE claims ADD COLUMN context_model TEXT"),
        ("context_version",          "ALTER TABLE claims ADD COLUMN context_version TEXT"),
        ("context_extracted_at",     "ALTER TABLE claims ADD COLUMN context_extracted_at TEXT"),
    ):
        try:
            c.execute(ddl)
        except sqlite3.OperationalError:
            pass

    # paper_summary: short, claim-grounded summary per source paper, written
    # by Sprint 0 for the 41k papers that have at least one deep_v1 claim.
    # Used as paper-level context in the Sprint 1 contextualization prompt
    # AND as a separate field in the future Sprint 2 multi-field FTS index.
    for col, ddl in (
        ("paper_summary",                "ALTER TABLE sources ADD COLUMN paper_summary TEXT"),
        ("paper_summary_model",          "ALTER TABLE sources ADD COLUMN paper_summary_model TEXT"),
        ("paper_summary_version",        "ALTER TABLE sources ADD COLUMN paper_summary_version TEXT"),
        ("paper_summary_extracted_at",   "ALTER TABLE sources ADD COLUMN paper_summary_extracted_at TEXT"),
    ):
        try:
            c.execute(ddl)
        except sqlite3.OperationalError:
            pass

    users_sql = c.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()
    if users_sql and "UNIQUE" in (users_sql[0] or ""):
        c.executescript("""
            CREATE TABLE users_new (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT,
                clerk_id TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );
            INSERT INTO users_new (user_id, email, display_name, clerk_id, created_at, last_login_at)
                SELECT user_id, email, display_name, clerk_id, created_at, last_login_at FROM users;
            DROP TABLE users;
            ALTER TABLE users_new RENAME TO users;
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """)

    try:
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_clerk "
            "ON users(clerk_id) WHERE clerk_id IS NOT NULL"
        )
    except sqlite3.OperationalError:
        pass

    import secrets as _secrets

    for row in c.execute("SELECT id FROM subscriptions WHERE manage_token IS NULL").fetchall():
        c.execute(
            "UPDATE subscriptions SET manage_token = ? WHERE id = ?",
            (_secrets.token_urlsafe(24), row[0]),
        )

    conn.commit()
    conn.close()


def build_searchable_text(claim_data: dict) -> str:
    """Build a searchable text blob from claim fields."""
    parts = []
    for key in ['verbatim_quote', 'source_paper_title', 'claim_type',
                'reaction_type', 'subject', 'property_name', 'technique_name',
                'what_it_achieves', 'process_described', 'comparison_result',
                'key_innovation']:
        val = claim_data.get(key, '')
        if val:
            parts.append(str(val))

    for key in ['reactants', 'products', 'compared_items', 'steps', 'key_intermediates']:
        val = claim_data.get(key, [])
        if val:
            for item in val:
                if isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if v)
                elif isinstance(item, str):
                    parts.append(item)

    for key in ['conditions', 'outcomes']:
        val = claim_data.get(key, {})
        if val and isinstance(val, dict):
            parts.extend(str(v) for v in val.values() if v)
        elif val and isinstance(val, str):
            parts.append(val)

    return ' '.join(parts)


def build_sources_fts():
    """Populate the sources_fts index from the sources table (title + abstract only).

    For enriched paper_text from claims, use build_paper_searchable_text() instead.
    """
    init_db()
    path = get_db_path()
    conn = sqlite3.connect(str(path))
    conn.execute("DELETE FROM sources_fts")
    rows = conn.execute(
        "SELECT doi, title, abstract FROM sources WHERE title IS NOT NULL"
    ).fetchall()
    batch = []
    for r in rows:
        batch.append((r[0], r[1] or '', r[2] or '', ''))
        if len(batch) >= 5000:
            conn.executemany(
                "INSERT INTO sources_fts (doi, title, abstract, paper_text) "
                "VALUES (?, ?, ?, ?)",
                batch,
            )
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT INTO sources_fts (doi, title, abstract, paper_text) "
            "VALUES (?, ?, ?, ?)",
            batch,
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM sources_fts").fetchone()[0]
    conn.close()
    print(f"Indexed {count:,} papers in sources_fts")
    return count


def _make_paper_text(claims_data: list[dict]) -> str:
    """Build a compact, search-friendly text blob from a paper's claims.

    Extracts curated high-signal fields (subjects, property names, reaction
    types, technique names, claim types) and a few truncated verbatim quotes
    to produce ~500-1000 chars of domain vocabulary per paper.
    """
    subjects: set[str] = set()
    property_names: set[str] = set()
    reaction_types: set[str] = set()
    technique_names: set[str] = set()
    claim_types: set[str] = set()
    quotes: list[str] = []

    for c in claims_data:
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except (json.JSONDecodeError, TypeError):
                continue

        for field, target in [
            ('subject', subjects),
            ('property_name', property_names),
            ('reaction_type', reaction_types),
            ('technique_name', technique_names),
        ]:
            val = c.get(field)
            if val and isinstance(val, str) and len(val) > 1:
                target.add(val)

        ct = c.get('claim_type')
        if ct:
            claim_types.add(ct)

        vq = c.get('verbatim_quote', '')
        if vq and len(quotes) < 3:
            quotes.append(vq[:150])

        # Also capture key terms from structured sub-fields
        for list_field in ('reactants', 'products', 'compared_items'):
            items = c.get(list_field, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = item.get('name') or item.get('formula') or ''
                        if name:
                            subjects.add(str(name))
                    elif isinstance(item, str):
                        subjects.add(item)

        conds = c.get('conditions', {})
        if isinstance(conds, dict):
            for v in conds.values():
                if isinstance(v, str) and len(v) > 2:
                    subjects.add(v)

    parts: list[str] = []
    if subjects:
        parts.append(' '.join(sorted(subjects)[:30]))
    if property_names:
        parts.append(' '.join(sorted(property_names)[:15]))
    if reaction_types:
        parts.append(' '.join(sorted(reaction_types)[:10]))
    if technique_names:
        parts.append(' '.join(sorted(technique_names)[:10]))
    if claim_types:
        parts.append(' '.join(sorted(claim_types)))
    for q in quotes:
        parts.append(q)

    return ' '.join(parts)[:2000]


def build_paper_searchable_text():
    """Rebuild sources_fts with enriched paper_text derived from claims.

    For each paper, aggregates curated claim fields (subject, property_name,
    reaction_type, technique_name, verbatim_quote snippets) into a compact
    text blob and stores it in the paper_text column of sources_fts alongside
    the original title and abstract.
    """
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Recreate the FTS table to ensure schema matches
    conn.execute("DROP TABLE IF EXISTS sources_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE sources_fts USING fts5(
            doi UNINDEXED,
            title,
            abstract,
            paper_text
        )
    """)
    conn.commit()

    # Load all source metadata
    sources = {}
    for r in conn.execute("SELECT doi, title, abstract FROM sources WHERE title IS NOT NULL"):
        sources[r[0]] = {'title': r[1] or '', 'abstract': r[2] or ''}
    print(f"Loaded {len(sources):,} sources")

    # Build paper text from claims in streaming batches
    paper_claims: dict[str, list[dict]] = {}
    cursor = conn.execute("SELECT source_doi, data FROM claims WHERE source_doi != ''")
    claim_count = 0
    while True:
        rows = cursor.fetchmany(10_000)
        if not rows:
            break
        for doi, data_json in rows:
            claim_count += 1
            try:
                claim = json.loads(data_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if doi not in paper_claims:
                paper_claims[doi] = []
            paper_claims[doi].append(claim)

        if claim_count % 200_000 == 0:
            print(f"  Read {claim_count:,} claims...", flush=True)

    print(f"Read {claim_count:,} claims across {len(paper_claims):,} papers")

    # Insert into sources_fts
    batch: list[tuple[str, str, str, str]] = []
    papers_with_text = 0
    for doi, meta in sources.items():
        claims = paper_claims.get(doi, [])
        pt = _make_paper_text(claims) if claims else ''
        if pt:
            papers_with_text += 1
        batch.append((doi, meta['title'], meta['abstract'], pt))
        if len(batch) >= 5000:
            conn.executemany(
                "INSERT INTO sources_fts (doi, title, abstract, paper_text) "
                "VALUES (?, ?, ?, ?)",
                batch,
            )
            batch.clear()
            conn.commit()

    if batch:
        conn.executemany(
            "INSERT INTO sources_fts (doi, title, abstract, paper_text) "
            "VALUES (?, ?, ?, ?)",
            batch,
        )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM sources_fts").fetchone()[0]
    conn.close()
    print(f"Indexed {count:,} papers in sources_fts "
          f"({papers_with_text:,} with enriched paper_text)")
    return count


def build_claim_view_map():
    """Populate the claim_view_map junction table from claims.view_paths.

    For each claim, inserts one row per (view_id, leaf_path).  Also inserts
    rows for every ancestor prefix so subtree queries via
    ``WHERE path = ? OR path LIKE ?||'/%'`` work efficiently.
    """
    init_db()
    path = get_db_path()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("DELETE FROM claim_view_map")
    conn.commit()

    cursor = conn.execute(
        "SELECT claim_id, view_paths FROM claims "
        "WHERE view_paths IS NOT NULL AND view_paths != '{}'"
    )

    batch: list[tuple[str, str, str]] = []
    total_rows = 0
    claims_processed = 0

    while True:
        rows = cursor.fetchmany(5000)
        if not rows:
            break
        for cid, vp_json in rows:
            claims_processed += 1
            try:
                vp = json.loads(vp_json)
            except (json.JSONDecodeError, TypeError):
                continue
            for view_id, paths in vp.items():
                if not isinstance(paths, list) or not paths:
                    continue
                leaf_path = '/'.join(str(s) for s in paths)
                # Insert the leaf path and all ancestor prefixes
                segments = leaf_path.split('/')
                for depth in range(1, len(segments) + 1):
                    prefix = '/'.join(segments[:depth])
                    batch.append((cid, view_id, prefix))

            if len(batch) >= 50_000:
                conn.executemany(
                    "INSERT OR IGNORE INTO claim_view_map (claim_id, view_id, path) "
                    "VALUES (?, ?, ?)",
                    batch,
                )
                total_rows += len(batch)
                batch.clear()
                conn.commit()
                if claims_processed % 100_000 == 0:
                    print(f"  Processed {claims_processed:,} claims, "
                          f"~{total_rows:,} map rows...", flush=True)

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO claim_view_map (claim_id, view_id, path) "
            "VALUES (?, ?, ?)",
            batch,
        )
        total_rows += len(batch)

    conn.commit()
    final_count = conn.execute("SELECT COUNT(*) FROM claim_view_map").fetchone()[0]
    conn.close()
    print(f"Built claim_view_map: {final_count:,} rows "
          f"from {claims_processed:,} claims")
    return final_count


# ── Tree-based recall (BFS) ─────────────────────────────────────────────────

_TreeNode = tuple[str, str, tuple[str, ...], frozenset[str], tuple[str, ...]]
# (view_id, path, node_words, node_stem_set, node_stems_tuple)
_tree_node_cache: list[_TreeNode] | None = None
_tree_node_cache_time: float = 0
_TREE_CACHE_TTL = 600  # 10 min


def _load_tree_node_index() -> list[_TreeNode]:
    """Load all tree-node tuples for node matching, pre-stemmed.

    The taxonomy has ~296K nodes (not the 4K the original comment
    assumed). Re-stemming every node on every query inside
    ``_match_tree_nodes`` is the dominant runtime cost — see the δ1
    profile (5M ``_stem`` calls per query, 58 % of total search
    latency). We pay the cost once here, at load time, then cache
    ``(view_id, path, words, stem_set, stem_tuple)`` so the matcher
    does zero stemming on the hot path.
    """
    global _tree_node_cache, _tree_node_cache_time
    now = _time.time()
    if _tree_node_cache is not None and (now - _tree_node_cache_time) < _TREE_CACHE_TTL:
        return _tree_node_cache

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT view_id, path FROM tree_nodes "
            "WHERE path != '' AND level >= 1"
        ).fetchall()

    index: list[_TreeNode] = []
    for r in rows:
        view_id = r['view_id']
        path = r['path']
        if view_id in ('by_paper', 'by_time_period', 'by_claim_type'):
            continue
        segments = path.split('/')
        words: list[str] = []
        for seg in segments:
            words.extend(seg.split('_'))
        stems = tuple(_stem(w) for w in words)
        index.append((view_id, path, tuple(words), frozenset(stems), stems))

    _tree_node_cache = index
    _tree_node_cache_time = now
    return index


def _stem(word: str) -> str:
    """Minimal chemistry-aware stemmer: strip common suffixes for matching."""
    if len(word) <= 3:
        return word
    # Strip simple plural first (handles metals->metal, catalysts->catalyst)
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        word = word[:-1]
    # Then strip derivational suffixes
    for suffix in ('ation', 'tion', 'sion', 'ment', 'ness',
                   'ical', 'ous', 'ive', 'ing', 'ity'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


# Stems of tokens that are too generic to be the *only* intersection between
# a multi-word user query and a taxonomy node.  Otherwise "Suzuki coupling"
# stem-matches only on ``coupl`` for nodes like ``spin_coupling`` / exchange
# coupling in condensed-matter paths, and tree recall floods irrelevant IDs.
_TREE_WEAK_SINGLE_OVERLAP_STEMS: frozenset[str] = frozenset(
    _stem(w)
    for w in (
        "coupling", "bonding", "interaction", "correlation", "relaxation",
        "dynamics", "reaction", "synthesis", "catalysis", "oxidation",
        "reduction", "exchange", "transfer", "binding", "adsorption",
        "desorption", "diffusion", "separation", "extraction", "complex",
        "association", "dissociation", "polymerization", "degradation",
        "spectroscopy", "microscopy", "diffraction", "calorimetry",
    )
)


def _match_tree_nodes(
    query: str,
    top_k: int = 10,
    restrict_view_id: str | None = None,
) -> list[tuple[str, str, float]]:
    """Map a free-text query to the best-matching tree node paths.

    Returns up to top_k (view_id, path, score) tuples sorted by score desc.

    If ``restrict_view_id`` is set, only nodes belonging to that taxonomy
    view are considered. This is used when ``search_claims(..., view=...)``
    filters results to one view: matching nodes from *other* views (e.g.
    reaction-type subtrees for "Suzuki coupling") must not inject recall
    scores into claims that merely carry an unrelated technique tag — a
    common source of irrelevant hits in the Technique/Method grouped search.

    Multi-word queries additionally ignore nodes whose only token overlap with
    the query is a very generic stem (e.g. ``coupl`` from "coupling") so named
    phrases like "Suzuki coupling" are not reduced to the noisy token
    "coupling" alone.

    Scoring:
      - Each query token that stem-matches a node's segment word earns 1 point
      - Bigram matches (consecutive query words matching consecutive segment words)
        earn a bonus point
      - Score is normalized by max(query_tokens, node_words) to balance precision
      - Deeper nodes (more path segments) get a depth bonus so specific matches
        rank above broad ones
    """
    node_index = _load_tree_node_index()
    if not node_index:
        return []

    raw = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    q_words = [w for w in raw.split() if w not in STOP_WORDS and len(w) > 1]
    if not q_words:
        return []

    q_stems = [_stem(w) for w in q_words]
    q_stem_set = set(q_stems)

    results: list[tuple[str, str, float]] = []
    for view_id, path, node_words, node_stem_set, node_stems in node_index:
        if restrict_view_id is not None and view_id != restrict_view_id:
            continue
        overlap = q_stem_set & node_stem_set
        if not overlap:
            continue

        # Bandaid: when a multi-word query overlaps a tree node only on a
        # generic stem (``coupl`` for "Suzuki coupling" vs spin-coupling /
        # exchange-coupling paths), drop the node. mxbai's dense channel
        # handles the same disambiguation natively, so this filter is a
        # candidate to retire in δ2.  Kill switch:
        # CHEMTREE_DISABLE_WEAK_STEM_SKIP=1.
        if (len(q_words) >= 2
                and len(overlap) == 1
                and os.environ.get(
                    "CHEMTREE_DISABLE_WEAK_STEM_SKIP", "0"
                ) != "1"):
            sole = next(iter(overlap))
            if sole in _TREE_WEAK_SINGLE_OVERLAP_STEMS:
                continue

        hit_count = len(overlap)

        # Bigram bonus: consecutive query stems matching consecutive node stems
        bigram_bonus = 0
        for i in range(len(q_stems) - 1):
            bi = (q_stems[i], q_stems[i + 1])
            for j in range(len(node_stems) - 1):
                if node_stems[j] == bi[0] and node_stems[j + 1] == bi[1]:
                    bigram_bonus += 1
                    break

        depth = path.count('/') + 1
        depth_bonus = 0.1 * depth

        denom = max(len(q_words), len(node_stem_set))
        score = (hit_count + bigram_bonus) / denom + depth_bonus

        results.append((view_id, path, score))

    results.sort(key=lambda x: -x[2])
    return results[:top_k]


def _tree_recall(
    query: str,
    conn,
    top_k: int = 200,
    restrict_view_id: str | None = None,
) -> list[str]:
    """Retrieve claim_ids from taxonomy subtrees matching the query.

    Pipeline:
      1. Match query words to tree node paths (optionally scoped to one view)
      2. For each matched node, fetch claim_ids from claim_view_map
         (the node itself + all descendants)
      3. Pool and deduplicate, with scores decayed by tree depth from
         the matched node
      4. Vector-rerank the pool against the query
      5. Return top-K claim_ids
    """
    matched_nodes = _match_tree_nodes(
        query, top_k=8, restrict_view_id=restrict_view_id,
    )
    if not matched_nodes:
        return []

    has_map = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claim_view_map'"
    ).fetchone()
    if not has_map:
        return []

    cid_scores: dict[str, float] = {}

    for view_id, path, match_score in matched_nodes:
        # Use prefix range scan instead of LIKE for efficient B-tree traversal.
        # path + '/' is the prefix, path + '0' is just past all children
        # ('0' > '/' in ASCII), so this captures the exact path plus all
        # descendant paths without a full table scan.
        rows = conn.execute(
            "SELECT claim_id, path FROM claim_view_map "
            "WHERE view_id = ? AND path >= ? AND path < ?",
            [view_id, path, path + '0'],
        ).fetchall()

        matched_depth = path.count('/') + 1
        for r in rows:
            cid = r['claim_id']
            claim_depth = r['path'].count('/') + 1
            extra_depth = claim_depth - matched_depth
            decay = 1.0 / (1.0 + 0.3 * extra_depth)
            score = match_score * decay
            if cid not in cid_scores or score > cid_scores[cid]:
                cid_scores[cid] = score

    if not cid_scores:
        return []

    # Pre-filter to top candidates by tree score before expensive reranking
    RERANK_POOL = top_k * 4
    ordered = sorted(cid_scores, key=lambda c: -cid_scores[c])
    pool = ordered[:RERANK_POOL]

    # May-15 ablation: the tree semantic_rerank step calls
    # embeddings_v2.semantic_rerank, which reconstructs up to 800 vectors
    # from the FAISS mmap before scoring — a major contributor to warm
    # latency on CPU. Two kill switches:
    #   CHEMTREE_DISABLE_TREE_RERANK=1        — always skip
    #   CHEMTREE_TREE_RERANK_POOL_CAP=<N>     — skip when pool > N
    if os.environ.get("CHEMTREE_DISABLE_TREE_RERANK", "0") == "1":
        return pool[:top_k]
    try:
        _tree_rerank_pool_cap = int(
            os.environ.get("CHEMTREE_TREE_RERANK_POOL_CAP", "0") or "0"
        )
    except ValueError:
        _tree_rerank_pool_cap = 0
    if _tree_rerank_pool_cap > 0 and len(pool) > _tree_rerank_pool_cap:
        return pool[:top_k]

    try:
        from askchem.retrieval import (
            is_loaded as embeddings_loaded, semantic_rerank,
        )
        if embeddings_loaded():
            try:
                tree_min = float(
                    os.environ.get("CHEMTREE_TREE_MIN_SCORE", "0.10")
                )
            except ValueError:
                tree_min = 0.10
            reranked = semantic_rerank(query, pool,
                                       top_k=top_k, min_score=tree_min)
            return [cid for cid, _ in reranked]
    except Exception:
        pass

    return pool[:top_k]


def _paper_recall(query: str, conn, top_k: int = 10) -> list[str]:
    """Find top-K relevant papers by searching titles/abstracts/paper_text.

    Strategy:
      1. Title/abstract queries (high signal, weighted 2x via double-insert)
      2. Full-column queries for the original terms
      3. paper_text-specific queries for synonym-expanded terms to find
         papers whose claims mention specific chemicals/methods
      4. RRF-merge all result lists
      5. Citation-boost: re-rank by combining RRF score with log(citations)
    """
    import math
    try:
        conn.execute("SELECT 1 FROM sources_fts LIMIT 1")
    except sqlite3.OperationalError:
        return []

    per_list_limit = top_k * 8
    ranked_lists: list[list[str]] = []

    def _run_fts(fts_q: str):
        try:
            hits = conn.execute(
                "SELECT doi FROM sources_fts "
                "WHERE sources_fts MATCH ? ORDER BY rank LIMIT ?",
                [fts_q, per_list_limit],
            ).fetchall()
            if hits:
                ranked_lists.append([h['doi'] for h in hits])
        except sqlite3.OperationalError:
            pass

    def _run_cite_ranked(fts_q: str):
        """Fetch papers matching the FTS query, ranked by citation count.

        This surfaces landmark papers that BM25 ranks low because their
        titles are short or don't repeat keywords densely.
        """
        try:
            hits = conn.execute(
                "SELECT f.doi FROM sources_fts f "
                "JOIN sources s ON s.doi = f.doi "
                "WHERE sources_fts MATCH ? "
                "ORDER BY s.citation_count DESC LIMIT ?",
                [fts_q, per_list_limit],
            ).fetchall()
            if hits:
                ranked_lists.append([h['doi'] for h in hits])
        except sqlite3.OperationalError:
            pass

    words = _clean_query_words(query)

    # (a) Title-specific queries (high signal — double-insert for 2x RRF weight)
    if len(words) > 1:
        _run_fts('{title}: ' + ' '.join(f'"{w}"' for w in words))
        _run_fts('{title}: ' + ' '.join(f'"{w}"' for w in words))
    else:
        _run_fts('{title}: ' + f'"{words[0]}"')
        _run_fts('{title}: ' + f'"{words[0]}"')

    # (b) All-column queries for original terms
    if len(words) > 1:
        _run_fts(' '.join(f'"{w}"' for w in words))
        _run_fts(f'NEAR({" ".join(words)}, 10)')
    else:
        _run_fts(f'"{words[0]}"')
        _run_fts(f'{words[0]}*')

    # (b2) Citation-ranked recall: queries ranked by citation count.
    # Surfaces landmark papers that BM25 underranks because 1000+ papers
    # match broad terms like "CO2 reduction".  Double-inserted for 2x
    # RRF weight (same technique as title queries) because citation
    # authority is a critical signal for recall quality.
    if len(words) > 1:
        _run_cite_ranked(' '.join(f'"{w}"' for w in words))
        _run_cite_ranked(' '.join(f'"{w}"' for w in words))
        # Citation-ranked on 2-word pairs — key recall path for broad
        # queries.  E.g. "CO2"+"reduction" cite-ranked finds landmark
        # papers that the full AND-of-all-terms never reaches.
        if len(words) >= 3:
            from itertools import combinations
            for combo in list(combinations(words, 2)):
                _run_cite_ranked(' '.join(f'"{w}"' for w in combo))
    else:
        _run_cite_ranked(f'"{words[0]}"')
        _run_cite_ranked(f'"{words[0]}"')

    # (b3) Sub-query recall for long queries: 2-word title combinations
    if len(words) >= 4:
        from itertools import combinations as _c
        for combo in list(_c(words, 2))[:8]:
            _run_fts('{title}: ' + ' '.join(f'"{w}"' for w in combo))

    # (c) Synonym-expanded queries targeting paper_text specifically.
    #     paper_text contains claim vocabulary (Cr, Pb, chitosan, etc.)
    #     that titles/abstracts may omit.
    lower_q = query.lower().strip()
    orig_words = _clean_query_words(query)
    expanded_terms: list[str] = []

    for i in range(len(orig_words) - 1):
        bigram = f"{orig_words[i].lower()} {orig_words[i+1].lower()}"
        expanded_terms.extend(CHEMISTRY_BIGRAM_SYNONYMS.get(bigram, []))

    for w in orig_words:
        expanded_terms.extend(CHEMISTRY_SYNONYMS.get(w.lower(), []))

    if expanded_terms:
        consumed = set()
        for i in range(len(orig_words) - 1):
            bigram = f"{orig_words[i].lower()} {orig_words[i+1].lower()}"
            if bigram in CHEMISTRY_BIGRAM_SYNONYMS:
                consumed.add(orig_words[i].lower())
                consumed.add(orig_words[i+1].lower())
        core_words = [w for w in orig_words if w.lower() not in consumed]
        if not core_words:
            core_words = orig_words[:1]

        core_and_synonyms: list[str] = list(core_words)
        for cw in core_words:
            core_and_synonyms.extend(CHEMISTRY_SYNONYMS.get(cw.lower(), []))

        # Deduplicate expanded terms
        seen_exp = set()
        unique_exp: list[str] = []
        for term in expanded_terms:
            tl = term.lower()
            if tl not in seen_exp and tl not in lower_q:
                seen_exp.add(tl)
                unique_exp.append(term)

        # Batch expansion: pair EACH expanded term with the top core word.
        # Creates one ranked list per expanded term (not per term*core pair).
        primary_core = core_and_synonyms[0] if core_and_synonyms else None
        for term in unique_exp[:12]:
            if primary_core:
                _run_fts(f'{{paper_text}}: "{term}" "{primary_core}"')
            else:
                _run_fts(f'{{paper_text}}: "{term}"')

        # A few all-column queries for top expansion terms
        for term in unique_exp[:6]:
            if primary_core:
                _run_fts(f'"{term}" "{primary_core}"')

    if not ranked_lists:
        return []

    merged = _rrf_merge(ranked_lists, k=10)

    # Citation boost: re-rank papers by RRF * (1 + α·log(1+cites))
    # This surfaces landmark/highly-cited papers that FTS alone may rank low.
    candidate_dois = [doi for doi, _ in merged[:top_k * 3]]
    if candidate_dois:
        ph = ','.join('?' * len(candidate_dois))
        cite_rows = conn.execute(
            f"SELECT doi, citation_count FROM sources WHERE doi IN ({ph})",
            candidate_dois,
        ).fetchall()
        cite_map = {r['doi']: r['citation_count'] or 0 for r in cite_rows}
        max_log = math.log(2 + max(cite_map.values())) if cite_map else 1.0
        CITE_ALPHA = 3.0
        boosted = []
        rrf_map = dict(merged)
        for doi in candidate_dois:
            rrf_score = rrf_map.get(doi, 0.0)
            cites = cite_map.get(doi, 0)
            cite_factor = math.log(1 + cites) / max_log
            boosted.append((doi, rrf_score * (1.0 + CITE_ALPHA * cite_factor)))
        boosted.sort(key=lambda x: -x[1])
        return [doi for doi, _ in boosted[:top_k]]

    return [doi for doi, _ in merged[:top_k]]


def _claim_guided_paper_recall(query: str, conn, top_k: int = 30) -> list[str]:
    """Find papers whose claims match the query, ranked by citation count.

    Unlike _paper_recall (which searches titles/abstracts), this searches
    the claims_fts index to find papers with matching claims, then ranks
    by citation count.  Very effective for queries where many papers match
    the broad topic but the authoritative ones have the highest citations.
    """
    words = _clean_query_words(query)
    if not words:
        return []

    from itertools import combinations
    # Generate FTS queries at decreasing specificity.
    # Cap at ~10 sub-queries to keep latency bounded (each runs a 3-way JOIN).
    fts_queries: list[str] = []
    if len(words) > 1:
        fts_queries.append(' '.join(f'"{w}"' for w in words))
        if len(words) >= 3:
            for combo in list(combinations(words, 3))[:6]:
                fts_queries.append(' '.join(f'"{w}"' for w in combo))
        if len(fts_queries) < 10:
            for combo in list(combinations(words, 2))[:max(0, 10 - len(fts_queries))]:
                fts_queries.append(' '.join(f'"{w}"' for w in combo))
    else:
        fts_queries.append(f'"{words[0]}"')

    import math
    doi_cites: dict[str, int] = {}
    doi_query_count: dict[str, int] = {}
    per_q_limit = top_k * 3

    for fts_q in fts_queries:
        try:
            # Single pass: BM25-ranked with citation info (avoids
            # running two separate 3-way JOINs per sub-query).
            rows = conn.execute(
                "SELECT DISTINCT c.source_doi, s.citation_count "
                "FROM claims_fts f "
                "JOIN claims c ON c.claim_id = f.claim_id "
                "JOIN sources s ON s.doi = c.source_doi "
                "WHERE claims_fts MATCH ? "
                "ORDER BY f.rank "
                "LIMIT ?",
                [fts_q, per_q_limit],
            ).fetchall()
            seen_this_q: set[str] = set()
            for r in rows:
                doi = r['source_doi']
                cites = r['citation_count'] or 0
                if doi not in doi_cites or cites > doi_cites[doi]:
                    doi_cites[doi] = cites
                if doi not in seen_this_q:
                    seen_this_q.add(doi)
                    doi_query_count[doi] = doi_query_count.get(doi, 0) + 1
        except Exception:
            continue

    # Score: query_count² * log(2 + citations).
    # Squaring query_count gives topically-focused papers (matching many
    # sub-queries) a strong boost, while citations still help distinguish
    # authoritative papers.  This allows 0-citation papers with high
    # query breadth to compete.
    scored = [
        (doi, (doi_query_count.get(doi, 1) ** 2) * math.log(2 + doi_cites.get(doi, 0)))
        for doi in doi_cites
    ]
    scored.sort(key=lambda x: -x[1])
    return [doi for doi, _ in scored[:top_k]]


def _get_claims_for_papers(dois: list[str], conn) -> list[dict]:
    """Fetch all claims from a list of paper DOIs."""
    if not dois:
        return []
    all_claims = []
    for i in range(0, len(dois), 999):
        batch = dois[i:i + 999]
        ph = ','.join('?' * len(batch))
        rows = conn.execute(
            f"SELECT claim_id, claim_type, source_doi, source_paper_title, "
            f"confidence, location_in_paper, verbatim_quote, "
            f"extraction_model, data, claim_contextualized "
            f"FROM claims WHERE source_doi IN ({ph})",
            batch,
        ).fetchall()
        for r in rows:
            claim = json.loads(r['data'])
            # DB columns are authoritative — always override JSON values
            for col in ('claim_id', 'claim_type', 'source_doi',
                        'source_paper_title', 'confidence',
                        'location_in_paper', 'verbatim_quote',
                        'extraction_model'):
                if r[col]:
                    claim[col] = r[col]
            if r['claim_contextualized']:
                claim['claim_contextualized'] = r['claim_contextualized']
            all_claims.append(claim)
    return all_claims


def import_from_filesystem(index_dir: Path, batch_size: int = 5000):
    """Import the filesystem-based index into SQLite."""
    init_db()
    path = get_db_path()
    conn = sqlite3.connect(str(path))
    c = conn.cursor()

    # Import views
    views_dir = index_dir / "views"
    if views_dir.exists():
        for vdir in sorted(views_dir.iterdir()):
            if not vdir.is_dir():
                continue
            view_file = vdir / "_view.json"
            if view_file.exists():
                with open(view_file) as f:
                    vdata = json.load(f)
                c.execute(
                    "INSERT OR REPLACE INTO views (view_id, name, description, data) VALUES (?,?,?,?)",
                    (vdata['view_id'], vdata.get('name', ''), vdata.get('description', ''), json.dumps(vdata))
                )
    conn.commit()
    print(f"  Imported views", flush=True)

    # Import sources
    sources_dir = index_dir / "sources"
    if sources_dir.exists():
        batch = []
        count = 0
        for sf in sources_dir.glob("*.json"):
            with open(sf) as f:
                sdata = json.load(f)
            batch.append((
                sdata.get('doi', ''),
                sdata.get('title', ''),
                json.dumps(sdata.get('authors', [])),
                sdata.get('year', 0),
                sdata.get('venue', ''),
                sdata.get('abstract', ''),
                sdata.get('citation_count', 0),
                sdata.get('open_access_url', ''),
                json.dumps(sdata),
            ))
            if len(batch) >= batch_size:
                c.executemany(
                    "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
                    batch
                )
                conn.commit()
                count += len(batch)
                print(f"  Sources: {count:,}", flush=True)
                batch = []
        if batch:
            c.executemany(
                "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            count += len(batch)
        print(f"  Sources total: {count:,}", flush=True)

    # Import claims
    claims_dir = index_dir / "claims"
    if claims_dir.exists():
        batch = []
        fts_batch = []
        count = 0
        for cf in claims_dir.glob("*.json"):
            with open(cf) as f:
                cdata = json.load(f)
            searchable = build_searchable_text(cdata)
            batch.append((
                cdata.get('claim_id', ''),
                cdata.get('claim_type', ''),
                cdata.get('source_doi', ''),
                cdata.get('source_paper_title', ''),
                cdata.get('confidence', ''),
                cdata.get('location_in_paper', ''),
                cdata.get('verbatim_quote', ''),
                cdata.get('extraction_model', ''),
                cdata.get('extraction_version', ''),
                cdata.get('extracted_at', ''),
                json.dumps(cdata.get('view_paths', {})),
                json.dumps(cdata),
            ))
            fts_batch.append((
                cdata.get('claim_id', ''),
                cdata.get('claim_type', ''),
                cdata.get('source_paper_title', ''),
                cdata.get('verbatim_quote', ''),
                searchable,
            ))
            if len(batch) >= batch_size:
                c.executemany(
                    "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch
                )
                conn.commit()
                count += len(batch)
                print(f"  Claims: {count:,}", flush=True)
                batch = []
                fts_batch = []
        if batch:
            c.executemany(
                "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            count += len(batch)
        print(f"  Claims total: {count:,}", flush=True)

    # Rebuild FTS index
    print("  Building full-text search index...", flush=True)
    c.execute("DELETE FROM claims_fts")
    conn.commit()

    rows = c.execute("SELECT data FROM claims").fetchall()
    fts_batch = []
    for row in rows:
        cdata = json.loads(row[0])
        searchable = build_searchable_text(cdata)
        fts_batch.append((
            cdata.get('claim_id', ''),
            cdata.get('claim_type', ''),
            cdata.get('source_paper_title', ''),
            cdata.get('verbatim_quote', ''),
            searchable,
        ))

    for i in range(0, len(fts_batch), batch_size):
        chunk = fts_batch[i:i+batch_size]
        c.executemany("""
            INSERT INTO claims_fts(claim_id, claim_type, source_paper_title, verbatim_quote, searchable_text)
            VALUES (?,?,?,?,?)
        """, chunk)
        conn.commit()
        print(f"  FTS indexed: {min(i+batch_size, len(fts_batch)):,}/{len(fts_batch):,}", flush=True)

    # Import tree nodes
    if views_dir.exists():
        node_count = 0
        for vdir in sorted(views_dir.iterdir()):
            if not vdir.is_dir():
                continue
            view_id = vdir.name
            batch = []
            for node_file in vdir.rglob("_node.json"):
                with open(node_file) as f:
                    ndata = json.load(f)
                rel = node_file.parent.relative_to(vdir)
                path_str = str(rel).replace(os.sep, '/')
                children_dirs = [d.name for d in node_file.parent.iterdir()
                                if d.is_dir() and (d / "_node.json").exists()]
                batch.append((
                    view_id,
                    path_str,
                    ndata.get('name', path_str.split('/')[-1]),
                    ndata.get('level', path_str.count('/') + 1),
                    ndata.get('claim_count', 0),
                    json.dumps(children_dirs),
                    json.dumps(ndata.get('claim_ids', [])[:100]),
                    json.dumps(ndata),
                ))
            if batch:
                c.executemany(
                    "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
                    batch
                )
                conn.commit()
                node_count += len(batch)
        print(f"  Tree nodes total: {node_count:,}", flush=True)

    # Also import root nodes
    for vdir in sorted(views_dir.iterdir()):
        if not vdir.is_dir():
            continue
        view_id = vdir.name
        root_file = vdir / "_root.json"
        if root_file.exists():
            with open(root_file) as f:
                rdata = json.load(f)
            children_dirs = [d.name for d in vdir.iterdir()
                            if d.is_dir() and (d / "_node.json").exists()]
            c.execute(
                "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
                (view_id, '', rdata.get('name', view_id), 0, rdata.get('claim_count', 0),
                 json.dumps(children_dirs), json.dumps([]), json.dumps(rdata))
            )
    conn.commit()

    # Store metadata
    total_claims = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    total_sources = c.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    total_nodes = c.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()[0]

    for k, v in [
        ('total_claims', str(total_claims)),
        ('total_sources', str(total_sources)),
        ('total_nodes', str(total_nodes)),
        ('total_views', '5'),
        ('version', '1.0.0'),
    ]:
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)", (k, v))
    conn.commit()
    conn.close()

    print(f"\n  Database built: {total_claims:,} claims, {total_sources:,} sources, {total_nodes:,} nodes")
    print(f"  Size: {get_db_path().stat().st_size / 1e6:.0f} MB")


def import_from_jsonl(dataset_dir: Path, batch_size: int = 10000):
    """Import the JSONL-based askchem dataset (claims.jsonl + sources.jsonl + hierarchy/) into SQLite."""
    init_db()
    path = get_db_path()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()

    claims_file = dataset_dir / "claims.jsonl"
    sources_file = dataset_dir / "sources.jsonl"
    hierarchy_dir = dataset_dir / "hierarchy"

    # Import sources
    if sources_file.exists():
        batch = []
        count = 0
        with open(sources_file) as f:
            for line in f:
                sdata = json.loads(line)
                batch.append((
                    sdata.get('doi', ''),
                    sdata.get('title', ''),
                    json.dumps(sdata.get('authors', [])),
                    sdata.get('year', 0),
                    sdata.get('venue', ''),
                    sdata.get('abstract', ''),
                    sdata.get('citation_count', 0),
                    sdata.get('open_access_url', ''),
                    json.dumps(sdata),
                ))
                if len(batch) >= batch_size:
                    c.executemany(
                        "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
                        batch
                    )
                    conn.commit()
                    count += len(batch)
                    print(f"  Sources: {count:,}", flush=True)
                    batch = []
        if batch:
            c.executemany(
                "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            count += len(batch)
        print(f"  Sources total: {count:,}", flush=True)

    # Import claims
    if claims_file.exists():
        batch = []
        count = 0
        with open(claims_file) as f:
            for line in f:
                cdata = json.loads(line)
                batch.append((
                    cdata.get('claim_id', ''),
                    cdata.get('claim_type', ''),
                    cdata.get('source_doi', ''),
                    cdata.get('source_paper_title', ''),
                    cdata.get('confidence', ''),
                    cdata.get('location_in_paper', ''),
                    cdata.get('verbatim_quote', ''),
                    cdata.get('extraction_model', ''),
                    cdata.get('extraction_version', ''),
                    cdata.get('extracted_at', ''),
                    json.dumps(cdata.get('view_paths', {})),
                    json.dumps(cdata),
                ))
                if len(batch) >= batch_size:
                    c.executemany(
                        "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        batch
                    )
                    conn.commit()
                    count += len(batch)
                    print(f"  Claims: {count:,}", flush=True)
                    batch = []
        if batch:
            c.executemany(
                "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            count += len(batch)
        print(f"  Claims total: {count:,}", flush=True)

    # Build FTS index
    print("  Building full-text search index...", flush=True)
    c.execute("DELETE FROM claims_fts")
    conn.commit()

    offset = 0
    fts_count = 0
    while True:
        rows = c.execute("SELECT data FROM claims LIMIT ? OFFSET ?", (batch_size, offset)).fetchall()
        if not rows:
            break
        fts_batch = []
        for row in rows:
            cdata = json.loads(row[0])
            searchable = build_searchable_text(cdata)
            fts_batch.append((
                cdata.get('claim_id', ''),
                cdata.get('claim_type', ''),
                cdata.get('source_paper_title', ''),
                cdata.get('verbatim_quote', ''),
                searchable,
            ))
        c.executemany("""
            INSERT INTO claims_fts(claim_id, claim_type, source_paper_title, verbatim_quote, searchable_text)
            VALUES (?,?,?,?,?)
        """, fts_batch)
        conn.commit()
        fts_count += len(fts_batch)
        print(f"  FTS indexed: {fts_count:,}", flush=True)
        offset += batch_size

    # Import hierarchy as tree nodes + views
    if hierarchy_dir.exists():
        node_count = 0
        for hfile in sorted(hierarchy_dir.glob("*.json")):
            with open(hfile) as f:
                vdata = json.load(f)
            view_id = vdata.get('view_id', hfile.stem)
            c.execute(
                "INSERT OR REPLACE INTO views (view_id, name, description, data) VALUES (?,?,?,?)",
                (view_id, vdata.get('name', view_id), vdata.get('description', ''), json.dumps(vdata))
            )
            nodes = vdata.get('nodes', [])
            if not nodes:
                continue
            # Build parent-children map from node_ids
            # node_id format: "by_reaction_type_addition/aldol_addition"
            # path = everything after first "_" following view prefix
            prefix = view_id + "_"
            path_map = {}
            for n in nodes:
                nid = n.get('node_id', '')
                if nid.startswith(prefix):
                    p = nid[len(prefix):]
                else:
                    p = nid
                path_map[p] = n

            # Find children for each node
            batch = []
            for p, n in path_map.items():
                level = p.count('/') + 1
                children = [
                    cp.split('/')[-1] for cp in path_map
                    if cp.startswith(p + '/') and cp.count('/') == p.count('/') + 1
                ]
                batch.append((
                    view_id, p,
                    n.get('name', p.split('/')[-1]),
                    level,
                    n.get('claim_count', 0),
                    json.dumps(children),
                    json.dumps([]),
                    json.dumps(n),
                ))
            # Root node
            top_level = [p for p in path_map if '/' not in p]
            total_claims = sum(path_map[p].get('claim_count', 0) for p in top_level)
            batch.append((
                view_id, '',
                vdata.get('name', view_id),
                0, total_claims,
                json.dumps(top_level),
                json.dumps([]),
                json.dumps({'name': vdata.get('name', view_id), 'view_id': view_id, 'claim_count': total_claims}),
            ))
            if batch:
                c.executemany(
                    "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
                    batch
                )
                conn.commit()
                node_count += len(batch)
        print(f"  Tree nodes total: {node_count:,}", flush=True)
    conn.commit()

    # Store metadata
    total_claims = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    total_sources = c.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    total_nodes = c.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()[0]
    total_views = c.execute("SELECT COUNT(*) FROM views").fetchone()[0]

    for k, v in [
        ('total_claims', str(total_claims)),
        ('total_sources', str(total_sources)),
        ('total_nodes', str(total_nodes)),
        ('total_views', str(total_views)),
        ('version', '1.0.0'),
    ]:
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)", (k, v))
    conn.commit()
    conn.close()

    print(f"\n  Database built: {total_claims:,} claims, {total_sources:,} sources, {total_nodes:,} nodes")
    print(f"  Size: {get_db_path().stat().st_size / 1e6:.0f} MB")


# ── Query functions ──────────────────────────────────────────────────────────

def get_stats() -> dict:
    global _stats_cache, _stats_cache_time
    now = _time.monotonic()
    if _stats_cache is not None and (now - _stats_cache_time) < _STATS_TTL:
        return _stats_cache

    with get_conn() as conn:
        stats = {}
        for row in conn.execute("SELECT key, value FROM metadata"):
            stats[row['key']] = row['value']

        type_counts = {}
        for row in conn.execute("SELECT claim_type, COUNT(*) as cnt FROM claims GROUP BY claim_type ORDER BY cnt DESC"):
            type_counts[row['claim_type']] = row['cnt']
        stats['claim_types'] = type_counts

        year_counts = {}
        for row in conn.execute("SELECT year, COUNT(*) as cnt FROM sources WHERE year > 1900 GROUP BY year ORDER BY year"):
            year_counts[row['year']] = row['cnt']
        stats['year_distribution'] = year_counts

        _stats_cache = stats
        _stats_cache_time = now
        return stats


CHEMISTRY_SYNONYMS = {
    # ── Spectroscopy / characterisation ─────────────────────────────────
    "nmr": ["nuclear magnetic resonance"],
    "nuclear magnetic resonance": ["nmr"],
    "1h nmr": ["proton nmr", "nuclear magnetic resonance"],
    "13c nmr": ["carbon nmr", "nuclear magnetic resonance"],
    "ir": ["infrared spectroscopy"],
    "infrared": ["ir spectroscopy", "ftir"],
    "ftir": ["fourier transform infrared spectroscopy"],
    "raman": ["raman spectroscopy", "raman scattering"],
    "uv-vis": ["ultraviolet visible spectroscopy", "uv visible"],
    "uv": ["ultraviolet", "uv-vis"],
    "ms": ["mass spectrometry"],
    "esi-ms": ["electrospray ionization mass spectrometry"],
    "maldi": ["matrix assisted laser desorption ionization"],
    "hrms": ["high-resolution mass spectrometry"],
    "gc-ms": ["gas chromatography mass spectrometry"],
    "lc-ms": ["liquid chromatography mass spectrometry"],
    "hplc": ["high performance liquid chromatography"],
    "gpc": ["gel permeation chromatography"],
    "sec": ["size exclusion chromatography"],
    "xrd": ["x-ray diffraction"],
    "x-ray diffraction": ["xrd"],
    "pxrd": ["powder x-ray diffraction"],
    "sxrd": ["single-crystal x-ray diffraction"],
    "xrf": ["x-ray fluorescence"],
    "xps": ["x-ray photoelectron spectroscopy"],
    "ups": ["ultraviolet photoelectron spectroscopy"],
    "pes": ["photoelectron spectroscopy"],
    "xanes": ["x-ray absorption near edge structure"],
    "exafs": ["extended x-ray absorption fine structure"],
    "xas": ["x-ray absorption spectroscopy"],
    "edx": ["energy dispersive x-ray spectroscopy"],
    "eds": ["energy dispersive spectroscopy"],
    "saxs": ["small-angle x-ray scattering"],
    "waxs": ["wide-angle x-ray scattering"],
    "dls": ["dynamic light scattering"],
    "sem": ["scanning electron microscopy"],
    "tem": ["transmission electron microscopy"],
    "haadf": ["high-angle annular dark-field imaging"],
    "stem": ["scanning transmission electron microscopy"],
    "afm": ["atomic force microscopy"],
    "stm": ["scanning tunneling microscopy"],
    "epr": ["electron paramagnetic resonance"],
    "esr": ["electron spin resonance"],
    "dsc": ["differential scanning calorimetry"],
    "tga": ["thermogravimetric analysis"],
    "dta": ["differential thermal analysis"],
    "bet": ["brunauer emmett teller", "surface area analysis"],
    "bjh": ["barrett joyner halenda"],
    "icp": ["inductively coupled plasma"],
    "icp-ms": ["inductively coupled plasma mass spectrometry"],
    "icp-oes": ["inductively coupled plasma optical emission spectroscopy"],
    "aas": ["atomic absorption spectroscopy"],
    "cd": ["circular dichroism"],
    "shg": ["second harmonic generation"],

    # ── Computational / theory ──────────────────────────────────────────
    "dft": ["density functional theory"],
    "density functional theory": ["dft"],
    "md": ["molecular dynamics"],
    "qm/mm": ["quantum mechanics molecular mechanics"],
    "tddft": ["time dependent density functional theory"],
    "ccsd": ["coupled cluster"],
    "mp2": ["moller plesset perturbation"],

    # ── Electrochemistry ────────────────────────────────────────────────
    "cv": ["cyclic voltammetry"],
    "lsv": ["linear sweep voltammetry"],
    "ca": ["chronoamperometry"],
    "eis": ["electrochemical impedance spectroscopy"],
    "rde": ["rotating disk electrode"],
    "rrde": ["rotating ring-disk electrode"],
    "dpv": ["differential pulse voltammetry"],
    "rhe": ["reversible hydrogen electrode"],
    "nhe": ["normal hydrogen electrode"],
    "sce": ["saturated calomel electrode"],
    "fe": ["faradaic efficiency"],
    "faradaic efficiency": ["fe"],
    "her": ["hydrogen evolution reaction"],
    "hydrogen evolution": ["her"],
    "oer": ["oxygen evolution reaction"],
    "oxygen evolution": ["oer"],
    "orr": ["oxygen reduction reaction"],
    "oxygen reduction": ["orr"],
    "co2rr": ["co2 reduction reaction", "carbon dioxide reduction"],
    "nrr": ["nitrogen reduction reaction"],

    # ── Named reactions & organic chemistry ─────────────────────────────
    "suzuki": ["suzuki coupling", "suzuki-miyaura", "suzuki miyaura"],
    "heck": ["heck reaction", "mizoroki-heck"],
    "sonogashira": ["sonogashira coupling"],
    "negishi": ["negishi coupling"],
    "stille": ["stille coupling"],
    "kumada": ["kumada coupling"],
    "buchwald": ["buchwald-hartwig", "buchwald hartwig amination"],
    "hartwig": ["buchwald-hartwig amination"],
    "wittig": ["wittig reaction", "wittig olefination"],
    "aldol": ["aldol reaction", "aldol condensation"],
    "mannich": ["mannich reaction"],
    "michael": ["michael addition", "conjugate addition"],
    "friedel-crafts": ["friedel crafts alkylation", "friedel crafts acylation"],
    "diels-alder": ["diels alder cycloaddition", "[4+2] cycloaddition"],
    "metathesis": ["olefin metathesis", "ring-closing metathesis", "romp"],
    "romp": ["ring-opening metathesis polymerization", "metathesis polymerization"],
    "rcm": ["ring-closing metathesis"],
    "adment": ["acyclic diene metathesis"],
    "click": ["click chemistry", "cuaac", "spaac"],
    "cuaac": ["copper-catalyzed azide alkyne cycloaddition", "click"],
    "spaac": ["strain-promoted azide alkyne cycloaddition"],
    "sn1": ["nucleophilic substitution unimolecular"],
    "sn2": ["nucleophilic substitution bimolecular"],
    "e1": ["elimination unimolecular"],
    "e2": ["elimination bimolecular"],
    "c-h": ["c-h activation", "c-h functionalization"],
    "c-n": ["c-n coupling", "amination"],
    "c-o": ["c-o coupling"],

    # ── Ligand / catalyst families ──────────────────────────────────────
    "nhc": ["n-heterocyclic carbene"],
    "binap": ["binaphthyl diphosphine"],
    "binol": ["binaphthol"],
    "duphos": ["duphos ligand"],
    "phox": ["phosphinooxazoline"],
    "pybox": ["pyridine bisoxazoline"],
    "salen": ["salicylidene ethylenediamine"],
    "cp": ["cyclopentadienyl"],
    "cp*": ["pentamethylcyclopentadienyl"],
    "ppy": ["phenylpyridine"],
    "bpy": ["bipyridine"],
    "phen": ["phenanthroline"],
    "dppe": ["diphenylphosphinoethane"],
    "dppf": ["diphenylphosphinoferrocene"],
    "xantphos": ["xantphos ligand"],
    "brookhart": ["brookhart catalyst"],
    "grubbs": ["grubbs catalyst", "olefin metathesis catalyst"],
    "hoveyda-grubbs": ["hoveyda grubbs catalyst"],
    "schrock": ["schrock catalyst"],
    "wilkinson": ["wilkinson catalyst"],

    # ── Materials / substances ─────────────────────────────────────────
    "mof": ["metal-organic framework", "metal organic framework"],
    "metal-organic framework": ["mof"],
    "cof": ["covalent organic framework"],
    "zif": ["zeolitic imidazolate framework"],
    "hof": ["hydrogen-bonded organic framework"],
    "pof": ["porous organic framework"],
    "mxene": ["transition metal carbide", "two-dimensional carbide"],
    "perovskite": ["abx3 structure", "metal halide perovskite"],
    "zeolite": ["aluminosilicate", "microporous crystalline"],
    "aerogel": ["silica aerogel", "porous nanostructure"],
    "hydrogel": ["polymeric hydrogel"],
    "graphene": ["graphene oxide", "reduced graphene oxide", "rgo"],
    "go": ["graphene oxide"],
    "rgo": ["reduced graphene oxide"],
    "cnt": ["carbon nanotube", "mwcnt", "swcnt"],
    "carbon nanotube": ["cnt", "mwcnt", "swcnt"],
    "mwcnt": ["multi-walled carbon nanotube", "carbon nanotube"],
    "swcnt": ["single-walled carbon nanotube", "carbon nanotube"],
    "qd": ["quantum dot"],
    "quantum dot": ["qd", "quantum dots"],
    "np": ["nanoparticle"],
    "nanoparticle": ["np", "nanoparticles", "nanomaterial"],
    "fullerene": ["c60", "buckminsterfullerene"],
    "ionic liquid": ["il", "room temperature ionic liquid"],
    "deep eutectic": ["deep eutectic solvent", "des"],
    "mip": ["molecularly imprinted polymer"],

    # ── Polymers ──────────────────────────────────────────────
    "peg": ["polyethylene glycol"],
    "polyethylene glycol": ["peg"],
    "peo": ["polyethylene oxide"],
    "pla": ["polylactic acid", "polylactide"],
    "pcl": ["polycaprolactone"],
    "ps": ["polystyrene"],
    "pp": ["polypropylene"],
    "pvc": ["polyvinyl chloride"],
    "pvdf": ["polyvinylidene fluoride"],
    "pdms": ["polydimethylsiloxane"],
    "pmma": ["polymethyl methacrylate"],
    "paa": ["polyacrylic acid"],
    "pei": ["polyethylenimine"],
    "pan": ["polyacrylonitrile"],
    "ptfe": ["polytetrafluoroethylene"],

    # ── Biomed / bio ─────────────────────────────────────────────
    "dna": ["deoxyribonucleic acid"],
    "rna": ["ribonucleic acid"],
    "mrna": ["messenger rna"],
    "pcr": ["polymerase chain reaction"],
    "sirna": ["small interfering rna"],
    "elisa": ["enzyme-linked immunosorbent assay"],
    "fret": ["forster resonance energy transfer"],

    # ── Energy / optoelectronics ──────────────────────────────────
    "led": ["light-emitting diode"],
    "oled": ["organic light-emitting diode"],
    "qled": ["quantum dot light-emitting diode"],
    "pled": ["polymer light-emitting diode"],
    "pv": ["photovoltaic", "solar cell"],
    "photovoltaic": ["pv", "solar cell"],
    "dssc": ["dye-sensitized solar cell"],
    "opv": ["organic photovoltaic"],
    "psc": ["perovskite solar cell"],
    "tco": ["transparent conducting oxide"],

    # ── Deposition / processing ──────────────────────────────────
    "ald": ["atomic layer deposition"],
    "cvd": ["chemical vapor deposition"],
    "pvd": ["physical vapor deposition"],
    "pecvd": ["plasma-enhanced chemical vapor deposition"],
    "mocvd": ["metalorganic chemical vapor deposition"],
    "pld": ["pulsed laser deposition"],
    "lb": ["langmuir blodgett"],
    "sam": ["self-assembled monolayer"],

    # ── Gases & small molecules ──────────────────────────────
    "co2": ["carbon dioxide"],
    "carbon dioxide": ["co2"],
    "ch4": ["methane"],
    "n2o": ["nitrous oxide"],
    "nh3": ["ammonia"],
    "h2o": ["water"],
    "h2o2": ["hydrogen peroxide"],
    "h2": ["hydrogen"],
    "o2": ["oxygen"],
    "n2": ["nitrogen"],
    "co": ["carbon monoxide"],
    "no": ["nitric oxide"],
    "no2": ["nitrogen dioxide"],
    "so2": ["sulfur dioxide"],

    # ── Metrics / properties ────────────────────────────────
    "ee": ["enantiomeric excess"],
    "de": ["diastereomeric excess"],
    "tof": ["turnover frequency"],
    "ton": ["turnover number"],
    "iec": ["ion exchange capacity"],
    "toc": ["total organic carbon"],
    "bod": ["biological oxygen demand"],
    "cod": ["chemical oxygen demand"],
    "tds": ["total dissolved solids"],

    # ── Synonymy of chemistry verbs (single-word) ────────────────────
    "adsorption": ["removal", "uptake", "sorption"],
    "removal": ["adsorption", "uptake", "elimination"],
    "uptake": ["adsorption", "sorption"],
    "catalyst": ["catalytic", "photocatalyst", "electrocatalyst"],
    "catalysis": ["catalytic", "catalyzed"],
    "synthesis": ["preparation", "fabrication", "formation"],
    "degradation": ["decomposition", "mineralization", "breakdown"],
    "oxidation": ["oxidized", "oxidative"],
    "reduction": ["reduced", "reductive"],
    "hydrogenation": ["hydrogenate", "reduce with hydrogen"],
    "polymerization": ["polymerisation", "polymerized"],
    "functionalization": ["functionalisation", "functionalized"],
    "photocatalysis": ["photocatalytic"],
    "electrocatalysis": ["electrocatalytic"],

    # ── Pharma / industrial ─────────────────────────────────
    "api": ["active pharmaceutical ingredient"],
    "sar": ["structure activity relationship"],
    "adme": ["absorption distribution metabolism excretion"],
}

CHEMISTRY_BIGRAM_SYNONYMS = {
    # Element-class expansions
    "heavy metal": ["Pb", "Cd", "Cr", "Hg", "Zn", "Cu", "Ni", "As",
                    "lead", "cadmium", "chromium", "mercury", "zinc",
                    "copper", "nickel", "arsenic"],
    "rare earth": ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
                   "Ho", "Er", "Tm", "Yb", "Lu", "lanthanide"],
    "noble metal": ["Au", "Ag", "Pt", "Pd", "Rh", "Ir", "Ru",
                    "gold", "silver", "platinum", "palladium"],
    "transition metal": ["Fe", "Co", "Ni", "Cu", "Zn", "Mn", "Ti", "V", "Cr",
                         "iron", "cobalt", "nickel", "copper"],
    "alkali metal": ["Li", "Na", "K", "Rb", "Cs",
                     "lithium", "sodium", "potassium"],
    "alkaline earth": ["Be", "Mg", "Ca", "Sr", "Ba"],
    "main group": ["B", "Al", "Si", "P", "S", "Ga", "Ge"],

    # Reactions / named processes
    "suzuki coupling": [
        "Suzuki-Miyaura",
        "Suzuki-Miyaura cross-coupling",
        "palladium-catalyzed cross-coupling",
        "aryl boronic acid",
    ],
    "cross coupling": ["Suzuki", "Heck", "Sonogashira", "Buchwald",
                       "Negishi", "Stille", "Kumada", "c-c coupling"],
    "c-c coupling": ["cross-coupling", "Suzuki", "Heck"],
    "c-n coupling": ["Buchwald-Hartwig", "amination"],
    "c-h activation": ["c-h functionalization", "c-h insertion", "c-h bond activation"],
    "c-h functionalization": ["c-h activation", "c-h insertion"],
    "olefin metathesis": ["Grubbs", "ROMP", "RCM", "CM", "ring-opening metathesis"],
    "ring-opening metathesis": ["ROMP", "olefin metathesis"],
    "ring-closing metathesis": ["RCM", "olefin metathesis"],
    "click chemistry": ["CuAAC", "SPAAC", "azide alkyne cycloaddition", "triazole"],
    "aldol reaction": ["aldol condensation", "Mukaiyama aldol", "stereoselective aldol"],
    "diels alder": ["[4+2] cycloaddition", "cycloaddition"],
    "asymmetric catalysis": ["enantioselective", "chiral catalyst", "asymmetric synthesis"],
    "asymmetric synthesis": ["enantioselective synthesis", "chiral synthesis"],
    "photoredox catalysis": ["visible light photoredox", "photocatalysis", "photocatalyst"],
    "single atom": ["SAC", "isolated site", "atomically dispersed", "single-atom catalyst"],
    "single atom catalyst": ["SAC", "isolated active site"],

    # Concepts
    "water splitting": ["hydrogen evolution", "oxygen evolution", "HER", "OER",
                        "photocatalytic H2"],
    "carbon capture": ["CO2 capture", "carbon dioxide capture",
                       "CO2 adsorption", "carbon sequestration"],
    "artificial photosynthesis": ["CO2 reduction", "water splitting", "solar fuels"],
    "greenhouse gas": ["CO2", "CH4", "N2O", "carbon dioxide", "methane"],
    "drug delivery": ["nanocarrier", "drug release", "controlled release",
                      "targeted delivery"],
    "drug design": ["medicinal chemistry", "pharmacophore", "SAR",
                    "structure-activity relationship"],
    "solar cell": ["photovoltaic", "perovskite solar", "dye-sensitized"],
    "perovskite solar": ["PSC", "methylammonium lead iodide", "MAPbI3"],
    "lithium ion": ["Li-ion", "lithium-ion battery", "LIB"],
    "sodium ion": ["Na-ion", "sodium-ion battery", "NIB"],
    "fuel cell": ["PEMFC", "SOFC", "hydrogen fuel cell"],
    "gas sensor": ["chemiresistor", "SnO2 sensor", "semiconductor gas sensor"],

    # Characterisation + spectroscopy contexts
    "powder diffraction": ["XRD", "PXRD", "x-ray diffraction"],
    "surface area": ["BET", "specific surface area"],

    # Materials / classes
    "quantum dot": ["QD", "CdSe", "CdS", "nanocrystal"],
    "ionic liquid": ["IL", "room temperature ionic liquid"],
    "porous material": ["zeolite", "MOF", "COF", "aerogel"],

    # Polymers / bio
    "polymer brush": ["ATRP brush", "controlled polymerization"],
    "cancer therapy": ["tumor", "chemotherapy", "anticancer"],
    "antimicrobial activity": ["antibacterial", "antifungal", "biocidal"],

    # Water / environmental
    "waste water": ["wastewater", "effluent treatment", "water purification"],
    "water treatment": ["wastewater treatment", "purification", "remediation"],
    "air pollution": ["particulate matter", "NOx", "SO2", "VOC"],
}


# Formula ↔ name lookups used by the query processor.  Round-trips in
# both directions so a user typing ``TiO2`` still matches ``titanium
# dioxide`` in claim text, and vice versa.
CHEMISTRY_FORMULAS = {
    "tio2": ["titanium dioxide", "titania"],
    "titanium dioxide": ["tio2", "titania"],
    "zno": ["zinc oxide"],
    "zinc oxide": ["zno"],
    "fe2o3": ["iron oxide", "hematite"],
    "fe3o4": ["magnetite", "iron oxide"],
    "al2o3": ["alumina", "aluminum oxide"],
    "sio2": ["silica", "silicon dioxide"],
    "silicon dioxide": ["sio2", "silica"],
    "cuo": ["copper oxide", "cupric oxide"],
    "cu2o": ["cuprous oxide", "copper oxide"],
    "mno2": ["manganese dioxide"],
    "moo3": ["molybdenum trioxide"],
    "wo3": ["tungsten trioxide"],
    "v2o5": ["vanadium pentoxide"],
    "bi2o3": ["bismuth oxide"],
    "bivo4": ["bismuth vanadate"],
    "srtio3": ["strontium titanate"],
    "srcoo3": ["strontium cobaltite"],
    "latmo3": ["lanthanum manganite"],
    "mapbi3": ["methylammonium lead iodide", "perovskite"],
    "faapbi3": ["formamidinium lead iodide"],
    "mos2": ["molybdenum disulfide"],
    "molybdenum disulfide": ["mos2"],
    "ws2": ["tungsten disulfide"],
    "wse2": ["tungsten diselenide"],
    "mose2": ["molybdenum diselenide"],
    "cds": ["cadmium sulfide"],
    "cdse": ["cadmium selenide"],
    "cdte": ["cadmium telluride"],
    "gan": ["gallium nitride"],
    "inas": ["indium arsenide"],
    "gaas": ["gallium arsenide"],
    "g-c3n4": ["graphitic carbon nitride", "carbon nitride"],
    "c3n4": ["graphitic carbon nitride"],
    "bn": ["boron nitride", "hexagonal boron nitride"],
    "h-bn": ["hexagonal boron nitride"],
    "zif-8": ["zeolitic imidazolate framework 8", "ZIF"],
    "zif-67": ["zeolitic imidazolate framework 67", "ZIF"],
    "uio-66": ["zirconium MOF", "UiO"],
    "mil-101": ["chromium MOF"],
    "hkust-1": ["copper MOF", "cu3btc2"],
}


# Domain-aware plural → singular table.  Runs AFTER stop-word filtering
# and is used purely to widen FTS coverage for common chemistry nouns
# whose plural form is not auto-generated by the unicode61 tokenizer.
CHEMISTRY_PLURALS = {
    "catalysts": "catalyst",
    "reactions": "reaction",
    "materials": "material",
    "frameworks": "framework",
    "particles": "particle",
    "nanoparticles": "nanoparticle",
    "ligands": "ligand",
    "polymers": "polymer",
    "compounds": "compound",
    "complexes": "complex",
    "electrodes": "electrode",
    "substrates": "substrate",
    "solvents": "solvent",
    "reagents": "reagent",
    "products": "product",
    "reactants": "reactant",
    "crystals": "crystal",
    "films": "film",
    "surfaces": "surface",
    "membranes": "membrane",
    "electrolytes": "electrolyte",
    "oxides": "oxide",
    "sulfides": "sulfide",
    "nitrides": "nitride",
    "carbides": "carbide",
    "halides": "halide",
    "amines": "amine",
    "alcohols": "alcohol",
    "acids": "acid",
    "aldehydes": "aldehyde",
    "ketones": "ketone",
    "esters": "ester",
    "ethers": "ether",
    "phosphines": "phosphine",
    "carbenes": "carbene",
    "fullerenes": "fullerene",
    "nanotubes": "nanotube",
    "heterocycles": "heterocycle",
    "applications": "application",
    "properties": "property",
    "structures": "structure",
    "mechanisms": "mechanism",
}


def _expand_synonyms(query: str) -> str:
    """Expand chemistry abbreviations/synonyms/formulas in the query.

    Runs against four tables:
      * CHEMISTRY_SYNONYMS      (unigram abbreviations + verbs)
      * CHEMISTRY_BIGRAM_SYNONYMS (bigrams: "heavy metal", "cross coupling", …)
      * CHEMISTRY_FORMULAS      (formula ↔ common name, both directions)
      * CHEMISTRY_PLURALS       (catalysts → catalyst etc.)

    All lookups are lowercase-normalised *after* Unicode cleanup so that
    queries like ``TiO₂`` and ``C–H`` reach the dictionaries as ``tio2``
    and ``c-h``.  Returns an OR-joined FTS5 clause of all expansions, or
    the empty string if nothing matched.
    """
    norm_q = _normalize_query_text(query).lower().strip()
    expansions: list[str] = []

    full_exp = CHEMISTRY_SYNONYMS.get(norm_q, [])
    if full_exp:
        expansions.extend(full_exp)
    full_exp = CHEMISTRY_FORMULAS.get(norm_q, [])
    if full_exp:
        expansions.extend(full_exp)

    words = [w.strip(_PUNCT_STRIP) for w in norm_q.split() if w.strip(_PUNCT_STRIP)]

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        for exp in CHEMISTRY_BIGRAM_SYNONYMS.get(bigram, []):
            if exp.lower() not in norm_q:
                expansions.append(exp)

    for w in words:
        for exp in CHEMISTRY_SYNONYMS.get(w, []):
            if exp.lower() not in norm_q:
                expansions.append(exp)
        for exp in CHEMISTRY_FORMULAS.get(w, []):
            if exp.lower() not in norm_q:
                expansions.append(exp)
        sing = CHEMISTRY_PLURALS.get(w)
        if sing and sing not in norm_q:
            expansions.append(sing)

    if not expansions:
        return ""

    seen = set()
    unique = []
    for e in expansions:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    parts = [f'"{e}"' for e in unique[:10]]
    return ' OR '.join(parts)


STOP_WORDS = frozenset({
    # Grammatical
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "how", "why", "when", "where", "which", "who", "whom",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "have", "has", "had", "it", "its", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "if", "then", "than", "so", "very",
    "about", "between", "through", "during", "before", "after",
    "i", "we", "you", "he", "she", "they", "me", "us", "my", "your",
    # Common scientific-paper filler (kept out of FTS but still
    # available to embedder because the raw query is what it sees)
    "role", "via", "some", "recent", "novel", "several",
    "respectively", "herein", "paper", "papers", "article", "articles",
    "review", "reviews", "discuss", "discussed", "summarize", "summary",
    "method", "methods", "approach", "approaches",
    "applications", "application", "property", "properties",
    "performance", "show", "shows", "shown", "present", "presents",
    "report", "reports", "reported", "describe", "described",
    "demonstrated", "examined",
})


_UNICODE_SUB_MAP = str.maketrans({
    # Subscript digits
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    # Superscript digits
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
    # Dashes → ASCII hyphen
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    # Degree, prime, interpunct, thin space, nbsp → space
    "\u00b0": " ", "\u2032": " ", "\u2033": " ", "\u00b7": " ",
    "\u2022": " ", "\u2009": " ", "\u00a0": " ",
    # Mathematical letters some journals use
    "\u03bc": "u",  # µ → u (micro)
    "\u00d7": "x",
})

_PUNCT_STRIP = ".,;:?!\"'()[]{}<>`\u00bf\u00a1"


def _normalize_query_text(query: str) -> str:
    """NFKC normalise + substitute common Unicode marks for ASCII so that
    tokens like ``H\u2082``, ``C\u2013H``, ``25 \u00b0C`` survive tokenisation.

    The returned string is safe to split on whitespace and to index into
    FTS5 tables that were built from ASCII-normalised text.
    """
    if not query:
        return ""
    import unicodedata
    q = unicodedata.normalize("NFKC", query)
    q = q.translate(_UNICODE_SUB_MAP)
    return q


def _clean_query_words(query: str) -> list[str]:
    """Normalise Unicode, strip punctuation and stop words, and return the
    remaining tokens.  Never returns an empty list — if every token is a
    stop word, the normalised (but still filtered) token list is returned
    verbatim.
    """
    if not query:
        return []
    q = _normalize_query_text(query).replace('"', '""').strip()
    tokens: list[str] = []
    for raw in q.split():
        t = raw.strip(_PUNCT_STRIP)
        if not t:
            continue
        tokens.append(t)
    filtered = [t for t in tokens if t.lower() not in STOP_WORDS]
    return filtered if filtered else tokens


def _build_fts_queries(query: str, mode: str = "auto") -> list[str]:
    """Build FTS5 queries from most specific to broadest.

    Generates multiple query granularities so that broad queries with many
    terms (e.g. 6-word queries) also produce targeted sub-queries from
    term pairs and triples.  All queries are run and merged by the caller.

    ``mode`` exposes Paperclip-style explicit query operators on top of
    the default cascade:

    * ``"phrase"`` — only the exact-phrase query.  Strict; useful when the
      caller wants the exact word order preserved.
    * ``"all"`` — only the AND-of-all-terms query (FTS5 implicit AND when
      space-separating quoted tokens).  Strict but permissive on order.
    * ``"any"`` — OR over the terms.  Broadest; useful as a recall escape
      hatch.
    * ``"auto"`` (default) — the historic cascade (phrase → NEAR → AND →
      combinations → plural → synonyms).
    """
    words = _clean_query_words(query)
    queries: list[str] = []
    if not words:
        return queries

    mode = (mode or "auto").lower()
    if mode == "phrase":
        if len(words) > 1:
            queries.append(f'"{" ".join(words)}"')
        else:
            queries.append(f'"{words[0]}"')
        return queries
    if mode == "all":
        queries.append(' '.join(f'"{w}"' for w in words))
        return queries
    if mode == "any":
        queries.append(' OR '.join(f'"{w}"' for w in words))
        return queries

    if len(words) > 1:
        queries.append(f'"{" ".join(words)}"')       # exact phrase
        queries.append(f'NEAR({" ".join(words)}, 5)') # proximity
        queries.append(' '.join(f'"{w}"' for w in words))  # AND of all terms

        # For long queries (4+ terms), generate sub-queries from 3-term
        # combinations.  This ensures claims matching e.g. "CO2 reduction
        # overpotential" are found even when the full AND query is too
        # restrictive.  Cap at 8 to keep query count manageable.
        if len(words) >= 4:
            from itertools import combinations
            for combo in list(combinations(words, 3))[:8]:
                queries.append(' '.join(f'"{w}"' for w in combo))

        # Plural-normalised variant for domain nouns.  Cheap insurance
        # against the FTS tokenizer not stemming (pre-Tier-C rebuilds).
        singular = [CHEMISTRY_PLURALS.get(w.lower(), w) for w in words]
        if singular != list(words):
            queries.append(' '.join(f'"{w}"' for w in singular))
    else:
        w = words[0]
        queries.append(f'"{w}"')   # exact match
        queries.append(f'{w}*')    # prefix match
        sing = CHEMISTRY_PLURALS.get(w.lower())
        if sing and sing != w:
            queries.append(f'"{sing}"')

    synonym_q = _expand_synonyms(query)
    if synonym_q:
        queries.append(synonym_q)

    return queries


def expand_query_variants(query: str) -> list[str]:
    """Generate expanded query variants using the dictionary tables.

    Returns a list of query strings (including the original) that cover
    vocabulary gaps — e.g. "heavy metal adsorption" also generates
    "Pb Cd Cr Hg adsorption" so FTS/vector can find specific-metal papers.
    Also produces a formula-expanded variant (``TiO2`` → ``titanium
    dioxide``) and, where applicable, a plural-normalised variant.

    All lookups operate on the Unicode-normalised query so that typed
    forms like ``TiO₂`` or ``C–H`` reach the dictionaries.
    """
    norm_q = _normalize_query_text(query).strip()
    lower_q = norm_q.lower()
    variants = [query]
    if norm_q and norm_q != query:
        variants.append(norm_q)

    words = [w.strip(_PUNCT_STRIP) for w in lower_q.split() if w.strip(_PUNCT_STRIP)]
    expanded_terms: list[str] = []

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        exps = CHEMISTRY_BIGRAM_SYNONYMS.get(bigram, [])
        if exps:
            expanded_terms.extend(exps[:8])

    for w in words:
        exps = CHEMISTRY_SYNONYMS.get(w, [])
        if exps:
            expanded_terms.extend(exps[:3])
        formula_exps = CHEMISTRY_FORMULAS.get(w, [])
        if formula_exps:
            expanded_terms.extend(formula_exps[:3])

    if expanded_terms:
        # Preserve every original word so the variant still carries the
        # user's exact anchor terms; the expansions are ADDED not substituted.
        seen = {w.lower() for w in words}
        unique_exp = []
        for t in expanded_terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                unique_exp.append(t)
        variants.append(' '.join(list(words) + unique_exp[:10]))

    if any(w in CHEMISTRY_PLURALS for w in words):
        singularised = [CHEMISTRY_PLURALS.get(w, w) for w in words]
        variant = ' '.join(singularised)
        if variant not in variants:
            variants.append(variant)

    # PAW expand_query variant (Phase 2 wiring, May-23 ft rollout).
    #
    # Until this PR, paw_functions.expand_query produced synonym lists but
    # nothing in db.search_claims read them — the May-14 PAW-off ablation
    # was flat at nDCG@10 partly because the only PAW touch point reaching
    # ranking was normalize_query's 0-hit rescue.  Gating on
    # CHEMTREE_PAW_REWRITES keeps prod safe (default off) while the A/B
    # is in flight; ``paw_functions._check_paw()`` short-circuits the call
    # entirely when ``CHEMTREE_DISABLE_PAW=1`` so the two kill-switches
    # compose correctly.
    if os.environ.get("CHEMTREE_PAW_REWRITES", "0") == "1":
        try:
            from askchem.paw_functions import expand_query as _paw_expand
            paw_terms = _paw_expand(query)
            if paw_terms:
                anchor_words = list(words) if words else norm_q.split()
                seen_anchor = {w.lower() for w in anchor_words}
                paw_unique: list[str] = []
                for t in paw_terms[:10]:
                    key = t.lower().strip()
                    if key and key not in seen_anchor:
                        seen_anchor.add(key)
                        paw_unique.append(t)
                if paw_unique:
                    variants.append(' '.join(anchor_words + paw_unique))
        except Exception:
            pass

    seen_v: set[str] = set()
    dedup: list[str] = []
    for v in variants:
        key = v.lower().strip()
        if key and key not in seen_v:
            seen_v.add(key)
            dedup.append(v)
    return dedup


def _multi_signal_score(claims: list[dict], bm25_scores: dict[str, float],
                        conn) -> list[dict]:
    """Re-score claims using multiple signals: BM25 + citations + confidence + key_result.

    Returns claims sorted by final score, each annotated with _relevance_score.
    """
    import math

    if not claims:
        return claims

    # Fetch citation counts for all source DOIs
    dois = list({c.get('source_doi', '') for c in claims if c.get('source_doi')})
    cite_map = {}
    if dois:
        for i in range(0, len(dois), 999):
            batch = dois[i:i+999]
            ph = ','.join('?' * len(batch))
            rows = conn.execute(
                f"SELECT doi, citation_count FROM sources WHERE doi IN ({ph})", batch
            ).fetchall()
            for r in rows:
                cite_map[r['doi']] = r['citation_count'] or 0

    max_cite = max(cite_map.values()) if cite_map else 1
    bm25_vals = list(bm25_scores.values()) if bm25_scores else [0]
    min_bm25 = min(bm25_vals)
    max_bm25 = max(bm25_vals)
    bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0

    confidence_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}

    for claim in claims:
        cid = claim.get('claim_id', '')
        raw_bm25 = bm25_scores.get(cid, min_bm25)
        bm25_norm = (raw_bm25 - min_bm25) / bm25_range

        cites = cite_map.get(claim.get('source_doi', ''), 0)
        cite_score = math.log(1 + cites) / math.log(2 + max_cite)

        conf = confidence_map.get(claim.get('confidence', ''), 0.5)
        key_res = 1.0 if claim.get('is_key_result') else 0.5

        score = (bm25_norm * 0.50) + (cite_score * 0.25) + (conf * 0.15) + (key_res * 0.10)
        claim['_relevance_score'] = round(score, 3)

    claims.sort(key=lambda c: c.get('_relevance_score', 0), reverse=True)
    return claims


def _run_fts_cascade(fts_queries: list[str], claim_type: str | None,
                     candidate_limit: int, conn) -> tuple[list, str | None]:
    """Run FTS queries in cascade, merging results via RRF.

    Runs up to `max_queries` queries (including sub-queries for broad
    coverage) and merges via RRF.  Stops early if a single tight query
    returns enough candidates.
    """
    ranked_lists: list[list[str]] = []
    used_fts_q = None
    per_q_limit = min(candidate_limit, 300)
    max_queries = 12

    for fts_q in fts_queries[:max_queries]:
        try:
            fts_rows = conn.execute(
                "SELECT claim_id, rank FROM claims_fts "
                "WHERE claims_fts MATCH ? ORDER BY rank LIMIT ?",
                [fts_q, per_q_limit],
            ).fetchall()
        except sqlite3.OperationalError:
            continue
        if not fts_rows:
            continue
        if used_fts_q is None:
            used_fts_q = fts_q
        ranked_lists.append([r['claim_id'] for r in fts_rows])

    if not ranked_lists:
        return [], None

    merged = _rrf_merge(ranked_lists, k=60)
    top_ids = [cid for cid, _ in merged[:candidate_limit]]

    if claim_type:
        ph = ','.join('?' * len(top_ids))
        data_rows = conn.execute(
            f"SELECT claim_id, source_doi, data, claim_contextualized "
            f"FROM claims "
            f"WHERE claim_id IN ({ph}) AND claim_type = ?",
            top_ids + [claim_type],
        ).fetchall()
    else:
        ph = ','.join('?' * len(top_ids))
        data_rows = conn.execute(
            f"SELECT claim_id, source_doi, data, claim_contextualized "
            f"FROM claims WHERE claim_id IN ({ph})",
            top_ids,
        ).fetchall()

    data_map = {dr['claim_id']: dr for dr in data_rows}
    rrf_map = dict(merged)

    rows = []
    for cid in top_ids:
        if cid in data_map:
            dr = data_map[cid]
            rows.append({
                'claim_id': cid,
                'source_doi': dr['source_doi'],
                'data': dr['data'],
                'claim_contextualized': dr['claim_contextualized'],
                'rank': -rrf_map.get(cid, 0),
            })
    rows.sort(key=lambda r: r['rank'])
    return rows, used_fts_q


def _rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: merge multiple ranked ID lists into one.

    Each item gets score = sum(1 / (k + rank_i)) across all lists it appears in.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


_PRF_STOP = STOP_WORDS | frozenset({
    'using', 'used', 'based', 'study', 'effect', 'effects',
    'high', 'low', 'new', 'novel', 'results', 'analysis',
    'via', 'use', 'show', 'shows', 'shown', 'also',
    'different', 'various', 'two', 'three', 'one', 'first',
    'may', 'however', 'well', 'such', 'due', 'found',
    'total', 'overall', 'other', 'more', 'less', 'most',
    'highly', 'significantly', 'respectively', 'compared',
    'properties', 'performance', 'activity', 'applications',
    'method', 'approach', 'recent', 'review', 'studies',
    'research', 'work', 'paper', 'data', 'experimental',
    'conditions', 'process', 'system', 'systems', 'material',
    'materials', 'structure', 'structures', 'efficient',
    'enhanced', 'improved', 'excellent', 'superior', 'good',
    'large', 'small', 'increasing', 'decreasing', 'between',
    'surface', 'reaction', 'reactions', 'catalysts', 'energy',
    'temperature', 'time', 'rate', 'value', 'values',
    'sample', 'samples', 'type', 'types', 'range',
})

_ELEMENT_SYMBOLS = frozenset({
    'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne',
    'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca',
    'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn',
    'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y', 'zr',
    'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn',
    'sb', 'te', 'i', 'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd',
    'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb',
    'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg',
    'tl', 'pb', 'bi', 'po', 'at', 'rn',
})


def _pseudo_relevance_feedback(
    query: str,
    top_claim_ids: list[str],
    paper_dois: list[str],
    conn,
    *,
    top_n: int = 30,
    max_expansion_queries: int = 8,
) -> list[list[str]]:
    """Pseudo-Relevance Feedback: extract distinctive terms from initial
    results and generate expansion FTS queries automatically.

    Uses multiple anchor words and prioritises chemical-specific terms
    (element symbols, material names) over generic words.

    Returns a list of ranked claim-ID lists (one per expansion query),
    suitable for feeding into the main RRF merge.
    """
    import re
    from collections import Counter

    query_words_raw = re.findall(r'[a-zA-Z0-9]+', query.lower())
    query_words = set(query_words_raw)
    query_prefixes = {w[:4] for w in query_words if len(w) >= 4}

    titles: list[str] = []
    if paper_dois:
        for i in range(0, len(paper_dois), 999):
            batch = paper_dois[i:i + 999]
            ph = ','.join('?' * len(batch))
            rows = conn.execute(
                f"SELECT title FROM sources WHERE doi IN ({ph})", batch
            ).fetchall()
            titles.extend(r['title'] for r in rows if r['title'])

    claim_texts: list[str] = []
    use_ids = top_claim_ids[:top_n]
    if use_ids:
        for i in range(0, len(use_ids), 999):
            batch = use_ids[i:i + 999]
            ph = ','.join('?' * len(batch))
            rows = conn.execute(
                f"SELECT verbatim_quote, source_paper_title "
                f"FROM claims WHERE claim_id IN ({ph})", batch
            ).fetchall()
            for r in rows:
                if r['verbatim_quote']:
                    claim_texts.append(r['verbatim_quote'])
                if r['source_paper_title']:
                    titles.append(r['source_paper_title'])

    if not titles and not claim_texts:
        return []

    term_freq = Counter()
    all_docs = titles + claim_texts
    for doc in all_docs:
        doc_terms = set(re.findall(r'[a-zA-Z0-9]{2,}', doc.lower()))
        for t in doc_terms:
            if t in _PRF_STOP or t in query_words:
                continue
            prefix = t[:4] if len(t) >= 4 else t
            if prefix in query_prefixes:
                continue
            term_freq[t] += 1

    if not term_freq:
        return []

    def _term_score(term: str, freq: int) -> float:
        """Score expansion terms: boost element symbols and short
        chemistry-specific terms that are underrepresented in generic
        frequency counts but high-signal for retrieval."""
        score = freq
        if term in _ELEMENT_SYMBOLS and len(term) <= 3:
            score *= 4.0
        elif len(term) <= 5:
            score *= 1.5
        return score

    scored = [
        (t, _term_score(t, f))
        for t, f in term_freq.items()
        if f >= 2 and len(t) >= 2
    ]
    scored.sort(key=lambda x: -x[1])
    expansion_terms = [t for t, _ in scored[:20]]

    if not expansion_terms:
        return []

    anchors = [
        w for w in query_words_raw
        if w not in STOP_WORDS and len(w) >= 2
    ]
    if len(anchors) > 3:
        anchors = sorted(set(anchors), key=lambda w: -len(w))[:3]

    expansion_queries: list[str] = []
    used = set()

    for term in expansion_terms:
        if len(expansion_queries) >= max_expansion_queries:
            break
        for anchor in anchors:
            pair = frozenset((anchor, term))
            if pair in used:
                continue
            used.add(pair)
            fts_q = f'"{anchor}" "{term}"'
            expansion_queries.append(fts_q)
            break

    remaining_terms = [t for t in expansion_terms if t not in
                       {t for fts_q in expansion_queries
                        for t in re.findall(r'"(\w+)"', fts_q)}]
    for i in range(0, len(remaining_terms) - 1, 2):
        if len(expansion_queries) >= max_expansion_queries:
            break
        t1, t2 = remaining_terms[i], remaining_terms[i + 1]
        fts_q = f'"{t1}" "{t2}"'
        expansion_queries.append(fts_q)

    ranked_lists: list[list[str]] = []
    per_q_limit = 200
    for fts_q in expansion_queries:
        try:
            rows = conn.execute(
                "SELECT claim_id FROM claims_fts "
                "WHERE claims_fts MATCH ? ORDER BY rank LIMIT ?",
                [fts_q, per_q_limit],
            ).fetchall()
            if rows:
                ranked_lists.append([r['claim_id'] for r in rows])
        except Exception:
            continue

    return ranked_lists


_AUTHOR_PREFIX_RE = re.compile(
    r'^\s*(?:papers?|publications?|articles?|works?)?\s*'
    r'(?:by|from|author(?:ed)?\s*(?:by)?|author[:\s])\s+(.+)$',
    re.IGNORECASE,
)
_CAPITALIZED_NAME_RE = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){0,3}$')
_GENERIC_NAME_FRAGMENTS = frozenset({
    'chemistry', 'catalysis', 'reactions', 'synthesis', 'chemical',
    'materials', 'paper', 'papers', 'author', 'authors', 'research',
})


def _extract_author_name(query: str) -> str | None:
    """Return a plausible author-name substring if the query looks like an
    author lookup; else None.  Filters out generic chemistry terms that
    happen to match a capitalized-phrase shape.
    """
    q = query.strip()
    if not q:
        return None
    m = _AUTHOR_PREFIX_RE.match(q)
    if m:
        name = m.group(1).strip().strip('"\'')
        if name and len(name) >= 2:
            return name
    if _CAPITALIZED_NAME_RE.match(q):
        tokens = [t.lower().strip('.') for t in q.split()]
        if not any(t in _GENERIC_NAME_FRAGMENTS for t in tokens):
            return q
    return None


def _author_matches(author_str: str, given: list[str], surname: str) -> bool:
    """Check whether any single author in the JSON-encoded authors list
    matches the given surname AND every given-name token (in full form or
    as a leading initial).  Running per-author (not per-string) prevents
    false positives when two authors of one paper together spell the name.
    """
    try:
        authors = json.loads(author_str) if author_str else []
    except Exception:
        return False
    surname_l = surname.lower()
    given_l = [g.lower() for g in given]
    for a in authors:
        al = (a or '').lower()
        if surname_l not in al:
            continue
        if not given_l:
            return True
        ok = True
        for g in given_l:
            if g in al:
                continue
            init = g[0]
            if f' {init}.' in f' {al}' or f' {init} ' in f' {al} ' or \
               al.startswith(f'{init}.') or al.startswith(f'{init} '):
                continue
            ok = False
            break
        if ok:
            return True
    return False


def _author_recall(author_query: str, conn, top_k: int = 60) -> tuple[list[str], set[str]]:
    """Find claim_ids whose source paper has an author matching the query.

    Two-phase:
      1. Broad SQL LIKE filter using both full-name and initial-form
         patterns (e.g. "Stephen Buchwald" → ``%Stephen%Buchwald%`` and
         ``%S. Buchwald%``).
      2. Per-row verification in Python by JSON-parsing the authors array
         and confirming at least one author satisfies the surname plus
         all given-name tokens (initials allowed).

    Returns ``(ranked_claim_ids, matched_doi_set)``.  The DOI set is used
    downstream as a hard filter when the query is unambiguously an author
    lookup.
    """
    if not author_query:
        return [], set()
    raw = [t.strip('.,;:') for t in author_query.split() if t.strip('.,;:')]
    tokens = [t for t in raw if t]
    if not tokens:
        return [], set()
    surname = tokens[-1]
    given = tokens[:-1]
    patterns: set[str] = set()
    patterns.add(f"%{surname}%")
    if given:
        patterns.add(f"%{' '.join(given)}%{surname}%")
        init_chain = ''.join(f"{g[0].upper()}. " for g in given)
        patterns.add(f"%{init_chain}{surname}%")
        patterns.add(f"%{given[0][0].upper()}. {surname}%")
        patterns.add(f"%{given[0][0].upper()} {surname}%")
    clauses = ' OR '.join('authors LIKE ?' for _ in patterns)
    params = list(patterns)
    try:
        rows = conn.execute(
            f"SELECT doi, authors, citation_count FROM sources "
            f"WHERE authors IS NOT NULL AND ({clauses}) "
            f"ORDER BY CASE WHEN citation_count IS NULL THEN 0 "
            f"ELSE citation_count END DESC LIMIT 2000",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return [], set()
    dois: list[str] = []
    for r in rows:
        if _author_matches(r['authors'], given, surname):
            dois.append(r['doi'])
    if not dois:
        return [], set()
    doi_set = {d.lower() for d in dois}
    ranked: list[str] = []
    by_doi: dict[str, list[str]] = {}
    for i in range(0, len(dois), 999):
        batch = dois[i:i + 999]
        ph = ','.join('?' * len(batch))
        claim_rows = conn.execute(
            f"SELECT claim_id, source_doi FROM claims "
            f"WHERE source_doi IN ({ph})",
            batch,
        ).fetchall()
        for cr in claim_rows:
            by_doi.setdefault(cr['source_doi'], []).append(cr['claim_id'])
    for doi in dois:
        for cid in by_doi.get(doi, [])[:3]:
            ranked.append(cid)
            if len(ranked) >= top_k:
                return ranked, doi_set
    return ranked, doi_set


def query_signals_organic_cross_coupling(query: str) -> bool:
    """True when the query clearly refers to a named organic cross-coupling.

    Backbone of two δ-era bandaids: the ``server.py`` intent override
    (forces ``reaction`` on PAW's ``method`` label) and the
    ``by_technique`` noise stripper below. Set
    ``CHEMTREE_DISABLE_COUPLING_INTENT_OVERRIDE=1`` to bypass the intent
    override and ``CHEMTREE_DISABLE_TECHNIQUE_STRIPPER=1`` to bypass the
    by_technique filter; either knob disables only its own consumer.
    """
    ql = (query or "").lower()
    markers = (
        "suzuki", "heck", "sonogashira", "negishi", "stille", "kumada",
        "buchwald", "hartwig", "miyaura",
        "cross-coupling", "cross coupling",
    )
    return any(m in ql for m in markers)


# "Strong" organic-coupling markers — at least one must appear in the
# claim's text or title for a by_technique hit to survive when the user
# is searching for an organic cross-coupling.
#
# Words like "cross-coupling", "biphenyl", or "phosphine" by themselves
# show up in plenty of condensed-matter / antenna / electromagnetics
# papers ("cross-polarization coupling", "biphenyl dithiol junction")
# so they intentionally don't qualify on their own.
_ORGANIC_COUPLING_MARKERS: tuple[str, ...] = (
    # Named reactions / scientists
    "suzuki", "heck", "sonogashira", "negishi", "stille", "kumada",
    "buchwald", "hartwig", "miyaura",
    # Palladium / nickel cross-coupling chemistry
    "palladium-cat", "pd-cat", "pd(0)", "pd(ii)", "pd(0)/pd(ii)",
    "ni-cat", "nickel-cat",
    # Substrate / coupling-partner classes
    "aryl halide", "aryl bromide", "aryl chloride", "aryl iodide",
    "aryl triflate", "vinyl halide",
    "boronic acid", "boronic ester", "boronate", "trifluoroborate",
    # Mechanistic terms specific to organometallic cross-coupling
    "oxidative addition", "transmetalation", "transmetalat",
    "reductive elimination",
    # Distinctive ligand families
    "binap", "xphos", "buchwald ligand",
)

_CLAIM_TEXT_FIELDS: tuple[str, ...] = (
    "reaction_type", "verbatim_quote", "technique_name",
    "what_it_achieves", "process_described", "hypothesis_text",
    "subject", "property_name", "comparison_result", "finding_text",
    "limitation_text", "direction_text", "key_innovation",
)


def _claim_text_blob(claim: dict) -> str:
    return " ".join(
        str(claim.get(k) or "") for k in _CLAIM_TEXT_FIELDS
    ).lower()


def _technique_claim_is_irrelevant_for_coupling_query(claim: dict) -> bool:
    """Drop a by_technique hit when an organic-coupling query matched it
    only via the generic word "coupling" (spin coupling, exciton coupling,
    light-matter coupling, …) and the claim has no organic-coupling marker.

    Looking at the title in addition to claim fields — many condensed-matter
    papers say "Suzuki" only in the body, but the claim-level text never
    mentions any organic-coupling marker, which is the real signal here.
    """
    segs = (claim.get("view_paths") or {}).get("by_technique")
    if not segs or not isinstance(segs, list):
        return False
    blob = _claim_text_blob(claim)
    blob += " " + (claim.get("source_paper_title") or "").lower()
    return not any(m in blob for m in _ORGANIC_COUPLING_MARKERS)


def search_time_distribution(query: str) -> dict:
    """Complete temporal histogram for a query, over ALL matched papers.

    Unlike ``search_claims`` (ranked + capped at 500), this aggregates every
    paper whose title/abstract/claim-text matches the query terms, grouped by
    publication year/decade via a single FTS5 GROUP BY. Used by the search
    "Time" view and the topic-views figure so the decade counts reflect the
    full matched corpus rather than the top-K sample.

    Returns ``{"decades": {"2020s": n, ...}, "years": {2021: m, ...},
    "total_papers": N}``.
    """
    empty = {"decades": {}, "years": {}, "total_papers": 0}
    qs = _build_fts_queries(query, mode="all") or _build_fts_queries(query, mode="any")
    if not qs:
        return empty
    fts_q = qs[0]
    try:
        with get_conn() as c:
            rows = c.execute(
                "SELECT s.year AS y, COUNT(DISTINCT f.doi) AS n "
                "FROM sources_fts f JOIN sources s ON s.doi = f.doi "
                "WHERE sources_fts MATCH ? AND s.year IS NOT NULL "
                "GROUP BY s.year",
                [fts_q],
            ).fetchall()
    except sqlite3.OperationalError:
        return empty
    years: dict[int, int] = {}
    decades: dict[str, int] = {}
    total = 0
    for r in rows:
        try:
            y = int(r["y"])
        except (TypeError, ValueError):
            continue
        if not (1900 <= y <= 2029):
            continue
        n = int(r["n"] or 0)
        years[y] = years.get(y, 0) + n
        dkey = f"{(y // 10) * 10}s"
        decades[dkey] = decades.get(dkey, 0) + n
        total += n
    return {"decades": decades, "years": years, "total_papers": total}


def search_claims(query: str, claim_type: str = None, view: str = None,
                  limit: int = 50, offset: int = 0,
                  use_semantic: bool = True,
                  mode: str = "auto", sort: str = "relevance",
                  _trace_into: dict | None = None) -> dict:
    """Hybrid search with paper-level recall and tree-based BFS recall.

    Pipeline:
      ① Query expansion (synonym/bigram → variant queries)
      ①b Tree recall (match query to taxonomy nodes, BFS subtree claims)
      ①c Author recall (when query looks like an author lookup)
      ② Paper-level recall (sources_fts on titles/abstracts → top-K papers)
      ③ Claim-level recall (FTS5 + vector on claim embeddings)
      ④ RRF merge across all signals (FTS + vector + tree + author)
      ⑤ Paper diversity injection
      ⑥ Per-paper cap (no paper dominates top-K)
      ⑦ Final ranked + paginated results

    ``mode`` (``auto`` / ``phrase`` / ``all`` / ``any``) tightens or
    loosens the FTS5 query construction — see ``_build_fts_queries``.
    Non-auto modes skip the cascade and use a single explicit FTS query
    shape; vector + tree + paper signals still contribute, so this is a
    soft preference rather than a hard restriction.

    ``sort`` (``relevance`` / ``date``) controls the final ordering: the
    default is the existing relevance score; ``date`` re-sorts by
    ``source_year`` descending (claims without a year drop to the tail,
    relevance breaks ties).
    """
    mode = (mode or "auto").lower()
    if mode not in ("auto", "phrase", "all", "any"):
        mode = "auto"
    sort = (sort or "relevance").lower()
    if sort not in ("relevance", "date"):
        sort = "relevance"
    experiment_config = _search_experiment_config()
    _profile_cache = _env_enabled("CHEMTREE_SEARCH_PROFILE")
    _profile_started = _time.monotonic() if _profile_cache else None
    # Result-LRU short-circuit (opt-in via CHEMTREE_SEARCH_CACHE=1).
    _cache_key = None
    if _search_cache_enabled():
        _cache_key = (
            query, claim_type, view, int(limit), int(offset), bool(use_semantic),
            mode, sort, experiment_config,
        )
        _cached = _search_cache_get(_cache_key)
        if _cached is not None:
            if _trace_into is not None:
                _trace_into["cache_hit"] = True
                _trace_into["experiment_config"] = dict(experiment_config)
            if _profile_cache:
                print("[search_profile] " + json.dumps({
                    "pid": os.getpid(),
                    "cache_hit": True,
                    "total_ms": round(
                        (_time.monotonic() - _profile_started) * 1000, 3,
                    ),
                    "config": dict(experiment_config),
                }, separators=(",", ":")), flush=True)
            return dict(_cached)
    if _trace_into is not None:
        _trace_into["cache_hit"] = False
        _trace_into["experiment_config"] = dict(experiment_config)
    _sc_debug = _env_enabled("CHEMTREE_SEARCH_PROFILE")
    _collect_timings = _sc_debug or _trace_into is not None
    if _collect_timings:
        import time as _t
        _sc_t0 = _t.monotonic()
        _sc_marks = [("start", _sc_t0)]
        def _mark(label):
            _sc_marks.append((label, _t.monotonic()))
    else:
        def _mark(label):
            pass
    _mark("entry")
    from askchem.retrieval import (
        vector_search, is_loaded as embeddings_loaded, load_embeddings,
        cross_rerank, cross_rerank_enabled,
    )
    _mark("imports")

    PAPER_TOP_K = 60
    TREE_TOP_K = 200
    CLAIM_FTS_LIMIT = max(300, limit * 4)
    VECTOR_K = max(200, limit * 4)
    # MIN_SEMANTIC_SCORE was tuned against MiniLM's cosine distribution
    # (typical max ≈ 0.55). mxbai's distribution is materially higher
    # (typical max > 0.70), so the same threshold is more permissive in
    # absolute terms. δ2 re-sweeps {0.10, 0.15, 0.20, 0.25} via the
    # CHEMTREE_DENSE_MIN_SCORE knob; default unchanged.
    try:
        MIN_SEMANTIC_SCORE = float(
            os.environ.get("CHEMTREE_DENSE_MIN_SCORE", "0.20")
        )
    except ValueError:
        MIN_SEMANTIC_SCORE = 0.20

    if use_semantic and not embeddings_loaded():
        try:
            load_embeddings()
        except Exception:
            pass

    query_variants = expand_query_variants(query)
    try:
        max_variants = int(
            os.environ.get("CHEMTREE_MAX_QUERY_VARIANTS", "0") or "0"
        )
    except ValueError:
        max_variants = 0
    if max_variants > 0:
        query_variants = query_variants[:max_variants]
    author_hint = _extract_author_name(query)
    _mark("variants")

    with get_conn() as conn:
        _mark("conn")
        # ── ①b Tree-based BFS recall ──
        tree_ranked: list[str] = []
        if not _env_enabled("CHEMTREE_DISABLE_TREE_RECALL"):
            tree_ranked = _tree_recall(
                query, conn, top_k=TREE_TOP_K, restrict_view_id=view,
            )
        if _trace_into is not None:
            _trace_into["tree_pool"] = list(tree_ranked)
        _mark("tree_recall")

        # ── ①c Author recall (only when query looks like an author lookup) ──
        author_ranked: list[str] = []
        author_doi_set: set[str] = set()
        if (author_hint
                and not _env_enabled("CHEMTREE_DISABLE_AUTHOR_RECALL")):
            author_ranked, author_doi_set = _author_recall(
                author_hint, conn, top_k=PAPER_TOP_K,
            )
        if _trace_into is not None:
            _trace_into["author_pool"] = list(author_ranked)
            _trace_into["author_doi_count"] = len(author_doi_set)
        _mark("author")

        # ── ② Paper-level recall (two complementary paths) ──
        # Path A: source_fts (title/abstract) recall with citation boost
        paper_dois_a: list[str] = []
        if not _env_enabled("CHEMTREE_DISABLE_SOURCE_PAPER_RECALL"):
            paper_dois_a = _paper_recall(query, conn, top_k=PAPER_TOP_K)
        _mark("source_paper_recall")
        # Path B: claim-guided recall (papers whose claims match, ranked
        # by citation count — catches authoritative papers BM25 misses)
        paper_dois_b: list[str] = []
        if not _env_enabled("CHEMTREE_DISABLE_CLAIM_GUIDED_PAPER_RECALL"):
            paper_dois_b = _claim_guided_paper_recall(
                query, conn, top_k=PAPER_TOP_K,
            )
        _mark("claim_guided_paper_recall")

        # Merge: deduplicate, keeping order of first appearance
        seen_paper = set()
        paper_dois: list[str] = []
        for d in paper_dois_a + paper_dois_b:
            dl = d.lower()
            if dl not in seen_paper:
                seen_paper.add(dl)
                paper_dois.append(d)

        paper_claims = _get_claims_for_papers(paper_dois, conn)
        _mark("paper_claim_hydration")
        paper_cid_set = {c.get('claim_id') for c in paper_claims if c.get('claim_id')}
        paper_claim_map = {c['claim_id']: c for c in paper_claims if c.get('claim_id')}

        from collections import defaultdict as _ddict
        _by_paper: dict[str, list[str]] = _ddict(list)
        for c in paper_claims:
            if c.get('claim_id'):
                _by_paper[c.get('source_doi', '').lower()].append(c['claim_id'])
        CLAIMS_PER_PAPER_RRF = 3
        paper_ranked: list[str] = []
        for doi in paper_dois:
            cids = _by_paper.get(doi.lower(), [])
            paper_ranked.extend(cids[:CLAIMS_PER_PAPER_RRF])
        if _trace_into is not None:
            _trace_into["source_paper_dois"] = list(paper_dois_a)
            _trace_into["claim_guided_paper_dois"] = list(paper_dois_b)
            _trace_into["source_paper_pool"] = [
                cid
                for doi in paper_dois_a
                for cid in _by_paper.get(doi.lower(), [])[:CLAIMS_PER_PAPER_RRF]
            ]
            _trace_into["claim_guided_paper_pool"] = [
                cid
                for doi in paper_dois_b
                for cid in _by_paper.get(doi.lower(), [])[:CLAIMS_PER_PAPER_RRF]
            ]
            _trace_into["paper_doi_count"] = len(paper_dois)
            _trace_into["paper_claims_loaded"] = len(paper_claims)
            _trace_into["paper_pool"] = list(paper_ranked)

        # ── ③ Claim-level FTS recall (run on all query variants) ──
        fts_ranked_all: list[str] = []
        fts_data: dict[str, dict] = {}
        fts_seen: set[str] = set()

        fts_variant_counts: list[dict] = []
        if not _env_enabled("CHEMTREE_DISABLE_FTS"):
            for variant in query_variants:
                fts_queries = _build_fts_queries(variant, mode=mode)
                rows, _ = _run_fts_cascade(
                    fts_queries, claim_type, CLAIM_FTS_LIMIT, conn,
                )
                fts_variant_counts.append({
                    "variant_index": len(fts_variant_counts),
                    "query_shapes": len(fts_queries),
                    "hits": len(rows),
                })
                for r in rows:
                    cid = r['claim_id']
                    if cid not in fts_seen:
                        fts_seen.add(cid)
                        fts_ranked_all.append(cid)
                        fts_data[cid] = r

        if (not fts_ranked_all
                and not _env_enabled("CHEMTREE_DISABLE_FTS")):
            try:
                from askchem.paw_functions import normalize_query
                normalized = normalize_query(query)
                if normalized.lower() != query.lower().strip():
                    norm_fts = _build_fts_queries(normalized, mode=mode)
                    rows, _ = _run_fts_cascade(norm_fts, claim_type,
                                               CLAIM_FTS_LIMIT, conn)
                    for r in rows:
                        cid = r['claim_id']
                        if cid not in fts_seen:
                            fts_seen.add(cid)
                            fts_ranked_all.append(cid)
                            fts_data[cid] = r
            except Exception:
                pass

        # PAW decompose_query rescue (Phase 2 wiring, May-23 ft rollout).
        #
        # When normalize_query also fails to produce any FTS hits, fall back
        # to splitting the query into multiple sub-topic queries and running
        # the FTS cascade on each.  Targets the ``multi`` family probes
        # (nDCG@10 = 0.704, the weakest family per the May-14 ablation).
        # Gated separately from CHEMTREE_PAW_REWRITES so the A/B can compare
        # "expand only" vs "expand + decompose" cleanly; if the env var is
        # unset, this block is a no-op and the existing fallback chain is
        # unchanged.
        if (not fts_ranked_all
                and not _env_enabled("CHEMTREE_DISABLE_FTS")
                and os.environ.get("CHEMTREE_PAW_REWRITES", "0") == "1"):
            try:
                from askchem.paw_functions import decompose_query
                sub_qs = decompose_query(query) or []
                for sub in sub_qs[:3]:
                    sub_fts = _build_fts_queries(sub, mode=mode)
                    rows, _ = _run_fts_cascade(sub_fts, claim_type,
                                               CLAIM_FTS_LIMIT, conn)
                    for r in rows:
                        cid = r['claim_id']
                        if cid not in fts_seen:
                            fts_seen.add(cid)
                            fts_ranked_all.append(cid)
                            fts_data[cid] = r
            except Exception:
                pass

        # ── ③ Claim-level vector recall ──
        _mark("fts")
        if _trace_into is not None:
            _trace_into["fts_pool"] = list(fts_ranked_all)
            _trace_into["query_variants"] = list(query_variants)
            _trace_into["fts_variant_counts"] = fts_variant_counts
        vec_ranked: list[str] = []
        vec_scores: dict[str, float] = {}
        if (use_semantic and embeddings_loaded()
                and not _env_enabled("CHEMTREE_DISABLE_DENSE")):
            from askchem.retrieval import embed_query as _eq
            _ = _eq(query)
            _mark("embed_query")
            vec_hits = vector_search(
                query, top_k=VECTOR_K, min_score=MIN_SEMANTIC_SCORE
            )
            vec_ranked = [cid for cid, _ in vec_hits]
            vec_scores = {cid: score for cid, score in vec_hits}
        if _trace_into is not None:
            _trace_into["vector_pool"] = list(vec_ranked)
        _mark("faiss_search")

        # ── ④ RRF merge (FTS + vector + tree + paper) ──
        ranked_lists: list[list[str]] = []
        if fts_ranked_all:
            ranked_lists.append(fts_ranked_all)
        if vec_ranked:
            ranked_lists.append(vec_ranked)
        if tree_ranked:
            ranked_lists.append(tree_ranked)
        if paper_ranked:
            ranked_lists.append(paper_ranked)
        if author_ranked:
            # Author match is a strong signal when triggered — include it
            # twice so RRF weights it more heavily than one of many BM25 lists.
            ranked_lists.append(author_ranked)
            ranked_lists.append(author_ranked)

        if not ranked_lists and not paper_ranked:
            _empty = {'results': [], 'total': 0, 'query': query,
                      'limit': limit, 'offset': offset}
            if _cache_key is not None:
                _search_cache_put(_cache_key, _empty)
            return _empty

        rrf_scores: dict[str, float] = {}
        if ranked_lists:
            for cid, score in _rrf_merge(ranked_lists):
                rrf_scores[cid] = score

        # ── ④b Pseudo-Relevance Feedback ──
        # Kill switch: CHEMTREE_DISABLE_PRF=1 skips the 8 extra FTS lookups
        # the PRF stage fans out (May-15 ablation).
        if os.environ.get("CHEMTREE_DISABLE_PRF", "0") != "1":
            initial_top = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:30]
            prf_lists = _pseudo_relevance_feedback(
                query, initial_top, paper_dois, conn
            )
            if prf_lists:
                ranked_lists.extend(prf_lists)
                rrf_scores.clear()
                for cid, score in _rrf_merge(ranked_lists):
                    rrf_scores[cid] = score

        all_cids_ordered = sorted(rrf_scores, key=lambda c: -rrf_scores[c])
        if _trace_into is not None:
            # RRF pool is the union of all recall channels post-merge. Cap
            # at 200 so the trace stays compact but covers more than the
            # rerank window (top-50).
            _trace_into["rrf_pool"] = list(all_cids_ordered[:200])

        # ── ⑤ Fetch claim data ──
        all_needed = set(all_cids_ordered) | paper_cid_set
        known_cids = set(fts_data.keys()) | set(paper_claim_map.keys())
        missing_cids = [cid for cid in all_needed if cid not in known_cids]
        extra_data: dict[str, dict] = {}
        if missing_cids:
            for i in range(0, len(missing_cids), 999):
                batch = missing_cids[i:i + 999]
                ph = ','.join('?' * len(batch))
                extra_rows = conn.execute(
                    f"SELECT claim_id, claim_type, source_doi, source_paper_title, "
                    f"confidence, location_in_paper, verbatim_quote, "
                    f"extraction_model, data, claim_contextualized "
                    f"FROM claims WHERE claim_id IN ({ph})",
                    batch,
                ).fetchall()
                for er in extra_rows:
                    extra_data[er['claim_id']] = er
        _mark("candidate_fetch")

        def _load_claim(cid: str) -> dict | None:
            if cid in paper_claim_map:
                return paper_claim_map[cid]
            if cid in fts_data:
                fr = fts_data[cid]
                c = json.loads(fr['data']) if fr['data'] else {}
                # _run_fts_cascade now ships claim_contextualized in the
                # row; fall through if older code paths predate that.
                if 'claim_contextualized' in fr.keys() and fr['claim_contextualized']:
                    c['claim_contextualized'] = fr['claim_contextualized']
                return c
            if cid in extra_data:
                er = extra_data[cid]
                c = json.loads(er['data'])
                for col in ('claim_id', 'claim_type', 'source_doi',
                            'source_paper_title', 'confidence',
                            'location_in_paper', 'verbatim_quote',
                            'extraction_model'):
                    if col not in c and er[col]:
                        c[col] = er[col]
                if er['claim_contextualized']:
                    c['claim_contextualized'] = er['claim_contextualized']
                return c
            return None

        # Build claim-level results
        claim_results: list[dict] = []
        for cid in all_cids_ordered:
            claim = _load_claim(cid)
            if claim:
                sem_score = vec_scores.get(cid, 0.0)
                claim['_relevance_score'] = round(
                    rrf_scores[cid] + sem_score * 0.05, 4
                )
                claim_results.append(claim)
        _mark("claim_json_hydration")

        # ── ⑤b Citation boost ──
        # Multiply relevance scores by (1 + α·normalized_log_citations).
        # Keeps relevance dominant but surfaces claims from landmark papers.
        if (claim_results
                and not _env_enabled("CHEMTREE_DISABLE_CITATION_BOOST")):
            import math
            doi_set = {c.get('source_doi') for c in claim_results
                       if c.get('source_doi')}
            if doi_set:
                cite_map: dict[str, int] = {}
                doi_list = list(doi_set)
                for i in range(0, len(doi_list), 999):
                    batch = doi_list[i:i+999]
                    ph = ','.join('?' * len(batch))
                    rows = conn.execute(
                        f"SELECT doi, citation_count FROM sources "
                        f"WHERE doi IN ({ph})", batch
                    ).fetchall()
                    for r in rows:
                        cite_map[r['doi']] = r['citation_count'] or 0

                if cite_map:
                    max_log = math.log(2 + max(cite_map.values()))
                    CLAIM_CITE_ALPHA = 1.0
                    for claim in claim_results:
                        doi = claim.get('source_doi', '')
                        cites = cite_map.get(doi, 0)
                        if cites > 0:
                            cite_factor = math.log(1 + cites) / max_log
                            claim['_relevance_score'] *= (
                                1.0 + CLAIM_CITE_ALPHA * cite_factor
                            )

                    claim_results.sort(
                        key=lambda c: c.get('_relevance_score', 0),
                        reverse=True,
                    )
        _mark("citation_boost")

        # ── ⑤c Cross-encoder rerank (γ1, opt-in via CHEMTREE_RETRIEVER_VERSION=v2) ──
        # Reorders only the top window — the bake-off picked
        # `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking the top-20 of the
        # dense ANN candidates as the production config (Δ +0.022 nDCG@10
        # over mxbai dense, p95 = 150 ms on Apple-MPS).  We rerank a
        # slightly bigger window (top-50) here so claims that the dense
        # stage put just outside the visible page can still climb in.
        # No-op when v1 is active or the v2 cross-encoder is disabled.
        try:
            _mark("pre_rerank")
            if (claim_results and cross_rerank_enabled()
                    and not _env_enabled("CHEMTREE_DISABLE_RERANK")):
                from askchem.embeddings import _claim_to_text
                # Hard cap at 50 (May-14 ablation: top-20 nDCG@10 unchanged
                # vs. uncapped max(50, limit*2)). At limit=500 the uncapped
                # value would be 1000, which pushes the cross-encoder past
                # the 90 s Nginx timeout on the 2-CPU VPS. Positions 51+
                # fall back to dense+FTS+RRF ordering, which is the same
                # quality AskChem shipped before Phase gamma1.
                # May-15 ablation knob: CHEMTREE_RERANK_WINDOW shrinks
                # the window (e.g. 30) to trade a small nDCG drop for
                # ~40% cross-encoder latency reduction on CPU.
                try:
                    RERANK_WINDOW = int(
                        os.environ.get("CHEMTREE_RERANK_WINDOW", "50") or "50"
                    )
                except ValueError:
                    RERANK_WINDOW = 50
                RERANK_WINDOW = max(1, RERANK_WINDOW)
                head = claim_results[:RERANK_WINDOW]
                # paper_summary lookup — one query for the unique DOIs in head
                head_dois = list({c.get('source_doi') for c in head
                                  if c.get('source_doi')})
                ps_map: dict[str, str] = {}
                if head_dois:
                    for i in range(0, len(head_dois), 999):
                        batch = head_dois[i:i + 999]
                        ph = ','.join('?' * len(batch))
                        rows = conn.execute(
                            f"SELECT doi, paper_summary FROM sources "
                            f"WHERE doi IN ({ph})", batch,
                        ).fetchall()
                        for r in rows:
                            if r['paper_summary']:
                                ps_map[r['doi']] = r['paper_summary']
                pairs: list[tuple[str, str]] = []
                for c in head:
                    cid = c.get('claim_id')
                    if not cid:
                        continue
                    text = _claim_to_text(
                        c,
                        claim_contextualized=c.get('claim_contextualized'),
                        paper_summary=ps_map.get(c.get('source_doi', '')),
                    )
                    if text:
                        pairs.append((cid, text))
                if pairs:
                    # Phase 1 (May-29) attribution wiring: when
                    # CHEMTREE_PAW_REWRITES_RERANK=1, feed the rerank an
                    # augmented query (anchor + top-K PAW expansion terms)
                    # so the cross-encoder sees the expanded vocabulary.
                    # Default (env unset) preserves prod behaviour.
                    rerank_query = query
                    if (os.environ.get(
                        "CHEMTREE_PAW_REWRITES_RERANK", "0"
                    ) == "1"
                        and os.environ.get(
                            "CHEMTREE_DISABLE_PAW", "0"
                        ) != "1"):
                        try:
                            from askchem.paw_functions import expand_query as _paw_expand
                            paw_terms = _paw_expand(query)[:8]
                            if paw_terms:
                                rerank_query = (
                                    query + " " + " ".join(paw_terms)
                                )
                        except Exception:
                            pass
                    if _trace_into is not None:
                        _trace_into["rerank_input"] = [cid for cid, _ in pairs]
                        _trace_into["rerank_query"] = rerank_query

                    # Phase 3 (May-29) PAW reranker wiring: when
                    # CHEMTREE_PAW_RERANK_ID is set to a PAW program id,
                    # gate the top-N of the cross-encoder output through
                    # a PAW relevance scorer for final ordering. Top-N
                    # is controlled by CHEMTREE_PAW_RERANK_TOPK (default
                    # 5). Cross-encoder still runs first to set the
                    # initial order; PAW only reorders the top-N for
                    # latency reasons.
                    reranked = cross_rerank(rerank_query, pairs,
                                            top_k=len(pairs))
                    paw_rerank_id = os.environ.get(
                        "CHEMTREE_PAW_RERANK_ID", ""
                    ).strip()
                    if paw_rerank_id and os.environ.get(
                        "CHEMTREE_DISABLE_PAW", "0"
                    ) != "1":
                        try:
                            topk = int(os.environ.get(
                                "CHEMTREE_PAW_RERANK_TOPK", "5"
                            ))
                        except ValueError:
                            topk = 5
                        try:
                            import programasweights as _paw
                            _paw_fn = _paw.function(
                                paw_rerank_id, n_gpu_layers=0,
                            )
                            text_by_id = {cid: t for cid, t in pairs}
                            paw_head = reranked[:topk]
                            paw_tail = reranked[topk:]
                            # Score each head item with PAW; map labels
                            # to integer scores so ordering is stable.
                            scored: list[tuple[str, float, float]] = []
                            for rank_pos, (cid, cscore) in enumerate(paw_head):
                                text = text_by_id.get(cid, "")
                                if not text:
                                    scored.append((cid, cscore, 0.0))
                                    continue
                                inp = f"QUERY: {query} CLAIM: {text}"
                                raw = (_paw_fn(inp) or "").strip().lower()
                                tok = (raw.split(",")[0].split()[0]
                                       if raw else "")
                                tok = tok.strip(".,;:!?'\"()[]{}")
                                paw_score = {
                                    "exact_match": 3.0,
                                    "highly_relevant": 2.0,
                                    "somewhat_relevant": 1.0,
                                    "not_relevant": 0.0,
                                }.get(tok, 0.5)
                                scored.append((cid, cscore, paw_score))
                            # Sort by PAW score desc; tiebreak by
                            # original cross-encoder score so we never
                            # regress within an equivalence class.
                            scored.sort(
                                key=lambda x: (-x[2], -x[1])
                            )
                            paw_new_head = [(cid, cscore)
                                            for cid, cscore, _ in scored]
                            reranked = paw_new_head + paw_tail
                        except Exception as exc:
                            print(f"[search] PAW rerank skipped: {exc}",
                                  file=sys.stderr)
                    if _trace_into is not None:
                        _trace_into["rerank_output"] = [cid for cid, _ in reranked]
                    rerank_score = {cid: s for cid, s in reranked}
                    rerank_pos = {cid: i for i, (cid, _) in enumerate(reranked)}
                    # Stable: keep tail order; reorder head by rerank_pos.
                    by_id = {c.get('claim_id'): c for c in head}
                    new_head = [by_id[cid] for cid, _ in reranked
                                if cid in by_id]
                    # Stash the score on each claim so it can be inspected
                    # client-side.  The dense / RRF / citation _relevance_score
                    # is preserved for tie-breaking outside the rerank window.
                    for c in new_head:
                        cid = c.get('claim_id')
                        if cid in rerank_score:
                            c['_rerank_score'] = round(rerank_score[cid], 4)
                    claim_results = new_head + claim_results[RERANK_WINDOW:]

        except Exception as exc:
            # Never let a rerank failure poison the v1 fallback path.
            print(f"[search] cross-encoder rerank skipped: {exc}",
                  file=sys.stderr)
        _mark("rerank")

        # ── ⑥ Paper diversity injection ──
        # When a paper made it into `paper_dois` (title/abstract match or
        # claim-aggregated match) but none of its claims surfaced in the
        # primary RRF top, we inject one claim from that paper so the
        # paper isn't invisible. CRITICAL: the injected claim must itself
        # be query-relevant — i.e. appear in either the FTS or the vector
        # recall list. Without this gate, a paper that only mentions the
        # query term in passing (e.g. a review whose abstract says
        # "Suzuki coupling" once but whose claims are about hydrothermal
        # carbonisation) injects an arbitrary unrelated claim into the
        # results. We always pick the strongest matching claim per paper,
        # ranked by vector similarity then FTS rank.
        INJECT_PER_PAPER = 1
        MAX_TOTAL_INJECTIONS = max(limit // 2, 15)
        # Pre-compute query-relevance set + a fast FTS-rank lookup, since
        # both the injection and (further down) the view filter use them.
        query_relevant_cids = fts_seen | set(vec_ranked)
        fts_rank_map = {cid: i for i, cid in enumerate(fts_ranked_all)}
        if paper_dois and claim_results:
            pre_inject_dois = {c.get('source_doi', '').lower()
                               for c in claim_results[:limit]}
            visible_dois = set(pre_inject_dois)
            result_cids = {c.get('claim_id') for c in claim_results[:limit]}

            from collections import defaultdict
            by_paper: dict[str, list[str]] = defaultdict(list)
            for cid in paper_ranked:
                c = paper_claim_map.get(cid)
                if c:
                    by_paper[c.get('source_doi', '').lower()].append(cid)

            injections: list[dict] = []
            target_pos = min(len(claim_results) - 1, limit // 2)
            inject_score = claim_results[target_pos]['_relevance_score']

            def _candidate_relevance(cid: str) -> tuple:
                # Higher is better. Prefer high vector score, then a
                # higher FTS rank (lower index), then any FTS hit.
                vec = vec_scores.get(cid, 0.0)
                in_fts = cid in fts_seen
                fts_neg_idx = -fts_rank_map.get(cid, 10**9)
                return (vec, in_fts, fts_neg_idx)

            for doi in paper_dois:
                doi_lower = doi.lower()
                if doi_lower in visible_dois:
                    continue
                candidates = by_paper.get(doi_lower, []) or by_paper.get(doi, [])
                if not candidates:
                    continue
                # Only consider claims that are themselves query-relevant.
                relevant_candidates = [cid for cid in candidates
                                       if cid in query_relevant_cids]
                if not relevant_candidates:
                    continue
                relevant_candidates.sort(key=_candidate_relevance, reverse=True)
                added = 0
                for cid in relevant_candidates:
                    if added >= INJECT_PER_PAPER:
                        break
                    if len(injections) >= MAX_TOTAL_INJECTIONS:
                        break
                    if cid not in result_cids:
                        src_claim = paper_claim_map.get(cid)
                        if src_claim is None:
                            continue
                        claim = dict(src_claim)
                        claim['_relevance_score'] = inject_score
                        claim['_from_paper_recall'] = True
                        injections.append(claim)
                        result_cids.add(cid)
                        added += 1

            if injections:
                claim_results.extend(injections)
                claim_results.sort(
                    key=lambda c: c.get('_relevance_score', 0),
                    reverse=True,
                )

        # ── ⑥b Per-paper cap ──
        # At most one claim per paper in the results so landmark papers
        # don't hog the first page.  Extra claims from the same paper are
        # demoted to the tail, where they remain reachable via pagination.
        MAX_PER_DOI = 1
        if claim_results:
            per_doi_count: dict[str, int] = {}
            primary: list[dict] = []
            overflow: list[dict] = []
            for c in claim_results:
                doi_key = (c.get('source_doi') or '').lower()
                if not doi_key:
                    primary.append(c)
                    continue
                if per_doi_count.get(doi_key, 0) < MAX_PER_DOI:
                    primary.append(c)
                    per_doi_count[doi_key] = per_doi_count.get(doi_key, 0) + 1
                else:
                    overflow.append(c)
            claim_results = primary + overflow

        # ── ⑥c Author hard filter ──
        # When the query is unambiguously an author lookup, only show
        # papers whose authors actually match.  This prevents bleed-through
        # of high-citation papers that happen to mention the author name
        # in unrelated text.
        if author_hint:
            if author_doi_set:
                claim_results = [
                    c for c in claim_results
                    if (c.get('source_doi') or '').lower() in author_doi_set
                ]
            else:
                claim_results = []

        results = claim_results

        if claim_type:
            results = [r for r in results
                       if r.get('claim_type') == claim_type]
        if view:
            # Two-pass view filter:
            #   ① keep only claims that have a non-empty path in this view
            #   ② additionally require the claim to be independently
            #      query-relevant — i.e. it appears in either the FTS or the
            #      vector recall list. Without this, paper-level recall and
            #      tree-level BFS recall pollute the post-filter ranking with
            #      claims that are NOT about the query but happen to live in
            #      a query-matching paper or taxonomy node AND happen to
            #      have the chosen view set. (See the "drug delivery" /
            #      "by_substance_class" pollution case in tests.)
            #
            # `query_relevant_cids` is the union of FTS hits and vector hits.
            # Author-recall is excluded on purpose because it's already
            # gated by `author_hint` above; tree/paper recall lists are
            # excluded because they're the source of the pollution.
            # (Already computed above for paper-diversity injection.)
            view_filtered: list[dict] = []
            for r in results:
                view_paths = r.get('view_paths') or {}
                segs = view_paths.get(view)
                if not segs or not isinstance(segs, list):
                    continue
                # Drop "not_applicable"/"none" markers; if nothing left, skip.
                if not [s for s in segs
                        if s and s not in ('not_applicable', 'none')]:
                    continue
                if (r.get('claim_id') or '') in query_relevant_cids:
                    view_filtered.append(r)
            # Fallback: if the strict filter empties the result set (e.g. a
            # rare query whose only matches all came from paper/tree recall),
            # relax to the loose view filter so the user gets *something*
            # rather than a blank page. This preserves recall for niche
            # queries while keeping common queries clean.
            if view_filtered:
                results = view_filtered
            else:
                results = [r for r in results
                           if (r.get('view_paths') or {}).get(view)]

            # Strip computational-modelling bucket noise for named coupling
            # queries when browsing Technique/Method — keeps ``Computational
            # Modeling'' from filling with spin-coupling / DFT jargon hits.
            # mxbai's homonym nDCG@10 of 0.942 means this filter is mostly
            # redundant on v2; live eval in δ2 verifies whether removing it
            # regresses. Kill switch: CHEMTREE_DISABLE_TECHNIQUE_STRIPPER=1.
            if (view == "by_technique"
                    and query_signals_organic_cross_coupling(query)
                    and os.environ.get(
                        "CHEMTREE_DISABLE_TECHNIQUE_STRIPPER", "0"
                    ) != "1"):
                results = [
                    r for r in results
                    if not _technique_claim_is_irrelevant_for_coupling_query(r)
                ]

        total = len(results)

        if sort == "date":
            # Two-pass sort: enrich the leading window with source_year so
            # we can order by recency, then page. Claims missing a year
            # drop to the tail (sentinel -1) and break ties by relevance.
            sort_window = min(len(results), max(200, (offset + limit) * 3))
            enrich_claims_with_source(results[:sort_window], conn)
            results = sorted(
                results,
                key=lambda r: (
                    -(int(r.get('source_year')) if r.get('source_year') else -1),
                    -(r.get('_relevance_score') or 0.0),
                ),
            )

        results = results[offset:offset + limit]
        enrich_claims_with_source(results, conn)
        _mark("done")

        if _collect_timings:
            timing_marks = [
                {
                    "stage": lab,
                    "elapsed_ms": round((t - _sc_marks[0][1]) * 1000, 3),
                    "duration_ms": round(
                        (t - _sc_marks[i - 1][1]) * 1000, 3
                    ) if i else 0.0,
                }
                for i, (lab, t) in enumerate(_sc_marks)
            ]
            if _trace_into is not None:
                _trace_into["timings"] = timing_marks
                _trace_into["counts"] = {
                    "query_variants": len(query_variants),
                    "tree_pool": len(tree_ranked),
                    "author_pool": len(author_ranked),
                    "paper_dois": len(paper_dois),
                    "paper_claims_loaded": len(paper_claims),
                    "paper_pool": len(paper_ranked),
                    "fts_pool": len(fts_ranked_all),
                    "vector_pool": len(vec_ranked),
                    "rrf_pool": len(all_cids_ordered),
                    "hydrated_candidates": len(claim_results),
                    "final_results": len(results),
                }
        if _sc_debug:
            print(
                "[search_profile] " + json.dumps({
                    "pid": os.getpid(),
                    "cache_hit": False,
                    "timings": timing_marks,
                    "counts": {
                        "query_variants": len(query_variants),
                        "tree_pool": len(tree_ranked),
                        "author_pool": len(author_ranked),
                        "paper_dois": len(paper_dois),
                        "paper_claims_loaded": len(paper_claims),
                        "paper_pool": len(paper_ranked),
                        "fts_pool": len(fts_ranked_all),
                        "vector_pool": len(vec_ranked),
                        "rrf_pool": len(all_cids_ordered),
                        "final_results": len(results),
                    },
                    "config": dict(experiment_config),
                }, separators=(",", ":")),
                flush=True,
            )

        _response = {
            'results': results,
            'total': total,
            'query': query,
            'limit': limit,
            'offset': offset,
            'mode': mode,
            'sort': sort,
        }
        if _trace_into is not None:
            _trace_into["final_top"] = [
                r.get("claim_id") for r in results
                if r.get("claim_id")
            ]
        if _cache_key is not None:
            _search_cache_put(_cache_key, _response)
        return _response


_CLAIM_TYPE_PLURALS = {
    'reaction': 'reactions', 'property': 'properties', 'method': 'methods',
    'mechanism': 'mechanisms', 'comparison': 'comparisons',
    'scope_entry': 'scope_entries', 'computational_result': 'computational_results',
    'hypothesis': 'hypotheses', 'conclusion': 'conclusions',
    'conclusions': 'conclusions', 'limitation': 'limitations',
    'future_direction': 'future_directions', 'surprising_finding': 'surprising_findings',
    'experimental_design': 'experimental_designs', 'structure': 'structures',
    'background': 'background', 'historical': 'historical',
    'definition': 'definitions', 'observation': 'observations',
}


# `search_claims_grouped` was removed in the May-2026 "one search mode"
# consolidation. The grouped/tree view is now built client-side over the
# same `/api/search` response (see `buildClientTree` in web/index.html).
# Server-side grouping was a presentation wrapper over `search_claims` and
# offered no retrieval benefit beyond filtering by `view`.


_smiles_index: dict | None = None
_smiles_index_ts: float = 0


def _build_smiles_index() -> dict:
    """Build/return cached index: {smiles: (RDKit Mol, [claim_id, ...])}."""
    global _smiles_index, _smiles_index_ts
    import time
    now = time.monotonic()
    if _smiles_index is not None and now - _smiles_index_ts < 3600:
        return _smiles_index

    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # Build a SMILES -> (mol, claim_ids) mapping using smiles_validations table
    # which only has ~3K rows vs scanning 1.2M claims
    index: dict = {}
    with get_conn() as conn:
        has_sv = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='smiles_validations'"
        ).fetchone()
        if has_sv:
            rows = conn.execute(
                "SELECT sv.claim_id, sv.smiles FROM smiles_validations sv "
                "WHERE sv.is_valid = 1"
            ).fetchall()
            for r in rows:
                smi = r["smiles"]
                if smi not in index:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    index[smi] = {"mol": mol, "claim_ids": []}
                index[smi]["claim_ids"].append(r["claim_id"])
        else:
            # Fallback: scan claims table for subject_smiles
            rows = conn.execute(
                "SELECT claim_id, data FROM claims WHERE data LIKE '%\"subject_smiles\"%' LIMIT 200000"
            ).fetchall()
            for r in rows:
                d = json.loads(r["data"])
                smi = d.get("subject_smiles", "")
                if not smi or len(smi) < 2 or smi.startswith("not "):
                    continue
                if smi not in index:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    index[smi] = {"mol": mol, "claim_ids": []}
                index[smi]["claim_ids"].append(r["claim_id"])

    _smiles_index = index
    _smiles_index_ts = now
    return index


def search_by_structure(smiles: str, search_type: str = "substructure",
                        limit: int = 50, offset: int = 0) -> dict:
    """Search claims by molecular structure using RDKit.

    Uses a pre-built in-memory index of ~500 distinct SMILES with their
    claim_ids for sub-second lookups after initial cache build.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        return {"results": [], "total": 0, "query_smiles": smiles,
                "search_type": search_type, "error": "RDKit not installed"}

    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        return {"results": [], "total": 0, "query_smiles": smiles,
                "search_type": search_type, "error": "Invalid SMILES"}

    index = _build_smiles_index()

    # Phase 1: match query against ~500 distinct SMILES (milliseconds)
    matching: list[tuple[str, float, list[str]]] = []
    if search_type == "substructure":
        for smi, entry in index.items():
            if entry["mol"].HasSubstructMatch(query_mol):
                matching.append((smi, 1.0, entry["claim_ids"]))
    elif search_type == "similarity":
        fp_q = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        for smi, entry in index.items():
            fp_t = AllChem.GetMorganFingerprintAsBitVect(entry["mol"], 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp_q, fp_t)
            if sim >= 0.3:
                matching.append((smi, round(sim, 3), entry["claim_ids"]))
        matching.sort(key=lambda x: x[1], reverse=True)

    if not matching:
        return {"results": [], "total": 0, "query_smiles": smiles,
                "search_type": search_type}

    # Phase 2: collect claim_ids from matches and fetch the page
    all_claim_ids = []
    sim_map: dict[str, float] = {}
    for smi, score, cids in matching:
        sim_map[smi] = score
        all_claim_ids.extend(cids)

    total = len(all_claim_ids)
    page_ids = all_claim_ids[offset:offset + limit]

    if not page_ids:
        return {"results": [], "total": total, "query_smiles": smiles,
                "search_type": search_type}

    with get_conn() as conn:
        placeholders = ",".join("?" for _ in page_ids)
        rows = conn.execute(
            f"SELECT data FROM claims WHERE claim_id IN ({placeholders})",
            page_ids,
        ).fetchall()

        hits = []
        for r in rows:
            claim = json.loads(r["data"])
            if search_type == "similarity":
                claim["_similarity"] = sim_map.get(claim.get("subject_smiles", ""), 0)
            hits.append(claim)

        enrich_claims_with_source(hits, conn)

    return {"results": hits, "total": total,
            "query_smiles": smiles, "search_type": search_type}


def enrich_claims_with_source(claims: list[dict], conn=None) -> list[dict]:
    """Batch-enrich claims with venue, year, and authors from the sources table.

    Uses a module-level LRU cache for hot sources to avoid repeated DB lookups.
    """
    global _source_cache
    if not claims:
        return claims
    dois = list({c.get('source_doi', '') for c in claims if c.get('source_doi')})
    if not dois:
        return claims

    # Check cache first, only query uncached DOIs
    source_map = {}
    uncached_dois = []
    for doi in dois:
        if doi in _source_cache:
            source_map[doi] = _source_cache[doi]
        else:
            uncached_dois.append(doi)

    if uncached_dois:
        def _fetch(connection):
            placeholders = ','.join('?' * len(uncached_dois))
            rows = connection.execute(
                f"SELECT doi, venue, year, authors, citation_count, "
                f"paper_summary FROM sources "
                f"WHERE doi IN ({placeholders})",
                uncached_dois,
            ).fetchall()
            return {r['doi']: dict(r) for r in rows}

        if conn:
            fetched = _fetch(conn)
        else:
            with get_conn() as c:
                fetched = _fetch(c)

        source_map.update(fetched)

        # Populate cache, evict oldest if over limit
        for doi, data in fetched.items():
            if len(_source_cache) >= _SOURCE_CACHE_MAX:
                oldest = next(iter(_source_cache))
                del _source_cache[oldest]
            _source_cache[doi] = data

    for claim in claims:
        src = source_map.get(claim.get('source_doi', ''))
        if src:
            claim['source_venue'] = src.get('venue') or ''
            claim['source_year'] = src.get('year')
            claim['source_citation_count'] = src.get('citation_count')
            ps = src.get('paper_summary')
            if ps:
                claim['paper_summary'] = ps
            raw_authors = src.get('authors')
            if raw_authors and isinstance(raw_authors, str):
                try:
                    claim['source_authors'] = json.loads(raw_authors)
                except (json.JSONDecodeError, TypeError):
                    claim['source_authors'] = []
            elif isinstance(raw_authors, list):
                claim['source_authors'] = raw_authors
            else:
                claim['source_authors'] = []
    return claims


def _hydrate_claim_row(row) -> dict:
    """Parse a (data, claim_contextualized) row into a claim dict.

    Stuffs the Sprint-1 contextualized rewrite onto the claim under
    ``claim_contextualized`` so downstream renderers and the API see it
    as just another field on the claim. ``enrich_claims_with_source``
    later attaches ``paper_summary`` from the sources table.
    """
    claim = json.loads(row['data']) if row['data'] else {}
    ctx = row['claim_contextualized'] if 'claim_contextualized' in row.keys() else None
    if ctx:
        claim['claim_contextualized'] = ctx
    return claim


def get_claim(claim_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data, claim_contextualized FROM claims WHERE claim_id = ?",
            [claim_id],
        ).fetchone()
        if not row:
            return None
        claim = _hydrate_claim_row(row)
        enrich_claims_with_source([claim], conn)
        return claim


def get_claims_by_doi(doi: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT data, claim_contextualized FROM claims "
            "WHERE source_doi = ? COLLATE NOCASE",
            [doi],
        ).fetchall()
        claims = [_hydrate_claim_row(r) for r in rows]
        enrich_claims_with_source(claims, conn)
        return claims


def get_claims_bulk(claim_ids: list[str]) -> list[dict]:
    """Fetch multiple claims by ID in one query."""
    if not claim_ids:
        return []
    with get_conn() as conn:
        placeholders = ",".join("?" for _ in claim_ids)
        rows = conn.execute(
            f"SELECT data, claim_contextualized FROM claims "
            f"WHERE claim_id IN ({placeholders})",
            claim_ids,
        ).fetchall()
        claims = [_hydrate_claim_row(r) for r in rows]
        enrich_claims_with_source(claims, conn)
        return claims


def get_source(doi: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data FROM sources WHERE doi = ? COLLATE NOCASE",
            [doi]
        ).fetchone()
        return json.loads(row['data']) if row else None


def search_papers(q: str = None, limit: int = 50, offset: int = 0, sort: str = "citations") -> dict:
    """Search or browse papers. Returns papers with claim counts."""
    order = {
        "citations": "s.citation_count DESC",
        "year": "s.year DESC, s.citation_count DESC",
        "claims": "claim_count DESC, s.citation_count DESC",
    }.get(sort, "s.citation_count DESC")

    with get_conn() as conn:
        if q:
            words = q.strip().split()
            conditions = " AND ".join(["s.title LIKE ?"] * len(words))
            params = [f"%{w}%" for w in words]
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM sources s WHERE {conditions}", params
            ).fetchone()
            total = count_row[0]
            rows = conn.execute(
                f"SELECT s.doi, s.title, s.year, s.citation_count, s.venue, s.authors, "
                f"(SELECT COUNT(*) FROM claims c WHERE c.source_doi = s.doi) as claim_count "
                f"FROM sources s WHERE {conditions} "
                f"ORDER BY {order} LIMIT ? OFFSET ?",
                params + [limit, offset]
            ).fetchall()
        else:
            count_row = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
            total = count_row[0]
            rows = conn.execute(
                f"SELECT s.doi, s.title, s.year, s.citation_count, s.venue, s.authors, "
                f"(SELECT COUNT(*) FROM claims c WHERE c.source_doi = s.doi) as claim_count "
                f"FROM sources s ORDER BY {order} LIMIT ? OFFSET ?",
                [limit, offset]
            ).fetchall()

        papers = []
        for r in rows:
            authors_raw = r["authors"]
            if isinstance(authors_raw, str):
                try:
                    authors_list = json.loads(authors_raw)
                except (json.JSONDecodeError, TypeError):
                    authors_list = []
            else:
                authors_list = authors_raw or []
            papers.append({
                "doi": r["doi"],
                "title": r["title"],
                "year": r["year"],
                "citation_count": r["citation_count"],
                "venue": r["venue"],
                "authors": authors_list[:5],
                "claim_count": r["claim_count"],
            })

        return {"papers": papers, "total": total, "limit": limit, "offset": offset, "query": q}


def list_views() -> list[dict]:
    """Return view metadata, rehydrating stale 0 node_count/claim_count.

    Most views were imported without populating the level-0 root row, which
    also left the ``views.data`` summary stuck at 0 nodes / 0 claims. Rather
    than rebuilding those rows we recompute the totals on the fly from
    ``tree_nodes`` for views that look empty in their stored summary.
    """
    with get_conn() as conn:
        rows = conn.execute("SELECT view_id, data FROM views ORDER BY view_id").fetchall()
        views: list[dict] = []
        for r in rows:
            data = json.loads(r['data'])
            if not data.get('node_count') and not data.get('claim_count'):
                summary = conn.execute(
                    "SELECT COUNT(*) AS nodes, COALESCE(MAX(level), 0) AS max_level, "
                    "       MIN(level) AS min_level "
                    "FROM tree_nodes WHERE view_id = ?",
                    [r['view_id']],
                ).fetchone()
                if summary and summary['nodes']:
                    min_lvl = summary['min_level']
                    claim_row = conn.execute(
                        "SELECT COALESCE(SUM(claim_count), 0) AS total "
                        "FROM tree_nodes WHERE view_id = ? AND level = ?",
                        [r['view_id'], min_lvl],
                    ).fetchone()
                    data['node_count'] = summary['nodes']
                    data['claim_count'] = claim_row['total'] if claim_row else 0
                    data['max_depth'] = summary['max_level']
            views.append(data)
        return views


def _virtual_root_segments(conn: sqlite3.Connection, view_id: str) -> tuple[list[str], int, int]:
    """Return (children_paths, claim_count, min_level) for a virtual root.

    Used when a view's ``path=''`` row was never materialized (true for every
    view except ``by_data``). Children are the lowest-level nodes for the view.
    """
    min_row = conn.execute(
        "SELECT MIN(level) AS m FROM tree_nodes WHERE view_id = ?", [view_id]
    ).fetchone()
    if not min_row or min_row['m'] is None:
        return [], 0, 0
    min_lvl = min_row['m']
    rows = conn.execute(
        "SELECT path, COALESCE(claim_count, 0) AS cc FROM tree_nodes "
        "WHERE view_id = ? AND level = ?",
        [view_id, min_lvl],
    ).fetchall()
    if not rows:
        return [], 0, min_lvl
    paths = [r['path'] for r in rows]
    total = sum(r['cc'] for r in rows)
    return paths, total, min_lvl


def get_tree_node(view_id: str, path: str = '') -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data, children, claim_ids FROM tree_nodes WHERE view_id = ? AND path = ?",
            [view_id, path]
        ).fetchone()
        if row:
            node = json.loads(row['data'])
            node['_children_segments'] = json.loads(row['children']) if row['children'] else []
            node['_claim_ids_sample'] = json.loads(row['claim_ids']) if row['claim_ids'] else []
            return node

        # Virtual root fallback: synthesize a root from the min-level nodes for
        # views whose path='' row was never materialized.
        if path != '':
            return None
        view_row = conn.execute(
            "SELECT data FROM views WHERE view_id = ?", [view_id]
        ).fetchone()
        if not view_row:
            return None
        view_meta = json.loads(view_row['data'])
        segments, claim_count, _ = _virtual_root_segments(conn, view_id)
        if not segments:
            return None
        return {
            'node_id': f'{view_id}_root',
            'name': view_meta.get('name', view_id),
            'view_id': view_id,
            'path': [],
            'description': view_meta.get('description', ''),
            'level': 0,
            'claim_count': claim_count,
            '_virtual_root': True,
            '_children_segments': sorted(segments),
            '_claim_ids_sample': [],
        }


# When a parent node has more than this many direct children (e.g. by_data
# L1 categories like "physical" hold ~55k measurement leaves), don't try to
# materialize them all into a sidebar tree — that response gets tens of MB
# big and the browser DOM chokes. We fall back to a SQL-side top-N-by-claim
# query and the frontend exposes a search box for finding niche children.
WIDE_NODE_THRESHOLD = 500
WIDE_NODE_TOP_N = 500


def _fetch_direct_children_top_n(
    conn: sqlite3.Connection,
    view_id: str,
    parent_path: str,
    limit: int,
) -> list[sqlite3.Row]:
    """Fast path: pull top-N direct children by claim_count via one SQL call.

    Uses the tree_nodes.level column (parent_level + 1) so we don't pick up
    grand-children in subtree LIKE matches. Falls back to a simpler
    ``path NOT LIKE 'parent/%/%'`` heuristic if level is missing.
    """
    parent_row = conn.execute(
        "SELECT level FROM tree_nodes WHERE view_id = ? AND path = ?",
        [view_id, parent_path],
    ).fetchone()
    parent_level = parent_row['level'] if parent_row else None
    pattern = (parent_path + '/%') if parent_path else '%'

    if parent_level is not None:
        return conn.execute(
            "SELECT path, name, claim_count, children "
            "FROM tree_nodes WHERE view_id = ? AND level = ? "
            "AND path LIKE ? "
            "ORDER BY claim_count DESC LIMIT ?",
            [view_id, parent_level + 1, pattern, limit],
        ).fetchall()
    # No level recorded — exclude grandchildren by structural pattern.
    grand_pattern = (parent_path + '/%/%') if parent_path else '%/%'
    return conn.execute(
        "SELECT path, name, claim_count, children "
        "FROM tree_nodes WHERE view_id = ? AND path LIKE ? "
        "AND path NOT LIKE ? "
        "ORDER BY claim_count DESC LIMIT ?",
        [view_id, pattern, grand_pattern, limit],
    ).fetchall()


def get_tree_children(view_id: str, parent_path: str = '',
                      limit_top_n: int | None = None) -> list[dict]:
    """Return the direct children of ``parent_path`` in ``view_id``.

    For wide parents (>WIDE_NODE_THRESHOLD children) we never load all
    of them — we cap to top-N by claim_count via a single SQL query.
    Set ``limit_top_n`` explicitly to force the cap regardless of width.
    """
    with get_conn() as conn:
        parent_node = conn.execute(
            "SELECT children FROM tree_nodes WHERE view_id = ? AND path = ?",
            [view_id, parent_path]
        ).fetchone()
        children_segments: list[str] = []
        if parent_node and parent_node['children']:
            children_segments = json.loads(parent_node['children'])

        # Virtual-root fallback: views without a materialized path='' row
        # (every view except ``by_data``) have their top-level nodes at
        # level=MIN(level). Synthesize the children list from SQL.
        if not children_segments and parent_path == '':
            segments, _, _ = _virtual_root_segments(conn, view_id)
            children_segments = sorted(segments)

        if not children_segments:
            return []

        # Wide-parent fast path — used for by_data L1 categories.
        is_wide = len(children_segments) > WIDE_NODE_THRESHOLD
        if is_wide or (limit_top_n and len(children_segments) > limit_top_n):
            cap = limit_top_n or WIDE_NODE_TOP_N
            rows = _fetch_direct_children_top_n(conn, view_id, parent_path, cap)
            results = []
            for r in rows:
                try:
                    name = r['name'] or r['path'].split('/')[-1].replace('_', ' ').title()
                except Exception:
                    name = r['path']
                child_children = json.loads(r['children']) if r['children'] else []
                results.append({
                    'name': name,
                    'view_id': view_id,
                    'claim_count': r['claim_count'] or 0,
                    'has_children': len(child_children) > 0,
                    '_path': r['path'],
                    '_truncated': True,  # signals frontend to expose search
                })
            return results

        child_paths = []
        for seg in sorted(children_segments):
            child_paths.append(f"{parent_path}/{seg}" if parent_path else seg)

        # Batch fetch: chunk into groups of 900 to stay under SQLite variable limit
        row_map: dict[str, sqlite3.Row] = {}
        CHUNK = 900
        for i in range(0, len(child_paths), CHUNK):
            batch = child_paths[i : i + CHUNK]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT path, data, children FROM tree_nodes "
                f"WHERE view_id = ? AND path IN ({placeholders})",
                [view_id] + batch,
            ).fetchall()
            for r in rows:
                row_map[r["path"]] = r

        results = []
        for cp in child_paths:
            row = row_map.get(cp)
            if row:
                child = json.loads(row['data'])
                child_children = json.loads(row['children']) if row['children'] else []
                child['has_children'] = len(child_children) > 0
                child['_path'] = cp
                results.append(child)
        return results


def search_tree_children(view_id: str, parent_path: str, query: str,
                         limit: int = 100) -> list[dict]:
    """Search direct children of ``parent_path`` whose path or name matches.

    Designed for the by_data view, where an L1 category like "physical" holds
    50k+ measurement leaves and a top-N listing isn't enough — researchers
    often want to find a specific measurement (e.g. "thermal conductivity")
    by name. Matches are case-insensitive substring; ranked by claim_count.
    """
    q = (query or '').strip().lower()
    if not q:
        return []
    pattern = f"%{q}%"
    with get_conn() as conn:
        parent_row = conn.execute(
            "SELECT level FROM tree_nodes WHERE view_id = ? AND path = ?",
            [view_id, parent_path],
        ).fetchone()
        parent_level = parent_row['level'] if parent_row else None
        path_pattern = (parent_path + '/%') if parent_path else '%'
        if parent_level is not None:
            rows = conn.execute(
                "SELECT path, name, claim_count FROM tree_nodes "
                "WHERE view_id = ? AND level = ? AND path LIKE ? "
                "AND (LOWER(path) LIKE ? OR LOWER(name) LIKE ?) "
                "ORDER BY claim_count DESC LIMIT ?",
                [view_id, parent_level + 1, path_pattern,
                 pattern, pattern, limit],
            ).fetchall()
        else:
            grand_pattern = (parent_path + '/%/%') if parent_path else '%/%'
            rows = conn.execute(
                "SELECT path, name, claim_count FROM tree_nodes "
                "WHERE view_id = ? AND path LIKE ? AND path NOT LIKE ? "
                "AND (LOWER(path) LIKE ? OR LOWER(name) LIKE ?) "
                "ORDER BY claim_count DESC LIMIT ?",
                [view_id, path_pattern, grand_pattern,
                 pattern, pattern, limit],
            ).fetchall()
    return [
        {
            'path': r['path'],
            'name': r['name'] or r['path'].split('/')[-1].replace('_', ' ').title(),
            'claim_count': r['claim_count'] or 0,
        }
        for r in rows
    ]


def get_tree_with_depth(view_id: str, path: str = '', depth: int = 1) -> dict:
    node = get_tree_node(view_id, path)
    if not node:
        return None

    if depth > 0:
        children = get_tree_children(view_id, path)
        if depth > 1:
            for child in children:
                child_path = child.get('_path', '')
                child['children_data'] = get_tree_children(view_id, child_path)
        node['children_data'] = children
    return node


def get_claims_at_node(view_id: str, path: str, limit: int = 50,
                       offset: int = 0) -> dict:
    """Returns {claims: [...], total: int, children_summary: [...]} for pagination.

    If the node has direct claim_ids, returns those (leaf node).
    If not (parent/category node), returns children_summary with subcategory
    info so the frontend can render a navigable grid instead.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT claim_ids, data, children FROM tree_nodes "
            "WHERE view_id = ? AND path = ?",
            [view_id, path]
        ).fetchone()
        if not row:
            return {"claims": [], "total": 0}

        claim_ids = json.loads(row['claim_ids']) if row['claim_ids'] else []
        node_data = json.loads(row['data']) if row['data'] else {}
        # Prefer the legacy inline ``data.children`` (some older builders wrote
        # this) but fall back to the authoritative tree_nodes.children column —
        # most views, including by_data, only populate the SQL column.
        children_names = node_data.get('children') or (
            json.loads(row['children']) if row['children'] else []
        )

        if children_names:
            aggregate_total = node_data.get('claim_count', 0)

            top_children = node_data.get('top_children')
            if top_children:
                return {
                    "claims": [],
                    "total": aggregate_total,
                    "children_summary": top_children,
                }

            # For wide nodes (>2000 direct children, e.g. by_data L1) skip the
            # batched IN-list fetch and use a single ORDER BY + LIMIT query.
            CHILD_SUMMARY_TOP_N = 200
            if len(children_names) > 2000:
                top_rows = _fetch_direct_children_top_n(
                    conn, view_id, path, CHILD_SUMMARY_TOP_N,
                )
                children_summary = [
                    {
                        "name": cr['name'] or cr['path'].split('/')[-1].replace('_', ' ').title(),
                        "path": cr['path'],
                        "claim_count": cr['claim_count'] or 0,
                    }
                    for cr in top_rows
                ]
                return {
                    "claims": [],
                    "total": aggregate_total,
                    "children_summary": children_summary,
                }

            results = []
            for i in range(0, len(children_names), 999):
                batch = children_names[i:i+999]
                paths = [path + '/' + c if path else c for c in batch]
                ph = ','.join('?' * len(paths))
                child_rows = conn.execute(
                    f"SELECT path, name, claim_count FROM tree_nodes "
                    f"WHERE view_id = ? AND path IN ({ph})",
                    [view_id] + paths
                ).fetchall()
                results.extend(child_rows)

            children_summary = sorted(
                [
                    {
                        "name": cr['name'] or cr['path'].split('/')[-1].replace('_', ' ').title(),
                        "path": cr['path'],
                        "claim_count": cr['claim_count'] or 0,
                    }
                    for cr in results
                ],
                key=lambda x: x['claim_count'],
                reverse=True,
            )[:CHILD_SUMMARY_TOP_N]

            return {
                "claims": [],
                "total": aggregate_total,
                "children_summary": children_summary,
            }

        if not claim_ids:
            if path:
                # Use claim_view_map table for fast lookups across all views
                has_map = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claim_view_map'"
                ).fetchone()
                if has_map:
                    total_row = conn.execute(
                        "SELECT COUNT(*) FROM claim_view_map "
                        "WHERE view_id = ? AND (path = ? OR path LIKE ?)",
                        [view_id, path, path + '/%']
                    ).fetchone()
                    total = total_row[0]
                    rows = conn.execute(
                        "SELECT c.data FROM claims c JOIN claim_view_map m "
                        "ON c.claim_id = m.claim_id "
                        "WHERE m.view_id = ? AND (m.path = ? OR m.path LIKE ?) "
                        "LIMIT ? OFFSET ?",
                        [view_id, path, path + '/%', limit, offset]
                    ).fetchall()
                    claims = [json.loads(r['data']) for r in rows]
                    enrich_claims_with_source(claims, conn)
                    return {"claims": claims, "total": total}
                elif view_id == 'by_claim_type':
                    leaf_type = path.split('/')[-1]
                    total_row = conn.execute(
                        "SELECT COUNT(*) FROM claims WHERE claim_type = ?",
                        [leaf_type]
                    ).fetchone()
                    total = total_row[0]
                    rows = conn.execute(
                        "SELECT data FROM claims WHERE claim_type = ? "
                        "LIMIT ? OFFSET ?",
                        [leaf_type, limit, offset]
                    ).fetchall()
                    claims = [json.loads(r['data']) for r in rows]
                    enrich_claims_with_source(claims, conn)
                    return {"claims": claims, "total": total}
            return {"claims": [], "total": 0}

        total = len(claim_ids)
        selected = claim_ids[offset:offset+limit]
        if not selected:
            return {"claims": [], "total": total}
        placeholders = ','.join('?' * len(selected))
        rows = conn.execute(
            f"SELECT data FROM claims WHERE claim_id IN ({placeholders})",
            selected
        ).fetchall()
        claims = [json.loads(r['data']) for r in rows]
        enrich_claims_with_source(claims, conn)
        return {"claims": claims, "total": total}


def get_contradictions(
    view_id: Optional[str] = None,
    node_path: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    confirmed_only: bool = True,
) -> dict:
    """Return stored contradictions, optionally filtered by view/node."""
    with get_conn() as conn:
        where_parts = []
        params: list = []
        if confirmed_only:
            where_parts.append("ct.gemini_verdict = 'confirmed'")
        if view_id:
            where_parts.append("ct.view_id = ?")
            params.append(view_id)
        if node_path:
            where_parts.append("(ct.node_path = ? OR ct.node_path LIKE ?)")
            params.append(node_path)
            params.append(node_path + "/%")

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        total = conn.execute(
            f"SELECT COUNT(*) FROM contradictions ct {where_sql}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT ct.*, "
            f"c1.data as data_1, c1.claim_contextualized as ctx_1, "
            f"c2.data as data_2, c2.claim_contextualized as ctx_2 "
            f"FROM contradictions ct "
            f"LEFT JOIN claims c1 ON ct.claim_id_1 = c1.claim_id "
            f"LEFT JOIN claims c2 ON ct.claim_id_2 = c2.claim_id "
            f"{where_sql} "
            f"ORDER BY ct.confidence DESC "
            f"LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        # Hydrate each side into the same shape as /api/search results so
        # the frontend can run them through `renderClaim` / formatAcsCitation
        # uniformly. `claim_contextualized` lives on the claims column (not
        # in `data`), and authors / venue / year live on the sources table
        # keyed by DOI — pulling both in here is what gives the contradiction
        # cards visual parity with normal search cards.
        def _build_side(data_json, ctx, claim_id):
            claim = json.loads(data_json) if data_json else {}
            if ctx:
                claim["claim_contextualized"] = ctx
            claim.setdefault("claim_id", claim_id)
            return claim

        sides: list[dict] = []
        pairs: list[tuple[dict, dict]] = []
        for r in rows:
            c1 = _build_side(r["data_1"], r["ctx_1"], r["claim_id_1"])
            c2 = _build_side(r["data_2"], r["ctx_2"], r["claim_id_2"])
            pairs.append((c1, c2))
            sides.append(c1)
            sides.append(c2)

        # One enrichment pass across the full window — avoids N+1 over the
        # `sources` table when the panel asks for limit=50.
        if sides:
            enrich_claims_with_source(sides, conn)

        results = []
        for r, (c1, c2) in zip(rows, pairs):
            results.append({
                "id": r["id"],
                "claim_1": c1,
                "claim_2": c2,
                "paw_verdict": r["paw_verdict"],
                "gemini_verdict": r["gemini_verdict"],
                "gemini_explanation": r["gemini_explanation"],
                "confidence": r["confidence"],
                "view_id": r["view_id"],
                "node_path": r["node_path"],
            })
        return {"contradictions": results, "total": total}


def get_reading_list(view_id: str, path: str, limit: int = 15) -> dict:
    """Build a tiered reading list of papers contributing to a topic.

    Groups papers into Foundational (>50 cites), Key Results (10-50 cites or
    top recent), and Recent Advances (<10 cites, last 2 years).
    """
    current_year = datetime.now().year

    with get_conn() as conn:
        row = conn.execute(
            "SELECT name FROM tree_nodes WHERE view_id = ? AND path = ?",
            [view_id, path],
        ).fetchone()
        topic_name = (row["name"] if row else None) or (path.split("/")[-1] if path else "Root")

        # Gather claim_ids for this topic from tree_nodes (fast, pre-indexed)
        all_cids: list[str] = []
        node_rows = conn.execute(
            "SELECT claim_ids FROM tree_nodes "
            "WHERE view_id = ? AND (path = ? OR path LIKE ?) "
            "AND claim_ids IS NOT NULL AND claim_ids != '[]'",
            [view_id, path, path + '/%'],
        ).fetchall()
        for nr in node_rows:
            try:
                all_cids.extend(json.loads(nr["claim_ids"]))
            except (json.JSONDecodeError, TypeError):
                pass

        if all_cids:
            cid_set = list(set(all_cids))[:3000]
            placeholders = ",".join("?" for _ in cid_set)
            doi_rows = conn.execute(
                f"SELECT source_doi, COUNT(*) as cnt FROM claims "
                f"WHERE claim_id IN ({placeholders}) AND source_doi != '' "
                f"GROUP BY source_doi ORDER BY cnt DESC",
                cid_set,
            ).fetchall()
        else:
            # Fallback: use claim_view_map or view_paths LIKE
            has_map = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claim_view_map'"
            ).fetchone()
            if has_map:
                # Two-step: get claim_ids from map, then group by DOI
                map_cids = conn.execute(
                    "SELECT claim_id FROM claim_view_map "
                    "WHERE view_id = ? AND (path = ? OR path LIKE ?) LIMIT 5000",
                    [view_id, path, path + '/%'],
                ).fetchall()
                if map_cids:
                    cids = [r["claim_id"] for r in map_cids]
                    placeholders = ",".join("?" for _ in cids)
                    doi_rows = conn.execute(
                        f"SELECT source_doi, COUNT(*) as cnt FROM claims "
                        f"WHERE claim_id IN ({placeholders}) AND source_doi != '' "
                        f"GROUP BY source_doi ORDER BY cnt DESC",
                        cids,
                    ).fetchall()
                else:
                    doi_rows = []
            else:
                like_pattern = f'%"{view_id}"%{path}%'
                doi_rows = conn.execute(
                    "SELECT source_doi, COUNT(*) as cnt FROM claims "
                    "WHERE view_paths LIKE ? AND source_doi != '' "
                    "GROUP BY source_doi ORDER BY cnt DESC LIMIT 100",
                    [like_pattern],
                ).fetchall()

        if not doi_rows:
            return {"topic": topic_name, "view_id": view_id, "path": path,
                    "total_papers": 0, "tiers": []}

        doi_claim_counts = {r["source_doi"]: r["cnt"] for r in doi_rows}
        dois = list(doi_claim_counts.keys())

        # Fetch source metadata
        doi_placeholders = ",".join("?" for _ in dois)
        source_rows = conn.execute(
            f"SELECT data FROM sources WHERE doi IN ({doi_placeholders})",
            dois,
        ).fetchall()

    papers = []
    for sr in source_rows:
        src = json.loads(sr["data"])
        doi = src.get("doi", "")
        authors = src.get("authors", [])
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except (json.JSONDecodeError, TypeError):
                authors = [authors] if authors else []

        papers.append({
            "doi": doi,
            "title": src.get("title", ""),
            "authors": authors,
            "year": src.get("year", 0) or 0,
            "venue": src.get("venue", ""),
            "citation_count": src.get("citation_count", 0) or 0,
            "abstract": (src.get("abstract") or "")[:300],
            "claim_count": doi_claim_counts.get(doi, 0),
            "open_access_url": src.get("open_access_url", ""),
        })

    # Tier assignment
    foundational = []
    key_results = []
    recent = []

    for p in papers:
        cites = p["citation_count"]
        year = p["year"]
        if cites > 50:
            foundational.append(p)
        elif cites >= 10 or (year >= current_year - 2 and cites >= 3):
            key_results.append(p)
        else:
            recent.append(p)

    for tier in [foundational, key_results, recent]:
        tier.sort(key=lambda x: x["citation_count"], reverse=True)

    # Distribute the limit budget across tiers (total ~15 papers)
    n_tiers = bool(foundational) + bool(key_results) + bool(recent)
    per_tier = max(3, limit // max(n_tiers, 1))

    tiers = []
    if foundational:
        tiers.append({"name": "Foundational", "papers": foundational[:per_tier]})
    if key_results:
        tiers.append({"name": "Key Results", "papers": key_results[:per_tier]})
    if recent:
        tiers.append({"name": "Recent Advances", "papers": recent[:per_tier]})

    total = len(foundational) + len(key_results) + len(recent)
    return {
        "topic": topic_name,
        "view_id": view_id,
        "path": path,
        "total_papers": total,
        "tiers": tiers,
    }


def get_temporal_overlay(view_id: str, path: str) -> dict:
    """Get year-by-year breakdown of claims at a node.

    Returns {year: {claim_count, types: {type: count}, is_surge, is_decline}}.
    Detects surges (>2x year-over-year growth) and declines.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT claim_ids FROM tree_nodes WHERE view_id = ? AND path = ?",
            [view_id, path],
        ).fetchone()
        if not row or not row["claim_ids"]:
            return {"years": {}, "total": 0}

        claim_ids = json.loads(row["claim_ids"])
        if not claim_ids:
            return {"years": {}, "total": 0}

        placeholders = ",".join("?" * len(claim_ids))
        rows = conn.execute(
            f"SELECT c.claim_type, s.year "
            f"FROM claims c JOIN sources s ON c.source_doi = s.doi "
            f"WHERE c.claim_id IN ({placeholders})",
            claim_ids,
        ).fetchall()

    from collections import Counter, defaultdict
    year_types = defaultdict(Counter)
    for r in rows:
        year = r["year"] or 0
        if year > 0:
            year_types[year][r["claim_type"]] += 1

    years = {}
    sorted_years = sorted(year_types.keys())
    for i, year in enumerate(sorted_years):
        count = sum(year_types[year].values())
        prev_count = sum(year_types[sorted_years[i - 1]].values()) if i > 0 else 0
        is_surge = prev_count > 0 and count > 2 * prev_count
        is_decline = prev_count > 5 and count < prev_count * 0.5
        years[year] = {
            "claim_count": count,
            "types": dict(year_types[year]),
            "is_surge": is_surge,
            "is_decline": is_decline,
        }

    return {"years": years, "total": sum(sum(c.values()) for c in year_types.values())}


def get_evolution_timeline(view_id: str, path: str) -> dict:
    """Rich evolution timeline for a node: claims per year, top papers, surges.

    Returns {view_id, path, years: {year: {claim_count, types, top_papers, is_surge}},
             new_subtopics: {year: [topics]}}.
    """
    with get_conn() as conn:
        # Gather claim_ids from this node and its children
        all_cids = []
        node_rows = conn.execute(
            "SELECT claim_ids FROM tree_nodes "
            "WHERE view_id = ? AND (path = ? OR path LIKE ?) "
            "AND claim_ids IS NOT NULL AND claim_ids != '[]'",
            [view_id, path, path + '/%'],
        ).fetchall()
        for nr in node_rows:
            try:
                all_cids.extend(json.loads(nr["claim_ids"]))
            except (json.JSONDecodeError, TypeError):
                pass

        if not all_cids:
            # Fallback: use view_paths LIKE match
            like_pattern = f'%"{view_id}"%{path}%'
            rows = conn.execute(
                "SELECT c.claim_type, c.source_doi, c.source_paper_title, "
                "       s.year, s.citation_count "
                "FROM claims c JOIN sources s ON c.source_doi = s.doi "
                "WHERE c.view_paths LIKE ? LIMIT 5000",
                [like_pattern],
            ).fetchall()
        else:
            cid_set = list(set(all_cids))[:5000]
            placeholders = ",".join("?" * len(cid_set))
            rows = conn.execute(
                f"SELECT c.claim_type, c.source_doi, c.source_paper_title, "
                f"       s.year, s.citation_count "
                f"FROM claims c JOIN sources s ON c.source_doi = s.doi "
                f"WHERE c.claim_id IN ({placeholders})",
                cid_set,
            ).fetchall()

        if not rows:
            return {"view_id": view_id, "path": path, "years": {}}

    from collections import Counter, defaultdict
    year_data = defaultdict(lambda: {"types": Counter(), "papers": {}})

    for r in rows:
        year = r["year"] or 0
        if year <= 0:
            continue
        yd = year_data[year]
        yd["types"][r["claim_type"]] += 1
        doi = r["source_doi"]
        if doi not in yd["papers"] or (r["citation_count"] or 0) > yd["papers"][doi].get("citations", 0):
            yd["papers"][doi] = {
                "doi": doi,
                "title": r["source_paper_title"],
                "citations": r["citation_count"] or 0,
            }

    years = {}
    sorted_years = sorted(year_data.keys())
    for i, year in enumerate(sorted_years):
        yd = year_data[year]
        count = sum(yd["types"].values())
        prev_count = sum(year_data[sorted_years[i - 1]]["types"].values()) if i > 0 else 0
        top_papers = sorted(yd["papers"].values(), key=lambda p: p["citations"], reverse=True)[:5]
        years[year] = {
            "claim_count": count,
            "types": dict(yd["types"]),
            "top_papers": top_papers,
            "is_surge": prev_count > 0 and count > 2 * prev_count,
        }

    return {"view_id": view_id, "path": path, "years": years}


def get_by_time_period(decade: str = None, year: int = None, quarter: str = None) -> dict:
    """Browse claims organized by time period.

    Hierarchy: decade -> year -> quarter -> claims (grouped by dominant topic).
    """
    with get_conn() as conn:
        if quarter and year:
            q_map = {"q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12)}
            m1, m2 = q_map.get(quarter.lower(), (1, 12))
            rows = conn.execute(
                "SELECT c.data, s.year FROM claims c JOIN sources s ON c.source_doi = s.doi "
                "WHERE s.year = ?",
                [year],
            ).fetchall()
            claims = [json.loads(r["data"]) for r in rows]
            return {"period": f"{year}_{quarter}", "claims": claims[:200], "total": len(claims)}

        elif year:
            rows = conn.execute(
                "SELECT c.claim_type, COUNT(*) as cnt "
                "FROM claims c JOIN sources s ON c.source_doi = s.doi "
                "WHERE s.year = ? GROUP BY c.claim_type ORDER BY cnt DESC",
                [year],
            ).fetchall()
            total = sum(r["cnt"] for r in rows)

            claim_rows = conn.execute(
                "SELECT c.data FROM claims c JOIN sources s ON c.source_doi = s.doi "
                "WHERE s.year = ? ORDER BY s.citation_count DESC LIMIT 50",
                [year],
            ).fetchall()
            claims = [json.loads(r["data"]) for r in claim_rows]
            enrich_claims_with_source(claims, conn)

            return {
                "period": str(year),
                "type_distribution": {r["claim_type"]: r["cnt"] for r in rows},
                "total": total,
                "claims": claims,
            }

        elif decade:
            start = int(decade.rstrip("s"))
            rows = conn.execute(
                "SELECT s.year, COUNT(*) as cnt "
                "FROM claims c JOIN sources s ON c.source_doi = s.doi "
                "WHERE s.year >= ? AND s.year < ? GROUP BY s.year ORDER BY s.year",
                [start, start + 10],
            ).fetchall()
            return {
                "period": decade,
                "years": {r["year"]: r["cnt"] for r in rows},
                "total": sum(r["cnt"] for r in rows),
            }

        else:
            rows = conn.execute(
                "SELECT CAST((s.year / 10) * 10 AS TEXT) || 's' as decade, COUNT(*) as cnt "
                "FROM claims c JOIN sources s ON c.source_doi = s.doi "
                "WHERE s.year > 0 GROUP BY decade ORDER BY decade",
            ).fetchall()
            return {
                "view": "by_time_period",
                "decades": {r["decade"]: r["cnt"] for r in rows},
                "total": sum(r["cnt"] for r in rows),
            }


def insert_claim(claim_data: dict, skip_validation: bool = False):
    """Insert a single claim into the database (for live updates).

    Validates the claim before insertion. Invalid claims are logged and skipped
    unless skip_validation is True.
    """
    if not skip_validation:
        from askchem.validation import validate_claim
        result = validate_claim(claim_data)
        if not result.is_valid:
            import logging
            logging.getLogger(__name__).warning(
                "Claim %s failed validation (%s): %s",
                claim_data.get('claim_id', '?'), result.summary(),
                "; ".join(e.message for e in result.errors),
            )
            return False

    with get_conn(readonly=False) as conn:
        searchable = build_searchable_text(claim_data)
        conn.execute(
            "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim_data.get('claim_id', ''),
                claim_data.get('claim_type', ''),
                claim_data.get('source_doi', ''),
                claim_data.get('source_paper_title', ''),
                claim_data.get('confidence', ''),
                claim_data.get('location_in_paper', ''),
                claim_data.get('verbatim_quote', ''),
                claim_data.get('extraction_model', ''),
                claim_data.get('extraction_version', ''),
                claim_data.get('extracted_at', ''),
                json.dumps(claim_data.get('view_paths', {})),
                json.dumps(claim_data),
            )
        )
        conn.execute("DELETE FROM claims_fts WHERE claim_id = ?", [claim_data['claim_id']])
        conn.execute(
            "INSERT INTO claims_fts(claim_id, claim_type, source_paper_title, verbatim_quote, searchable_text) VALUES (?,?,?,?,?)",
            (claim_data['claim_id'], claim_data.get('claim_type', ''),
             claim_data.get('source_paper_title', ''), claim_data.get('verbatim_quote', ''), searchable)
        )
        conn.commit()
    return True


def insert_source(source_data: dict):
    """Insert a single source into the database (for live updates)."""
    with get_conn(readonly=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                source_data.get('doi', ''),
                source_data.get('title', ''),
                json.dumps(source_data.get('authors', [])),
                source_data.get('year', 0),
                source_data.get('venue', ''),
                source_data.get('abstract', ''),
                source_data.get('citation_count', 0),
                source_data.get('open_access_url', ''),
                json.dumps(source_data),
            )
        )
        conn.commit()


# ── Community Flags ──────────────────────────────────────────────────────────

def add_flag(claim_id: str, flag_type: str, category: str = '',
             comment: str = '', suggested_fix: str = '',
             reporter_name: str = '', reporter_email: str = '') -> int:
    """Add a community flag for a claim. Returns flag ID."""
    valid_types = ('wrong_claim', 'wrong_classification', 'not_chemistry',
                   'duplicate', 'low_quality', 'other')
    if flag_type not in valid_types:
        raise ValueError(f"flag_type must be one of {valid_types}")
    with get_runtime_conn(readonly=False) as conn:
        c = conn.execute(
            "INSERT INTO community_flags "
            "(claim_id, flag_type, category, comment, suggested_fix, "
            "reporter_name, reporter_email, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (claim_id, flag_type, category, comment, suggested_fix,
             reporter_name, reporter_email, 'open',
             datetime.now().isoformat())
        )
        conn.commit()
        return c.lastrowid


def get_flags_for_claim(claim_id: str) -> list[dict]:
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM community_flags WHERE claim_id = ? ORDER BY created_at DESC",
            [claim_id]
        ).fetchall()
        return [dict(r) for r in rows]


def get_flag_summary() -> dict:
    """Aggregate flag counts by type and status."""
    with get_runtime_conn() as conn:
        by_type = conn.execute(
            "SELECT flag_type, COUNT(*) as cnt FROM community_flags "
            "GROUP BY flag_type ORDER BY cnt DESC"
        ).fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM community_flags "
            "GROUP BY status ORDER BY cnt DESC"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM community_flags").fetchone()[0]
        flagged_claims = conn.execute(
            "SELECT COUNT(DISTINCT claim_id) FROM community_flags WHERE status = 'open'"
        ).fetchone()[0]
        return {
            "total_flags": total,
            "flagged_claims": flagged_claims,
            "by_type": {r['flag_type']: r['cnt'] for r in by_type},
            "by_status": {r['status']: r['cnt'] for r in by_status},
        }


# ── Living-tree feedback ─────────────────────────────────────────────────────

_LTREE_FEEDBACK_KINDS = ('mislabeled', 'misplaced', 'duplicate', 'wrong_parent',
                         'missing', 'other')


def _ensure_ltree_feedback(conn):
    """Create the table on demand so hosts that never ran init_db (e.g. the
    HF-distributed prod DB) still accept feedback."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ltree_feedback ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, view_id TEXT NOT NULL, node_id TEXT, "
        "doi TEXT, kind TEXT NOT NULL, comment TEXT, reporter_name TEXT, "
        "reporter_email TEXT, ip_hash TEXT, status TEXT DEFAULT 'open', "
        "created_at TEXT NOT NULL, reviewed_at TEXT, reviewer_notes TEXT)")


def add_ltree_feedback(view_id: str, kind: str, node_id: str = None, doi: str = None,
                       comment: str = '', reporter_name: str = '',
                       reporter_email: str = '', ip_hash: str = '') -> int:
    """Record community feedback on a living-tree node/placement. Returns row id."""
    if kind not in _LTREE_FEEDBACK_KINDS:
        raise ValueError(f"kind must be one of {_LTREE_FEEDBACK_KINDS}")
    with get_runtime_conn(readonly=False) as conn:
        _ensure_ltree_feedback(conn)
        c = conn.execute(
            "INSERT INTO ltree_feedback (view_id,node_id,doi,kind,comment,"
            "reporter_name,reporter_email,ip_hash,status,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (view_id, node_id, doi, kind, comment, reporter_name, reporter_email,
             ip_hash, 'open', datetime.now().isoformat()))
        conn.commit()
        return c.lastrowid


def get_ltree_feedback_summary() -> dict:
    """Aggregate living-tree feedback counts (safe if the table is absent)."""
    with get_runtime_conn() as conn:
        try:
            total = conn.execute("SELECT COUNT(*) FROM ltree_feedback").fetchone()[0]
            open_ct = conn.execute(
                "SELECT COUNT(*) FROM ltree_feedback WHERE status='open'").fetchone()[0]
            by_kind = conn.execute(
                "SELECT kind, COUNT(*) c FROM ltree_feedback GROUP BY kind "
                "ORDER BY c DESC").fetchall()
        except Exception:
            return {"total": 0, "open": 0, "by_kind": {}}
        return {"total": total, "open": open_ct,
                "by_kind": {r['kind']: r['c'] for r in by_kind}}


def list_flags(status: str = None, flag_type: str = None,
               limit: int = 50, offset: int = 0, public_only: bool = False) -> list[dict]:
    sql = "SELECT f.*, c.data as claim_data FROM community_flags f " \
          "LEFT JOIN claims c ON f.claim_id = c.claim_id WHERE 1=1"
    params = []
    if public_only:
        sql += " AND f.status IN ('resolved', 'reviewed', 'dismissed')"
    if status:
        sql += " AND f.status = ?"
        params.append(status)
    if flag_type:
        sql += " AND f.flag_type = ?"
        params.append(flag_type)
    sql += " ORDER BY f.created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with get_runtime_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get('claim_data'):
                d['claim_data'] = json.loads(d['claim_data'])
            results.append(d)
        return results


def resolve_flag(flag_id: int, status: str, reviewer_notes: str = ''):
    valid = ('reviewed', 'resolved', 'dismissed')
    if status not in valid:
        raise ValueError(f"status must be one of {valid}")
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "UPDATE community_flags SET status = ?, reviewed_at = ?, reviewer_notes = ? "
            "WHERE id = ?",
            (status, datetime.now().isoformat(), reviewer_notes, flag_id)
        )
        conn.commit()


# ── Paper Validation ─────────────────────────────────────────────────────────

def save_paper_validation(doi: str, validation: dict):
    with get_conn(readonly=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO paper_validations "
            "(doi, crossref_verified, has_abstract, is_retracted, journal, publisher, "
            "is_chemistry, validation_score, validated_at, validation_data) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (doi, validation.get('crossref_verified', 0),
             validation.get('has_abstract', 0),
             validation.get('is_retracted', 0),
             validation.get('journal', ''),
             validation.get('publisher', ''),
             validation.get('is_chemistry', 1),
             validation.get('validation_score', 0),
             datetime.now().isoformat(),
             json.dumps(validation))
        )
        conn.commit()


def get_paper_validation(doi: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM paper_validations WHERE doi = ?", [doi]
        ).fetchone()
        return dict(row) if row else None


def add_submission(doi: str, name: str = '', email: str = '', notes: str = '') -> int:
    """Record a user paper submission. Returns the submission ID."""
    from datetime import datetime
    with get_runtime_conn(readonly=False) as conn:
        c = conn.execute(
            "INSERT INTO submissions (doi, submitted_at, status, submitter_name, submitter_email, notes) VALUES (?,?,?,?,?,?)",
            (doi, datetime.utcnow().isoformat(), 'pending', name, email, notes)
        )
        conn.commit()
        return c.lastrowid


def get_submission(submission_id: int) -> Optional[dict]:
    with get_runtime_conn() as conn:
        row = conn.execute("SELECT * FROM submissions WHERE id = ?", [submission_id]).fetchone()
        return dict(row) if row else None


def list_submissions(status: str = None, limit: int = 50) -> list[dict]:
    with get_runtime_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM submissions WHERE status = ? ORDER BY submitted_at DESC LIMIT ?",
                [status, limit]
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM submissions ORDER BY submitted_at DESC LIMIT ?",
                [limit]
            ).fetchall()
        return [dict(r) for r in rows]


def update_submission(submission_id: int, status: str, result: dict = None):
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "UPDATE submissions SET status = ?, result = ? WHERE id = ?",
            (status, json.dumps(result) if result else None, submission_id)
        )
        conn.commit()


def add_subscription(
    user_id: str,
    sub_type: str,
    target: str,
    frequency: str = "weekly",
    email: str | None = None,
) -> dict:
    """Add a subscription for a logged-in user. Returns subscription_id."""
    with get_runtime_conn(readonly=False) as conn:
        existing = conn.execute(
            "SELECT id FROM subscriptions "
            "WHERE user_id = ? AND sub_type = ? AND target = ? AND is_active = 1 LIMIT 1",
            (user_id, sub_type, target),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE subscriptions SET frequency = ?, email = COALESCE(?, email) "
                "WHERE id = ?",
                (frequency, email, existing["id"]),
            )
            conn.commit()
            return {"subscription_id": existing["id"], "reactivated": False}

        cursor = conn.execute(
            "INSERT INTO subscriptions "
            "(email, sub_type, target, frequency, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (email, sub_type, target, frequency, datetime.now().isoformat(), user_id),
        )
        conn.commit()
        return {"subscription_id": cursor.lastrowid, "reactivated": False}


def get_user_subscriptions(user_id: str) -> list[dict]:
    """List active subscriptions for a logged-in user."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT id, sub_type, target, frequency, created_at, last_notified_at, email "
            "FROM subscriptions WHERE user_id = ? AND is_active = 1 "
            "ORDER BY created_at DESC",
            [user_id],
        ).fetchall()
    return [dict(r) for r in rows]


def get_subscription_row(sub_id: int) -> dict | None:
    """Return one subscription row or None."""
    with get_runtime_conn() as conn:
        row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", [sub_id]).fetchone()
    return dict(row) if row else None


def cancel_user_subscription(user_id: str, sub_id: int) -> None:
    """Cancel a subscription owned by this user."""
    with get_runtime_conn(readonly=False) as conn:
        row = conn.execute(
            "SELECT user_id FROM subscriptions WHERE id = ?", [sub_id]
        ).fetchone()
        if not row:
            raise ValueError("subscription not found")
        if row["user_id"] != user_id:
            raise ValueError("not your subscription")
        conn.execute("UPDATE subscriptions SET is_active = 0 WHERE id = ?", [sub_id])
        conn.commit()


# ── Bookmarks ──────────────────────────────────────────────────────────────

def add_bookmark(
    user_id: str,
    target_type: str,
    target_id: str,
    title: str | None = None,
    note: str | None = None,
) -> dict:
    """Add or update a bookmark. Idempotent on (user_id, target_type, target_id)."""
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "INSERT INTO bookmarks (user_id, target_type, target_id, title, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, target_type, target_id) DO UPDATE SET "
            "title = COALESCE(excluded.title, bookmarks.title), "
            "note = COALESCE(excluded.note, bookmarks.note)",
            (user_id, target_type, target_id, title, note, datetime.now().isoformat()),
        )
        row = conn.execute(
            "SELECT id, created_at FROM bookmarks "
            "WHERE user_id = ? AND target_type = ? AND target_id = ?",
            (user_id, target_type, target_id),
        ).fetchone()
        conn.commit()
    return {"id": row["id"], "created_at": row["created_at"]}


def remove_bookmark(user_id: str, target_type: str, target_id: str) -> bool:
    with get_runtime_conn(readonly=False) as conn:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE user_id = ? AND target_type = ? AND target_id = ?",
            (user_id, target_type, target_id),
        )
        conn.commit()
        return cur.rowcount > 0


def list_bookmarks(user_id: str, target_type: str | None = None, limit: int = 200) -> list[dict]:
    with get_runtime_conn() as conn:
        if target_type:
            rows = conn.execute(
                "SELECT id, target_type, target_id, title, note, created_at "
                "FROM bookmarks WHERE user_id = ? AND target_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, target_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, target_type, target_id, title, note, created_at "
                "FROM bookmarks WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def is_bookmarked(user_id: str, target_type: str, target_id: str) -> bool:
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM bookmarks WHERE user_id = ? AND target_type = ? AND target_id = ? LIMIT 1",
            (user_id, target_type, target_id),
        ).fetchone()
    return row is not None


# ── Saved searches ─────────────────────────────────────────────────────────

def add_saved_search(
    user_id: str,
    query: str,
    view: str | None = None,
    filters: dict | None = None,
    name: str | None = None,
) -> dict:
    filters_json = json.dumps(filters) if filters else None
    with get_runtime_conn(readonly=False) as conn:
        cur = conn.execute(
            "INSERT INTO saved_searches (user_id, name, query, view, filters, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, name, query, view, filters_json, datetime.now().isoformat()),
        )
        conn.commit()
        return {"id": cur.lastrowid}


def list_saved_searches(user_id: str, limit: int = 200) -> list[dict]:
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, query, view, filters, created_at FROM saved_searches "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("filters"):
            try:
                d["filters"] = json.loads(d["filters"])
            except Exception:
                d["filters"] = None
        out.append(d)
    return out


def delete_saved_search(user_id: str, saved_id: int) -> bool:
    with get_runtime_conn(readonly=False) as conn:
        cur = conn.execute(
            "DELETE FROM saved_searches WHERE id = ? AND user_id = ?",
            (saved_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ── Reading lists ──────────────────────────────────────────────────────────

def create_reading_list(
    user_id: str,
    name: str,
    description: str | None = None,
    is_public: bool = False,
) -> dict:
    now = datetime.now().isoformat()
    with get_runtime_conn(readonly=False) as conn:
        cur = conn.execute(
            "INSERT INTO reading_lists "
            "(user_id, name, description, is_public, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, name, description, 1 if is_public else 0, now, now),
        )
        conn.commit()
    return {"id": cur.lastrowid, "name": name, "created_at": now}


def list_reading_lists(user_id: str) -> list[dict]:
    """Return the user's reading lists with item counts."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT l.id, l.name, l.description, l.is_public, "
            "       l.created_at, l.updated_at, "
            "       (SELECT COUNT(*) FROM reading_list_items i WHERE i.list_id = l.id) AS item_count "
            "FROM reading_lists l "
            "WHERE l.user_id = ? "
            "ORDER BY l.updated_at DESC",
            [user_id],
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_reading_list(list_id: int, user_id: str | None = None) -> dict | None:
    """Return one user-owned reading list with its items.

    If ``user_id`` is ``None``, only public lists are returned.
    If ``user_id`` is provided, the owner can see their own private lists too.
    """
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT id, user_id, name, description, is_public, created_at, updated_at "
            "FROM reading_lists WHERE id = ?",
            [list_id],
        ).fetchone()
        if not row:
            return None
        lst = dict(row)
        if not lst["is_public"] and lst["user_id"] != user_id:
            return None
        items = conn.execute(
            "SELECT id, target_type, target_id, title, note, position, added_at "
            "FROM reading_list_items WHERE list_id = ? "
            "ORDER BY position ASC, added_at DESC",
            [list_id],
        ).fetchall()
        lst["items"] = [dict(i) for i in items]
        lst["item_count"] = len(lst["items"])
    return lst


def update_reading_list(
    list_id: int,
    user_id: str,
    name: str | None = None,
    description: str | None = None,
    is_public: bool | None = None,
) -> bool:
    """Rename/change a list. Ownership enforced."""
    fields = []
    params: list = []
    if name is not None:
        fields.append("name = ?")
        params.append(name)
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    if is_public is not None:
        fields.append("is_public = ?")
        params.append(1 if is_public else 0)
    if not fields:
        return True
    fields.append("updated_at = ?")
    params.append(datetime.now().isoformat())
    params.extend([list_id, user_id])
    with get_runtime_conn(readonly=False) as conn:
        cur = conn.execute(
            "UPDATE reading_lists SET " + ", ".join(fields) +
            " WHERE id = ? AND user_id = ?",
            params,
        )
        conn.commit()
        return cur.rowcount > 0


def delete_reading_list(list_id: int, user_id: str) -> bool:
    """Delete a list and its items. Ownership enforced."""
    with get_runtime_conn(readonly=False) as conn:
        row = conn.execute(
            "SELECT user_id FROM reading_lists WHERE id = ?", [list_id]
        ).fetchone()
        if not row or row["user_id"] != user_id:
            return False
        conn.execute("DELETE FROM reading_list_items WHERE list_id = ?", [list_id])
        cur = conn.execute(
            "DELETE FROM reading_lists WHERE id = ? AND user_id = ?",
            (list_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def _assert_list_ownership(conn, list_id: int, user_id: str) -> None:
    row = conn.execute(
        "SELECT user_id FROM reading_lists WHERE id = ?", [list_id]
    ).fetchone()
    if not row:
        raise ValueError("list not found")
    if row["user_id"] != user_id:
        raise ValueError("not your list")


def add_reading_list_item(
    list_id: int,
    user_id: str,
    target_type: str,
    target_id: str,
    title: str | None = None,
    note: str | None = None,
) -> dict:
    """Add an item to a reading list. Idempotent on (list_id, target_type, target_id)."""
    now = datetime.now().isoformat()
    with get_runtime_conn(readonly=False) as conn:
        _assert_list_ownership(conn, list_id, user_id)
        conn.execute(
            "INSERT INTO reading_list_items "
            "(list_id, target_type, target_id, title, note, position, added_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(list_id, target_type, target_id) DO UPDATE SET "
            "title = COALESCE(excluded.title, reading_list_items.title), "
            "note  = COALESCE(excluded.note,  reading_list_items.note)",
            (list_id, target_type, target_id, title, note, now),
        )
        conn.execute(
            "UPDATE reading_lists SET updated_at = ? WHERE id = ?",
            (now, list_id),
        )
        row = conn.execute(
            "SELECT id, added_at FROM reading_list_items "
            "WHERE list_id = ? AND target_type = ? AND target_id = ?",
            (list_id, target_type, target_id),
        ).fetchone()
        conn.commit()
    return {"id": row["id"], "added_at": row["added_at"]}


def remove_reading_list_item(
    list_id: int,
    user_id: str,
    target_type: str,
    target_id: str,
) -> bool:
    with get_runtime_conn(readonly=False) as conn:
        _assert_list_ownership(conn, list_id, user_id)
        cur = conn.execute(
            "DELETE FROM reading_list_items "
            "WHERE list_id = ? AND target_type = ? AND target_id = ?",
            (list_id, target_type, target_id),
        )
        if cur.rowcount:
            conn.execute(
                "UPDATE reading_lists SET updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), list_id),
            )
        conn.commit()
        return cur.rowcount > 0


def get_lists_containing(
    user_id: str,
    target_type: str,
    target_id: str,
) -> list[int]:
    """Return IDs of the user's lists that already contain this target."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT l.id FROM reading_lists l "
            "JOIN reading_list_items i ON i.list_id = l.id "
            "WHERE l.user_id = ? AND i.target_type = ? AND i.target_id = ?",
            (user_id, target_type, target_id),
        ).fetchall()
    return [r["id"] for r in rows]


# ── User-owned API keys ────────────────────────────────────────────────────

def list_user_api_keys(user_id: str) -> list[dict]:
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT key_id, name, tier, rate_limit, created_at, last_used_at, is_active, "
            "COALESCE(total_requests, 0) AS total_requests "
            "FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            [user_id],
        ).fetchall()
    return [dict(r) for r in rows]


def create_user_api_key(
    user_id: str,
    name: str,
    email: str = "",
    tier: str = "tier_1",
) -> dict:
    """Create an API key owned by user_id. Returns {key_id, api_key, name, tier}."""
    import secrets
    key_id = secrets.token_hex(8)
    raw_key = f"ac-{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    rpm = 1000 if tier in ("pro", "registered", "tier_3") else (
        500 if tier == "tier_2" else 200
    )
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "INSERT INTO api_keys "
            "(key_id, key_hash, name, email, tier, rate_limit, created_at, total_requests, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (key_id, key_hash, name, email, tier, rpm,
             datetime.now().isoformat(), user_id),
        )
        conn.commit()
    return {"key_id": key_id, "api_key": raw_key, "name": name, "tier": tier, "rate_limit": rpm}


def revoke_user_api_key(user_id: str, key_id: str) -> bool:
    with get_runtime_conn(readonly=False) as conn:
        row = conn.execute(
            "SELECT user_id FROM api_keys WHERE key_id = ?", [key_id]
        ).fetchone()
        if not row:
            return False
        if row["user_id"] != user_id:
            raise ValueError("not your api key")
        cur = conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE key_id = ?", [key_id]
        )
        conn.commit()
        return cur.rowcount > 0


def get_due_subscriptions() -> list[dict]:
    """Get active subscriptions that are due for notification.

    Falls back to the user's email from the users table when the subscription
    row has no email set (all authenticated subs).
    """
    now = datetime.now()
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT s.id, COALESCE(NULLIF(s.email,''), u.email) AS email, "
            "       s.sub_type, s.target, s.frequency, s.created_at, "
            "       s.last_notified_at, s.user_id "
            "FROM subscriptions s LEFT JOIN users u ON u.user_id = s.user_id "
            "WHERE s.is_active = 1",
        ).fetchall()

    due = []
    for r in rows:
        row = dict(r)
        last = row.get("last_notified_at")
        freq = row.get("frequency", "weekly")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
            except (ValueError, TypeError):
                last_dt = datetime.min
        else:
            last_dt = datetime.min

        if freq == "daily" and (now - last_dt).total_seconds() >= 86400:
            due.append(row)
        elif freq == "weekly" and (now - last_dt).total_seconds() >= 7 * 86400:
            due.append(row)
        elif not last:
            due.append(row)
    return due


def update_subscription_notified(sub_id: int):
    """Mark a subscription as just notified."""
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "UPDATE subscriptions SET last_notified_at = ? WHERE id = ?",
            (datetime.now().isoformat(), sub_id),
        )
        conn.commit()


def log_notification(subscription_id: int, claim_count: int,
                     status: str = "sent", error: str = None):
    """Log a notification attempt."""
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "INSERT INTO notification_log (subscription_id, sent_at, claim_count, status, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (subscription_id, datetime.now().isoformat(), claim_count, status, error),
        )
        conn.commit()


def get_notification_history(sub_id: int, limit: int = 20) -> list[dict]:
    """Get notification history for a subscription."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT id, sent_at, claim_count, status, error "
            "FROM notification_log WHERE subscription_id = ? "
            "ORDER BY sent_at DESC LIMIT ?",
            (sub_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_new_claims_for_subscription(sub: dict) -> list[dict]:
    """Get new claims matching a subscription since last notification."""
    since = sub.get("last_notified_at") or sub.get("created_at")
    sub_type = sub["sub_type"]
    target = sub["target"]

    with get_conn() as conn:
        if sub_type == "topic":
            row = conn.execute(
                "SELECT claim_ids FROM tree_nodes WHERE view_id || '/' || path = ? "
                "OR path = ?",
                [target, target],
            ).fetchone()
            if not row or not row["claim_ids"]:
                return []
            claim_ids = json.loads(row["claim_ids"])
            if not claim_ids:
                return []
            placeholders = ",".join("?" for _ in claim_ids)
            rows = conn.execute(
                f"SELECT data FROM claims WHERE claim_id IN ({placeholders}) "
                f"AND extracted_at > ?",
                claim_ids + [since],
            ).fetchall()
            claims = [json.loads(r["data"]) for r in rows]

        elif sub_type == "author":
            rows = conn.execute(
                "SELECT c.data FROM claims c "
                "JOIN sources s ON c.source_doi = s.doi "
                "WHERE s.authors LIKE ? AND c.extracted_at > ? "
                "ORDER BY c.extracted_at DESC LIMIT 50",
                (f"%{target}%", since),
            ).fetchall()
            claims = [json.loads(r["data"]) for r in rows]

        elif sub_type == "query":
            result = search_claims(target, limit=50)
            all_claims = result.get("claims", [])
            claims = [c for c in all_claims
                      if c.get("extracted_at", "") > since]

        else:
            claims = []

        enrich_claims_with_source(claims, conn)
        return claims


def get_discoveries_feed(limit: int = 20, days: int = 7) -> list[dict]:
    """Get the discoveries feed: highest-surprise claims from recent papers.

    Falls back to recent high-citation claims when surprise_scores is empty.
    Progressively widens the time window (7d -> 30d -> 90d -> 365d -> all)
    to ensure results are returned.
    """
    from datetime import timedelta

    with get_conn() as conn:
        has_scores = False
        try:
            has_scores = conn.execute(
                "SELECT COUNT(*) FROM surprise_scores"
            ).fetchone()[0] > 0
        except Exception:
            pass

        if has_scores:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            rows = conn.execute(
                "SELECT c.data, ss.total_score, ss.structural_score, "
                "ss.temporal_score, ss.content_score "
                "FROM claims c "
                "JOIN surprise_scores ss ON c.claim_id = ss.claim_id "
                "WHERE c.extracted_at > ? "
                "ORDER BY ss.total_score DESC "
                "LIMIT ?",
                [cutoff, limit],
            ).fetchall()
            results = []
            for r in rows:
                claim = json.loads(r["data"])
                results.append({
                    "claim": claim,
                    "surprise_score": r["total_score"],
                    "scores": {
                        "structural": r["structural_score"],
                        "temporal": r["temporal_score"],
                        "content": r["content_score"],
                    },
                })
            if results:
                enrich_claims_with_source([r["claim"] for r in results], conn)
                return results

        # Fallback: recent claims from high-citation papers
        # Widen time window progressively until we get results
        for window_days in [days, 30, 90, 365, 3650]:
            cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
            rows = conn.execute(
                "SELECT c.data FROM claims c "
                "JOIN sources s ON c.source_doi = s.doi "
                "WHERE c.extracted_at > ? "
                "ORDER BY s.citation_count DESC, c.extracted_at DESC "
                "LIMIT ?",
                [cutoff, limit],
            ).fetchall()
            if rows:
                break

        if not rows:
            # Ultimate fallback: ignore extracted_at entirely
            rows = conn.execute(
                "SELECT c.data FROM claims c "
                "JOIN sources s ON c.source_doi = s.doi "
                "ORDER BY s.citation_count DESC "
                "LIMIT ?",
                [limit],
            ).fetchall()

        claims = [json.loads(r["data"]) for r in rows]
        enrich_claims_with_source(claims, conn)
        return [{"claim": c, "surprise_score": 0, "scores": {}} for c in claims]


def get_top_authors(view_id: str = None, path: str = None, limit: int = 200) -> list[dict]:
    """Get top authors by paper count, optionally filtered by tree node."""
    with get_conn() as conn:
        if not _has_author_index(conn):
            return _fallback_get_top_authors(view_id=view_id, path=path, limit=limit)
        if view_id and path:
            row = conn.execute(
                "SELECT claim_ids FROM tree_nodes WHERE view_id = ? AND path = ?",
                [view_id, path],
            ).fetchone()
            if not row or not row["claim_ids"]:
                return []
            claim_ids = json.loads(row["claim_ids"])
            if not claim_ids:
                return []
            placeholders = ",".join("?" * min(len(claim_ids), 500))
            subset = claim_ids[:500]
            rows = conn.execute(
                f"SELECT pa.author_id, COUNT(DISTINCT pa.doi) as paper_count, a.cited_by_count "
                f"FROM claims c "
                f"JOIN paper_authors pa ON c.source_doi = pa.doi "
                f"JOIN authors a ON pa.author_id = a.author_id "
                f"WHERE c.claim_id IN ({placeholders}) "
                f"GROUP BY pa.author_id "
                f"ORDER BY a.cited_by_count DESC "
                f"LIMIT ?",
                subset + [limit],
            ).fetchall()
            results = []
            for r in rows:
                row2 = conn.execute("SELECT data FROM authors WHERE author_id = ?", [r["author_id"]]).fetchone()
                if row2:
                    d = json.loads(row2["data"])
                    d["matching_papers"] = r["paper_count"]
                    results.append(d)
            return results
        else:
            rows = conn.execute(
                "SELECT a.data, COUNT(pa.doi) as pc "
                "FROM authors a "
                "JOIN paper_authors pa ON a.author_id = pa.author_id "
                "GROUP BY a.author_id "
                "ORDER BY pc DESC LIMIT ?",
                [limit],
            ).fetchall()
            results = []
            for r in rows:
                d = json.loads(r["data"])
                d["papers_in_index"] = r["pc"]
                results.append(d)
            return results


def search_authors(query: str, limit: int = 200) -> list[dict]:
    """Search authors by name. Puts exact matches first, then partial matches sorted by paper count."""
    words = query.strip().split()
    if not words:
        return []
    with get_conn() as conn:
        if not _has_author_index(conn):
            return _fallback_search_authors(query, limit=limit)
        conditions = " AND ".join(["a.name LIKE ?"] * len(words))
        like_params = [f"%{w}%" for w in words]
        exact_lower = query.strip().lower()
        rows = conn.execute(
            f"SELECT a.data, COUNT(pa.doi) as pc, "
            f"CASE WHEN LOWER(a.name) = ? THEN 0 ELSE 1 END as exact_rank "
            f"FROM authors a "
            f"LEFT JOIN paper_authors pa ON a.author_id = pa.author_id "
            f"WHERE {conditions} "
            f"GROUP BY a.author_id "
            f"ORDER BY exact_rank ASC, pc DESC LIMIT ?",
            [exact_lower] + like_params + [limit],
        ).fetchall()
        results = []
        for r in rows:
            d = json.loads(r["data"])
            d["papers_in_index"] = r["pc"]
            results.append(d)
        return results


def get_authors_for_doi(doi: str) -> list[dict]:
    """Get disambiguated authors for a paper."""
    with get_conn() as conn:
        if not _has_author_index(conn):
            row = conn.execute(
                "SELECT authors FROM sources WHERE doi = ? COLLATE NOCASE", [doi]
            ).fetchone()
            if not row:
                return []
            return [
                {
                    "author_id": _local_author_id(name),
                    "name": name,
                    "position": "unknown",
                    "institution": "",
                    "h_index": 0,
                }
                for name in _parse_source_authors(row["authors"])
            ]
        rows = conn.execute(
            "SELECT pa.author_id, pa.position, a.name, a.institution, a.h_index "
            "FROM paper_authors pa "
            "LEFT JOIN authors a ON pa.author_id = a.author_id "
            "WHERE pa.doi = ? COLLATE NOCASE "
            "ORDER BY CASE pa.position WHEN 'first' THEN 0 WHEN 'middle' THEN 1 WHEN 'last' THEN 2 ELSE 3 END",
            [doi],
        ).fetchall()
    return [
        {
            "author_id": r["author_id"],
            "name": r["name"] or r["author_id"],
            "position": r["position"],
            "institution": r["institution"] or "",
            "h_index": r["h_index"] or 0,
        }
        for r in rows
    ]


def index_authors_for_doi(doi: str):
    """Fetch disambiguated authors from OpenAlex and add to DB."""
    import urllib.request
    try:
        url = (f"https://api.openalex.org/works?filter=doi:{doi}"
               f"&per-page=1&select=doi,authorships&mailto=askchem@mit.edu")
        req = urllib.request.Request(url, headers={"User-Agent": "AskChem/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        works = data.get("results", [])
        if not works:
            return

        work = works[0]
        with get_conn(readonly=False) as conn:
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                oa_id = author.get("id", "")
                if not oa_id:
                    continue
                author_id = oa_id.split("/")[-1]
                display_name = author.get("display_name", "")
                orcid = (author.get("orcid") or "").replace("https://orcid.org/", "")
                position = authorship.get("author_position", "middle")
                institutions = authorship.get("institutions", [])
                inst_name = institutions[0].get("display_name", "") if institutions else ""
                inst_country = institutions[0].get("country_code", "") if institutions else ""

                author_data = {
                    "author_id": author_id, "name": display_name,
                    "openalex_id": oa_id, "orcid": orcid,
                    "institution": inst_name, "institution_country": inst_country,
                    "h_index": 0, "works_count": 0, "cited_by_count": 0,
                }
                conn.execute(
                    "INSERT OR IGNORE INTO authors "
                    "(author_id,name,openalex_id,orcid,institution,institution_country,"
                    "h_index,works_count,cited_by_count,concepts,data) "
                    "VALUES (?,?,?,?,?,?,0,0,0,'[]',?)",
                    (author_id, display_name, oa_id, orcid, inst_name, inst_country,
                     json.dumps(author_data)),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO paper_authors (doi,author_id,position) VALUES (?,?,?)",
                    (doi, author_id, position),
                )
            conn.commit()
    except Exception:
        pass


def find_experts(topic: str, limit: int = 200) -> list[dict]:
    """Find experts for a topic by searching claims and counting authors.

    Uses stop-word-filtered FTS queries and requires authors to have
    at least 2 topic-relevant papers to qualify (relaxed to 1 if
    fewer than 5 experts meet the threshold).
    """
    with get_conn() as conn:
        if not _has_author_index(conn):
            return _fallback_find_experts(topic, limit=limit)
        fts_queries = _build_fts_queries(topic)
        fts_rows = []
        for fts_q in fts_queries:
            try:
                fts_rows = conn.execute(
                    "SELECT rowid FROM claims_fts WHERE claims_fts MATCH ? LIMIT 50000",
                    [fts_q],
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            if len(fts_rows) >= 100:
                break
        if not fts_rows:
            return []

        rowids = [r[0] for r in fts_rows]
        placeholders = ",".join("?" * len(rowids))
        doi_rows = conn.execute(
            f"SELECT DISTINCT source_doi FROM claims WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        dois = [r["source_doi"] for r in doi_rows if r["source_doi"]]
        if not dois:
            return []

        doi_placeholders = ",".join("?" * len(dois))
        pa_rows = conn.execute(
            f"SELECT pa.author_id, COUNT(DISTINCT pa.doi) as paper_count, "
            f"COALESCE(SUM(s.citation_count), 0) as total_cites "
            f"FROM paper_authors pa "
            f"JOIN sources s ON pa.doi = s.doi "
            f"WHERE pa.doi IN ({doi_placeholders}) "
            f"GROUP BY pa.author_id "
            f"ORDER BY paper_count DESC, total_cites DESC "
            f"LIMIT ?",
            dois + [limit * 2],
        ).fetchall()

        # Require min 2 topic-relevant papers; relax if too few qualify
        strong = [r for r in pa_rows if r["paper_count"] >= 2]
        if len(strong) >= 5:
            pa_rows = strong[:limit]
        else:
            pa_rows = pa_rows[:min(limit, 20)]

        results = []
        for r in pa_rows:
            author_row = conn.execute(
                "SELECT data FROM authors WHERE author_id = ?", [r["author_id"]]
            ).fetchone()
            if author_row:
                d = json.loads(author_row["data"])
                d["topic_papers"] = r["paper_count"]
                d["topic_citations"] = r["total_cites"]
                d["topic"] = topic
                results.append(d)
        return results


def get_author_profile(author_id: str) -> dict | None:
    """Get full author profile with papers, claims, and research breakdown."""
    with get_conn() as conn:
        if not _has_author_index(conn):
            return _fallback_get_author_profile(author_id)
        author_row = conn.execute(
            "SELECT data FROM authors WHERE author_id = ?", [author_id]
        ).fetchone()
        if not author_row:
            return None

        profile = json.loads(author_row["data"])

        pa_rows = conn.execute(
            "SELECT doi, position FROM paper_authors WHERE author_id = ?",
            [author_id],
        ).fetchall()
        dois = [r["doi"] for r in pa_rows]
        positions = {r["doi"]: r["position"] for r in pa_rows}
        author_name = profile.get("name") or ""

        papers = []
        if dois:
            doi_placeholders = ",".join("?" * len(dois))
            source_rows = conn.execute(
                f"SELECT doi, title, year, citation_count, authors FROM sources WHERE doi IN ({doi_placeholders})",
                dois,
            ).fetchall()
            for sr in source_rows:
                ordered_authors = _parse_source_authors(sr["authors"])
                # Prefer the stored position when available; otherwise
                # derive it from the ordered author list on the source
                # row so we never surface a bare "unknown".
                stored = positions.get(sr["doi"])
                if stored and stored not in ("unknown", ""):
                    pos = stored
                else:
                    pos = _position_in_authors(author_name, ordered_authors)
                papers.append({
                    "doi": sr["doi"],
                    "title": sr["title"],
                    "year": sr["year"],
                    "citation_count": sr["citation_count"],
                    "position": pos,
                    "authors": ordered_authors,
                })
            papers.sort(key=lambda p: p.get("year") or 0, reverse=True)

        view_breakdown = {}
        if dois:
            claim_rows = conn.execute(
                f"SELECT data FROM claims WHERE source_doi IN ({doi_placeholders})",
                dois,
            ).fetchall()
            for cr in claim_rows:
                cd = json.loads(cr["data"])
                vp = cd.get("view_paths", {})
                for view_id, path in vp.items():
                    if view_id == "by_claim_type":
                        continue
                    if path and isinstance(path, list) and len(path) >= 1:
                        l1 = path[0]
                        key = f"{view_id}/{l1}"
                        view_breakdown[key] = view_breakdown.get(key, 0) + 1

        top_areas = sorted(view_breakdown.items(), key=lambda x: -x[1])[:20]

        profile["papers"] = papers
        profile["paper_count"] = len(papers)
        profile["research_areas"] = [
            {"view_path": k, "claim_count": v} for k, v in top_areas
        ]
        return profile


def _has_coauthor_edges(conn) -> bool:
    """Whether the precomputed coauthor_edges table exists and is populated.

    The edge table is only created by the offline author-index build
    (build_author_index.py / populate_authors.py), not by core schema init,
    so many deployments have authors + paper_authors but no edges.
    """
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='coauthor_edges'"
        ).fetchone()
        if not exists:
            return False
        return bool(conn.execute("SELECT COUNT(*) FROM coauthor_edges").fetchone()[0])
    except Exception:
        return False


def _coauthors_of(conn, author_id: str, limit: int, use_edges: bool, exclude=()) -> list:
    """Return [(coauthor_id, paper_count), ...] for an author.

    Uses the precomputed coauthor_edges table when available, otherwise
    computes co-authorship on the fly from the indexed paper_authors table
    (authors who share at least one DOI). This keeps the ego network
    populated even when the offline edge build has not been run.
    """
    exclude = set(exclude) | {author_id}
    fetch = limit + len(exclude)
    if use_edges:
        rows = conn.execute(
            "SELECT author_id_2 AS coauthor, paper_count FROM coauthor_edges "
            "WHERE author_id_1 = ? "
            "UNION ALL "
            "SELECT author_id_1 AS coauthor, paper_count FROM coauthor_edges "
            "WHERE author_id_2 = ? "
            "ORDER BY paper_count DESC LIMIT ?",
            [author_id, author_id, fetch],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT pa2.author_id AS coauthor, "
            "COUNT(DISTINCT pa1.doi) AS paper_count "
            "FROM paper_authors pa1 "
            "JOIN paper_authors pa2 ON pa1.doi = pa2.doi "
            "WHERE pa1.author_id = ? AND pa2.author_id != pa1.author_id "
            "GROUP BY pa2.author_id "
            "ORDER BY paper_count DESC LIMIT ?",
            [author_id, fetch],
        ).fetchall()
    out = [(r["coauthor"], r["paper_count"]) for r in rows]
    return [(cid, pc) for cid, pc in out if cid not in exclude][:limit]


def get_coauthor_network(author_id: str, depth: int = 1, limit: int = 30) -> dict:
    """Get co-authorship ego network.

    Resolution order, so the network stays populated across deployments:
    (1) precomputed ``coauthor_edges`` table when present, (2) computed on
    the fly from the indexed ``paper_authors`` table, (3) the JSON source
    fallback. The endpoint never raises: any failure degrades to the
    fallback rather than a 500.
    """
    try:
        with get_conn() as conn:
            if not _has_author_index(conn):
                return _fallback_get_coauthor_network(author_id, depth=depth, limit=limit)

            author_row = conn.execute(
                "SELECT data FROM authors WHERE author_id = ?", [author_id]
            ).fetchone()
            if not author_row:
                return {"center": author_id, "nodes": [], "edges": []}

            use_edges = _has_coauthor_edges(conn)

            def _node(aid, data, depth_val):
                return {
                    "id": aid,
                    "name": data.get("name", ""),
                    "institution": data.get("institution", ""),
                    "h_index": data.get("h_index", 0),
                    "papers_in_index": data.get("papers_in_index", 0),
                    "depth": depth_val,
                }

            def _author_data(aid):
                row = conn.execute(
                    "SELECT data FROM authors WHERE author_id = ?", [aid]
                ).fetchone()
                return json.loads(row["data"]) if row else {}

            nodes = {author_id: _node(author_id, json.loads(author_row["data"]), 0)}
            edges = []

            for cid, weight in _coauthors_of(conn, author_id, limit, use_edges):
                edges.append({"source": author_id, "target": cid, "weight": weight})
                if cid not in nodes:
                    nodes[cid] = _node(cid, _author_data(cid), 1)

            if depth >= 2:
                for d1_id in [n for n in nodes if n != author_id][:10]:
                    for did, weight in _coauthors_of(
                        conn, d1_id, 5, use_edges, exclude=nodes.keys()
                    ):
                        edges.append({"source": d1_id, "target": did, "weight": weight})
                        if did not in nodes:
                            nodes[did] = _node(did, _author_data(did), 2)

            result = {"center": author_id, "nodes": list(nodes.values()), "edges": edges}
            # If the indexed path returned an isolated node, try the JSON fallback.
            if len(result["nodes"]) <= 1:
                fb = _fallback_get_coauthor_network(author_id, depth=depth, limit=limit)
                if len(fb.get("nodes", [])) > 1:
                    return fb
            return result
    except Exception:
        try:
            return _fallback_get_coauthor_network(author_id, depth=depth, limit=limit)
        except Exception:
            return {"center": author_id, "nodes": [], "edges": []}


def _has_author_index(conn) -> bool:
    """Whether disambiguated author tables are populated."""
    try:
        count = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        pa_count = conn.execute("SELECT COUNT(*) FROM paper_authors").fetchone()[0]
        return bool(count and pa_count)
    except Exception:
        return False


def _normalize_author_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


def _normalize_author_key(name: str) -> str:
    return _normalize_author_name(name).casefold()


def _local_author_id(name: str) -> str:
    norm = _normalize_author_key(name)
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return f"local:{digest}"


def _position_in_authors(author_name: str, source_authors: list[str]) -> str:
    """Derive first/middle/last position for `author_name` from the paper's
    ordered author list. Falls back to 'unknown' when no match is found
    (e.g. name variant the normalizer doesn't catch).

    The paper_authors table is currently empty in the corpus, so this is
    the canonical way author position is computed for the website.
    """
    if not author_name or not source_authors:
        return "unknown"
    target = _normalize_author_key(author_name)
    if not target:
        return "unknown"
    idx = -1
    for i, n in enumerate(source_authors):
        if _normalize_author_key(n) == target:
            idx = i
            break
    if idx < 0:
        return "unknown"
    if len(source_authors) == 1:
        return "first"
    if idx == 0:
        return "first"
    if idx == len(source_authors) - 1:
        return "last"
    return "middle"


def _parse_source_authors(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            items = [raw]
    else:
        items = [raw]

    out: list[str] = []
    seen = set()
    for item in items:
        name = _normalize_author_name(str(item))
        if len(name) < 2:
            continue
        key = _normalize_author_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _get_fallback_author_index() -> dict:
    global _fallback_author_cache, _fallback_author_cache_time
    now = _time.monotonic()
    if _fallback_author_cache is not None and now - _fallback_author_cache_time < _FALLBACK_AUTHOR_TTL:
        return _fallback_author_cache

    by_id: dict[str, dict] = {}
    doi_to_author_ids: dict[str, list[str]] = {}

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT doi, authors, citation_count FROM sources "
            "WHERE authors IS NOT NULL AND authors != '' AND authors != '[]'"
        ).fetchall()

    for row in rows:
        doi = row["doi"] or ""
        if not doi:
            continue
        names = _parse_source_authors(row["authors"])
        author_ids: list[str] = []
        cites = int(row["citation_count"] or 0)
        for name in names:
            author_id = _local_author_id(name)
            rec = by_id.setdefault(
                author_id,
                {
                    "author_id": author_id,
                    "name": name,
                    "openalex_id": "",
                    "orcid": "",
                    "institution": "",
                    "institution_country": "",
                    "h_index": 0,
                    "works_count": 0,
                    "cited_by_count": 0,
                    "papers_in_index": 0,
                    "dois": [],
                },
            )
            rec["papers_in_index"] += 1
            rec["works_count"] = rec["papers_in_index"]
            rec["cited_by_count"] += cites
            rec["dois"].append(doi)
            author_ids.append(author_id)
        if author_ids:
            doi_to_author_ids[doi] = author_ids

    ordered_ids = sorted(
        by_id,
        key=lambda aid: (
            -by_id[aid]["papers_in_index"],
            -by_id[aid]["cited_by_count"],
            by_id[aid]["name"].casefold(),
        ),
    )

    _fallback_author_cache = {
        "by_id": by_id,
        "doi_to_author_ids": doi_to_author_ids,
        "ordered_ids": ordered_ids,
    }
    _fallback_author_cache_time = now
    return _fallback_author_cache


def _fallback_author_record(author_id: str) -> dict | None:
    return _get_fallback_author_index()["by_id"].get(author_id)


def _fallback_author_records_for_dois(dois: list[str]) -> list[dict]:
    if not dois:
        return []
    cache = _get_fallback_author_index()
    counts: dict[str, dict] = {}
    doi_set = {d for d in dois if d}
    with get_conn() as conn:
        cite_rows = conn.execute(
            f"SELECT doi, citation_count FROM sources WHERE doi IN ({','.join('?' * len(doi_set))})",
            list(doi_set),
        ).fetchall() if doi_set else []
    doi_cites = {r["doi"]: int(r["citation_count"] or 0) for r in cite_rows}

    for doi in doi_set:
        for author_id in cache["doi_to_author_ids"].get(doi, []):
            rec = cache["by_id"].get(author_id)
            if not rec:
                continue
            agg = counts.setdefault(
                author_id,
                {
                    "author_id": author_id,
                    "name": rec["name"],
                    "papers_in_index": 0,
                    "cited_by_count": 0,
                },
            )
            agg["papers_in_index"] += 1
            agg["cited_by_count"] += doi_cites.get(doi, 0)

    results = list(counts.values())
    results.sort(key=lambda r: (-r["papers_in_index"], -r["cited_by_count"], r["name"].casefold()))
    return results


def _fallback_get_top_authors(view_id: str = None, path: str = None, limit: int = 200) -> list[dict]:
    cache = _get_fallback_author_index()
    if not (view_id and path):
        return [
            {k: v for k, v in cache["by_id"][aid].items() if k != "dois"}
            for aid in cache["ordered_ids"][:limit]
        ]

    with get_conn() as conn:
        has_map = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claim_view_map'"
        ).fetchone()
        if has_map:
            doi_rows = conn.execute(
                "SELECT DISTINCT c.source_doi FROM claims c "
                "JOIN claim_view_map m ON c.claim_id = m.claim_id "
                "WHERE m.view_id = ? AND (m.path = ? OR m.path LIKE ?)",
                [view_id, path, path + "/%"],
            ).fetchall()
        else:
            doi_rows = conn.execute(
                "SELECT DISTINCT source_doi FROM claims WHERE view_paths LIKE ?",
                [f'%"{view_id}"%{path}%'],
            ).fetchall()

    dois = [r[0] for r in doi_rows if r[0]]
    results = _fallback_author_records_for_dois(dois)[:limit]
    return results


def _fallback_search_authors(query: str, limit: int = 200) -> list[dict]:
    words = [_normalize_author_key(w) for w in query.strip().split() if w.strip()]
    if not words:
        return []
    cache = _get_fallback_author_index()
    exact = _normalize_author_key(query)
    matches = []
    for aid in cache["ordered_ids"]:
        rec = cache["by_id"][aid]
        name_key = _normalize_author_key(rec["name"])
        if all(word in name_key for word in words):
            matches.append(rec)
    matches.sort(
        key=lambda r: (
            0 if _normalize_author_key(r["name"]) == exact else 1,
            -r["papers_in_index"],
            -r["cited_by_count"],
            r["name"].casefold(),
        )
    )
    return [{k: v for k, v in r.items() if k != "dois"} for r in matches[:limit]]


def _fallback_find_experts(topic: str, limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        fts_queries = _build_fts_queries(topic)
        fts_rows = []
        for fts_q in fts_queries:
            try:
                fts_rows = conn.execute(
                    "SELECT rowid FROM claims_fts WHERE claims_fts MATCH ? LIMIT 50000",
                    [fts_q],
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            if len(fts_rows) >= 100:
                break
        if not fts_rows:
            return []

        rowids = [r[0] for r in fts_rows]
        placeholders = ",".join("?" * len(rowids))
        doi_rows = conn.execute(
            f"SELECT DISTINCT source_doi FROM claims WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()

    dois = [r["source_doi"] for r in doi_rows if r["source_doi"]]
    results = _fallback_author_records_for_dois(dois)
    for rec in results:
        rec["topic_papers"] = rec["papers_in_index"]
        rec["topic_citations"] = rec["cited_by_count"]
        rec["topic"] = topic
    return results[:limit]


def _fallback_get_author_profile(author_id: str) -> dict | None:
    rec = _fallback_author_record(author_id)
    if not rec:
        return None

    dois = rec.get("dois", [])
    profile = {k: v for k, v in rec.items() if k != "dois"}

    papers = []
    view_breakdown: dict[str, int] = {}
    author_name = (rec.get("name") or "") if isinstance(rec, dict) else ""
    if dois:
        with get_conn() as conn:
            doi_placeholders = ",".join("?" * len(dois))
            source_rows = conn.execute(
                f"SELECT doi, title, year, citation_count, authors FROM sources WHERE doi IN ({doi_placeholders})",
                dois,
            ).fetchall()
            claim_rows = conn.execute(
                f"SELECT data FROM claims WHERE source_doi IN ({doi_placeholders})",
                dois,
            ).fetchall()

        for sr in source_rows:
            ordered_authors = _parse_source_authors(sr["authors"])
            papers.append(
                {
                    "doi": sr["doi"],
                    "title": sr["title"],
                    "year": sr["year"],
                    "citation_count": sr["citation_count"],
                    "position": _position_in_authors(author_name, ordered_authors),
                    "authors": ordered_authors,
                }
            )
        papers.sort(key=lambda p: ((p.get("year") or 0), (p.get("citation_count") or 0)), reverse=True)

        for cr in claim_rows:
            cd = json.loads(cr["data"])
            vp = cd.get("view_paths", {})
            for view_id, path in vp.items():
                if view_id == "by_claim_type":
                    continue
                if path and isinstance(path, list) and len(path) >= 1:
                    key = f"{view_id}/{path[0]}"
                    view_breakdown[key] = view_breakdown.get(key, 0) + 1

    top_areas = sorted(view_breakdown.items(), key=lambda x: -x[1])[:20]
    profile["papers"] = papers
    profile["paper_count"] = len(papers)
    profile["research_areas"] = [{"view_path": k, "claim_count": v} for k, v in top_areas]
    return profile


def _fallback_get_coauthor_network(author_id: str, depth: int = 1, limit: int = 30) -> dict:
    rec = _fallback_author_record(author_id)
    if not rec:
        return {"nodes": [], "edges": []}

    cache = _get_fallback_author_index()
    nodes = {
        author_id: {
            "id": author_id,
            "name": rec["name"],
            "institution": "",
            "h_index": 0,
            "papers_in_index": rec.get("papers_in_index", 0),
            "depth": 0,
        }
    }
    co_counts: dict[str, int] = {}
    for doi in rec.get("dois", []):
        for coauthor_id in cache["doi_to_author_ids"].get(doi, []):
            if coauthor_id == author_id:
                continue
            co_counts[coauthor_id] = co_counts.get(coauthor_id, 0) + 1

    top = sorted(co_counts.items(), key=lambda x: -x[1])[:limit]
    edges = []
    for coauthor_id, weight in top:
        corec = cache["by_id"].get(coauthor_id)
        if not corec:
            continue
        nodes[coauthor_id] = {
            "id": coauthor_id,
            "name": corec["name"],
            "institution": "",
            "h_index": 0,
            "papers_in_index": corec.get("papers_in_index", 0),
            "depth": 1,
        }
        edges.append({"source": author_id, "target": coauthor_id, "weight": weight})

    return {"center": author_id, "nodes": list(nodes.values()), "edges": edges}


def _compute_graduated_tier(tier: str, total_requests: int, created_at: str) -> str:
    """Auto-upgrade tier based on usage (OpenAI-style)."""
    from datetime import datetime, timezone

    t = (tier or "tier_1").lower()
    if t in ("pro", "registered", "tier_3"):
        return "tier_3"
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        return t or "tier_1"
    now = datetime.now(timezone.utc)
    if created.tzinfo is None:
        age_days = (datetime.now() - created).days
    else:
        age_days = (now - created.astimezone(timezone.utc)).days
    tr = int(total_requests or 0)
    if t in ("free", "tier_1", "") and tr >= 10000 and age_days >= 7:
        return "tier_2"
    if t == "tier_2" and tr >= 100000 and age_days >= 30:
        return "tier_3"
    return t if t else "tier_1"


def create_api_key(name: str, email: str = "", tier: str = "tier_1") -> dict:
    """Create a new API key. Returns {key_id, api_key, name, tier}. Prefix ac- (AskChem)."""
    import secrets
    key_id = secrets.token_hex(8)
    raw_key = f"ac-{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    rpm = 1000 if tier in ("pro", "registered", "tier_3") else (
        500 if tier == "tier_2" else 200
    )

    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "INSERT INTO api_keys (key_id, key_hash, name, email, tier, rate_limit, created_at, total_requests) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (key_id, key_hash, name, email, tier, rpm, datetime.now().isoformat()),
        )
        conn.commit()

    return {"key_id": key_id, "api_key": raw_key, "name": name, "tier": tier}


def validate_api_key(raw_key: str) -> dict | None:
    """Validate an API key. Returns key info or None."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT key_id, name, tier, rate_limit, is_active, created_at, "
            "COALESCE(total_requests, 0) AS total_requests FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        ).fetchone()
    if not row or not row["is_active"]:
        return None
    tier = row["tier"] or "tier_1"
    total_requests = int(row["total_requests"] or 0)
    created_at = row["created_at"]
    new_tier = _compute_graduated_tier(tier, total_requests, created_at)
    if new_tier != tier:
        try:
            with get_runtime_conn(readonly=False) as conn:
                conn.execute(
                    "UPDATE api_keys SET tier = ? WHERE key_id = ?",
                    (new_tier, row["key_id"]),
                )
                conn.commit()
        except Exception:
            pass
        tier = new_tier
    try:
        with get_runtime_conn(readonly=False) as conn:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (datetime.now().isoformat(), key_hash),
            )
            conn.commit()
    except Exception:
        pass
    return {
        "key_id": row["key_id"],
        "name": row["name"],
        "tier": tier,
        "rate_limit": row["rate_limit"],
        "total_requests": total_requests,
        "created_at": created_at,
    }


def get_key_rpd_today(key_id: str) -> int:
    """Requests counted today (UTC) for RPD enforcement."""
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).date().isoformat()
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT request_count FROM key_usage WHERE key_id = ? AND date = ?",
            (key_id, day),
        ).fetchone()
    return int(row["request_count"]) if row else 0


def record_authenticated_api_request(key_id: str) -> None:
    """Increment daily and lifetime counters after a successful rate-limited request."""
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).date().isoformat()
    try:
        with get_runtime_conn(readonly=False) as conn:
            conn.execute(
                "INSERT INTO key_usage (key_id, date, request_count) VALUES (?, ?, 1) "
                "ON CONFLICT(key_id, date) DO UPDATE SET request_count = request_count + 1",
                (key_id, day),
            )
            conn.execute(
                "UPDATE api_keys SET total_requests = COALESCE(total_requests, 0) + 1 WHERE key_id = ?",
                (key_id,),
            )
            conn.commit()
    except Exception:
        pass


def get_api_key_usage_summary(key_id: str, days: int = 30) -> dict:
    """Daily request counts and total_requests for /api/usage."""
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days - 1)).isoformat()
    with get_runtime_conn() as conn:
        total_row = conn.execute(
            "SELECT COALESCE(total_requests, 0) FROM api_keys WHERE key_id = ?",
            (key_id,),
        ).fetchone()
        total_requests = int(total_row[0]) if total_row else 0
        rows = conn.execute(
            "SELECT date, request_count FROM key_usage WHERE key_id = ? AND date >= ? "
            "ORDER BY date DESC",
            (key_id, cutoff),
        ).fetchall()
    daily = [{"date": r["date"], "requests": int(r["request_count"])} for r in rows]
    return {"total_requests": total_requests, "daily_usage": daily}


def log_security_event(event_type: str, ip_hash: str = "", details: str = ""):
    try:
        with get_runtime_conn(readonly=False) as conn:
            conn.execute(
                "INSERT INTO security_log (timestamp, event_type, ip_hash, details) VALUES (?,?,?,?)",
                (datetime.now().isoformat(), event_type, ip_hash or None, details or None),
            )
            conn.commit()
    except Exception:
        pass


def get_security_events(days: int = 7, event_type: str = None, limit: int = 500) -> list[dict]:
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with get_runtime_conn() as conn:
        sql = "SELECT * FROM security_log WHERE timestamp >= ?"
        params: list = [cutoff]
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def export_claims(claim_type: str = None, since: str = None,
                  limit: int = 10000, offset: int = 0) -> dict:
    """Export claims in bulk for downstream consumers."""
    with get_conn() as conn:
        sql = "SELECT data FROM claims WHERE 1=1"
        params: list = []
        if claim_type:
            sql += " AND claim_type = ?"
            params.append(claim_type)
        if since:
            sql += " AND extracted_at >= ?"
            params.append(since)
        sql += " ORDER BY claim_id LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        claims = [json.loads(r["data"]) for r in rows]

        count_sql = "SELECT COUNT(*) FROM claims WHERE 1=1"
        count_params: list = []
        if claim_type:
            count_sql += " AND claim_type = ?"
            count_params.append(claim_type)
        if since:
            count_sql += " AND extracted_at >= ?"
            count_params.append(since)
        total = conn.execute(count_sql, count_params).fetchone()[0]

    return {"claims": claims, "total": total, "limit": limit, "offset": offset}


def get_changelog(since: str = None, limit: int = 100) -> dict:
    """Get recently added/updated claims as a changelog."""
    with get_conn() as conn:
        sql = "SELECT claim_id, claim_type, source_doi, source_paper_title, extracted_at FROM claims"
        params: list = []
        if since:
            sql += " WHERE extracted_at >= ?"
            params.append(since)
        sql += " ORDER BY extracted_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        entries = [dict(r) for r in rows]

    return {"entries": entries, "count": len(entries), "since": since}


def log_query(query: str, endpoint: str, view: str = None, filters: str = None,
              result_count: int = 0, latency_ms: float = 0,
              user_agent: str = None, ip_hash: str = None,
              user_id: str = None) -> int | None:
    """Log a search query for analytics. Returns the inserted query_log.id."""
    try:
        with get_runtime_conn(readonly=False) as conn:
            cur = conn.execute(
                "INSERT INTO query_log "
                "(timestamp, query, endpoint, view, filters, result_count, "
                " latency_ms, user_agent, ip_hash, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query, endpoint, view, filters,
                 result_count, latency_ms, user_agent, ip_hash, user_id),
            )
            conn.commit()
            return cur.lastrowid
    except Exception:
        return None


def get_user_query_history(user_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """Return recent queries by a user, newest first."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, query, endpoint, view, filters, "
            "       result_count, latency_ms "
            "FROM query_log WHERE user_id = ? "
            "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [user_id, limit, offset],
        ).fetchall()
    return [dict(r) for r in rows]


def log_click(target_type: str, target_id: str,
              query: str = None, query_log_id: int = None,
              position: int = None, user_id: str = None,
              ip_hash: str = None) -> None:
    """Log that a user clicked a search result (claim / paper / link)."""
    try:
        with get_runtime_conn(readonly=False) as conn:
            conn.execute(
                "INSERT INTO click_log "
                "(timestamp, query_log_id, query, target_type, target_id, "
                " position, user_id, ip_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query_log_id, query,
                 target_type, target_id, position, user_id, ip_hash),
            )
            conn.commit()
    except Exception:
        pass


def get_query_stats(days: int = 30, limit: int = 50) -> dict:
    """Get query analytics for the last N days."""
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with get_runtime_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]

        top_queries = conn.execute(
            "SELECT query, COUNT(*) as cnt, AVG(latency_ms) as avg_latency "
            "FROM query_log WHERE timestamp > ? "
            "GROUP BY query ORDER BY cnt DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()

        daily = conn.execute(
            "SELECT DATE(timestamp) as day, COUNT(*) as cnt "
            "FROM query_log WHERE timestamp > ? "
            "GROUP BY day ORDER BY day",
            (cutoff,),
        ).fetchall()

        by_endpoint = conn.execute(
            "SELECT endpoint, COUNT(*) as cnt "
            "FROM query_log WHERE timestamp > ? "
            "GROUP BY endpoint ORDER BY cnt DESC",
            (cutoff,),
        ).fetchall()

    return {
        "total_queries": total,
        "period_days": days,
        "top_queries": [{"query": r[0], "count": r[1], "avg_latency_ms": round(r[2], 1)} for r in top_queries],
        "daily_counts": [{"date": r[0], "count": r[1]} for r in daily],
        "by_endpoint": [{"endpoint": r[0], "count": r[1]} for r in by_endpoint],
    }


# ── Upsert functions for direct writes ────────────────────────────────────────

def upsert_source(source_data: dict):
    """Insert or update a source. Accepts either a raw dict or a structured one."""
    doi = source_data.get('doi', '')
    if not doi:
        return
    with get_conn(readonly=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sources "
            "(doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                doi,
                source_data.get('title', ''),
                json.dumps(source_data.get('authors', [])),
                source_data.get('year', 0),
                source_data.get('venue', ''),
                source_data.get('abstract', ''),
                source_data.get('citation_count', 0),
                source_data.get('open_access_url', ''),
                json.dumps(source_data),
            )
        )
        conn.commit()


def upsert_claim(claim_data: dict):
    """Insert or update a claim with FTS entry."""
    claim_id = claim_data.get('claim_id', '')
    if not claim_id:
        return
    searchable = build_searchable_text(claim_data)
    with get_conn(readonly=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO claims "
            "(claim_id,claim_type,source_doi,source_paper_title,confidence,"
            "location_in_paper,verbatim_quote,extraction_model,extraction_version,"
            "extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim_id,
                claim_data.get('claim_type', ''),
                claim_data.get('source_doi', ''),
                claim_data.get('source_paper_title', ''),
                claim_data.get('confidence', ''),
                claim_data.get('location_in_paper', ''),
                claim_data.get('verbatim_quote', ''),
                claim_data.get('extraction_model', ''),
                claim_data.get('extraction_version', ''),
                claim_data.get('extracted_at', ''),
                json.dumps(claim_data.get('view_paths', {})),
                json.dumps(claim_data),
            )
        )
        conn.execute("DELETE FROM claims_fts WHERE claim_id = ?", [claim_id])
        conn.execute(
            "INSERT INTO claims_fts(claim_id, claim_type, source_paper_title, "
            "verbatim_quote, searchable_text) VALUES (?,?,?,?,?)",
            (claim_id, claim_data.get('claim_type', ''),
             claim_data.get('source_paper_title', ''),
             claim_data.get('verbatim_quote', ''), searchable)
        )
        conn.commit()


def upsert_tree_node(view_id: str, path_str: str, name: str, level: int,
                     claim_ids: list[str] = None, children: list[str] = None,
                     data: dict = None):
    """Insert or update a tree node."""
    cids = claim_ids or []
    kids = children or []
    node_data = data or {}
    with get_conn(readonly=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tree_nodes "
            "(view_id,path,name,level,claim_count,children,claim_ids,data) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (view_id, path_str, name, level, len(cids),
             json.dumps(kids), json.dumps(cids[:100]), json.dumps(node_data))
        )
        conn.commit()


def append_claim_to_node(view_id: str, path_str: str, claim_id: str):
    """Append a claim_id to an existing tree node, creating it if needed."""
    with get_conn(readonly=False) as conn:
        row = conn.execute(
            "SELECT claim_ids, claim_count FROM tree_nodes WHERE view_id = ? AND path = ?",
            [view_id, path_str]
        ).fetchone()
        if row:
            existing = json.loads(row['claim_ids']) if row['claim_ids'] else []
            if claim_id not in existing:
                existing.append(claim_id)
                conn.execute(
                    "UPDATE tree_nodes SET claim_ids = ?, claim_count = ? "
                    "WHERE view_id = ? AND path = ?",
                    (json.dumps(existing[-100:]), (row['claim_count'] or 0) + 1,
                     view_id, path_str)
                )
        else:
            segments = path_str.split('/') if path_str else []
            conn.execute(
                "INSERT INTO tree_nodes "
                "(view_id,path,name,level,claim_count,children,claim_ids,data) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (view_id, path_str, smart_title(segments[-1]) if segments else view_id,
                 len(segments), 1, json.dumps([]), json.dumps([claim_id]),
                 json.dumps({'node_id': f'{view_id}_{path_str}', 'name': smart_title(segments[-1]) if segments else view_id}))
            )
            if segments:
                parent_path = '/'.join(segments[:-1])
                child_seg = segments[-1]
                parent_row = conn.execute(
                    "SELECT children FROM tree_nodes WHERE view_id = ? AND path = ?",
                    [view_id, parent_path]
                ).fetchone()
                if parent_row:
                    kids = json.loads(parent_row['children']) if parent_row['children'] else []
                    if child_seg not in kids:
                        kids.append(child_seg)
                        conn.execute(
                            "UPDATE tree_nodes SET children = ? WHERE view_id = ? AND path = ?",
                            (json.dumps(kids), view_id, parent_path)
                        )
        conn.commit()


def upsert_claims_batch(claims: list[dict], batch_size: int = 1000):
    """Batch insert/update claims with FTS. Much faster than one-by-one."""
    with get_conn(readonly=False) as conn:
        claim_batch = []
        fts_batch = []
        for cd in claims:
            claim_id = cd.get('claim_id', '')
            if not claim_id:
                continue
            searchable = build_searchable_text(cd)
            claim_batch.append((
                claim_id, cd.get('claim_type', ''), cd.get('source_doi', ''),
                cd.get('source_paper_title', ''), cd.get('confidence', ''),
                cd.get('location_in_paper', ''), cd.get('verbatim_quote', ''),
                cd.get('extraction_model', ''), cd.get('extraction_version', ''),
                cd.get('extracted_at', ''), json.dumps(cd.get('view_paths', {})),
                json.dumps(cd),
            ))
            fts_batch.append((
                claim_id, cd.get('claim_type', ''),
                cd.get('source_paper_title', ''),
                cd.get('verbatim_quote', ''), searchable,
            ))
            if len(claim_batch) >= batch_size:
                conn.executemany(
                    "INSERT OR REPLACE INTO claims "
                    "(claim_id,claim_type,source_doi,source_paper_title,confidence,"
                    "location_in_paper,verbatim_quote,extraction_model,extraction_version,"
                    "extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    claim_batch)
                for cid, *_ in fts_batch:
                    conn.execute("DELETE FROM claims_fts WHERE claim_id = ?", [cid])
                conn.executemany(
                    "INSERT INTO claims_fts(claim_id,claim_type,source_paper_title,"
                    "verbatim_quote,searchable_text) VALUES (?,?,?,?,?)",
                    fts_batch)
                conn.commit()
                claim_batch = []
                fts_batch = []
        if claim_batch:
            conn.executemany(
                "INSERT OR REPLACE INTO claims "
                "(claim_id,claim_type,source_doi,source_paper_title,confidence,"
                "location_in_paper,verbatim_quote,extraction_model,extraction_version,"
                "extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                claim_batch)
            for cid, *_ in fts_batch:
                conn.execute("DELETE FROM claims_fts WHERE claim_id = ?", [cid])
            conn.executemany(
                "INSERT INTO claims_fts(claim_id,claim_type,source_paper_title,"
                "verbatim_quote,searchable_text) VALUES (?,?,?,?,?)",
                fts_batch)
            conn.commit()


def update_metadata_counts():
    """Recount and update metadata table."""
    with get_conn(readonly=False) as conn:
        total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        total_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        total_nodes = conn.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()[0]
        for k, v in [
            ('total_claims', str(total_claims)),
            ('total_sources', str(total_sources)),
            ('total_nodes', str(total_nodes)),
            ('last_updated', datetime.now().isoformat()),
        ]:
            conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)", (k, v))
        conn.commit()
    return {'total_claims': total_claims, 'total_sources': total_sources, 'total_nodes': total_nodes}


# ── Deep PDF claim merger ─────────────────────────────────────────────────────

def merge_deep_claims(data_dir: Path = None):
    """Merge deep PDF extraction claims into the SQLite database.

    Reads from data/deep_results/*.json and data/classify_pipeline/classifications.json.
    These are full-paper claims extracted by gpt-5.4 with richer detail than abstract claims.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"

    results_dir = data_dir / "deep_results"
    classify_file = data_dir / "classify_pipeline" / "classifications.json"
    corpus_dir = data_dir / "corpus_checkpoints"

    if not results_dir.exists():
        print(f"  No deep results at {results_dir}")
        return

    print("Loading deep extraction results...", flush=True)
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get('num_claims', 0) > 0:
                results.append(d)
        except Exception:
            pass
    print(f"  {len(results)} papers with claims", flush=True)

    classifications = {}
    if classify_file.exists():
        print("Loading classifications...", flush=True)
        classifications = json.loads(classify_file.read_text())
        print(f"  {len(classifications):,} classified claims", flush=True)

    print("Loading corpus metadata...", flush=True)
    corpus = {}
    if corpus_dir.exists():
        for shard in sorted(f for f in os.listdir(corpus_dir) if f.endswith('.jsonl')):
            with open(corpus_dir / shard) as fh:
                for line in fh:
                    paper = json.loads(line)
                    doi = (paper.get('externalIds') or {}).get('DOI', '')
                    if doi and doi.lower() not in corpus:
                        corpus[doi.lower()] = paper
    print(f"  {len(corpus):,} papers in corpus", flush=True)

    CLAIM_TYPE_LABELS = {
        'reaction': 'reactions', 'property': 'properties', 'method': 'methods',
        'mechanism': 'mechanisms', 'comparison': 'comparisons',
        'scope_entry': 'scope_entries', 'computational_result': 'computational_results',
        'hypothesis': 'hypotheses', 'conclusion': 'conclusions',
        'conclusions': 'conclusions', 'limitation': 'limitations',
        'future_direction': 'future_directions', 'surprising_finding': 'surprising_findings',
        'experimental_design': 'experimental_designs', 'structure': 'structures',
        'background': 'background', 'historical': 'historical',
        'definition': 'definitions', 'observation': 'observations',
    }
    from askchem.taxonomy import ALL_CONTENT_VIEWS

    sources_added = 0
    claims_added = 0
    sources_seen = set()
    claim_batch = []

    with get_conn(readonly=False) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        for ri, result in enumerate(results):
            doi = result.get('doi', '')
            if not doi:
                continue

            if doi.lower() not in sources_seen:
                sources_seen.add(doi.lower())
                cp = corpus.get(doi.lower(), {})
                authors = [a.get('name', '') for a in (cp.get('authors') or [])[:20]]
                source_data = {
                    'doi': doi, 'title': cp.get('title', ''),
                    'authors': authors, 'year': cp.get('year') or 0,
                    'venue': cp.get('venue', ''),
                    'abstract': cp.get('abstract', ''),
                    'citation_count': cp.get('citationCount', 0) or 0,
                    'open_access_url': (cp.get('openAccessPdf') or {}).get('url', ''),
                }
                conn.execute(
                    "INSERT OR IGNORE INTO sources "
                    "(doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (doi, source_data['title'], json.dumps(authors),
                     source_data['year'], source_data['venue'], source_data['abstract'],
                     source_data['citation_count'], source_data['open_access_url'],
                     json.dumps(source_data))
                )
                sources_added += 1

            paper_knowledge = result.get('data', {}).get('paper_knowledge', {})
            subfield = paper_knowledge.get('subfield', '')
            source_title = corpus.get(doi.lower(), {}).get('title', '')

            for raw_claim in result.get('data', {}).get('claims', []):
                claim_type = raw_claim.get('claim_type', 'unknown')
                content_hash = hashlib.sha256(
                    json.dumps(raw_claim, sort_keys=True).encode()
                ).hexdigest()[:12]

                from askchem.models import Claim
                claim_id = Claim.generate_id(doi, claim_type, content_hash)

                view_paths = {}
                llm_paths = classifications.get(claim_id, {})
                for view_id in ALL_CONTENT_VIEWS:
                    p = llm_paths.get(view_id)
                    if not p or p == ['not_applicable']:
                        continue
                    if p and isinstance(p[0], list):
                        p = p[0]
                    p = [str(s).replace('\x00', '').replace('/', '_').strip()
                         for s in p if isinstance(s, (str, int, float))]
                    p = [s for s in p if s and s != 'not_applicable']
                    if p:
                        view_paths[view_id] = p

                ct_l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
                ct_path = [ct_l1]
                if subfield:
                    ct_path.append(subfield.lower().replace(' ', '_'))
                view_paths['by_claim_type'] = ct_path

                claim_data = dict(raw_claim)
                claim_data.update({
                    'claim_id': claim_id,
                    'source_doi': doi,
                    'source_paper_title': source_title,
                    'extraction_model': 'gpt-5.4',
                    'extraction_version': 'deep_v1',
                    'view_paths': view_paths,
                })

                searchable = build_searchable_text(claim_data)
                conn.execute(
                    "INSERT OR IGNORE INTO claims "
                    "(claim_id,claim_type,source_doi,source_paper_title,confidence,"
                    "location_in_paper,verbatim_quote,extraction_model,extraction_version,"
                    "extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (claim_id, claim_type, doi, source_title,
                     raw_claim.get('confidence', 'high'),
                     raw_claim.get('location_in_paper', ''),
                     raw_claim.get('verbatim_quote', ''),
                     'gpt-5.4', 'deep_v1',
                     result.get('collected_at', datetime.now().isoformat()),
                     json.dumps(view_paths), json.dumps(claim_data))
                )
                existing_fts = conn.execute(
                    "SELECT claim_id FROM claims_fts WHERE claim_id = ? LIMIT 1",
                    [claim_id]
                ).fetchone()
                if not existing_fts:
                    conn.execute(
                        "INSERT INTO claims_fts"
                        "(claim_id,claim_type,source_paper_title,verbatim_quote,searchable_text) "
                        "VALUES (?,?,?,?,?)",
                        (claim_id, claim_type, source_title,
                         raw_claim.get('verbatim_quote', ''), searchable)
                    )
                claims_added += 1

            if (ri + 1) % 100 == 0:
                conn.commit()
                print(f"  Processed {ri+1}/{len(results)} papers "
                      f"({claims_added:,} claims)", flush=True)

        conn.commit()

    counts = update_metadata_counts()
    print(f"\nDeep merge complete:")
    print(f"  Sources added: {sources_added:,}")
    print(f"  Claims added: {claims_added:,}")
    print(f"  DB totals: {counts['total_claims']:,} claims, "
          f"{counts['total_sources']:,} sources")


# ---------------------------------------------------------------------------
# User auth & feedback helpers
# ---------------------------------------------------------------------------

def create_user(email: str, display_name: str = "") -> dict:
    """Create a new user or return existing. Returns {user_id, email, display_name, created_at}."""
    import uuid
    now = datetime.now().isoformat()
    with get_runtime_conn(readonly=False) as conn:
        existing = conn.execute(
            "SELECT user_id, email, display_name, created_at FROM users WHERE email = ?",
            [email],
        ).fetchone()
        if existing:
            conn.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?",
                         [now, existing["user_id"]])
            conn.commit()
            return dict(existing)
        user_id = uuid.uuid4().hex[:16]
        conn.execute(
            "INSERT INTO users (user_id, email, display_name, created_at, last_login_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, email, display_name or email.split("@")[0], now, now),
        )
        conn.commit()
    return {"user_id": user_id, "email": email,
            "display_name": display_name or email.split("@")[0], "created_at": now}


def get_or_create_clerk_user(clerk_id: str, email: str = "") -> dict:
    """Find user by Clerk ID, or create a new one. Returns user dict."""
    import uuid as _uuid
    now = datetime.now().isoformat()
    with get_runtime_conn(readonly=False) as conn:
        row = conn.execute(
            "SELECT user_id, email, display_name, clerk_id, created_at "
            "FROM users WHERE clerk_id = ?",
            [clerk_id],
        ).fetchone()
        if row:
            updates = ["last_login_at = ?"]
            params: list = [now]
            if email and email != row["email"]:
                updates.append("email = ?")
                params.append(email)
            params.append(row["user_id"])
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?",
                params,
            )
            conn.commit()
            result = dict(row)
            if email:
                result["email"] = email
            return result

        existing_email = None
        if email:
            existing_email = conn.execute(
                "SELECT user_id, email, display_name, created_at "
                "FROM users WHERE email = ? AND (clerk_id IS NULL OR clerk_id = '')",
                [email],
            ).fetchone()
        if existing_email:
            conn.execute(
                "UPDATE users SET clerk_id = ?, last_login_at = ? WHERE user_id = ?",
                (clerk_id, now, existing_email["user_id"]),
            )
            conn.commit()
            result = dict(existing_email)
            result["clerk_id"] = clerk_id
            return result

        user_id = _uuid.uuid4().hex[:16]
        display_name = email.split("@")[0] if email else clerk_id[:8]
        conn.execute(
            "INSERT INTO users (user_id, email, display_name, clerk_id, created_at, last_login_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, email, display_name, clerk_id, now, now),
        )
        conn.commit()
    return {"user_id": user_id, "email": email, "display_name": display_name,
            "clerk_id": clerk_id, "created_at": now}


def get_user_by_email(email: str) -> dict | None:
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT user_id, email, display_name, created_at FROM users WHERE email = ?",
            [email],
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT user_id, email, display_name, created_at FROM users WHERE user_id = ?",
            [user_id],
        ).fetchone()
    return dict(row) if row else None


def create_session(user_id: str, ttl_days: int = 30) -> str:
    """Create a session token for a user. Returns the token string."""
    import secrets
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    expires = now + timedelta(days=ttl_days)
    with get_runtime_conn(readonly=False) as conn:
        conn.execute(
            "INSERT INTO user_sessions (session_token, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires.isoformat()),
        )
        conn.commit()
    return token


def validate_session(token: str) -> dict | None:
    """Validate a session token. Returns user dict or None if expired/invalid."""
    with get_runtime_conn() as conn:
        row = conn.execute(
            "SELECT s.user_id, s.expires_at, u.email, u.display_name "
            "FROM user_sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.session_token = ?",
            [token],
        ).fetchone()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        expire_session(token)
        return None
    return {"user_id": row["user_id"], "email": row["email"],
            "display_name": row["display_name"]}


def expire_session(token: str) -> None:
    with get_runtime_conn(readonly=False) as conn:
        conn.execute("DELETE FROM user_sessions WHERE session_token = ?", [token])
        conn.commit()


def cleanup_expired_sessions() -> int:
    """Delete expired sessions. Returns count deleted."""
    with get_runtime_conn(readonly=False) as conn:
        cursor = conn.execute(
            "DELETE FROM user_sessions WHERE expires_at < ?",
            [datetime.now().isoformat()],
        )
        conn.commit()
    return cursor.rowcount


def upsert_feedback(user_id: str, target_type: str, target_id: str,
                    feedback_type: str, comment: str = "") -> dict:
    """Insert or toggle feedback. If the same vote exists, remove it (toggle off).
    If the opposite vote exists, replace it. Returns {action, feedback_type}."""
    now = datetime.now().isoformat()
    with get_runtime_conn(readonly=False) as conn:
        existing = conn.execute(
            "SELECT id, feedback_type FROM feedback "
            "WHERE user_id = ? AND target_type = ? AND target_id = ? "
            "AND feedback_type IN ('upvote', 'downvote')",
            [user_id, target_type, target_id],
        ).fetchone()

        if feedback_type in ("upvote", "downvote"):
            if existing and existing["feedback_type"] == feedback_type:
                conn.execute("DELETE FROM feedback WHERE id = ?", [existing["id"]])
                conn.commit()
                return {"action": "removed", "feedback_type": feedback_type}
            if existing:
                conn.execute("DELETE FROM feedback WHERE id = ?", [existing["id"]])

        conn.execute(
            "INSERT INTO feedback (user_id, target_type, target_id, feedback_type, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, target_type, target_id, feedback_type, comment, now),
        )
        conn.commit()
    return {"action": "added", "feedback_type": feedback_type}


def get_feedback_summary(target_type: str, target_id: str,
                         user_id: str = None) -> dict:
    """Get vote counts for a target. Optionally includes current user's vote."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT feedback_type, COUNT(*) as cnt FROM feedback "
            "WHERE target_type = ? AND target_id = ? AND feedback_type IN ('upvote', 'downvote') "
            "GROUP BY feedback_type",
            [target_type, target_id],
        ).fetchall()
        counts = {r["feedback_type"]: r["cnt"] for r in rows}
        result = {"upvotes": counts.get("upvote", 0),
                  "downvotes": counts.get("downvote", 0),
                  "user_vote": None}
        if user_id:
            vote = conn.execute(
                "SELECT feedback_type FROM feedback "
                "WHERE user_id = ? AND target_type = ? AND target_id = ? "
                "AND feedback_type IN ('upvote', 'downvote')",
                [user_id, target_type, target_id],
            ).fetchone()
            if vote:
                result["user_vote"] = vote["feedback_type"]
    return result


def get_feedback_batch(target_type: str, target_ids: list[str],
                       user_id: str = None) -> dict[str, dict]:
    """Get vote summaries for multiple targets in one query."""
    if not target_ids:
        return {}
    with get_runtime_conn() as conn:
        ph = ",".join("?" * len(target_ids))
        rows = conn.execute(
            f"SELECT target_id, feedback_type, COUNT(*) as cnt FROM feedback "
            f"WHERE target_type = ? AND target_id IN ({ph}) "
            f"AND feedback_type IN ('upvote', 'downvote') "
            f"GROUP BY target_id, feedback_type",
            [target_type] + target_ids,
        ).fetchall()
        result: dict[str, dict] = {}
        for r in rows:
            tid = r["target_id"]
            if tid not in result:
                result[tid] = {"upvotes": 0, "downvotes": 0, "user_vote": None}
            if r["feedback_type"] == "upvote":
                result[tid]["upvotes"] = r["cnt"]
            else:
                result[tid]["downvotes"] = r["cnt"]
        if user_id:
            user_votes = conn.execute(
                f"SELECT target_id, feedback_type FROM feedback "
                f"WHERE user_id = ? AND target_type = ? AND target_id IN ({ph}) "
                f"AND feedback_type IN ('upvote', 'downvote')",
                [user_id, target_type] + target_ids,
            ).fetchall()
            for v in user_votes:
                tid = v["target_id"]
                if tid not in result:
                    result[tid] = {"upvotes": 0, "downvotes": 0, "user_vote": None}
                result[tid]["user_vote"] = v["feedback_type"]
    return result


def get_user_feedback(user_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """Get a user's feedback history."""
    with get_runtime_conn() as conn:
        rows = conn.execute(
            "SELECT target_type, target_id, feedback_type, comment, created_at "
            "FROM feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [user_id, limit, offset],
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        index_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent.parent / "chemtree_index"
        print(f"Building SQLite database from {index_dir}...")
        import_from_filesystem(index_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "merge-deep":
        data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        print("Merging deep PDF claims into SQLite...")
        merge_deep_claims(data_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "build-jsonl":
        dataset_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent.parent / "askchem"
        print(f"Building SQLite database from JSONL dataset at {dataset_dir}...")
        import_from_jsonl(dataset_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "build-view-map":
        print("Building claim_view_map junction table...")
        build_claim_view_map()
    elif len(sys.argv) > 1 and sys.argv[1] == "build-paper-text":
        print("Building paper-level searchable text in sources_fts...")
        build_paper_searchable_text()
    else:
        print("Usage:")
        print("  python -m askchem.db build [index_dir]       # Build from filesystem index")
        print("  python -m askchem.db build-jsonl [dataset_dir] # Build from JSONL dataset (claims.jsonl)")
        print("  python -m askchem.db merge-deep [data_dir]   # Merge deep PDF claims")
        print("  python -m askchem.db build-view-map           # Build claim_view_map junction table")
        print("  python -m askchem.db build-paper-text         # Build paper searchable text in sources_fts")
