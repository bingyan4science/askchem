"""Private mutable service state, isolated from the public corpus database."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

RUNTIME_TABLES = (
    "submissions",
    "query_log",
    "click_log",
    "api_keys",
    "subscriptions",
    "notification_log",
    "community_flags",
    "ltree_feedback",
    "key_usage",
    "security_log",
    "edge_jobs",
    "users",
    "user_sessions",
    "feedback",
    "bookmarks",
    "saved_searches",
    "reading_lists",
    "reading_list_items",
)

_SCHEMA_PATH = Path(__file__).with_name("runtime_schema.sql")
_INIT_LOCK = threading.Lock()
_INITIALIZED: set[tuple[Path, Path]] = set()


def runtime_db_path(corpus_path: Path) -> Path:
    configured = os.environ.get("ASKCHEM_RUNTIME_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    return corpus_path.with_name(f"{corpus_path.stem}.runtime.db")


def _attach_corpus(conn: sqlite3.Connection, corpus_path: Path) -> None:
    uri = f"file:{corpus_path}?mode=ro"
    conn.execute("ATTACH DATABASE ? AS corpus", (uri,))


def _copy_legacy_rows(
    conn: sqlite3.Connection, corpus_path: Path,
) -> None:
    migrated = conn.execute(
        "SELECT value FROM runtime_metadata WHERE key = 'legacy_import_complete'"
    ).fetchone()
    if migrated:
        return
    _attach_corpus(conn, corpus_path)
    try:
        available = {
            row[0] for row in conn.execute(
                "SELECT name FROM corpus.sqlite_master WHERE type = 'table'"
            )
        }
        for table in RUNTIME_TABLES:
            if table not in available:
                continue
            columns = [
                row[1] for row in conn.execute(
                    f"PRAGMA corpus.table_info({table})"
                )
            ]
            if not columns:
                continue
            quoted = ", ".join(f'"{column}"' for column in columns)
            conn.execute(
                f'INSERT OR IGNORE INTO main."{table}" ({quoted}) '
                f'SELECT {quoted} FROM corpus."{table}"'
            )
        conn.execute(
            "INSERT OR REPLACE INTO runtime_metadata(key, value) "
            "VALUES ('legacy_import_complete', datetime('now'))"
        )
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE corpus")


def initialize_runtime_db(corpus_path: Path) -> Path:
    runtime_path = runtime_db_path(corpus_path)
    key = (corpus_path.resolve(), runtime_path.resolve())
    if key in _INITIALIZED:
        return runtime_path
    with _INIT_LOCK:
        if key in _INITIALIZED:
            return runtime_path
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(runtime_path, uri=True)
        try:
            conn.executescript(_SCHEMA_PATH.read_text())
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runtime_metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.commit()
            _copy_legacy_rows(conn, corpus_path)
        finally:
            conn.close()
        _INITIALIZED.add(key)
    return runtime_path


@contextmanager
def get_runtime_conn(corpus_path: Path, readonly: bool = True):
    runtime_path = initialize_runtime_db(corpus_path)
    conn = sqlite3.connect(
        runtime_path, check_same_thread=False, uri=True,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _attach_corpus(conn, corpus_path)
    try:
        yield conn
    finally:
        conn.close()
