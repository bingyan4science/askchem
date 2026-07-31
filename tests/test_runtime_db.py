import sqlite3

from askchem import db
from askchem import runtime_db


def test_private_user_state_is_written_outside_corpus(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.db"
    runtime = tmp_path / "runtime.db"
    conn = sqlite3.connect(corpus)
    conn.executescript("""
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        );
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            email TEXT,
            display_name TEXT,
            clerk_id TEXT,
            created_at TEXT,
            last_login_at TEXT
        );
    """)
    conn.close()
    monkeypatch.setenv("ASKCHEM_DB", str(corpus))
    monkeypatch.setenv("ASKCHEM_RUNTIME_DB", str(runtime))
    runtime_db._INITIALIZED.clear()

    created = db.create_user("chemist@example.org", "Chemist")

    runtime_conn = sqlite3.connect(runtime)
    assert runtime_conn.execute(
        "SELECT email FROM users WHERE user_id = ?", (created["user_id"],)
    ).fetchone()[0] == "chemist@example.org"
    runtime_conn.close()
    corpus_conn = sqlite3.connect(corpus)
    assert corpus_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    corpus_conn.close()


def test_public_database_uses_table_allowlist(tmp_path):
    from upload_to_hf import build_public_database

    service_db = tmp_path / "service.db"
    public_db = tmp_path / "public.db"
    conn = sqlite3.connect(service_db)
    conn.executescript("""
        CREATE TABLE claims (claim_id TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE sources (doi TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE users (user_id TEXT PRIMARY KEY, email TEXT);
        CREATE TABLE query_log (id INTEGER PRIMARY KEY, query TEXT);
        INSERT INTO claims VALUES ('c1', '{}');
        INSERT INTO users VALUES ('u1', 'private@example.org');
        INSERT INTO query_log(query) VALUES ('private query');
    """)
    conn.close()

    build_public_database(service_db, public_db)

    conn = sqlite3.connect(public_db)
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "claims" in tables
    assert "sources" in tables
    assert "users" not in tables
    assert "query_log" not in tables
    conn.close()
