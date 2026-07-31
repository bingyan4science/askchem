import sqlite3

from upload_to_hf import build_public_database


def test_public_database_copies_allowlist_without_private_bytes(tmp_path):
    source = tmp_path / "service.db"
    public = tmp_path / "public.db"
    secret = "PRIVATE-QUERY-MARKER-7df6cbb1"

    conn = sqlite3.connect(source)
    conn.executescript(
        """
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE claims_fts USING fts5(claim_id, text);
        CREATE TABLE query_log (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL
        );
        INSERT INTO claims VALUES ('claim-1', 'Suzuki coupling');
        INSERT INTO claims_fts VALUES ('claim-1', 'Suzuki coupling');
        """
    )
    conn.execute("INSERT INTO query_log(query) VALUES (?)", (secret,))
    conn.commit()
    conn.close()

    build_public_database(source, public)

    conn = sqlite3.connect(public)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "claims" in tables
    assert "claims_fts" in tables
    assert "query_log" not in tables
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    assert conn.execute(
        "SELECT claim_id FROM claims_fts WHERE claims_fts MATCH 'Suzuki'"
    ).fetchone()[0] == "claim-1"
    conn.close()

    assert secret.encode() not in public.read_bytes()
