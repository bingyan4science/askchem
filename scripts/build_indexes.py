"""Build indexes for performance: subject_smiles and claim_view_map."""
import sqlite3
import json
import sys
import time

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # 1. Create index on json_extract(data, '$.subject_smiles') for fast structure search
    print("Creating subject_smiles index...")
    t0 = time.time()
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_smiles "
            "ON claims(json_extract(data, '$.subject_smiles')) "
            "WHERE json_extract(data, '$.subject_smiles') IS NOT NULL "
            "AND json_extract(data, '$.subject_smiles') != ''"
        )
        conn.commit()
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # 2. Build claim_view_map for fast reading list and node lookups
    print("\nBuilding claim_view_map table...")
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS claim_view_map")
    conn.execute("""
        CREATE TABLE claim_view_map (
            claim_id TEXT NOT NULL,
            view_id TEXT NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (claim_id, view_id, path)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cvm_view_path ON claim_view_map(view_id, path)")
    conn.commit()

    BATCH_SIZE = 5000
    offset = 0
    inserted = 0

    while True:
        rows = conn.execute(
            "SELECT claim_id, view_paths FROM claims "
            "WHERE view_paths IS NOT NULL AND view_paths != '' AND view_paths != '{}' "
            "LIMIT ? OFFSET ?",
            [BATCH_SIZE, offset],
        ).fetchall()
        if not rows:
            break
        offset += len(rows)

        batch = []
        for r in rows:
            try:
                vp = json.loads(r["view_paths"])
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(vp, dict):
                continue

            for view_id, paths in vp.items():
                if isinstance(paths, list):
                    # Reconstruct hierarchical paths from the leaf segments
                    full_path = "/".join(paths)
                    batch.append((r["claim_id"], view_id, full_path))
                    # Also add intermediate paths for subtree queries
                    for i in range(1, len(paths)):
                        batch.append((r["claim_id"], view_id, "/".join(paths[:i])))

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO claim_view_map (claim_id, view_id, path) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)

        if offset % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {offset} claims, {inserted} mappings, {elapsed:.0f}s")

    elapsed = time.time() - t0
    total = conn.execute("SELECT COUNT(*) FROM claim_view_map").fetchone()[0]
    print(f"  Done in {elapsed:.0f}s: {total} mappings from {offset} claims")

    conn.close()


if __name__ == "__main__":
    main()
