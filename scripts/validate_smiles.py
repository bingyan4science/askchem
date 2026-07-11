"""Validate SMILES strings in claims using RDKit and store results."""
import sqlite3
import json
import sys
import time

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"
BATCH_SIZE = 5000


def main():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Create smiles_validations table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS smiles_validations (
            claim_id TEXT PRIMARY KEY,
            smiles TEXT,
            is_valid INTEGER DEFAULT 0,
            validated_at TEXT
        )
    """)
    conn.commit()

    already = set(r[0] for r in conn.execute("SELECT claim_id FROM smiles_validations").fetchall())
    print(f"Already validated: {len(already)}")

    # Find all claims with subject_smiles
    print("Scanning claims for SMILES...")
    total = 0
    valid = 0
    invalid = 0
    skipped = 0
    too_short = 0
    start = time.time()

    offset = 0
    while True:
        rows = conn.execute(
            "SELECT claim_id, data FROM claims LIMIT ? OFFSET ?",
            [BATCH_SIZE, offset],
        ).fetchall()
        if not rows:
            break
        offset += len(rows)

        batch_inserts = []
        for r in rows:
            claim_id = r["claim_id"]
            if claim_id in already:
                continue

            d = json.loads(r["data"])
            smiles = d.get("subject_smiles")
            if not smiles or str(smiles).strip() in ('', 'None', 'null', 'N/A', 'none'):
                continue

            smiles = str(smiles).strip()
            total += 1

            if len(smiles) < 2:
                too_short += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            is_valid = 1 if mol is not None else 0
            if is_valid:
                valid += 1
            else:
                invalid += 1

            batch_inserts.append((claim_id, smiles, is_valid))

        if batch_inserts:
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            conn.executemany(
                "INSERT OR REPLACE INTO smiles_validations (claim_id, smiles, is_valid, validated_at) VALUES (?,?,?,?)",
                [(cid, sm, iv, now) for cid, sm, iv in batch_inserts],
            )
            conn.commit()

        if offset % 50000 == 0:
            elapsed = time.time() - start
            print(f"  Scanned {offset} claims | SMILES found: {total} | valid: {valid} | invalid: {invalid} | too_short: {too_short} | {elapsed:.0f}s")

    elapsed = time.time() - start
    sv_total = conn.execute("SELECT COUNT(*) FROM smiles_validations").fetchone()[0]
    sv_valid = conn.execute("SELECT COUNT(*) FROM smiles_validations WHERE is_valid = 1").fetchone()[0]
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"SMILES found: {total} (excluding empty/None)")
    print(f"  Too short (<2 chars): {too_short}")
    print(f"  Validated: {valid + invalid}")
    print(f"  Valid: {valid} ({valid/(valid+invalid)*100:.1f}% if any)" if (valid + invalid) > 0 else "  No SMILES to validate")
    print(f"  Invalid: {invalid}")
    print(f"smiles_validations table: {sv_total} rows, {sv_valid} valid")
    conn.close()


if __name__ == "__main__":
    main()
