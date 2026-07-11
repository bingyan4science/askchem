"""Fix invalid SMILES in the database.

Handles three categories:
1. Text placeholders ("not determinable", "not applicable") -> mark as cleaned
2. Structurally valid but RDKit-unfriendly SMILES -> attempt canonicalization
3. Genuinely malformed SMILES -> clear from claim data
"""
import sqlite3
import json
import sys

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"

TEXT_PLACEHOLDERS = {
    "not determinable", "not applicable", "not available",
    "none", "n/a", "null", "unknown", "various", "mixture",
}


def main():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute(
        "SELECT claim_id, smiles FROM smiles_validations WHERE is_valid = 0"
    ).fetchall()
    print(f"Invalid SMILES to fix: {len(rows)}")

    cleaned_text = 0
    fixed_rdkit = 0
    cleared = 0

    for r in rows:
        claim_id = r["claim_id"]
        smi = r["smiles"]

        if smi.lower().strip() in TEXT_PLACEHOLDERS:
            _clear_smiles_from_claim(conn, claim_id)
            conn.execute(
                "DELETE FROM smiles_validations WHERE claim_id = ?", [claim_id]
            )
            cleaned_text += 1
            continue

        # Try sanitizing: remove charges, kekulize
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol)
                if canonical and Chem.MolFromSmiles(canonical) is not None:
                    _update_smiles_in_claim(conn, claim_id, canonical)
                    conn.execute(
                        "UPDATE smiles_validations SET smiles = ?, is_valid = 1 WHERE claim_id = ?",
                        [canonical, claim_id],
                    )
                    fixed_rdkit += 1
                    continue
            except Exception:
                pass

        # Try SMARTS interpretation (for patterns like [CX4])
        mol = Chem.MolFromSmarts(smi)
        if mol is not None:
            _clear_smiles_from_claim(conn, claim_id)
            conn.execute(
                "DELETE FROM smiles_validations WHERE claim_id = ?", [claim_id]
            )
            cleared += 1
            continue

        # Genuinely broken: clear from claim
        _clear_smiles_from_claim(conn, claim_id)
        conn.execute(
            "DELETE FROM smiles_validations WHERE claim_id = ?", [claim_id]
        )
        cleared += 1

    conn.commit()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM smiles_validations WHERE is_valid = 0"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM smiles_validations").fetchone()[0]
    valid = conn.execute(
        "SELECT COUNT(*) FROM smiles_validations WHERE is_valid = 1"
    ).fetchone()[0]

    print(f"\nResults:")
    print(f"  Text placeholders cleaned: {cleaned_text}")
    print(f"  Fixed via RDKit sanitize: {fixed_rdkit}")
    print(f"  Cleared (SMARTS/broken): {cleared}")
    print(f"  Remaining invalid: {remaining}")
    print(f"  Total in table: {total}, valid: {valid}")
    if total > 0:
        print(f"  Validation rate: {valid/total*100:.1f}%")

    conn.close()


def _clear_smiles_from_claim(conn, claim_id):
    row = conn.execute("SELECT data FROM claims WHERE claim_id = ?", [claim_id]).fetchone()
    if not row:
        return
    d = json.loads(row["data"])
    if "subject_smiles" in d:
        d["subject_smiles"] = ""
        conn.execute("UPDATE claims SET data = ? WHERE claim_id = ?", [json.dumps(d), claim_id])


def _update_smiles_in_claim(conn, claim_id, new_smiles):
    row = conn.execute("SELECT data FROM claims WHERE claim_id = ?", [claim_id]).fetchone()
    if not row:
        return
    d = json.loads(row["data"])
    d["subject_smiles"] = new_smiles
    conn.execute("UPDATE claims SET data = ? WHERE claim_id = ?", [json.dumps(d), claim_id])


if __name__ == "__main__":
    main()
