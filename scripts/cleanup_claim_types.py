#!/usr/bin/env python3
"""Normalize claim types in the askchem database.

Maps 129+ LLM-generated claim types down to ~13 canonical types by:
- Mapping known non-canonical types to canonical equivalents
- Fixing typos (conparison → comparison, etc.)
- Splitting compound types (property|comparison → property)
- Collapsing plurals (conclusions → finding)
- Mapping junk/meta types to 'observation'

After updating the claims table, rebuilds the FTS index.
"""
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from askchem.db import get_db_path, build_searchable_text

CANONICAL_TYPES = {
    "property", "method", "comparison", "mechanism", "computational",
    "reaction", "finding", "limitation", "future_direction", "hypothesis",
    "application", "observation", "spectroscopic",
}

DIRECT_MAP = {
    # Large legitimate types
    "computational_result": "computational",
    "scope_entry": "property",
    "surprising_finding": "finding",
    "structure": "property",
    "experimental_design": "method",
    "conclusion": "finding",
    "conclusions": "finding",
    "background": "observation",
    # Typos
    "conparison": "comparison",
    "comparision": "comparison",
    "conputational_result": "computational",
    "comparative_result": "comparison",
    # Plurals
    "limitations": "limitation",
    # Near-canonical
    "experimental_result": "finding",
    "result": "finding",
    "negative_result": "finding",
    "analytical_result": "finding",
    "finding": "finding",
    "outcome": "finding",
    "prediction": "hypothesis",
    "recommendation": "future_direction",
    "outlook": "future_direction",
    "future_scope": "future_direction",
    "future_opportunity": "future_direction",
    "novelty": "finding",
    "significance": "finding",
    "benefit": "property",
    "demonstration": "finding",
    "interpretation": "mechanism",
    "explanation": "mechanism",
    "implication": "finding",
    "optimization": "method",
    "measurement": "property",
    "classification": "property",
    "composition": "property",
    "material": "property",
    "product": "property",
    "synthesis": "reaction",
    "biological_property": "property",
    "background_property": "property",
    "control_experiment": "method",
    "control": "method",
    "field_observations": "observation",
    "literature_observation": "observation",
    "historical": "observation",
    "historical_fact": "observation",
    "historical_milestone": "observation",
    "historical_trend": "observation",
    "historical_comparison": "comparison",
    # Junk / meta
    "metadata": "observation",
    "claim": "observation",
    "claim_type": "observation",
    "claim_type:method": "method",
    "claim_type:comparison": "comparison",
    "other": "observation",
    "general": "observation",
    "general_fact": "observation",
    "context": "observation",
    "definition": "observation",
    "assumption": "observation",
    "purpose": "observation",
    "goal": "observation",
    "goal_statement": "observation",
    "problem": "observation",
    "problem_statement": "observation",
    "challenge": "observation",
    "motivation": "observation",
    "scope": "property",
    "conditions": "method",
    "dataset": "observation",
    "paper_type": "observation",
    "meta": "observation",
    "assignment": "property",
    "initial_state_assignment": "property",
    "methodology_outlook": "method",
    "metadata_based_method": "method",
    "analysis": "method",
    "projection/method": "method",
    "scale_up": "method",
}


def resolve_type(raw: str) -> str:
    """Map a raw claim type to a canonical type."""
    if not raw:
        return "observation"

    t = raw.strip().lower()

    if t in CANONICAL_TYPES:
        return t

    if t in DIRECT_MAP:
        return DIRECT_MAP[t]

    # Compound types: split on | or / and take the first component
    if "|" in t or "/" in t:
        first = re.split(r"[|/]", t)[0].strip()
        return resolve_type(first)

    # Parenthesized types like "claim (method/property)"
    paren = re.search(r"\(([^)]+)\)", t)
    if paren:
        return resolve_type(paren.group(1))

    return "observation"


def main():
    db_path = get_db_path()
    print(f"Database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    cur = conn.execute(
        "SELECT json_extract(data, '$.claim_type') as ct, COUNT(*) as n "
        "FROM claims GROUP BY ct ORDER BY n DESC"
    )
    before = {row[0]: row[1] for row in cur.fetchall()}
    total = sum(before.values())
    print(f"\nBEFORE: {len(before)} distinct types across {total:,} claims")
    for ct, n in sorted(before.items(), key=lambda x: -x[1])[:20]:
        canon = resolve_type(ct) if ct not in CANONICAL_TYPES else ct
        tag = f" → {canon}" if ct not in CANONICAL_TYPES else ""
        print(f"  {n:>8}  {ct}{tag}")
    if len(before) > 20:
        print(f"  ... and {len(before) - 20} more types")

    # Build update batches: group by (old_type → new_type)
    updates = {}
    for ct in before:
        new_ct = resolve_type(ct)
        if new_ct != ct:
            updates[ct] = new_ct

    affected = sum(before.get(old, 0) for old in updates)
    print(f"\n{len(updates)} types to remap, affecting {affected:,} claims")

    if not updates:
        print("Nothing to do.")
        return

    # Update claims in batches by old type
    changed = 0
    for old_type, new_type in sorted(updates.items(), key=lambda x: -before.get(x[0], 0)):
        count = before.get(old_type, 0)
        if count == 0:
            continue

        rows = conn.execute(
            "SELECT claim_id, data FROM claims "
            "WHERE json_extract(data, '$.claim_type') = ?",
            (old_type,),
        ).fetchall()

        batch = []
        for claim_id, data_str in rows:
            data = json.loads(data_str)
            data["claim_type"] = new_type
            batch.append((json.dumps(data, ensure_ascii=False), new_type, claim_id))

        conn.executemany(
            "UPDATE claims SET data = ?, claim_type = ? WHERE claim_id = ?",
            batch,
        )
        conn.commit()
        changed += len(batch)
        print(f"  {old_type} → {new_type}: {len(batch):,} claims")

    print(f"\nUpdated {changed:,} claims total")

    # Rebuild FTS index
    print("\nRebuilding FTS index...")
    conn.execute("DELETE FROM claims_fts")
    conn.commit()

    batch_size = 5000
    offset = 0
    fts_count = 0
    while True:
        rows = conn.execute(
            "SELECT data FROM claims LIMIT ? OFFSET ?", (batch_size, offset)
        ).fetchall()
        if not rows:
            break
        fts_batch = []
        for (data_str,) in rows:
            cdata = json.loads(data_str)
            searchable = build_searchable_text(cdata)
            fts_batch.append((
                cdata.get("claim_id", ""),
                cdata.get("claim_type", ""),
                cdata.get("source_paper_title", ""),
                cdata.get("verbatim_quote", ""),
                searchable,
            ))
        conn.executemany(
            "INSERT INTO claims_fts(claim_id, claim_type, source_paper_title, "
            "verbatim_quote, searchable_text) VALUES (?,?,?,?,?)",
            fts_batch,
        )
        conn.commit()
        fts_count += len(fts_batch)
        if fts_count % 50000 == 0 or len(rows) < batch_size:
            print(f"  FTS indexed: {fts_count:,}", flush=True)
        offset += batch_size

    print(f"  FTS rebuild complete: {fts_count:,} entries")

    # Print after distribution
    cur = conn.execute(
        "SELECT json_extract(data, '$.claim_type') as ct, COUNT(*) as n "
        "FROM claims GROUP BY ct ORDER BY n DESC"
    )
    after = {row[0]: row[1] for row in cur.fetchall()}
    print(f"\nAFTER: {len(after)} distinct types across {sum(after.values()):,} claims")
    for ct, n in sorted(after.items(), key=lambda x: -x[1]):
        print(f"  {n:>8}  {ct}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
