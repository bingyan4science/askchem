"""One-off: relabel deep_v1 claims with the actual extraction model (Gemini 3.1 Pro).

Background: integrate_deep.py + classify_papers.py rebuild had a bug where they
hardcoded extraction_model='gpt-5.4' for all deep_v1 rows, even though the
underlying extraction was performed with Gemini 3.1 Pro via Vertex/Portkey.

This script fixes both:
  - the extraction_model column
  - the embedded JSON 'data' column

Run as: python scripts/fix_deep_v1_model.py [path/to/chemtree.db]
"""
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "chemtree.db"
NEW_MODEL = "gemini-3.1-pro"


def main(db_path: Path) -> int:
    print(f"db: {db_path} ({db_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    print("\n[before]", flush=True)
    for v, n in cur.execute(
        "SELECT extraction_model, COUNT(*) FROM claims "
        "WHERE extraction_version='deep_v1' GROUP BY extraction_model"
    ):
        print(f"  {v!r}: {n:,}", flush=True)

    print(f"\n[1/2] UPDATE extraction_model column -> {NEW_MODEL!r} ...", flush=True)
    t0 = time.time()
    cur.execute(
        "UPDATE claims SET extraction_model=? "
        "WHERE extraction_version='deep_v1' AND extraction_model<>?",
        (NEW_MODEL, NEW_MODEL),
    )
    print(f"  rows updated: {cur.rowcount:,}  ({time.time()-t0:.1f}s)", flush=True)
    con.commit()

    print(f"\n[2/2] UPDATE embedded JSON ('data' column) -> {NEW_MODEL!r} ...", flush=True)
    t0 = time.time()
    cur.execute(
        "UPDATE claims SET data = json_set(data, '$.extraction_model', ?) "
        "WHERE extraction_version='deep_v1'",
        (NEW_MODEL,),
    )
    print(f"  rows updated: {cur.rowcount:,}  ({time.time()-t0:.1f}s)", flush=True)
    con.commit()

    print("\n[after]", flush=True)
    for v, n in cur.execute(
        "SELECT extraction_model, COUNT(*) FROM claims "
        "WHERE extraction_version='deep_v1' GROUP BY extraction_model"
    ):
        print(f"  {v!r}: {n:,}", flush=True)
    sample = cur.execute(
        "SELECT data FROM claims WHERE extraction_version='deep_v1' LIMIT 1"
    ).fetchone()
    import json
    print("  sample data.extraction_model:",
          json.loads(sample[0]).get("extraction_model"), flush=True)
    con.close()
    print("\ndone.", flush=True)
    return 0


if __name__ == "__main__":
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    sys.exit(main(db))
