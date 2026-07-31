#!/usr/bin/env python3
"""Build a checksum-verified AskChem release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def artifact(path: Path) -> dict:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--taxonomy-version", required=True)
    parser.add_argument("--schema-version", default="1")
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--evaluation-report", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    counts = {}
    for table in ("claims", "sources", "tree_nodes", "claim_view_map"):
        counts[table] = conn.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]
    integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
    conn.close()
    if integrity != "ok":
        raise RuntimeError(f"database quick_check failed: {integrity}")

    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": args.taxonomy_version,
        "schema_version": args.schema_version,
        "git_sha": git_sha(),
        "huggingface_revision": args.hf_revision,
        "database": artifact(args.db),
        "counts": counts,
        "artifacts": [artifact(path) for path in args.artifact],
        "evaluation_reports": args.evaluation_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
