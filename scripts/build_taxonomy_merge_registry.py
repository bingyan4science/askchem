#!/usr/bin/env python3
"""Build a versioned, path-scoped taxonomy merge registry from reviewed audits."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from askchem.taxonomy import CANONICAL_L2, CANONICAL_L3  # noqa: E402
from snap_l3_aliases import CURATED_ALIASES  # noqa: E402


def close_l2_map(raw: dict[tuple[str, str, str], str]) -> dict:
    closed = {}
    for key, target in raw.items():
        view, l1, _ = key
        visited = {key[2]}
        while (view, l1, target) in raw:
            if target in visited:
                raise ValueError(f"cycle in L2 merge map at {key!r}")
            visited.add(target)
            target = raw[(view, l1, target)]
        closed[key] = target
    return closed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "askchem.db")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/audits/l2/gemini_validation_cache.json",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--view", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    allowed_views = set(args.view or [])
    decisions = json.loads(args.cache.read_text())
    raw_l2 = {}
    evidence = {}
    for record in decisions.values():
        if record.get("decision") != "merge":
            continue
        if allowed_views and record["view"] not in allowed_views:
            continue
        key = (record["view"], record["l1"], record["small"])
        raw_l2[key] = record["big"]
        evidence[key] = record
    l2_map = close_l2_map(raw_l2)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    mappings = []
    statuses = Counter()
    rows = conn.execute(
        "SELECT view_id, path, claim_count FROM tree_nodes ORDER BY view_id, path"
    )
    for row in rows:
        view = row["view_id"]
        if allowed_views and view not in allowed_views:
            continue
        parts = row["path"].split("/")
        if len(parts) < 2:
            continue
        key = (view, parts[0], parts[1])
        target_l2 = l2_map.get(key)
        if target_l2:
            proposed = [parts[0], target_l2]
            status = "approved"
            note = "reviewed L2 synonym merge"
            if len(parts) >= 3:
                allowed_l3 = CANONICAL_L3.get(view, {}).get(
                    (parts[0], target_l2)
                )
                if allowed_l3 is not None and parts[2] in allowed_l3:
                    proposed.append(parts[2])
                else:
                    status = "needs_l3_review"
                    note = "L3 is not canonical under merged parent"
            if len(parts) > 3:
                status = "needs_l3_review"
                note = "source path exceeds taxonomy-v2 depth"
            rec = evidence[key]
            mappings.append({
                "view": view,
                "old_path": row["path"],
                "new_path": "/".join(proposed),
                "level": "l2",
                "status": status,
                "reason": rec.get("reason"),
                "confidence": rec.get("confidence"),
                "affected_claims": int(row["claim_count"] or 0),
                "note": note,
            })
            statuses[status] += 1
            continue

        if len(parts) == 3:
            aliases = CURATED_ALIASES.get((view, parts[0], parts[1]), {})
            target_l3 = aliases.get(parts[2])
            if target_l3:
                mappings.append({
                    "view": view,
                    "old_path": row["path"],
                    "new_path": "/".join([parts[0], parts[1], target_l3]),
                    "level": "l3",
                    "status": "approved",
                    "reason": "curated L3 alias",
                    "confidence": "high",
                    "affected_claims": int(row["claim_count"] or 0),
                    "note": "target is canonical under the same parent",
                })
                statuses["approved"] += 1

    conn.close()
    payload = {
        "taxonomy_version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(args.db.resolve()),
        "source_audit": str(args.cache),
        "views": sorted(allowed_views) if allowed_views else "all",
        "policy": {
            "max_depth": 3,
            "l2_demotions": "excluded",
            "needs_l3_review": "not applied",
        },
        "counts": {
            "reviewed_l2_merges": len(l2_map),
            "mappings": len(mappings),
            "by_status": dict(statuses),
        },
        "mappings": mappings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
