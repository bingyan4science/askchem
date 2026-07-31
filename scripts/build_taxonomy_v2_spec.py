#!/usr/bin/env python3
"""Build the canonical taxonomy-v2 spec from reviewed merges and survivors."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["ASKCHEM_DISABLE_TAXONOMY_V2"] = "1"

from askchem.taxonomy import CANONICAL_L1, CANONICAL_L2, CANONICAL_L3  # noqa: E402


def sorted_bucket(values) -> list[str]:
    unique = {value for value in values if value}
    unique.discard("other")
    return sorted(unique) + ["other"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "askchem.db")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--l2-audit",
        type=Path,
        default=ROOT / "data/audits/l2/gemini_validation_cache.json",
    )
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument(
        "--survivor-min-claims",
        type=int,
        default=250,
        help="Retain high-support L2 nodes even if absent from pair audit",
    )
    parser.add_argument(
        "--l3-survivor-min-claims",
        type=int,
        default=250,
        help="Retain high-support L3 nodes under retained L2 parents",
    )
    parser.add_argument(
        "--l3-additions",
        type=Path,
        default=ROOT / "data/audits/l3/proposed_l3_additions_cleaned.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    l2 = {
        view: {
            l1: set(values)
            for l1, values in parents.items()
        }
        for view, parents in CANONICAL_L2.items()
    }
    # Expand only with Gemini-reviewed distinct categories. Unreviewed
    # free-form L2 labels remain outside the spec and migrate to "other".
    l2_audit = json.loads(args.l2_audit.read_text())
    for record in l2_audit.values():
        view = record["view"]
        l1 = record["l1"]
        if view not in CANONICAL_L1 or l1 not in CANONICAL_L1[view]:
            continue
        if record.get("decision") == "merge":
            candidates = [(record["big"], record.get("big_n", 0))]
        elif record.get("decision") in {
            "keep_separate", "demote_small_to_l3",
        }:
            candidates = [
                (record["big"], record.get("big_n", 0)),
                (record["small"], record.get("small_n", 0)),
            ]
        else:
            continue
        for candidate, support in candidates:
            if int(support or 0) >= args.min_support:
                l2.setdefault(view, {}).setdefault(l1, set()).add(candidate)

    # Pair audits only cover lexical neighbors. Preserve high-support distinct
    # nodes so tightening does not collapse major scientific areas to "other".
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    for view, path in conn.execute(
        "SELECT view_id, path FROM tree_nodes "
        "WHERE level = 2 AND claim_count >= ?",
        (args.survivor_min_claims,),
    ):
        parts = path.split("/")
        if (view in CANONICAL_L1 and len(parts) == 2
                and parts[0] in CANONICAL_L1[view]):
            l2.setdefault(view, {}).setdefault(parts[0], set()).add(parts[1])
    l3 = {
        view: {
            f"{l1}/{l2_name}": list(values)
            for (l1, l2_name), values in parents.items()
        }
        for view, parents in CANONICAL_L3.items()
    }
    if args.l3_additions.exists():
        additions = json.loads(args.l3_additions.read_text()).get("additions", [])
        for record in additions:
            key = f"{record['l1']}/{record['l2']}"
            l3.setdefault(record["view"], {}).setdefault(key, ["other"]).append(
                record["new_l3"]
            )
    for view, path in conn.execute(
        "SELECT view_id, path FROM tree_nodes "
        "WHERE level = 3 AND claim_count >= ?",
        (args.l3_survivor_min_claims,),
    ):
        parts = path.split("/")
        if len(parts) != 3 or view not in l2:
            continue
        if parts[1] not in l2.get(view, {}).get(parts[0], set()):
            continue
        l3.setdefault(view, {}).setdefault(
            f"{parts[0]}/{parts[1]}", ["other"],
        ).append(parts[2])
    conn.close()

    payload = {
        "taxonomy_version": registry["taxonomy_version"],
        "source_registry": str(args.registry),
        "policy": {
            "l1": "fixed",
            "l2": "reviewed synonym merges plus distinct surviving nodes",
            "l3": "canonical definitions plus reviewed cleaned additions",
            "max_depth": 3,
            "l2_demotions": "excluded",
        },
        "canonical_l1": CANONICAL_L1,
        "canonical_l2": {
            view: {
                l1: sorted_bucket(values)
                for l1, values in parents.items()
            }
            for view, parents in l2.items()
        },
        "canonical_l3": {
            view: {
                key: sorted_bucket(values)
                for key, values in parents.items()
            }
            for view, parents in l3.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "taxonomy_version": payload["taxonomy_version"],
        "l2_nodes": sum(
            len(values)
            for parents in payload["canonical_l2"].values()
            for values in parents.values()
        ),
        "l3_nodes": sum(
            len(values)
            for parents in payload["canonical_l3"].values()
            for values in parents.values()
        ),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
