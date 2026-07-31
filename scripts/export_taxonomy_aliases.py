#!/usr/bin/env python3
"""Export approved taxonomy registry mappings as runtime path aliases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from askchem.taxonomy_semantics import assert_formula_safe_alias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    aliases: dict[str, dict[str, str]] = defaultdict(dict)
    for record in registry["mappings"]:
        if record.get("status") != "approved":
            continue
        if record["old_path"] != record["new_path"]:
            assert_formula_safe_alias(
                record["old_path"], record["new_path"],
            )
            aliases[record["view"]][record["old_path"]] = record["new_path"]
    payload = {
        "taxonomy_version": registry["taxonomy_version"],
        "aliases": {
            view: dict(sorted(paths.items()))
            for view, paths in sorted(aliases.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(sum(len(paths) for paths in aliases.values()), "aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
