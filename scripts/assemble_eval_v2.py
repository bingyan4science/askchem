#!/usr/bin/env python3
"""Assemble eval-v2 probes from the stable v1 set and reviewed additions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "data/eval/probes_v1.jsonl",
    )
    parser.add_argument(
        "--additions",
        type=Path,
        default=ROOT / "data/eval/probes_v2_additions.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/eval/probes_v2.jsonl",
    )
    args = parser.parse_args()

    records = []
    seen = set()
    for source in (args.base, args.additions):
        for line_number, raw in enumerate(source.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            probe_id = record["id"]
            if probe_id in seen:
                raise ValueError(
                    f"duplicate probe id {probe_id!r} at {source}:{line_number}"
                )
            seen.add(probe_id)
            records.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n"
                for record in records)
    )
    print(f"wrote {len(records)} probes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
