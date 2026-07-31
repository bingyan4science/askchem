#!/usr/bin/env python3
"""Pool candidates from every AskChem retrieval channel for eval v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem import db, retrieval  # noqa: E402
from eval_common import PROBES_PATH, load_probes, write_jsonl  # noqa: E402


CHANNEL_KEYS = {
    "fts": "fts_pool",
    "dense": "vector_pool",
    "tree": "tree_pool",
    "source_paper": "source_paper_pool",
    "claim_guided_paper": "claim_guided_paper_pool",
    "author": "author_pool",
    "production": "final_top",
}


def pool_probe(probe, per_channel: int) -> dict:
    trace: dict = {}
    db.search_claims(
        probe.q,
        claim_type=probe.claim_type,
        view=probe.view,
        limit=max(50, per_channel),
        mode=probe.mode,
        sort=probe.sort,
        _trace_into=trace,
    )
    candidate_ids: list[str] = []
    seen: set[str] = set()
    sources: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for channel, key in CHANNEL_KEYS.items():
        ids = [
            cid for cid in trace.get(key, [])[:per_channel]
            if isinstance(cid, str) and cid
        ]
        counts[channel] = len(ids)
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                candidate_ids.append(cid)
            sources.setdefault(cid, []).append(channel)
    counts["union"] = len(candidate_ids)
    return {
        "id": probe.id,
        "q": probe.q,
        "family": probe.family,
        "view": probe.view,
        "claim_type": probe.claim_type,
        "mode": probe.mode,
        "sort": probe.sort,
        "candidate_ids": candidate_ids,
        "sources": sources,
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data/eval/candidates_v2.jsonl",
    )
    parser.add_argument("--per-channel", type=int, default=30)
    args = parser.parse_args()

    # Candidate generation must execute every channel; cached final responses
    # do not contain intermediate pools.
    os.environ["CHEMTREE_SEARCH_CACHE"] = "0"
    retrieval.load_embeddings()
    probes = load_probes(args.probes)
    rows = []
    for index, probe in enumerate(probes, 1):
        row = pool_probe(probe, args.per_channel)
        rows.append(row)
        print(
            f"[{index:>3}/{len(probes)}] {probe.id}: "
            + " ".join(f"{k}={v}" for k, v in row["counts"].items()),
            flush=True,
        )
    write_jsonl(args.out, rows)
    print(json.dumps({
        "probes": len(rows),
        "candidates": sum(r["counts"]["union"] for r in rows),
        "output": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
