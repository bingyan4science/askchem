#!/usr/bin/env python3
"""Classify claims from dual-approved composite taxonomy nodes into splits."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"
SYSTEM = """You are classifying chemistry claims after an ontology node was
split. Assign every claim to exactly one allowed canonical path. Base the choice
on the scientific subject of the claim, not incidental words. Use the fallback
`other` path only when no specific child is supported. Chemical elements and
formulas are exact identities. Return only JSON:
{"assignments":[{"claim_id":"...", "new_path":"slash/path",
"confidence":"high|medium|low", "rationale":"brief"}]}"""

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_claims(db: Path, splits: list[dict]) -> dict[tuple[str, str], list[dict]]:
    wanted = {
        (record["view"], tuple(record["old_path"].split("/"))): record
        for record in splits
    }
    result = {key: [] for key in wanted}
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    for row in conn.execute(
        "SELECT claim_id,claim_type,source_paper_title,verbatim_quote,view_paths "
        "FROM claims WHERE view_paths IS NOT NULL"
    ):
        try:
            view_paths = json.loads(row["view_paths"])
        except (json.JSONDecodeError, TypeError):
            continue
        for key in wanted:
            view, old_path = key
            path = view_paths.get(view)
            if isinstance(path, list) and tuple(path[:3]) == old_path:
                result[key].append({
                    "claim_id": row["claim_id"],
                    "claim_type": row["claim_type"],
                    "paper_title": (row["source_paper_title"] or "")[:300],
                    "claim": (row["verbatim_quote"] or "")[:1000],
                })
    conn.close()
    return result


def classify_batch(
    split: dict, claims: list[dict], model: str, timeout: int,
) -> list[dict]:
    key = os.environ.get("PORTKEY_API_KEY")
    if not key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    allowed = [*split["new_paths"], split["fallback_path"]]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "view": split["view"],
                "old_composite_path": split["old_path"],
                "allowed_paths": allowed,
                "claims": claims,
            }, ensure_ascii=False)},
        ],
        "thinking_level": "high",
        "temperature": 0.1,
        "max_completion_tokens": 32768,
        "response_format": {"type": "json_object"},
    }
    response = _session().post(
        GATEWAY,
        headers={
            "x-portkey-api-key": key,
            "x-portkey-provider": PROVIDER,
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text[:500]}")
    payload = response.json()
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    assignments = parsed.get("assignments") or []
    expected = {claim["claim_id"] for claim in claims}
    actual = {item.get("claim_id") for item in assignments}
    if expected != actual:
        raise ValueError(
            f"assignment coverage mismatch missing={expected - actual} "
            f"extra={actual - expected}"
        )
    for item in assignments:
        if item.get("new_path") not in allowed:
            raise ValueError(f"invalid split path: {item.get('new_path')}")
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    splits = json.loads(args.registry.read_text()).get("splits", [])
    claims_by_split = load_claims(args.db, splits)
    split_by_key = {
        (record["view"], tuple(record["old_path"].split("/"))): record
        for record in splits
    }
    cache = json.loads(args.output.read_text()) if args.output.exists() else {}
    jobs = []
    for key, claims in claims_by_split.items():
        pending = [
            claim for claim in claims
            if f"{key[0]}:{claim['claim_id']}" not in cache
        ]
        for start in range(0, len(pending), args.batch_size):
            jobs.append((key, pending[start:start + args.batch_size]))
    lock = threading.Lock()
    failures = []

    def run(key, claims):
        error = ""
        for attempt in range(3):
            try:
                return classify_batch(
                    split_by_key[key], claims, args.model, args.timeout,
                )
            except Exception as exc:
                error = str(exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(error)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run, key, claims): (key, claims)
            for key, claims in jobs
        }
        completed = 0
        for future in as_completed(futures):
            key, claims = futures[future]
            try:
                assignments = future.result()
            except Exception as exc:
                failures.append({
                    "view": key[0], "old_path": "/".join(key[1]),
                    "claim_ids": [claim["claim_id"] for claim in claims],
                    "error": str(exc),
                })
                continue
            with lock:
                split = split_by_key[key]
                for item in assignments:
                    cache[f"{key[0]}:{item['claim_id']}"] = {
                        **item,
                        "view": key[0],
                        "old_path": split["old_path"],
                        "model": args.model,
                        "thinking_level": "high",
                    }
                _atomic(args.output, cache)
                completed += 1
                print(f"[{completed}/{len(jobs)}] classified split batch", flush=True)
    if failures:
        failure_path = args.output.with_suffix(args.output.suffix + ".failures.json")
        _atomic(failure_path, failures)
        raise RuntimeError(f"{len(failures)} split batches failed: {failure_path}")
    print(json.dumps({
        "splits": len(splits),
        "assignments": len(cache),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
