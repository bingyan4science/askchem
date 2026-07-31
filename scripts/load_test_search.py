#!/usr/bin/env python3
"""Small dependency-free load test for AskChem search and browse endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_QUERIES = [
    "Suzuki coupling",
    "TiO2 photocatalysis",
    "metal organic framework MOF",
    "solid state electrolyte",
    "NMR spectroscopy solid state",
    "hydrogen evolution reaction HER",
    "operando spectroscopy electrocatalysis",
    "cryo-EM protein structure",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1)
    return ordered[max(0, idx)]


def fetch(url: str, timeout: float) -> dict:
    started = time.monotonic()
    status = 0
    error = None
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = response.status
            size = len(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "url": url,
        "status": status,
        "latency_ms": (time.monotonic() - started) * 1000,
        "bytes": size,
        "error": error,
    }


def summarize(rows: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in rows]
    failures = [r for r in rows if r["status"] != 200]
    return {
        "requests": len(rows),
        "failures": len(failures),
        "error_rate": len(failures) / max(1, len(rows)),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95), 1) if latencies else None,
        "max_ms": round(max(latencies), 1) if latencies else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://askchem.org")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries:
        queries = [
            line.strip()
            for line in args.queries.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not queries:
        parser.error("query list is empty")

    base = args.base_url.rstrip("/")
    search_urls = [
        f"{base}/api/search?"
        + urllib.parse.urlencode({
            "q": queries[i % len(queries)],
            "limit": args.limit,
        })
        for i in range(args.requests)
    ]
    browse_urls = [
        f"{base}/api/views",
        f"{base}/api/stats",
        f"{base}/api/tree/by_reaction_type/?depth=1",
    ] * max(1, math.ceil(args.requests / 3))
    browse_urls = browse_urls[:args.requests]

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.concurrency),
    ) as pool:
        search_rows = list(pool.map(
            lambda url: fetch(url, args.timeout), search_urls,
        ))
    search_wall = time.monotonic() - started

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.concurrency),
    ) as pool:
        browse_rows = list(pool.map(
            lambda url: fetch(url, args.timeout), browse_urls,
        ))

    report = {
        "base_url": base,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "search": {
            **summarize(search_rows),
            "wall_seconds": round(search_wall, 3),
            "throughput_rps": round(len(search_rows) / max(search_wall, 1e-9), 3),
        },
        "browse": summarize(browse_rows),
        "errors": [
            {"url": r["url"], "status": r["status"], "error": r["error"]}
            for r in search_rows + browse_rows
            if r["status"] != 200
        ][:20],
        "rows": {"search": search_rows, "browse": browse_rows},
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 1 if report["search"]["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
