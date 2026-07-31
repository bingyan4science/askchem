#!/usr/bin/env python3
"""Pre-warm askchem's LRU result cache with likely internal-test queries.

Run on a self-hosted instance (or via cron) shortly after a service restart
and periodically thereafter.
Each query hits the local FastAPI endpoint; the search_claims LRU
short-circuits subsequent identical requests for ~50 ms response.

Usage:
    python scripts/prewarm_cache.py

Environment overrides:
    ASKCHEM_LOCAL_URL   default http://127.0.0.1:8420
    PREWARM_LIMIT       default 20 (results per query, matches typical UI)
    PREWARM_TIMEOUT_S   default 60
    PREWARM_PARALLEL    default 1 (sequential = predictable, no
                                  thread oversubscription on the box)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_URL = os.environ.get("ASKCHEM_LOCAL_URL", "http://127.0.0.1:8420")
LIMIT = int(os.environ.get("PREWARM_LIMIT", "20"))
TIMEOUT_S = int(os.environ.get("PREWARM_TIMEOUT_S", "60"))
PARALLEL = int(os.environ.get("PREWARM_PARALLEL", "1"))

# Representative query set for chemistry-literature retrieval. Covers
# the four big "intent families" we benchmark against (reaction,
# material, property, technique) plus a handful of homonym / multi
# queries that surface different recall channels.
QUERIES: list[str] = [
    # --- reactions / catalysis (high traffic) ---
    "Suzuki coupling",
    "Suzuki-Miyaura coupling boronic acid",
    "Buchwald-Hartwig amination",
    "Heck reaction palladium",
    "Sonogashira coupling",
    "Negishi coupling zinc",
    "Stille coupling tin",
    "Mizoroki-Heck reaction",
    "Ullmann coupling copper",
    "olefin metathesis Grubbs",
    "asymmetric hydrogenation",
    "C-H activation",
    "photoredox catalysis visible light",
    "Wittig reaction",
    "Diels-Alder cycloaddition",
    "click chemistry azide alkyne",
    "ring-closing metathesis",
    "carbonylation palladium",
    "hydroformylation rhodium",
    "asymmetric organocatalysis proline",
    # --- materials / energy ---
    "perovskite solar cell efficiency",
    "halide perovskite stability",
    "metal organic framework MOF",
    "covalent organic framework",
    "lithium battery cathode",
    "lithium iron phosphate LFP",
    "sodium ion battery",
    "solid state electrolyte",
    "PEM fuel cell platinum",
    "graphene field effect",
    "MXene electrode supercapacitor",
    "TiO2 photocatalysis",
    "ZnO photocatalyst hydrogen evolution",
    "carbon nanotube transistor",
    "bismuth telluride thermoelectric",
    # --- properties / phenomena ---
    "CO2 reduction electrocatalyst",
    "CO2 capture amine",
    "oxygen reduction reaction ORR",
    "hydrogen evolution reaction HER",
    "nitrogen reduction reaction ammonia",
    "spin crossover iron complex",
    "single molecule magnet dysprosium",
    "anti-aromatic compound",
    # --- techniques / methods (sometimes confused with materials) ---
    "X-ray diffraction crystal structure",
    "NMR spectroscopy solid state",
    "DFT calculation transition state",
    "operando spectroscopy electrocatalysis",
    "cryo-EM protein structure",
    "mass spectrometry proteomics",
    # --- homonyms / multi-intent (good cache-coverage stress) ---
    "lead halide",
    "copper catalyst",
    "iron nitrogen carbon ORR",
    "anti-aromatic Hückel",
]


def hit(q: str) -> tuple[str, float, int, str]:
    url = f"{LOCAL_URL}/api/search?q={urllib.parse.quote(q)}&limit={LIMIT}"
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            body = json.loads(r.read())
        dt = (time.monotonic() - t0) * 1000
        n = len(body.get("results", []))
        return (q, dt, n, "ok")
    except Exception as exc:
        dt = (time.monotonic() - t0) * 1000
        return (q, dt, 0, f"FAIL: {exc!r}")


def main() -> int:
    print(
        f"[prewarm] {len(QUERIES)} queries  limit={LIMIT}  "
        f"parallel={PARALLEL}  url={LOCAL_URL}",
        flush=True,
    )
    t_total = time.monotonic()
    results: list[tuple[str, float, int, str]] = []
    if PARALLEL <= 1:
        for q in QUERIES:
            r = hit(q)
            results.append(r)
            print(f"  {r[0]!r:50}  {r[2]:3d} res  {r[1]:6.0f} ms  {r[3]}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            for r in pool.map(hit, QUERIES):
                results.append(r)
                print(
                    f"  {r[0]!r:50}  {r[2]:3d} res  {r[1]:6.0f} ms  {r[3]}",
                    flush=True,
                )
    total_ms = (time.monotonic() - t_total) * 1000
    oks = [r for r in results if r[3] == "ok"]
    fails = [r for r in results if r[3] != "ok"]
    times = sorted(r[1] for r in oks)
    p50 = times[len(times) // 2] if times else 0
    p95 = times[int(len(times) * 0.95)] if times else 0
    print(
        f"\n[prewarm] done in {total_ms / 1000:.1f}s  "
        f"ok={len(oks)}  fail={len(fails)}  "
        f"p50={p50:.0f} ms  p95={p95:.0f} ms",
        flush=True,
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
