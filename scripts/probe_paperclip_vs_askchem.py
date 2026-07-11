#!/usr/bin/env python3
"""Side-by-side probe of AskChem /api/search vs Paperclip hybrid search.

For each probe question we pick the first AskChem-rewriter sub-query
(stored in the bench JSON) and run it against both retrieval backends
with top-10. Output is a JSON file with per-query top-10 DOIs, paper
years, and overlap. Used by ``data/eval/paperclip_search_2026-05-20.md``.

Usage::

    PAPERCLIP=... python3 scripts/probe_paperclip_vs_askchem.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
ASKCHEM_API = os.environ.get("ASKCHEM_API", "https://askchem.org/api").rstrip("/")
PAPERCLIP_PYTHON = os.environ.get("PAPERCLIP_PYTHON", str(REPO_ROOT / ".venv-bench" / "bin" / "python3.14"))
PAPERCLIP_SCRIPT = REPO_ROOT / "scripts" / "paperclip_bench_client.py"

PROBES = [
    ("ca02", "CA", "Suzuki-Miyaura cross-coupling aryl chlorides palladium catalyst yield"),
    ("ca04", "CA", "CO2 electroreduction CO Au Ag catalysts Faradaic efficiency"),
    ("tc05", "TC", "TiO2 photocatalysis mechanism electron hole hydroxyl radical"),
    ("tc06", "TC", "CO2 electrochemical reduction mechanism copper selectivity"),
    ("cs02", "CS", "DFT functional benchmark thermochemistry accuracy"),
    ("cs10", "CS", "nanoparticle size effect catalytic activity structure sensitive"),
]


def search_askchem(query: str, limit: int = 10) -> list[dict]:
    r = requests.get(f"{ASKCHEM_API}/search", params={"q": query, "limit": limit}, timeout=60)
    r.raise_for_status()
    return r.json().get("results") or []


def search_paperclip(query: str, limit: int = 10) -> list[dict]:
    payload = json.dumps({
        "fn": "search_with_flags",
        "kwargs": {"query": query, "limit": limit, "ranking": "hybrid",
                   "source": "pmc,arxiv", "timeout": 60},
    })
    env = os.environ.copy()
    if env.get("PAPERCLIP") and not env.get("PAPERCLIP_API_KEY"):
        env["PAPERCLIP_API_KEY"] = env["PAPERCLIP"]
    proc = subprocess.run(
        [PAPERCLIP_PYTHON, str(PAPERCLIP_SCRIPT), "rpc", payload],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout).get("papers") or []


def _normalize_doi(s: str) -> str:
    return (s or "").lower().strip().lstrip("/")


def run_probe(qid: str, task: str, query: str) -> dict:
    started = time.time()
    rec: dict = {"qid": qid, "task": task, "query": query}

    try:
        ac = search_askchem(query)
        rec["askchem"] = [
            {
                "doi": _normalize_doi(c.get("source_doi") or ""),
                "title": (c.get("source_paper_title") or "")[:140],
                "year": c.get("source_year"),
                "verbatim": (c.get("verbatim_quote") or "")[:160],
                "claim_type": c.get("claim_type"),
            }
            for c in ac[:10]
        ]
    except Exception as exc:
        rec["askchem"] = []
        rec["askchem_error"] = str(exc)[:200]

    try:
        pc = search_paperclip(query)
        rec["paperclip"] = [
            {
                "doi": _normalize_doi(p.get("source_doi") or ""),
                "title": (p.get("title") or "")[:140],
                "year": p.get("source_year"),
                "venue": (p.get("source_venue") or "")[:60],
            }
            for p in pc[:10]
        ]
    except Exception as exc:
        rec["paperclip"] = []
        rec["paperclip_error"] = str(exc)[:200]

    ac_dois = {r["doi"] for r in rec.get("askchem", []) if r["doi"]}
    pc_dois = {r["doi"] for r in rec.get("paperclip", []) if r["doi"]}
    rec["overlap_dois"] = sorted(ac_dois & pc_dois)
    rec["overlap_n"] = len(rec["overlap_dois"])
    rec["askchem_unique_n"] = len(ac_dois - pc_dois)
    rec["paperclip_unique_n"] = len(pc_dois - ac_dois)
    rec["elapsed_s"] = round(time.time() - started, 2)
    return rec


def main() -> int:
    out_path = REPO_ROOT / "data" / "eval" / "paperclip_probe_2026-05-20.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(run_probe, qid, task, query): qid for (qid, task, query) in PROBES}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            print(f"  [{rec['qid']}] {rec['task']} ovr={rec['overlap_n']}/{len(rec.get('askchem',[]))}+{len(rec.get('paperclip',[]))} "
                  f"({rec['elapsed_s']}s)")

    results.sort(key=lambda r: r["qid"])
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path} ({len(results)} probes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
