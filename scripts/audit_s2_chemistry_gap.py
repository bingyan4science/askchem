#!/usr/bin/env python3
"""
Semantic Scholar Chemistry-Wide Coverage Audit for AskChem.

Uses the S2 bulk search API to enumerate all chemistry papers by year,
compares DOIs against the local chemtree.db, and produces a gap summary.

Resumable: checkpoints after each year so interrupted runs can continue.

Usage:
    python scripts/audit_s2_chemistry_gap.py                # Full run (2015-2026)
    python scripts/audit_s2_chemistry_gap.py --years 2020-2026
    python scripts/audit_s2_chemistry_gap.py --resume       # Resume interrupted run
    python scripts/audit_s2_chemistry_gap.py --status       # Show progress
    python scripts/audit_s2_chemistry_gap.py --summarize    # Print completed summary
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "chemtree.db"
AUDIT_DIR = REPO_ROOT / "data" / "s2_audit"
CHECKPOINT_FILE = AUDIT_DIR / "checkpoint.json"
MISSING_FILE = AUDIT_DIR / "missing_dois.jsonl"
SUMMARY_FILE = AUDIT_DIR / "gap_summary.json"

S2_API_KEY = os.environ.get("S2_API_KEY", "")
S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "externalIds,year,citationCount,isOpenAccess,s2FieldsOfStudy,venue"

CHEMISTRY_FOS = ["Chemistry", "Materials Science", "Chemical Engineering"]
REQUEST_DELAY = 1.05


def s2_headers() -> dict:
    h = {}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


def get_askchem_dois() -> set[str]:
    if not DB_PATH.exists():
        print(f"  WARNING: {DB_PATH} not found", flush=True)
        return set()
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT doi FROM sources").fetchall()
    conn.close()
    return {r[0].strip().lower() for r in rows if r[0]}


def citation_bucket(count: int | None) -> str:
    if count is None:
        return "unknown"
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    if count <= 100:
        return "21-100"
    if count <= 500:
        return "101-500"
    return "500+"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return new_checkpoint()


def new_checkpoint() -> dict:
    return {
        "completed_years": {},
        "in_progress_year": None,
        "in_progress_token": None,
        "in_progress_page": 0,
        "totals": {
            "s2_chemistry": 0,
            "s2_chemistry_with_doi": 0,
            "in_askchem": 0,
            "missing": 0,
            "missing_oa": 0,
        },
        "year_dist_missing": {},
        "fos_dist_missing": {},
        "cite_dist_missing": {},
        "year_dist_all": {},
        "fos_dist_all": {},
    }


def save_checkpoint(ckpt: dict):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, indent=2))
    tmp.rename(CHECKPOINT_FILE)


def fetch_year(year: int, fos: str, askchem_dois: set[str],
               ckpt: dict, missing_fh, top_missing: list,
               resume_token: str | None = None, resume_page: int = 0):
    """Paginate through all papers for a given year + fieldsOfStudy."""
    params: dict = {
        "query": "",
        "fieldsOfStudy": fos,
        "fields": S2_FIELDS,
        "year": f"{year}-{year}",
    }
    if resume_token:
        params["token"] = resume_token

    page = resume_page
    year_total = 0
    year_doi = 0
    year_in_ac = 0
    year_missing = 0
    year_missing_oa = 0

    yr_key = str(year)
    fos_counter = Counter()
    cite_counter = Counter()

    while True:
        try:
            resp = requests.get(S2_BULK_URL, params=params, headers=s2_headers(), timeout=60)
        except Exception as e:
            print(f"    Network error page {page}: {e}, retrying in 5s", flush=True)
            time.sleep(5)
            continue

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            print(f"    Rate limited, waiting {retry_after}s", flush=True)
            time.sleep(retry_after)
            continue

        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
            time.sleep(3)
            continue

        data = resp.json()
        papers = data.get("data", [])
        if not papers:
            break

        for p in papers:
            year_total += 1
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI")

            fos_cats = [f.get("category", "") for f in (p.get("s2FieldsOfStudy") or [])]
            for c in fos_cats:
                ckpt["fos_dist_all"][c] = ckpt["fos_dist_all"].get(c, 0) + 1

            if not doi:
                continue

            year_doi += 1
            doi_lower = doi.strip().lower()

            if doi_lower in askchem_dois:
                year_in_ac += 1
            else:
                year_missing += 1
                cites = p.get("citationCount")
                is_oa = p.get("isOpenAccess", False)
                venue = p.get("venue", "")

                if is_oa:
                    year_missing_oa += 1

                cb = citation_bucket(cites)
                cite_counter[cb] += 1
                for c in fos_cats:
                    fos_counter[c] += 1

                entry = {
                    "doi": doi,
                    "year": year,
                    "citations": cites,
                    "oa": is_oa,
                    "fos": fos_cats,
                    "venue": venue,
                }
                missing_fh.write(json.dumps(entry) + "\n")

                if cites and cites >= 100 and len(top_missing) < 2000:
                    top_missing.append(entry)

        page += 1
        token = data.get("token")

        if page % 50 == 0:
            print(f"    page {page}: {year_total:,} papers, "
                  f"{year_missing:,} missing", flush=True)
            ckpt["in_progress_token"] = token
            ckpt["in_progress_page"] = page
            save_checkpoint(ckpt)
            missing_fh.flush()

        if not token:
            break

        params["token"] = token
        time.sleep(REQUEST_DELAY)

    ckpt["totals"]["s2_chemistry"] += year_total
    ckpt["totals"]["s2_chemistry_with_doi"] += year_doi
    ckpt["totals"]["in_askchem"] += year_in_ac
    ckpt["totals"]["missing"] += year_missing
    ckpt["totals"]["missing_oa"] += year_missing_oa

    ckpt["year_dist_missing"][yr_key] = ckpt["year_dist_missing"].get(yr_key, 0) + year_missing
    ckpt["year_dist_all"][yr_key] = ckpt["year_dist_all"].get(yr_key, 0) + year_total
    for k, v in fos_counter.items():
        ckpt["fos_dist_missing"][k] = ckpt["fos_dist_missing"].get(k, 0) + v
    for k, v in cite_counter.items():
        ckpt["cite_dist_missing"][k] = ckpt["cite_dist_missing"].get(k, 0) + v

    print(f"  {fos} {year}: {year_total:,} total, {year_doi:,} DOI, "
          f"{year_in_ac:,} in AskChem, {year_missing:,} missing "
          f"({page} pages)", flush=True)


def run_audit(year_start: int, year_end: int, resume: bool = False):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AskChem DOI inventory...", flush=True)
    askchem_dois = get_askchem_dois()
    print(f"  {len(askchem_dois):,} DOIs in AskChem", flush=True)

    ckpt = load_checkpoint() if resume else new_checkpoint()

    if not resume and MISSING_FILE.exists():
        MISSING_FILE.unlink()

    years = list(range(year_end, year_start - 1, -1))
    top_missing: list[dict] = []

    mode = "a" if resume else "w"
    with open(MISSING_FILE, mode) as mf:
        for year in years:
            for fos in CHEMISTRY_FOS:
                year_fos_key = f"{year}_{fos}"

                if year_fos_key in ckpt.get("completed_years", {}):
                    print(f"  {fos} {year}: already done, skipping", flush=True)
                    continue

                resume_token = None
                resume_page = 0
                if (resume and ckpt.get("in_progress_year") == year_fos_key):
                    resume_token = ckpt.get("in_progress_token")
                    resume_page = ckpt.get("in_progress_page", 0)
                    print(f"  Resuming {fos} {year} from page {resume_page}",
                          flush=True)

                ckpt["in_progress_year"] = year_fos_key
                ckpt["in_progress_token"] = None
                ckpt["in_progress_page"] = 0
                save_checkpoint(ckpt)

                try:
                    fetch_year(year, fos, askchem_dois, ckpt, mf,
                               top_missing, resume_token, resume_page)
                except Exception as e:
                    print(f"  ERROR {fos} {year}: {e}", flush=True)
                    save_checkpoint(ckpt)
                    continue

                ckpt["completed_years"][year_fos_key] = True
                ckpt["in_progress_year"] = None
                ckpt["in_progress_token"] = None
                save_checkpoint(ckpt)

            cov = 100 * ckpt["totals"]["in_askchem"] / max(ckpt["totals"]["s2_chemistry_with_doi"], 1)
            print(f"\n  Year {year} done. Running: "
                  f"{ckpt['totals']['s2_chemistry']:,} S2 chem | "
                  f"{ckpt['totals']['in_askchem']:,} in AskChem | "
                  f"{ckpt['totals']['missing']:,} missing | "
                  f"coverage {cov:.2f}%\n", flush=True)

    top_missing.sort(key=lambda x: -(x.get("citations") or 0))

    t = ckpt["totals"]
    cov = round(100 * t["in_askchem"] / max(t["s2_chemistry_with_doi"], 1), 3)

    summary = {
        "year_range": f"{year_start}-{year_end}",
        "askchem_dois": len(askchem_dois),
        "s2_chemistry_papers": t["s2_chemistry"],
        "s2_chemistry_with_doi": t["s2_chemistry_with_doi"],
        "in_askchem": t["in_askchem"],
        "missing_from_askchem": t["missing"],
        "coverage_pct": cov,
        "missing_oa_papers": t["missing_oa"],
        "year_distribution_missing": dict(sorted(ckpt["year_dist_missing"].items())),
        "fos_distribution_missing": dict(sorted(
            ckpt["fos_dist_missing"].items(), key=lambda x: -x[1]
        )),
        "citation_buckets_missing": ckpt["cite_dist_missing"],
        "year_distribution_all": dict(sorted(ckpt["year_dist_all"].items())),
        "fos_distribution_all": dict(sorted(
            ckpt["fos_dist_all"].items(), key=lambda x: -x[1]
        )),
        "top_missing_high_citation": top_missing[:500],
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}", flush=True)
    print("AUDIT COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Year range:              {year_start}-{year_end}", flush=True)
    print(f"S2 chemistry papers:     {t['s2_chemistry']:>12,}", flush=True)
    print(f"  with DOI:              {t['s2_chemistry_with_doi']:>12,}", flush=True)
    print(f"In AskChem:              {t['in_askchem']:>12,}", flush=True)
    print(f"Missing from AskChem:    {t['missing']:>12,}", flush=True)
    print(f"Coverage:                {cov:>11.2f}%", flush=True)
    print(f"Missing OA papers:       {t['missing_oa']:>12,}", flush=True)
    print(f"\nSummary: {SUMMARY_FILE}", flush=True)
    print(f"Missing: {MISSING_FILE}", flush=True)


def show_status():
    if not CHECKPOINT_FILE.exists():
        print("No audit in progress.")
        return
    ckpt = json.loads(CHECKPOINT_FILE.read_text())
    t = ckpt["totals"]
    done = len(ckpt.get("completed_years", {}))
    cov = 100 * t["in_askchem"] / max(t["s2_chemistry_with_doi"], 1)
    print(f"Year-FoS combos done: {done}")
    print(f"In progress:     {ckpt.get('in_progress_year', 'none')}")
    print(f"S2 chem papers:  {t['s2_chemistry']:,}")
    print(f"  with DOI:      {t['s2_chemistry_with_doi']:,}")
    print(f"In AskChem:      {t['in_askchem']:,}")
    print(f"Missing:         {t['missing']:,}")
    print(f"Coverage:        {cov:.2f}%")


def show_summary():
    if not SUMMARY_FILE.exists():
        print("No summary yet. Run the audit first.")
        return
    s = json.loads(SUMMARY_FILE.read_text())
    for k, v in s.items():
        if k == "top_missing_high_citation":
            print(f"{k}: {len(v)} entries (showing top 10)")
            for item in v[:10]:
                print(f"  {item['doi']} | {item.get('citations',0)} cites | "
                      f"{item.get('year','')} | {item.get('venue','')}")
        elif isinstance(v, dict) and len(v) > 10:
            print(f"{k}: ({len(v)} entries)")
            for sk, sv in list(v.items())[:10]:
                print(f"  {sk}: {sv:,}" if isinstance(sv, int) else f"  {sk}: {sv}")
            print("  ...")
        else:
            print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="S2 Chemistry Coverage Audit")
    parser.add_argument("--years", default="2015-2026",
                        help="Year range, e.g. 2015-2026")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.summarize:
        show_summary()
    else:
        parts = args.years.split("-")
        year_start, year_end = int(parts[0]), int(parts[1])
        run_audit(year_start, year_end, resume=args.resume)


if __name__ == "__main__":
    main()
