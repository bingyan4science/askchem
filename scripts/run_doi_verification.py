#!/usr/bin/env python3
"""Verify source DOIs against CrossRef and store results in paper_validations."""

import argparse
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


CHEM_KEYWORDS = {
    "chemistry", "chemical", "catalysis", "polymer", "materials",
    "electrochemistry", "biochemistry", "organic", "inorganic",
    "analytical", "physical chemistry", "pharmaceutical", "medicinal",
    "nanoscience", "nanotechnology",
}

_TLS = threading.local()
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def session() -> requests.Session:
    if not hasattr(_TLS, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "AskChem/1.0 (mailto:admin@askchem.org)"})
        _TLS.session = s
    return _TLS.session


def acquire_request_slot(min_interval: float) -> None:
    global _NEXT_REQUEST_AT
    with _RATE_LOCK:
        now = time.time()
        wait = max(0.0, _NEXT_REQUEST_AT - now)
        if wait:
            time.sleep(wait)
        _NEXT_REQUEST_AT = max(_NEXT_REQUEST_AT, time.time()) + min_interval


def validate_doi(doi: str, min_interval: float) -> dict:
    val = {
        "doi_format_valid": False,
        "crossref_verified": False,
        "has_abstract": False,
        "is_retracted": False,
        "journal": "",
        "publisher": "",
        "is_chemistry": True,
        "validation_score": 0.0,
        "issues": [],
    }

    if not re.match(r"^10\.\d{4,}/", doi):
        val["issues"].append("Invalid DOI format")
        return val
    val["doi_format_valid"] = True

    for attempt in range(6):
        try:
            acquire_request_slot(min_interval)
            resp = session().get(
                f"https://api.crossref.org/works/{doi}",
                params={"mailto": "admin@askchem.org"},
                timeout=20,
            )
            if resp.status_code == 200:
                msg = resp.json().get("message", {})
                val["crossref_verified"] = True
                val["journal"] = (msg.get("container-title") or [""])[0]
                val["publisher"] = msg.get("publisher", "")
                val["has_abstract"] = bool(msg.get("abstract"))

                for upd in msg.get("update-to") or []:
                    if upd.get("type") == "retraction":
                        val["is_retracted"] = True
                        val["issues"].append("Paper has been RETRACTED")

                subjects = msg.get("subject") or []
                title_str = " ".join(msg.get("title") or []).lower()
                subjects_lower = " ".join(subjects).lower()
                if subjects and not any(k in subjects_lower for k in CHEM_KEYWORDS):
                    if not any(k in title_str for k in CHEM_KEYWORDS):
                        val["is_chemistry"] = False
                        val["issues"].append(f"Paper may not be chemistry-related (subjects: {subjects})")
            elif resp.status_code == 429:
                if attempt < 5:
                    time.sleep(min(10 * (attempt + 1), 60))
                    continue
                val["issues"].append("CrossRef lookup failed (HTTP 429)")
            else:
                val["issues"].append(f"CrossRef lookup failed (HTTP {resp.status_code})")
            break
        except Exception as e:
            if attempt == 5:
                val["issues"].append(f"CrossRef check error: {str(e)[:100]}")
            else:
                time.sleep(min(2 ** attempt, 30))

    score = 0.0
    if val["doi_format_valid"]:
        score += 0.2
    if val["crossref_verified"]:
        score += 0.3
    if val["is_chemistry"]:
        score += 0.3
    if not val["is_retracted"]:
        score += 0.2
    val["validation_score"] = round(score, 2)
    return val


def save_batch(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    if not batch:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO paper_validations "
        "(doi, crossref_verified, has_abstract, is_retracted, journal, publisher, "
        "is_chemistry, validation_score, validated_at, validation_data) "
        "VALUES (?,?,?,?,?,?,?,?,datetime('now'),?)",
        batch,
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="chemtree.db")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--min-interval", type=float, default=0.2)
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    if args.refresh:
        query = "SELECT doi FROM sources WHERE doi != '' ORDER BY doi"
    else:
        query = (
            "SELECT s.doi FROM sources s "
            "LEFT JOIN paper_validations pv ON pv.doi = s.doi "
            "WHERE s.doi != '' AND pv.doi IS NULL "
            "ORDER BY s.doi"
        )

    dois = [row[0] for row in conn.execute(query)]
    if args.limit > 0:
        dois = dois[:args.limit]
    print(f"DOIs to verify: {len(dois):,}", flush=True)
    if not dois:
        return

    saved = []
    done = 0
    verified = 0
    retracted = 0
    chemistry = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_doi, doi, args.min_interval): doi for doi in dois}
        for fut in as_completed(futures):
            doi = futures[fut]
            val = fut.result()
            saved.append((
                doi,
                1 if val.get("crossref_verified") else 0,
                1 if val.get("has_abstract") else 0,
                1 if val.get("is_retracted") else 0,
                val.get("journal", ""),
                val.get("publisher", ""),
                1 if val.get("is_chemistry", True) else 0,
                float(val.get("validation_score", 0)),
                json.dumps(val),
            ))
            done += 1
            verified += 1 if val.get("crossref_verified") else 0
            retracted += 1 if val.get("is_retracted") else 0
            chemistry += 1 if val.get("is_chemistry", True) else 0

            if len(saved) >= 200:
                save_batch(conn, saved)
                saved.clear()

            if done % 1000 == 0:
                elapsed = max(time.time() - t0, 1)
                rate = done / elapsed
                print(
                    f"Progress: {done:,}/{len(dois):,}  "
                    f"verified={verified:,}  retracted={retracted:,}  "
                    f"chemistry={chemistry:,}  rate={rate:.1f}/s",
                    flush=True,
                )

    save_batch(conn, saved)
    conn.close()
    elapsed = time.time() - t0
    print(
        f"Done in {elapsed/60:.1f} min: {done:,} processed, "
        f"{verified:,} CrossRef-verified, {retracted:,} retracted",
        flush=True,
    )


if __name__ == "__main__":
    main()
