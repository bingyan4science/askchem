"""Validate DOIs against CrossRef API and populate paper_validations table."""
import sqlite3
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"
# CrossRef's polite pool (with mailto) tolerates ~50 req/s comfortably; we
# stay well under that with 8 workers and ~0.2s inter-batch breathing room.
# If we hit 429s, the per-request retry loop with exponential backoff handles it.
BATCH_SIZE = 200
MAX_WORKERS = 8
INTER_BATCH_SLEEP = 0.2
POLITE_EMAIL = "askchem@askchem.org"

def check_doi(doi: str) -> dict:
    """Check a single DOI against CrossRef with retry on rate limit."""
    result = {
        "doi": doi,
        "crossref_verified": 0,
        "has_abstract": 0,
        "is_retracted": 0,
        "journal": "",
        "publisher": "",
        "is_chemistry": 1,
        "validation_score": 0.0,
    }
    url = f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='')}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": f"AskChem/1.0 (mailto:{POLITE_EMAIL})",
                "Accept": "application/json",
            })
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
            msg = data.get("message", {})

            result["crossref_verified"] = 1
            result["has_abstract"] = 1 if msg.get("abstract") else 0
            result["journal"] = (msg.get("container-title") or [""])[0][:200] if msg.get("container-title") else ""
            result["publisher"] = (msg.get("publisher") or "")[:200]

            if "retraction" in str(msg.get("update-to", "")).lower():
                result["is_retracted"] = 1

            score = 0.5
            if result["has_abstract"]:
                score += 0.2
            if result["journal"]:
                score += 0.2
            if result["publisher"]:
                score += 0.1
            result["validation_score"] = round(score, 2)
            return result

        except urllib.error.HTTPError as e:
            if e.code == 404:
                result["crossref_verified"] = 0
                result["validation_score"] = 0.0
                return result
            elif e.code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            else:
                result["crossref_verified"] = -1
                return result
        except Exception:
            time.sleep(1)
            continue

    result["crossref_verified"] = -1
    return result


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    already = set()
    for r in conn.execute("SELECT doi FROM paper_validations").fetchall():
        already.add(r["doi"])

    all_dois = [r["doi"] for r in conn.execute(
        "SELECT doi FROM sources WHERE doi IS NOT NULL AND doi != ''"
    ).fetchall()]
    pending = [d for d in all_dois if d not in already]

    print(f"Total DOIs: {len(all_dois)}, already validated: {len(already)}, pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        conn.close()
        return

    validated = 0
    verified = 0
    errors = 0
    start = time.time()

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(check_doi, doi): doi for doi in batch}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                    if r["crossref_verified"] == 1:
                        verified += 1
                    elif r["crossref_verified"] == -1:
                        errors += 1
                except Exception:
                    errors += 1

        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT OR REPLACE INTO paper_validations "
            "(doi, crossref_verified, has_abstract, is_retracted, journal, publisher, "
            "is_chemistry, validation_score, validated_at, validation_data) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(r["doi"], r["crossref_verified"], r["has_abstract"], r["is_retracted"],
              r["journal"], r["publisher"], r["is_chemistry"], r["validation_score"],
              now, json.dumps(r))
             for r in results],
        )
        conn.commit()
        validated += len(results)

        elapsed = time.time() - start
        rate = validated / elapsed if elapsed > 0 else 0
        pct = validated / len(pending) * 100
        print(f"  {validated}/{len(pending)} ({pct:.1f}%) | verified={verified} errors={errors} | {rate:.1f} DOIs/s",
              flush=True)
        time.sleep(INTER_BATCH_SLEEP)

    total_v = conn.execute("SELECT COUNT(*) FROM paper_validations WHERE crossref_verified = 1").fetchone()[0]
    total_all = conn.execute("SELECT COUNT(*) FROM paper_validations").fetchone()[0]
    print(f"\nDone! {total_all} total validations, {total_v} CrossRef verified ({total_v/total_all*100:.1f}%)")
    conn.close()


if __name__ == "__main__":
    main()
