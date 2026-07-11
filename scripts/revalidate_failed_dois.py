"""Re-validate the DOIs that the CrossRef pass marked as not-found.

Two systematic causes were found in the initial sweep:

1. arXiv DOIs (`10.48550/arXiv.*`) are registered with DataCite, not
   CrossRef. We hit api.datacite.org for those and accept any "findable"
   record as verified.

2. A small tail of malformed/dirty DOIs from PDF extraction (trailing
   `.`, concatenated with `;`, `R1` revision suffix, query strings, etc.).
   We apply a few cheap cleanup heuristics and retry CrossRef.

Idempotent: only touches paper_validations rows where crossref_verified=0.
Re-running is safe and only acts on rows that are still failing.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemtree.db"
POLITE_EMAIL = "askchem@askchem.org"

CROSSREF_URL = "https://api.crossref.org/works/{enc}"
DATACITE_URL = "https://api.datacite.org/dois/{enc}"


def _http_json(url: str, headers: dict | None = None) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.getcode(), json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return -1, None


def _datacite_check(doi: str) -> dict | None:
    """Returns a validation summary if the arXiv DOI is registered with DataCite."""
    enc = urllib.request.quote(doi, safe="")
    code, payload = _http_json(DATACITE_URL.format(enc=enc))
    if code != 200 or not payload:
        return None
    attrs = (payload.get("data") or {}).get("attributes") or {}
    if attrs.get("state") != "findable":
        return None
    titles = attrs.get("titles") or [{}]
    title = titles[0].get("title", "") if titles else ""
    publisher = attrs.get("publisher") or ""
    score = 0.5 + (0.2 if title else 0) + (0.2 if publisher else 0)
    return {
        "crossref_verified": 1,  # we treat datacite-findable as "verified"
        "has_abstract": 1 if attrs.get("descriptions") else 0,
        "is_retracted": 0,
        "journal": (publisher or "")[:200],
        "publisher": "DataCite/arXiv",
        "is_chemistry": 1,
        "validation_score": round(score, 2),
        "_source": "datacite",
    }


def _crossref_check(doi: str) -> dict | None:
    enc = urllib.request.quote(doi, safe="")
    headers = {
        "User-Agent": f"AskChem/1.0 (mailto:{POLITE_EMAIL})",
        "Accept": "application/json",
    }
    for attempt in range(3):
        code, payload = _http_json(CROSSREF_URL.format(enc=enc), headers=headers)
        if code == 200 and payload:
            msg = payload.get("message") or {}
            if not msg:
                return None
            journal = ((msg.get("container-title") or [""])[0]) or ""
            publisher = msg.get("publisher") or ""
            has_abstract = 1 if msg.get("abstract") else 0
            is_retracted = 1 if "retraction" in str(msg.get("update-to", "")).lower() else 0
            score = 0.5 + (0.2 if has_abstract else 0) + (0.2 if journal else 0) + (0.1 if publisher else 0)
            return {
                "crossref_verified": 1,
                "has_abstract": has_abstract,
                "is_retracted": is_retracted,
                "journal": journal[:200],
                "publisher": str(publisher)[:200],
                "is_chemistry": 1,
                "validation_score": round(score, 2),
                "_source": "crossref",
            }
        if code == 429:
            time.sleep(2 ** attempt + 1)
            continue
        return None
    return None


_TRAILING_GARBAGE = re.compile(
    r"(?i)("
    r"\.$|"               # trailing period
    r"R\d+$|"             # revision suffix R1, R2
    r"/pdf$|"             # /pdf URL fragment
    r"/abstract$|"        # /abstract URL fragment
    r"/full$|"            # /full URL fragment
    r"\?.*$"              # query string
    r")"
)


def _cleanup_candidates(doi: str) -> list[str]:
    """Generate a few alternative spellings to try for a malformed DOI."""
    cands: list[str] = []
    seen: set[str] = set()

    def add(c: str) -> None:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            cands.append(c)

    # Strip trailing garbage iteratively
    s = doi
    for _ in range(3):
        new = _TRAILING_GARBAGE.sub("", s).strip()
        if new == s:
            break
        s = new
    add(s)

    # If concatenated with ; or , take each half
    for sep in (";", ","):
        if sep in doi:
            for piece in doi.split(sep):
                add(piece)

    # Strip /full/<digits>/<digits> path fragment (commonly seen)
    if "/" in doi:
        parts = doi.split("/")
        # keep only the first 2 path segments after the prefix
        if len(parts) > 2:
            add("/".join(parts[:2]))

    return cands


def _is_arxiv(doi: str) -> bool:
    return doi.lower().startswith("10.48550/arxiv")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    failed = [
        r["doi"] for r in conn.execute(
            "SELECT doi FROM paper_validations WHERE crossref_verified=0"
        )
    ]
    print(f"failed DOIs to re-check: {len(failed):,}")
    if not failed:
        return

    arxiv = [d for d in failed if _is_arxiv(d)]
    other = [d for d in failed if not _is_arxiv(d)]
    print(f"  arXiv (DataCite path): {len(arxiv):,}")
    print(f"  other (CrossRef cleanup path): {len(other):,}")

    fixed_arxiv = 0
    fixed_other = 0
    still_404 = 0
    started = time.time()

    def _commit(rows: list[tuple]) -> None:
        if not rows:
            return
        conn.executemany(
            "INSERT OR REPLACE INTO paper_validations "
            "(doi, crossref_verified, has_abstract, is_retracted, journal, publisher, "
            "is_chemistry, validation_score, validated_at, validation_data) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()

    pending_rows: list[tuple] = []
    BATCH = 200

    for i, doi in enumerate(arxiv, 1):
        summary = _datacite_check(doi)
        if summary:
            fixed_arxiv += 1
            now = datetime.now(timezone.utc).isoformat()
            pending_rows.append((
                doi,
                summary["crossref_verified"],
                summary["has_abstract"],
                summary["is_retracted"],
                summary["journal"],
                summary["publisher"],
                summary["is_chemistry"],
                summary["validation_score"],
                now,
                json.dumps(summary),
            ))
        else:
            still_404 += 1
        if len(pending_rows) >= BATCH:
            _commit(pending_rows)
            pending_rows.clear()
        if i % 200 == 0 or i == len(arxiv):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            print(f"  arxiv: {i:>5,}/{len(arxiv):,} | fixed={fixed_arxiv:,} "
                  f"still_404={still_404:,} | {rate:.1f}/s", flush=True)
        time.sleep(0.05)  # gentle on DataCite

    _commit(pending_rows)
    pending_rows.clear()

    print()
    started2 = time.time()
    for i, doi in enumerate(other, 1):
        summary = None
        for cand in _cleanup_candidates(doi):
            summary = _crossref_check(cand)
            if summary:
                summary["_cleaned_to"] = cand
                break
        if summary:
            fixed_other += 1
            now = datetime.now(timezone.utc).isoformat()
            pending_rows.append((
                doi,  # keep original DOI as the row key; record what worked in JSON
                summary["crossref_verified"],
                summary["has_abstract"],
                summary["is_retracted"],
                summary["journal"],
                summary["publisher"],
                summary["is_chemistry"],
                summary["validation_score"],
                now,
                json.dumps(summary),
            ))
        else:
            still_404 += 1
        if len(pending_rows) >= 50:
            _commit(pending_rows)
            pending_rows.clear()
        if i % 25 == 0 or i == len(other):
            elapsed = time.time() - started2
            rate = i / elapsed if elapsed else 0
            print(f"  other: {i:>5,}/{len(other):,} | fixed={fixed_other:,} "
                  f"| {rate:.1f}/s", flush=True)
        time.sleep(0.1)

    _commit(pending_rows)

    print()
    print(f"DONE in {time.time()-started:.1f}s")
    print(f"  arxiv DOIs verified via DataCite: {fixed_arxiv:,}")
    print(f"  other DOIs recovered via cleanup: {fixed_other:,}")
    print(f"  remaining truly missing         : {still_404:,}")

    total_v = conn.execute(
        "SELECT COUNT(*) FROM paper_validations WHERE crossref_verified=1"
    ).fetchone()[0]
    total_all = conn.execute("SELECT COUNT(*) FROM paper_validations").fetchone()[0]
    rate = total_v / total_all * 100 if total_all else 0
    print(f"\n  paper_validations: {total_all:,} rows, {total_v:,} verified ({rate:.1f}%)")
    conn.close()


if __name__ == "__main__":
    main()
