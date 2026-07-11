"""
Incremental AskChem Index Update.

Discovers new chemistry papers from multiple sources (arXiv, ChemRxiv, journal
RSS feeds, Semantic Scholar), extracts claims (from abstracts or full-text PDFs
when available), classifies them into the 5-view hierarchy, and writes to SQLite.

Sources:
  - arXiv OAI-PMH: chem-ph + materials science daily feed
  - ChemRxiv API: preprints (graceful fallback if blocked)
  - Journal RSS: JACS, Angew. Chem., Nature Chemistry, etc.
  - Semantic Scholar: bulk Chemistry + Materials + ChemEng + Biochem sweep

Modes:
  - Real-time (default): async API calls, results in minutes
  - Batch (--batch): OpenAI Batch API, 50% cheaper, up to 24h

Usage:
    python src/update_index.py                          # All sources, real-time, since last run
    python src/update_index.py --days 7                 # Look back 7 days
    python src/update_index.py --source arxiv           # arXiv only
    python src/update_index.py --source s2 --batch      # S2 sweep via Batch API
    python src/update_index.py --batch --poll            # Check batch status
    python src/update_index.py --batch --collect         # Download results & index
    python src/update_index.py --dois 10.1038/s41586-024-00001-1
    python src/update_index.py --download-pdfs           # Also download OA PDFs
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import requests as http_requests
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).parent))

from askchem.models import Claim
from askchem.llm import get_async_client, MODELS
from askchem.display import smart_title
from askchem.db import (
    get_conn, upsert_source, upsert_claims_batch,
    append_claim_to_node, update_metadata_counts,
    upsert_tree_node, index_authors_for_doi,
)

INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"
DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers_full"
BATCH_DIR = INDEX_DIR / "_batch_update"

S2_BULK_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "paperId,title,abstract,year,citationCount,venue,openAccessPdf,authors,fieldsOfStudy,externalIds"
S2_MIN_DELAY = 1.1

CONCURRENCY = 20
ABSTRACT_MODEL = MODELS["fast"]       # gpt-5-mini for abstracts
DEEP_MODEL = MODELS["strong"]         # gpt-5.4 for full PDFs (fallback)
GEMINI_MODEL = "@vertexai-gemini-kc119-2/gemini-3.1-pro-preview"
CLASSIFICATION_MODEL = MODELS["fast"]

MAX_PDF_BYTES = 50 * 1024 * 1024      # skip PDFs > 50MB
MAX_BATCH_FILE_BYTES = 90 * 1024 * 1024  # 90MB per batch JSONL file

# ── Last-run tracking ────────────────────────────────────────────────────────

LAST_RUN_FILE = INDEX_DIR / "_last_discovery.json"


def _load_last_run() -> dict[str, str]:
    if LAST_RUN_FILE.exists():
        with open(LAST_RUN_FILE) as f:
            return json.load(f)
    return {}


def _save_last_run(data: dict[str, str]):
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_RUN_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_from_date(source: str, days: int) -> str:
    """Get the from-date for a source: last run time or N days ago."""
    last_run = _load_last_run()
    if source in last_run:
        return last_run[source][:10]  # YYYY-MM-DD
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


# ── Discovery helpers ────────────────────────────────────────────────────────

ARXIV_OAI_BASE = "http://export.arxiv.org/oai2"
ARXIV_SETS = ["physics:physics:chem-ph", "physics:cond-mat:mtrl-sci"]

CHEMRXIV_API = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"

JOURNAL_FEEDS = {
    "JACS": "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=jacsat",
    "Angew_Chem": "https://onlinelibrary.wiley.com/action/showFeed?jc=15213773&type=etoc",
    "Nature_Chemistry": "https://www.nature.com/nchem.rss",
    "Nature_Catalysis": "https://www.nature.com/natcatal.rss",
    "ACS_Catalysis": "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=accacs",
    "Chem_Rev": "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=chreay",
    "Chem_Sci": "https://pubs.rsc.org/en/journals/journalissues/sc#!recentarticles&adv",
    "ACS_Nano": "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=ancac3",
}

S2_FIELDS_OF_STUDY = ["Chemistry", "Materials Science", "Chemical Engineering", "Biochemistry"]

# ── CrossRef per-publisher metadata ──────────────────────────────────────────
# Metadata-only discovery for closed-access journals — NO PDF downloads, so
# this is fully compliant with the NYU library policy that flagged the
# Science.org bulk-PDF incident in Apr 2026. We grab DOI + title + abstract
# + venue + year + license here, then route abstract-only papers through the
# Stage-4b Gemini Batch extractor.
CROSSREF_BASE = "https://api.crossref.org/works"
# CrossRef member IDs from https://api.crossref.org/members
CROSSREF_MEMBERS: dict[int, str] = {
    316:   "ACS",               # American Chemical Society
    311:   "Wiley",
    78:    "Elsevier",
    81:    "RSC",               # Royal Society of Chemistry
    297:   "Springer Nature",
    301:   "Taylor & Francis",
    1968:  "Nature Research",
    47168: "ChemRxiv (CrossRef)",  # ChemRxiv records mirrored into CrossRef
}
CROSSREF_ROWS = 200
CROSSREF_MAX_PAGES = 25            # cap per publisher; 25 × 200 = 5000 papers
CROSSREF_MIN_DELAY_S = 1.1         # polite-pool rate
CROSSREF_RETRY_BACKOFF = (15, 30, 60, 120, 240)


def _s2_headers():
    key = os.environ.get("S2_API_KEY", "")
    return {"x-api-key": key} if key else {}


def _crossref_headers() -> dict:
    """Polite-pool User-Agent (CrossRef rate-limits anonymous traffic harder)."""
    contact = os.environ.get("CROSSREF_CONTACT", "bingyan@nyu.edu")
    return {
        "User-Agent": f"AskChem/1.0 (https://askchem.org; mailto:{contact})",
        "Accept": "application/json",
    }


def _strip_jats(text: str) -> str:
    """Strip JATS XML tags that CrossRef sometimes embeds in abstracts."""
    if not text:
        return ""
    out = re.sub(r"<jats:[^>]+>|</jats:[^>]+>", "", text)
    out = re.sub(r"<[^>]+>", "", out)
    return re.sub(r"\s+", " ", out).strip()


def get_existing_dois() -> set[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT doi FROM sources").fetchall()
    return {r['doi'].lower() for r in rows if r['doi']}


# ── arXiv OAI-PMH ────────────────────────────────────────────────────────────

def discover_arxiv(from_date: str) -> list[dict]:
    """Fetch recent chemistry papers from arXiv via OAI-PMH."""
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    all_papers = []

    for arxiv_set in ARXIV_SETS:
        token = None
        page = 0
        while True:
            try:
                if token:
                    params = {"verb": "ListRecords", "resumptionToken": token}
                else:
                    params = {
                        "verb": "ListRecords",
                        "set": arxiv_set,
                        "from": from_date,
                        "metadataPrefix": "oai_dc",
                    }
                resp = http_requests.get(ARXIV_OAI_BASE, params=params,
                                         timeout=60, allow_redirects=True)
                if resp.status_code != 200:
                    print(f"  arXiv {arxiv_set}: HTTP {resp.status_code}", flush=True)
                    break

                root = ET.fromstring(resp.text)
                error = root.find(".//oai:error", ns)
                if error is not None:
                    if page == 0:
                        print(f"  arXiv {arxiv_set}: {error.text}", flush=True)
                    break

                records = root.findall(".//oai:record", ns)
                for record in records:
                    title_el = record.find(".//dc:title", ns)
                    abstract_el = record.find(".//dc:description", ns)
                    identifier = record.find(".//oai:identifier", ns)
                    if title_el is None or identifier is None:
                        continue
                    arxiv_id = identifier.text.replace("oai:arXiv.org:", "")
                    all_papers.append({
                        "source": "arxiv",
                        "arxiv_id": arxiv_id,
                        "title": title_el.text or "",
                        "abstract": abstract_el.text if abstract_el is not None else "",
                        "doi": f"arXiv:{arxiv_id}",
                        "externalIds": {"ArXiv": arxiv_id},
                    })

                page += 1
                rt = root.find(".//oai:resumptionToken", ns)
                if rt is not None and rt.text:
                    token = rt.text
                    time.sleep(1)
                else:
                    break

            except Exception as e:
                print(f"  arXiv {arxiv_set} error: {e}", flush=True)
                break

        print(f"  arXiv [{arxiv_set.split(':')[-1]}]: {len(all_papers)} papers "
              f"(from {from_date})", flush=True)
        time.sleep(1)

    return all_papers


# ── ChemRxiv via CrossRef DOI prefix ─────────────────────────────────────────
# The direct ChemRxiv API at chemrxiv.org sits behind Cloudflare and returns
# a 403 bot-detection challenge from NYU IP ranges (confirmed live, May 20).
# ChemRxiv-hosted preprints are however indexed in CrossRef under DOI prefix
# ``10.26434`` (member 316, ACS — they operate ChemRxiv). Pulling from
# CrossRef gives us DOI + title + abstract directly without ever touching
# chemrxiv.org, which also stays inside NYU library policy.

CHEMRXIV_DOI_PREFIX = "10.26434"


def discover_chemrxiv(from_date: str) -> list[dict]:
    """Fetch recent ChemRxiv preprints via CrossRef (DOI prefix 10.26434).

    Hits ``/prefixes/10.26434/works?filter=from-pub-date:<from_date>``,
    paginates with cursor. Output shape matches ``discover_arxiv`` /
    ``discover_crossref``. ChemRxiv's preprint records carry abstracts
    nearly always (~95% coverage observed), so Stage 4b can extract
    standalone claims even without the PDF.
    """
    all_papers: list[dict] = []
    cursor = "*"
    base = "https://api.crossref.org/prefixes/10.26434/works"
    for page in range(CROSSREF_MAX_PAGES):
        params = {
            "filter": f"from-pub-date:{from_date}",
            "rows": str(CROSSREF_ROWS),
            "cursor": cursor,
            "select": "DOI,title,abstract,published-online,published,"
                      "container-title,author,type",
        }
        attempt = 0
        data = None
        while attempt < len(CROSSREF_RETRY_BACKOFF):
            try:
                time.sleep(CROSSREF_MIN_DELAY_S)
                resp = http_requests.get(
                    base, params=params,
                    headers=_crossref_headers(), timeout=60,
                )
                if resp.status_code in (429, 403, 503):
                    wait = int(resp.headers.get("Retry-After") or
                               CROSSREF_RETRY_BACKOFF[attempt])
                    print(f"  ChemRxiv (via CrossRef) HTTP {resp.status_code}, "
                          f"waiting {wait}s (attempt {attempt+1})...", flush=True)
                    time.sleep(wait)
                    attempt += 1
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                wait = CROSSREF_RETRY_BACKOFF[min(attempt, len(CROSSREF_RETRY_BACKOFF) - 1)]
                print(f"  ChemRxiv error: {exc} -- waiting {wait}s", flush=True)
                time.sleep(wait)
                attempt += 1
        if not data:
            break

        msg = data.get("message", {})
        items = msg.get("items", [])
        if not items:
            break

        for it in items:
            doi = (it.get("DOI") or "").strip()
            if not doi:
                continue
            title = ""
            if isinstance(it.get("title"), list) and it["title"]:
                title = _strip_jats(it["title"][0])
            year = 0
            for k in ("published-online", "published"):
                date_parts = ((it.get(k) or {}).get("date-parts") or [[None]])[0]
                if date_parts and date_parts[0]:
                    year = int(date_parts[0])
                    break
            authors = []
            for a in (it.get("author") or [])[:20]:
                full = " ".join(filter(None, [a.get("given"), a.get("family")]))
                if full:
                    authors.append({"name": full})

            all_papers.append({
                "source": "chemrxiv",
                "doi": doi,
                "title": title,
                "abstract": _strip_jats(it.get("abstract", "")),
                "externalIds": {"DOI": doi},
                "authors": authors,
                "venue": "ChemRxiv",
                "year": year or datetime.now().year,
            })

        next_cursor = msg.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    n_with_abstract = sum(1 for p in all_papers if p.get("abstract"))
    print(f"  ChemRxiv: {len(all_papers)} papers (from {from_date}; "
          f"{n_with_abstract} with abstract)", flush=True)
    return all_papers


# ── Journal RSS feeds ────────────────────────────────────────────────────────

def discover_rss() -> list[dict]:
    """Fetch recent papers from journal RSS/Atom feeds."""
    all_papers = []

    for journal, url in JOURNAL_FEEDS.items():
        try:
            resp = http_requests.get(url, timeout=15, headers={
                "User-Agent": "AskChem/1.0 (academic research)",
            })
            if resp.status_code != 200:
                print(f"  RSS {journal}: HTTP {resp.status_code}", flush=True)
                continue

            root = ET.fromstring(resp.text)
            items = (root.findall(".//item") or
                     root.findall(".//{http://www.w3.org/2005/Atom}entry"))

            count = 0
            for item in items[:30]:
                title_el = (item.find("title") or
                            item.find("{http://www.w3.org/2005/Atom}title"))
                link_el = (item.find("link") or
                           item.find("{http://www.w3.org/2005/Atom}link"))
                doi_el = item.find(
                    "{http://prismstandard.org/namespaces/basic/2.0/}doi")

                title = title_el.text if title_el is not None else ""
                link = ""
                if link_el is not None:
                    link = link_el.text or link_el.get("href", "")
                doi = doi_el.text if doi_el is not None else ""

                if not doi and "doi.org/" in link:
                    doi = link.split("doi.org/", 1)[1]
                if not doi:
                    continue

                all_papers.append({
                    "source": "rss",
                    "journal": journal,
                    "doi": doi,
                    "title": title,
                    "externalIds": {"DOI": doi},
                    "venue": journal.replace("_", " "),
                })
                count += 1

            print(f"  RSS {journal}: {count} papers", flush=True)

        except Exception as e:
            print(f"  RSS {journal} error: {e}", flush=True)

        time.sleep(0.5)

    return all_papers


# ── CrossRef per-publisher metadata discovery ────────────────────────────────

def _crossref_query_one(member_id: int, member_name: str,
                        from_date: str, existing_dois: set[str]) -> list[dict]:
    """Paginate CrossRef for one publisher's papers since ``from_date``.

    Filters to ``type:journal-article`` (no datasets, no books) and
    ``from-pub-date:<from_date>``. Subject filter (Chemistry) is applied
    client-side because CrossRef's ``subject`` filter is unreliable.

    Returns S2-shaped paper dicts:
        {source, doi, title, abstract, year, venue, externalIds, license,
         crossref_member}
    """
    out: list[dict] = []
    cursor = "*"
    filter_str = f"type:journal-article,from-pub-date:{from_date},member:{member_id}"
    for page in range(CROSSREF_MAX_PAGES):
        params = {
            "filter": filter_str,
            "rows": str(CROSSREF_ROWS),
            "cursor": cursor,
            "select": "DOI,title,abstract,published,published-print,published-online,"
                      "container-title,subject,license,author,type",
        }
        attempt = 0
        data = None
        while attempt < len(CROSSREF_RETRY_BACKOFF):
            try:
                time.sleep(CROSSREF_MIN_DELAY_S)
                resp = http_requests.get(
                    CROSSREF_BASE, params=params,
                    headers=_crossref_headers(), timeout=60,
                )
                if resp.status_code in (429, 403, 503):
                    wait = int(resp.headers.get("Retry-After") or
                               CROSSREF_RETRY_BACKOFF[attempt])
                    print(f"  CrossRef[{member_name}] HTTP {resp.status_code}, "
                          f"waiting {wait}s (attempt {attempt+1})...", flush=True)
                    time.sleep(wait)
                    attempt += 1
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                wait = CROSSREF_RETRY_BACKOFF[min(attempt, len(CROSSREF_RETRY_BACKOFF) - 1)]
                print(f"  CrossRef[{member_name}] error: {exc} -- waiting {wait}s",
                      flush=True)
                time.sleep(wait)
                attempt += 1
        if not data:
            break

        msg = data.get("message", {})
        items = msg.get("items", [])
        if not items:
            break

        for it in items:
            doi = (it.get("DOI") or "").strip()
            if not doi:
                continue
            dl = doi.lower()
            if dl in existing_dois:
                continue
            title = ""
            if isinstance(it.get("title"), list) and it["title"]:
                title = _strip_jats(it["title"][0])
            venue = ""
            if isinstance(it.get("container-title"), list) and it["container-title"]:
                venue = _strip_jats(it["container-title"][0])
            abstract = _strip_jats(it.get("abstract", ""))
            # Year: prefer print, fall back to online, then any published.
            year = 0
            for k in ("published-print", "published-online", "published"):
                date_parts = ((it.get(k) or {}).get("date-parts") or [[None]])[0]
                if date_parts and date_parts[0]:
                    year = int(date_parts[0])
                    break
            license_url = ""
            if isinstance(it.get("license"), list) and it["license"]:
                license_url = it["license"][0].get("URL", "")
            authors = []
            for a in (it.get("author") or [])[:20]:
                full = " ".join(filter(None, [a.get("given"), a.get("family")]))
                if full:
                    authors.append({"name": full})

            out.append({
                "source": f"crossref:{member_name.lower().replace(' ', '_')}",
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": year,
                "venue": venue,
                "externalIds": {"DOI": doi},
                "license": license_url,
                "authors": authors,
                "crossref_member": member_id,
            })

        next_cursor = msg.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return out


def discover_crossref(from_date: str, existing_dois: set[str]) -> list[dict]:
    """Discover papers since ``from_date`` from every configured CrossRef
    member (publisher).

    Output is shaped like ``discover_arxiv`` so the rest of the pipeline is
    indifferent. NYU library compliant: metadata-only, no PDF downloads, no
    publisher-site scraping.
    """
    all_papers: list[dict] = []
    for member_id, name in CROSSREF_MEMBERS.items():
        try:
            papers = _crossref_query_one(member_id, name, from_date, existing_dois)
        except Exception as exc:
            print(f"  CrossRef[{name}] fatal: {exc}", flush=True)
            papers = []
        n_with_abstract = sum(1 for p in papers if p.get("abstract"))
        print(f"  CrossRef[{name}]: {len(papers)} new papers "
              f"({n_with_abstract} with abstract)", flush=True)
        all_papers.extend(papers)
    print(f"  CrossRef total: {len(all_papers)} new papers", flush=True)
    return all_papers


# ── Semantic Scholar bulk sweep ──────────────────────────────────────────────

# S2 retry policy: the 2026-05-20 run hit a transient 403 on the first
# request to /bulk and the old flat 10-s retry-loop burned through all 5
# attempts in under a minute. The fix is exponential backoff that also
# treats 403 as transient (the bulk endpoint sometimes 403s under load)
# and respects the ``Retry-After`` header when present.
S2_RETRY_BACKOFF = (15, 30, 60, 120, 240)
S2_PAGES_PER_FIELD = 12


def _s2_get_with_backoff(url: str, params: dict, *, label: str) -> dict | None:
    headers = _s2_headers()
    attempt = 0
    while attempt < len(S2_RETRY_BACKOFF):
        try:
            time.sleep(S2_MIN_DELAY)
            resp = http_requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code in (429, 403, 502, 503):
                wait = int(resp.headers.get("Retry-After") or
                           S2_RETRY_BACKOFF[attempt])
                print(f"  S2 [{label}] HTTP {resp.status_code}, "
                      f"waiting {wait}s (attempt {attempt+1})...", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            resp.raise_for_status()
            return resp.json()
        except http_requests.exceptions.HTTPError as exc:
            wait = S2_RETRY_BACKOFF[min(attempt, len(S2_RETRY_BACKOFF) - 1)]
            print(f"  S2 [{label}] http error: {exc} -- waiting {wait}s", flush=True)
            time.sleep(wait)
            attempt += 1
        except Exception as exc:
            wait = S2_RETRY_BACKOFF[min(attempt, len(S2_RETRY_BACKOFF) - 1)]
            print(f"  S2 [{label}] error: {exc} -- waiting {wait}s", flush=True)
            time.sleep(wait)
            attempt += 1
    return None


def discover_s2(from_date: str, existing_dois: set[str]) -> list[dict]:
    """Bulk-discover new chemistry papers from Semantic Scholar.

    S2's bulk filter has only year-level resolution, so we fetch the full
    year and client-side filter to ``publicationDate >= from_date``. Loop
    over four fields of study (Chemistry / Materials / Chemical Engineering
    / Biochemistry); the four are kept sequential so we never blow the
    polite-pool rate. Retry policy now treats 403/429/5xx as transient
    with exponential backoff (see ``_s2_get_with_backoff``).
    """
    from_year = int(from_date[:4])
    all_papers: list[dict] = []
    seen_dois: set[str] = set()

    for field in S2_FIELDS_OF_STUDY:
        token = None
        pages = 0
        field_count = 0
        kept = 0

        while pages < S2_PAGES_PER_FIELD:
            params = {
                "fields": S2_FIELDS + ",publicationDate",
                "fieldsOfStudy": field,
                "year": f"{from_year}-",
                "minCitationCount": 0,
            }
            if token:
                params["token"] = token

            data = _s2_get_with_backoff(S2_BULK_SEARCH, params, label=field)
            if not data:
                break

            papers = data.get("data", []) or []
            if not papers:
                break
            field_count += len(papers)

            for p in papers:
                doi = (p.get("externalIds") or {}).get("DOI", "")
                if not doi or not p.get("abstract"):
                    continue
                pub_date = p.get("publicationDate") or ""
                if pub_date and pub_date < from_date:
                    # Client-side day-precise filter — S2's bulk filter
                    # only handles year, not day.
                    continue
                dl = doi.lower()
                if dl in existing_dois or dl in seen_dois:
                    continue
                seen_dois.add(dl)
                kept += 1
                all_papers.append(p)

            pages += 1
            token = data.get("token")
            if not token:
                break

        print(f"  S2 [{field}]: scanned {field_count} -> kept {kept} "
              f"new papers ({pages} pages)", flush=True)

    all_papers.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)
    print(f"  S2 total: {len(all_papers)} new papers", flush=True)
    return all_papers


# ── S2 metadata enrichment ───────────────────────────────────────────────────

def enrich_via_s2(papers: list[dict]) -> list[dict]:
    """Enrich papers that lack abstracts by looking up via Semantic Scholar.

    Uses the same exponential-backoff retry policy as ``discover_s2`` —
    treats 403/429/5xx as transient. A 404 just means S2 doesn't have the
    paper, so we pass through unchanged (no retry).
    """
    enriched = []
    looked_up = 0
    n_404 = 0

    for paper in papers:
        if paper.get("abstract"):
            enriched.append(paper)
            continue

        doi = (paper.get("externalIds") or {}).get("DOI", paper.get("doi", ""))
        if not doi:
            enriched.append(paper)
            continue

        lookup_id = doi
        if paper.get("arxiv_id"):
            lookup_id = f"ArXiv:{paper['arxiv_id']}"

        s2_data = None
        attempt = 0
        while attempt < len(S2_RETRY_BACKOFF):
            try:
                time.sleep(S2_MIN_DELAY)
                resp = http_requests.get(
                    f"{S2_PAPER}/{lookup_id}",
                    params={"fields": S2_FIELDS},
                    headers=_s2_headers(), timeout=30,
                )
                if resp.status_code == 404:
                    n_404 += 1
                    break  # paper not in S2; pass through
                if resp.status_code in (429, 403, 502, 503):
                    wait = int(resp.headers.get("Retry-After") or
                               S2_RETRY_BACKOFF[attempt])
                    time.sleep(wait)
                    attempt += 1
                    continue
                if resp.status_code == 200:
                    s2_data = resp.json()
                    break
                # Other non-2xx: give up on this paper
                break
            except Exception:
                wait = S2_RETRY_BACKOFF[min(attempt, len(S2_RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                attempt += 1

        if s2_data:
            real_doi = (s2_data.get("externalIds") or {}).get("DOI", "")
            if real_doi:
                s2_data["_original_doi"] = doi
                enriched.append(s2_data)
                looked_up += 1
            elif s2_data.get("abstract"):
                paper["abstract"] = s2_data["abstract"]
                paper["year"] = s2_data.get("year") or paper.get("year")
                paper["citationCount"] = s2_data.get("citationCount", 0)
                paper["venue"] = s2_data.get("venue") or paper.get("venue", "")
                paper["authors"] = s2_data.get("authors") or paper.get("authors", [])
                if not paper.get("externalIds", {}).get("DOI"):
                    paper["externalIds"] = s2_data.get("externalIds") or paper.get("externalIds", {})
                enriched.append(paper)
                looked_up += 1
            else:
                enriched.append(paper)
        else:
            enriched.append(paper)

    if looked_up or n_404:
        print(f"  Enriched {looked_up} papers via S2 metadata lookup "
              f"({n_404} not in S2)", flush=True)
    return enriched


def fetch_papers_by_doi(dois: list[str]) -> list[dict]:
    """Fetch specific papers by DOI from Semantic Scholar."""
    headers = _s2_headers()
    papers = []
    for doi in dois:
        try:
            time.sleep(S2_MIN_DELAY)
            resp = http_requests.get(
                f"{S2_PAPER}/{doi}", params={"fields": S2_FIELDS},
                headers=headers, timeout=30,
            )
            if resp.status_code == 200:
                papers.append(resp.json())
            else:
                print(f"  Could not fetch {doi}: HTTP {resp.status_code}", flush=True)
        except Exception as e:
            print(f"  Error fetching {doi}: {e}", flush=True)
    return papers


# ── Unified discovery ────────────────────────────────────────────────────────

def discover_all(sources: list[str], days: int, existing_dois: set[str]) -> list[dict]:
    """Run all requested discovery sources and return deduplicated papers."""
    all_new: dict[str, dict] = {}  # doi.lower() -> paper

    def _add(paper):
        doi = (paper.get("externalIds") or {}).get("DOI", paper.get("doi", ""))
        if not doi:
            return
        dl = doi.lower()
        if dl in existing_dois:
            return
        if dl in all_new:
            existing = all_new[dl]
            if paper.get("abstract") and not existing.get("abstract"):
                all_new[dl] = paper
            return
        all_new[dl] = paper

    if "arxiv" in sources or "all" in sources:
        print("\n[arXiv OAI-PMH]", flush=True)
        from_date = _get_from_date("arxiv", days)
        for p in discover_arxiv(from_date):
            _add(p)

    if "chemrxiv" in sources or "all" in sources:
        print("\n[ChemRxiv]", flush=True)
        from_date = _get_from_date("chemrxiv", days)
        for p in discover_chemrxiv(from_date):
            _add(p)

    if "rss" in sources:
        # Explicit ``rss`` request only — RSS feeds expose only the most
        # recent 20-30 items per journal and have proven structurally
        # unsuitable for catch-up windows > 1 week. CrossRef is the
        # supported replacement (per-publisher, date-precise). RSS still
        # works if the caller explicitly asks for it.
        print("\n[Journal RSS] (explicit only; use 'crossref' for periodic catch-up)",
              flush=True)
        for p in discover_rss():
            _add(p)

    if "crossref" in sources or "all" in sources:
        print("\n[CrossRef per-publisher metadata]", flush=True)
        from_date = _get_from_date("crossref", days)
        for p in discover_crossref(from_date, existing_dois | set(all_new.keys())):
            _add(p)

    if "s2" in sources or "all" in sources:
        print("\n[Semantic Scholar]", flush=True)
        from_date = _get_from_date("s2", max(days, 7))
        for p in discover_s2(from_date, existing_dois | set(all_new.keys())):
            _add(p)

    papers = list(all_new.values())
    papers_needing_enrichment = [p for p in papers if not p.get("abstract")]
    if papers_needing_enrichment:
        print(f"\n[Enrichment] {len(papers_needing_enrichment)} papers need S2 metadata...",
              flush=True)
        enriched = enrich_via_s2(papers_needing_enrichment)
        enriched_by_doi = {}
        for p in enriched:
            doi = (p.get("externalIds") or {}).get("DOI", p.get("doi", ""))
            if doi:
                enriched_by_doi[doi.lower()] = p
        papers = [
            enriched_by_doi.get(
                (p.get("externalIds") or {}).get("DOI", p.get("doi", "")).lower(), p
            )
            for p in papers
        ]

    papers = [p for p in papers if p.get("abstract")]
    papers.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)

    # Update last-run timestamps
    last_run = _load_last_run()
    now = datetime.now().isoformat()
    active = (sources if "all" not in sources
              else ["arxiv", "chemrxiv", "crossref", "s2"])
    for src in active:
        last_run[src] = now
    _save_last_run(last_run)

    return papers


# ── PDF helpers ──────────────────────────────────────────────────────────────

def doi_to_hash(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def find_pdf_for_paper(paper: dict) -> str | None:
    """Return local PDF path if the paper has one on disk, else None."""
    doi = (paper.get("externalIds") or {}).get("DOI", "")
    if not doi:
        return None
    # Try exact DOI first, then lowercase (S2 sometimes returns uppercase DOIs)
    for variant in [doi, doi.lower()]:
        fname = doi_to_hash(variant) + ".pdf"
        path = PAPERS_DIR / fname
        if path.exists():
            size = path.stat().st_size
            if 0 < size <= MAX_PDF_BYTES:
                return str(path)
    return None


def download_pdf(paper: dict) -> str | None:
    """Try to download the open-access PDF. Returns local path or None."""
    oa = paper.get("openAccessPdf") or {}
    url = oa.get("url", "")
    if not url:
        return None
    doi = (paper.get("externalIds") or {}).get("DOI", "")
    if not doi:
        return None

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    fname = doi_to_hash(doi) + ".pdf"
    path = PAPERS_DIR / fname
    if path.exists() and path.stat().st_size > 0:
        return str(path) if path.stat().st_size <= MAX_PDF_BYTES else None

    try:
        resp = http_requests.get(url, timeout=30, stream=True,
                                 headers={"User-Agent": "AskChem/1.0"})
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and "octet" not in content_type:
            return None
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        size = path.stat().st_size
        if size == 0 or size > MAX_PDF_BYTES:
            path.unlink(missing_ok=True)
            return None
        return str(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None


# ── Extraction prompts ───────────────────────────────────────────────────────

ABSTRACT_EXTRACTION_PROMPT = """You are a chemistry expert. Extract structured knowledge claims from this paper's abstract and metadata.

Paper metadata:
Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Abstract: {abstract}

Extract ALL factual claims. Return a JSON object with:
{{
  "claims": [
    {{
      "claim_id": sequential number,
      "claim_type": "reaction|property|method|mechanism|comparison|computational_result",
      "confidence": "high|medium|low",

      // For reactions:
      "reaction_type": "e.g., Suzuki coupling",
      "reactants": [{{"name": "...", "smiles": "or null", "role": "substrate|reagent|catalyst"}}],
      "products": [{{"name": "...", "smiles": "or null"}}],
      "conditions": {{"catalyst": "...", "solvent": "...", "temperature": "...", "other": "..."}},
      "outcomes": {{"yield_percent": null, "selectivity": "...", "other": "..."}},

      // For properties:
      "subject": "molecule/material",
      "property_name": "e.g., BET surface area",
      "value": "numeric or descriptive",
      "unit": "if applicable",
      "measurement_method": "technique",

      // For methods:
      "technique_name": "name",
      "what_it_achieves": "description",

      // For mechanisms:
      "process_described": "what process",
      "steps": ["step1", "step2"],

      // For all:
      "verbatim_quote": "exact text from abstract"
    }}
  ]
}}

Extract 3-10 claims from the abstract. Focus on the main findings."""


DEEP_EXTRACTION_PROMPT = """You are a chemistry expert performing EXHAUSTIVE knowledge extraction from a research paper.

Extract EVERY piece of structured knowledge. Target 20-50 claims per paper. Do NOT summarize — extract individual data points.

Return a JSON object with:
{
  "paper_knowledge": {
    "hypothesis": "The central hypothesis or research question",
    "experimental_design": "Brief description of the experimental approach",
    "conclusions": ["Main conclusion 1", "Main conclusion 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "future_directions": ["Future direction 1", "Future direction 2"],
    "surprising_findings": ["Any unexpected or counter-intuitive results"],
    "paper_type": "research_article|review|communication|computational_study|methods_paper",
    "subfield": "organic_synthesis|inorganic|materials|catalysis|physical_chemistry|biochemistry|computational|electrochemistry|photochemistry|polymer|environmental|analytical|other"
  },
  "claims": [
    {
      "claim_id": sequential number,
      "claim_type": "reaction|property|method|mechanism|comparison|scope_entry|computational_result|structure|hypothesis|experimental_design|limitation|future_direction|surprising_finding",
      "confidence": "high|medium|low",
      "location_in_paper": "Table 1, entry 3" or "Figure 2" or "Results, paragraph 4",

      // FOR REACTIONS (including each scope entry as a separate claim):
      "reaction_type": "e.g., Suzuki coupling, C-H activation, MOF synthesis",
      "reactants": [{"name": "...", "smiles": "... or null", "role": "substrate|reagent|catalyst|ligand|additive"}],
      "products": [{"name": "...", "smiles": "... or null", "role": "major|minor|byproduct"}],
      "conditions": {"catalyst": "...", "ligand": "...", "solvent": "...", "temperature": "...", "time": "...", "atmosphere": "...", "additives": ["..."], "other": "..."},
      "outcomes": {"yield_percent": null, "ee_percent": null, "selectivity": "...", "conversion_percent": null, "turnover_number": null},
      "is_key_result": true/false,

      // FOR PROPERTIES:
      "subject": "molecule/material name",
      "property_name": "e.g., melting point, BET surface area, IC50",
      "property_category": "physical|chemical|biological|spectroscopic|electrochemical|mechanical|optical|thermal",
      "value": "numerical value with units",
      "measurement_method": "instrument/technique",

      // FOR MECHANISMS:
      "process_described": "what reaction/process",
      "steps": ["step 1", "step 2"],
      "key_intermediates": ["..."],

      // FOR METHODS:
      "technique_name": "name",
      "what_it_achieves": "description",
      "key_innovation": "what's new",

      // FOR COMPARISONS:
      "compared_items": ["item A", "item B"],
      "metric": "what's being compared",
      "comparison_result": "A is better/worse/equal to B by X",

      // FOR ALL:
      "verbatim_quote": "exact sentence(s) from paper supporting this claim"
    }
  ]
}

CRITICAL: Extract EVERY entry from substrate scope and optimization tables as separate claims.
Include control experiments and negative results. A typical paper should yield 20-50 claims."""


# ── Real-time extraction (async) ─────────────────────────────────────────────

async def extract_one_abstract(paper: dict, semaphore: asyncio.Semaphore) -> list[dict]:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if not abstract:
        return []
    authors = [a.get("name", "") for a in (paper.get("authors") or [])[:5]]
    prompt = ABSTRACT_EXTRACTION_PROMPT.format(
        title=title, authors=", ".join(authors),
        year=paper.get("year", ""), venue=paper.get("venue", ""),
        abstract=abstract,
    )
    async with semaphore:
        for attempt in range(3):
            try:
                aclient = get_async_client()
                response = await aclient.chat.completions.create(
                    model=ABSTRACT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if content:
                    return json.loads(content).get("claims", [])
            except Exception:
                pass
            await asyncio.sleep(1)
    return []


def _get_portkey_client():
    """Lazy-init Portkey client for Gemini access via NYU gateway."""
    from portkey_ai import Portkey
    return Portkey(
        base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
        api_key=os.environ.get("PORTKEY_API_KEY", ""),
    )


def _extract_deep_gemini_sync(pdf_path: str) -> list[dict]:
    """Deep-extract using Gemini via Portkey (synchronous, faster for PDFs)."""
    pdf_bytes = open(pdf_path, 'rb').read()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')
    client = _get_portkey_client()
    delays = [5, 15, 45]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEEP_EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:application/pdf;base64,{pdf_b64}"}},
                    ],
                }],
                max_completion_tokens=65536,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if content:
                parsed = json.loads(content)
                return parsed.get("claims", [])
        except Exception as e:
            if attempt < 2:
                print(f"    Gemini attempt {attempt+1} failed: {str(e)[:80]}, "
                      f"retry in {delays[attempt]}s", flush=True)
                time.sleep(delays[attempt])
            else:
                print(f"    Gemini failed after 3 attempts: {str(e)[:80]}", flush=True)
    return []


async def extract_one_deep(paper: dict, pdf_path: str,
                           semaphore: asyncio.Semaphore) -> list[dict]:
    """Deep-extract from a PDF. Tries Gemini first, falls back to GPT-5.4."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        claims = await loop.run_in_executor(None, _extract_deep_gemini_sync, pdf_path)
        if claims:
            return claims

        # Fallback to GPT-5.4
        print(f"    Falling back to GPT-5.4 for {os.path.basename(pdf_path)}", flush=True)
        for attempt in range(2):
            try:
                pdf_bytes = open(pdf_path, 'rb').read()
                pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')
                aclient = get_async_client()
                response = await aclient.chat.completions.create(
                    model=DEEP_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DEEP_EXTRACTION_PROMPT},
                            {"type": "file", "file": {
                                "filename": os.path.basename(pdf_path),
                                "file_data": f"data:application/pdf;base64,{pdf_b64}",
                            }},
                        ],
                    }],
                    max_completion_tokens=16384,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if content:
                    return json.loads(content).get("claims", [])
            except Exception:
                pass
            await asyncio.sleep(2)
    return []


async def extract_batch_realtime(papers: list[dict],
                                 pdf_map: dict[str, str]) -> dict[str, list[dict]]:
    """Extract claims from all papers concurrently. Uses deep extraction when PDF available."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    deep_semaphore = asyncio.Semaphore(5)  # fewer concurrent deep extractions (larger payloads)
    results = {}
    completed = 0
    total = len(papers)
    deep_count = 0
    abstract_count = 0

    async def process(paper):
        nonlocal completed, deep_count, abstract_count
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        pdf_path = pdf_map.get(doi)
        if pdf_path:
            claims = await extract_one_deep(paper, pdf_path, deep_semaphore)
            deep_count += 1
        else:
            claims = await extract_one_abstract(paper, semaphore)
            abstract_count += 1
        results[doi] = claims
        completed += 1
        if completed % 20 == 0:
            print(f"  Extracted {completed}/{total} papers "
                  f"({deep_count} deep, {abstract_count} abstract)", flush=True)

    await asyncio.gather(*[process(p) for p in papers])
    print(f"  Final: {deep_count} deep extractions, {abstract_count} abstract extractions",
          flush=True)
    return results


# ── Batch API mode ───────────────────────────────────────────────────────────

def _build_abstract_batch_request(paper: dict, custom_id: str) -> dict:
    """Build a Batch API request for abstract extraction."""
    title = paper.get("title", "")
    authors = [a.get("name", "") for a in (paper.get("authors") or [])[:5]]
    prompt = ABSTRACT_EXTRACTION_PROMPT.format(
        title=title, authors=", ".join(authors),
        year=paper.get("year", ""), venue=paper.get("venue", ""),
        abstract=paper.get("abstract", ""),
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": ABSTRACT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"},
        },
    }


def _build_deep_batch_request(paper: dict, pdf_path: str, custom_id: str) -> dict:
    """Build a Batch API request for deep PDF extraction."""
    pdf_bytes = open(pdf_path, 'rb').read()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": DEEP_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": DEEP_EXTRACTION_PROMPT},
                    {"type": "file", "file": {
                        "filename": os.path.basename(pdf_path),
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    }},
                ],
            }],
            "max_completion_tokens": 16384,
            "response_format": {"type": "json_object"},
        },
    }


def _build_classify_batch_request(claim_summary: dict, custom_id: str,
                                  prompt_template: str) -> dict:
    from askchem.taxonomy import CLASSIFICATION_SYSTEM_PROMPT
    claim_json = json.dumps(claim_summary, indent=2)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": CLASSIFICATION_MODEL,
            "messages": [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this claim:\n{claim_json}"},
            ],
            "max_completion_tokens": 2048,
            "response_format": {"type": "json_object"},
        },
    }


def batch_prepare(papers: list[dict], pdf_map: dict[str, str]):
    """Build batch JSONL files for extraction and save metadata."""
    from openai import OpenAI
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    doi_to_custom = {}
    batch_idx = 0
    current_file = None
    current_size = 0
    papers_in_batch = 0
    batch_files = []
    deep_count = 0
    abstract_count = 0
    skipped = 0

    for paper in papers:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        if not doi:
            continue
        custom_id = f"ext_{doi_to_hash(doi)}"
        doi_to_custom[doi] = custom_id
        pdf_path = pdf_map.get(doi)

        try:
            if pdf_path:
                request = _build_deep_batch_request(paper, pdf_path, custom_id)
                deep_count += 1
            else:
                request = _build_abstract_batch_request(paper, custom_id)
                abstract_count += 1
            line = json.dumps(request) + "\n"
            line_bytes = len(line.encode('utf-8'))
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip {doi}: {str(e)[:60]}", flush=True)
            continue

        if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {papers_in_batch} papers, "
                      f"{current_size / 1e6:.1f} MB", flush=True)
            batch_idx += 1
            fname = BATCH_DIR / f"extract_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            current_size = 0
            papers_in_batch = 0

        current_file.write(line)
        current_size += line_bytes
        papers_in_batch += 1

    if current_file:
        current_file.close()
        if papers_in_batch > 0:
            print(f"  {batch_files[-1].name}: {papers_in_batch} papers, "
                  f"{current_size / 1e6:.1f} MB", flush=True)

    meta = {
        "phase": "extract",
        "total_papers": len(doi_to_custom),
        "deep_count": deep_count,
        "abstract_count": abstract_count,
        "skipped": skipped,
        "batch_files": [f.name for f in batch_files],
        "doi_to_custom": doi_to_custom,
        "created_at": datetime.now().isoformat(),
    }
    with open(BATCH_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(BATCH_DIR / "papers.json", "w") as f:
        json.dump(papers, f)

    print(f"\n  Prepared {len(doi_to_custom)} extraction requests "
          f"({deep_count} deep, {abstract_count} abstract) "
          f"in {len(batch_files)} batch files", flush=True)
    if skipped:
        print(f"  Skipped: {skipped}", flush=True)

    # Submit
    print("\nSubmitting extraction batches...", flush=True)
    client = OpenAI()
    tracker = {}
    for fpath in batch_files:
        size_mb = fpath.stat().st_size / 1e6
        print(f"  Uploading {fpath.name} ({size_mb:.1f} MB)...", flush=True)
        uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        tracker[fpath.name] = {
            "batch_id": batch.id, "file_id": uploaded.id,
            "status": batch.status,
            "submitted_at": datetime.now().isoformat(),
        }
        print(f"  Batch {batch.id} ({batch.status})", flush=True)
        time.sleep(2)

    with open(BATCH_DIR / "extract_tracker.json", "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"\n  {len(tracker)} extraction batches submitted.", flush=True)
    print(f"  Poll with: python src/update_index.py --batch --poll", flush=True)


def batch_poll():
    """Check status of submitted batches (extraction and classification)."""
    from openai import OpenAI
    client = OpenAI()

    for phase in ["extract", "classify"]:
        tracker_file = BATCH_DIR / f"{phase}_tracker.json"
        if not tracker_file.exists():
            continue
        with open(tracker_file) as f:
            tracker = json.load(f)

        print(f"\n=== {phase.upper()} BATCHES ===", flush=True)
        all_done = True
        total_completed = 0
        total_total = 0
        for fname, info in sorted(tracker.items()):
            batch = client.batches.retrieve(info["batch_id"])
            info["status"] = batch.status
            if batch.output_file_id:
                info["output_file_id"] = batch.output_file_id
            if batch.error_file_id:
                info["error_file_id"] = batch.error_file_id
            rc = batch.request_counts
            status_str = batch.status
            if rc:
                status_str += f" ({rc.completed}/{rc.total} done, {rc.failed} failed)"
                total_completed += rc.completed
                total_total += rc.total
            print(f"  {fname}: {status_str}", flush=True)
            if batch.status not in ("completed", "failed", "expired", "cancelled"):
                all_done = False

        with open(tracker_file, "w") as f:
            json.dump(tracker, f, indent=2)
        print(f"  Overall: {total_completed}/{total_total} completed", flush=True)

        if all_done and phase == "extract":
            print(f"\n  All extraction batches done!", flush=True)
            print(f"  Collect with: python src/update_index.py --batch --collect", flush=True)
        elif all_done and phase == "classify":
            print(f"\n  All classification batches done!", flush=True)
            print(f"  Collect with: python src/update_index.py --batch --collect", flush=True)


def batch_collect():
    """Download batch results, run classification if needed, and index."""
    from openai import OpenAI
    client = OpenAI()

    meta_file = BATCH_DIR / "meta.json"
    if not meta_file.exists():
        print("No batch metadata found. Run --batch first.", flush=True)
        return
    with open(meta_file) as f:
        meta = json.load(f)

    phase = meta.get("phase", "extract")

    if phase == "extract":
        _collect_extractions(client, meta)
    elif phase == "classify":
        _collect_classifications(client, meta)


def _collect_extractions(client, meta):
    """Download extraction results and submit classification batches."""
    tracker_file = BATCH_DIR / "extract_tracker.json"
    if not tracker_file.exists():
        print("No extraction tracker found.", flush=True)
        return
    with open(tracker_file) as f:
        tracker = json.load(f)

    doi_to_custom = meta.get("doi_to_custom", {})
    custom_to_doi = {v: k for k, v in doi_to_custom.items()}

    extractions = {}
    total_claims = 0
    errors = 0

    raw_dir = BATCH_DIR / "raw_extract"
    raw_dir.mkdir(exist_ok=True)

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            print(f"  {fname}: no output (status={info.get('status')})", flush=True)
            continue

        raw_path = raw_dir / fname
        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            with open(raw_path, "wb") as f:
                f.write(content.read())

        with open(raw_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    response = result.get("response", {})
                    body = response.get("body", {})
                    if response.get("status_code") != 200:
                        errors += 1
                        continue
                    choices = body.get("choices", [])
                    if not choices:
                        errors += 1
                        continue
                    text = choices[0].get("message", {}).get("content", "")
                    parsed = json.loads(text)
                    claims = parsed.get("claims", [])
                    doi = custom_to_doi.get(custom_id, custom_id.replace("ext_", ""))
                    extractions[doi] = claims
                    total_claims += len(claims)
                except Exception:
                    errors += 1

    print(f"\n  Collected {len(extractions)} papers, {total_claims} claims, {errors} errors",
          flush=True)

    with open(BATCH_DIR / "extractions.json", "w") as f:
        json.dump(extractions, f)

    # Now build classification requests
    print("\nBuilding classification batches...", flush=True)
    with open(BATCH_DIR / "papers.json") as f:
        papers = json.load(f)

    canonical = load_canonical_categories()
    prompt_template = build_constrained_classification_prompt(canonical)

    claim_objects_meta = []
    for paper in papers:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        title = paper.get("title", "")
        for raw in extractions.get(doi, []):
            claim_type = raw.get("claim_type", "unknown") or "unknown"
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)
            claim_summary = {
                "claim_type": claim_type,
                "reaction_type": raw.get("reaction_type", ""),
                "subject": raw.get("subject", ""),
                "property_name": raw.get("property_name", ""),
                "technique_name": raw.get("technique_name", ""),
                "process_described": raw.get("process_described", ""),
                "verbatim_quote": (raw.get("verbatim_quote") or "")[:200],
                "reactants": (raw.get("reactants") or [])[:3],
                "products": (raw.get("products") or [])[:3],
                "conditions": raw.get("conditions") or {},
            }
            claim_summary = {k: v for k, v in claim_summary.items() if v}
            claim_objects_meta.append((claim_id, claim_summary))

    print(f"  {len(claim_objects_meta)} claims to classify", flush=True)

    batch_idx = 0
    current_file = None
    current_size = 0
    items_in_batch = 0
    batch_files = []
    id_map = {}

    for claim_id, summary in claim_objects_meta:
        custom_id = f"cls_{claim_id}"
        id_map[custom_id] = claim_id
        request = _build_classify_batch_request(summary, custom_id, prompt_template)
        line = json.dumps(request) + "\n"
        line_bytes = len(line.encode('utf-8'))

        if current_file is None or current_size + line_bytes > MAX_BATCH_FILE_BYTES:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {items_in_batch} claims, "
                      f"{current_size / 1e6:.1f} MB", flush=True)
            batch_idx += 1
            fname = BATCH_DIR / f"classify_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, 'w')
            current_size = 0
            items_in_batch = 0

        current_file.write(line)
        current_size += line_bytes
        items_in_batch += 1

    if current_file:
        current_file.close()
        if items_in_batch > 0:
            print(f"  {batch_files[-1].name}: {items_in_batch} claims, "
                  f"{current_size / 1e6:.1f} MB", flush=True)

    # Submit classification batches
    print(f"\nSubmitting {len(batch_files)} classification batches...", flush=True)
    cls_tracker = {}
    for fpath in batch_files:
        size_mb = fpath.stat().st_size / 1e6
        print(f"  Uploading {fpath.name} ({size_mb:.1f} MB)...", flush=True)
        uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        cls_tracker[fpath.name] = {
            "batch_id": batch.id, "file_id": uploaded.id,
            "status": batch.status,
            "submitted_at": datetime.now().isoformat(),
        }
        print(f"  Batch {batch.id} ({batch.status})", flush=True)
        time.sleep(2)

    with open(BATCH_DIR / "classify_tracker.json", "w") as f:
        json.dump(cls_tracker, f, indent=2)

    meta["phase"] = "classify"
    meta["classify_id_map"] = id_map
    with open(BATCH_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  {len(cls_tracker)} classification batches submitted.", flush=True)
    print(f"  Poll with: python src/update_index.py --batch --poll", flush=True)


def _collect_classifications(client, meta):
    """Download classification results and write everything to the index."""
    tracker_file = BATCH_DIR / "classify_tracker.json"
    if not tracker_file.exists():
        print("No classification tracker found.", flush=True)
        return
    with open(tracker_file) as f:
        tracker = json.load(f)

    id_map = meta.get("classify_id_map", {})
    classifications = []
    errors = 0

    raw_dir = BATCH_DIR / "raw_classify"
    raw_dir.mkdir(exist_ok=True)

    for fname, info in sorted(tracker.items()):
        output_id = info.get("output_file_id")
        if not output_id:
            continue
        raw_path = raw_dir / fname
        if not raw_path.exists():
            print(f"  Downloading {fname}...", flush=True)
            content = client.files.content(output_id)
            with open(raw_path, "wb") as f:
                f.write(content.read())

        with open(raw_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    response = result.get("response", {})
                    body = response.get("body", {})
                    if response.get("status_code") != 200:
                        errors += 1
                        continue
                    choices = body.get("choices", [])
                    if not choices:
                        errors += 1
                        continue
                    text = choices[0].get("message", {}).get("content", "")
                    paths = json.loads(text)
                    claim_id = id_map.get(custom_id, custom_id.replace("cls_", ""))
                    classifications.append({"claim_id": claim_id, "paths": paths})
                except Exception:
                    errors += 1

    print(f"  Collected {len(classifications)} classifications, {errors} errors", flush=True)

    # Load papers and extractions
    with open(BATCH_DIR / "papers.json") as f:
        papers = json.load(f)
    with open(BATCH_DIR / "extractions.json") as f:
        extractions = json.load(f)

    # Write to index
    print("\nWriting to SQLite database...", flush=True)
    result = add_to_index(papers, extractions, classifications)

    print(f"\n{'='*60}", flush=True)
    print("BATCH UPDATE COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  New sources:  {result['sources_added']}", flush=True)
    print(f"  New claims:   {result['claims_added']}", flush=True)
    print(f"  Assignments:  {result['assignments']}", flush=True)

    with get_conn() as conn:
        total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        total_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    print(f"\n  Database now: {total_claims:,} claims, {total_sources:,} sources",
          flush=True)

    # Update embeddings for new claims
    print("\nUpdating claim embeddings...", flush=True)
    try:
        from askchem.embeddings import update_embeddings
        n_new = update_embeddings()
        if n_new:
            print(f"  Embedded {n_new} new claims", flush=True)
        else:
            print("  No new claims to embed", flush=True)
    except Exception as e:
        print(f"  Embedding update failed: {e}", flush=True)

    # Cleanup
    import shutil
    shutil.rmtree(BATCH_DIR)
    print("  Cleaned up batch directory", flush=True)

    # Sync
    print("\nSyncing to HuggingFace...", flush=True)
    try:
        from upload_to_hf import sync_to_hf
        sync_to_hf()
    except Exception as e:
        print(f"  HuggingFace sync failed: {e}", flush=True)


# ── Classification (constrained to canonical L1s) ────────────────────────────

def load_canonical_categories() -> dict[str, list[str]]:
    from askchem.taxonomy import CANONICAL_L1
    return CANONICAL_L1


def build_constrained_classification_prompt(canonical: dict) -> str:
    """Build the classification prompt template (L1/L2 only; L3 assigned separately)."""
    from askchem.taxonomy import _TAXONOMY_TEXT

    return """You are classifying a chemistry knowledge claim into a hierarchical index with 5 views.

The claim:
{{claim_json}}

Rules:
- L1 MUST be one of the listed categories (exactly one per view).
- L2 MUST be one of the listed subcategories under that L1.
- L3 is NOT needed here — it will be assigned separately.
- Use lowercase_with_underscores.
- If the claim does not fit a view, use ["not_applicable"].

Canonical categories (L1 → allowed L2):
""" + _TAXONOMY_TEXT + """

Return a JSON object:
{{
  "by_reaction_type": ["l1", "l2"],
  "by_substance_class": ["l1", "l2"],
  "by_application": ["l1", "l2"],
  "by_technique": ["l1", "l2"],
  "by_mechanism": ["l1", "l2"]
}}"""


async def classify_one(claim: Claim, prompt_template: str,
                       semaphore: asyncio.Semaphore) -> dict:
    from askchem.taxonomy import FULL_CLASSIFICATION_SYSTEM_PROMPT
    claim_summary = {
        "claim_type": claim.claim_type or "",
        "reaction_type": claim.reaction_type or "",
        "subject": claim.subject or "",
        "property_name": claim.property_name or "",
        "technique_name": claim.technique_name or "",
        "process_described": claim.process_described or "",
        "verbatim_quote": (claim.verbatim_quote or "")[:200],
        "reactants": (claim.reactants or [])[:3],
        "products": (claim.products or [])[:3],
        "conditions": claim.conditions or {},
    }
    claim_summary = {k: v for k, v in claim_summary.items() if v}
    claim_json = json.dumps(claim_summary, indent=2)

    async with semaphore:
        for attempt in range(3):
            try:
                aclient = get_async_client()
                response = await aclient.chat.completions.create(
                    model=CLASSIFICATION_MODEL,
                    messages=[
                        {"role": "system", "content": FULL_CLASSIFICATION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Classify this claim:\n{claim_json}"},
                    ],
                    max_completion_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if content:
                    return {"claim_id": claim.claim_id, "paths": json.loads(content)}
            except Exception:
                pass
            await asyncio.sleep(1)
    return {"claim_id": claim.claim_id, "paths": {}}


async def classify_batch_realtime(claims: list[Claim], prompt_template: str) -> list[dict]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    completed = 0
    total = len(claims)
    results = []

    async def process(claim):
        nonlocal completed
        result = await classify_one(claim, prompt_template, semaphore)
        results.append(result)
        completed += 1
        if completed % 50 == 0:
            print(f"  Classified {completed}/{total} claims", flush=True)

    await asyncio.gather(*[process(c) for c in claims])
    return results


# ── Index Integration ────────────────────────────────────────────────────────

def add_to_index(papers: list[dict],
                 extractions: dict[str, list[dict]],
                 classifications: list[dict]):
    """Add new sources, claims, and tree nodes directly to SQLite."""
    from askchem.taxonomy import normalize_path, build_claim_type_path, ALL_CONTENT_VIEWS

    claims_added = 0
    sources_added = 0
    assignments = 0

    class_by_id = {c["claim_id"]: c for c in classifications}
    all_claim_dicts = []
    paper_claim_ids: dict[str, list[str]] = {}

    for paper in papers:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        if not doi:
            continue

        title = paper.get("title", "")
        authors = [a.get("name", "") for a in (paper.get("authors") or [])[:10]]
        source_data = {
            'doi': doi, 'title': title, 'authors': authors,
            'year': paper.get("year") or 0, 'venue': paper.get("venue", ""),
            'abstract': paper.get("abstract", ""),
            'citation_count': paper.get("citationCount", 0) or 0,
            'open_access_url': (paper.get("openAccessPdf") or {}).get("url", ""),
        }
        upsert_source(source_data)
        sources_added += 1

        raw_claims = extractions.get(doi, [])
        for raw in raw_claims:
            claim_type = raw.get("claim_type", "unknown") or "unknown"
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            classification = class_by_id.get(claim_id, {})
            raw_paths = classification.get("paths", {})

            view_paths = {}
            for view_id in ALL_CONTENT_VIEWS:
                normalized = normalize_path(view_id, raw_paths.get(view_id, []))
                if normalized:
                    view_paths[view_id] = normalized

            ct_path = build_claim_type_path(claim_type)
            view_paths['by_claim_type'] = ct_path

            claim_data = dict(raw)
            claim_data.update({
                'claim_id': claim_id,
                'claim_type': claim_type,
                'source_doi': doi,
                'source_paper_title': title,
                'confidence': raw.get('confidence', 'medium'),
                'location_in_paper': raw.get('location_in_paper', 'abstract'),
                'extraction_model': DEEP_MODEL if raw.get('location_in_paper') else ABSTRACT_MODEL,
                'extraction_version': 'v4-normalized',
                'extracted_at': datetime.now().isoformat(),
                'view_paths': view_paths,
            })
            all_claim_dicts.append(claim_data)
            claims_added += 1

            paper_claim_ids.setdefault(doi, []).append(claim_id)

            for view_id, path in view_paths.items():
                for depth in range(len(path)):
                    partial_path = '/'.join(path[:depth + 1])
                    append_claim_to_node(view_id, partial_path, claim_id)
                assignments += 1

    if all_claim_dicts:
        upsert_claims_batch(all_claim_dicts)

    for paper in papers:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        if doi and doi in paper_claim_ids:
            doi_path = doi.replace('/', '__')
            title = paper.get("title", doi)
            cids = paper_claim_ids[doi]
            upsert_tree_node(
                'by_paper', doi_path, name=title, level=1,
                claim_ids=cids,
                data={
                    'view_id': 'by_paper', 'path': doi_path,
                    'name': title, 'level': 1,
                    'claim_count': len(cids), 'children': [],
                    'claim_ids': cids, 'doi': doi,
                },
            )

    for paper in papers:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        if doi:
            try:
                index_authors_for_doi(doi)
            except Exception:
                pass

    update_metadata_counts()

    return {
        "sources_added": sources_added,
        "claims_added": claims_added,
        "assignments": assignments,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Incrementally update AskChem index")
    parser.add_argument("--source", type=str, nargs="*",
                        default=["all"],
                        choices=["arxiv", "chemrxiv", "rss", "crossref", "s2", "all"],
                        help="Discovery sources (default: all)")
    parser.add_argument("--days", type=int, default=1,
                        help="Look back N days for new papers (default: 1)")
    parser.add_argument("--dois", type=str, nargs="*",
                        help="Specific DOIs to add (skips discovery)")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API (50%% cheaper, up to 24h)")
    parser.add_argument("--poll", action="store_true",
                        help="Poll batch status (use with --batch)")
    parser.add_argument("--collect", action="store_true",
                        help="Collect batch results and index (use with --batch)")
    parser.add_argument("--download-pdfs", action="store_true",
                        help="Download open-access PDFs for discovered papers")
    parser.add_argument("--no-deep", action="store_true",
                        help="Skip deep extraction even if PDFs available")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip subscription notifications after indexing")
    args = parser.parse_args()

    # Handle batch poll/collect without discovery
    if args.batch and args.poll:
        batch_poll()
        return
    if args.batch and args.collect:
        batch_collect()
        return

    print(f"{'='*60}", flush=True)
    print(f"AskChem Index Update — {datetime.now().isoformat()}", flush=True)
    print(f"Mode: {'Batch API' if args.batch else 'Real-time'}", flush=True)
    print(f"Sources: {', '.join(args.source)}", flush=True)
    print(f"Lookback: {args.days} day(s)", flush=True)
    print(f"{'='*60}\n", flush=True)

    existing_dois = get_existing_dois()
    print(f"Existing index: {len(existing_dois):,} sources\n", flush=True)

    # ── Step 1: Discover new papers ──────────────────────────────────────
    print("Step 1: Discovering new papers...", flush=True)

    if args.dois:
        papers = fetch_papers_by_doi(args.dois)
        papers = [p for p in papers
                  if (p.get("externalIds") or {}).get("DOI", "").lower()
                  not in existing_dois]
    else:
        papers = discover_all(args.source, args.days, existing_dois)

    if not papers:
        print("\nNo new papers found. Index is up to date.", flush=True)
        return

    print(f"\n  Found {len(papers)} new papers to process\n", flush=True)

    # ── Step 1b: Find or download PDFs ───────────────────────────────────
    pdf_map = {}  # doi -> local pdf path
    if not args.no_deep:
        print("Step 1b: Checking for PDFs...", flush=True)
        for paper in papers:
            doi = (paper.get("externalIds") or {}).get("DOI", "")
            if not doi:
                continue
            pdf_path = find_pdf_for_paper(paper)
            if pdf_path:
                pdf_map[doi] = pdf_path

        if args.download_pdfs:
            print(f"  {len(pdf_map)} PDFs already on disk, downloading more...", flush=True)
            downloaded = 0
            for paper in papers:
                doi = (paper.get("externalIds") or {}).get("DOI", "")
                if not doi or doi in pdf_map:
                    continue
                path = download_pdf(paper)
                if path:
                    pdf_map[doi] = path
                    downloaded += 1
            print(f"  Downloaded {downloaded} new PDFs", flush=True)

        print(f"  {len(pdf_map)}/{len(papers)} papers have PDFs (will use deep extraction)",
              flush=True)
        print(f"  {len(papers) - len(pdf_map)} papers abstract-only\n", flush=True)

    # ── Batch mode: prepare and submit ───────────────────────────────────
    if args.batch:
        batch_prepare(papers, pdf_map)
        return

    # ── Real-time mode ───────────────────────────────────────────────────

    # ── Step 2: Extract claims ───────────────────────────────────────────
    update_dir = INDEX_DIR / "_update_checkpoints"
    update_dir.mkdir(parents=True, exist_ok=True)
    extraction_cache = update_dir / "extractions.json"
    papers_cache = update_dir / "papers.json"

    if extraction_cache.exists() and papers_cache.exists():
        print("Step 2: Loading cached extractions...", flush=True)
        with open(extraction_cache) as f:
            extractions = json.load(f)
        with open(papers_cache) as f:
            papers = json.load(f)
        total_claims = sum(len(v) for v in extractions.values())
        print(f"  Loaded {total_claims} claims from {len(papers)} papers (cached)\n",
              flush=True)
    else:
        print("Step 2: Extracting claims...", flush=True)
        t0 = time.time()
        extractions = asyncio.run(extract_batch_realtime(papers, pdf_map))
        total_claims = sum(len(v) for v in extractions.values())
        elapsed = time.time() - t0
        print(f"  Extracted {total_claims} claims from {len(papers)} papers "
              f"in {elapsed:.0f}s\n", flush=True)

        with open(extraction_cache, "w") as f:
            json.dump(extractions, f)
        with open(papers_cache, "w") as f:
            json.dump(papers, f)

    if total_claims == 0:
        print("No claims extracted. Nothing to add.", flush=True)
        return

    # ── Step 3: Build Claim objects for classification ───────────────────
    def _s(val, default=""):
        if val is None:
            return default
        return val

    print("Step 3: Building claim objects...", flush=True)
    claim_objects = []
    for paper in papers:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        title = paper.get("title", "")
        for raw in extractions.get(doi, []):
            claim_type = _s(raw.get("claim_type"), "unknown")
            content_hash = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)
            claim_objects.append(Claim(
                claim_id=claim_id,
                claim_type=claim_type,
                source_doi=doi,
                source_paper_title=title,
                verbatim_quote=_s(raw.get("verbatim_quote")),
                reaction_type=_s(raw.get("reaction_type")),
                subject=_s(raw.get("subject")),
                property_name=_s(raw.get("property_name")),
                technique_name=_s(raw.get("technique_name")),
                process_described=_s(raw.get("process_described")),
                reactants=raw.get("reactants") or [],
                products=raw.get("products") or [],
                conditions=raw.get("conditions") or {},
            ))
    print(f"  {len(claim_objects)} claims ready for classification\n", flush=True)

    # ── Step 4: Classify with constrained canonical L1s ──────────────────
    print("Step 4: Classifying claims (constrained to canonical L1s)...", flush=True)
    canonical = load_canonical_categories()
    prompt_template = build_constrained_classification_prompt(canonical)
    print(f"  Using canonical L1 taxonomy "
          f"({sum(len(v) for v in canonical.values())} categories "
          f"across {len(canonical)} views)", flush=True)

    t0 = time.time()
    classifications = asyncio.run(classify_batch_realtime(claim_objects, prompt_template))
    elapsed = time.time() - t0
    print(f"  Classified {len(classifications)} claims in {elapsed:.0f}s\n", flush=True)

    # ── Step 5: Add to index ─────────────────────────────────────────────
    print("Step 5: Writing to SQLite database...", flush=True)
    result = add_to_index(papers, extractions, classifications)

    print(f"\n{'='*60}", flush=True)
    print("UPDATE COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  New sources:  {result['sources_added']}", flush=True)
    print(f"  New claims:   {result['claims_added']}", flush=True)
    print(f"  Assignments:  {result['assignments']}", flush=True)

    with get_conn() as conn:
        total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        total_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    print(f"\n  Database now: {total_claims:,} claims, {total_sources:,} sources",
          flush=True)

    update_dir = INDEX_DIR / "_update_checkpoints"
    if update_dir.exists():
        import shutil
        shutil.rmtree(update_dir)
        print("  Cleaned up update checkpoints", flush=True)

    # ── Step 5b: Update embeddings for new claims ────────────────────────
    print("\nStep 5b: Updating claim embeddings...", flush=True)
    try:
        from askchem.embeddings import update_embeddings
        n_new = update_embeddings()
        if n_new:
            print(f"  Embedded {n_new} new claims", flush=True)
        else:
            print("  No new claims to embed", flush=True)
    except Exception as e:
        print(f"  Embedding update failed: {e}", flush=True)
        print("  You can manually rebuild with: python -m askchem.embeddings build", flush=True)

    # ── Step 6: Sync to HuggingFace ─────────────────────────────────────
    print("\nStep 6: Syncing to HuggingFace...", flush=True)
    try:
        from upload_to_hf import sync_to_hf
        sync_to_hf()
    except Exception as e:
        print(f"  HuggingFace sync failed: {e}", flush=True)
        print("  You can manually sync later with: python src/upload_to_hf.py", flush=True)

    # ── Step 7: Pre-generate reading lists for L1 topics ────────────────
    print("\nStep 7: Pre-generating reading lists for L1 topics...", flush=True)
    try:
        _pregen_reading_lists()
    except Exception as e:
        print(f"  Reading list pre-generation failed: {e}", flush=True)

    # ── Step 8: Send subscription notifications ──────────────────────────
    if not args.no_notify:
        print("\nStep 8: Checking subscriptions...", flush=True)
        try:
            from askchem.notify import check_subscriptions
            sent = check_subscriptions()
            if sent:
                print(f"  Sent {sent} notification(s)", flush=True)
            else:
                print("  No notifications due", flush=True)
        except Exception as e:
            print(f"  Notification check failed: {e}", flush=True)


def _pregen_reading_lists():
    """Pre-generate curated reading lists for all L1 topic nodes."""
    from askchem.db import get_conn, get_reading_list
    from askchem.llm import chat_full

    CONTENT_VIEWS = [
        "by_reaction_type", "by_substance_class",
        "by_application", "by_technique", "by_mechanism",
    ]

    CURATION_PROMPT = """You are a chemistry professor creating a guided reading list for a student entering the research area of "{topic}".

Below are the papers contributing to this topic, with their metadata. Create a pedagogical reading order.

Papers:
{papers_json}

Return a JSON object:
{{
  "reading_order": [
    {{
      "doi": "the paper DOI",
      "reading_order": 1,
      "annotation": "One sentence explaining why to read this paper and what the reader will learn."
    }}
  ],
  "summary": "2-3 sentence overview of this research area and what the reading list covers."
}}

Rules:
- Put review articles and foundational/seminal works first
- Then key methodological advances
- Then recent results and frontier work
- Include ALL papers from the input list
- Keep annotations concise (1 sentence each)"""

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT view_id, path, name FROM tree_nodes WHERE level = 1 "
            "AND view_id IN (?, ?, ?, ?, ?)",
            CONTENT_VIEWS,
        ).fetchall()

    l1_nodes = [(r["view_id"], r["path"], r["name"]) for r in rows]
    print(f"  Found {len(l1_nodes)} L1 nodes to pre-generate", flush=True)

    generated = 0
    for view_id, path, name in l1_nodes:
        try:
            rl = get_reading_list(view_id, path, limit=30)
            if rl["total_papers"] == 0:
                continue

            all_papers = []
            for tier in rl["tiers"]:
                all_papers.extend(tier["papers"])
            if not all_papers:
                continue

            papers_for_llm = []
            for p in all_papers[:50]:
                authors = p.get("authors", [])
                if isinstance(authors, list) and authors:
                    author_names = [
                        a if isinstance(a, str) else a.get("name", "")
                        for a in authors[:5]
                    ]
                    author_str = ", ".join(author_names)
                else:
                    author_str = ""
                papers_for_llm.append({
                    "doi": p["doi"], "title": p["title"],
                    "authors": author_str, "year": p["year"],
                    "venue": p["venue"],
                    "citation_count": p["citation_count"],
                    "claim_count": p["claim_count"],
                    "abstract": p.get("abstract", "")[:200],
                })

            topic_name = name or path
            prompt = CURATION_PROMPT.format(
                topic=topic_name,
                papers_json=json.dumps(papers_for_llm, indent=2),
            )

            chat_full(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                max_completion_tokens=4096,
            )
            generated += 1
            if generated % 10 == 0:
                print(f"  Generated {generated}/{len(l1_nodes)} reading lists", flush=True)

        except Exception as e:
            print(f"  Error for {view_id}/{path}: {e}", flush=True)

    print(f"  Pre-generated {generated} curated reading lists", flush=True)


if __name__ == "__main__":
    main()
