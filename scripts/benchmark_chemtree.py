#!/usr/bin/env python3
from __future__ import annotations
"""
AskChem-Bench: Quantitative evaluation of structured scientific retrieval.

Compares GPT-only answers against AskChem-assisted answers across 30 chemistry
questions spanning three formal task types:
  Task A - Cross-Paper Condition Aggregation (CA)
  Task B - Temporal Claim Tracking (TC)
  Task C - Contradiction Surfacing (CS)

This refreshed benchmark keeps the old "strict grounded" setting and adds a
more realistic "retrieval assisted" setting. It can also run Edison Scientific
as an external baseline on a balanced 9-question subset.

Usage:
    export OPENAI_API_KEY=sk-...
    export EDISON=...
    BENCH_MODEL=gpt-5.4 python scripts/benchmark_chemtree.py

Output: scripts/benchmark_results_<model>.json
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai requests")
    sys.exit(1)

try:
    from edison_client import EdisonClient, JobNames
except ImportError:
    EdisonClient = None
    JobNames = None

ASKCHEM_API = os.environ.get("ASKCHEM_API", "https://askchem.org/api")
MODEL = os.environ.get("BENCH_MODEL", "gpt-5.5")
DEFAULT_OUTPUT_FILE = Path(__file__).parent / f"benchmark_results_{MODEL.replace('/', '_')}.json"
OUTPUT_FILE = Path(os.environ.get("BENCH_OUTPUT_FILE", str(DEFAULT_OUTPUT_FILE)))
CACHE_FILE = Path(os.environ.get("BENCH_CACHE_FILE", str(DEFAULT_OUTPUT_FILE)))
CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_HEADERS = {
    "User-Agent": "AskChemBench/1.0 (mailto:askchem@askchem.org)"
}
# Unified mode (the only AskChem mode emitted on new runs): LLM rewriter
# fans the question into 3-4 short keyword sub-queries, concurrent
# /api/search calls (the one hybrid retriever; FTS + paper-level + tree
# recall + vector all fused in search_claims), merge by claim_id,
# diversify to ≤40 claims with ≤4 per source, then a grounded synthesiser.
#
# Concurrency note: prod /api/search runs hybrid retrieval (FTS + vector
# rerank) on a single VPS CPU and takes ~7-9 s per short query. Hammering
# it with 4 concurrent calls saturates the backend; 2 in-flight is the
# sweet spot — enough overlap to shorten the wall-clock without stacking
# rerankers. Timeout is generous (60 s) because the first call after a
# warm-up gap can take >20 s.
UNIFIED_MAX_CLAIMS = 40
UNIFIED_MAX_PER_SOURCE = 4
UNIFIED_LIMIT_PER_QUERY = 20
UNIFIED_QUERY_TIMEOUT = 60
UNIFIED_MAX_SUB_QUERIES = 4
UNIFIED_WORKERS = 2
# Paperclip retrieval (gxl-paperclip 0.3.0; requires Python 3.10+ subprocess).
PAPERCLIP_PYTHON = os.environ.get(
    "PAPERCLIP_PYTHON",
    shutil.which("python3.14") or shutil.which("python3.12") or "",
)
PAPERCLIP_SCRIPT = Path(__file__).parent / "paperclip_bench_client.py"
PAPERCLIP_LIMIT_PER_QUERY = 15
PAPERCLIP_MAX_PAPERS = 40
PAPERCLIP_MAX_PER_AUTHOR = 4
PAPERCLIP_SNIPPET_LINES = 40
PAPERCLIP_SOURCE_FILTER = os.environ.get("PAPERCLIP_SOURCES", "pmc,arxiv")
PAPERCLIP_SOURCE_FALLBACK = os.environ.get(
    "PAPERCLIP_SOURCES_FALLBACK", "pmc,arxiv,abstracts_only"
)
PAPERCLIP_QUERY_TIMEOUT = 120
PAPERCLIP_WORKERS = 2
_DOI_IN_QUERY_RE = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+|PMC\d+|PMID[:\s]?\d+)\b", re.I
)
RESUME_EXISTING = os.environ.get("BENCH_RESUME", "1") != "0"
EDISON_API = os.environ.get("EDISON_API", "https://api.platform.edisonscientific.com").rstrip("/")
EDISON_JOB_NAME = os.environ.get("EDISON_JOB_NAME", "literature-20260216")
EDISON_POLL_SECONDS = float(os.environ.get("EDISON_POLL_SECONDS", "5"))
BENCH_TASKS = {
    item.strip().upper() for item in os.environ.get("BENCH_TASKS", "").split(",") if item.strip()
}
BENCH_IDS = {
    item.strip().lower() for item in os.environ.get("BENCH_IDS", "").split(",") if item.strip()
}
EDISON_SUBSET_IDS = [
    # original balanced-9 (Edison answers cached from Apr 15 local_deep run;
    # answers are model-independent so the v1-era snapshot is reused without
    # leaking gpt-5.4 answer-head outputs into the gpt-5.5 columns).
    "ca02", "ca04", "ca10",
    "tc01", "tc03", "tc05",
    "cs01", "cs03", "cs08",
    # balanced-18 expansion (May 11, freshly scored by Edison).
    # Picks are domain-disjoint from the existing 9.
    "ca05", "ca08", "ca09",   # photocatalysis, synthesis, biochemistry
    "tc02", "tc06", "tc08",   # MOFs, CO2_reduction, 2D_materials
    "cs02", "cs05", "cs09",   # computational, perovskite, batteries
    # full-30 expansion (May 15, EDISON_2 credits) -- the 7 quota-blocked
    # cells from May 11 are retried because edison_seed is None for them,
    # and the remaining 12 fresh ids below get their first Edison call.
    "ca01", "ca03", "ca06", "ca07",
    "tc04", "tc07", "tc09", "tc10",
    "cs04", "cs06", "cs07", "cs10",
]

# ---------------------------------------------------------------------------
# Question Bank: 30 questions across 3 task types
# ---------------------------------------------------------------------------

BENCHMARK_QUESTIONS = [
    # ── Task A: Cross-Paper Condition Aggregation (CA) ──────────────────
    {
        "id": "ca01", "task": "CA", "domain": "coupling",
        "question": (
            "What catalysts and conditions have been used for C-N coupling "
            "of heteroaryl chlorides? Give specific catalysts, ligands, "
            "solvents, temperatures, and yields from the literature."
        ),
        "askchem_params": {"q": "C-N coupling catalyst palladium conditions", "limit": 50},
    },
    {
        "id": "ca02", "task": "CA", "domain": "coupling",
        "question": (
            "What catalysts, solvents, bases, and temperatures have been "
            "reported for Suzuki-Miyaura cross-coupling of aryl chlorides? "
            "Include specific yields."
        ),
        "askchem_params": {"q": "Suzuki Miyaura coupling aryl chloride catalyst", "limit": 50},
    },
    {
        "id": "ca03", "task": "CA", "domain": "catalysis",
        "question": (
            "What homogeneous catalysts have been used for asymmetric "
            "hydrogenation? List specific catalyst systems, substrates, "
            "ee values, and conditions."
        ),
        "askchem_params": {"q": "asymmetric hydrogenation catalyst", "limit": 50},
    },
    {
        "id": "ca04", "task": "CA", "domain": "electrocatalysis",
        "question": (
            "What electrocatalysts have been reported for CO2 reduction "
            "to CO or formate? Give specific materials, overpotentials, "
            "Faradaic efficiencies, and current densities."
        ),
        "askchem_params": {"q": "electrocatalytic CO2 reduction catalyst Faradaic efficiency", "limit": 50},
    },
    {
        "id": "ca05", "task": "CA", "domain": "photocatalysis",
        "question": (
            "What photocatalysts and conditions have been used for "
            "visible-light-driven water splitting? Report specific "
            "materials, cocatalysts, light sources, and hydrogen evolution rates."
        ),
        "askchem_params": {"q": "photocatalytic water splitting visible light hydrogen evolution", "limit": 50},
    },
    {
        "id": "ca06", "task": "CA", "domain": "polymerization",
        "question": (
            "What catalysts and conditions have been reported for "
            "ring-opening metathesis polymerization (ROMP)? Include "
            "specific initiators, monomers, solvents, and molecular weights."
        ),
        "askchem_params": {"q": "ring-opening metathesis polymerization ROMP catalyst", "limit": 50},
    },
    {
        "id": "ca07", "task": "CA", "domain": "oxidation",
        "question": (
            "What catalytic systems have been used for selective oxidation "
            "of alcohols to aldehydes? Report catalysts, oxidants, solvents, "
            "temperatures, and selectivities."
        ),
        "askchem_params": {"q": "oxidation catalyst selective alcohol", "limit": 50},
    },
    {
        "id": "ca08", "task": "CA", "domain": "synthesis",
        "question": (
            "What conditions have been reported for Heck coupling of "
            "aryl halides with olefins? Give catalysts, ligands, bases, "
            "solvents, and temperatures with yields."
        ),
        "askchem_params": {"q": "Heck coupling palladium aryl", "limit": 50},
    },
    {
        "id": "ca09", "task": "CA", "domain": "biochemistry",
        "question": (
            "What enzyme systems have been used for biocatalytic "
            "synthesis? Report specific enzymes, substrates, products, "
            "and yields or turnover numbers."
        ),
        "askchem_params": {"q": "enzyme biocatalysis synthesis", "limit": 50},
    },
    {
        "id": "ca10", "task": "CA", "domain": "adsorption",
        "question": (
            "What adsorbent materials and conditions have been used "
            "for heavy metal removal from water? Report specific "
            "materials, adsorption capacities, pH, and contact times."
        ),
        "askchem_params": {"q": "adsorption heavy metal removal water capacity", "limit": 50},
    },

    # ── Task B: Temporal Claim Tracking (TC) ────────────────────────────
    {
        "id": "tc01", "task": "TC", "domain": "perovskites",
        "question": (
            "How has the scientific understanding of perovskite degradation "
            "mechanisms evolved over time? Describe the key shifts in the "
            "field's understanding, citing specific findings and years."
        ),
        "askchem_params": {"q": "perovskite degradation mechanism", "limit": 100},
    },
    {
        "id": "tc02", "task": "TC", "domain": "MOFs",
        "question": (
            "How has the understanding of metal-organic framework (MOF) "
            "stability in aqueous environments evolved? Trace the key "
            "findings and when they were reported."
        ),
        "askchem_params": {"q": "metal-organic framework stability", "limit": 100},
    },
    {
        "id": "tc03", "task": "TC", "domain": "batteries",
        "question": (
            "How has the understanding of solid electrolyte interphase (SEI) "
            "formation in lithium-ion batteries evolved over time? "
            "Cite specific discoveries and their years."
        ),
        "askchem_params": {"q": "solid electrolyte interphase SEI lithium battery formation", "limit": 100},
    },
    {
        "id": "tc04", "task": "TC", "domain": "nanomedicine",
        "question": (
            "How has the scientific understanding of nanoparticle-protein "
            "corona formation evolved? Describe shifts in understanding "
            "with dates and citations."
        ),
        "askchem_params": {"q": "nanoparticle protein corona formation mechanism", "limit": 100},
    },
    {
        "id": "tc05", "task": "TC", "domain": "photocatalysis",
        "question": (
            "How has the mechanistic understanding of TiO2 photocatalysis "
            "evolved from its discovery to the present? Cite key findings "
            "and years."
        ),
        "askchem_params": {"q": "TiO2 photocatalysis mechanism", "limit": 100},
    },
    {
        "id": "tc06", "task": "TC", "domain": "CO2_reduction",
        "question": (
            "How has the understanding of CO2 electrochemical reduction "
            "mechanisms and selectivity evolved? Trace the key discoveries "
            "with citations and years."
        ),
        "askchem_params": {"q": "CO2 reduction selectivity mechanism", "limit": 100},
    },
    {
        "id": "tc07", "task": "TC", "domain": "drug_delivery",
        "question": (
            "How has the understanding of nanoparticle drug delivery "
            "systems evolved? Trace the progression of key findings "
            "with citations and years."
        ),
        "askchem_params": {"q": "drug delivery nanoparticle release", "limit": 100},
    },
    {
        "id": "tc08", "task": "TC", "domain": "2D_materials",
        "question": (
            "How has the understanding of defects in graphene and their "
            "effects on electronic properties evolved over time? Trace "
            "key discoveries."
        ),
        "askchem_params": {"q": "graphene defect electronic properties", "limit": 100},
    },
    {
        "id": "tc09", "task": "TC", "domain": "nanotoxicology",
        "question": (
            "How has the scientific understanding of nanoparticle "
            "cytotoxicity mechanisms evolved? Trace shifts in understanding "
            "with dates and citations."
        ),
        "askchem_params": {"q": "nanoparticle cytotoxicity", "limit": 100},
    },
    {
        "id": "tc10", "task": "TC", "domain": "batteries_electrolyte",
        "question": (
            "How has the understanding of lithium battery electrolyte "
            "stability and decomposition evolved? Cite key findings "
            "and the years they appeared."
        ),
        "askchem_params": {"q": "lithium battery electrolyte stability", "limit": 100},
    },

    # ── Task C: Contradiction Surfacing (CS) ────────────────────────────
    {
        "id": "cs01", "task": "CS", "domain": "nanotoxicology",
        "question": (
            "Are silver nanoparticles toxic or safe for biomedical use? "
            "What contradictory findings exist about the antimicrobial "
            "activity versus cytotoxicity of silver nanoparticles?"
        ),
        "askchem_params": {"q": "silver nanoparticle antimicrobial", "limit": 50},
    },
    {
        "id": "cs02", "task": "CS", "domain": "computational",
        "question": (
            "Which DFT functionals are most accurate for chemical "
            "calculations? Are there contradictory results about the "
            "accuracy of different density functional methods?"
        ),
        "askchem_params": {"q": "density functional theory DFT computational", "limit": 50},
    },
    {
        "id": "cs03", "task": "CS", "domain": "MOF_stability",
        "question": (
            "Are metal-organic frameworks stable enough for practical "
            "applications? What conflicting reports exist about MOF "
            "stability under real-world conditions?"
        ),
        "askchem_params": {"q": "metal-organic framework stability", "limit": 50},
    },
    {
        "id": "cs04", "task": "CS", "domain": "solvent_effects",
        "question": (
            "Does the choice of solvent fundamentally alter reaction "
            "mechanisms? What contradictory findings exist about "
            "solvent effects on reaction selectivity and mechanism?"
        ),
        "askchem_params": {"q": "solvent effect reaction mechanism", "limit": 50},
    },
    {
        "id": "cs05", "task": "CS", "domain": "perovskite",
        "question": (
            "Does interface passivation reliably improve perovskite "
            "solar cell stability? What conflicting results exist "
            "about passivation strategies?"
        ),
        "askchem_params": {"q": "perovskite solar cell stability", "limit": 50},
    },
    {
        "id": "cs06", "task": "CS", "domain": "polymer",
        "question": (
            "Is polylactic acid (PLA) truly biodegradable in natural "
            "environments? What contradictory findings exist about PLA "
            "and polymer degradation rates?"
        ),
        "askchem_params": {"q": "polylactic acid PLA degradation", "limit": 50},
    },
    {
        "id": "cs07", "task": "CS", "domain": "graphene",
        "question": (
            "Does nitrogen doping improve or degrade the electrocatalytic "
            "activity of graphene? What conflicting results have been "
            "reported?"
        ),
        "askchem_params": {"q": "nitrogen doped graphene catalysis", "limit": 50},
    },
    {
        "id": "cs08", "task": "CS", "domain": "CO2_reduction",
        "question": (
            "Is copper the best electrocatalyst for CO2 reduction, "
            "or do alternatives outperform it? What contradictory "
            "findings exist about CO2 reduction selectivity?"
        ),
        "askchem_params": {"q": "CO2 reduction selectivity mechanism", "limit": 50},
    },
    {
        "id": "cs09", "task": "CS", "domain": "batteries",
        "question": (
            "Do solid-state electrolytes solve the safety problems of "
            "lithium batteries, or introduce new ones? What conflicting "
            "results exist about electrolyte stability?"
        ),
        "askchem_params": {"q": "lithium battery electrolyte stability", "limit": 50},
    },
    {
        "id": "cs10", "task": "CS", "domain": "nanoparticle_catalysis",
        "question": (
            "Do smaller nanoparticles always have higher catalytic activity? "
            "What contradictory findings exist about the nanoparticle size "
            "effect in catalysis?"
        ),
        "askchem_params": {"q": "nanoparticle size catalytic activity surface", "limit": 50},
    },
]


# ---------------------------------------------------------------------------
# DOI extraction and verification
# ---------------------------------------------------------------------------

# Match the DOI prefix only (registrant/agency segment). The suffix is
# walked character-by-character in extract_dois() because legacy Wiley
# SICI and Elsevier S-series DOIs contain balanced parens and a `;2-X`
# checksum tail that the old `[^\s,;)}\]\"']+` character class would
# truncate. The truncation cost AskChem-Bench measurable accuracy on
# pre-2003 Wiley DOIs (e.g. Littke/Fu 2002 Angew. Chem., Knowles 2002
# Nobel lecture) and on Chinese J. Catal. 2020 papers — see
# https://www.crossref.org/blog/dois-and-matching-regular-expressions/
# for why DOI suffixes are intentionally permissive.
DOI_PATTERN = re.compile(r"10\.\d{4,}/", re.IGNORECASE)


def extract_dois(text: str) -> list[str]:
    """Extract unique DOIs from text, preserving balanced parens and SICI suffixes.

    Walks each candidate DOI character-by-character starting from the
    `10.PREFIX/` anchor, stopping at the first unambiguous terminator
    (whitespace, quotes, bracketed prose markup) while keeping balanced
    `()` pairs intact. The legacy SICI checksum tail (`;2-X`) is also
    preserved — a bare `;` terminates the DOI in modern prose but
    `;2-Y`, `;2-1`, `;2-U` etc. are valid SICI checksums.

    Examples handled correctly:
      - 10.1002/1521-3773(20021115)41:22<4176::AID-ANIE4176>3.0.CO;2-U
      - 10.1002/(SICI)1521-3773(19990315)38:6<838::AID-ANIE838>3.0.CO;2-O
      - 10.1016/S1872-2067(20)63754-8
      - 10.1021/jacs.5c20640                       (modern, unchanged)
      - "see (DOI: 10.X/Y), also Z"                (trailing ')' trimmed)
    """
    seen: set[str] = set()
    out: list[str] = []
    n = len(text)
    for m in DOI_PATTERN.finditer(text):
        start = m.start()
        i = m.end()
        # Single depth counter for both `()` and `<>` — they appear
        # sequentially in SICI DOIs (e.g. `(YYYY)VOL:ISSUE<PAGE::AID-...>`),
        # not nested, so one counter is enough to keep the suffix intact.
        depth = 0
        while i < n:
            ch = text[i]
            if ch.isspace():
                break
            if ch in "(<":
                depth += 1
            elif ch in ")>":
                if depth == 0:
                    break
                depth -= 1
            elif ch in '"\'':
                break
            elif depth == 0 and ch in "]}":
                break
            elif depth == 0 and ch == ",":
                break
            elif depth == 0 and ch == ";":
                # Preserve the SICI checksum tail `;2-X` (single
                # alphanumeric); otherwise treat `;` as prose terminator.
                if i + 2 < n and text[i + 1] == "2" and text[i + 2] == "-":
                    pass
                else:
                    break
            i += 1
        # Trailing punctuation cleanup: drop characters that commonly
        # attach to a DOI in prose (sentence-ending '.', markdown '*',
        # bracket close, etc.) but never appear at the end of a real DOI.
        # We deliberately preserve the SICI suffix because the walker
        # already validated balanced ()/<>.
        doi = text[start:i].rstrip(".,;:*)]}>")
        if not doi:
            continue
        key = doi.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(doi)
    return out


def _crossref_extract_year(message: dict) -> Optional[int]:
    """Pick the earliest published date (online or print) and return its year."""
    candidates = []
    for field in ("published-online", "published-print", "issued", "created"):
        block = message.get(field) or {}
        parts = (block.get("date-parts") or [[]])[0]
        if parts and isinstance(parts[0], int) and 1800 <= parts[0] <= 2100:
            candidates.append(parts[0])
    return min(candidates) if candidates else None


_CROSSREF_CACHE_PATH = Path(__file__).parent / "crossref_cache.json"
_CROSSREF_CACHE: dict = {}
_CROSSREF_CACHE_LOADED = False


def _load_crossref_cache() -> dict:
    global _CROSSREF_CACHE_LOADED
    if _CROSSREF_CACHE_LOADED:
        return _CROSSREF_CACHE
    if _CROSSREF_CACHE_PATH.exists():
        try:
            _CROSSREF_CACHE.update(json.loads(_CROSSREF_CACHE_PATH.read_text()))
        except Exception:
            pass
    _CROSSREF_CACHE_LOADED = True
    return _CROSSREF_CACHE


def _save_crossref_cache() -> None:
    if not _CROSSREF_CACHE_LOADED:
        return
    try:
        _CROSSREF_CACHE_PATH.write_text(json.dumps(_CROSSREF_CACHE, indent=2))
    except Exception:
        pass


def _verify_doi_handle(doi: str) -> bool:
    """Best-effort existence check via the doi.org handle resolver.

    Returns True iff the DOI redirects to a publisher landing page.
    Used as a fallback when CrossRef's metadata API returns 404 — this
    happens for:

      * arXiv DataCite DOIs  (10.48550/arXiv.NNNN.NNNNN; CrossRef does
        not index arXiv, but doi.org redirects to arxiv.org/abs/...).
      * DOIs registered with mEDRA / DataCite / non-CrossRef agencies
        (e.g. small/non-English publishers like the Iranian J. Pharm.
        Res., Brieflands).
      * Legacy Wiley/Elsevier SICI DOIs that were issued before the
        publisher's bulk CrossRef migration.

    HTTP redirects (301/302/303/307/308) indicate the handle resolved;
    we deliberately do NOT follow them so that a publisher returning
    403/410 from the landing page (with no Referer) still counts as
    "DOI exists" — the handle resolution succeeded, and the paper was
    indexed by the publisher at some point.
    """
    try:
        resp = requests.get(
            f"https://doi.org/{doi}",
            headers={
                "User-Agent": "AskChem-Bench/1.0 (mailto:contact@askchem.org)",
            },
            allow_redirects=False,
            timeout=10,
        )
        return resp.status_code in (301, 302, 303, 307, 308)
    except Exception:
        return False


def verify_doi_crossref(doi: str, use_cache: bool = True) -> dict:
    """Check a DOI against CrossRef, with a doi.org handle fallback.

    Returns existence status, title, and Phase 1b paper-quality fields
    (is-referenced-by-count, year, abstract, type). When CrossRef
    doesn't have the DOI (404 or non-200), we hit the doi.org handle
    resolver as a secondary check and mark ``exists=True`` with empty
    metadata if it resolves. Downstream consumers (relevance judge,
    citation-count means, etc.) gate on ``isinstance(...)`` checks of
    the typed fields, so fallback-verified DOIs raise DOI existence
    without polluting the relevance / citation-impact aggregates.

    Cache key is the lowercase DOI; ``verified_via`` records which
    path resolved the DOI ("crossref" or "doi.org") for downstream
    diagnostics.
    """
    cache = _load_crossref_cache() if use_cache else {}
    key = doi.lower().strip()
    if use_cache and key in cache:
        return cache[key]
    try:
        resp = requests.get(
            f"{CROSSREF_API}/{doi}",
            headers=CROSSREF_HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get("message", {})
            titles = msg.get("title", [])
            title = titles[0] if titles else ""
            abstract = msg.get("abstract", "") or ""
            # CrossRef abstracts often arrive wrapped in <jats:p>…</jats:p>
            # XML; strip the tags for downstream consumption.
            if abstract:
                abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
                abstract = re.sub(r"\s+", " ", abstract)[:2000]
            result = {
                "exists": True,
                "title": title,
                "citation_count": msg.get("is-referenced-by-count", 0),
                "year": _crossref_extract_year(msg),
                "abstract": abstract,
                "type": msg.get("type", ""),
                "verified_via": "crossref",
            }
        else:
            # CrossRef gap — confirm via doi.org handle before declaring
            # the DOI fabricated. arXiv DataCite DOIs and pre-2003 Wiley
            # SICI DOIs commonly land here.
            if _verify_doi_handle(doi):
                result = {
                    "exists": True,
                    "title": "",
                    "citation_count": None,
                    "year": None,
                    "abstract": "",
                    "type": "",
                    "verified_via": "doi.org",
                }
            else:
                result = {
                    "exists": False,
                    "title": "",
                    "citation_count": None,
                    "year": None,
                    "abstract": "",
                    "type": "",
                    "verified_via": None,
                }
    except Exception:
        # Network/timeout error against CrossRef — still try doi.org so
        # one flaky CrossRef call doesn't drop a real DOI to exists=False.
        if _verify_doi_handle(doi):
            result = {
                "exists": True,
                "title": "",
                "citation_count": None,
                "year": None,
                "abstract": "",
                "type": "",
                "verified_via": "doi.org",
            }
        else:
            result = {
                "exists": False,
                "title": "",
                "citation_count": None,
                "year": None,
                "abstract": "",
                "type": "",
                "verified_via": None,
            }
    if use_cache:
        cache[key] = result
    return result


def tokenize(text: str) -> set[str]:
    """Simple whitespace + lowercased tokenizer, dropping short tokens."""
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower())}


# ── Paper-relevance LLM judge (Phase 1b, May 2026) ────────────────────────
# Replaces the broken title-only Jaccard with a chemistry-aware
# gemini-3.1-pro-preview judge that sees the question + paper title +
# abstract and scores 0/1/2/3.
PORTKEY_GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions"
PORTKEY_PROVIDER = "@vertexai-gemini-kc119-2"
PAPER_RELEVANCE_MODEL = "gemini-3.1-pro-preview"

_PAPER_RELEVANCE_CACHE_PATH = Path(__file__).parent / "llm_relevance_cache.json"
_PAPER_RELEVANCE_CACHE: dict = {}
_PAPER_RELEVANCE_CACHE_LOADED = False

_PAPER_RELEVANCE_PROMPT_PAPER = """You are a chemistry-domain relevance judge.

Given a research QUESTION and a paper described by its TITLE and ABSTRACT, score how directly this paper answers the question. Output a single JSON object on one line:

{{"score": 0|1|2|3, "rationale": "<one short sentence>"}}

Rubric:
  3 = DIRECTLY ANSWERS. The paper is primary evidence for the question.
      Example: question "What catalysts have been reported for Suzuki coupling of aryl chlorides?" + paper reporting a new Pd-catalyst system with yields for that exact reaction.

  2 = ON TOPIC. The paper covers the same specific sub-area, but the abstract alone doesn't confirm it directly answers the question.
      Example: question "Suzuki coupling of aryl chlorides" + paper on Pd-catalysed cross-coupling of aryl bromides with a brief mention of chlorides.

  1 = LOOSELY RELATED. Same broader field, but doesn't address the specific question.
      Example: question "Suzuki coupling of aryl chlorides" + paper on a different Pd-catalysed C-C bond formation (e.g. Heck, Negishi).

  0 = IRRELEVANT. Different field, homonym usage, or off-topic.
      Example: question "Suzuki coupling" + paper on spin-orbit coupling in solid-state physics.

When unsure between 1 and 2, prefer 1. When unsure between 2 and 3, prefer 2. Output JSON ONLY - no surrounding prose, no markdown fences.

QUESTION: {question}

TITLE: {title}

ABSTRACT: {abstract}
"""

_PAPER_RELEVANCE_PROMPT_CLAIM = """You are a chemistry-domain relevance judge.

You are scoring how directly a CLAIM extracted from a paper answers a research QUESTION. The paper's TITLE and ABSTRACT are provided as background context, but you are judging the CLAIM. Output a single JSON object on one line:

{{"score": 0|1|2|3, "rationale": "<one short sentence>"}}

Rubric (apply to the CLAIM, using TITLE/ABSTRACT only as context):
  3 = DIRECTLY ANSWERS. The claim is primary evidence for the question.
      Example: question "What catalysts have been reported for Suzuki coupling of aryl chlorides?" + claim "Pd-PEPPSI-IPent catalysed Suzuki coupling of aryl chlorides in 92% yield."

  2 = ON TOPIC. The claim is in the same specific sub-area as the question but doesn't directly answer it.
      Example: question "Suzuki coupling of aryl chlorides" + claim about Pd-catalysed coupling of aryl bromides.

  1 = LOOSELY RELATED. Same broader field, but the claim doesn't address the specific question.
      Example: question "Suzuki coupling of aryl chlorides" + claim about a Heck reaction.

  0 = IRRELEVANT. Different field, homonym, or off-topic.
      Example: question "Suzuki coupling" + claim about spin-orbit coupling.

When unsure between 1 and 2, prefer 1. When unsure between 2 and 3, prefer 2. Output JSON ONLY - no surrounding prose, no markdown fences.

QUESTION: {question}

CLAIM(S) EXTRACTED FROM PAPER:
{claim_text}

PAPER TITLE: {title}

PAPER ABSTRACT: {abstract}
"""


def _relevance_cache_key(qid: str, doi: str, claim_text: Optional[str] = None) -> str:
    """Cache key for the paper-relevance judge.

    Paper-level keys stay as ``"{qid}|{doi}"`` (back-compat with the existing
    1654 cached entries). When a claim_text is supplied (claim-aware judge for
    AskChem ``unified``), the key gets an extra ``|c:<hash>`` suffix so the
    two judgments live in separate cache slots and we don't silently reuse a
    paper-only score for a claim-grounded request.
    """
    base = f"{qid}|{doi.lower().strip()}"
    if claim_text:
        import hashlib as _hashlib

        h = _hashlib.sha1(claim_text.encode("utf-8")).hexdigest()[:12]
        return f"{base}|c:{h}"
    return base


def _load_paper_relevance_cache() -> dict:
    global _PAPER_RELEVANCE_CACHE_LOADED
    if _PAPER_RELEVANCE_CACHE_LOADED:
        return _PAPER_RELEVANCE_CACHE
    if _PAPER_RELEVANCE_CACHE_PATH.exists():
        try:
            _PAPER_RELEVANCE_CACHE.update(json.loads(_PAPER_RELEVANCE_CACHE_PATH.read_text()))
        except Exception:
            pass
    _PAPER_RELEVANCE_CACHE_LOADED = True
    return _PAPER_RELEVANCE_CACHE


def _save_paper_relevance_cache() -> None:
    if not _PAPER_RELEVANCE_CACHE_LOADED:
        return
    try:
        _PAPER_RELEVANCE_CACHE_PATH.write_text(json.dumps(_PAPER_RELEVANCE_CACHE, indent=2))
    except Exception:
        pass


_SCORE_REGEX = re.compile(r'"score"\s*:\s*(\d)')


def _parse_relevance_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction. Pro frequently truncates the JSON
    mid-rationale because reasoning tokens count against max_completion
    _tokens; when strict + lenient JSON parsing both fail we fall back to
    a regex on the leading ``"score": N`` field so we at least keep the
    rating even without a clean rationale.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip("`").strip()
        try:
            return json.loads(s)
        except Exception:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    # Last resort: regex out the leading score. Rationale is lost but the
    # 0-3 rating is the only field we aggregate over.
    m = _SCORE_REGEX.search(text)
    if m:
        return {"score": int(m.group(1)), "rationale": "(truncated; recovered from prefix)"}
    return None


def score_paper_relevance(
    question_id: str,
    doi: str,
    question: str,
    title: str,
    abstract: str,
    use_cache: bool = True,
    timeout: int = 60,
    retries: int = 3,
    claim_text: Optional[str] = None,
) -> dict:
    """Score paper or claim relevance via gemini-3.1-pro-preview.

    Returns ``{'score': int 0-3, 'rationale': str}`` or
    ``{'score': None, 'error': str}`` on persistent failure. Cached to
    llm_relevance_cache.json. When ``claim_text`` is supplied the judge
    scores the CLAIM (with title/abstract as context) and the cache key
    includes a claim-hash suffix so paper-only and claim-grounded
    judgments stay in separate cache slots.
    """
    cache = _load_paper_relevance_cache() if use_cache else {}
    key = _relevance_cache_key(question_id, doi, claim_text)
    # Return cache hits only when the previous run produced a real score.
    # Cached failure entries (score=None) are retried on next pass so a
    # transient network blip or token-budget truncation doesn't poison
    # the row forever.
    if use_cache and key in cache and cache[key].get("score") is not None:
        return cache[key]

    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        result = {"score": None, "error": "PORTKEY_API_KEY not set"}
        if use_cache:
            cache[key] = result
        return result

    has_claim = bool(claim_text and claim_text.strip())
    if has_claim:
        prompt = _PAPER_RELEVANCE_PROMPT_CLAIM.format(
            question=question,
            title=title or "(no title)",
            abstract=(abstract or "(not available)")[:1500],
            claim_text=claim_text.strip()[:2000],
        )
    else:
        prompt = _PAPER_RELEVANCE_PROMPT_PAPER.format(
            question=question,
            title=title or "(no title)",
            abstract=(abstract or "(not available)")[:1500],
        )

    body = {
        "model": PAPER_RELEVANCE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # Pro burns ~300-1500 tokens on internal reasoning before emitting
        # the JSON body, so 512 was clipping the rationale on most calls.
        # 2048 lets the rationale finish cleanly and keeps the per-call
        # cost bounded to about $0.005 worst-case.
        "max_completion_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "x-portkey-api-key": api_key,
        "x-portkey-provider": PORTKEY_PROVIDER,
        "Content-Type": "application/json",
    }
    last_err = ""
    for attempt in range(retries):
        try:
            r = requests.post(PORTKEY_GATEWAY, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            last_err = f"network: {e}"
            time.sleep(min(2 ** attempt, 20))
            continue
        if r.status_code != 200:
            last_err = f"http {r.status_code}: {r.text[:160]}"
            time.sleep(min(2 ** attempt, 20))
            continue
        try:
            resp = r.json()
        except Exception as e:
            last_err = f"json: {e}"
            time.sleep(min(2 ** attempt, 20))
            continue
        choices = resp.get("choices") or []
        content = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
        parsed = _parse_relevance_json(content)
        if not parsed or "score" not in parsed:
            last_err = f"parse: {content[:160]!r}"
            time.sleep(min(2 ** attempt, 20))
            continue
        score = parsed.get("score")
        try:
            score = int(score)
        except Exception:
            last_err = f"bad score: {score!r}"
            time.sleep(min(2 ** attempt, 20))
            continue
        if score not in (0, 1, 2, 3):
            last_err = f"out-of-range score: {score}"
            time.sleep(min(2 ** attempt, 20))
            continue
        result = {
            "score": score,
            "rationale": str(parsed.get("rationale", ""))[:300],
            "model": PAPER_RELEVANCE_MODEL,
            "judged_with_claim": has_claim,
        }
        if use_cache:
            cache[key] = result
        return result
    result = {"score": None, "error": last_err, "judged_with_claim": has_claim}
    if use_cache:
        cache[key] = result
    return result


def title_relevance(crossref_title: str, context: str) -> float:
    """Jaccard similarity between CrossRef title tokens and answer context."""
    t1 = tokenize(crossref_title)
    t2 = tokenize(context)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def verify_dois_in_text(
    text: str, question: str = "", sleep_between: float = 0.3
) -> dict:
    """Extract and verify all DOIs in a text block.

    Captures the full CrossRef enrichment (existence, title, citation count,
    year, abstract, type) so the bench output is self-contained and
    ``compute_metrics`` can compute paper-quality aggregates without a
    separate backfill pass. Repeats are O(1) via the on-disk CrossRef cache.
    """
    dois = extract_dois(text)
    results = {}
    for doi in dois:
        info = verify_doi_crossref(doi)
        relevance = 0.0
        if info["exists"] and info["title"]:
            context = text + " " + question
            relevance = title_relevance(info["title"], context)
        results[doi] = {
            "exists": info["exists"],
            "crossref_title": info["title"],
            "relevance": round(relevance, 3),
            "citation_count": info.get("citation_count"),
            "year": info.get("year"),
            "abstract": info.get("abstract", ""),
            "type": info.get("type", ""),
        }
        time.sleep(sleep_between)
    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
# Cap digit length to 4 + require non-word context on both sides so that
# claim-ID hashes like "00100k" don't masquerade as temperatures.
TEMP_PATTERN = re.compile(
    r"(?<![\w])\d{1,4}\s*°\s*C|(?<![\w])\d{1,4}\s*K(?![\w])",
    re.IGNORECASE,
)
YIELD_PATTERN = re.compile(r"\d+\.?\d*\s*%")
QUANTITY_PATTERN = re.compile(
    r"\d+\.?\d*\s*(?:nm|μm|mm|cm|mV|eV|GPa|MPa|kPa|mol|mmol|"
    r"mg|μg|mL|μL|ppm|ppb|mA|μA|kJ|kcal|Hz|MHz|GHz|"
    r"cm-1|cm⁻¹|wt%|vol%|M\b|mM\b)"
)
CONTRADICTION_MARKERS = re.compile(
    r"\b(?:however|in contrast|contradicts?|conflicting|"
    r"disagree|debat|on the other hand|conversely|"
    r"opposite|inconsisten|paradox|contrasting)\b",
    re.IGNORECASE,
)

# A sentence boundary: . ! or ? followed by whitespace and a capital letter or
# bracket. Matches in the middle of a string (not lookbehind for ^) so the very
# first sentence is captured implicitly by splitting on these boundaries.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])")
# Inline DOI in answer prose (matches the same shape extract_dois finds).
_DOI_INLINE = re.compile(r"10\.\d{4,9}/[^\s\]\)\"';,]+", re.IGNORECASE)
# Various inline citation markers an answer head might emit:
#   [12], [4-7], [Paper 3], [DOI: 10.x/y], (Smith et al. 2019), (Lee, 2020),
#   (manolikakes2008anefficientsilanepromoted pages 5-6)  ← paperqa3 / Edison
_CITATION_MARKER = re.compile(
    r"\[\s*\d+(?:\s*[\-\u2013,]\s*\d+)*\s*\]"     # [12], [4-7], [4,5,7]
    r"|\[\s*(?:Paper|DOI|Ref)\s*[:#\d]"          # [Paper 3], [DOI: ...], [Ref 1
    r"|\(\s*[A-Z][\w'\-]+(?:\s+et\s+al\.?)?\s*[,\s]\s*(?:19|20)\d{2}"   # (Smith, 2019) / (Lee et al. 2020)
    r"|\(\s*[a-z][\w\-]+\s+pages\s+\d",          # paperqa3-style (authorYYYYkeyword pages 5-6)
    re.IGNORECASE,
)
# Section-heading regex. Quantitative-token attribution is scoped to a
# section: a citation marker that appears anywhere in the section makes
# every subsequent quantitative token in that section count as grounded.
# Without this carry-forward, the metric punishes the common style where
# a system cites the source once in the section's lead sentence and then
# enumerates yields / conditions / temps in following bullets or rows.
# Headings we treat as section boundaries:
#   * Markdown headings (# / ## / ### ... Title)
#   * Numbered section openers ("1. Title", "1.1 Title", "1) Title")
#   * All-caps lines >= 5 chars
#   * Markdown horizontal rules (--- / ===)
_SECTION_HEADER = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s+\S"               # # Title / ## Title / ...
    r"|\d+(?:\.\d+)*[\.\)]\s+[A-Z]"  # 1. Title / 1.1 Title / 1) Title
    r"|[A-Z][A-Z\s\-]{4,}\s*$"   # ALL CAPS LINE
    r"|-{3,}\s*$|={3,}\s*$"      # --- / === rules
    r")"
)


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter (no NLTK dep). Good enough for citation binding.

    Splits on:
      * standard ``[.!?]<ws><Capital|bracket>`` boundaries,
      * newlines that start a markdown bullet (``- `` / ``* `` / ``1. ``),
      * newlines that start a markdown table row (lines beginning with
        ``|``) so each table row becomes its own "sentence" — without
        this, AskChem-style table answers collapse to one or two
        gigantic sentences and quantitative tokens stop being
        per-row-attributable.
    """
    if not text:
        return []
    # First pass: split on hard sentence boundaries.
    chunks = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    for chunk in chunks:
        # Second pass: also split on lines that look like list items or
        # table rows. Keeps prose chunks intact (no newline → no extra
        # split).
        for line in re.split(r"\n(?=\s*(?:\||[*\-+]\s|\d+[\.\)]\s))", chunk):
            if line.strip():
                out.append(line)
    return out


def _sentence_is_grounded(sentence: str) -> bool:
    """A sentence counts as 'grounded' if it carries a citation signal."""
    return bool(_DOI_INLINE.search(sentence) or _CITATION_MARKER.search(sentence))


def _count_quantitative_tokens(text: str) -> int:
    """How many temperature/yield/quantity tokens appear in this chunk of text."""
    return (
        len(TEMP_PATTERN.findall(text))
        + len(YIELD_PATTERN.findall(text))
        + len(QUANTITY_PATTERN.findall(text))
    )


def _split_sections(answer: str) -> list[str]:
    """Split the answer into sections at section-header lines.

    A header line itself is treated as the *start* of the next section, so
    "## 2. Foo" carries any prose on the same logical block into the new
    section and resets the citation context (so a citation in section 1
    does NOT carry over to specifics in section 2).
    """
    if not answer:
        return []
    sections: list[list[str]] = []
    current: list[str] = []
    for line in answer.splitlines():
        if _SECTION_HEADER.match(line):
            if current:
                sections.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def compute_grounded_specificity(answer: str) -> int:
    """Count quantitative tokens (yields/temps/units) that fall in a
    section where a citation marker has already appeared.

    A citation is "live" from the first sentence of the section that
    contains it through the end of that section. Section boundaries
    (markdown headings, numbered "1. Title" openers, ALL-CAPS lines,
    horizontal rules) reset the context. This captures the common style
    where the source is cited once in the lead sentence and the
    specifics (yields, temps, mol%) are enumerated in subsequent
    sentences, bullets, or table rows without re-citing — while still
    refusing to credit a wall of numbers that never names a source.

    Robust to two failure modes:
      * llm_alone-style: numbers in standalone tables with no citation.
        Those land in sections whose citation never fires; they don't
        count.
      * over-strict per-sentence binding: NotebookLM / AskChem / Paperclip
        all cite once and elaborate; the old per-sentence rule scored
        them ~0 even when every specific was attributable to a cited
        paper.
    """
    total = 0
    for section in _split_sections(answer):
        cite_live = False
        for sentence in _split_sentences(section):
            if not cite_live and _sentence_is_grounded(sentence):
                cite_live = True
            if cite_live:
                total += _count_quantitative_tokens(sentence)
    return total


def compute_grounded_specificity_strict(answer: str) -> int:
    """Legacy per-sentence variant kept for diagnostics / ablation.

    Quantitative tokens count only when the SAME sentence carries the
    citation marker. Stricter than ``compute_grounded_specificity`` and
    therefore far more sensitive to citation style — used to be the
    primary metric and is retained so we can audit the section-scoped
    vs. sentence-scoped delta on demand.
    """
    total = 0
    for sentence in _split_sentences(answer):
        if _sentence_is_grounded(sentence):
            total += _count_quantitative_tokens(sentence)
    return total


def compute_metrics(
    answer: str,
    verified_dois: dict,
    task: str,
    edison_dois: Optional[set] = None,
) -> dict:
    """Compute all quantitative metrics for a single answer.

    ``edison_dois`` is the set of CrossRef-existing DOIs cited by Edison
    Scientific for the **same question**. When provided, we record
    ``edison_overlap_rate`` = |own_dois ∩ edison_dois| / |edison_dois|
    (recall against the Edison baseline). When ``None`` (e.g. this *is*
    the Edison cell, or the bench was run before Edison existed) the
    field is omitted from the metrics dict so the aggregate ignores it.
    """
    dois_list = list(verified_dois.keys())
    n_dois = len(dois_list)
    n_exist = sum(1 for v in verified_dois.values() if v["exists"])
    n_relevant = sum(
        1 for v in verified_dois.values()
        if v["exists"] and v["relevance"] >= 0.15
    )

    doi_existence_rate = n_exist / n_dois if n_dois else 0.0
    doi_relevance_rate = n_relevant / n_exist if n_exist else 0.0
    citation_density = n_exist

    # Legacy specificity: raw regex count over the whole answer. Kept for
    # backward-compat with older runs but no longer surfaced as headline.
    specificity = _count_quantitative_tokens(answer)
    # New citation-anchored specificity (May 2026). Only counts numbers
    # that live in the same sentence as a citation marker, so honest "no
    # value reported" abstentions don't lose to confident fabrications.
    grounded_specificity = compute_grounded_specificity(answer)

    # ── Paper-quality aggregates (Phase 1b, May 2026) ───────────────────────
    # Driven entirely by the per-DOI enrichment in verified_dois (citation_count,
    # year, llm_relevance). All are NaN-safe (0 when no DOIs / no enrichment).
    cited_with_count = [
        v["citation_count"] for v in verified_dois.values()
        if v.get("exists") and isinstance(v.get("citation_count"), (int, float))
    ]
    cited_with_year = [
        (v["citation_count"], v["year"]) for v in verified_dois.values()
        if v.get("exists") and isinstance(v.get("citation_count"), (int, float))
        and isinstance(v.get("year"), int)
    ]
    cited_with_relevance = [
        v["llm_relevance"] for v in verified_dois.values()
        if v.get("exists") and isinstance(v.get("llm_relevance"), (int, float))
    ]

    if cited_with_count:
        cc_sorted = sorted(cited_with_count)
        mid = len(cc_sorted) // 2
        citation_count_mean = sum(cited_with_count) / len(cited_with_count)
        citation_count_median = (
            cc_sorted[mid] if len(cc_sorted) % 2 == 1
            else (cc_sorted[mid - 1] + cc_sorted[mid]) / 2
        )
        high_impact_rate = (
            sum(1 for c in cited_with_count if c >= 100) / len(cited_with_count)
        )
    else:
        citation_count_mean = 0.0
        citation_count_median = 0.0
        high_impact_rate = 0.0

    # "Recent high impact": >= 50 citations AND published in the last 5
    # calendar years. Catches papers that punch above their weight relative
    # to their age (a 2024 paper with 80 citations is more impactful than a
    # 2010 paper with 80 citations).
    from datetime import datetime as _dt
    _current_year = _dt.utcnow().year
    if cited_with_year:
        recent_high_impact_rate = (
            sum(1 for c, y in cited_with_year if c >= 50 and y >= _current_year - 5)
            / len(cited_with_year)
        )
    else:
        recent_high_impact_rate = 0.0

    # ``paper_relevance_*`` is left as None when no cited DOI has been
    # scored by the LLM judge yet. This signals to the UI that the metric
    # is pending (so the row renders an em-dash) rather than implying that
    # the system retrieved zero-relevance papers, which would penalise any
    # unscored mode unfairly. Once backfill_paper_relevance.py runs over
    # the new DOIs the means will be populated.
    if cited_with_relevance:
        paper_relevance_mean = round(
            sum(cited_with_relevance) / len(cited_with_relevance), 3
        )
        paper_relevance_high_rate = round(
            sum(1 for r in cited_with_relevance if r >= 2) / len(cited_with_relevance),
            3,
        )
    else:
        paper_relevance_mean = None
        paper_relevance_high_rate = None

    own_exists_dois = {
        d.lower() for d, v in verified_dois.items()
        if v.get("exists")
    }
    if edison_dois is None:
        edison_overlap_rate: Optional[float] = None
        edison_overlap_count: Optional[int] = None
    else:
        edison_lower = {d.lower() for d in edison_dois}
        n_overlap = len(own_exists_dois & edison_lower)
        edison_overlap_count = n_overlap
        edison_overlap_rate = (
            round(n_overlap / len(edison_lower), 3) if edison_lower else 0.0
        )

    metrics = {
        "dois_cited": n_dois,
        "dois_exist": n_exist,
        "dois_relevant": n_relevant,
        "doi_existence_rate": round(doi_existence_rate, 3),
        "doi_relevance_rate": round(doi_relevance_rate, 3),  # legacy Jaccard
        "citation_density": citation_density,
        "specificity_score": specificity,  # legacy raw-regex count
        "grounded_specificity": grounded_specificity,
        "grounded_specificity_strict": compute_grounded_specificity_strict(answer),
        "citation_count_mean": round(citation_count_mean, 1),
        "citation_count_median": round(citation_count_median, 1),
        "high_impact_rate": round(high_impact_rate, 3),
        "recent_high_impact_rate": round(recent_high_impact_rate, 3),
        "paper_relevance_mean": paper_relevance_mean,
        "paper_relevance_high_rate": paper_relevance_high_rate,
        "edison_overlap_rate": edison_overlap_rate,
        "edison_overlap_count": edison_overlap_count,
        "n_papers_with_citation_count": len(cited_with_count),
        "n_papers_with_relevance": len(cited_with_relevance),
    }

    if task == "TC":
        years = [int(y) for y in YEAR_PATTERN.findall(answer)]
        unique_years = sorted(set(years))
        metrics["years_mentioned"] = len(unique_years)
        metrics["temporal_span"] = (
            max(unique_years) - min(unique_years) if len(unique_years) >= 2 else 0
        )

    if task == "CS":
        markers = CONTRADICTION_MARKERS.findall(answer)
        metrics["contradiction_markers"] = len(markers)

    return metrics


# ---------------------------------------------------------------------------
# LLM + AskChem pipeline
# ---------------------------------------------------------------------------

def _local_search(params: dict) -> tuple[list[dict], int]:
    """Search the local chemtree.db directly (bypass remote API)."""
    from askchem.db import search_claims
    result = search_claims(
        query=params.get("q", ""),
        claim_type=params.get("claim_type"),
        view=params.get("view"),
        limit=int(params.get("limit", 50)),
    )
    return result.get("results", []), result.get("total", 0)


def query_askchem(params: dict) -> tuple[list[dict], int]:
    """Query AskChem search API and return (claims, total)."""
    if ASKCHEM_API == "local":
        return _local_search(params)
    resp = requests.get(f"{ASKCHEM_API}/search", params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", []), data.get("total", 0)


# ── Unified mode: LLM rewriter + concurrent /api/search + diversify ───────
#
# We replaced the old `strict_grounded` and `retrieval_assisted` modes with a
# single canonical "unified" pipeline. The architectural insight that drove
# the consolidation: `search_claims` in `src/askchem/db.py` is already the
# hybrid retriever (FTS + paper-level + tree-recall + vector all fused via
# RRF), and `/api/search/grouped` is a presentation wrapper over the same
# function. There is only ONE retrieval engine — `/api/search` — and the
# legacy split was an artefact of how the bench tested two prompt strategies
# over that single engine.
#
# The remaining variation worth testing is purely retrieval-side: long
# natural-language questions sent verbatim to `/api/search` time out (504
# at nginx after 30 s), so we keep an LLM rewriter that produces 3-4 short
# keyword sub-queries, run them concurrently, and merge.

_LLM_REWRITER_CACHE_PATH = Path(__file__).parent / "llm_rewriter_cache.json"


def _load_rewriter_cache() -> dict:
    if _LLM_REWRITER_CACHE_PATH.exists():
        try:
            return json.loads(_LLM_REWRITER_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_rewriter_cache(cache: dict) -> None:
    try:
        _LLM_REWRITER_CACHE_PATH.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False)
        )
    except Exception:
        pass


_REWRITER_TASK_HINT = {
    "CA": (
        "Cross-Paper Aggregation: each sub-query should target a different "
        "catalyst class, material family, or experimental dimension so the "
        "merged claim pool spans many systems."
    ),
    "TC": (
        "Temporal Tracking: each sub-query should probe a different "
        "mechanism, era, or aspect of how the field's understanding "
        "shifted."
    ),
    "CS": (
        "Contradiction Surfacing: each sub-query should probe a different "
        "viewpoint, condition, or material where conflicting results have "
        "been reported."
    ),
}


_REWRITER_PROMPT = """You are a chemistry literature search query rewriter.

The user has a research question. AskChem indexes chemistry claims and exposes a hybrid retrieval endpoint at /api/search (FTS + paper-level + taxonomy + vector fused via RRF). Your job is to rewrite the question into 3-4 SHORT KEYWORD sub-queries so the retriever can surface relevant claims with breadth.

Hard rules:
- Each sub-query is a keyword bag of 4-10 tokens (NOT a full sentence).
- Long natural-language queries time out the retriever, so brevity is mandatory.
- Sub-queries must cover DIFFERENT facets so the merged result has breadth, not redundancy.
- Use canonical chemistry terminology ("Suzuki Miyaura cross coupling aryl chloride", not "the famous Pd coupling reaction").
- Do not invent yields, dates, or paper findings - the retriever will surface those.

Task type: {task}
{task_hint}

Question: {question}

Return a single JSON object with this exact shape (no markdown fences):
{{"sub_queries": ["...", "...", "..."], "rationale": "<one short sentence>"}}"""


def _llm_rewrite_query(question: str, task: str, question_id: str) -> dict:
    """LLM-rewrite a research question into 3-4 short keyword sub-queries.

    Returns ``{"sub_queries": [str, ...], "rationale": str}``. Cached to
    ``scripts/llm_rewriter_cache.json`` keyed by ``question_id``. The cache
    survives across runs so the rewriter cost is paid once per question.

    Falls back gracefully (empty ``sub_queries``) when the LLM call fails
    or no ``OPENAI_API_KEY`` is set; the dispatcher then uses the seed
    keyword query baked into the question record.
    """
    cache = _load_rewriter_cache()
    cached = cache.get(question_id) or {}
    if cached.get("sub_queries"):
        return cached

    if not os.environ.get("OPENAI_API_KEY"):
        return {"sub_queries": [], "rationale": "OPENAI_API_KEY not set"}

    prompt = _REWRITER_PROMPT.format(
        task=task,
        task_hint=_REWRITER_TASK_HINT.get(task, ""),
        question=question,
    )

    try:
        client = OpenAI()
        resp = _call_llm(client, [
            {
                "role": "system",
                "content": (
                    "You are a precise chemistry search query rewriter. "
                    "Output strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ], max_tokens=400)
    except Exception as e:
        print(f"        LLM rewriter failed: {e}")
        return {"sub_queries": [], "rationale": f"error: {e}"}

    text = (resp or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        # Salvage attempt: find the first {...} block in the response.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except Exception:
                parsed = {}
        else:
            parsed = {}

    sub_queries = parsed.get("sub_queries") if isinstance(parsed, dict) else None
    if not isinstance(sub_queries, list):
        sub_queries = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for sq in sub_queries:
        norm = " ".join(str(sq).split())
        if not norm or norm.lower() in seen:
            continue
        seen.add(norm.lower())
        cleaned.append(norm)
    cleaned = cleaned[:UNIFIED_MAX_SUB_QUERIES]
    rationale = str((parsed or {}).get("rationale", ""))[:300]

    rec = {"sub_queries": cleaned, "rationale": rationale}
    cache[question_id] = rec
    _save_rewriter_cache(cache)
    return rec


def _query_one_search(
    query_text: str, limit: int = UNIFIED_LIMIT_PER_QUERY
) -> tuple[list[dict], int]:
    """One /api/search call with a tight timeout. Returns ``([], 0)`` on
    failure so a single bad sub-query doesn't take down the whole unified
    retrieval."""
    if ASKCHEM_API == "local":
        return _local_search({"q": query_text, "limit": limit})
    try:
        resp = requests.get(
            f"{ASKCHEM_API}/search",
            params={"q": query_text, "limit": limit},
            timeout=UNIFIED_QUERY_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []), data.get("total", 0)
    except Exception as e:
        print(f"        /api/search failed for {query_text!r}: {e}")
        return [], 0


def query_askchem_unified(q: dict) -> tuple[list[dict], dict]:
    """Concurrent multi-query retrieval against /api/search.

    Pipeline:
      1. LLM rewriter turns the question into 3-4 short keyword sub-queries.
      2. Concurrent /api/search calls (ThreadPool of ``UNIFIED_WORKERS``,
         ``UNIFIED_QUERY_TIMEOUT`` s per call).
      3. Merge by claim_id, keeping first-seen ordering.
      4. Diversify to ``UNIFIED_MAX_CLAIMS`` claims with
         ``UNIFIED_MAX_PER_SOURCE`` per source via ``select_diverse_claims``.

    Returns ``(diversified_claims, retrieval_meta)``.
    """
    rewrite = _llm_rewrite_query(q["question"], q["task"], q["id"])
    sub_queries = list(rewrite.get("sub_queries") or [])
    if not sub_queries:
        # Fall back to the canonical seed query stamped on the question
        # record so we still produce something usable when the rewriter
        # fails or no API key is set.
        fallback = (q.get("askchem_params") or {}).get("q") or q["question"][:80]
        sub_queries = [fallback]

    query_stats: list[dict] = []
    merged: list[dict] = []
    seen: set = set()

    with ThreadPoolExecutor(max_workers=UNIFIED_WORKERS) as pool:
        future_to_query = {
            pool.submit(_query_one_search, qt): qt for qt in sub_queries
        }
        for fut in as_completed(future_to_query):
            qt = future_to_query[fut]
            try:
                claims, total = fut.result()
            except Exception as e:
                claims, total = [], 0
                print(f"        /api/search worker failed for {qt!r}: {e}")
            query_stats.append({
                "query": qt,
                "total": total,
                "returned": len(claims),
            })
            for claim in claims:
                key = claim.get("claim_id") or (
                    claim.get("source_doi"),
                    claim.get("claim_type"),
                    claim.get("verbatim_quote"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(claim)

    diversified = select_diverse_claims(
        merged, UNIFIED_MAX_CLAIMS, UNIFIED_MAX_PER_SOURCE
    )
    meta = {
        "total_claims": len(merged),
        "diversified_claims": len(diversified),
        "sub_queries": sub_queries,
        "rewriter_rationale": rewrite.get("rationale", ""),
        "queries": query_stats,
    }
    return diversified, meta


def paperclip_enabled() -> bool:
  """True when PAPERCLIP key + Python 3.10+ runner are available."""
  key = os.environ.get("PAPERCLIP") or os.environ.get("PAPERCLIP_API_KEY")
  if not key:
    return False
  if not PAPERCLIP_PYTHON or not Path(PAPERCLIP_PYTHON).exists():
    return False
  if not PAPERCLIP_SCRIPT.exists():
    return False
  return True


def _paperclip_rpc(fn: str, **kwargs: Any) -> dict:
  """Call ``paperclip_bench_client`` in a 3.10+ subprocess (main bench may be 3.9)."""
  env = os.environ.copy()
  if os.environ.get("PAPERCLIP") and not env.get("PAPERCLIP_API_KEY"):
    env["PAPERCLIP_API_KEY"] = os.environ["PAPERCLIP"]
  payload = json.dumps({"fn": fn, "kwargs": kwargs})
  proc = subprocess.run(
    [PAPERCLIP_PYTHON, str(PAPERCLIP_SCRIPT), "rpc", payload],
    capture_output=True,
    text=True,
    env=env,
    timeout=PAPERCLIP_QUERY_TIMEOUT + 30,
  )
  if proc.returncode != 0:
    raise RuntimeError(
      f"paperclip rpc {fn} failed: {proc.stderr.strip() or proc.stdout.strip()}"
    )
  return json.loads(proc.stdout)


def _paperclip_search_subquery(sub_q: str, *, task: str, limit: int) -> tuple[list[dict], dict]:
  """Route one rewriter sub-query to the correct Paperclip search mode."""
  sub_q = sub_q.strip()
  if not sub_q:
    return [], {"query": sub_q, "raw_count": 0}

  doi_m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", sub_q, re.I)
  if doi_m:
    out = _paperclip_rpc(
      "lookup_field", field="doi", value=doi_m.group(1), limit=limit
    )
    papers, stat = out["papers"], out["stat"]
    stat["query"] = sub_q
    stat["route"] = "lookup_doi"
    return papers, stat

  pmc_m = re.search(r"\b(PMC\d+)\b", sub_q, re.I)
  if pmc_m:
    out = _paperclip_rpc(
      "lookup_field", field="pmc", value=pmc_m.group(1), limit=limit
    )
    papers, stat = out["papers"], out["stat"]
    stat["query"] = sub_q
    stat["route"] = "lookup_pmc"
    return papers, stat

  quoted = bool(re.match(r'^["\'].*["\']$', sub_q))
  tokens = sub_q.split()
  short_kw = len(tokens) <= 4
  kwargs: dict[str, Any] = {
    "query": sub_q,
    "limit": limit,
    "source": PAPERCLIP_SOURCE_FILTER,
    "ranking": "hybrid",
    "exact": quoted,
    "timeout": PAPERCLIP_QUERY_TIMEOUT,
  }
  route = "hybrid_default"

  if quoted or (len(tokens) >= 4 and "-" in sub_q):
    kwargs["exact"] = True
    route = "hybrid_phrase"
  elif short_kw:
    kwargs["ranking"] = "bm25"
    kwargs["mode"] = "all"
    route = "bm25_keywords"
  if task == "TC":
    kwargs["sort"] = "date"
    route += "+date_sort"

  out = _paperclip_rpc("search_with_flags", **kwargs)
  papers, stat = out["papers"], out["stat"]
  stat["query"] = sub_q
  stat["route"] = route

  if len(papers) < 3:
    fallback = dict(kwargs)
    fallback["ranking"] = "vector"
    fallback["source"] = PAPERCLIP_SOURCE_FALLBACK
    out2 = _paperclip_rpc("search_with_flags", **fallback)
    stat["fallback_used"] = "vector+wider_source"
    seen = {p.get("source_doi") or p.get("paper_id") for p in papers}
    for p in out2["papers"]:
      key = p.get("source_doi") or p.get("paper_id")
      if key and key not in seen:
        papers.append(p)
        seen.add(key)
  return papers, stat


def _diversify_papers(
  papers: list[dict],
  max_papers: int = PAPERCLIP_MAX_PAPERS,
  max_per_author: int = PAPERCLIP_MAX_PER_AUTHOR,
) -> list[dict]:
  selected: list[dict] = []
  author_counts: dict[str, int] = defaultdict(int)

  def first_author(p: dict) -> str:
    authors = p.get("source_authors") or []
    if authors:
      return str(authors[0]).lower()
    a = (p.get("authors") or "").split(",")
    return a[0].strip().lower() if a and a[0].strip() else ""

  for p in papers:
    fa = first_author(p)
    if fa and author_counts[fa] >= max_per_author:
      continue
    if fa:
      author_counts[fa] += 1
    selected.append(p)
    if len(selected) >= max_papers:
      break
  return selected


def query_paperclip_unified(q: dict) -> tuple[list[dict], dict]:
  """Paperclip retrieval mirroring ``query_askchem_unified`` shape."""
  rewrite = _llm_rewrite_query(q["question"], q["task"], q["id"])
  sub_queries = list(rewrite.get("sub_queries") or [])
  if not sub_queries:
    fallback = (q.get("askchem_params") or {}).get("q") or q["question"][:80]
    sub_queries = [fallback]

  query_stats: list[dict] = []
  merged: list[dict] = []
  seen: set[str] = set()

  with ThreadPoolExecutor(max_workers=PAPERCLIP_WORKERS) as pool:
    futures = {
      pool.submit(_paperclip_search_subquery, sq, task=q["task"], limit=PAPERCLIP_LIMIT_PER_QUERY): sq
      for sq in sub_queries
    }
    for fut in as_completed(futures):
      sq = futures[fut]
      try:
        papers, stat = fut.result()
      except Exception as exc:
        papers, stat = [], {"query": sq, "error": str(exc)}
        print(f"        paperclip search failed for {sq!r}: {exc}")
      stat["returned"] = len(papers)
      query_stats.append(stat)
      for p in papers:
        key = (p.get("source_doi") or "").strip() or (p.get("paper_id") or "").strip()
        if not key or key in seen:
          continue
        seen.add(key)
        merged.append(p)

  diversified = _diversify_papers(merged)
  for p in diversified[:12]:
    if not p.get("snippet") and p.get("paper_id"):
      try:
        sn = _paperclip_rpc(
          "paper_snippet",
          paper_id=p["paper_id"],
          lines=PAPERCLIP_SNIPPET_LINES,
        ).get("snippet", "")
        if sn:
          p["snippet"] = sn[:1200]
      except Exception:
        pass

  meta = {
    "total_papers": len(merged),
    "diversified_papers": len(diversified),
    "sub_queries": sub_queries,
    "rewriter_rationale": rewrite.get("rationale", ""),
    "queries": query_stats,
  }
  return diversified, meta


def format_papers_for_llm(papers: list[dict], max_papers: int = PAPERCLIP_MAX_PAPERS) -> str:
  lines = []
  for i, p in enumerate(papers[:max_papers]):
    title = p.get("title") or p.get("source_paper_title") or "?"
    authors = p.get("authors") or ", ".join(p.get("source_authors") or [])
    year = p.get("source_year") or ""
    doi = p.get("source_doi") or ""
    line = f"[{i+1}] {title}"
    if authors:
      line += f" — {authors}"
    if year:
      line += f" ({year})"
    line += f" [DOI: {doi}]" if doi else ""
    snippet = (p.get("snippet") or "").strip()
    if snippet:
      line += f"\nExcerpt: {snippet[:800]}"
    lines.append(line)
  return "\n\n".join(lines)


def select_diverse_claims(
    claims: list[dict],
    max_claims: int = UNIFIED_MAX_CLAIMS,
    max_per_source: int = UNIFIED_MAX_PER_SOURCE,
) -> list[dict]:
    """Cap the claim pool while spreading evidence across many source
    papers (no single paper dominates). Used by the unified retrieval
    dispatcher after merging the concurrent /api/search fan-out."""
    selected_indices = []
    source_counts = defaultdict(int)

    def source_key(claim: dict, fallback_idx: int) -> str:
        return (
            str(claim.get("source_doi") or "").strip()
            or str(claim.get("source_paper_title") or "").strip()
            or str(claim.get("source_id") or "").strip()
            or f"claim_{fallback_idx}"
        )

    for idx, claim in enumerate(claims):
        key = source_key(claim, idx)
        if source_counts[key] >= max_per_source:
            continue
        selected_indices.append(idx)
        source_counts[key] += 1
        if len(selected_indices) >= max_claims:
            return [claims[i] for i in selected_indices]

    seen = set(selected_indices)
    for idx, claim in enumerate(claims):
        if idx in seen:
            continue
        selected_indices.append(idx)
        if len(selected_indices) >= max_claims:
            break
    return [claims[i] for i in selected_indices]


def format_claims_for_llm(claims: list[dict], max_claims: int = 40) -> str:
    """Format claims into a text block for LLM consumption."""
    lines = []
    for i, c in enumerate(claims[:max_claims]):
        quote = c.get("verbatim_quote", "") or ""
        doi = c.get("source_doi", "")
        ctype = c.get("claim_type", "")
        title = c.get("source_paper_title", "")
        year = str(c.get("source_year", ""))

        line = f"[{i+1}] [{ctype}] {quote}"
        if title:
            line += f" — Paper: {title}"
        if year:
            line += f" ({year})"
        line += f" [DOI: {doi}]"
        lines.append(line)
    return "\n\n".join(lines)


SYSTEM_LLM_ALONE = (
    "You are a chemistry expert. Answer the question concisely "
    "but with specific details: catalyst names, conditions, "
    "yields, DOIs where possible. If you're unsure about a "
    "specific detail, say so."
)

SYSTEM_LLM_GROUNDED = (
    "You are a chemistry expert. Synthesize a concise, "
    "authoritative answer to the question using ONLY the "
    "research claims provided below. Cite specific DOIs in "
    "parentheses. Highlight any contradictions or nuances "
    "between different papers. Do not add information beyond "
    "what the claims support."
)

MAX_RETRIES = 3


_USES_NEW_API = None


def _model_uses_new_api(client: OpenAI) -> bool:
    """Detect if model requires max_completion_tokens / developer role."""
    global _USES_NEW_API
    if _USES_NEW_API is not None:
        return _USES_NEW_API
    try:
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            temperature=0.0,
        )
        _USES_NEW_API = False
    except Exception:
        _USES_NEW_API = True
    return _USES_NEW_API


def _call_llm(client: OpenAI, messages: list[dict], max_tokens: int = 1200) -> str:
    new_api = _model_uses_new_api(client)
    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict = {"model": MODEL, "timeout": 120}
            if new_api:
                converted = []
                for m in messages:
                    role = "developer" if m["role"] == "system" else m["role"]
                    converted.append({"role": role, "content": m["content"]})
                kwargs["messages"] = converted
                kwargs["max_completion_tokens"] = max(max_tokens, 8000)
            else:
                kwargs["messages"] = messages
                kwargs["temperature"] = 0.3
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 5 * (attempt + 1)
                print(f"        Retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def llm_alone(client: OpenAI, question: str) -> str:
    return _call_llm(client, [
        {"role": "system", "content": SYSTEM_LLM_ALONE},
        {"role": "user", "content": question},
    ])


def llm_with_askchem(
    client: OpenAI,
    question: str,
    claims_text: str,
    total_claims: int,
    *,
    system_prompt: str,
    prompt_label: str,
    claims_shown: int,
    max_tokens: int = 1500,
) -> str:
    return _call_llm(client, [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"The following {total_claims} research claims were retrieved "
                f"from the AskChem index. Below is the {prompt_label} evidence set "
                f"(showing {claims_shown} claims):\n\n"
                f"{claims_text}"
            ),
        },
    ], max_tokens=max_tokens)


def llm_unified(
    client: OpenAI, question: str, claims_text: str, total_claims: int, claims_shown: int
) -> str:
    """Synthesise a grounded answer over the unified-merged claim pool.

    Reuses ``SYSTEM_LLM_GROUNDED`` and the numbered-claim format that gave
    the old strict mode its citation tightness, applied to the broader
    claim pool produced by the rewriter-driven fan-out.
    """
    return llm_with_askchem(
        client,
        question,
        claims_text,
        total_claims,
        system_prompt=SYSTEM_LLM_GROUNDED,
        prompt_label="unified",
        claims_shown=claims_shown,
        max_tokens=1800,
    )


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_askchem_snapshot() -> dict:
    """Capture lightweight corpus metadata for the benchmark output."""
    snapshot = {"api": ASKCHEM_API, "retrieved_at": iso_now()}
    if ASKCHEM_API == "local":
        snapshot["mode"] = "local_db"
        return snapshot
    for endpoint, key in [("stats", "stats"), ("quality", "quality")]:
        try:
            resp = requests.get(f"{ASKCHEM_API}/{endpoint}", timeout=120)
            resp.raise_for_status()
            snapshot[key] = resp.json()
        except Exception as exc:
            snapshot[f"{key}_error"] = str(exc)
    return snapshot


def init_edison_client():
    if not os.environ.get("EDISON"):
        return None
    if EdisonClient is not None:
        return EdisonClient(api_key=os.environ["EDISON"])
    return RawEdisonClient(api_key=os.environ["EDISON"], base_url=EDISON_API)


def _edison_job_name():
    if JobNames is None:
        return EDISON_JOB_NAME
    if hasattr(JobNames, "LITERATURE"):
        return JobNames.LITERATURE
    if hasattr(JobNames, "from_string"):
        return JobNames.from_string("literature")
    return EDISON_JOB_NAME


class RawEdisonClient:
    """Minimal Edison REST client fallback for Python <3.11 environments."""

    TERMINAL_STATES = {"success", "fail", "failed", "cancelled", "truncated"}

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.jwt = ""

    def login(self) -> str:
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"api_key": self.api_key},
            timeout=120,
        )
        response.raise_for_status()
        self.jwt = response.json()["access_token"]
        return self.jwt

    def request(self, method: str, path: str, **kwargs) -> dict:
        if not self.jwt:
            self.login()
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {self.jwt}"
        headers.setdefault("x-client", "sdk")
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            timeout=120,
            **kwargs,
        )
        if response.status_code in {401, 403}:
            self.login()
            headers["Authorization"] = f"Bearer {self.jwt}"
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=120,
                **kwargs,
            )
        response.raise_for_status()
        return response.json()

    def run_tasks_until_done(self, task_data: dict):
        create_response = self.request(
            "POST",
            "/v0.1/crows",
            json={"name": task_data["name"], "query": task_data["query"]},
        )
        task_id = create_response["trajectory_id"]
        while True:
            task = self.request("GET", f"/v0.1/trajectories/{task_id}")
            status = str(task.get("status", "")).strip().lower()
            if status in self.TERMINAL_STATES:
                return [task]
            time.sleep(EDISON_POLL_SECONDS)


def _extract_edison_answer(payload) -> str:
    """Walk a paperqa3-flavoured trajectory and return the formatted answer.

    Edison's "literature-20260216" (retired Apr 2026) returned the answer at
    the trajectory root. The current `@FutureHouse/paperqa3` job nests it at
        environment_frame.state.state.response.answer.formatted_answer
    (with `.answer` / `.raw_answer` as fallbacks for shorter variants).

    We do a bounded BFS for the first non-empty `formatted_answer` field
    anywhere in the tree, falling back to `answer`, then `content`, then the
    pre-existing object-attribute walk for older SDK objects.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload

    from collections import deque

    seen: set[int] = set()
    queue: deque = deque([payload])
    fallback_answer = ""
    fallback_content = ""

    while queue:
        node = queue.popleft()
        if id(node) in seen:
            continue
        seen.add(id(node))

        if isinstance(node, dict):
            fa = node.get("formatted_answer")
            if isinstance(fa, str) and fa.strip():
                return fa
            if not fallback_answer:
                a = node.get("answer")
                if isinstance(a, str) and a.strip() and len(a) > 50:
                    fallback_answer = a
            if not fallback_content:
                c = node.get("content")
                if isinstance(c, str) and c.strip() and len(c) > 50:
                    fallback_content = c
            for v in node.values():
                if isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list, str)):
                    queue.append(v)
        else:
            for attr in ("formatted_answer", "answer", "content", "result",
                         "response", "data", "environment_frame", "state"):
                if hasattr(node, attr):
                    try:
                        queue.append(getattr(node, attr))
                    except Exception:
                        continue

    return fallback_answer or fallback_content


def edison_literature(client, question: str) -> str:
    """Run Edison Scientific literature synthesis and return the formatted answer."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.run_tasks_until_done(
                {"name": _edison_job_name(), "query": question}
            )
            answer = _extract_edison_answer(response)
            if not answer:
                raise ValueError(f"Unexpected Edison response: {response!r}")
            return answer
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                wait = 5 * (attempt + 1)
                print(f"        Edison retry {attempt+1}/{MAX_RETRIES} after {wait}s: {exc}")
                time.sleep(wait)
                continue
            raise


def has_answer(block: dict | None) -> bool:
    return bool(block and isinstance(block, dict) and block.get("answer"))


def load_existing_questions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {q.get("id", ""): q for q in data.get("questions", []) if q.get("id")}


def build_output(results: list[dict], askchem_snapshot: dict) -> dict:
    return {
        "model": MODEL,
        "generated_at": iso_now(),
        "askchem_snapshot": askchem_snapshot,
        "methods": {
            "llm_alone": {
                "label": "GPT-only baseline",
                "description": "Reuse existing GPT answers with no AskChem retrieval.",
            },
            "unified": {
                "label": "AskChem (LLM rewriter + hybrid /api/search + grounded synthesis)",
                "description": (
                    "Canonical AskChem usage: a small LLM rewriter turns the "
                    "question into 3-4 short keyword sub-queries, fans them "
                    "out to the single hybrid retrieval endpoint /api/search "
                    "(which already exploits FTS, paper-level, taxonomy, and "
                    "vector signals via RRF), merges and diversifies the "
                    "claim pool to <= 40 claims with <= 4 per source, then "
                    "the answer head synthesises strictly from those claims."
                ),
            },
            "edison_scientific": {
                "label": "Edison Scientific literature baseline",
                "description": "External literature synthesis baseline (all 30 questions).",
            },
            "paperclip_unified": {
                "label": "Paperclip (hybrid/bm25 search + GPT-5.5 synthesis)",
                "description": (
                    "Same rewriter and synthesis as AskChem unified, but retrieval "
                    "uses Paperclip search (ranking hybrid by default, bm25 for short "
                    "keyword sub-queries, lookup for DOIs) over pmc+arxiv, then "
                    "GPT-5.5 grounded synthesis."
                ),
            },
        },
        "subsets": {
            "balanced_9": {
                "description": "Balanced Edison subset with 3 questions each from CA, TC, and CS.",
                "question_ids": EDISON_SUBSET_IDS,
            },
        },
        "questions": results,
        "aggregate": aggregate(results),
        "aggregate_subsets": {
            "balanced_9": aggregate(results, allowed_ids=set(EDISON_SUBSET_IDS))
        },
    }


def save_output(results: list[dict], askchem_snapshot: dict):
    OUTPUT_FILE.write_text(
        json.dumps(build_output(results, askchem_snapshot), indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_single(
    client: OpenAI,
    q: dict,
    idx: int,
    total: int,
    *,
    edison_client=None,
    use_paperclip: bool = False,
    cached_entry: dict | None = None,
    baseline_entry: dict | None = None,
) -> dict:
    """Run a single benchmark question using cached baseline + refreshed methods."""
    print(f"\n{'='*60}")
    print(f"[{idx}/{total}] {q['id']} ({q['task']}) — {q['domain']}")
    print(f"Q: {q['question'][:90]}...")
    print(f"{'='*60}")

    cached_entry = cached_entry or {}
    baseline_entry = baseline_entry or {}
    needs_edison = edison_client is not None
    needs_paperclip = use_paperclip
    if (
        RESUME_EXISTING
        and has_answer(cached_entry.get("llm_alone"))
        and has_answer(cached_entry.get("unified"))
        and (not needs_paperclip or has_answer(cached_entry.get("paperclip_unified")))
        and (not needs_edison or has_answer(cached_entry.get("edison_scientific")))
    ):
        print("  [cached] Reusing completed result.")
        return cached_entry

    llm_alone_seed = cached_entry.get("llm_alone") or baseline_entry.get("llm_alone")
    unified_seed = cached_entry.get("unified")
    paperclip_seed = cached_entry.get("paperclip_unified")
    edison_seed = cached_entry.get("edison_scientific") or baseline_entry.get("edison_scientific")
    n_steps = 2 + (1 if needs_paperclip else 0) + (1 if needs_edison else 0)
    step = 0

    # 1. LLM alone (reused from prior runs when available — answer-head only)
    step += 1
    if has_answer(llm_alone_seed):
        llm_alone_result = llm_alone_seed
        print(f"  [{step}/{n_steps}] LLM alone... cached ({llm_alone_result.get('time_s', 0)}s)")
    else:
        print(f"  [{step}/{n_steps}] LLM alone...")
        t0 = time.time()
        answer_alone = llm_alone(client, q["question"])
        t_alone = round(time.time() - t0, 1)
        dois_alone = verify_dois_in_text(answer_alone, q["question"], sleep_between=0.3)
        llm_alone_result = {
            "answer": answer_alone,
            "time_s": t_alone,
            "dois_verified": dois_alone,
            "metrics": compute_metrics(answer_alone, dois_alone, q["task"]),
        }
        print(f"        {t_alone}s, {len(answer_alone)} chars")

    # 2. Unified AskChem (rewriter → concurrent /api/search → diversify →
    #    grounded synthesis)
    step += 1
    if has_answer(unified_seed):
        unified_result = unified_seed
        print(f"  [{step}/{n_steps}] AskChem unified... cached ({unified_result.get('time_s', 0)}s)")
    else:
        print(f"  [{step}/{n_steps}] AskChem unified retrieval + synthesis...")
        t_retrieval_start = time.time()
        unified_claims, unified_meta = query_askchem_unified(q)
        t_retrieval = round(time.time() - t_retrieval_start, 1)
        print(
            f"        rewriter: {len(unified_meta['sub_queries'])} sub-queries → "
            f"{unified_meta['total_claims']} merged, "
            f"{unified_meta['diversified_claims']} diversified in {t_retrieval}s"
        )
        for stat in unified_meta.get("queries", []):
            print(
                f"          - {stat.get('query', '?')!r}: "
                f"{stat.get('returned', 0)} returned / {stat.get('total', 0)} matched"
            )

        unified_claims_text = format_claims_for_llm(unified_claims, UNIFIED_MAX_CLAIMS)
        t_synth_start = time.time()
        answer_unified = llm_unified(
            client,
            q["question"],
            unified_claims_text,
            unified_meta["diversified_claims"],
            len(unified_claims),
        )
        t_synth = round(time.time() - t_synth_start, 1)
        dois_unified = verify_dois_in_text(
            answer_unified, q["question"], sleep_between=0.3
        )
        unified_result = {
            "answer": answer_unified,
            "time_s": t_synth,
            "retrieval_time_s": t_retrieval,
            "claims_shown": len(unified_claims),
            "retrieval_meta": unified_meta,
            "dois_verified": dois_unified,
            "metrics": compute_metrics(answer_unified, dois_unified, q["task"]),
        }
        print(f"        synth: {t_synth}s, {len(answer_unified)} chars")

    paperclip_result = paperclip_seed
    if needs_paperclip:
        step += 1
        if has_answer(paperclip_result):
            print(
                f"  [{step}/{n_steps}] Paperclip unified... cached "
                f"({paperclip_result.get('time_s', 0)}s)"
            )
        else:
            print(f"  [{step}/{n_steps}] Paperclip unified retrieval + synthesis...")
            t_retrieval_start = time.time()
            try:
                pc_papers, pc_meta = query_paperclip_unified(q)
                t_retrieval = round(time.time() - t_retrieval_start, 1)
                print(
                    f"        rewriter: {len(pc_meta['sub_queries'])} sub-queries → "
                    f"{pc_meta['total_papers']} merged, "
                    f"{pc_meta['diversified_papers']} diversified in {t_retrieval}s"
                )
                for stat in pc_meta.get("queries", []):
                    print(
                        f"          - {stat.get('query', '?')!r} "
                        f"[{stat.get('route', '?')}]: {stat.get('returned', 0)} papers"
                    )
                pc_text = format_papers_for_llm(pc_papers)
                t_synth_start = time.time()
                answer_pc = llm_unified(
                    client,
                    q["question"],
                    pc_text,
                    pc_meta["diversified_papers"],
                    len(pc_papers),
                )
                t_synth = round(time.time() - t_synth_start, 1)
                dois_pc = verify_dois_in_text(
                    answer_pc, q["question"], sleep_between=0.3
                )
                paperclip_result = {
                    "answer": answer_pc,
                    "time_s": t_synth,
                    "retrieval_time_s": t_retrieval,
                    "papers_shown": len(pc_papers),
                    "retrieval_meta": pc_meta,
                    "dois_verified": dois_pc,
                    "metrics": compute_metrics(answer_pc, dois_pc, q["task"]),
                }
                print(f"        synth: {t_synth}s, {len(answer_pc)} chars")
            except Exception as exc:
                print(f"        Paperclip ERROR: {exc}")
                paperclip_result = None

    edison_result = edison_seed
    edison_error = None
    if needs_edison:
        step += 1
        if has_answer(edison_result):
            print(f"  [{step}/{n_steps}] Edison Scientific... cached ({edison_result.get('time_s', 0)}s)")
        else:
            print(f"  [{step}/{n_steps}] Edison Scientific...")
            t0 = time.time()
            try:
                answer_edison = edison_literature(edison_client, q["question"])
                t_edison = round(time.time() - t0, 1)
                dois_edison = verify_dois_in_text(answer_edison, q["question"], sleep_between=0.3)
                edison_result = {
                    "answer": answer_edison,
                    "time_s": t_edison,
                    "dois_verified": dois_edison,
                    "metrics": compute_metrics(answer_edison, dois_edison, q["task"]),
                }
                print(f"        {t_edison}s, {len(answer_edison)} chars")
            except Exception as exc:
                # Don't lose the rest of the question's results to an Edison
                # outage / 404 / quota error. Record the failure so the
                # aggregator skips this question's Edison column and
                # downstream readers can see what went wrong.
                t_edison = round(time.time() - t0, 1)
                edison_error = f"{type(exc).__name__}: {exc}"
                edison_result = None
                print(f"        ERROR after {t_edison}s: {edison_error}")
    result = {
        "id": q["id"],
        "task": q["task"],
        "domain": q["domain"],
        "question": q["question"],
        "llm_alone": llm_alone_result,
        "unified": unified_result,
        **(
            {"paperclip_unified": paperclip_result}
            if needs_paperclip and paperclip_result
            else {}
        ),
        **({"edison_scientific": edison_result} if needs_edison and edison_result else {}),
        **({"edison_error": edison_error} if edison_error else {}),
    }
    # Carry forward legacy diagnostic cells from prior runs (strict_grounded,
    # retrieval_assisted, llm_plus_askchem, askchem). The bench no longer
    # re-emits them, but they remain in the JSON for transparency so the
    # May-11 snapshot is reproducible from the same file.
    for legacy_key in ("strict_grounded", "retrieval_assisted",
                       "llm_plus_askchem", "askchem"):
        if legacy_key in cached_entry:
            result[legacy_key] = cached_entry[legacy_key]

    # If Edison ran, re-stamp the non-Edison cells' metrics with the
    # ``edison_overlap_rate`` now that we know which DOIs Edison cited
    # for this question. Edison's own cell stays at edison_dois=None so
    # the aggregate doesn't report a trivial 100% overlap.
    edison_cell = result.get("edison_scientific") or {}
    edison_dois_set = {
        doi.lower() for doi, info in (edison_cell.get("dois_verified") or {}).items()
        if isinstance(info, dict) and info.get("exists")
    }
    if edison_dois_set:
        for mode_key in ("llm_alone", "unified", "paperclip_unified",
                         "strict_grounded", "retrieval_assisted",
                         "llm_plus_askchem", "askchem"):
            cell = result.get(mode_key)
            if not isinstance(cell, dict):
                continue
            verified = cell.get("dois_verified") or {}
            if not verified:
                continue
            cell["metrics"] = compute_metrics(
                cell.get("answer", ""), verified, q.get("task", ""),
                edison_dois=edison_dois_set,
            )
    return result


def aggregate(results: list[dict], allowed_ids: set[str] | None = None) -> dict:
    """Compute per-task aggregate statistics for the current bench methods.

    The bench emits three methods on new runs:
      * ``llm_alone``        - GPT-only baseline
      * ``unified``          - the canonical AskChem mode
      * ``edison_scientific`` - external literature baseline on the
                                balanced Edison subset

    Historical cells (``strict_grounded``, ``retrieval_assisted``) from
    pre-consolidation runs may still live in the per-question JSON for
    transparency, but they are intentionally NOT surfaced in the aggregate
    so the website renders a single canonical AskChem row.
    """
    method_map = {
        "llm_alone": "alone",
        "unified": "unified",
        "paperclip_unified": "paperclip_unified",
        "edison_scientific": "edison_scientific",
    }
    by_task = defaultdict(lambda: defaultdict(list))
    for r in results:
        if allowed_ids is not None and r["id"] not in allowed_ids:
            continue
        for result_key, summary_key in method_map.items():
            method_result = r.get(result_key)
            if method_result and method_result.get("metrics"):
                by_task[r["task"]][summary_key].append(method_result["metrics"])

    def _numeric(lst, key):
        # Skip None values - they signal "metric not yet computed for this
        # cell" (currently used for paper_relevance_* before the gemini
        # backfill runs). Returning these as zeros would unfairly penalise
        # any mode whose answers have not been judged yet.
        return [m[key] for m in lst if isinstance(m.get(key), (int, float))]

    def avg(lst, key):
        vals = _numeric(lst, key)
        return round(sum(vals) / len(vals), 3) if vals else None

    def std(lst, key):
        vals = _numeric(lst, key)
        if len(vals) < 2:
            return None
        mean = sum(vals) / len(vals)
        return round((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 3)

    summary = {}
    for task, methods in sorted(by_task.items()):
        summary[task] = {}
        for method_name, metric_list in methods.items():
            keys = [
                # Existence + density (unchanged)
                "doi_existence_rate", "doi_relevance_rate",
                "citation_density",
                # Specificity: legacy raw + new citation-anchored (Phase 1a)
                "specificity_score", "grounded_specificity",
                # Paper-quality (Phase 1b)
                "citation_count_mean", "citation_count_median",
                "high_impact_rate", "recent_high_impact_rate",
                "paper_relevance_mean", "paper_relevance_high_rate",
                # Overlap with Edison Scientific as a retrieval baseline.
                # None for the edison_scientific cell itself (by definition
                # 100%, uninformative); aggregates skip None via _numeric.
                "edison_overlap_rate",
            ]
            if task == "TC":
                keys += ["years_mentioned", "temporal_span"]
            if task == "CS":
                keys += ["contradiction_markers"]
            method_summary = {"n_questions": len(metric_list)}
            for k in keys:
                method_summary[k] = {"mean": avg(metric_list, k), "std": std(metric_list, k)}
            summary[task][method_name] = method_summary
    return summary


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI()
    edison_client = init_edison_client()
    use_paperclip = paperclip_enabled()
    askchem_snapshot = fetch_askchem_snapshot()
    existing = load_existing_questions(OUTPUT_FILE) if RESUME_EXISTING else {}
    baseline_cache = load_existing_questions(CACHE_FILE) if RESUME_EXISTING else {}

    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Cache: {CACHE_FILE}")
    if RESUME_EXISTING and existing:
        print(f"Resume: loaded {len(existing)} question records from existing output")
    if edison_client is None:
        print("Edison: disabled (set EDISON to enable)")
    else:
        print("Edison: enabled (all 30 questions)")
    if use_paperclip:
        print(f"Paperclip: enabled via {PAPERCLIP_PYTHON}")
    else:
        print(
            "Paperclip: disabled (set PAPERCLIP + install 0.3.0 CLI; "
            f"PAPERCLIP_PYTHON={PAPERCLIP_PYTHON or 'not found'})"
        )
    questions = [
        q for q in BENCHMARK_QUESTIONS
        if (not BENCH_TASKS or q["task"].upper() in BENCH_TASKS)
        and (not BENCH_IDS or q["id"].lower() in BENCH_IDS)
    ]
    total = len(questions)
    iterating_ids = {q["id"] for q in questions}

    # Seed results with any cached cells we are NOT touching this run, so
    # an interactive `BENCH_IDS=ca02 ...` invocation never overwrites the
    # 29 other questions sitting in the JSON. They get passed through and
    # save_output re-stitches the full file.
    results: list[dict] = []
    untouched_ids: list[str] = []
    for qid, cached in existing.items():
        if qid not in iterating_ids and cached:
            results.append(cached)
            untouched_ids.append(qid)
    if untouched_ids:
        print(f"Preserve: {len(untouched_ids)} untouched cached questions "
              f"({', '.join(sorted(untouched_ids)[:5])}{'...' if len(untouched_ids) > 5 else ''})")

    for i, q in enumerate(questions, 1):
        try:
            result = run_single(
                client,
                q,
                i,
                total,
                edison_client=edison_client,
                use_paperclip=use_paperclip,
                cached_entry=existing.get(q["id"], {}),
                baseline_entry=baseline_cache.get(q["id"], {}),
            )
            results.append(result)
            save_output(results, askchem_snapshot)
        except Exception as e:
            print(f"  ERROR on {q['id']}: {e}")
            continue

    # Re-sort to canonical question order so the on-disk JSON is stable.
    order = {q["id"]: i for i, q in enumerate(BENCHMARK_QUESTIONS)}
    results.sort(key=lambda r: order.get(r.get("id", ""), 999))

    output = build_output(results, askchem_snapshot)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {len(results)}/{total} questions")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"\n--- AGGREGATE SUMMARY ---")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
