#!/usr/bin/env python3
"""Score every cited DOI for relevance via gemini-3.1-pro-preview.

Tasks are keyed by ``(question_id, doi, mode)`` because the same DOI can
be cited by multiple modes (e.g. ``unified`` and ``paperclip_unified``)
with different evidence available:

* ``unified`` (and any other claim-based AskChem mode): the judge sees the
  paper TITLE + ABSTRACT *plus* the verbatim CLAIM text AskChem extracted
  for the question's sub-queries. Scored against the claim.
* ``paperclip_unified``, ``edison_scientific``, ``llm_alone``: title +
  abstract only, scored against the paper as a whole. Back-compatible with
  the pre-existing ``{qid}|{doi}`` cache entries (no claim hash suffix).

Pipeline:

1. Enrich ``unified.dois_verified[doi]`` cells with ``claim_texts``
   (fetched concurrently from ``/api/search`` using the same sub-queries
   AskChem originally used for the question).
2. Collect per-mode tasks and dispatch to the Gemini judge concurrently.
3. Stamp each task's score onto its specific (mode, doi) cell, recompute
   ``compute_metrics`` per cell, and write the aggregate.

Usage::

    PORTKEY_API_KEY=... python scripts/backfill_paper_relevance.py
    PORTKEY_API_KEY=... python scripts/backfill_paper_relevance.py --workers 12 --max 50
    PORTKEY_API_KEY=... python scripts/backfill_paper_relevance.py --skip-enrich
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_askchem import (  # noqa: E402
    score_paper_relevance,
    _load_paper_relevance_cache,
    _save_paper_relevance_cache,
    _relevance_cache_key,
    compute_metrics,
    aggregate,
    EDISON_SUBSET_IDS,
)


CANONICAL = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json"
MIRROR = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5_v2-prod_may11.json"

ASKCHEM_API = os.environ.get("ASKCHEM_API", "https://askchem.org/api").rstrip("/")

# Claim-aware modes get claim_texts fetched and the claim-grounded judge.
# Paper-only modes use title+abstract only.
CLAIM_AWARE_MODES = ("unified", "strict_grounded", "retrieval_assisted")
PAPER_ONLY_MODES = ("llm_alone", "paperclip_unified", "edison_scientific")
ALL_MODES = CLAIM_AWARE_MODES + PAPER_ONLY_MODES

CLAIMS_PER_DOI = 3
ENRICH_TIMEOUT_S = 45
ENRICH_WORKERS = 3
ENRICH_RETRIES = 4


def _fetch_sub_query(query: str) -> list[dict]:
    """Hit /api/search with retry on 5xx/429 and connect timeouts."""
    url = f"{ASKCHEM_API}/search"
    last_err = ""
    for attempt in range(ENRICH_RETRIES):
        try:
            r = requests.get(
                url, params={"q": query, "limit": 50}, timeout=ENRICH_TIMEOUT_S
            )
            if r.status_code in (429, 502, 503, 504):
                last_err = f"http {r.status_code}"
                time.sleep(min(2 ** attempt + 1, 20))
                continue
            r.raise_for_status()
            return r.json().get("results") or []
        except requests.exceptions.RequestException as exc:
            last_err = str(exc)[:120]
            time.sleep(min(2 ** attempt + 1, 20))
    print(f"    ! /api/search failed for {query[:60]!r}: {last_err}")
    return []


def enrich_unified_with_claims(data: dict) -> int:
    """Populate ``unified.dois_verified[doi]['claim_texts']`` in-place.

    Re-runs the rewriter sub-queries that AskChem used originally
    (stored under ``unified.retrieval_meta.sub_queries``), groups returned
    claims by ``source_doi``, and stamps up to ``CLAIMS_PER_DOI`` verbatim
    quotes onto each cited DOI. Idempotent: cells that already carry
    ``claim_texts`` are skipped.

    Returns the number of (qid, doi) cells freshly enriched.
    """
    questions = data.get("questions") or []
    work: list[tuple[str, list[str], list[str]]] = []
    for q in questions:
        for mode in CLAIM_AWARE_MODES:
            cell = q.get(mode) or {}
            verified = cell.get("dois_verified") or {}
            if not verified:
                continue
            needs = [
                doi for doi, info in verified.items()
                if isinstance(info, dict) and info.get("exists")
                and not (info.get("claim_texts") or [])
            ]
            if not needs:
                continue
            sub_qs = list(
                ((cell.get("retrieval_meta") or {}).get("sub_queries") or [])
            )
            if not sub_qs:
                seed = (q.get("askchem_params") or {}).get("q") or q["question"][:80]
                sub_qs = [seed]
            work.append((q["id"], sub_qs, needs))

    if not work:
        print("  enrichment: every unified DOI already carries claim_texts; nothing to do")
        return 0

    unique_queries: list[str] = []
    seen_q: set[str] = set()
    for _, sub_qs, _ in work:
        for sq in sub_qs:
            if sq and sq not in seen_q:
                seen_q.add(sq)
                unique_queries.append(sq)
    print(
        f"  enrichment: {len(work)} (qid, mode) cells need claim_texts; "
        f"{len(unique_queries)} unique sub-queries to fan out"
    )

    query_results: dict[str, list[dict]] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        futs = {pool.submit(_fetch_sub_query, sq): sq for sq in unique_queries}
        done = 0
        for fut in as_completed(futs):
            sq = futs[fut]
            try:
                query_results[sq] = fut.result()
            except Exception as exc:
                print(f"    ! fetch error for {sq!r}: {exc}")
                query_results[sq] = []
            done += 1
            if done % 20 == 0:
                rate = done / max(time.time() - started, 1e-3)
                print(f"    fetched {done}/{len(unique_queries)} sub-queries ({rate:.1f}/s)")
    print(f"  enrichment: fetched {len(query_results)} sub-queries in "
          f"{time.time() - started:.1f}s")

    by_qid_doi: dict[tuple[str, str], list[str]] = {}
    seen_pairs: set[tuple[str, str, str]] = set()
    for qid, sub_qs, _needs in work:
        for sq in sub_qs:
            for claim in query_results.get(sq, []):
                doi = (claim.get("source_doi") or "").strip().lower()
                if not doi:
                    continue
                quote = (claim.get("verbatim_quote") or "").strip()
                if not quote:
                    continue
                key = (qid, doi, quote[:60])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                by_qid_doi.setdefault((qid, doi), []).append(quote)

    enriched = 0
    for q in questions:
        qid = q["id"]
        for mode in CLAIM_AWARE_MODES:
            cell = q.get(mode) or {}
            verified = cell.get("dois_verified") or {}
            for doi, info in verified.items():
                if not isinstance(info, dict) or not info.get("exists"):
                    continue
                if info.get("claim_texts"):
                    continue
                quotes = by_qid_doi.get((qid, doi.lower()), [])
                info["claim_texts"] = quotes[:CLAIMS_PER_DOI]
                if info["claim_texts"]:
                    enriched += 1
    print(f"  enrichment: stamped claim_texts onto {enriched} (qid, doi) cells")
    return enriched


def _cell_claim_text(info: dict) -> str:
    """Concatenate up to ``CLAIMS_PER_DOI`` verbatim quotes for the judge."""
    quotes = [q for q in (info.get("claim_texts") or []) if isinstance(q, str) and q.strip()]
    if not quotes:
        return ""
    return "\n\n".join(f"- {q.strip()}" for q in quotes[:CLAIMS_PER_DOI])


Task = tuple[str, str, str, str, str, str, str]
# (qid, doi, mode, question, title, abstract, claim_text)


def collect_tasks(data: dict) -> list[Task]:
    """One task per (qid, doi, mode) cited DOI."""
    tasks: list[Task] = []
    for q in data.get("questions") or []:
        qid = q["id"]
        question = q["question"]
        for mode in ALL_MODES:
            cell = q.get(mode) or {}
            verified = cell.get("dois_verified") or {}
            for doi, info in verified.items():
                if not isinstance(info, dict) or not info.get("exists"):
                    continue
                title = info.get("crossref_title") or info.get("title") or ""
                abstract = info.get("abstract") or ""
                claim_text = _cell_claim_text(info) if mode in CLAIM_AWARE_MODES else ""
                tasks.append((qid, doi.lower(), mode, question, title, abstract, claim_text))
    return tasks


def _build_edison_doi_index(data: dict) -> dict[str, set[str]]:
    """Map each qid -> set of CrossRef-existing DOIs Edison cited for it."""
    out: dict[str, set[str]] = {}
    for q in data.get("questions") or []:
        ed = (q.get("edison_scientific") or {}).get("dois_verified") or {}
        out[q["id"]] = {
            doi.lower() for doi, info in ed.items()
            if isinstance(info, dict) and info.get("exists")
        }
    return out


def merge_relevance_into_data(
    data: dict, scored: dict[tuple[str, str, str], dict]
) -> int:
    """Stamp llm_relevance onto every (mode, doi) cell using per-mode scores."""
    stamped = 0
    edison_index = _build_edison_doi_index(data)
    for q in data.get("questions") or []:
        qid = q["id"]
        for mode in ALL_MODES:
            cell = q.get(mode) or {}
            verified = cell.get("dois_verified") or {}
            for doi, info in verified.items():
                if not isinstance(info, dict) or not info.get("exists"):
                    continue
                rec = scored.get((qid, doi.lower(), mode))
                if not rec:
                    continue
                score = rec.get("score")
                if score is None:
                    continue
                info["llm_relevance"] = int(score)
                if rec.get("rationale"):
                    info["llm_relevance_rationale"] = rec["rationale"]
                if rec.get("judged_with_claim") is not None:
                    info["llm_relevance_judged_with_claim"] = bool(
                        rec["judged_with_claim"]
                    )
                stamped += 1
            if verified and "metrics" in cell:
                # Edison-overlap is only meaningful for OTHER modes; pass
                # None for the edison_scientific cell itself so the
                # aggregate doesn't report a trivial 100%.
                edison_dois = None if mode == "edison_scientific" else edison_index.get(qid, set())
                cell["metrics"] = compute_metrics(
                    cell.get("answer", ""), verified, q.get("task", ""),
                    edison_dois=edison_dois,
                )
    return stamped


def recompute_aggregate(data: dict) -> None:
    questions = data.get("questions") or []
    data["aggregate"] = aggregate(questions)
    edison_subset = set(EDISON_SUBSET_IDS)
    data["aggregate_subsets"] = {
        "balanced_18_edison": aggregate(questions, allowed_ids=edison_subset),
    }


def run_concurrent_judge(
    tasks: list[Task],
    workers: int,
    cache_flush_every: int = 25,
) -> dict[tuple[str, str, str], dict]:
    cache = _load_paper_relevance_cache()
    out: dict[tuple[str, str, str], dict] = {}

    pending: list[Task] = []
    for t in tasks:
        qid, doi, mode, _q, _ti, _ab, claim_text = t
        ck = _relevance_cache_key(qid, doi, claim_text if claim_text else None)
        if ck in cache and cache[ck].get("score") is not None:
            out[(qid, doi, mode)] = cache[ck]
        else:
            pending.append(t)

    print(f"  cached hits:       {len(out)}")
    print(f"  pending to score:  {len(pending)}")
    if not pending:
        return out

    started = time.time()
    done = 0
    fails = 0
    sample_scores: list[int] = []

    def _score_one(item: Task):
        qid, doi, mode, question, title, abstract, claim_text = item
        rec = score_paper_relevance(
            qid, doi, question, title, abstract,
            use_cache=True,
            claim_text=claim_text or None,
        )
        return (qid, doi, mode), rec

    last_flush = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_score_one, t) for t in pending]
        for fut in as_completed(futs):
            try:
                key, rec = fut.result()
            except Exception as e:
                fails += 1
                print(f"  ! fatal {e}")
                continue
            out[key] = rec
            done += 1
            if rec.get("score") is None:
                fails += 1
            else:
                sample_scores.append(rec["score"])

            if done - last_flush >= cache_flush_every:
                _save_paper_relevance_cache()
                last_flush = done
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0
                eta = (len(pending) - done) / rate if rate else 0
                tally = {s: sample_scores.count(s) for s in (0, 1, 2, 3)}
                print(f"  [{done:>4}/{len(pending)}] rate={rate:.1f}/s "
                      f"eta={eta/60:.1f}min fails={fails} dist={tally}")

    _save_paper_relevance_cache()
    elapsed = time.time() - started
    tally = {s: sample_scores.count(s) for s in (0, 1, 2, 3)}
    print(f"  done in {elapsed/60:.1f} min; fails={fails}; score distribution: {tally}")
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max", type=int, default=None,
                   help="Cap on # tasks (for cheap dry runs).")
    p.add_argument("--paths", nargs="+", default=[str(CANONICAL), str(MIRROR)])
    p.add_argument("--skip-enrich", action="store_true",
                   help="Skip the /api/search claim-text enrichment pass.")
    args = p.parse_args(argv)

    print("== paper/claim-relevance backfill (gemini-3.1-pro-preview) ==")
    print(f"  paths       : {args.paths}")
    print(f"  workers     : {args.workers}")
    print(f"  askchem api : {ASKCHEM_API}")

    primary_path = Path(args.paths[0])
    primary = json.loads(primary_path.read_text())

    if not args.skip_enrich:
        print("\n-- enriching unified cells with claim_texts from /api/search")
        enriched = enrich_unified_with_claims(primary)
        if enriched:
            primary_path.write_text(
                json.dumps(primary, indent=2, ensure_ascii=False)
            )
            print(f"  wrote claim_texts back to {primary_path.name}")

    tasks = collect_tasks(primary)
    print(f"\n  (qid, doi, mode) tasks: {len(tasks)}")
    if args.max:
        tasks = tasks[: args.max]
        print(f"  capped to {len(tasks)} for this run")

    scored = run_concurrent_judge(tasks, workers=args.workers)

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"\n-- skipping (not found): {path}")
            continue
        print(f"\n-- merging into {path.name}")
        data = json.loads(path.read_text())
        if not args.skip_enrich and path != primary_path:
            enrich_unified_with_claims(data)
        n_stamped = merge_relevance_into_data(data, scored)
        recompute_aggregate(data)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  stamped {n_stamped} (qid, doi, mode) cells; recomputed aggregate")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
