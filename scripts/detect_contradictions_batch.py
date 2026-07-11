#!/usr/bin/env python3
"""Batch contradiction detection: PAW pre-filter + Gemini verification.

Scans claims grouped by tree node, generates candidate pairs, pre-filters
with PAW, and sends flagged pairs to Gemini for verification.

Run this script with a Python environment that can import `programasweights`
(on this machine that is `/opt/homebrew/bin/python3.10`). Gemini verification
uses plain HTTP, so it no longer depends on the `openai` package being present
in the same interpreter.

Usage:
    # Full pipeline (PAW → Gemini → DB)
    python scripts/detect_contradictions_batch.py --view by_reaction_type

    # PAW-only pass (no Gemini, saves flagged pairs for later)
    python scripts/detect_contradictions_batch.py --paw-only

    # Gemini-only pass (verify previously PAW-flagged pairs)
    python scripts/detect_contradictions_batch.py --gemini-only

Requires: PORTKEY_API_KEY env var for Gemini verification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askchem.db import get_db_path

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-2.5-flash"

MAX_PAIRS_PER_NODE = 500
MAX_CLAIMS_PER_TYPE_GROUP = 100
PAW_TEXT_LIMIT = 280
GEMINI_BATCH_SIZE = 5
MAX_GEMINI_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = 45


def _to_str(val) -> str:
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v).strip()
    return (str(val) if val else "").strip()


def _subjects_overlap(c1: dict, c2: dict) -> bool:
    s1 = _to_str(c1.get("subject")).lower()
    s2 = _to_str(c2.get("subject")).lower()
    if s1 and s2 and s1 == s2:
        return True
    rt1 = _to_str(c1.get("reaction_type")).lower()
    rt2 = _to_str(c2.get("reaction_type")).lower()
    if rt1 and rt2 and rt1 == rt2:
        return True
    sm1 = _to_str(c1.get("subject_smiles"))
    sm2 = _to_str(c2.get("subject_smiles"))
    if sm1 and sm2 and sm1 == sm2:
        return True
    return False


def generate_candidates(conn, view_id: str) -> list[dict]:
    """Generate candidate contradiction pairs from tree nodes."""
    nodes = conn.execute(
        "SELECT path, claim_ids FROM tree_nodes "
        "WHERE view_id = ? AND claim_ids IS NOT NULL AND claim_ids != '[]'",
        (view_id,),
    ).fetchall()
    print(f"Processing {len(nodes)} nodes in view '{view_id}'...", flush=True)

    all_candidates = []
    seen_pairs = set()
    nodes_with_candidates = 0

    for idx, node_row in enumerate(nodes):
        path = node_row[0]
        claim_ids = json.loads(node_row[1]) if node_row[1] else []
        if len(claim_ids) < 2:
            continue

        # Limit to avoid O(n²) explosion on huge nodes
        if len(claim_ids) > 2000:
            claim_ids = claim_ids[:2000]

        ph = ",".join("?" * len(claim_ids))
        rows = conn.execute(
            f"SELECT claim_id, claim_type, data FROM claims WHERE claim_id IN ({ph})",
            claim_ids,
        ).fetchall()

        by_type = defaultdict(list)
        for r in rows:
            by_type[r[1]].append((r[0], json.loads(r[2])))

        node_pairs = 0
        for claim_type, type_claims in by_type.items():
            if len(type_claims) < 2:
                continue
            claims_subset = type_claims[:MAX_CLAIMS_PER_TYPE_GROUP]
            for i in range(len(claims_subset)):
                for j in range(i + 1, len(claims_subset)):
                    cid1, d1 = claims_subset[i]
                    cid2, d2 = claims_subset[j]

                    if d1.get("source_doi") == d2.get("source_doi"):
                        continue

                    if not _subjects_overlap(d1, d2):
                        continue

                    pair_key = tuple(sorted([cid1, cid2]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    vq1 = (d1.get("verbatim_quote", "") or "")[:500]
                    vq2 = (d2.get("verbatim_quote", "") or "")[:500]
                    if len(vq1) < 50 or len(vq2) < 50:
                        continue

                    all_candidates.append({
                        "claim_id_1": cid1,
                        "claim_id_2": cid2,
                        "claim_type": claim_type,
                        "quote_1": vq1,
                        "quote_2": vq2,
                        "doi_1": d1.get("source_doi", ""),
                        "doi_2": d2.get("source_doi", ""),
                        "subject": d1.get("subject", ""),
                        "view_id": view_id,
                        "node_path": path,
                    })
                    node_pairs += 1
                    if node_pairs >= MAX_PAIRS_PER_NODE:
                        break
                if node_pairs >= MAX_PAIRS_PER_NODE:
                    break
            if node_pairs >= MAX_PAIRS_PER_NODE:
                break

        if node_pairs > 0:
            nodes_with_candidates += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(nodes)} nodes, "
                  f"{len(all_candidates)} candidates so far", flush=True)

    print(f"Generated {len(all_candidates)} candidate pairs from "
          f"{nodes_with_candidates} nodes", flush=True)
    return all_candidates


OPPOSING_PAIRS = [
    ({"high", "increase", "increased", "increases", "enhance", "enhanced", "stable",
      "retain", "retained", "excellent", "superior", "effective", "efficient"},
     {"low", "decrease", "decreased", "decreases", "degrade", "degraded", "unstable",
      "lose", "lost", "poor", "inferior", "ineffective", "inefficient"}),
    ({"improve", "improved", "promotes", "facilitates", "accelerate"},
     {"worsen", "worsened", "inhibits", "hinders", "decelerate"}),
    ({"positive", "beneficial", "favorable"},
     {"negative", "detrimental", "unfavorable"}),
]


def _tokenize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_+-]*", text.lower()))


def _cue_conflict_score(cand: dict) -> int:
    """Cheap lexical safety net to recover obvious PAW misses."""
    w1 = _tokenize_words(cand["quote_1"])
    w2 = _tokenize_words(cand["quote_2"])
    score = 0
    for pos_set, neg_set in OPPOSING_PAIRS:
        if (w1 & pos_set and w2 & neg_set) or (w1 & neg_set and w2 & pos_set):
            score += 1
    return score


def heuristic_filter(candidates: list[dict]) -> list[dict]:
    """Fast keyword-based pre-filter for potential contradictions."""
    flagged = []
    for cand in candidates:
        score = _cue_conflict_score(cand)
        if score > 0:
            cand["paw_verdict"] = "heuristic_flagged"
            cand["prefilter_reason"] = "cue_conflict"
            flagged.append(cand)

    print(f"Heuristic filter: {len(flagged)}/{len(candidates)} flagged "
          f"({len(flagged)/len(candidates)*100:.1f}%)" if candidates
          else "Heuristic: no candidates", flush=True)
    return flagged


def paw_filter(candidates: list[dict]) -> list[dict]:
    """Run PAW pre-filter on candidate pairs.

    PAW remains the primary local filter, but a cheap lexical conflict score
    acts as a safety net for obvious pros-vs-cons pairs that PAW may still miss.
    """
    from askchem.paw_functions import detect_contradiction

    flagged = []
    total = len(candidates)
    t0 = time.time()
    for idx, cand in enumerate(candidates):
        q1 = cand["quote_1"][:PAW_TEXT_LIMIT]
        q2 = cand["quote_2"][:PAW_TEXT_LIMIT]
        verdict = detect_contradiction(q1, q2)
        cand["paw_verdict"] = verdict
        cue_score = _cue_conflict_score(cand)
        cand["cue_conflict_score"] = cue_score
        if verdict in ("contradicts", "unclear"):
            cand["prefilter_reason"] = f"paw:{verdict}"
            flagged.append(cand)
        elif cue_score > 0:
            cand["prefilter_reason"] = "cue_conflict"
            flagged.append(cand)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate / 60
            print(f"  PAW: {idx+1}/{total} processed, "
                  f"{len(flagged)} flagged, {rate:.1f}/s, ETA {eta:.0f}min",
                  flush=True)

    print(f"PAW filter: {len(flagged)}/{total} flagged "
          f"({len(flagged)/total*100:.1f}%)" if total else "PAW: no candidates",
          flush=True)
    return flagged


def _build_gemini_prompt(pairs: list[dict]) -> str:
    lines = [
        "You are a chemistry expert checking whether pairs of claims contradict each other.",
        "Only mark a pair as confirmed when the claims make genuinely incompatible assertions",
        "about the same subject or property. Different conditions, different materials, or",
        "different aspects of the same material should be rejected.",
        "",
    ]
    for i, pair in enumerate(pairs, 1):
        lines.append(f"--- Pair {i} ---")
        if pair.get("subject"):
            lines.append(f"Subject: {pair['subject']}")
        lines.append(f"Claim A (DOI: {pair['doi_1']}): \"{pair['quote_1']}\"")
        lines.append(f"Claim B (DOI: {pair['doi_2']}): \"{pair['quote_2']}\"")
        lines.append("")

    lines.append(
        "Respond with a JSON array, one object per pair in order:\n"
        '[{"pair": 1, "verdict": "confirmed"|"rejected", '
        '"explanation": "one sentence", "confidence": 0.0-1.0}, ...]\n'
        "Return JSON only."
    )
    return "\n".join(lines)


def _parse_gemini_results(text: str) -> list[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        results = []
        for match in re.finditer(r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
        return results


def _call_gemini_batch(
    pairs: list[dict],
    api_key: str,
    model: str,
    batch_idx: int,
    timeout: int,
) -> list[dict]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": _build_gemini_prompt(pairs)}],
        "temperature": 0.1,
        "max_tokens": 2000 + 300 * len(pairs),
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{GATEWAY}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-portkey-api-key": api_key,
            "x-portkey-provider": PROVIDER,
        },
        method="POST",
    )

    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"].strip()
            results = _parse_gemini_results(text)

            for i, pair in enumerate(pairs):
                if i < len(results):
                    result = results[i]
                    pair["gemini_verdict"] = result.get("verdict", "rejected")
                    pair["gemini_explanation"] = result.get("explanation", "")
                    pair["confidence"] = float(result.get("confidence", 0.5))
                else:
                    pair["gemini_verdict"] = "error"
                    pair["gemini_explanation"] = (
                        f"Missing from Gemini response ({len(results)}/{len(pairs)} parsed)"
                    )
                    pair["confidence"] = 0.0
            return pairs
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            err = f"HTTP {e.code}: {body_text[:200]}"
            is_retryable = e.code in (408, 409, 429, 500, 502, 503, 504)
        except Exception as e:
            err = str(e)
            lowered = err.lower()
            is_retryable = (
                "429" in err or "rate" in lowered or "quota" in lowered
                or "timed out" in lowered or "connection" in lowered
            )

        if attempt < 4 and is_retryable:
            wait = min(2 ** attempt * 5, 60)
            if attempt >= 1:
                print(
                    f"    Batch {batch_idx} attempt {attempt+1} error: {err[:100]}; "
                    f"retrying in {wait}s",
                    flush=True,
                )
            time.sleep(wait)
            continue

        for pair in pairs:
            pair["gemini_verdict"] = "error"
            pair["gemini_explanation"] = err[:200]
            pair["confidence"] = 0.0
        return pairs

    return pairs


def gemini_verify(
    flagged: list[dict],
    batch_size: int = GEMINI_BATCH_SIZE,
    workers: int = MAX_GEMINI_WORKERS,
    model: str = MODEL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> list[dict]:
    """Verify flagged pairs with Gemini in concurrent batches."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PORTKEY_API_KEY not set. Run with --paw-only or export the key for Gemini verification."
        )

    batches = [
        flagged[i:i + batch_size]
        for i in range(0, len(flagged), batch_size)
    ]
    print(
        f"Gemini verification: {len(flagged)} pairs in {len(batches)} batches "
        f"(size {batch_size}, {workers} workers, model={model})",
        flush=True,
    )

    verified: list[dict] = []
    confirmed = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_call_gemini_batch, batch, api_key, model, idx, timeout): idx
            for idx, batch in enumerate(batches)
        }
        for done, future in enumerate(as_completed(futures), 1):
            result_pairs = future.result()
            verified.extend(result_pairs)
            for pair in result_pairs:
                if pair.get("gemini_verdict") == "confirmed":
                    confirmed += 1
                if pair.get("gemini_verdict") == "error":
                    errors += 1

            if done % 10 == 0 or done == len(batches):
                elapsed = max(time.time() - t0, 0.1)
                rate = len(verified) / elapsed
                remaining = len(flagged) - len(verified)
                eta_min = remaining / max(rate, 0.1) / 60
                print(
                    f"  Gemini: {done}/{len(batches)} batches done, "
                    f"{confirmed} confirmed, {errors} errors, "
                    f"{rate:.1f} pairs/s, ETA {eta_min:.0f}min",
                    flush=True,
                )

    print(
        f"Gemini verification: {confirmed} confirmed out of "
        f"{len(flagged)} flagged ({errors} errors)",
        flush=True,
    )
    return verified


def store_results(conn, results: list[dict]):
    """Store verified contradictions in the database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id_1 TEXT NOT NULL,
            claim_id_2 TEXT NOT NULL,
            view_id TEXT,
            node_path TEXT,
            paw_verdict TEXT,
            gemini_verdict TEXT,
            gemini_explanation TEXT,
            confidence REAL,
            detected_at TEXT,
            UNIQUE(claim_id_1, claim_id_2)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_view "
                 "ON contradictions(view_id, node_path)")

    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for r in results:
        ids = sorted([r["claim_id_1"], r["claim_id_2"]])
        try:
            conn.execute(
                "INSERT OR REPLACE INTO contradictions "
                "(claim_id_1, claim_id_2, view_id, node_path, "
                "paw_verdict, gemini_verdict, gemini_explanation, "
                "confidence, detected_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (ids[0], ids[1], r.get("view_id"), r.get("node_path"),
                 r.get("paw_verdict"), r.get("gemini_verdict"),
                 r.get("gemini_explanation", ""), r.get("confidence", 0.0),
                 now),
            )
            inserted += 1
        except Exception as e:
            print(f"  Insert error: {e}", flush=True)

    conn.commit()
    print(f"Stored {inserted} contradiction records in DB", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Batch contradiction detection")
    parser.add_argument("--view", default="by_reaction_type",
                        help="Tree view to scan")
    parser.add_argument("--paw-only", action="store_true",
                        help="Run PAW filter only, save flagged pairs to JSON")
    parser.add_argument("--heuristic", action="store_true",
                        help="Use fast keyword heuristic instead of PAW")
    parser.add_argument("--gemini-only", action="store_true",
                        help="Verify previously flagged pairs from JSON")
    parser.add_argument("--gemini-batch-size", type=int, default=GEMINI_BATCH_SIZE,
                        help="Pairs per Gemini request")
    parser.add_argument("--gemini-workers", type=int, default=MAX_GEMINI_WORKERS,
                        help="Concurrent Gemini requests")
    parser.add_argument("--gemini-model", default=MODEL,
                        help="Gemini model name for verification")
    parser.add_argument("--gemini-timeout", type=int, default=REQUEST_TIMEOUT_SECONDS,
                        help="HTTP timeout per Gemini request in seconds")
    parser.add_argument("--flagged-file", type=Path,
                        default=Path("data/paw_flagged_pairs.json"),
                        help="JSON file for intermediate flagged pairs")
    args = parser.parse_args()

    db_path = get_db_path()
    print(f"Database: {db_path}", flush=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA mmap_size=268435456")

    if args.gemini_only:
        if not args.flagged_file.exists():
            print(f"ERROR: {args.flagged_file} not found. Run PAW first.", flush=True)
            sys.exit(1)
        flagged = json.loads(args.flagged_file.read_text())
        print(f"Loaded {len(flagged)} PAW-flagged pairs", flush=True)
    else:
        candidates = generate_candidates(conn, args.view)
        if not candidates:
            print("No candidate pairs found.", flush=True)
            return

        if args.heuristic:
            flagged = heuristic_filter(candidates)
        else:
            flagged = paw_filter(candidates)
        if not flagged:
            print("No pairs flagged by PAW.", flush=True)
            return

        args.flagged_file.parent.mkdir(parents=True, exist_ok=True)
        args.flagged_file.write_text(json.dumps(flagged, indent=2))
        print(f"Saved {len(flagged)} flagged pairs to {args.flagged_file}", flush=True)

        if args.paw_only:
            print("PAW-only mode, stopping here.", flush=True)
            return

    verified = gemini_verify(
        flagged,
        batch_size=args.gemini_batch_size,
        workers=args.gemini_workers,
        model=args.gemini_model,
        timeout=args.gemini_timeout,
    )
    store_results(conn, verified)

    confirmed = [v for v in verified if v.get("gemini_verdict") == "confirmed"]
    print(f"\n{'='*60}", flush=True)
    print(f"Summary: {len(confirmed)} confirmed contradictions", flush=True)
    for c in confirmed[:10]:
        print(f"  [{c.get('confidence', 0):.2f}] {c['subject'][:40]}:", flush=True)
        print(f"    A: {c['quote_1'][:80]}...", flush=True)
        print(f"    B: {c['quote_2'][:80]}...", flush=True)
        print(f"    → {c.get('gemini_explanation', '')}", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
