"""
AskChem Surprise Scoring Engine.

Computes a surprise score (0-100) for each claim based on:
  1. Structural signals: claim in unusual view combination, outlier values
  2. Temporal signals: first claim in empty node, rapidly growing area
  3. Content signals: LLM-based novelty assessment (batch, gpt-5-mini)

Usage:
    python src/surprise_scoring.py compute          # Compute scores for all claims
    python src/surprise_scoring.py top --limit 50   # Show top surprising claims
    python src/surprise_scoring.py status            # Show scoring progress
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from askchem import db

DATA_DIR = Path(__file__).parent.parent / "data"
SCORES_FILE = DATA_DIR / "surprise_scores.json"


def compute_structural_score(claim: dict, node_stats: dict) -> float:
    """Score based on structural position in the tree (0-40 points).

    Signals:
    - Claim is in a sparse node (few siblings) -> higher surprise
    - Claim type is rare for its node -> higher surprise
    - Claim has outlier numerical values -> higher surprise
    """
    score = 0.0

    view_paths = claim.get("view_paths", {})
    if isinstance(view_paths, str):
        try:
            view_paths = json.loads(view_paths)
        except (json.JSONDecodeError, TypeError):
            view_paths = {}

    for view_id, path_segments in view_paths.items():
        if not path_segments:
            continue
        path_key = "/".join(path_segments) if isinstance(path_segments, list) else str(path_segments)
        stats = node_stats.get(f"{view_id}/{path_key}", {})
        sibling_count = stats.get("sibling_claims", 100)

        if sibling_count <= 3:
            score += 10
        elif sibling_count <= 10:
            score += 5

    claim_type = claim.get("claim_type", "")
    if claim_type in ("surprising_finding", "limitation", "hypothesis"):
        score += 10
    elif claim_type == "future_direction":
        score += 5

    outcomes = claim.get("outcomes", {})
    if isinstance(outcomes, dict):
        yield_pct = outcomes.get("yield_percent")
        ee_pct = outcomes.get("ee_percent")
        if yield_pct is not None and isinstance(yield_pct, (int, float)):
            if yield_pct > 99:
                score += 5
            elif yield_pct < 10:
                score += 8
        if ee_pct is not None and isinstance(ee_pct, (int, float)):
            if ee_pct > 99:
                score += 5

    return min(score, 40.0)


def compute_temporal_score(claim: dict, doi_years: dict) -> float:
    """Score based on temporal signals (0-30 points).

    Signals:
    - Claim from a very recent paper -> higher surprise
    - Claim in a rapidly growing area -> higher surprise
    """
    score = 0.0

    year = doi_years.get(claim.get("source_doi", ""), 0)

    current_year = datetime.now().year
    if year >= current_year:
        score += 15
    elif year >= current_year - 1:
        score += 10
    elif year >= current_year - 2:
        score += 5

    return min(score, 30.0)


def compute_content_score(claim: dict) -> float:
    """Score based on content signals (0-30 points).

    Signals:
    - Claim explicitly flagged as surprising -> high score
    - Claim contains contradiction language -> higher score
    - Claim has "first", "novel", "unprecedented" language -> higher score
    """
    score = 0.0

    if claim.get("claim_type") == "surprising_finding":
        score += 20

    quote = (claim.get("verbatim_quote") or "").lower()
    surprise_words = ["surprisingly", "unexpectedly", "contrary to", "unprecedented",
                      "first report", "first example", "never before", "counterintuitive",
                      "remarkable", "striking", "anomalous"]
    for word in surprise_words:
        if word in quote:
            score += 5
            break

    novelty_words = ["novel", "new class", "first-in-class", "breakthrough",
                     "paradigm", "record-breaking", "state-of-the-art"]
    for word in novelty_words:
        if word in quote:
            score += 5
            break

    return min(score, 30.0)


def _build_node_stats() -> dict:
    """Build node statistics from the tree_nodes table.

    Returns a dict mapping "view_id/path" -> {"sibling_claims": int, "claim_count": int}
    so the structural scorer can detect sparse nodes.
    """
    stats = {}
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT view_id, path, claim_count, children FROM tree_nodes"
        ).fetchall()

    parent_children: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in rows:
        view_id = r["view_id"]
        path = r["path"]
        claim_count = r["claim_count"] or 0
        full_key = f"{view_id}/{path}" if path else view_id

        parts = path.split("/") if path else []
        if len(parts) > 0:
            parent_path = "/".join(parts[:-1])
            parent_key = f"{view_id}/{parent_path}" if parent_path else view_id
            parent_children[parent_key].append((full_key, claim_count))

        stats[full_key] = {"claim_count": claim_count, "sibling_claims": claim_count}

    for parent_key, children in parent_children.items():
        for child_key, child_count in children:
            sibling_total = sum(c for _, c in children)
            stats[child_key]["sibling_claims"] = sibling_total

    return stats


def compute_all_scores():
    """Compute surprise scores for all claims."""
    print("Loading claims and source years...", flush=True)
    with db.get_conn() as conn:
        rows = conn.execute("SELECT data FROM claims").fetchall()
        year_rows = conn.execute("SELECT doi, year FROM sources").fetchall()

    claims = [json.loads(r["data"]) for r in rows]
    doi_years = {r["doi"]: (r["year"] or 0) for r in year_rows}
    print(f"Loaded {len(claims):,} claims, {len(doi_years):,} source years", flush=True)

    print("Building node statistics...", flush=True)
    node_stats = _build_node_stats()

    scores = {}
    for i, claim in enumerate(claims):
        s_struct = compute_structural_score(claim, node_stats)
        s_temporal = compute_temporal_score(claim, doi_years)
        s_content = compute_content_score(claim)

        total = s_struct + s_temporal + s_content
        scores[claim.get("claim_id", "")] = {
            "total": round(total, 1),
            "structural": round(s_struct, 1),
            "temporal": round(s_temporal, 1),
            "content": round(s_content, 1),
        }

        if (i + 1) % 20000 == 0:
            print(f"  Scored {i+1:,}/{len(claims):,}", flush=True)

    # Write to JSON file
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f)

    # Also write to DB for the feed endpoint
    print("Writing scores to database...", flush=True)
    with db.get_conn(readonly=False) as conn:
        batch = []
        for claim_id, s in scores.items():
            batch.append((claim_id, s["total"], s["structural"], s["temporal"],
                          s["content"], datetime.now().isoformat()))
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT OR REPLACE INTO surprise_scores "
                    "(claim_id, total_score, structural_score, temporal_score, content_score, computed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)", batch)
                conn.commit()
                batch = []
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO surprise_scores "
                "(claim_id, total_score, structural_score, temporal_score, content_score, computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", batch)
            conn.commit()

    high_surprise = sum(1 for s in scores.values() if s["total"] >= 50)
    medium_surprise = sum(1 for s in scores.values() if 25 <= s["total"] < 50)
    print(f"\nDone: {len(scores):,} claims scored")
    print(f"  High surprise (>=50): {high_surprise:,}")
    print(f"  Medium surprise (25-50): {medium_surprise:,}")


def show_top(limit: int = 50):
    """Show top surprising claims."""
    if not SCORES_FILE.exists():
        print("No scores computed yet. Run 'compute' first.")
        return

    with open(SCORES_FILE) as f:
        scores = json.load(f)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)

    print(f"Top {limit} most surprising claims:\n")
    with db.get_conn() as conn:
        for claim_id, score_data in sorted_scores[:limit]:
            row = conn.execute("SELECT data FROM claims WHERE claim_id = ?", [claim_id]).fetchone()
            if not row:
                continue
            claim = json.loads(row["data"])
            print(f"  Score: {score_data['total']:5.1f} "
                  f"(S:{score_data['structural']:.0f} T:{score_data['temporal']:.0f} C:{score_data['content']:.0f}) "
                  f"| [{claim.get('claim_type', '')}] "
                  f"{(claim.get('verbatim_quote') or '')[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="AskChem Surprise Scoring")
    parser.add_argument("command", choices=["compute", "top", "status"])
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.command == "compute":
        compute_all_scores()
    elif args.command == "top":
        show_top(limit=args.limit)
    elif args.command == "status":
        if SCORES_FILE.exists():
            with open(SCORES_FILE) as f:
                scores = json.load(f)
            print(f"Scores computed: {len(scores):,}")
            high = sum(1 for s in scores.values() if s["total"] >= 50)
            print(f"High surprise (>=50): {high:,}")
        else:
            print("No scores computed yet.")


if __name__ == "__main__":
    main()
