"""
AskChem Contradiction Detection.

Identifies potential contradictions between claims at the same tree node.
Two claims may contradict if they report different values for the same
property/reaction under similar conditions.

Heuristic approach (no LLM needed):
1. Group claims by tree node path
2. Within each group, find pairs with same claim_type and overlapping subjects
3. Compare numerical outcomes (yield, ee, etc.) for significant disagreements
4. Flag pairs where the same measurement differs by >20% or contradicts direction

Usage:
    python src/detect_contradictions.py scan
    python src/detect_contradictions.py scan --view by_reaction_type --min-confidence medium
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from askchem import db


def _extract_numeric(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip().rstrip("%"))
        except (ValueError, TypeError):
            return None
    return None


def _subjects_overlap(c1: dict, c2: dict) -> bool:
    """Check if two claims are about similar subjects."""
    s1 = (c1.get("subject", "") or "").lower().strip()
    s2 = (c2.get("subject", "") or "").lower().strip()
    if s1 and s2 and s1 == s2:
        return True

    rt1 = (c1.get("reaction_type", "") or "").lower().strip()
    rt2 = (c2.get("reaction_type", "") or "").lower().strip()
    if rt1 and rt2 and rt1 == rt2:
        return True

    sm1 = c1.get("subject_smiles", "")
    sm2 = c2.get("subject_smiles", "")
    if sm1 and sm2 and sm1 == sm2:
        return True

    return False


COMPARABLE_FIELDS = [
    ("outcomes", "yield_percent"),
    ("outcomes", "ee_percent"),
    ("outcomes", "conversion_percent"),
]


def find_contradictions_in_group(claims: list[dict], threshold: float = 20.0) -> list[dict]:
    """Find contradicting claim pairs within a group."""
    contradictions = []

    by_type = defaultdict(list)
    for c in claims:
        by_type[c.get("claim_type", "")].append(c)

    for claim_type, type_claims in by_type.items():
        if len(type_claims) < 2:
            continue

        for i in range(len(type_claims)):
            for j in range(i + 1, len(type_claims)):
                c1, c2 = type_claims[i], type_claims[j]

                if c1.get("source_doi") == c2.get("source_doi"):
                    continue

                if not _subjects_overlap(c1, c2):
                    continue

                for parent_field, field_name in COMPARABLE_FIELDS:
                    v1_raw = (c1.get(parent_field, {}) or {}).get(field_name)
                    v2_raw = (c2.get(parent_field, {}) or {}).get(field_name)
                    v1 = _extract_numeric(v1_raw)
                    v2 = _extract_numeric(v2_raw)

                    if v1 is None or v2 is None:
                        continue

                    diff = abs(v1 - v2)
                    if diff >= threshold:
                        contradictions.append({
                            "claim_1_id": c1.get("claim_id"),
                            "claim_2_id": c2.get("claim_id"),
                            "claim_type": claim_type,
                            "field": f"{parent_field}.{field_name}",
                            "value_1": v1,
                            "value_2": v2,
                            "difference": round(diff, 1),
                            "source_1": c1.get("source_doi"),
                            "source_2": c2.get("source_doi"),
                            "subject": c1.get("subject", "") or c1.get("reaction_type", ""),
                        })

                pn1 = (c1.get("property_name", "") or "").lower()
                pn2 = (c2.get("property_name", "") or "").lower()
                if pn1 and pn1 == pn2:
                    v1 = _extract_numeric(c1.get("value"))
                    v2 = _extract_numeric(c2.get("value"))
                    if v1 is not None and v2 is not None:
                        denom = max(abs(v1), abs(v2), 1)
                        rel_diff = abs(v1 - v2) / denom * 100
                        if rel_diff >= threshold:
                            contradictions.append({
                                "claim_1_id": c1.get("claim_id"),
                                "claim_2_id": c2.get("claim_id"),
                                "claim_type": claim_type,
                                "field": f"property:{pn1}",
                                "value_1": v1,
                                "value_2": v2,
                                "difference": round(rel_diff, 1),
                                "source_1": c1.get("source_doi"),
                                "source_2": c2.get("source_doi"),
                                "subject": c1.get("subject", ""),
                            })

    return contradictions


def scan_all_contradictions(view_id: str = "by_reaction_type",
                            threshold: float = 20.0) -> list[dict]:
    """Scan all tree nodes for contradictions."""
    print(f"Loading claims for view '{view_id}'...", flush=True)

    with db.get_conn() as conn:
        rows = conn.execute("SELECT data FROM claims").fetchall()

    claims = [json.loads(r["data"]) for r in rows]
    print(f"Loaded {len(claims):,} claims", flush=True)

    by_node: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        view_paths = claim.get("view_paths", {})
        if isinstance(view_paths, str):
            try:
                view_paths = json.loads(view_paths)
            except (json.JSONDecodeError, TypeError):
                continue

        path_segments = view_paths.get(view_id)
        if not path_segments or not isinstance(path_segments, list):
            continue

        for depth in range(1, len(path_segments) + 1):
            node_key = "/".join(path_segments[:depth])
            by_node[node_key].append(claim)

    print(f"Grouped into {len(by_node):,} nodes", flush=True)

    all_contradictions = []
    for node_path, node_claims in by_node.items():
        if len(node_claims) < 2:
            continue
        contras = find_contradictions_in_group(node_claims, threshold)
        for c in contras:
            c["node_path"] = node_path
            c["view_id"] = view_id
        all_contradictions.extend(contras)

    seen = set()
    unique = []
    for c in all_contradictions:
        key = (c["claim_1_id"], c["claim_2_id"], c["field"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique.sort(key=lambda x: x["difference"], reverse=True)
    print(f"Found {len(unique):,} unique contradictions", flush=True)
    return unique


def main():
    parser = argparse.ArgumentParser(description="AskChem Contradiction Detection")
    sub = parser.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan", help="Scan for contradictions")
    scan_p.add_argument("--view", default="by_reaction_type")
    scan_p.add_argument("--threshold", type=float, default=20.0)
    scan_p.add_argument("--output", type=Path, help="Save results to JSON")
    scan_p.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()

    if args.command == "scan":
        results = scan_all_contradictions(view_id=args.view, threshold=args.threshold)
        results = results[:args.limit]

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} contradictions to {args.output}")
        else:
            for c in results[:20]:
                print(f"  [{c['claim_type']}] {c['subject']}: "
                      f"{c['field']} = {c['value_1']} vs {c['value_2']} "
                      f"(diff: {c['difference']}%) "
                      f"@ {c['node_path']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
