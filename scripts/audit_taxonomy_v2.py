#!/usr/bin/env python3
"""Audit canonical-path quality and tree shape for an AskChem database."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.taxonomy import (  # noqa: E402
    ALL_CONTENT_VIEWS,
    CANONICAL_L1,
    CANONICAL_L2,
    CANONICAL_L3,
)
from askchem.taxonomy_semantics import (  # noqa: E402
    chemical_family_key,
    chemical_identities,
    concept_key,
    concept_signature,
    duplicate_groups,
    high_confidence_near_synonym_pairs,
    ancestor_redundancy_pairs,
    near_synonym_pairs,
    scoped_concept_key,
    shallow_deep_duplicates,
    soft_concept_key,
)


class AuditReportClass(str, Enum):
    """Stable machine-readable classes used by review and release gates."""

    EXACT_ALIAS = "exact_alias"
    SOFT_ALIAS = "soft_alias"
    SHALLOW_DEEP = "shallow_deep"
    NEAR_SYNONYM = "near_synonym"
    FORMULA_CONFLICT = "formula_conflict"
    SCOPED_LABEL_DUPLICATE = "scoped_label_duplicate"
    AXIS_LEAKAGE = "axis_leakage"
    SEMANTIC_ID_DUPLICATE = "semantic_id_duplicate"
    UNRESOLVED_ADJUDICATION = "unresolved_adjudication"
    FANOUT = "fanout"


@dataclass(frozen=True)
class FanoutViolation:
    view: str
    parent_path: str
    level: int
    child_count: int
    limit: int


@dataclass(frozen=True)
class HardGateResult:
    passed: bool
    single_placement_violations: int
    fanout_violations: int
    failed_gates: tuple[str, ...]
    high_confidence_near_synonyms: int = 0
    ancestor_redundancies: int = 0
    scoped_label_duplicates: int = 0
    axis_leakage_violations: int = 0
    repeated_semantic_ids: int = 0
    unresolved_adjudications: int = 0


ADJUDICATION_STATUSES = frozenset({"merge", "rehome", "keep", "remove"})

# Contracts describe concepts that belong to another primary taxonomy axis.
# They are deliberately narrow: chemistry words shared by several axes are
# not evidence of leakage.
VIEW_AXIS_CONTRACTS = {
    "by_reaction_type": {
        "forbidden": {
            "spectroscopy", "microscopy", "chromatography", "diffraction",
            "battery", "sensor", "drug_delivery",
        },
    },
    "by_substance_class": {
        "forbidden": {
            "oxidation", "reduction", "coupling", "spectroscopy",
            "microscopy", "catalysis_application",
        },
    },
    "by_technique": {
        "forbidden": {
            "oxidation", "reduction", "hydrogenation", "polymerization",
            "drug_delivery", "energy_storage",
        },
    },
    "by_application": {
        "forbidden": {
            "spectroscopy", "microscopy", "chromatography", "diffraction",
            "oxidation", "reduction",
        },
    },
    "by_mechanism": {
        "forbidden": {
            "spectroscopy", "microscopy", "chromatography", "drug_delivery",
            "energy_storage",
        },
    },
    # These views intentionally admit broad topical labels; keeping an
    # explicit empty contract prevents another axis's rules being inferred.
    "by_claim_type": {"forbidden": set()},
    "by_author": {"forbidden": set()},
    "by_data": {"forbidden": set()},
}


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return float(ordered[index])


def duplicate_values(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def canonical_paths(view: str) -> list[str]:
    """Return every canonical path in deterministic order."""
    paths = set(CANONICAL_L1.get(view, []))
    for l1, values in CANONICAL_L2.get(view, {}).items():
        paths.update(f"{l1}/{value}" for value in values)
    for (l1, l2), values in CANONICAL_L3.get(view, {}).items():
        paths.update(f"{l1}/{l2}/{value}" for value in values)
    return sorted(paths)


def install_candidate_spec(path: Path) -> None:
    """Install an explicit generated spec for pre-deployment release gates."""
    global ALL_CONTENT_VIEWS, CANONICAL_L1, CANONICAL_L2, CANONICAL_L3
    payload = json.loads(path.read_text())
    CANONICAL_L1 = payload["canonical_l1"]
    CANONICAL_L2 = payload["canonical_l2"]
    CANONICAL_L3 = {
        view: {
            tuple(parent.split("/", 1)): children
            for parent, children in parents.items()
        }
        for view, parents in payload["canonical_l3"].items()
    }
    ALL_CONTENT_VIEWS = list(CANONICAL_L1)


def find_fanout_violations(
    view: str, limit: int,
) -> list[FanoutViolation]:
    """Check root and canonical parents against one explicit fanout limit."""
    records = []
    root_count = len(CANONICAL_L1.get(view, []))
    if root_count > limit:
        records.append(FanoutViolation(view, "", 1, root_count, limit))
    for parent, children in sorted(CANONICAL_L2.get(view, {}).items()):
        if len(children) > limit:
            records.append(
                FanoutViolation(view, parent, 2, len(children), limit)
            )
    for parent, children in sorted(CANONICAL_L3.get(view, {}).items()):
        if len(children) > limit:
            records.append(
                FanoutViolation(view, "/".join(parent), 3, len(children), limit)
            )
    return records


def semantic_placement_report(view: str) -> dict:
    """Classify deterministic and review-only placement candidates."""
    paths = canonical_paths(view)
    exact = defaultdict(set)
    soft = defaultdict(set)
    signatures = defaultdict(set)
    scoped = defaultdict(set)
    for path in paths:
        leaf = path.rsplit("/", 1)[-1]
        signature = concept_signature(view, path)
        if signature.concept and signature.concept != "other":
            exact_context = "/".join(
                (*signature.context, concept_key(leaf))
            )
            if signature.chemical_identities:
                exact_context += "@" + ",".join(signature.chemical_identities)
            exact[f"{signature.view}:{exact_context}"].add(path)
            soft[signature.stable_id].add(path)
            signatures[signature.stable_id].add(path)
            scoped[scoped_concept_key(path)].add(path)

    exact_groups = {
        key: sorted(values)
        for key, values in sorted(exact.items())
        if len(values) > 1
    }
    soft_groups = {
        key: sorted(values)
        for key, values in sorted(soft.items())
        if len(values) > 1
        and len({concept_key(path.rsplit("/", 1)[-1]) for path in values}) > 1
    }
    signature_groups = {
        key: sorted(values)
        for key, values in sorted(signatures.items())
        if len(values) > 1
    }
    scoped_groups = {
        key: sorted(values)
        for key, values in sorted(scoped.items())
        if key and len(values) > 1
        and len({soft_concept_key(path.rsplit("/", 1)[-1]) for path in values}) > 1
    }
    return {
        "exact_aliases": exact_groups,
        "soft_aliases": soft_groups,
        "path_signature_duplicates": signature_groups,
        "repeated_semantic_ids": signature_groups,
        "scoped_label_duplicates": scoped_groups,
        "shallow_deep": [
            asdict(pair) for pair in shallow_deep_duplicates(view, paths)
        ],
        "ancestor_redundancy": [
            asdict(pair) for pair in ancestor_redundancy_pairs(view, paths)
        ],
        "near_synonyms": [
            asdict(pair) for pair in near_synonym_pairs(view, paths)
        ],
        "high_confidence_near_synonyms": [
            asdict(pair)
            for pair in high_confidence_near_synonym_pairs(view, paths)
        ],
    }


def find_axis_leakage(
    view: str,
    paths: list[str],
    contracts: dict | None = None,
) -> list[dict]:
    """Return deterministic paths violating a view's explicit axis contract."""
    contract = (
        VIEW_AXIS_CONTRACTS if contracts is None else contracts
    ).get(view, {})
    forbidden = set(contract.get("forbidden", ()))
    records = []
    for path in sorted(set(paths)):
        leaf = soft_concept_key(path.rsplit("/", 1)[-1])
        matches = sorted(
            term for term in forbidden
            if leaf == term or leaf.startswith(term + "_") or leaf.endswith("_" + term)
        )
        if matches:
            records.append({
                "path": path,
                "semantic_id": concept_signature(view, path).stable_id,
                "forbidden_concepts": matches,
            })
    return records


def _candidate_id(report_class: str, view: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return f"{report_class}:{view}:{digest}"


def adjudication_report(report: dict, decisions: object = None) -> dict:
    """List every actionable candidate and require a valid terminal decision."""
    decision_map = {}
    if isinstance(decisions, dict):
        source = decisions.get("adjudications", decisions)
        if isinstance(source, dict):
            decision_map = source
        elif isinstance(source, list):
            decision_map = {
                str(item.get("candidate_id")): item.get("status")
                for item in source if isinstance(item, dict)
            }
    elif isinstance(decisions, list):
        decision_map = {
            str(item.get("candidate_id")): item.get("status")
            for item in decisions if isinstance(item, dict)
        }

    candidate_sources = (
        ("high_confidence_near_synonym", "high_confidence_near_synonyms"),
        ("ancestor_redundancy", "ancestor_redundancy"),
        ("scoped_label_duplicate", "scoped_label_duplicates"),
        ("semantic_id_duplicate", "repeated_semantic_ids"),
    )
    candidates = []
    for view, placement in sorted(report.get("concept_placement", {}).items()):
        for report_class, key in candidate_sources:
            values = placement.get(key, {})
            iterable = (
                [{"semantic_id": group, "paths": paths}
                 for group, paths in sorted(values.items())]
                if isinstance(values, dict) else values
            )
            for payload in iterable:
                candidate_id = _candidate_id(report_class, view, payload)
                raw = decision_map.get(candidate_id)
                status = raw.get("status") if isinstance(raw, dict) else raw
                candidates.append({
                    "candidate_id": candidate_id,
                    "class": report_class,
                    "view": view,
                    "candidate": payload,
                    "status": status if status in ADJUDICATION_STATUSES else "unresolved",
                })
    for view, records in sorted(report.get("axis_leakage", {}).items()):
        for payload in records:
            candidate_id = _candidate_id("axis_leakage", view, payload)
            raw = decision_map.get(candidate_id)
            status = raw.get("status") if isinstance(raw, dict) else raw
            candidates.append({
                "candidate_id": candidate_id,
                "class": "axis_leakage",
                "view": view,
                "candidate": payload,
                "status": status if status in ADJUDICATION_STATUSES else "unresolved",
            })
    counts = Counter(item["status"] for item in candidates)
    return {
        "allowed_statuses": sorted(ADJUDICATION_STATUSES),
        "candidate_count": len(candidates),
        "status_counts": dict(sorted(counts.items())),
        "unresolved_count": counts["unresolved"],
        "complete": counts["unresolved"] == 0,
        "candidates": candidates,
    }


def evaluate_hard_gates(report: dict) -> HardGateResult:
    """Summarize deterministic release-gate failures from an audit report."""
    placement = 0
    for record in report.get("concept_placement", {}).values():
        placement += len(record.get("exact_aliases", {}))
        placement += len(record.get("soft_aliases", {}))
        placement += len(record.get("shallow_deep", []))
    fanout = sum(
        len(records)
        for records in report.get("fanout_violations", {}).values()
    )
    high_confidence = sum(
        len(record.get("high_confidence_near_synonyms", []))
        for record in report.get("concept_placement", {}).values()
    )
    ancestor = sum(
        len(record.get("ancestor_redundancy", []))
        for record in report.get("concept_placement", {}).values()
    )
    scoped = sum(
        len(record.get("scoped_label_duplicates", {}))
        for record in report.get("concept_placement", {}).values()
    )
    semantic_ids = sum(
        len(record.get("repeated_semantic_ids", {}))
        for record in report.get("concept_placement", {}).values()
    )
    axis = sum(len(values) for values in report.get("axis_leakage", {}).values())
    unresolved = report.get("adjudication", {}).get("unresolved_count", 0)
    failed = []
    if placement:
        failed.append("single_placement")
    if fanout:
        failed.append("fanout")
    if high_confidence:
        failed.append("high_confidence_near_synonym")
    if ancestor:
        failed.append("ancestor_redundancy")
    if scoped:
        failed.append("scoped_label_duplicate")
    if axis:
        failed.append("axis_leakage")
    if semantic_ids:
        failed.append("semantic_id_duplicate")
    if unresolved:
        failed.append("unresolved_adjudication")
    return HardGateResult(
        passed=not failed,
        single_placement_violations=placement,
        fanout_violations=fanout,
        failed_gates=tuple(failed),
        high_confidence_near_synonyms=high_confidence,
        ancestor_redundancies=ancestor,
        scoped_label_duplicates=scoped,
        axis_leakage_violations=axis,
        repeated_semantic_ids=semantic_ids,
        unresolved_adjudications=unresolved,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "askchem.db")
    parser.add_argument(
        "--spec",
        type=Path,
        help="Audit an explicit generated taxonomy spec before installation",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-fanout", type=int, default=30)
    parser.add_argument(
        "--include-view",
        action="append",
        default=[],
        metavar="VIEW_ID",
        help="Audit an additional retained/generated view (repeatable)",
    )
    parser.add_argument(
        "--fail-on-single-placement",
        "--fail-on-duplicates",
        action="store_true",
        dest="fail_on_single_placement",
    )
    parser.add_argument("--fail-on-fanout", action="store_true")
    parser.add_argument("--fail-on-near-synonyms", action="store_true")
    parser.add_argument("--fail-on-ancestor-redundancy", action="store_true")
    parser.add_argument("--fail-on-scoped-duplicates", action="store_true")
    parser.add_argument("--fail-on-axis-leakage", action="store_true")
    parser.add_argument("--fail-on-semantic-ids", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    parser.add_argument(
        "--adjudications",
        type=Path,
        help="JSON decisions keyed by candidate_id (merge/rehome/keep/remove)",
    )
    parser.add_argument(
        "--hard-gate",
        action="store_true",
        help="Fail on every taxonomy release gate",
    )
    args = parser.parse_args()
    if args.max_fanout < 1:
        parser.error("--max-fanout must be positive")
    if args.spec:
        install_candidate_spec(args.spec)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    retained_views = sorted(
        set(ALL_CONTENT_VIEWS)
        | set(args.include_view)
        | set(CANONICAL_L1)
        | set(CANONICAL_L2)
        | set(CANONICAL_L3)
        | {
            row[0] for row in conn.execute(
                "SELECT DISTINCT view_id FROM tree_nodes"
            ) if row[0]
        }
    )
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.db.resolve()),
        "retained_views": retained_views,
        "claim_count": conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
        "tree_node_count": conn.execute(
            "SELECT COUNT(*) FROM tree_nodes"
        ).fetchone()[0],
        "claim_view_map_count": conn.execute(
            "SELECT COUNT(*) FROM claim_view_map"
        ).fetchone()[0],
        "views": {},
        "canonical_definition_duplicates": {},
        "semantic_definition_duplicates": {},
        "ancestor_descendant_duplicates": {},
        "cross_path_concept_duplicates": {},
        "formula_guard_conflicts": {},
        "concept_placement": {},
        "axis_leakage": {},
        "fanout_violations": {},
        "report_classes": {
            item.value: (
                "hard_violation"
                if item not in {AuditReportClass.NEAR_SYNONYM,
                                AuditReportClass.FORMULA_CONFLICT}
                else "review_candidate"
            )
            for item in AuditReportClass
        },
    }

    stats = {
        view: {
            "assignments": 0,
            "depth": Counter(),
            "other_assignments": 0,
            "invalid_l1": Counter(),
            "invalid_l2": Counter(),
            "invalid_l3": Counter(),
        }
        for view in retained_views
    }
    cursor = conn.execute(
        "SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL"
    )
    while True:
        rows = cursor.fetchmany(20_000)
        if not rows:
            break
        for row in rows:
            try:
                view_paths = json.loads(row["view_paths"])
            except (json.JSONDecodeError, TypeError):
                continue
            # Include generated views (notably by_data) when supplied in data.
            for view in retained_views:
                path = view_paths.get(view)
                if not isinstance(path, list) or not path:
                    continue
                rec = stats[view]
                rec["assignments"] += 1
                rec["depth"][len(path)] += 1
                if "other" in path:
                    rec["other_assignments"] += 1
                l1 = path[0]
                if l1 not in CANONICAL_L1.get(view, []):
                    rec["invalid_l1"][l1] += 1
                    continue
                if len(path) >= 2:
                    l2 = path[1]
                    allowed_l2 = CANONICAL_L2.get(view, {}).get(l1, [])
                    if l2 not in allowed_l2:
                        rec["invalid_l2"][f"{l1}/{l2}"] += 1
                        continue
                    if len(path) >= 3:
                        allowed_l3 = CANONICAL_L3.get(view, {}).get((l1, l2))
                        if allowed_l3 is None or path[2] not in allowed_l3:
                            rec["invalid_l3"]["/".join(path[:3])] += 1

    for view in retained_views:
        tree_rows = conn.execute(
            "SELECT level, children FROM tree_nodes WHERE view_id = ?",
            (view,),
        ).fetchall()
        level_counts = Counter()
        fanouts = []
        for row in tree_rows:
            level_counts[int(row["level"])] += 1
            try:
                fanouts.append(len(json.loads(row["children"] or "[]")))
            except (json.JSONDecodeError, TypeError):
                fanouts.append(0)
        rec = stats[view]
        assignments = rec["assignments"]
        report["views"][view] = {
            "assignments": assignments,
            "depth_counts": {
                str(depth): count
                for depth, count in sorted(rec["depth"].items())
            },
            "max_depth": max(rec["depth"], default=0),
            "other_assignments": rec["other_assignments"],
            "other_rate": round(
                rec["other_assignments"] / max(1, assignments), 6
            ),
            "invalid_l1_assignments": sum(rec["invalid_l1"].values()),
            "invalid_l2_assignments": sum(rec["invalid_l2"].values()),
            "invalid_l3_assignments": sum(rec["invalid_l3"].values()),
            "top_invalid_l2": rec["invalid_l2"].most_common(25),
            "top_invalid_l3": rec["invalid_l3"].most_common(25),
            "tree_nodes_by_level": {
                str(level): count
                for level, count in sorted(level_counts.items())
            },
            "fanout": {
                "median": statistics.median(fanouts) if fanouts else 0,
                "p95": percentile(fanouts, 0.95),
                "max": max(fanouts, default=0),
            },
        }

        duplicate_l2 = {}
        for l1, values in CANONICAL_L2.get(view, {}).items():
            duplicates = duplicate_values(values)
            if duplicates:
                duplicate_l2[l1] = duplicates
        duplicate_l3 = {}
        for (l1, l2), values in CANONICAL_L3.get(view, {}).items():
            duplicates = duplicate_values(values)
            if duplicates:
                duplicate_l3[f"{l1}/{l2}"] = duplicates
        report["canonical_definition_duplicates"][view] = {
            "l1": duplicate_values(CANONICAL_L1.get(view, [])),
            "l2": duplicate_l2,
            "l3": duplicate_l3,
        }

        semantic_l2 = {}
        formula_conflicts = []
        for l1, values in CANONICAL_L2.get(view, {}).items():
            groups = duplicate_groups(values)
            if groups:
                semantic_l2[l1] = groups
            by_family = defaultdict(list)
            for value in values:
                identities = chemical_identities(value)
                if identities:
                    by_family[chemical_family_key(value)].append(
                        (value, sorted(identities))
                    )
            for family, labels in by_family.items():
                identities = {tuple(item[1]) for item in labels}
                if family and len(labels) > 1 and len(identities) > 1:
                    formula_conflicts.append({
                        "parent": l1,
                        "family": family,
                        "labels": [
                            {"name": name, "identities": ids}
                            for name, ids in labels
                        ],
                    })

        semantic_l3 = {}
        ancestor_duplicates = []
        for (l1, l2), values in CANONICAL_L3.get(view, {}).items():
            groups = duplicate_groups(values)
            if groups:
                semantic_l3[f"{l1}/{l2}"] = groups
            ancestor_keys = {concept_key(l1), concept_key(l2)}
            for value in values:
                if concept_key(value) in ancestor_keys:
                    ancestor_duplicates.append(f"{l1}/{l2}/{value}")

        report["semantic_definition_duplicates"][view] = {
            "l1": duplicate_groups(CANONICAL_L1.get(view, [])),
            "l2": semantic_l2,
            "l3": semantic_l3,
        }
        report["ancestor_descendant_duplicates"][view] = sorted(
            ancestor_duplicates
        )
        paths_by_concept = defaultdict(set)
        for l1 in CANONICAL_L1.get(view, []):
            if concept_key(l1) != "other":
                paths_by_concept[concept_key(l1)].add(l1)
        for l1, values in CANONICAL_L2.get(view, {}).items():
            for value in values:
                if concept_key(value) != "other":
                    paths_by_concept[concept_key(value)].add(f"{l1}/{value}")
        for (l1, l2), values in CANONICAL_L3.get(view, {}).items():
            for value in values:
                if concept_key(value) != "other":
                    paths_by_concept[concept_key(value)].add(
                        f"{l1}/{l2}/{value}"
                    )
        report["cross_path_concept_duplicates"][view] = {
            concept: sorted(paths)
            for concept, paths in sorted(paths_by_concept.items())
            if len(paths) > 1
        }
        report["formula_guard_conflicts"][view] = formula_conflicts
        report["concept_placement"][view] = semantic_placement_report(view)
        report["axis_leakage"][view] = find_axis_leakage(
            view, canonical_paths(view)
        )
        report["fanout_violations"][view] = [
            asdict(record)
            for record in find_fanout_violations(view, args.max_fanout)
        ]

    conn.close()
    decisions = None
    if args.adjudications:
        try:
            decisions = json.loads(args.adjudications.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --adjudications: {exc}")
    report["adjudication"] = adjudication_report(report, decisions)
    gate = evaluate_hard_gates(report)
    report["hard_gates"] = asdict(gate)
    report["counts"] = {
        "retained_views": len(retained_views),
        "single_placement_violations": gate.single_placement_violations,
        "fanout_violations": gate.fanout_violations,
        "high_confidence_near_synonyms": gate.high_confidence_near_synonyms,
        "ancestor_redundancies": gate.ancestor_redundancies,
        "scoped_label_duplicates": gate.scoped_label_duplicates,
        "axis_leakage_violations": gate.axis_leakage_violations,
        "repeated_semantic_ids": gate.repeated_semantic_ids,
        "adjudication_candidates": report["adjudication"]["candidate_count"],
        "unresolved_adjudications": gate.unresolved_adjudications,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    fail_placement = (
        (args.hard_gate or args.fail_on_single_placement)
        and gate.single_placement_violations > 0
    )
    fail_fanout = (
        (args.hard_gate or args.fail_on_fanout)
        and gate.fanout_violations > 0
    )
    fail_near = (
        (args.hard_gate or args.fail_on_near_synonyms)
        and gate.high_confidence_near_synonyms > 0
    )
    fail_ancestor = (
        (args.hard_gate or args.fail_on_ancestor_redundancy)
        and gate.ancestor_redundancies > 0
    )
    fail_scoped = (
        (args.hard_gate or args.fail_on_scoped_duplicates)
        and gate.scoped_label_duplicates > 0
    )
    fail_axis = (
        (args.hard_gate or args.fail_on_axis_leakage)
        and gate.axis_leakage_violations > 0
    )
    fail_semantic_ids = (
        (args.hard_gate or args.fail_on_semantic_ids)
        and gate.repeated_semantic_ids > 0
    )
    fail_unresolved = (
        (args.hard_gate or args.fail_on_unresolved)
        and gate.unresolved_adjudications > 0
    )
    return 1 if any((
        fail_placement, fail_fanout, fail_near, fail_ancestor, fail_scoped,
        fail_axis, fail_semantic_ids, fail_unresolved,
    )) else 0


if __name__ == "__main__":
    raise SystemExit(main())
