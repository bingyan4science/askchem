#!/usr/bin/env python3
"""Build and Gemini-review every canonical taxonomy sibling set.

The generated review units contain every L1/L2/L3 node, its siblings, support,
and representative claims.  Judging is resumable and writes one atomic JSON
cache keyed by review-unit ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.taxonomy_semantics import (  # noqa: E402
    chemical_identities,
    concept_key,
    normalize_slug,
)

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions"
PROVIDER = "@vertexai-gemini-kc119-2"
PRIMARY_MODEL = "gemini-3.1-pro-preview"
CHALLENGER_MODEL = "gemini-3.6-flash"
VIEW_CONTRACTS = {
    "by_reaction_type": (
        "Classify by the chemical transformation. Catalyst, energy source, "
        "and operating mode are subordinate qualifiers, not competing roots."
    ),
    "by_application": (
        "Classify by the end use or practical purpose, not by material, "
        "technique, or reaction mechanism."
    ),
    "by_technique": (
        "Classify by the experimental or computational method being used."
    ),
    "by_mechanism": (
        "Classify by the causal phenomenon or mechanistic process, not by "
        "measurement technique or application."
    ),
    "by_substance_class": (
        "This legacy view mixes composition and material form. Flag every "
        "mixed-axis node for migration to by_composition or by_material_form."
    ),
    "by_composition": "Classify only by chemical identity or composition.",
    "by_material_form": (
        "Classify only by morphology, dimensionality, architecture, or "
        "physical material form."
    ),
}
ALLOWED_ACTIONS = {
    "keep", "rename", "merge", "move", "promote", "demote", "split", "remove",
}

SYSTEM_PROMPT = """You are the senior ontology editor for a structured chemistry
knowledge base. Audit the complete sibling set supplied by the user. The view is
an independent classification facet; every parent-child edge must be an "is-a"
relationship appropriate to that facet.

For every node, assess:
1. scientific validity and fit under its parent;
2. overlap or synonymy with siblings or ancestors;
3. consistent granularity and useful specificity;
4. whether representative claims actually fit;
5. whether it should be kept, renamed, merged, moved, promoted, demoted, split,
   or removed.

Chemical formulas are exact identities. CO (carbon monoxide) is distinct from
CO2 (carbon dioxide); likewise preserve distinctions such as NO/NO2, N2/NH3,
and oxidation/reduction. Formula/name forms of the same species are aliases.
A high claim count is not evidence that a category is valid.

When review_scope is cross_path_exact, the nodes are competing placements of
one concept in one view. Select exactly one canonical path and merge or move
every other node to it. Do not keep multiple placements. Use the supplied
view_contract to choose the scientifically correct parent. A generic leaf in
different chemical contexts may be renamed into context-specific concepts only
when the full paths describe genuinely different entities.

When review_scope is cross_path_candidate, the nodes are a deterministic
semantic-overlap candidate, not a presumed duplicate. Keep both only when they
represent scientifically distinct concepts or a valid parent/subtype or
contrast relationship. Otherwise select exactly one canonical path and merge,
move, or remove the competing placement. Explicitly protect formula identity,
stoichiometry, opposing mechanisms, and meaningful composition distinctions.

Return one decision for every node_id, and only valid JSON:
{"decisions":[{"node_id":"...", "action":"keep|rename|merge|move|promote|demote|split|remove",
"canonical_label":"lowercase_underscore_slug", "canonical_path":"slash/path",
"confidence":"high|medium|low", "rationale":"concise chemistry rationale"}],
"set_assessment":{"coherent":true,"issues":["..."]}}

For keep, canonical_path must equal the current path. For merge/rename/move/
promote/demote, provide the exact proposed canonical path. Use split only when
the representative claims demonstrate multiple genuinely distinct concepts.
Default to keep with low confidence when evidence is insufficient."""


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _canonical_nodes(spec: dict) -> list[dict]:
    nodes = []
    for view, l1_values in spec["canonical_l1"].items():
        nodes.append({
            "unit_id": f"{view}:root",
            "view": view,
            "level": 1,
            "parent_path": "",
            "children": list(l1_values),
        })
        for l1 in l1_values:
            l2_values = spec["canonical_l2"].get(view, {}).get(l1, [])
            if l2_values:
                nodes.append({
                    "unit_id": f"{view}:{l1}",
                    "view": view,
                    "level": 2,
                    "parent_path": l1,
                    "children": list(l2_values),
                })
            for l2 in l2_values:
                key = f"{l1}/{l2}"
                l3_values = spec["canonical_l3"].get(view, {}).get(key, [])
                if l3_values:
                    nodes.append({
                        "unit_id": f"{view}:{key}",
                        "view": view,
                        "level": 3,
                        "parent_path": key,
                        "children": list(l3_values),
                    })
    return nodes


def build_units(db_path: Path, spec_path: Path, output: Path) -> None:
    spec = json.loads(spec_path.read_text())
    units = _canonical_nodes(spec)
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    # Older tree builds stored direct rather than recursive support. Aggregate
    # claim_view_map prefixes so every node gets its true subtree count.
    support: dict[tuple[str, str], int] = {}
    samples_by_path: dict[tuple[str, str], list[str]] = {}
    content_views = set(spec["canonical_l1"])
    for row in conn.execute(
        "SELECT claim_id,view_paths FROM claims WHERE view_paths IS NOT NULL"
    ):
        try:
            view_paths = json.loads(row["view_paths"])
        except (json.JSONDecodeError, TypeError):
            continue
        for view, path in view_paths.items():
            if view not in content_views or not isinstance(path, list):
                continue
            parts = [normalize_slug(part) for part in path if part][:3]
            for depth in range(1, len(parts) + 1):
                key = (view, "/".join(parts[:depth]))
                support[key] = support.get(key, 0) + 1
                sample = samples_by_path.setdefault(key, [])
                if len(sample) < 2:
                    sample.append(row["claim_id"])
    sample_ids = set()
    for unit in units:
        for child in unit["children"]:
            path = "/".join(filter(None, [unit["parent_path"], child]))
            sample_ids.update(samples_by_path.get((unit["view"], path), []))
    quotes = {}
    ids = sorted(sample_ids)
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT claim_id,verbatim_quote FROM claims "
            f"WHERE claim_id IN ({marks})",
            chunk,
        ):
            quotes[row["claim_id"]] = (row["verbatim_quote"] or "")[:500]
    conn.close()

    rendered = []
    rendered_nodes = {}
    for unit in units:
        children = []
        for child in unit.pop("children"):
            path = "/".join(filter(None, [unit["parent_path"], child]))
            children.append({
                "node_id": f"{unit['view']}:{path}",
                "label": child,
                "path": path,
                "claim_count": support.get((unit["view"], path), 0),
                "concept_key": concept_key(child),
                "chemical_identities": sorted(chemical_identities(child)),
                "sample_claims": [
                    quotes[claim_id]
                    for claim_id in samples_by_path.get(
                        (unit["view"], path), []
                    )
                    if quotes.get(claim_id)
                ],
            })
        rendered_unit = {**unit, "review_scope": "siblings", "nodes": children}
        rendered.append(rendered_unit)
        for node in children:
            rendered_nodes[(unit["view"], node["path"])] = node

    # A sibling-only review cannot see duplicate concepts placed under another
    # branch. Add explicit within-view cross-path units so nested duplicates
    # such as electrocatalysis/reduction/CO2 reduction are adjudicated together.
    by_concept: dict[tuple[str, str], list[dict]] = {}
    for (view, _), node in rendered_nodes.items():
        if node["concept_key"] and node["concept_key"] != "other":
            by_concept.setdefault((view, node["concept_key"]), []).append(node)
    cross_groups: dict[str, list[tuple[str, list[dict]]]] = {}
    for (view, concept), concept_nodes in sorted(by_concept.items()):
        unique_paths = {node["path"] for node in concept_nodes}
        if len(unique_paths) < 2:
            continue
        cross_groups.setdefault(view, []).append((concept, concept_nodes))
    for view, groups in sorted(cross_groups.items()):
        for concept, concept_nodes in groups:
            cross_nodes = [
                {**node, "cross_path_concept": concept}
                for node in concept_nodes
            ]
            rendered.append({
                "unit_id": f"{view}:cross_path:{concept}",
                "view": view,
                "level": "cross_path",
                "parent_path": "",
                "review_scope": "cross_path_exact",
                "view_contract": VIEW_CONTRACTS.get(view, ""),
                "required_disposition": "one_canonical_path",
                "nodes": sorted(
                    cross_nodes,
                    key=lambda node: node["path"],
                ),
            })
    _atomic_json(output, {
        "taxonomy_version": spec["taxonomy_version"],
        "source_db": str(db_path.resolve()),
        "source_spec": str(spec_path),
        "unit_count": len(rendered),
        "node_count": sum(len(unit["nodes"]) for unit in rendered),
        "units": rendered,
    })


def build_candidate_units(
    db_path: Path,
    spec_path: Path,
    audit_path: Path,
    output: Path,
) -> None:
    """Build one review unit for every deterministic semantic candidate."""
    spec = json.loads(spec_path.read_text())
    audit = json.loads(audit_path.read_text())
    contracts = {
        view: VIEW_CONTRACTS.get(view, "")
        for view in spec["canonical_l1"]
    }
    candidates: list[tuple[str, dict]] = []
    for view, report in audit.get("concept_placement", {}).items():
        if view not in contracts:
            continue
        for key in (
            "near_synonyms",
            "high_confidence_near_synonyms",
            "shallow_deep",
            "ancestor_redundancy",
            "scoped_duplicates",
            "scoped_label_duplicates",
            "semantic_id_duplicates",
            "repeated_semantic_ids",
        ):
            values = report.get(key, [])
            if isinstance(values, dict):
                expanded = []
                for semantic_id, paths in values.items():
                    if not isinstance(paths, list):
                        continue
                    for left, right in combinations(sorted(set(paths)), 2):
                        expanded.append({
                            "left": left,
                            "right": right,
                            "kind": key,
                            "semantic_id": semantic_id,
                        })
                values = expanded
            for value in values:
                if value.get("left") and value.get("right"):
                    candidates.append((view, {**value, "report_class": key}))
    for view, records in audit.get("axis_leakage", {}).items():
        if view not in contracts:
            continue
        for value in records:
            if value.get("path"):
                candidates.append((view, {
                    **value,
                    "paths": [value["path"]],
                    "kind": "axis_leakage",
                    "report_class": "axis_leakage",
                }))

    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    node_rows = {}
    sample_ids: set[str] = set()
    for row in conn.execute(
        "SELECT view_id,path,claim_count,claim_ids FROM tree_nodes"
    ):
        key = (row["view_id"], row["path"])
        try:
            claim_ids = json.loads(row["claim_ids"] or "[]")[:2]
        except (json.JSONDecodeError, TypeError):
            claim_ids = []
        node_rows[key] = {
            "claim_count": row["claim_count"],
            "claim_ids": claim_ids,
        }
        sample_ids.update(claim_ids)
    quotes = {}
    ids = sorted(sample_ids)
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT claim_id,verbatim_quote FROM claims "
            f"WHERE claim_id IN ({marks})",
            chunk,
        ):
            quotes[row["claim_id"]] = (row["verbatim_quote"] or "")[:500]
    conn.close()

    units = []
    seen = set()
    for view, candidate in candidates:
        paths = tuple(sorted(set(
            candidate.get("paths")
            or (candidate["left"], candidate["right"])
        )))
        dedup_key = (view, paths)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        digest = hashlib.sha256(
            f"{view}:{':'.join(paths)}".encode()
        ).hexdigest()[:16]
        nodes = []
        for path in paths:
            row = node_rows.get((view, path), {})
            leaf = path.rsplit("/", 1)[-1]
            nodes.append({
                "node_id": f"{view}:{path}",
                "label": leaf,
                "path": path,
                "claim_count": row.get("claim_count", 0),
                "concept_key": concept_key(leaf),
                "chemical_identities": sorted(chemical_identities(path)),
                "sample_claims": [
                    quotes[claim_id]
                    for claim_id in row.get("claim_ids", [])
                    if quotes.get(claim_id)
                ],
            })
        units.append({
            "unit_id": f"{view}:semantic_candidate:{digest}",
            "view": view,
            "level": "cross_path",
            "parent_path": "",
            "review_scope": "cross_path_candidate",
            "view_contract": contracts[view],
            "candidate_kind": candidate.get("kind", "semantic_overlap"),
            "candidate_score": candidate.get("score"),
            "report_class": candidate["report_class"],
            "nodes": nodes,
        })
    _atomic_json(output, {
        "taxonomy_version": spec["taxonomy_version"],
        "source_db": str(db_path.resolve()),
        "source_spec": str(spec_path),
        "source_audit": str(audit_path),
        "unit_count": len(units),
        "node_count": sum(len(unit["nodes"]) for unit in units),
        "units": units,
    })


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def call_gemini(
    unit: dict, model: str, timeout: int, review_role: str,
) -> dict:
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    role_instruction = (
        "\nAct as an independent adversarial reviewer. Prefer a different "
        "analysis route, actively look for false merges and invalid hierarchy "
        "edges, and do not assume another review is correct."
        if review_role == "challenger" else ""
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + role_instruction},
            {"role": "user", "content": json.dumps(unit, ensure_ascii=False)},
        ],
        "thinking_level": "high",
        "temperature": 0.1,
        "max_completion_tokens": 32768,
        "response_format": {"type": "json_object"},
    }
    response = _session().post(
        GATEWAY,
        headers={
            "x-portkey-api-key": api_key,
            "x-portkey-provider": PROVIDER,
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini {model} returned {response.status_code}: "
            f"{response.text[:500]}"
        )
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Gemini {model} returned no choices")
    parsed = json.loads(choices[0]["message"]["content"])
    decisions = parsed.get("decisions")
    expected = {node["node_id"] for node in unit["nodes"]}
    actual = {
        decision.get("node_id")
        for decision in decisions or []
        if isinstance(decision, dict)
    }
    if actual != expected:
        raise ValueError(
            f"decision coverage mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for decision in decisions:
        if decision.get("action") not in ALLOWED_ACTIONS:
            raise ValueError(f"invalid action: {decision.get('action')!r}")
    return {
        "model": model,
        "thinking_level": "high",
        "review_role": review_role,
        "result": parsed,
        "usage": payload.get("usage") or {},
    }


def judge(
    input_path: Path,
    output: Path,
    model: str,
    workers: int,
    timeout: int,
    limit: int,
    unit_id: str | None,
    review_role: str,
) -> None:
    units = json.loads(input_path.read_text())["units"]
    if unit_id:
        units = [unit for unit in units if unit["unit_id"] == unit_id]
        if not units:
            raise ValueError(f"unknown review unit: {unit_id}")
    cache = json.loads(output.read_text()) if output.exists() else {}
    pending = [unit for unit in units if unit["unit_id"] not in cache]
    if limit:
        pending = pending[:limit]
    lock = threading.Lock()
    failures = []

    def run(unit: dict) -> tuple[str, dict]:
        last_error = ""
        for attempt in range(3):
            try:
                return unit["unit_id"], call_gemini(
                    unit, model, timeout, review_role,
                )
            except Exception as exc:  # network/model errors are retried
                last_error = str(exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(last_error)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, unit): unit for unit in pending}
        completed = 0
        for future in as_completed(futures):
            unit = futures[future]
            try:
                unit_id, result = future.result()
            except Exception as exc:
                failures.append({"unit_id": unit["unit_id"], "error": str(exc)})
                continue
            with lock:
                cache[unit_id] = result
                _atomic_json(output, cache)
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] reviewed {unit_id}",
                    flush=True,
                )
    if failures:
        failure_path = output.with_suffix(output.suffix + ".failures.json")
        _atomic_json(failure_path, failures)
        raise RuntimeError(
            f"{len(failures)} review units failed; see {failure_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--db", type=Path, default=ROOT / "askchem.db")
    build.add_argument(
        "--spec", type=Path, default=ROOT / "src/askchem/taxonomy_v2.json",
    )
    build.add_argument("--output", type=Path, required=True)
    candidates = sub.add_parser("build-candidates")
    candidates.add_argument("--db", type=Path, default=ROOT / "askchem.db")
    candidates.add_argument(
        "--spec", type=Path, default=ROOT / "src/askchem/taxonomy_v2.json",
    )
    candidates.add_argument("--audit", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("judge")
    review.add_argument("--input", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--model", default=PRIMARY_MODEL)
    review.add_argument("--workers", type=int, default=4)
    review.add_argument("--timeout", type=int, default=600)
    review.add_argument("--limit", type=int, default=0)
    review.add_argument("--unit")
    review.add_argument(
        "--review-role", choices=("primary", "challenger"), default="primary",
    )
    args = parser.parse_args()
    if args.command == "build":
        build_units(args.db, args.spec, args.output)
    elif args.command == "build-candidates":
        build_candidate_units(args.db, args.spec, args.audit, args.output)
    else:
        judge(
            args.input, args.output, args.model,
            args.workers, args.timeout, args.limit, args.unit,
            args.review_role,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
