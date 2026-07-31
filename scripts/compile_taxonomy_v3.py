#!/usr/bin/env python3
"""Reconcile taxonomy reviews and compile a formula-safe canonical v3 spec."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.taxonomy_semantics import (  # noqa: E402
    FORMULA_NAMES,
    assert_formula_safe_alias,
    chemical_identities,
    concept_key,
    duplicate_groups,
    normalize_slug,
    soft_concept_key,
)

ACTIONABLE = {"rename", "merge", "move", "promote", "demote", "remove"}
FORMULA_TOKENS = set(FORMULA_NAMES)
GENERIC_SUFFIXES = {"reaction", "reactions", "process", "method", "technique"}
CURATED_SPLITS = {
    (
        "by_reaction_type",
        "synthesis/total_synthesis/protecting_group_free_and_flow_syntheses",
    ): ["protecting_group_free_synthesis", "flow_synthesis"],
    (
        "by_substance_class",
        "inorganic_compounds/metal_oxides/alumina_and_silica",
    ): ["alumina", "silica"],
    (
        "by_substance_class",
        "inorganic_compounds/other/sulfates_and_nitrates",
    ): ["sulfates", "nitrates"],
    (
        "by_substance_class",
        "coordination_compounds/transition_metal_complexes/"
        "palladium_platinum_gold_complexes",
    ): ["palladium_complexes", "platinum_complexes", "gold_complexes"],
    (
        "by_substance_class",
        "coordination_compounds/transition_metal_complexes/"
        "ruthenium_rhodium_iridium_complexes",
    ): ["ruthenium_complexes", "rhodium_complexes", "iridium_complexes"],
}
CURATED_CANONICAL_MAPPINGS = {
    # Reaction-type paths must use one hierarchy for each chemical
    # transformation.  The catalytic modality is retained as L3 instead of
    # duplicating CO2/CO reduction under both "reduction" and "catalysis".
    (
        "by_reaction_type",
        "reduction/co2_reduction",
    ): "reduction/carbon_dioxide_reduction",
    (
        "by_reaction_type",
        "catalysis/electrocatalysis/carbon_dioxide_reduction",
    ): "reduction/carbon_dioxide_reduction/electrocatalytic_reduction",
    (
        "by_reaction_type",
        "catalysis/photocatalysis/carbon_dioxide_reduction",
    ): "reduction/carbon_dioxide_reduction/photocatalytic_reduction",
    (
        "by_reaction_type",
        "catalysis/photocatalysis/co2_photoreduction",
    ): "reduction/carbon_dioxide_reduction/photocatalytic_reduction",
    (
        "by_reaction_type",
        "catalysis/electrocatalysis/carbon_monoxide_reduction",
    ): "reduction/carbon_monoxide_reduction/electrocatalytic_reduction",
    (
        "by_mechanism",
        "adsorption_and_surface/host_guest_interactions",
    ): "molecular_recognition/host_guest_interactions",
    (
        "by_substance_class",
        "nanomaterials/two_dimensional_materials",
    ): "semiconductors/two_dimensional_materials",
    (
        "by_application",
        "catalysis/electrocatalysis/hydrogen_evolution_reaction",
    ): "energy/water_splitting/hydrogen_evolution_reaction",
    (
        "by_application",
        "catalysis/electrocatalysis/oxygen_evolution_reaction",
    ): "energy/water_splitting/oxygen_evolution_reaction",
    (
        "by_mechanism",
        "conformational_and_structural/thermodynamics",
    ): "conformational_and_structural/other",
    (
        "by_mechanism",
        "conformational_and_structural/chirality",
    ): "conformational_and_structural/other",
}


def _sorted(values: set[str], include_other: bool = False) -> list[str]:
    values.discard("")
    values.discard("other")
    return sorted(values) + (["other"] if include_other else [])


def _preferred_label(labels: list[str]) -> str:
    def rank(label: str) -> tuple:
        tokens = label.split("_")
        return (
            any(token in FORMULA_TOKENS for token in tokens),
            bool(tokens and tokens[-1] in GENERIC_SUFFIXES),
            len(tokens),
            label,
        )
    return min(labels, key=rank)


def _all_paths(units: dict) -> dict[tuple[str, str], dict]:
    result = {}
    for unit in units["units"]:
        for node in unit["nodes"]:
            result[(unit["view"], node["path"])] = node
    return result


def _decisions(
    review: dict, review_scopes: dict[str, str],
) -> dict[str, dict]:
    result = {}
    ordered = sorted(
        review.items(),
        key=lambda item: (
            review_scopes.get(item[0], "").startswith("cross_path")
            or ":cross_path" in item[0],
            item[0],
        ),
    )
    for _, record in ordered:
        for decision in record.get("result", {}).get("decisions", []):
            result[decision["node_id"]] = decision
    return result


def _normalize_path(value: str) -> str:
    return "/".join(
        normalize_slug(part) for part in str(value).strip("/").split("/")
        if normalize_slug(part)
    )


def _apply_mapping(
    view: str, path: str, mappings: dict[tuple[str, str], str],
) -> str:
    current = path
    seen = set()
    while current not in seen:
        seen.add(current)
        replacement = mappings.get((view, current))
        if replacement:
            current = replacement
            continue
        parts = current.split("/")
        changed = False
        for depth in range(len(parts) - 1, 0, -1):
            prefix = "/".join(parts[:depth])
            target = mappings.get((view, prefix))
            if target:
                target_parts = target.split("/")
                current = "/".join([*target_parts, *parts[depth:]][:3])
                changed = True
                break
        if not changed:
            return current
    # Independent reviews can create a parent/child cycle (for example moving
    # a specific node under a parent that another decision removes). Preserve
    # the most specific reviewed state; the emitted registry is cycle-free
    # because it stores only this resolved result.
    return max(seen, key=lambda value: (len(value.split("/")), len(value)))


def _single_placement_key(view: str, path: str) -> tuple[str, tuple[str, ...]]:
    """Identify one concept while retaining chemical context from ancestors."""
    leaf = path.rsplit("/", 1)[-1]
    identities = tuple(sorted(chemical_identities(path.replace("/", "_"))))
    return soft_concept_key(leaf), identities


def _enforce_single_placement(
    nodes: dict[tuple[str, str], dict],
    mapping: dict[tuple[str, str], str],
) -> int:
    """Map every exact concept to one context-safe path in each view."""
    merges = 0
    while True:
        grouped: dict[tuple[str, str, tuple[str, ...]], dict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        for (view, old_path), node in nodes.items():
            resolved = _apply_mapping(view, old_path, mapping)
            concept, identities = _single_placement_key(view, resolved)
            if concept and concept != "other":
                grouped[(view, concept, identities)][resolved] += int(
                    node.get("claim_count") or 0
                )

        pass_merges = 0
        for (view, _, _), support in sorted(grouped.items()):
            if len(support) < 2:
                continue
            target = max(
                support,
                key=lambda path: (
                    support[path],
                    len(path.split("/")),
                    -len(path),
                    path,
                ),
            )
            for old_path in support:
                if old_path == target:
                    continue
                assert_formula_safe_alias(old_path, target)
                mapping[(view, old_path)] = target
                pass_merges += 1
        merges += pass_merges
        if not pass_merges:
            break
    return merges


def _map_generated_duplicate_paths(
    final_paths: dict[str, set[str]],
    mapping: dict[tuple[str, str], str],
) -> int:
    """Collapse duplicate parent paths introduced by mapped descendants."""
    merges = 0
    for view, paths in final_paths.items():
        grouped: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
        for path in paths:
            concept, identities = _single_placement_key(view, path)
            if concept and concept != "other":
                grouped[(concept, identities)].append(path)
        for candidates in grouped.values():
            if len(candidates) < 2:
                continue
            target = max(
                candidates,
                key=lambda path: (len(path.split("/")), -len(path), path),
            )
            for old_path in candidates:
                if old_path == target:
                    continue
                assert_formula_safe_alias(old_path, target)
                if mapping.get((view, old_path)) == target:
                    continue
                mapping[(view, old_path)] = target
                merges += 1
    return merges


def _enforce_fanout(
    nodes: dict[tuple[str, str], dict],
    mapping: dict[tuple[str, str], str],
    limit: int = 30,
) -> int:
    """Group low-support children under ``other`` until every parent is bounded."""
    merges = 0
    while True:
        children: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for (view, old_path), node in nodes.items():
            resolved = _apply_mapping(view, old_path, mapping)
            parts = resolved.split("/")
            support = int(node.get("claim_count") or 0)
            for depth, child in enumerate(parts):
                parent = "/".join(parts[:depth])
                children[(view, parent)][child] += support
        overfull = [
            (view, parent, support)
            for (view, parent), support in children.items()
            if len(support) > limit
        ]
        if not overfull:
            break
        pass_merges = 0
        for view, parent, support in sorted(overfull):
            named = set(support) - {"other"}
            keep = set(sorted(
                named,
                key=lambda child: (support[child], child),
                reverse=True,
            )[:limit - 1])
            for child in named - keep:
                old_path = "/".join(filter(None, [parent, child]))
                target = "/".join(filter(None, [parent, "other"]))
                if mapping.get((view, old_path)) == target:
                    continue
                mapping[(view, old_path)] = target
                pass_merges += 1
        merges += pass_merges
        if not pass_merges:
            break
    return merges


def compile_taxonomy(
    units: dict,
    primary: dict,
    challenger: dict,
    version: str,
    legacy_aliases: dict | None = None,
) -> tuple[dict, dict]:
    nodes = _all_paths(units)
    review_scopes = {
        unit["unit_id"]: unit.get("review_scope", "siblings")
        for unit in units["units"]
    }
    primary_decisions = _decisions(primary, review_scopes)
    challenger_decisions = _decisions(challenger, review_scopes)
    approved: dict[tuple[str, str], dict] = {}
    quarantined = []
    locked_canonical: set[tuple[str, str]] = set()
    allowed_roots: dict[str, set[str]] = defaultdict(set)
    for view, path in nodes:
        allowed_roots[view].add(path.split("/", 1)[0])

    # Deterministic sibling aliases are stronger than probabilistic judgments.
    for unit in units["units"]:
        labels = [node["label"] for node in unit["nodes"]]
        for concept, aliases in duplicate_groups(labels).items():
            target_label = _preferred_label(aliases)
            target_path = "/".join(
                filter(None, [unit["parent_path"], target_label])
            )
            locked_canonical.add((unit["view"], target_path))
            for alias in aliases:
                if alias == target_label:
                    continue
                old_path = "/".join(
                    filter(None, [unit["parent_path"], alias])
                )
                assert_formula_safe_alias(old_path, target_path)
                approved[(unit["view"], old_path)] = {
                    "view": unit["view"],
                    "old_path": old_path,
                    "new_path": target_path,
                    "status": "approved",
                    "action": "merge",
                    "confidence": "deterministic",
                    "reason": f"chemistry-aware alias group {concept}",
                    "source": "deterministic_semantic_normalization",
                }

    for view, aliases in (legacy_aliases or {}).get("aliases", {}).items():
        for old_path, new_path in aliases.items():
            if (view, old_path) not in nodes:
                continue
            assert_formula_safe_alias(old_path, new_path)
            approved[(view, old_path)] = {
                "view": view,
                "old_path": old_path,
                "new_path": new_path,
                "status": "approved",
                "action": "merge",
                "confidence": "curated",
                "reason": "carried forward from reviewed legacy alias registry",
                "source": "legacy_alias_registry",
            }
            locked_canonical.add((view, new_path))

    for (view, old_path), node in nodes.items():
        node_id = node["node_id"]
        left = primary_decisions.get(node_id)
        right = challenger_decisions.get(node_id)
        if not left or not right:
            quarantined.append({
                "view": view, "old_path": old_path,
                "status": "needs_review", "reason": "missing independent review",
            })
            continue
        if (view, old_path) in locked_canonical:
            continue
        action = left.get("action")
        target = _normalize_path(left.get("canonical_path", ""))
        if action == "remove" and "/" in old_path:
            target = old_path.rsplit("/", 1)[0]
        right_target = _normalize_path(right.get("canonical_path", ""))
        if right.get("action") == "remove" and "/" in old_path:
            right_target = old_path.rsplit("/", 1)[0]
        left_parent, _, left_leaf = target.rpartition("/")
        right_parent, _, right_leaf = right_target.rpartition("/")
        if (
            left_parent == right_parent
            and concept_key(left_leaf)
            and concept_key(left_leaf) == concept_key(right_leaf)
        ):
            preferred = _preferred_label([left_leaf, right_leaf])
            target = "/".join(filter(None, [left_parent, preferred]))
            right_target = target
        agrees = (
            action == right.get("action")
            and target == right_target
        )
        high = (
            left.get("confidence") == "high"
            and right.get("confidence") == "high"
        )
        if not agrees or not high or action not in ACTIONABLE:
            if left.get("action") != "keep" or right.get("action") != "keep":
                quarantined.append({
                    "view": view,
                    "old_path": old_path,
                    "status": "needs_review",
                    "primary": left,
                    "challenger": right,
                    "reason": "review disagreement, low confidence, or split/remove",
                })
            continue
        if action == "remove" and "/" not in old_path:
            quarantined.append({
                "view": view, "old_path": old_path,
                "status": "needs_review",
                "reason": "removing an L1 requires claim reclassification",
                "primary": left, "challenger": right,
            })
            continue
        if not target or len(target.split("/")) > 3 or target == old_path:
            quarantined.append({
                "view": view, "old_path": old_path, "new_path": target,
                "status": "rejected",
                "reason": "missing, over-depth, or unchanged actionable target",
                "primary": left, "challenger": right,
            })
            continue
        old_depth = len(old_path.split("/"))
        new_depth = len(target.split("/"))
        invalid_shape = (
            (action in {"rename", "move"} and new_depth != old_depth)
            or (action == "promote" and new_depth >= old_depth)
            or (action == "demote" and new_depth <= old_depth)
            or (action == "remove" and new_depth != old_depth - 1)
            or (
                target.split("/", 1)[0] not in allowed_roots[view]
                and not (old_depth == 1 and action == "rename")
            )
        )
        if invalid_shape:
            quarantined.append({
                "view": view, "old_path": old_path, "new_path": target,
                "status": "rejected",
                "reason": "invalid action depth or unknown L1 target",
                "primary": left, "challenger": right,
            })
            continue
        if action != "remove":
            try:
                assert_formula_safe_alias(old_path, target)
            except ValueError as exc:
                quarantined.append({
                    "view": view, "old_path": old_path, "new_path": target,
                    "status": "rejected", "reason": str(exc),
                    "primary": left, "challenger": right,
                })
                continue
        approved.setdefault((view, old_path), {
            "view": view,
            "old_path": old_path,
            "new_path": target,
            "status": "approved",
            "action": action,
            "confidence": "high",
            "reason": left.get("rationale", ""),
            "source": "independent_gemini_agreement",
        })

    mapping = {
        key: record["new_path"] for key, record in approved.items()
    }
    for key, target in CURATED_CANONICAL_MAPPINGS.items():
        assert_formula_safe_alias(key[1], target)
        reverse_key = (key[0], target)
        if mapping.get(reverse_key) == key[1]:
            mapping.pop(reverse_key)
        mapping[key] = target
    single_placement_merges = _enforce_single_placement(nodes, mapping)
    fanout_merges = _enforce_fanout(nodes, mapping)
    while True:
        final_paths: dict[str, set[str]] = defaultdict(set)
        for view, old_path in nodes:
            new_path = _apply_mapping(view, old_path, mapping)
            parts = new_path.split("/")
            if not 1 <= len(parts) <= 3:
                raise ValueError(f"invalid compiled path: {view}:{new_path}")
            for depth in range(1, len(parts) + 1):
                final_paths[view].add("/".join(parts[:depth]))
        generated_merges = _map_generated_duplicate_paths(final_paths, mapping)
        single_placement_merges += generated_merges
        generated_fanout_merges = _enforce_fanout(nodes, mapping)
        fanout_merges += generated_fanout_merges
        if not generated_merges and not generated_fanout_merges:
            break

    aliases = []
    for view, old_path in nodes:
        new_path = _apply_mapping(view, old_path, mapping)
        if new_path != old_path:
            aliases.append({
                "view": view, "old_path": old_path, "new_path": new_path,
                "status": "approved", "action": "resolved",
                "confidence": "compiled",
                "reason": "resolved exact/prefix canonical mapping",
                "source": "compiler",
            })

    splits = []
    for (view, old_path), labels in CURATED_SPLITS.items():
        resolved = _apply_mapping(view, old_path, mapping)
        if (view, old_path) not in nodes and resolved not in final_paths[view]:
            continue
        parent = resolved.rsplit("/", 1)[0]
        final_paths[view].discard(resolved)
        new_paths = [f"{parent}/{label}" for label in labels]
        final_paths[view].update(new_paths)
        aliases = [
            item for item in aliases
            if not (item["view"] == view and item["old_path"] == old_path)
        ]
        aliases.append({
            "view": view,
            "old_path": old_path,
            "new_path": parent,
            "status": "approved",
            "action": "split",
            "confidence": "high",
            "reason": "dual high-confidence split; legacy path resolves to parent",
            "source": "curated_split",
        })
        splits.append({
            "view": view,
            "old_path": old_path,
            "new_paths": new_paths,
            "fallback_path": f"{parent}/other",
            "status": "approved",
        })

    canonical_l1 = {}
    canonical_l2 = {}
    canonical_l3 = {}
    for view, paths in final_paths.items():
        l1 = set()
        l2: dict[str, set[str]] = defaultdict(set)
        l3: dict[str, set[str]] = defaultdict(set)
        for path in paths:
            parts = path.split("/")
            l1.add(parts[0])
            if len(parts) >= 2:
                l2[parts[0]].add(parts[1])
            if len(parts) == 3:
                l3["/".join(parts[:2])].add(parts[2])
        canonical_l1[view] = _sorted(l1)
        canonical_l2[view] = {
            parent: _sorted(values, include_other=True)
            for parent, values in sorted(l2.items())
        }
        canonical_l3[view] = {
            parent: _sorted(values, include_other=True)
            for parent, values in sorted(l3.items())
            if not parent.endswith("/other")
        }

    spec = {
        "taxonomy_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_taxonomy_version": units["taxonomy_version"],
        "policy": {
            "max_depth": 3,
            "formula_safe": True,
            "high_confidence_independent_agreement_required": True,
            "support_does_not_imply_validity": True,
        },
        "canonical_l1": canonical_l1,
        "canonical_l2": canonical_l2,
        "canonical_l3": canonical_l3,
    }
    registry = {
        "taxonomy_version": version,
        "generated_at": spec["generated_at"],
        "source_taxonomy_version": units["taxonomy_version"],
        "models": sorted({
            record.get("model")
            for review in (primary, challenger)
            for record in review.values()
            if record.get("model")
        }),
        "counts": {
            "approved_direct": len(approved),
            "compiled_aliases": len(aliases),
            "single_placement_merges": single_placement_merges,
            "fanout_merges": fanout_merges,
            "quarantined": len(quarantined),
        },
        "mappings": sorted(
            aliases, key=lambda item: (item["view"], item["old_path"])
        ),
        "splits": splits,
        "quarantined": quarantined,
    }
    return spec, registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--version", default="taxonomy-v3-2026-07")
    parser.add_argument("--legacy-aliases", type=Path)
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    args = parser.parse_args()
    spec, registry = compile_taxonomy(
        json.loads(args.units.read_text()),
        json.loads(args.primary.read_text()),
        json.loads(args.challenger.read_text()),
        args.version,
        (
            json.loads(args.legacy_aliases.read_text())
            if args.legacy_aliases else None
        ),
    )
    args.spec_output.parent.mkdir(parents=True, exist_ok=True)
    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    args.spec_output.write_text(json.dumps(spec, indent=2) + "\n")
    args.registry_output.write_text(json.dumps(registry, indent=2) + "\n")
    print(json.dumps(registry["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
