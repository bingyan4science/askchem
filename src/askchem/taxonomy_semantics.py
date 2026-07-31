"""Chemistry-aware normalization and validation for taxonomy concepts.

This module intentionally uses a small, explicit vocabulary.  Formula tokens
must never be normalized with fuzzy string similarity alone: CO and CO2, for
example, name different reactants even though their labels are nearly equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections.abc import Iterable


FORMULA_NAMES = {
    "co": "carbon_monoxide",
    "co2": "carbon_dioxide",
    "h2": "hydrogen",
    "h2o": "water",
    "h2o2": "hydrogen_peroxide",
    "n2": "nitrogen",
    "n2o": "nitrous_oxide",
    "nh3": "ammonia",
    "no": "nitric_oxide",
    "no2": "nitrogen_dioxide",
    "n2o3": "dinitrogen_trioxide",
    "n2o4": "dinitrogen_tetroxide",
    "n2o5": "dinitrogen_pentoxide",
    "o2": "oxygen",
    "o3": "ozone",
    "so2": "sulfur_dioxide",
    "so3": "sulfur_trioxide",
    "ch4": "methane",
}

_NAME_TO_FORMULA = {name: formula for formula, name in FORMULA_NAMES.items()}
_DISPLAY_FORMULAS = {
    "carbon_dioxide": "CO₂",
    "carbon_monoxide": "CO",
    "hydrogen_peroxide": "H₂O₂",
    "nitrous_oxide": "N₂O",
    "nitrogen_dioxide": "NO₂",
    "nitric_oxide": "NO",
    "sulfur_dioxide": "SO₂",
    "sulfur_trioxide": "SO₃",
}
_OPTIONAL_SUFFIXES = {
    "reaction",
    "reactions",
    "process",
    "processes",
    "method",
    "methods",
    "technique",
    "techniques",
}
_CO_REACTION_CONTEXT = {
    "adsorption",
    "conversion",
    "electrooxidation",
    "hydrogenation",
    "methanation",
    "oxidation",
    "poisoning",
    "reduction",
}
_REACTION_EQUIVALENT_CONCEPTS = {
    frozenset({"water_reduction", "hydrogen_evolution"}),
    frozenset({"water_splitting", "hydrogen_evolution"}),
    frozenset({"water_splitting", "oxygen_evolution"}),
}
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_IRREGULAR_SINGULARS = {
    "analyses": "analysis",
    "bacteria": "bacterium",
    "classes": "class",
    "criteria": "criterion",
    "data": "datum",
    "indices": "index",
    "matrices": "matrix",
    "media": "medium",
    "properties": "property",
    "species": "species",
    "syntheses": "synthesis",
}
_MORPHOLOGY_EQUIVALENTS = {
    "characterisation": "characterization",
    "functionalisation": "functionalization",
    "modelling": "modeling",
    "optimisation": "optimization",
    "polymerisation": "polymerization",
    "stabilisation": "stabilization",
}
_NON_SINGULAR_WORDS = {
    "analysis",
    "catalysis",
    "class",
    "glass",
    "kinetics",
    "mass",
    "process",
    "species",
    "synthesis",
}
_CONTEXTUAL_LEAVES = {
    "analysis",
    "applications",
    "characterization",
    "mechanism",
    "modeling",
    "other",
    "properties",
    "synthesis",
}
_RELATION_WORDS = {"and", "of", "the"}
_CONTRAST_PAIRS = {
    frozenset({"aerobic", "anaerobic"}),
    frozenset({"organic", "inorganic"}),
    frozenset({"symmetric", "asymmetric"}),
}
_ORDERED_MECHANISMS = {
    ("pt", "then", "et"),
    ("et", "then", "pt"),
}


@dataclass(frozen=True, order=True)
class ConceptSignature:
    """A deterministic concept identifier retaining disambiguating path context."""

    view: str
    concept: str
    context: tuple[str, ...] = ()
    chemical_identities: tuple[str, ...] = ()

    @property
    def stable_id(self) -> str:
        context = "/".join((*self.context, self.concept))
        if self.chemical_identities:
            context += "@" + ",".join(self.chemical_identities)
        return f"{self.view}:{context}" if self.view else context


@dataclass(frozen=True, order=True)
class ConceptPair:
    """A deterministic pair emitted by semantic duplicate detectors."""

    left: str
    right: str
    kind: str
    score: float = 1.0


def normalize_slug(value: str) -> str:
    """Return a stable lowercase underscore slug."""
    return _SLUG_RE.sub("_", str(value).strip().lower()).strip("_")


def concept_key(value: str) -> str:
    """Normalize lexical aliases while preserving chemical identity."""
    tokens = [token for token in normalize_slug(value).split("_") if token]
    expanded: list[str] = []
    for token in tokens:
        formula_name = FORMULA_NAMES.get(token)
        if token == "co" and not (_CO_REACTION_CONTEXT & set(tokens)):
            formula_name = None
        expanded.extend((formula_name or token).split("_"))
    while expanded and expanded[-1] in _OPTIONAL_SUFFIXES:
        expanded.pop()
    return "_".join(expanded)


def _singular_token(token: str) -> str:
    if token in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[token]
    if token in _NON_SINGULAR_WORDS or token in FORMULA_NAMES:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(
        ("ss", "is", "us")
    ):
        return token[:-1]
    return token


def soft_concept_key(value: str) -> str:
    """Return a conservative plural- and spelling-aware concept key.

    Formula expansion is performed first, so morphology never compares raw
    formula strings.  In particular, CO and CO2 retain distinct identities.
    """
    key = concept_key(value)
    tokens = [
        _MORPHOLOGY_EQUIVALENTS.get(_singular_token(token), _singular_token(token))
        for token in key.split("_")
        if token
    ]
    return "_".join(tokens)


def scoped_concept_key(path: str | Iterable[str]) -> str:
    """Normalize a label after removing redundant ancestor scope words.

    This catches scoped spellings such as ``metals_and_metal_alloys`` under
    ``electrode_materials`` without making the unscoped concept identifier
    depend on its path.
    """
    parts = (
        [part for part in path.strip("/").split("/") if part]
        if isinstance(path, str)
        else [str(part) for part in path if str(part).strip()]
    )
    if not parts:
        return ""
    leaf = soft_concept_key(parts[-1]).split("_")
    ancestor_tokens = {
        token
        for part in parts[:-1]
        for token in soft_concept_key(part).split("_")
        if token not in _RELATION_WORDS
    }
    result = []
    for token in leaf:
        if token in _RELATION_WORDS:
            continue
        # Ancestor scope is descriptive, not part of the leaf identity.
        if token in ancestor_tokens and len(leaf) > 1:
            continue
        if not result or result[-1] != token:
            result.append(token)
    # "metals and metal alloys" should have one semantic "metal".
    unique = []
    for token in result:
        if token not in unique:
            unique.append(token)
    return "_".join(unique)


def _contrast_guard(left: str, right: str) -> bool:
    """Return false for short labels whose difference is scientifically real."""
    left_tokens = tuple(soft_concept_key(left).split("_"))
    right_tokens = tuple(soft_concept_key(right).split("_"))
    left_set, right_set = set(left_tokens), set(right_tokens)
    for pair in _CONTRAST_PAIRS:
        if pair & left_set and pair & right_set and (pair & left_set) != (
            pair & right_set
        ):
            return False
    if left_tokens in _ORDERED_MECHANISMS and right_tokens in _ORDERED_MECHANISMS:
        return left_tokens == right_tokens
    # Roman numeral families (II-VI and III-V) are distinct material classes.
    romans = {"i", "ii", "iii", "iv", "v", "vi"}
    left_roman = tuple(token for token in left_tokens if token in romans)
    right_roman = tuple(token for token in right_tokens if token in romans)
    if left_roman and right_roman and left_roman != right_roman:
        return False
    return True


def concept_signature(
    view: str, path: str | Iterable[str],
) -> ConceptSignature:
    """Build a path-aware signature for a taxonomy node.

    Specific leaves keep a context-free concept within their view so misplaced
    copies can still be detected. Generic leaves retain their nearest parent,
    preventing unrelated ``other`` or ``analysis`` nodes from collapsing.
    """
    parts = (
        [part for part in path.strip("/").split("/") if part]
        if isinstance(path, str)
        else [str(part) for part in path if str(part).strip()]
    )
    keys = [soft_concept_key(part) for part in parts]
    concept = keys[-1] if keys else ""
    contextual = (
        concept in {soft_concept_key(item) for item in _CONTEXTUAL_LEAVES}
        or concept in {"", "other"}
    )
    context = ()
    if contextual and len(keys) > 1:
        generic = {soft_concept_key(item) for item in _CONTEXTUAL_LEAVES}
        parent = next(
            (key for key in reversed(keys[:-1]) if key not in generic),
            keys[-2],
        )
        context = (parent,)
    return ConceptSignature(
        view=normalize_slug(view),
        concept=concept,
        context=context,
        chemical_identities=tuple(
            sorted(chemical_identities("_".join(parts) if parts else ""))
        ),
    )


def display_label(value: str) -> str:
    """Render canonical slugs with chemically correct formula typography."""
    slug = normalize_slug(value)
    for name, formula in sorted(
        _DISPLAY_FORMULAS.items(), key=lambda item: len(item[0]), reverse=True,
    ):
        if slug == name:
            return formula
        if slug.startswith(name + "_"):
            suffix = slug[len(name) + 1:].replace("_", " ").title()
            return f"{formula} {suffix}"
    return slug.replace("_", " ").title()


def chemical_identities(value: str) -> frozenset[str]:
    """Extract explicit formula/name identities from a taxonomy label."""
    slug = normalize_slug(value)
    identities = set()
    tokens = set(slug.split("_"))
    for formula, name in FORMULA_NAMES.items():
        formula_present = formula in tokens
        if formula == "co" and not (_CO_REACTION_CONTEXT & tokens):
            formula_present = False
        if formula_present or name in slug:
            identities.add(formula)
    return frozenset(identities)


def chemical_family_key(value: str) -> str:
    """Normalize a label while replacing a named species with a placeholder."""
    key = concept_key(value)
    for name in sorted(_NAME_TO_FORMULA, key=len, reverse=True):
        key = re.sub(
            rf"(^|_){re.escape(name)}(?=_|$)",
            r"\1chemical_species",
            key,
        )
    return key


def chemically_compatible(left: str, right: str) -> bool:
    """Reject aliases that explicitly refer to different chemical species."""
    left_ids = chemical_identities(left)
    right_ids = chemical_identities(right)
    if frozenset({concept_key(left), concept_key(right)}) in (
        _REACTION_EQUIVALENT_CONCEPTS
    ):
        return True
    return not left_ids or not right_ids or bool(left_ids & right_ids)


def assert_formula_safe_alias(old_path: str, new_path: str) -> None:
    """Raise when an alias changes an explicit chemical identity."""
    old_leaf = old_path.strip("/").rsplit("/", 1)[-1]
    new_leaf = new_path.strip("/").rsplit("/", 1)[-1]
    if not chemically_compatible(old_leaf, new_leaf):
        raise ValueError(
            f"chemically unsafe taxonomy alias: {old_path!r} -> {new_path!r}"
        )


def assert_formula_safe_merge(paths: Iterable[str]) -> None:
    """Raise unless every explicit formula identity in a merge is compatible."""
    values = list(paths)
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            assert_formula_safe_alias(left, right)


def formula_safe_merge_groups(
    groups: dict[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Return sorted merge groups after validating all formula identities."""
    result = {}
    for key, values in sorted(groups.items()):
        unique = sorted(set(values))
        assert_formula_safe_merge(unique)
        if len(unique) > 1:
            result[key] = unique
    return result


def shallow_deep_duplicates(
    view: str, paths: Iterable[str],
) -> list[ConceptPair]:
    """Find a concept repeated at different depths in the same view."""
    normalized = sorted(set(path.strip("/") for path in paths if path.strip("/")))
    records = [
        (
            path,
            path.split("/"),
            soft_concept_key(path.rsplit("/", 1)[-1]),
            concept_signature(view, path),
        )
        for path in normalized
    ]
    by_leaf: dict[str, list[int]] = {}
    for index, (_, _, key, _) in enumerate(records):
        by_leaf.setdefault(key, []).append(index)

    candidates = set()
    for indexes in by_leaf.values():
        for offset, left_index in enumerate(indexes):
            for right_index in indexes[offset + 1:]:
                left = records[left_index]
                right = records[right_index]
                if (
                    len(left[1]) != len(right[1])
                    and left[3].context == right[3].context
                ):
                    candidates.add((left_index, right_index))
    return [
        ConceptPair(records[left][0], records[right][0], "shallow_deep")
        for left, right in sorted(candidates)
        if chemically_compatible(
            records[left][1][-1], records[right][1][-1],
        )
    ]


def near_synonym_pairs(
    view: str,
    paths: Iterable[str],
    *,
    threshold: float = 0.84,
) -> list[ConceptPair]:
    """Find conservative lexical near-synonyms without crossing formulas."""
    del view  # Included for a uniform detector API and future view contracts.
    normalized = sorted(set(path.strip("/") for path in paths if path.strip("/")))
    items = []
    token_index: dict[str, list[int]] = {}
    prefix_index: dict[str, list[int]] = {}
    for index, path in enumerate(normalized):
        leaf = path.rsplit("/", 1)[-1]
        key = soft_concept_key(leaf)
        tokens = set(key.split("_")) - {"and", "of", "the"}
        items.append((path, leaf, key, tokens))
        for token in tokens:
            token_index.setdefault(token, []).append(index)
        prefix_index.setdefault(key[:5], []).append(index)

    candidate_pairs = set()
    for indexes in [*token_index.values(), *prefix_index.values()]:
        # Very common generic tokens create millions of low-value pairs.
        if len(indexes) > 200:
            continue
        for offset, left_index in enumerate(indexes):
            for right_index in indexes[offset + 1:]:
                candidate_pairs.add((left_index, right_index))

    result = []
    for left_index, right_index in sorted(candidate_pairs):
        left, left_leaf, left_key, left_tokens = items[left_index]
        right, right_leaf, right_key, right_tokens = items[right_index]
        if not left_key or left_key == right_key:
            continue
        if not chemically_compatible(left_leaf, right_leaf):
            continue
        if not _contrast_guard(left_leaf, right_leaf):
            continue
        overlap = len(left_tokens & right_tokens) / max(
            1, len(left_tokens | right_tokens)
        )
        sequence = SequenceMatcher(None, left_key, right_key).ratio()
        score = max(overlap, sequence)
        # Token overlap must be meaningful; SequenceMatcher alone can
        # dangerously group short chemistry labels.
        if score >= threshold and left_tokens & right_tokens:
            result.append(
                ConceptPair(left, right, "near_synonym", round(score, 6))
            )
    return result


def high_confidence_near_synonym_pairs(
    view: str,
    paths: Iterable[str],
    *,
    threshold: float = 0.9,
) -> list[ConceptPair]:
    """Return near synonyms safe enough to use as release-gate candidates."""
    return [
        ConceptPair(pair.left, pair.right, "high_confidence_near_synonym", pair.score)
        for pair in near_synonym_pairs(view, paths, threshold=threshold)
        if _contrast_guard(pair.left.rsplit("/", 1)[-1], pair.right.rsplit("/", 1)[-1])
    ]


def ancestor_redundancy_pairs(
    view: str, paths: Iterable[str],
) -> list[ConceptPair]:
    """Find child labels that restate an ancestor or repeat across depths."""
    del view
    result = set()
    for path in sorted(set(item.strip("/") for item in paths if item.strip("/"))):
        parts = path.split("/")
        leaf_key = soft_concept_key(parts[-1])
        leaf_tokens = set(leaf_key.split("_")) - _RELATION_WORDS
        for depth, ancestor in enumerate(parts[:-1], start=1):
            ancestor_key = soft_concept_key(ancestor)
            ancestor_tokens = set(ancestor_key.split("_")) - _RELATION_WORDS
            if (
                leaf_key == ancestor_key
                or (ancestor_tokens and ancestor_tokens < leaf_tokens)
            ):
                result.add(("/".join(parts[:depth]), path))
    return [
        ConceptPair(left, right, "ancestor_redundancy")
        for left, right in sorted(result)
    ]


def duplicate_groups(values: Iterable[str]) -> dict[str, list[str]]:
    """Group labels that reduce to the same chemistry-aware concept key."""
    grouped: dict[str, list[str]] = {}
    for value in values:
        grouped.setdefault(concept_key(value), []).append(value)
    return {
        key: sorted(set(labels))
        for key, labels in grouped.items()
        if key and len(set(labels)) > 1
    }
