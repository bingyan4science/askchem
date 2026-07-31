"""Tests for askchem.taxonomy — canonical categories and path normalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askchem.taxonomy import (
    CANONICAL_L1,
    CLAIM_TYPE_LABELS,
    ALL_CONTENT_VIEWS,
    normalize_path,
    build_claim_type_path,
    build_classification_prompt,
)


class TestCanonicalL1:
    def test_all_content_views_present(self):
        expected = {
            "by_reaction_type", "by_substance_class",
            "by_technique", "by_application", "by_mechanism",
        }
        assert expected == set(CANONICAL_L1.keys())

    def test_each_view_has_categories(self):
        for view_id, cats in CANONICAL_L1.items():
            assert len(cats) >= 5, f"{view_id} has too few categories"

    def test_categories_are_lowercase_underscore(self):
        for view_id, cats in CANONICAL_L1.items():
            for cat in cats:
                assert cat == cat.lower(), f"Category '{cat}' in {view_id} not lowercase"
                assert " " not in cat, f"Category '{cat}' in {view_id} has spaces"

    def test_no_duplicate_categories(self):
        for view_id, cats in CANONICAL_L1.items():
            assert len(cats) == len(set(cats)), f"Duplicates in {view_id}"


class TestClaimTypeLabels:
    def test_all_expected_types_mapped(self):
        expected_types = {
            "reaction", "scope_entry", "property", "structure",
            "method", "experimental_design", "mechanism",
            "comparison", "computational_result",
            "hypothesis", "conclusion", "conclusions",
            "limitation", "future_direction", "surprising_finding",
        }
        assert expected_types == set(CLAIM_TYPE_LABELS.keys())

    def test_scope_entry_maps_to_reaction(self):
        assert CLAIM_TYPE_LABELS["scope_entry"] == "reaction"

    def test_structure_maps_to_property(self):
        assert CLAIM_TYPE_LABELS["structure"] == "property"


class TestAllContentViews:
    def test_matches_canonical_keys(self):
        assert set(ALL_CONTENT_VIEWS) == set(CANONICAL_L1.keys())


class TestNormalizePath:
    def test_valid_path(self):
        result = normalize_path("by_reaction_type", ["catalysis", "heterogeneous_catalysis"])
        assert result is not None
        assert result[0] == "catalysis"
        assert result[1] == "heterogeneous_catalysis"

    def test_invalid_l1_returns_none(self):
        result = normalize_path("by_reaction_type", ["bogus_category", "sub"])
        assert result is None

    def test_not_applicable_filtered(self):
        result = normalize_path("by_reaction_type", ["not_applicable"])
        assert result is None

    def test_none_filtered(self):
        result = normalize_path("by_reaction_type", ["none"])
        assert result is None

    def test_empty_path_returns_none(self):
        assert normalize_path("by_reaction_type", []) is None

    def test_none_path_returns_none(self):
        assert normalize_path("by_reaction_type", None) is None

    def test_cleans_dashes_and_spaces(self):
        result = normalize_path("by_reaction_type", ["catalysis", "heterogeneous-catalysis"])
        assert result is not None
        assert result[1] == "heterogeneous_catalysis"

    def test_invalid_l2_becomes_other(self):
        result = normalize_path("by_reaction_type", ["catalysis", "nonexistent_subcategory"])
        assert result is not None
        assert result[1] == "other"

    def test_unknown_view_allows_any_l1(self):
        result = normalize_path("by_unknown_view", ["anything", "goes"])
        assert result is not None
        assert result[0] == "anything"

    def test_legacy_alias_is_resolved_before_validation(self):
        result = normalize_path(
            "by_reaction_type",
            ["coupling", "cross_coupling", "suzuki_coupling"],
        )
        assert result == ["coupling", "cross_coupling", "suzuki_miyaura"]


class TestBuildClaimTypePath:
    def test_reaction(self):
        assert build_claim_type_path("reaction") == ["reaction"]

    def test_scope_entry_maps_to_reaction(self):
        assert build_claim_type_path("scope_entry") == ["reaction"]

    def test_unknown_type_uses_itself(self):
        assert build_claim_type_path("unknown_type") == ["unknown_type"]


class TestBuildClassificationPrompt:
    def test_returns_string(self):
        result = build_classification_prompt("reaction", "some quote", "Some Paper")
        assert isinstance(result, str)

    def test_contains_claim_info(self):
        result = build_classification_prompt("property", "melting point is 150C", "My Paper")
        assert "property" in result
        assert "melting point" in result
        assert "My Paper" in result

    def test_classification_messages_contain_views(self):
        from askchem.taxonomy import build_classification_messages
        msgs = build_classification_messages("reaction", "quote", "title")
        system_content = msgs[0]["content"]
        for view_id in CANONICAL_L1:
            assert view_id in system_content
