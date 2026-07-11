"""Tests for askchem.validation — claim validation logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askchem.validation import (
    validate_claim,
    validate_batch,
    ValidationResult,
    VALID_CLAIM_TYPES,
    VALID_CONFIDENCE,
)


def _make_claim(**overrides) -> dict:
    """Create a minimal valid claim dict, with optional overrides."""
    base = {
        "claim_id": "abc123",
        "claim_type": "reaction",
        "source_doi": "10.1234/test",
        "source_paper_title": "Test Paper",
        "confidence": "high",
        "verbatim_quote": "The reaction proceeded in 95% yield under mild conditions.",
        "extraction_model": "gpt-5.4",
        "extraction_version": "v2",
    }
    base.update(overrides)
    return base


# ── Valid claims ──


class TestValidClaims:
    def test_minimal_valid_claim(self):
        r = validate_claim(_make_claim())
        assert r.is_valid
        assert r.error_count == 0

    def test_all_claim_types_accepted(self):
        for ct in VALID_CLAIM_TYPES:
            r = validate_claim(_make_claim(claim_type=ct))
            assert r.is_valid, f"claim_type '{ct}' should be valid"

    def test_all_confidence_levels(self):
        for conf in VALID_CONFIDENCE:
            r = validate_claim(_make_claim(confidence=conf))
            assert r.is_valid

    def test_claim_with_view_paths(self):
        r = validate_claim(_make_claim(view_paths={
            "by_reaction_type": ["catalysis", "cross_coupling"],
            "by_technique": ["spectroscopy"],
        }))
        assert r.is_valid


# ── Missing required fields ──


class TestMissingFields:
    def test_missing_claim_id(self):
        r = validate_claim(_make_claim(claim_id=""))
        assert not r.is_valid
        assert any(e.field == "claim_id" for e in r.errors)

    def test_missing_claim_type(self):
        r = validate_claim(_make_claim(claim_type=""))
        assert not r.is_valid
        assert any(e.field == "claim_type" for e in r.errors)

    def test_invalid_claim_type(self):
        r = validate_claim(_make_claim(claim_type="bogus_type"))
        assert not r.is_valid
        assert any("Invalid claim_type" in e.message for e in r.errors)

    def test_missing_source_doi(self):
        r = validate_claim(_make_claim(source_doi=""))
        assert not r.is_valid
        assert any(e.field == "source_doi" for e in r.errors)

    def test_missing_verbatim_quote(self):
        r = validate_claim(_make_claim(verbatim_quote=""))
        assert not r.is_valid
        assert any(e.field == "verbatim_quote" for e in r.errors)

    def test_short_quote_warns(self):
        r = validate_claim(_make_claim(verbatim_quote="short"))
        assert r.is_valid  # warning, not error
        assert r.warning_count > 0


# ── Type-specific recommended fields ──


class TestRecommendedFields:
    def test_reaction_missing_reaction_type_warns(self):
        r = validate_claim(_make_claim(claim_type="reaction"))
        assert r.is_valid
        assert any("reaction_type" in w.field for w in r.warnings)

    def test_property_missing_subject_warns(self):
        r = validate_claim(_make_claim(claim_type="property"))
        assert r.is_valid
        assert any("subject" in w.field for w in r.warnings)

    def test_strict_mode_promotes_warnings_to_errors(self):
        r = validate_claim(_make_claim(claim_type="reaction"), strict=True)
        assert not r.is_valid
        assert any("reaction_type" in e.field for e in r.errors)

    def test_method_with_technique_name_no_warning(self):
        r = validate_claim(_make_claim(claim_type="method", technique_name="HPLC"))
        assert not any("technique_name" in w.field for w in r.warnings)


# ── Numeric validation ──


class TestNumericValidation:
    def test_valid_yield(self):
        r = validate_claim(_make_claim(outcomes={"yield_percent": 95}))
        assert r.is_valid
        assert not any("yield_percent" in w.field for w in r.warnings)

    def test_yield_out_of_range(self):
        r = validate_claim(_make_claim(outcomes={"yield_percent": 150}))
        assert any("outside expected 0-100 range" in w.message for w in r.warnings)

    def test_non_numeric_yield(self):
        r = validate_claim(_make_claim(outcomes={"yield_percent": "not a number"}))
        assert any("Expected numeric" in w.message for w in r.warnings)

    def test_string_numeric_yield(self):
        r = validate_claim(_make_claim(outcomes={"yield_percent": "95%"}))
        assert not any("Expected numeric" in w.message for w in r.warnings)


# ── Reaction field validation ──


class TestReactionFields:
    def test_valid_reactants(self):
        r = validate_claim(_make_claim(
            claim_type="reaction",
            reactants=[{"name": "benzene", "smiles": "c1ccccc1"}],
        ))
        assert not any("reactants" in w.field for w in r.warnings)

    def test_reactant_missing_name(self):
        r = validate_claim(_make_claim(
            claim_type="reaction",
            reactants=[{"smiles": "c1ccccc1"}],
        ))
        assert any("reactants" in w.field and "missing 'name'" in w.message for w in r.warnings)

    def test_reactants_not_list(self):
        r = validate_claim(_make_claim(claim_type="reaction", reactants="benzene"))
        assert any("reactants" in w.field for w in r.warnings)


# ── View paths validation ──


class TestViewPaths:
    def test_non_list_path_warns(self):
        r = validate_claim(_make_claim(view_paths={"by_reaction_type": "catalysis"}))
        assert any("view_paths" in w.field for w in r.warnings)

    def test_empty_path_warns(self):
        r = validate_claim(_make_claim(view_paths={"by_reaction_type": []}))
        assert any("view_paths" in w.field for w in r.warnings)


# ── Non-standard confidence ──


class TestConfidence:
    def test_nonstandard_confidence_warns(self):
        r = validate_claim(_make_claim(confidence="very_high"))
        assert r.is_valid
        assert any("confidence" in w.field for w in r.warnings)


# ── Batch validation ──


class TestBatchValidation:
    def test_batch_counts(self):
        claims = [
            _make_claim(),
            _make_claim(claim_id=""),
            _make_claim(claim_type="bogus"),
            _make_claim(claim_type="property"),
        ]
        result = validate_batch(claims)
        assert result["total"] == 4
        assert result["valid"] == 2
        assert result["invalid"] == 2

    def test_empty_batch(self):
        result = validate_batch([])
        assert result["total"] == 0
        assert result["valid"] == 0

    def test_batch_error_counts(self):
        claims = [
            _make_claim(claim_id=""),
            _make_claim(claim_id=""),
        ]
        result = validate_batch(claims)
        assert "claim_id" in result["error_counts"]
        assert result["error_counts"]["claim_id"] == 2
