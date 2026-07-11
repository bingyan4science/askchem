"""
Claim validation for AskChem.

Validates LLM-extracted claims before database insertion:
- Enum validation for claim_type, confidence
- Type-specific required field checks
- Numeric field validation
- SMILES validation (optional, requires rdkit)

Usage:
    from askchem.validation import validate_claim, ValidationResult

    result = validate_claim(claim_dict)
    if result.is_valid:
        db.insert_claim(claim_dict)
    else:
        log_quarantine(claim_dict, result.errors)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

VALID_CLAIM_TYPES = frozenset({
    "reaction", "property", "method", "mechanism", "comparison",
    "scope_entry", "computational_result", "structure",
    "hypothesis", "experimental_design",
    "limitation", "future_direction", "surprising_finding",
    "conclusion", "conclusions",
})

VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

VALID_PAPER_LOCATIONS = frozenset({
    "abstract", "introduction", "results", "discussion", "methods",
    "experimental", "conclusion", "supporting_information", "supplementary",
})

TYPE_REQUIRED_FIELDS: dict[str, list[str]] = {
    "reaction": ["verbatim_quote"],
    "scope_entry": ["verbatim_quote"],
    "property": ["verbatim_quote"],
    "structure": ["verbatim_quote"],
    "method": ["verbatim_quote"],
    "mechanism": ["verbatim_quote"],
    "comparison": ["verbatim_quote"],
    "computational_result": ["verbatim_quote"],
    "hypothesis": ["verbatim_quote"],
    "experimental_design": ["verbatim_quote"],
    "limitation": ["verbatim_quote"],
    "future_direction": ["verbatim_quote"],
    "surprising_finding": ["verbatim_quote"],
}

TYPE_RECOMMENDED_FIELDS: dict[str, list[str]] = {
    "reaction": ["reaction_type"],
    "scope_entry": ["reaction_type"],
    "property": ["subject", "property_name"],
    "structure": ["subject"],
    "method": ["technique_name"],
    "mechanism": ["process_described"],
    "comparison": ["comparison_result"],
    "computational_result": ["technique_name"],
}

NUMERIC_FIELDS = {
    "yield_percent", "ee_percent", "dr", "conversion_percent",
    "turnover_number", "citation_count",
}


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: Severity


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, field_name: str, message: str):
        self.errors.append(ValidationIssue(field_name, message, Severity.ERROR))
        self.is_valid = False

    def add_warning(self, field_name: str, message: str):
        self.warnings.append(ValidationIssue(field_name, message, Severity.WARNING))

    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        return ", ".join(parts) if parts else "valid"


def validate_claim(claim: dict, strict: bool = False) -> ValidationResult:
    """
    Validate a claim dictionary.

    Args:
        claim: The claim data dict as returned by LLM extraction.
        strict: If True, recommended fields become required.

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # -- Required metadata --
    claim_id = claim.get("claim_id")
    if not claim_id:
        result.add_error("claim_id", "Missing claim_id")

    claim_type = claim.get("claim_type", "")
    if not claim_type:
        result.add_error("claim_type", "Missing claim_type")
    elif claim_type not in VALID_CLAIM_TYPES:
        result.add_error("claim_type", f"Invalid claim_type: '{claim_type}'")

    source_doi = claim.get("source_doi", "")
    if not source_doi:
        result.add_error("source_doi", "Missing source_doi")

    confidence = claim.get("confidence", "")
    if confidence and confidence not in VALID_CONFIDENCE:
        result.add_warning("confidence", f"Non-standard confidence: '{confidence}'")

    # -- Verbatim quote --
    quote = claim.get("verbatim_quote", "")
    if not quote:
        result.add_error("verbatim_quote", "Missing verbatim_quote (source grounding required)")
    elif len(quote) < 10:
        result.add_warning("verbatim_quote", f"Very short quote ({len(quote)} chars)")

    # -- Type-specific required fields --
    if claim_type in TYPE_REQUIRED_FIELDS:
        for f in TYPE_REQUIRED_FIELDS[claim_type]:
            if not claim.get(f):
                result.add_error(f, f"Required field '{f}' missing for {claim_type}")

    # -- Type-specific recommended fields --
    if claim_type in TYPE_RECOMMENDED_FIELDS:
        for f in TYPE_RECOMMENDED_FIELDS[claim_type]:
            if not claim.get(f):
                if strict:
                    result.add_error(f, f"Recommended field '{f}' missing for {claim_type}")
                else:
                    result.add_warning(f, f"Recommended field '{f}' missing for {claim_type}")

    # -- Numeric field validation --
    _validate_numeric_fields(claim, result)

    # -- SMILES validation (if rdkit available) --
    _validate_smiles(claim, result)

    # -- Reactants/products structure --
    if claim_type in ("reaction", "scope_entry"):
        _validate_reaction_fields(claim, result)

    # -- View paths --
    view_paths = claim.get("view_paths", {})
    if view_paths and isinstance(view_paths, dict):
        for view_id, path in view_paths.items():
            if not isinstance(path, list):
                result.add_warning("view_paths", f"View path for '{view_id}' is not a list")
            elif len(path) == 0:
                result.add_warning("view_paths", f"Empty path for view '{view_id}'")

    return result


def _validate_numeric_fields(claim: dict, result: ValidationResult):
    """Check that fields expected to be numeric are actually numeric."""
    outcomes = claim.get("outcomes", {})
    if isinstance(outcomes, dict):
        for key in ("yield_percent", "ee_percent", "dr", "conversion_percent", "turnover_number"):
            val = outcomes.get(key)
            if val is not None and val != "":
                if not _is_numeric(val):
                    result.add_warning(
                        f"outcomes.{key}",
                        f"Expected numeric value, got: '{val}'"
                    )
                else:
                    num = _to_float(val)
                    if num is not None:
                        if key in ("yield_percent", "ee_percent", "conversion_percent"):
                            if num < 0 or num > 100:
                                result.add_warning(
                                    f"outcomes.{key}",
                                    f"Value {num} outside expected 0-100 range"
                                )


def _validate_reaction_fields(claim: dict, result: ValidationResult):
    """Validate structure of reactants and products lists."""
    for field_name in ("reactants", "products"):
        items = claim.get(field_name, [])
        if not isinstance(items, list):
            result.add_warning(field_name, f"Expected list, got {type(items).__name__}")
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                result.add_warning(field_name, f"Item {i} is not a dict")
            elif not item.get("name"):
                result.add_warning(field_name, f"Item {i} missing 'name'")


def _validate_smiles(claim: dict, result: ValidationResult):
    """Validate SMILES strings if rdkit is available."""
    smiles_fields = []

    subject_smiles = claim.get("subject_smiles", "")
    if subject_smiles:
        smiles_fields.append(("subject_smiles", subject_smiles))

    for field_name in ("reactants", "products"):
        items = claim.get(field_name, [])
        if isinstance(items, list):
            for i, item in enumerate(items):
                if isinstance(item, dict) and item.get("smiles"):
                    smiles_fields.append((f"{field_name}[{i}].smiles", item["smiles"]))

    if not smiles_fields:
        return

    try:
        from rdkit import Chem
        for field_name, smi in smiles_fields:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                result.add_warning(field_name, f"Invalid SMILES: '{smi}'")
    except ImportError:
        pass


def _is_numeric(val) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        try:
            float(val.strip().rstrip("%"))
            return True
        except (ValueError, AttributeError):
            return False
    return False


def _to_float(val) -> Optional[float]:
    try:
        if isinstance(val, str):
            return float(val.strip().rstrip("%"))
        return float(val)
    except (ValueError, TypeError):
        return None


def validate_batch(claims: list[dict], strict: bool = False) -> dict:
    """
    Validate a batch of claims and return summary statistics.

    Returns:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "warnings_only": int,
            "error_counts": {field: count},
            "warning_counts": {field: count},
            "results": [ValidationResult, ...]
        }
    """
    from collections import Counter
    results = []
    error_counts: Counter = Counter()
    warning_counts: Counter = Counter()
    valid = 0
    invalid = 0
    warnings_only = 0

    for claim in claims:
        r = validate_claim(claim, strict=strict)
        results.append(r)
        if r.is_valid:
            if r.warnings:
                warnings_only += 1
            valid += 1
        else:
            invalid += 1
        for e in r.errors:
            error_counts[e.field] += 1
        for w in r.warnings:
            warning_counts[w.field] += 1

    return {
        "total": len(claims),
        "valid": valid,
        "invalid": invalid,
        "warnings_only": warnings_only,
        "error_counts": dict(error_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "results": results,
    }
