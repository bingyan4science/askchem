import pytest

from askchem.taxonomy_semantics import (
    assert_formula_safe_alias,
    assert_formula_safe_merge,
    chemical_family_key,
    chemically_compatible,
    concept_key,
    concept_signature,
    display_label,
    duplicate_groups,
    formula_safe_merge_groups,
    high_confidence_near_synonym_pairs,
    near_synonym_pairs,
    scoped_concept_key,
    shallow_deep_duplicates,
    soft_concept_key,
)


def test_formula_and_expanded_name_share_concept_key():
    assert concept_key("CO2 reduction") == concept_key(
        "carbon dioxide reduction reaction"
    )
    assert concept_key("CO reduction") == concept_key(
        "carbon monoxide reduction"
    )


def test_co_and_co2_remain_chemically_distinct():
    assert concept_key("CO reduction") != concept_key("CO2 reduction")
    assert not chemically_compatible("CO reduction", "CO2 reduction")
    assert chemical_family_key("CO reduction") == chemical_family_key(
        "CO2 reduction"
    )


def test_co_prefix_is_not_always_carbon_monoxide():
    assert concept_key("co_assembly") == "co_assembly"
    assert chemically_compatible("co_assembly", "co_crystallization")


def test_known_reaction_relationships_are_formula_safe():
    assert chemically_compatible("water_reduction", "hydrogen_evolution")
    assert chemically_compatible("water_splitting", "oxygen_evolution")


def test_duplicate_groups_find_aliases_without_conflating_species():
    groups = duplicate_groups([
        "co2_reduction",
        "carbon_dioxide_reduction",
        "co2_reduction_reaction",
        "co_reduction",
        "carbon_monoxide_reduction",
    ])
    assert groups["carbon_dioxide_reduction"] == [
        "carbon_dioxide_reduction",
        "co2_reduction",
        "co2_reduction_reaction",
    ]
    assert groups["carbon_monoxide_reduction"] == [
        "carbon_monoxide_reduction",
        "co_reduction",
    ]


def test_formula_guard_rejects_species_changing_alias():
    with pytest.raises(ValueError, match="chemically unsafe"):
        assert_formula_safe_alias(
            "electrocatalysis/co_reduction",
            "electrocatalysis/carbon_dioxide_reduction",
        )


def test_formula_guard_accepts_expanded_alias():
    assert_formula_safe_alias(
        "electrocatalysis/co2_reduction",
        "electrocatalysis/carbon_dioxide_reduction",
    )


def test_display_label_uses_scientific_formula_typography():
    assert display_label("carbon_dioxide_reduction") == "CO₂ Reduction"
    assert display_label("carbon_monoxide_reduction") == "CO Reduction"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("nanoparticles", "nanoparticle"),
        ("material_properties", "material_property"),
        ("polymerisation_methods", "polymerization_method"),
        ("analyses", "analysis"),
    ],
)
def test_soft_concept_key_handles_deterministic_morphology(left, right):
    assert soft_concept_key(left) == soft_concept_key(right)


def test_soft_concept_key_preserves_formula_identity():
    assert soft_concept_key("CO reductions") != soft_concept_key("CO2 reduction")


def test_concept_signature_uses_context_for_generic_labels():
    spectroscopy = concept_signature(
        "by_technique", "spectroscopy/other/analysis"
    )
    microscopy = concept_signature(
        "by_technique", "microscopy/other/analysis"
    )
    assert spectroscopy != microscopy
    assert spectroscopy.stable_id == "by_technique:spectroscopy/analysis"
    assert concept_signature(
        "by_reaction_type", "reduction/co2_reduction"
    ) == concept_signature(
        "by_reaction_type", "catalysis/co2_reduction"
    )


def test_shallow_deep_detector_finds_repeated_specific_concept():
    pairs = shallow_deep_duplicates(
        "by_reaction_type",
        [
            "carbon_dioxide_reduction",
            "reduction/carbon_dioxide_reductions",
            "oxidation/carbon_monoxide_oxidation",
        ],
    )
    assert [(pair.left, pair.right) for pair in pairs] == [
        ("carbon_dioxide_reduction", "reduction/carbon_dioxide_reductions")
    ]


def test_near_synonym_detector_is_formula_safe():
    pairs = near_synonym_pairs(
        "by_reaction_type",
        ["reduction/co2_reduction", "catalysis/co2_reductive_process"],
        threshold=0.7,
    )
    assert len(pairs) == 1
    assert not near_synonym_pairs(
        "by_reaction_type",
        ["reduction/co_reduction", "reduction/co2_reduction"],
        threshold=0.5,
    )


def test_formula_safe_merge_helpers_validate_whole_group():
    groups = formula_safe_merge_groups({
        "co2": ["reduction/co2_reduction", "reduction/carbon_dioxide_reduction"]
    })
    assert groups["co2"] == [
        "reduction/carbon_dioxide_reduction",
        "reduction/co2_reduction",
    ]
    with pytest.raises(ValueError, match="chemically unsafe"):
        assert_formula_safe_merge(
            ["reduction/co_reduction", "reduction/co2_reduction"]
        )


def test_scoped_key_finds_reworded_duplicate():
    assert scoped_concept_key(
        "electrode_materials/metals_and_metal_alloys"
    ) == scoped_concept_key("materials/metals_and_alloys")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("ii_vi_semiconductors", "iii_v_semiconductors"),
        ("organic", "inorganic"),
        ("aerobic", "anaerobic"),
        ("symmetric", "asymmetric"),
        ("pt_then_et", "et_then_pt"),
        ("co_reduction", "co2_reduction"),
    ],
)
def test_high_confidence_detector_preserves_valid_siblings(left, right):
    assert not high_confidence_near_synonym_pairs(
        "view", [left, right], threshold=0.5
    )
