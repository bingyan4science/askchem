import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compile_taxonomy_v3 import compile_taxonomy


def _decision(node_id, action, path, confidence="high"):
    return {
        "node_id": node_id,
        "action": action,
        "canonical_label": path.rsplit("/", 1)[-1],
        "canonical_path": path,
        "confidence": confidence,
        "rationale": "test",
    }


def _review(decisions):
    return {
        "unit": {
            "model": "gemini-3.1-pro-preview",
            "result": {"decisions": decisions},
        }
    }


def test_compiler_collapses_formula_aliases_but_keeps_co_distinct():
    labels = [
        "carbon_dioxide_reduction",
        "co2_reduction",
        "carbon_monoxide_reduction",
        "co_reduction",
        "other",
    ]
    nodes = [
        {
            "node_id": f"by_reaction_type:electrocatalysis/{label}",
            "label": label,
            "path": f"electrocatalysis/{label}",
        }
        for label in labels
    ]
    units = {
        "taxonomy_version": "v2",
        "units": [
            {
                "unit_id": "by_reaction_type:root",
                "view": "by_reaction_type",
                "level": 1,
                "parent_path": "",
                "nodes": [{
                    "node_id": "by_reaction_type:electrocatalysis",
                    "label": "electrocatalysis",
                    "path": "electrocatalysis",
                }],
            },
            {
                "unit_id": "unit",
                "view": "by_reaction_type",
                "level": 2,
                "parent_path": "electrocatalysis",
                "nodes": nodes,
            },
        ],
    }
    decisions = [
        _decision("by_reaction_type:electrocatalysis", "keep", "electrocatalysis"),
        *[
            _decision(node["node_id"], "keep", node["path"])
            for node in nodes
        ],
    ]

    spec, registry = compile_taxonomy(
        units, _review(decisions), _review(decisions), "v3",
    )

    children = spec["canonical_l2"]["by_reaction_type"]["electrocatalysis"]
    assert "carbon_dioxide_reduction" in children
    assert "carbon_monoxide_reduction" in children
    assert "co2_reduction" not in children
    assert "co_reduction" not in children
    aliases = {
        item["old_path"]: item["new_path"] for item in registry["mappings"]
    }
    assert aliases["electrocatalysis/co2_reduction"] == (
        "electrocatalysis/carbon_dioxide_reduction"
    )
    assert aliases["electrocatalysis/co_reduction"] == (
        "electrocatalysis/carbon_monoxide_reduction"
    )


def test_compiler_quarantines_formula_changing_agreement():
    node_id = "by_reaction_type:electrocatalysis/co_reduction"
    units = {
        "taxonomy_version": "v2",
        "units": [{
            "unit_id": "unit",
            "view": "by_reaction_type",
            "level": 2,
            "parent_path": "electrocatalysis",
            "nodes": [{
                "node_id": node_id,
                "label": "co_reduction",
                "path": "electrocatalysis/co_reduction",
            }],
        }],
    }
    unsafe = _decision(
        node_id, "merge", "electrocatalysis/carbon_dioxide_reduction",
    )

    _, registry = compile_taxonomy(
        units, _review([unsafe]), _review([unsafe]), "v3",
    )

    assert registry["counts"]["quarantined"] == 1
    assert "chemically unsafe" in registry["quarantined"][0]["reason"]


def test_compiler_uses_one_reaction_hierarchy_for_co2_and_co_reduction():
    paths = [
        "reduction/co2_reduction",
        "catalysis/electrocatalysis/carbon_dioxide_reduction",
        "catalysis/electrocatalysis/carbon_monoxide_reduction",
        "catalysis/photocatalysis/co2_photoreduction",
    ]
    units = {
        "taxonomy_version": "v2",
        "units": [{
            "unit_id": "unit",
            "view": "by_reaction_type",
            "level": 3,
            "parent_path": "",
            "nodes": [
                {
                    "node_id": f"by_reaction_type:{path}",
                    "label": path.rsplit("/", 1)[-1],
                    "path": path,
                }
                for path in paths
            ],
        }],
    }
    decisions = [
        _decision(f"by_reaction_type:{path}", "keep", path)
        for path in paths
    ]

    spec, registry = compile_taxonomy(
        units, _review(decisions), _review(decisions), "v3",
    )

    reduction_l2 = spec["canonical_l2"]["by_reaction_type"]["reduction"]
    assert "carbon_dioxide_reduction" in reduction_l2
    assert "carbon_monoxide_reduction" in reduction_l2
    assert "co2_reduction" not in reduction_l2
    co2_modes = spec["canonical_l3"]["by_reaction_type"][
        "reduction/carbon_dioxide_reduction"
    ]
    assert "electrocatalytic_reduction" in co2_modes
    assert "photocatalytic_reduction" in co2_modes
    aliases = {
        item["old_path"]: item["new_path"] for item in registry["mappings"]
    }
    assert aliases["reduction/co2_reduction"] == (
        "reduction/carbon_dioxide_reduction"
    )
    assert aliases[
        "catalysis/electrocatalysis/carbon_monoxide_reduction"
    ] == "reduction/carbon_monoxide_reduction/electrocatalytic_reduction"


def test_compiler_prunes_cross_path_exact_concepts_to_one_placement():
    paths = [
        "biomolecules/amino_acids",
        "organic_compounds/amino_acids",
    ]
    units = {
        "taxonomy_version": "v3",
        "units": [{
            "unit_id": "cross",
            "view": "by_substance_class",
            "level": "cross_path",
            "parent_path": "",
            "review_scope": "cross_path_exact",
            "nodes": [
                {
                    "node_id": f"by_substance_class:{path}",
                    "label": "amino_acids",
                    "path": path,
                    "claim_count": count,
                }
                for path, count in zip(paths, [100, 10])
            ],
        }],
    }
    decisions = [
        _decision(f"by_substance_class:{path}", "keep", path)
        for path in paths
    ]

    spec, registry = compile_taxonomy(
        units, _review(decisions), _review(decisions), "v4",
    )

    all_paths = {
        f"{l1}/{l2}"
        for l1, children in spec["canonical_l2"][
            "by_substance_class"
        ].items()
        for l2 in children
    }
    assert "biomolecules/amino_acids" in all_paths
    assert "organic_compounds/amino_acids" not in all_paths
    assert registry["counts"]["single_placement_merges"] == 1


def test_compiler_keeps_generic_leaf_when_chemical_context_differs():
    paths = [
        "reduction/carbon_dioxide_reduction/electrocatalytic_reduction",
        "reduction/carbon_monoxide_reduction/electrocatalytic_reduction",
    ]
    units = {
        "taxonomy_version": "v3",
        "units": [{
            "unit_id": "cross",
            "view": "by_reaction_type",
            "level": "cross_path",
            "parent_path": "",
            "review_scope": "cross_path_exact",
            "nodes": [
                {
                    "node_id": f"by_reaction_type:{path}",
                    "label": "electrocatalytic_reduction",
                    "path": path,
                    "claim_count": 10,
                }
                for path in paths
            ],
        }],
    }
    decisions = [
        _decision(f"by_reaction_type:{path}", "keep", path)
        for path in paths
    ]

    spec, registry = compile_taxonomy(
        units, _review(decisions), _review(decisions), "v4",
    )

    assert registry["counts"]["single_placement_merges"] == 0
    assert set(spec["canonical_l3"]["by_reaction_type"]) == {
        "reduction/carbon_dioxide_reduction",
        "reduction/carbon_monoxide_reduction",
    }
