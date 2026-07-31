import json
import sqlite3
import sys
from pathlib import Path

from askchem.taxonomy_aliases import resolve_tree_path, taxonomy_version

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from migrate_taxonomy_v2 import (
    canonicalize_path,
    load_facet_mapping,
    load_registry,
    mapped_facets,
    mapped_path,
    migrate_claims,
    migrate_view_metadata,
)


def test_resolves_approved_taxonomy_alias():
    canonical, aliased = resolve_tree_path(
        "by_reaction_type",
        "coupling/cross_coupling/suzuki_coupling",
    )

    assert canonical == "coupling/cross_coupling/suzuki_miyaura"
    assert aliased is True


def test_unknown_taxonomy_path_is_unchanged():
    canonical, aliased = resolve_tree_path(
        "by_reaction_type", "not/a/known/path",
    )

    assert canonical == "not/a/known/path"
    assert aliased is False


def test_legacy_l4_path_resolves_through_canonical_prefix():
    canonical, aliased = resolve_tree_path(
        "by_reaction_type",
        "coupling/cross_coupling/suzuki_coupling/ligand_effects",
    )
    assert canonical == "coupling/cross_coupling/suzuki_miyaura"
    assert aliased is True


def test_alias_artifact_has_version():
    assert taxonomy_version() == "taxonomy-v4-2026-07"


def test_taxonomy_v2_canonicalization_is_idempotent():
    original = [
        "coupling", "cross_coupling", "suzuki_miyaura", "hidden_l4",
    ]
    canonical = canonicalize_path("by_reaction_type", original)

    assert canonical == ["coupling", "cross_coupling", "suzuki_miyaura"]
    assert canonicalize_path("by_reaction_type", canonical) == canonical


def test_migration_registry_rejects_formula_changing_alias(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        """{
          "taxonomy_version": "unsafe",
          "views": ["by_reaction_type"],
          "mappings": [{
            "view": "by_reaction_type",
            "old_path": "electrocatalysis/co_reduction",
            "new_path": "electrocatalysis/carbon_dioxide_reduction",
            "status": "approved"
          }]
        }"""
    )

    import pytest
    with pytest.raises(ValueError, match="chemically unsafe"):
        load_registry(registry)


def test_prefix_mapping_collapses_deep_nested_duplicate():
    mapping = {
        (
            "by_reaction_type",
            ("electrocatalysis", "reduction", "co2_reduction"),
        ): ["electrocatalysis", "carbon_dioxide_reduction"],
    }
    assert mapped_path(
        "by_reaction_type",
        [
            "electrocatalysis",
            "reduction",
            "co2_reduction",
            "catalyst_design",
        ],
        mapping,
    ) == [
        "electrocatalysis",
        "carbon_dioxide_reduction",
        "catalyst_design",
    ]


def test_prefix_mapping_preserves_descendant_for_same_depth_rename():
    mapping = {
        (
            "by_reaction_type",
            ("electrocatalysis", "co2_reduction"),
        ): ["electrocatalysis", "carbon_dioxide_reduction"],
    }
    assert mapped_path(
        "by_reaction_type",
        ["electrocatalysis", "co2_reduction", "mechanism"],
        mapping,
    ) == [
        "electrocatalysis",
        "carbon_dioxide_reduction",
        "mechanism",
    ]


def test_unified_mapping_maps_composition_to_substance():
    artifact = {
        "new_views": ["by_substance_class"],
        "mappings": {
            "nanomaterials/metal_nanoparticles": {
                "by_substance_class": "nanomaterials/metal_nanoparticles",
            },
        },
    }
    assert mapped_facets(
        ["nanomaterials", "metal_nanoparticles"], artifact,
    ) == {
        "by_substance_class": ["nanomaterials", "metal_nanoparticles"],
    }


def test_facet_mapping_preserves_descendant_with_depth_cap():
    artifact = {
        "new_views": ["by_substance_class"],
        "mappings": {
            "nanomaterials/metal_nanoparticles": {
                "by_substance_class": "nanomaterials/metal_nanoparticles",
            },
        },
    }
    result = mapped_facets(
        ["nanomaterials", "metal_nanoparticles", "gold"], artifact,
    )
    assert result["by_substance_class"] == [
        "nanomaterials", "metal_nanoparticles", "gold",
    ]


def test_loads_singular_unified_mapping_artifact(tmp_path):
    artifact_path = tmp_path / "substance_mapping.json"
    artifact_path.write_text(json.dumps({
        "taxonomy_version": "test",
        "source_view": "by_composition",
        "target_view": "by_substance_class",
        "policy": {"material_form_is_metadata": True},
        "mappings": {
            "nanomaterials": {
                "path": "nanomaterials",
                "material_forms": ["particle_based/nanoparticles"],
            },
        },
    }))

    artifact = load_facet_mapping(artifact_path)

    assert artifact["old_view"] == "by_composition"
    assert artifact["new_views"] == ["by_substance_class"]
    assert artifact["removed_views"] == [
        "by_composition", "by_material_form",
    ]
    assert artifact["material_form_metadata_key"] == "material_form"
    assert mapped_facets(["nanomaterials"], artifact) == {
        "by_substance_class": ["nanomaterials"],
    }


def test_legacy_facet_mapping_cannot_expose_deprecated_views(tmp_path):
    import pytest

    artifact_path = tmp_path / "legacy_mapping.json"
    artifact_path.write_text(json.dumps({
        "taxonomy_version": "test",
        "old_view": "by_substance_class",
        "new_views": ["by_composition", "by_material_form"],
        "mappings": {},
    }))

    with pytest.raises(ValueError, match="deprecated views"):
        load_facet_mapping(artifact_path)


def test_migration_unifies_paths_and_preserves_material_form_metadata():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE claims "
        "(claim_id TEXT PRIMARY KEY, view_paths TEXT, data TEXT)"
    )
    view_paths = {
        "by_composition": ["organic_compounds", "other"],
        "by_material_form": ["particle_based", "nanoparticles"],
    }
    conn.execute(
        "INSERT INTO claims VALUES (?, ?, ?)",
        ("claim-1", json.dumps(view_paths), json.dumps({"view_paths": view_paths})),
    )
    artifact = {
        "old_view": "by_composition",
        "new_views": ["by_substance_class"],
        "removed_views": ["by_composition", "by_material_form"],
        "material_form_metadata_key": "material_form",
        "mappings": {
            "organic_compounds/other": "organic_compounds/other",
        },
    }

    migrate_claims(
        conn, "test", {"by_substance_class"}, {}, {}, 100, artifact,
    )
    raw_paths, raw_data = conn.execute(
        "SELECT view_paths, data FROM claims WHERE claim_id = 'claim-1'"
    ).fetchone()
    migrated_paths = json.loads(raw_paths)
    migrated_data = json.loads(raw_data)

    assert migrated_paths == {
        "by_substance_class": ["organic_compounds", "other"],
    }
    assert migrated_data["view_paths"] == migrated_paths
    assert migrated_data["material_form"] == [
        "particle_based", "nanoparticles",
    ]


def test_migration_publishes_only_substance_view_metadata():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE views "
        "(view_id TEXT PRIMARY KEY, name TEXT, description TEXT, data TEXT)"
    )
    conn.execute(
        "CREATE TABLE tree_nodes "
        "(view_id TEXT, level INTEGER, claim_count INTEGER)"
    )
    conn.executemany(
        "INSERT INTO views VALUES (?, ?, '', '{}')",
        [
            ("by_composition", "Composition"),
            ("by_material_form", "Material Form"),
        ],
    )

    migrate_view_metadata(conn, None)

    rows = conn.execute(
        "SELECT view_id, name FROM views ORDER BY view_id"
    ).fetchall()
    assert rows == [("by_substance_class", "Substance")]
