import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from review_taxonomy_semantics import build_candidate_units, build_units


def test_review_units_use_recursive_claim_support(tmp_path):
    database = tmp_path / "claims.db"
    spec_path = tmp_path / "taxonomy.json"
    output = tmp_path / "units.json"
    spec_path.write_text(json.dumps({
        "taxonomy_version": "test",
        "canonical_l1": {"by_reaction_type": ["electrocatalysis"]},
        "canonical_l2": {
            "by_reaction_type": {
                "electrocatalysis": ["carbon_dioxide_reduction"],
            },
        },
        "canonical_l3": {
            "by_reaction_type": {
                "electrocatalysis/carbon_dioxide_reduction": ["mechanism"],
            },
        },
    }))
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE claims("
        "claim_id TEXT, view_paths TEXT, verbatim_quote TEXT)"
    )
    for index in range(3):
        conn.execute(
            "INSERT INTO claims VALUES (?,?,?)",
            (
                f"c{index}",
                json.dumps({
                    "by_reaction_type": [
                        "electrocatalysis",
                        "carbon_dioxide_reduction",
                        "mechanism",
                    ],
                }),
                f"claim {index}",
            ),
        )
    conn.commit()
    conn.close()

    build_units(database, spec_path, output)

    payload = json.loads(output.read_text())
    by_id = {unit["unit_id"]: unit for unit in payload["units"]}
    l1 = by_id["by_reaction_type:root"]["nodes"][0]
    l2 = by_id["by_reaction_type:electrocatalysis"]["nodes"][0]
    assert l1["claim_count"] == 3
    assert l2["claim_count"] == 3
    assert l2["sample_claims"] == ["claim 0", "claim 1"]


def test_cross_path_review_units_isolate_one_concept(tmp_path):
    database = tmp_path / "claims.db"
    spec_path = tmp_path / "taxonomy.json"
    output = tmp_path / "units.json"
    spec_path.write_text(json.dumps({
        "taxonomy_version": "test",
        "canonical_l1": {"by_application": ["energy", "catalysis"]},
        "canonical_l2": {
            "by_application": {
                "energy": ["carbon_dioxide_reduction"],
                "catalysis": ["co2_reduction"],
            },
        },
        "canonical_l3": {"by_application": {}},
    }))
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE claims("
        "claim_id TEXT, view_paths TEXT, verbatim_quote TEXT)"
    )
    conn.commit()
    conn.close()

    build_units(database, spec_path, output)

    payload = json.loads(output.read_text())
    cross = [
        unit for unit in payload["units"]
        if unit["review_scope"] == "cross_path_exact"
    ]
    assert len(cross) == 1
    assert cross[0]["unit_id"].endswith(":carbon_dioxide_reduction")
    assert {node["path"] for node in cross[0]["nodes"]} == {
        "energy/carbon_dioxide_reduction",
        "catalysis/co2_reduction",
    }
    assert cross[0]["required_disposition"] == "one_canonical_path"


def test_candidate_units_cover_every_unique_semantic_pair(tmp_path):
    database = tmp_path / "claims.db"
    spec_path = tmp_path / "taxonomy.json"
    audit_path = tmp_path / "audit.json"
    output = tmp_path / "candidate_units.json"
    spec_path.write_text(json.dumps({
        "taxonomy_version": "test",
        "canonical_l1": {"by_application": ["energy", "catalysis"]},
        "canonical_l2": {"by_application": {}},
        "canonical_l3": {"by_application": {}},
    }))
    audit_path.write_text(json.dumps({
        "concept_placement": {
            "by_application": {
                "near_synonyms": [
                    {
                        "kind": "near_synonym",
                        "left": "catalysis/electrocatalysis",
                        "right": "energy/electrocatalysis_and_reactions",
                        "score": 1.0,
                    },
                    {
                        "kind": "near_synonym",
                        "left": "energy/electrocatalysis_and_reactions",
                        "right": "catalysis/electrocatalysis",
                        "score": 1.0,
                    },
                ],
            },
        },
    }))
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE claims("
        "claim_id TEXT, view_paths TEXT, verbatim_quote TEXT)"
    )
    conn.execute(
        "CREATE TABLE tree_nodes("
        "view_id TEXT, path TEXT, claim_count INTEGER, claim_ids TEXT)"
    )
    conn.execute(
        "INSERT INTO claims VALUES (?,?,?)",
        ("c1", "{}", "Electrocatalysis supports energy conversion."),
    )
    conn.execute(
        "INSERT INTO tree_nodes VALUES (?,?,?,?)",
        (
            "by_application",
            "catalysis/electrocatalysis",
            10,
            json.dumps(["c1"]),
        ),
    )
    conn.execute(
        "INSERT INTO tree_nodes VALUES (?,?,?,?)",
        (
            "by_application",
            "energy/electrocatalysis_and_reactions",
            5,
            "[]",
        ),
    )
    conn.commit()
    conn.close()

    build_candidate_units(database, spec_path, audit_path, output)

    payload = json.loads(output.read_text())
    assert payload["unit_count"] == 1
    unit = payload["units"][0]
    assert unit["review_scope"] == "cross_path_candidate"
    assert unit["candidate_score"] == 1.0
    assert {node["path"] for node in unit["nodes"]} == {
        "catalysis/electrocatalysis",
        "energy/electrocatalysis_and_reactions",
    }
    assert unit["nodes"][0]["sample_claims"] == [
        "Electrocatalysis supports energy conversion.",
    ]
