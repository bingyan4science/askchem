from dataclasses import asdict

import scripts.audit_taxonomy_v2 as audit


def install_spec(monkeypatch):
    monkeypatch.setattr(
        audit,
        "CANONICAL_L1",
        {"view": ["reduction", "spectroscopy", "microscopy"]},
    )
    monkeypatch.setattr(
        audit,
        "CANONICAL_L2",
        {
            "view": {
                "reduction": [
                    "co2_reductions",
                    "carbon_dioxide_reduction",
                    "co_reduction",
                ],
                "spectroscopy": ["analysis"],
                "microscopy": ["analysis"],
            }
        },
    )
    monkeypatch.setattr(
        audit,
        "CANONICAL_L3",
        {"view": {("reduction", "co2_reductions"): ["kinetic_analyses"]}},
    )


def test_semantic_report_is_path_aware_and_formula_safe(monkeypatch):
    install_spec(monkeypatch)
    report = audit.semantic_placement_report("view")

    exact_paths = list(report["exact_aliases"].values())
    soft_paths = list(report["soft_aliases"].values())
    assert [
        "reduction/carbon_dioxide_reduction",
        "reduction/co2_reductions",
    ] in soft_paths
    assert not any(
        {"spectroscopy/analysis", "microscopy/analysis"} <= set(paths)
        for paths in exact_paths
    )
    assert not any(
        {"reduction/co_reduction", "reduction/co2_reductions"} <= set(paths)
        for paths in exact_paths
    )


def test_fanout_and_hard_gate_reports_are_machine_readable(monkeypatch):
    install_spec(monkeypatch)
    violations = audit.find_fanout_violations("view", limit=2)
    assert [record.parent_path for record in violations] == ["", "reduction"]

    report = {
        "concept_placement": {
            "view": {
                "exact_aliases": {"co2": ["a", "b"]},
                "soft_aliases": {},
                "shallow_deep": [],
            }
        },
        "fanout_violations": {
            "view": [asdict(record) for record in violations]
        },
    }
    gate = audit.evaluate_hard_gates(report)
    assert not gate.passed
    assert gate.single_placement_violations == 1
    assert gate.fanout_violations == 2
    assert gate.failed_gates == ("single_placement", "fanout")


def test_scoped_duplicates_and_adjudications_are_hard_gates(monkeypatch):
    monkeypatch.setattr(audit, "CANONICAL_L1", {"view": ["electrode_materials"]})
    monkeypatch.setattr(
        audit,
        "CANONICAL_L2",
        {"view": {"electrode_materials": [
            "metals_and_metal_alloys", "metals_and_alloys",
        ]}},
    )
    monkeypatch.setattr(audit, "CANONICAL_L3", {"view": {}})

    placement = audit.semantic_placement_report("view")
    assert list(placement["scoped_label_duplicates"].values()) == [[
        "electrode_materials/metals_and_alloys",
        "electrode_materials/metals_and_metal_alloys",
    ]]

    report = {"concept_placement": {"view": placement}, "axis_leakage": {}}
    adjudication = audit.adjudication_report(report)
    report["adjudication"] = adjudication
    gate = audit.evaluate_hard_gates(report)
    assert gate.scoped_label_duplicates == 1
    assert gate.unresolved_adjudications == adjudication["candidate_count"]
    assert "scoped_label_duplicate" in gate.failed_gates
    assert "unresolved_adjudication" in gate.failed_gates

    decisions = {
        item["candidate_id"]: "keep"
        for item in adjudication["candidates"]
    }
    assert audit.adjudication_report(report, decisions)["complete"]


def test_axis_contract_reports_only_explicit_leakage():
    records = audit.find_axis_leakage(
        "by_technique",
        ["spectroscopy/raman", "reaction_types/oxidation"],
    )
    assert [record["path"] for record in records] == [
        "reaction_types/oxidation"
    ]

