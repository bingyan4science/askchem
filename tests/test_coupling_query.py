"""Tests for organic cross-coupling query signals and technique-view filtering."""

import askchem.db as db


def test_query_signals_organic_cross_coupling():
    assert db.query_signals_organic_cross_coupling("Suzuki coupling")
    assert db.query_signals_organic_cross_coupling(" Buchwald-Hartwig ")
    assert not db.query_signals_organic_cross_coupling("NMR spin coupling")
    assert not db.query_signals_organic_cross_coupling("")


def test_technique_irrelevance_filter_drops_non_organic_coupling_buckets():
    """Spin / phonon / exciton coupling claims must be dropped regardless of
    which by_technique bucket they happen to live in, when the user is
    searching for an organic cross-coupling term."""
    spin = {
        "view_paths": {"by_technique": ["spectroscopy", "optical_spectroscopy"]},
        "verbatim_quote": "Strong coupling between excitons and Bloch surface waves.",
        "source_paper_title": "Strong coupling between excitons in organic semiconductors and Bloch surface waves",
    }
    assert db._technique_claim_is_irrelevant_for_coupling_query(spin)

    phonon = {
        "view_paths": {"by_technique": ["computational_modeling", "dft"]},
        "verbatim_quote": "The spin-spin coupling tensor was computed.",
    }
    assert db._technique_claim_is_irrelevant_for_coupling_query(phonon)


def test_technique_irrelevance_filter_keeps_real_organic_claims():
    organic = {
        "view_paths": {"by_technique": ["spectroscopy", "nmr"]},
        "verbatim_quote": "1H NMR characterisation of biaryl product after Suzuki coupling.",
        "source_paper_title": "Pd-catalysed cross-coupling of aryl halides",
    }
    assert not db._technique_claim_is_irrelevant_for_coupling_query(organic)

    organic_via_title = {
        "view_paths": {"by_technique": ["spectroscopy", "ir"]},
        "verbatim_quote": "Carbonyl stretches confirm conversion.",
        "source_paper_title": "Suzuki-Miyaura cross-coupling of aryl boronic acids",
    }
    assert not db._technique_claim_is_irrelevant_for_coupling_query(organic_via_title)


def test_technique_irrelevance_filter_ignores_claims_outside_view():
    no_path = {"view_paths": {}, "verbatim_quote": "spin coupling"}
    assert not db._technique_claim_is_irrelevant_for_coupling_query(no_path)
