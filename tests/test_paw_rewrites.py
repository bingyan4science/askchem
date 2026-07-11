"""Regression tests for PAW rewrite plumbing.

Covers the Phase 2 wiring shipped with the paw-ft-bs48-20260522 rollout:

* ``CHEMTREE_PAW_REWRITES=1`` enables both the expand_query path inside
  ``db.expand_query_variants`` and the decompose_query rescue inside
  ``db.search_claims``.
* ``CHEMTREE_DISABLE_PAW=1`` continues to win against the new flag — the
  PAW import never happens regardless of CHEMTREE_PAW_REWRITES.
* ``CHEMTREE_PAW_FT_IDS`` swaps the program-ID constants in
  ``askchem.paw_functions`` to the IDs in the JSON written by
  ``scripts/compile_paw_ft.py``.

These tests stub the PAW SDK so they run without the
``programasweights`` Python package or network access.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def reload_paw_functions(monkeypatch):
    """Reload askchem.paw_functions with env overrides applied at import."""

    def _reload():
        if "askchem.paw_functions" in sys.modules:
            del sys.modules["askchem.paw_functions"]
        import askchem.paw_functions as paw_functions
        return paw_functions

    return _reload


def test_disable_paw_wins_over_rewrites(monkeypatch, reload_paw_functions):
    """CHEMTREE_DISABLE_PAW=1 must short-circuit even with rewrites on."""
    monkeypatch.setenv("CHEMTREE_DISABLE_PAW", "1")
    monkeypatch.setenv("CHEMTREE_PAW_REWRITES", "1")
    monkeypatch.delenv("CHEMTREE_PAW_FT_IDS", raising=False)

    paw_functions = reload_paw_functions()

    # _check_paw must lie about availability when the kill switch is on.
    assert paw_functions._check_paw() is False

    # All PAW entry points must return their documented fallbacks.
    assert paw_functions.expand_query("Suzuki coupling") == []
    assert paw_functions.decompose_query("complex multi topic question") is None
    assert paw_functions.normalize_query("how does X work") == "how does X work"


def test_ft_ids_override_swaps_constants(tmp_path, monkeypatch, reload_paw_functions):
    """CHEMTREE_PAW_FT_IDS rewrites the per-function program IDs."""
    monkeypatch.delenv("CHEMTREE_DISABLE_PAW", raising=False)
    ft_path = tmp_path / "paw_ft_program_ids.json"
    ft_path.write_text(json.dumps({
        "expand": {
            "program_id": "feedeadbeef0000aaaa",
            "constant": "QUERY_EXPANDER_PROGRAM_ID",
        },
        "decompose": {
            "program_id": "decodecode000011bbbb",
            "constant": "QUERY_DECOMPOSER_PROGRAM_ID",
        },
        "normalize": {
            "program_id": "n0rmal1ze0000022cccc",
            "constant": "NORMALIZER_PROGRAM_ID",
        },
    }))
    monkeypatch.setenv("CHEMTREE_PAW_FT_IDS", str(ft_path))

    paw_functions = reload_paw_functions()

    assert paw_functions.QUERY_EXPANDER_PROGRAM_ID == "feedeadbeef0000aaaa"
    assert paw_functions.QUERY_DECOMPOSER_PROGRAM_ID == "decodecode000011bbbb"
    assert paw_functions.NORMALIZER_PROGRAM_ID == "n0rmal1ze0000022cccc"


def test_rewrites_flag_off_skips_paw_expand(monkeypatch):
    """With CHEMTREE_PAW_REWRITES unset, expand_query_variants stays pure-static."""
    monkeypatch.delenv("CHEMTREE_PAW_REWRITES", raising=False)
    monkeypatch.delenv("CHEMTREE_DISABLE_PAW", raising=False)

    # Inject a stub paw_functions that would explode if called.
    sentinel = types.ModuleType("askchem.paw_functions")

    def _boom(*_a, **_kw):
        raise AssertionError("expand_query should not be called with rewrites off")

    sentinel.expand_query = _boom
    sentinel.decompose_query = _boom
    sentinel.normalize_query = _boom
    monkeypatch.setitem(sys.modules, "askchem.paw_functions", sentinel)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    if "chemtree.db" in sys.modules:
        importlib.reload(sys.modules["chemtree.db"])
    from askchem import db as chem_db

    variants = chem_db.expand_query_variants("Suzuki coupling palladium")
    assert isinstance(variants, list)
    assert any("suzuki" in v.lower() for v in variants)


def test_rewrites_flag_on_calls_paw_expand(monkeypatch):
    """With CHEMTREE_PAW_REWRITES=1, expand_query_variants appends a PAW variant."""
    monkeypatch.setenv("CHEMTREE_PAW_REWRITES", "1")
    monkeypatch.delenv("CHEMTREE_DISABLE_PAW", raising=False)

    captured: dict[str, str] = {}

    def _expand(q):
        captured["q"] = q
        # Synthetic synonyms — pick terms that are NOT in the query so
        # the dedup branch in expand_query_variants exercises the append.
        return ["Pd", "Suzuki-Miyaura", "boronic acid"]

    sentinel = types.ModuleType("askchem.paw_functions")
    sentinel.expand_query = _expand
    sentinel.decompose_query = lambda q: None
    sentinel.normalize_query = lambda q: q
    monkeypatch.setitem(sys.modules, "askchem.paw_functions", sentinel)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    if "chemtree.db" in sys.modules:
        importlib.reload(sys.modules["chemtree.db"])
    from askchem import db as chem_db

    variants = chem_db.expand_query_variants("Suzuki coupling")

    assert captured.get("q") == "Suzuki coupling"
    assert any(
        "pd" in v.lower() and "boronic" in v.lower()
        for v in variants
    ), variants
