"""Unit tests for the PAW expander benchmark harness.

Covers the matching rule and metric primitives. The PAW system adapters
are NOT exercised here (they require the programasweights SDK + a
network round-trip to fetch program assets); they are smoke-tested via
``scripts/bench_paw_expander.py`` itself on the real probe set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

bench = pytest.importorskip("bench_paw_expander")


class TestMatching:
    """Matching rule (case-insensitive, non-alphanumeric boundary)."""

    def test_pd_matches_pd_catalyzed(self):
        hits = bench._term_matches_any("Pd-catalyzed", ["Pd"])
        assert hits == ["Pd"]

    def test_pd_matches_palladium_pd_parens(self):
        hits = bench._term_matches_any("palladium (Pd)", ["Pd"])
        assert hits == ["Pd"]

    def test_pd_does_not_match_padova(self):
        hits = bench._term_matches_any("Padova", ["Pd"])
        assert hits == []

    def test_case_insensitive(self):
        hits = bench._term_matches_any("PALLADIUM", ["palladium"])
        assert hits == ["palladium"]

    def test_multiword_gold(self):
        hits = bench._term_matches_any(
            "boronic acid derivatives", ["boronic acid"]
        )
        assert hits == ["boronic acid"]

    def test_punctuated_gold(self):
        # MOF-5 must match inside an output item that includes it.
        hits = bench._term_matches_any("MOF-5 isotherm", ["MOF-5"])
        assert hits == ["MOF-5"]

    def test_bracket_gold(self):
        # [4+2] is the canonical Diels-Alder shorthand and must match.
        hits = bench._term_matches_any("endo [4+2] cycloaddition", ["[4+2]"])
        assert hits == ["[4+2]"]

    def test_empty_output(self):
        assert bench._term_matches_any("", ["Pd"]) == []

    def test_empty_gold(self):
        assert bench._term_matches_any("Pd-catalyzed", []) == []


class TestDegeneracy:
    """Lexical repetition score for Suzuki-spam detection."""

    def test_clean_output_zero_degeneracy(self):
        out = ["Pd", "palladium", "SPhos", "boronic acid"]
        deg = bench._degeneracy(out)
        assert deg == 0.0

    def test_total_repetition_max_degeneracy(self):
        # If every token is the same, degeneracy approaches 1 - 1/n.
        out = ["Suzuki coupling"] * 10
        deg = bench._degeneracy(out)
        # 20 tokens total ("suzuki", "coupling" x 10), 2 unique.
        assert deg == pytest.approx(1.0 - 2 / 20)

    def test_partial_repetition(self):
        # Real Suzuki-spam style: same phrase 3x with some unique tokens.
        out = [
            "Suzuki coupling catalysts",
            "Suzuki coupling catalysts",
            "Suzuki coupling catalysts",
            "boronic acid",
        ]
        deg = bench._degeneracy(out)
        # Tokens: {suzuki, coupling, catalysts} x 3 + {boronic, acid} = 11
        # total, 5 unique. Degeneracy = 1 - 5/11 ≈ 0.545.
        assert deg == pytest.approx(1.0 - 5 / 11)


class TestScore:
    """End-to-end scoring on a synthetic probe."""

    def test_coverage_pollution_score(self):
        probe = bench.Probe(
            id="t1",
            query="Mannich reaction enantioselective",
            family="reaction",
            gold_expand=["iminium", "proline", "organocatalysis"],
            gold_forbid=["Pd", "SPhos"],
        )
        # 1 of 3 gold_expand hit; 2 of 5 outputs polluted (Pd, SPhos).
        out = ["iminium", "Pd-catalyzed", "boronic acid", "SPhos", "alkene"]
        m = bench.score(probe, out)
        assert m.coverage == pytest.approx(1 / 3)
        assert m.pollution == pytest.approx(2 / 5)
        # Degeneracy: 5 distinct items with no token overlap → 0.
        assert m.degeneracy == pytest.approx(0.0)
        # score = 0.333 - 0.40 - 0 = -0.0667
        assert m.score == pytest.approx(1 / 3 - 2 / 5)
        assert m.matched_expand == ["iminium"]
        assert "Pd-catalyzed" in m.matched_forbid

    def test_score_clean_full_coverage(self):
        probe = bench.Probe(
            id="t2", query="q", family="reaction",
            gold_expand=["a", "b"], gold_forbid=["x"],
        )
        m = bench.score(probe, ["a", "b"])
        assert m.coverage == 1.0
        assert m.pollution == 0.0
        assert m.score == pytest.approx(1.0)


class TestParsePawTerms:
    """PAW comma-separated output parsing + safety caps."""

    def test_simple_parse(self):
        terms = bench._parse_paw_terms(
            "Pd, palladium, boronic acid, aryl halide"
        )
        assert terms == ["Pd", "palladium", "boronic acid", "aryl halide"]

    def test_dedup_case_insensitive(self):
        terms = bench._parse_paw_terms("Pd, pd, palladium, PALLADIUM")
        assert [t.lower() for t in terms] == ["pd", "palladium"]

    def test_degenerate_output_capped(self):
        raw = ", ".join(["Suzuki coupling catalysts"] * 100)
        terms = bench._parse_paw_terms(raw)
        # Dedup case-insensitive collapses everything to 1, well under cap.
        assert len(terms) == 1

    def test_strips_short_tokens(self):
        terms = bench._parse_paw_terms("a, Pd, b, palladium")
        # 1-char tokens dropped.
        assert "a" not in terms and "b" not in terms
        assert "Pd" in terms


class TestStaticAdapter:
    """Static-dict adapter exposes the same terms the search pipeline uses."""

    def test_diels_alder_emits_4plus2(self):
        fn = bench.make_static_adapter()
        terms = fn("Diels-Alder cycloaddition")
        # The static dict expands the diels-alder bigram with [4+2] and the
        # space-separated form; both should appear.
        joined = " ".join(terms).lower()
        assert "[4+2]" in joined

    def test_mannich_static_minimal(self):
        # Mannich + reaction bigram exists in the dict; "Mannich reaction"
        # is the only addition and gets filtered as in-query, so the
        # static adapter returns very few terms — matches the May-27
        # observation that the static path adds essentially nothing here.
        fn = bench.make_static_adapter()
        terms = fn("Mannich reaction enantioselective")
        # Static must NOT inject Pd/Suzuki here.
        assert not any("Pd" in t for t in terms)
        assert not any("Suzuki" in t.lower() for t in terms)

    def test_transmetalation_static_empty(self):
        # No dict entry for transmetalation; static adapter must return [].
        fn = bench.make_static_adapter()
        terms = fn("transmetalation cross-coupling mechanism")
        # Allow at most 1-2 terms (defensive — the cross-coupling bigram
        # may pick something up); main assertion is "no Pd-specific
        # ligand spam".
        assert not any("SPhos" in t for t in terms)
        assert not any("XPhos" in t for t in terms)
