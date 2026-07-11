"""Tests for askchem.display — smart_title and chemistry formatting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askchem.display import smart_title


class TestSmartTitleBasic:
    def test_empty_string(self):
        assert smart_title("") == ""

    def test_single_word(self):
        assert smart_title("catalysis") == "Catalysis"

    def test_multi_word(self):
        assert smart_title("rotating_disk_electrode") == "Rotating Disk Electrode"


class TestSmartTitleAbbreviations:
    def test_nmr(self):
        assert smart_title("nmr_spectroscopy") == "NMR Spectroscopy"

    def test_dft(self):
        assert smart_title("dft_computational_methods") == "DFT Computational Methods"

    def test_mof(self):
        result = smart_title("mof_synthesis")
        assert result.startswith("MOF")

    def test_ml(self):
        result = smart_title("ml_for_drug_discovery")
        assert result.startswith("ML")
        assert "for" in result  # lowercase preposition

    def test_hplc(self):
        assert "HPLC" in smart_title("hplc_analysis")


class TestSmartTitleLowercaseWords:
    def test_prepositions_lowercase(self):
        result = smart_title("ml_for_drug_discovery")
        assert " for " in result

    def test_first_word_always_capitalized(self):
        result = smart_title("in_situ_measurement")
        assert result[0] == "I"  # "In" not "in"


class TestSmartTitleChemicalFormulas:
    def test_co2(self):
        result = smart_title("co2_reduction")
        assert "CO2" in result

    def test_tio2(self):
        result = smart_title("tio2_photocatalysis")
        assert "TiO2" in result

    def test_h2o(self):
        result = smart_title("h2o_splitting")
        assert "H2O" in result

    def test_fe2o3(self):
        result = smart_title("fe2o3_nanoparticles")
        assert "Fe2O3" in result


class TestSmartTitleBondPatterns:
    def test_c_h_activation(self):
        result = smart_title("c_h_activation")
        assert "C" in result and "H" in result
        assert "Activation" in result

    def test_c_c_bond(self):
        result = smart_title("c_c_bond_formation")
        assert "C" in result
        assert "Bond" in result or "bond" in result


class TestSmartTitleElements:
    def test_element_title_cased(self):
        result = smart_title("pd_catalyzed")
        assert result.startswith("Pd")

    def test_cu_nanoparticles(self):
        result = smart_title("cu_nanoparticles")
        assert result.startswith("Cu")
