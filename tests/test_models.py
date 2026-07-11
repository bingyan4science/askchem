"""Tests for askchem.models — core data models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askchem.models import Claim, Source, TreeNode, View, PaperKnowledge


class TestSource:
    def test_source_id_from_doi(self):
        s = Source(doi="10.1234/test.paper", title="Test", authors=["A"], year=2024)
        assert s.source_id == "10-1234_test-paper"

    def test_source_id_fallback_to_s2id(self):
        s = Source(doi="", title="Test", authors=["A"], year=2024,
                   semantic_scholar_id="abc123")
        assert s.source_id == "abc123"

    def test_roundtrip(self):
        s = Source(doi="10.1234/test", title="Test Paper",
                   authors=["Alice", "Bob"], year=2024, venue="Nature")
        d = s.to_dict()
        s2 = Source.from_dict(d)
        assert s2.doi == s.doi
        assert s2.authors == s.authors
        assert s2.year == s.year

    def test_from_dict_ignores_extra_keys(self):
        d = {"doi": "10.1234/test", "title": "T", "authors": [], "year": 2024,
             "extra_field": "ignored"}
        s = Source.from_dict(d)
        assert s.doi == "10.1234/test"


class TestClaim:
    def test_generate_id_deterministic(self):
        id1 = Claim.generate_id("10.1234/test", "reaction", "hash1")
        id2 = Claim.generate_id("10.1234/test", "reaction", "hash1")
        assert id1 == id2
        assert len(id1) == 16

    def test_generate_id_varies_with_input(self):
        id1 = Claim.generate_id("10.1234/test", "reaction", "hash1")
        id2 = Claim.generate_id("10.1234/test", "reaction", "hash2")
        assert id1 != id2

    def test_to_dict_omits_empty_fields(self):
        c = Claim(claim_id="abc", claim_type="reaction", source_doi="10.1234/test",
                  source_paper_title="Paper", confidence="high",
                  extraction_model="gpt-5.4", extraction_version="v2")
        d = c.to_dict()
        assert "claim_id" in d
        assert "steps" not in d  # empty list omitted
        assert "subject" not in d  # empty string omitted

    def test_to_dict_keeps_false_and_zero(self):
        c = Claim(claim_id="abc", claim_type="reaction", source_doi="10.1234/test",
                  source_paper_title="Paper", confidence="high",
                  extraction_model="gpt-5.4", extraction_version="v2",
                  is_key_result=False)
        d = c.to_dict()
        assert "is_key_result" in d
        assert d["is_key_result"] is False

    def test_roundtrip(self):
        c = Claim(claim_id="abc", claim_type="property", source_doi="10.1234/test",
                  source_paper_title="Paper", subject="benzene",
                  property_name="melting_point", value="5.5", unit="C",
                  confidence="high", extraction_model="gpt-5.4",
                  extraction_version="v2")
        d = c.to_dict()
        c2 = Claim.from_dict(d)
        assert c2.claim_id == c.claim_id
        assert c2.subject == "benzene"
        assert c2.property_name == "melting_point"

    def test_new_fields_exist(self):
        c = Claim(claim_id="abc", claim_type="hypothesis",
                  source_doi="10.1234/test", source_paper_title="Paper",
                  confidence="high", extraction_model="gpt-5.4",
                  extraction_version="v2",
                  hypothesis_text="We hypothesize that...",
                  extraction_tier="full_paper")
        assert c.hypothesis_text == "We hypothesize that..."
        assert c.extraction_tier == "full_paper"

    def test_extraction_tier_field(self):
        c = Claim(claim_id="abc", claim_type="reaction",
                  source_doi="10.1234/test", source_paper_title="Paper",
                  confidence="high", extraction_model="gpt-5-mini",
                  extraction_version="v2",
                  extraction_tier="abstract_only")
        assert c.extraction_tier == "abstract_only"


class TestTreeNode:
    def test_roundtrip(self):
        n = TreeNode(node_id="n1", name="Catalysis",
                     path=["catalysis"], view="by_reaction_type",
                     level=1, claim_count=42)
        d = n.to_dict()
        n2 = TreeNode.from_dict(d)
        assert n2.name == "Catalysis"
        assert n2.claim_count == 42

    def test_year_range_list_to_tuple(self):
        d = {"node_id": "n1", "name": "Test", "path": ["a"],
             "view": "by_reaction_type", "year_range": [2020, 2024]}
        n = TreeNode.from_dict(d)
        assert isinstance(n.year_range, tuple)
        assert n.year_range == (2020, 2024)

    def test_defaults_for_missing_fields(self):
        d = {"node_id": "n1", "name": "Test"}
        n = TreeNode.from_dict(d)
        assert n.path == []
        assert n.view == ""


class TestView:
    def test_roundtrip(self):
        v = View(view_id="by_reaction_type", name="By Reaction Type",
                 description="desc", organizing_principle="reaction_type")
        d = v.to_dict()
        v2 = View.from_dict(d)
        assert v2.view_id == "by_reaction_type"


class TestPaperKnowledge:
    def test_roundtrip(self):
        pk = PaperKnowledge(
            doi="10.1234/test",
            hypothesis="We hypothesize that...",
            conclusions=["It works"],
            limitations=["Small sample size"],
        )
        d = pk.to_dict()
        pk2 = PaperKnowledge.from_dict(d)
        assert pk2.hypothesis == "We hypothesize that..."
        assert pk2.conclusions == ["It works"]
