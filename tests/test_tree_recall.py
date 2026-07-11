"""Unit tests for taxonomy tree recall used in hybrid search."""

import pytest

import askchem.db as db


def _node(view_id: str, path: str, words: list[str]):
    """Build a tree-node tuple matching ``_load_tree_node_index``'s
    pre-stemmed format (view_id, path, words, stem_set, stem_tuple).
    The δ1 latency fix moved stemming from per-query to load-time, so
    the test fixture must mirror that shape."""
    stems = tuple(db._stem(w) for w in words)
    return (view_id, path, tuple(words), frozenset(stems), stems)


def test_match_tree_nodes_respects_view_restriction(monkeypatch):
    """When searching within one view, do not match nodes from other views."""
    fake_index = [
        _node("by_reaction_type", "cross_coupling/suzuki_miyaura",
              ["cross", "coupling", "suzuki", "miyaura"]),
        _node("by_technique", "chromatography/hplc",
              ["chromatography", "hplc"]),
    ]
    monkeypatch.setattr(db, "_load_tree_node_index", lambda: fake_index)

    open_hits = db._match_tree_nodes("suzuki hplc", top_k=10)
    views_open = {v for v, _, _ in open_hits}
    assert "by_reaction_type" in views_open
    assert "by_technique" in views_open

    tech_only = db._match_tree_nodes(
        "suzuki hplc", top_k=10, restrict_view_id="by_technique"
    )
    assert tech_only
    assert all(v == "by_technique" for v, _, _ in tech_only)
    assert not any(v == "by_reaction_type" for v, _, _ in tech_only)


def test_match_tree_nodes_rejects_generic_coupling_alone(monkeypatch):
    """A multi-word query must not match taxonomy nodes on 'coupling' alone."""
    fake_index = [
        _node("by_reaction_type", "cross_coupling/suzuki_miyaura",
              ["cross", "coupling", "suzuki", "miyaura"]),
        _node("by_mechanism", "condensed_matter/spin_coupling",
              ["condensed", "matter", "spin", "coupling"]),
    ]
    monkeypatch.setattr(db, "_load_tree_node_index", lambda: fake_index)

    hits = db._match_tree_nodes("Suzuki coupling", top_k=10)
    paths = {p for _, p, _ in hits}
    assert "cross_coupling/suzuki_miyaura" in paths
    assert "condensed_matter/spin_coupling" not in paths
