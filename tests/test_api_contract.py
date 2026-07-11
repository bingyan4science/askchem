"""Contract tests: verify API response shapes match what the frontend JS expects.

The frontend (web/index.html) accesses specific keys from API responses.
These tests ensure the backend always returns the shapes the UI relies on.
"""

import pytest

pytest.importorskip("fastapi")


class TestStatsContract:
    """Frontend init() reads: total_claims, total_sources, total_views, total_nodes."""

    def test_stats_shape(self, client):
        data = client.get("/api/stats").json()
        assert isinstance(data["total_claims"], int)
        assert isinstance(data["total_sources"], int)
        assert isinstance(data["total_views"], int)
        assert isinstance(data["total_nodes"], int)
        assert isinstance(data.get("claim_types", {}), dict)
        assert isinstance(data.get("year_distribution", {}), dict)


class TestSearchContract:
    """Frontend doSearchFlat reads: results (list), total (int), intent (str)."""

    def test_flat_search_shape(self, client):
        data = client.get("/api/search?q=test").json()
        assert isinstance(data["results"], list)
        assert isinstance(data["total"], int)
        assert "intent" in data

    def test_search_with_view_shape(self, client):
        """Frontend now groups client-side using `view_paths` from each claim
        in the /api/search response (the old /api/search/grouped endpoint
        was removed in the May-2026 'one search mode' consolidation). The
        explicit limit=500 here guards the cap that doSearchGrouped relies on;
        if a future commit lowers the cap below 500 the grouped search tree
        on askchem.org will return HTTP 422 again and this test will catch it.
        """
        r = client.get("/api/search?q=test&view=by_reaction_type&limit=500")
        assert r.status_code == 200, (
            f"/api/search must accept limit=500 to support client-side grouping; "
            f"got {r.status_code}: {r.text[:200]}"
        )
        data = r.json()
        assert isinstance(data["results"], list)
        assert isinstance(data["total"], int)
        # Each claim should carry view_paths so client-side grouping can find
        # the path in any view the user picks.
        if data["results"]:
            assert "view_paths" in data["results"][0]


class TestViewsContract:
    """Frontend loadViews reads: views (list of {view_id, name, description})."""

    def test_views_shape(self, client):
        data = client.get("/api/views").json()
        views = data["views"]
        assert isinstance(views, list)
        assert len(views) > 0
        for v in views:
            assert "view_id" in v
            assert "name" in v


class TestTreeContract:
    """Frontend selectView reads: tree.children_data (list of node objects)."""

    def test_tree_root_shape(self, client):
        data = client.get("/api/tree/by_reaction_type?depth=1").json()
        tree = data["tree"]
        assert "children_data" in tree

    def test_tree_node_shape(self, client):
        """Frontend loadNodeContent reads: claims (list), total_claims, node."""
        data = client.get("/api/tree/by_reaction_type/catalysis?depth=1").json()
        assert isinstance(data["claims"], list)
        assert isinstance(data["total_claims"], int)
        assert "node" in data


class TestClaimContract:
    """Frontend renderClaim reads: claim_id, claim_type, source_doi,
    source_paper_title, confidence, verbatim_quote."""

    def test_claim_shape(self, client):
        data = client.get("/api/claims/abc123").json()
        for key in ("claim_id", "claim_type", "source_doi",
                     "source_paper_title", "confidence", "verbatim_quote"):
            assert key in data, f"Missing key: {key}"


class TestSourceContract:
    """Frontend showPaperPanel reads: claims (list), authors (list),
    source (dict with title/venue/year/citation_count), count (int)."""

    def test_source_shape(self, client):
        data = client.get("/api/sources/10.1234/test").json()
        assert isinstance(data["claims"], list)
        assert isinstance(data["authors"], list)
        assert isinstance(data["count"], int)
        assert data["source"] is not None
        src = data["source"]
        assert "title" in src


class TestAuthorContract:
    """Frontend showAuthorPanel reads: name, institution, h_index,
    paper_count, papers (list), research_areas (list)."""

    def test_author_shape(self, client):
        data = client.get("/api/authors/A123").json()
        assert isinstance(data["name"], str)
        assert isinstance(data["papers"], list)
        assert isinstance(data["research_areas"], list)
        assert "h_index" in data

    def test_author_network_shape(self, client):
        """Frontend drawNetwork reads: nodes (list), edges (list)."""
        data = client.get("/api/authors/A123/network").json()
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)


class TestFeedContract:
    """Frontend reads: discoveries (list), count (int), period_days (int)."""

    def test_feed_shape(self, client):
        data = client.get("/api/feed").json()
        assert isinstance(data["discoveries"], list)
        assert isinstance(data["count"], int)
        assert isinstance(data["period_days"], int)


class TestSubmitContract:
    """Frontend submitPaper reads: status, submission_id, track_url, stream_url."""

    def test_submit_accepted_shape(self, client):
        data = client.post("/api/submit", json={
            "doi": "10.9999/new", "name": "T", "email": "t@t.com",
        }).json()
        assert data["status"] == "accepted"
        assert isinstance(data["submission_id"], int)
        assert "track_url" in data
        assert "stream_url" in data

    def test_submit_already_indexed_shape(self, client):
        data = client.post("/api/submit", json={
            "doi": "10.1234/test",
        }).json()
        assert data["status"] == "already_indexed"
        assert isinstance(data["claims_count"], int)


class TestApiRootContract:
    """Frontend/agents read: name, endpoints (dict), stats (dict)."""

    def test_api_root_shape(self, client):
        data = client.get("/api").json()
        assert data["name"] == "AskChem"
        assert isinstance(data["endpoints"], dict)
        assert isinstance(data["stats"], dict)
        assert "total_claims" in data["stats"]
