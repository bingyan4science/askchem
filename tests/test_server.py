"""Tests for askchem.server — FastAPI endpoint integration tests."""

import pytest

pytest.importorskip("fastapi")


# ── Health & Stats ───────────────────────────────────────────────────────────


class TestHealthAndStats:
    def test_health_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "total_claims" in data

    def test_stats(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_claims"] == 876000
        assert data["total_sources"] == 56000

    def test_stats_has_expected_keys(self, client):
        data = client.get("/api/stats").json()
        for key in ("total_claims", "total_sources", "total_views",
                     "total_nodes", "claim_types", "year_distribution"):
            assert key in data


# ── Search ───────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_text(self, client):
        r = client.get("/api/search?q=benzene")
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] == 1

    def test_search_missing_query(self, client):
        r = client.get("/api/search")
        assert r.status_code == 422

    def test_search_grouped_removed(self, client):
        """`/api/search/grouped` was removed in the May-2026 'one search mode'
        consolidation. The UI now groups client-side over `/api/search?view=...`."""
        r = client.get("/api/search/grouped?q=catalysis&view=by_reaction_type")
        assert r.status_code == 404

    def test_search_with_view_filter(self, client):
        """Server-side view filtering still works on /api/search; client
        groups the results into a tree using `view_paths` on each claim."""
        r = client.get("/api/search?q=catalysis&view=by_reaction_type")
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_search_structure(self, client):
        r = client.get("/api/search/structure?smiles=c1ccccc1&type=substructure")
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_search_structure_invalid_type(self, client):
        r = client.get("/api/search/structure?smiles=c1ccccc1&type=bogus")
        assert r.status_code == 400


# ── Claims ───────────────────────────────────────────────────────────────────


class TestClaims:
    def test_get_claim(self, client):
        r = client.get("/api/claims/abc123")
        assert r.status_code == 200
        data = r.json()
        assert data["claim_id"] == "abc123"
        assert data["claim_type"] == "reaction"

    def test_get_claim_not_found(self, client):
        r = client.get("/api/claims/nonexistent")
        assert r.status_code == 404

    def test_bulk_claims(self, client):
        r = client.post("/api/claims/bulk",
                        json={"claim_ids": ["abc123", "xyz"]})
        assert r.status_code == 200
        data = r.json()
        assert data["requested"] == 2
        assert data["count"] == 1  # only abc123 matched

    def test_bulk_claims_too_many(self, client):
        r = client.post("/api/claims/bulk",
                        json={"claim_ids": ["id"] * 201})
        assert r.status_code == 400


# ── Views & Tree ─────────────────────────────────────────────────────────────


class TestViewsAndTree:
    def test_list_views(self, client):
        r = client.get("/api/views")
        assert r.status_code == 200
        data = r.json()
        assert "views" in data
        assert len(data["views"]) == 5

    def test_tree_root(self, client):
        r = client.get("/api/tree/by_reaction_type?depth=1")
        assert r.status_code == 200
        data = r.json()
        assert data["view_id"] == "by_reaction_type"
        assert "tree" in data

    def test_tree_node(self, client):
        r = client.get("/api/tree/by_reaction_type/catalysis?depth=1")
        assert r.status_code == 200
        data = r.json()
        assert data["view_id"] == "by_reaction_type"
        assert "claims" in data

    def test_tree_not_found(self, client):
        r = client.get("/api/tree/nonexistent_view")
        assert r.status_code == 404

    def test_children_search(self, client):
        r = client.get(
            "/api/views/by_data/children-search"
            "?parent_path=physical&q=conductivity"
        )
        assert r.status_code == 200
        data = r.json()
        assert data["view_id"] == "by_data"
        assert data["parent_path"] == "physical"
        assert data["query"] == "conductivity"
        assert isinstance(data["results"], list)
        assert data["count"] == len(data["results"])

    def test_children_search_requires_query(self, client):
        r = client.get("/api/views/by_data/children-search?parent_path=physical")
        assert r.status_code == 422


# ── Papers & Sources ─────────────────────────────────────────────────────────


class TestPapers:
    def test_papers_list(self, client):
        r = client.get("/api/papers")
        assert r.status_code == 200
        assert "papers" in r.json()

    def test_papers_search(self, client):
        r = client.get("/api/papers?q=suzuki")
        assert r.status_code == 200

    def test_source_claims(self, client):
        r = client.get("/api/sources/10.1234/test")
        assert r.status_code == 200
        data = r.json()
        assert data["doi"] == "10.1234/test"
        assert "claims" in data
        assert "authors" in data
        assert data["count"] == 1


# ── Temporal ─────────────────────────────────────────────────────────────────


class TestTemporal:
    def test_time_browse(self, client):
        r = client.get("/api/time")
        assert r.status_code == 200

    def test_time_decade(self, client):
        r = client.get("/api/time?decade=2020s")
        assert r.status_code == 200

    def test_temporal_overlay(self, client):
        r = client.get("/api/temporal/by_reaction_type/catalysis")
        assert r.status_code == 200
        assert "years" in r.json()

    def test_evolution(self, client):
        r = client.get("/api/evolution/by_reaction_type/catalysis")
        assert r.status_code == 200
        assert "years" in r.json()


# ── Authors ──────────────────────────────────────────────────────────────────


class TestAuthors:
    def test_authors_search(self, client):
        r = client.get("/api/authors?q=Hartwig")
        assert r.status_code == 200
        assert "authors" in r.json()

    def test_authors_topic(self, client):
        r = client.get("/api/authors?topic=catalysis")
        assert r.status_code == 200
        assert "authors" in r.json()

    def test_author_profile(self, client):
        r = client.get("/api/authors/A123")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Test Author"
        assert data["h_index"] == 45

    def test_author_network(self, client):
        r = client.get("/api/authors/A123/network")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "edges" in data

    def test_author_not_found(self, client):
        r = client.get("/api/authors/nonexistent")
        assert r.status_code == 404


# ── Feed & Reading List ──────────────────────────────────────────────────────


class TestFeedAndReadingList:
    def test_feed(self, client):
        r = client.get("/api/feed?limit=5&days=7")
        assert r.status_code == 200
        data = r.json()
        assert "discoveries" in data
        assert data["period_days"] == 7

    def test_reading_list(self, client):
        r = client.get("/api/reading-list/by_reaction_type/catalysis")
        assert r.status_code == 200
        data = r.json()
        assert data["total_papers"] == 1
        assert "tiers" in data


# ── Community Features ───────────────────────────────────────────────────────


class TestCommunity:
    def test_flag_valid(self, client):
        r = client.post("/api/flag", json={
            "claim_id": "abc123",
            "flag_type": "wrong_claim",
            "comment": "This is incorrect",
        })
        assert r.status_code == 200
        assert r.json()["flag_id"] == 1

    def test_flag_claim_not_found(self, client):
        r = client.post("/api/flag", json={
            "claim_id": "nonexistent",
            "flag_type": "wrong_claim",
        })
        assert r.status_code == 404

    def test_list_flags(self, client):
        r = client.get("/api/flags")
        assert r.status_code == 200
        data = r.json()
        assert "flags" in data
        assert "summary" in data

    def test_subscribe(self, client):
        r = client.post("/api/subscribe", json={
            "email": "test@test.com",
            "sub_type": "topic",
            "target": "catalysis",
            "frequency": "weekly",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["subscription_id"] == 1
        assert "manage_token" in data

    def test_list_subscriptions(self, client):
        r = client.get("/api/subscriptions?email=test@test.com&token=test-mtok")
        assert r.status_code == 200
        data = r.json()
        assert "subscriptions" in data
        assert data["count"] == 0

    def test_cancel_subscription(self, client):
        r = client.delete("/api/subscriptions/1?token=test-mtok")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


# ── Paper Submission ─────────────────────────────────────────────────────────


class TestSubmissions:
    def test_submit_already_indexed(self, client):
        r = client.post("/api/submit", json={
            "doi": "10.1234/test",
            "name": "Test User",
            "email": "test@test.com",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "already_indexed"
        assert data["claims_count"] == 1

    def test_submit_new_paper(self, client):
        r = client.post("/api/submit", json={
            "doi": "10.9999/new-paper",
            "name": "Test User",
            "email": "test@test.com",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "accepted"
        assert data["submission_id"] == 1
        assert "track_url" in data

    def test_submission_status(self, client):
        r = client.get("/api/submissions/1")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 1
        assert data["status"] == "processing"

    def test_submission_not_found(self, client):
        r = client.get("/api/submissions/999")
        assert r.status_code == 404


# ── API Keys ─────────────────────────────────────────────────────────────────


class TestApiKeys:
    def test_request_key(self, client):
        r = client.post("/api/keys/request", json={
            "name": "Test User",
            "email": "test@university.edu",
            "intended_use": "research",
        })
        assert r.status_code == 200
        data = r.json()
        assert "api_key" in data
        assert data["tier"] == "tier_1"
        assert data["rate_limit"] == 200

    def test_request_key_missing_fields(self, client):
        r = client.post("/api/keys/request", json={
            "name": "",
            "email": "test@test.com",
        })
        assert r.status_code == 400

    def test_request_key_invalid_email(self, client):
        r = client.post("/api/keys/request", json={
            "name": "Test",
            "email": "not-an-email",
        })
        assert r.status_code == 400


# ── Admin ────────────────────────────────────────────────────────────────────


class TestAdmin:
    def test_admin_queries_no_token(self, client):
        r = client.get("/api/admin/queries")
        assert r.status_code == 401

    def test_admin_queries_with_token(self, client):
        r = client.get("/api/admin/queries",
                       headers={"Authorization": "Bearer test-admin-token"})
        assert r.status_code == 200
        data = r.json()
        assert "top_queries" in data

    def test_admin_create_key_no_token(self, client):
        r = client.post("/api/admin/keys?name=test")
        assert r.status_code == 401


# ── Versioned API (v1) ───────────────────────────────────────────────────────


class TestV1:
    def test_v1_search(self, client):
        r = client.get("/v1/search?q=test")
        assert r.status_code == 200
        assert "results" in r.json()

    def test_v1_stats(self, client):
        r = client.get("/v1/stats")
        assert r.status_code == 200
        assert r.json()["total_claims"] == 876000

    def test_v1_views(self, client):
        r = client.get("/v1/views")
        assert r.status_code == 200
        assert "views" in r.json()


# ── Bulk Export & Changelog ──────────────────────────────────────────────────


class TestExport:
    def test_export(self, client):
        r = client.get("/api/export")
        assert r.status_code == 200
        assert "claims" in r.json()

    def test_changelog(self, client):
        r = client.get("/api/changelog")
        assert r.status_code == 200
        assert "changes" in r.json()


# ── Quality & Validation ────────────────────────────────────────────────────


class TestQuality:
    def test_quality(self, client):
        r = client.get("/api/quality")
        assert r.status_code == 200
        data = r.json()
        assert data["total_claims"] == 876000
        assert "year_range" in data

    def test_paper_validation_exists(self, client):
        r = client.get("/api/paper/validation/10.1234/test")
        assert r.status_code == 200
        data = r.json()
        assert data["validated"] is True
        assert data["crossref_verified"] is True

    def test_paper_validation_not_found(self, client):
        r = client.get("/api/paper/validation/10.9999/missing")
        assert r.status_code == 200
        data = r.json()
        assert data["validated"] is False


# ── Static Serving & API Root ────────────────────────────────────────────────


class TestStatic:
    def test_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_api_root(self, client):
        r = client.get("/api")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "AskChem"
        assert "endpoints" in data
        assert "stats" in data
