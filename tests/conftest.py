# PAW integration harness: compiles/runs neural programs; needs programasweights + network.
collect_ignore = ["test_paw.py"]

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("ASKCHEM_DISABLE_PAW_WARMUP", "1")

# Pre-mock askchem.embeddings (requires numpy + sentence-transformers)
# before askchem.server can import it.
_mock_embeddings = MagicMock()
sys.modules["askchem.embeddings"] = _mock_embeddings

# ── Sample data shared across server tests ──────────────────────────────────

SAMPLE_STATS = {
    "total_claims": 876000,
    "total_sources": 56000,
    "total_views": 7,
    "total_nodes": 12000,
    "claim_types": {"reaction": 300000, "property": 200000},
    "year_distribution": {"2023": 8000, "2024": 9000},
    "citation_source": "Semantic Scholar",
    "citation_source_url": "https://api.semanticscholar.org/",
    "citations_updated_at": "2025-03-01T00:00:00",
}

SAMPLE_CLAIM = {
    "claim_id": "abc123",
    "claim_type": "reaction",
    "source_doi": "10.1234/test",
    "source_paper_title": "Test Paper on Suzuki Coupling",
    "confidence": "high",
    "verbatim_quote": "The Suzuki coupling proceeded in 95% yield.",
    "extraction_model": "gemini-3.1-pro",
    "extraction_version": "v2",
    "reaction_type": "cross_coupling",
    "view_paths": {
        "by_reaction_type": ["coupling", "cross_coupling", "suzuki_miyaura"],
        "by_substance_class": ["inorganic_compounds", "metals_and_alloys", "palladium"],
    },
}

SAMPLE_SOURCE = {
    "doi": "10.1234/test",
    "title": "Test Paper on Suzuki Coupling",
    "authors": ["Alice Smith", "Bob Jones"],
    "year": 2024,
    "venue": "J. Test Chem.",
    "citation_count": 42,
}

SAMPLE_VIEWS = [
    {"view_id": "by_reaction_type", "name": "By Reaction Type", "description": "Browse by reaction type"},
    {"view_id": "by_substance_class", "name": "By Substance Class", "description": "Browse by substance"},
    {"view_id": "by_technique", "name": "By Technique", "description": "Browse by technique"},
    {"view_id": "by_application", "name": "By Application", "description": "Browse by application"},
    {"view_id": "by_mechanism", "name": "By Mechanism", "description": "Browse by mechanism"},
]

SAMPLE_TREE = {
    "path": "",
    "name": "root",
    "level": 0,
    "claim_count": 876000,
    "children_data": [
        {"path": "catalysis", "name": "Catalysis", "level": 1, "claim_count": 50000},
    ],
}

SAMPLE_AUTHOR = {
    "author_id": "A123",
    "name": "Test Author",
    "institution": "MIT",
    "institution_country": "US",
    "h_index": 45,
    "works_count": 200,
    "cited_by_count": 12000,
    "orcid": "0000-0001-2345-6789",
    "openalex_id": "https://openalex.org/A123",
    "paper_count": 5,
    "papers": [
        {"doi": "10.1234/test", "title": "Test Paper", "year": 2024,
         "position": "first", "citation_count": 42},
    ],
    "research_areas": [
        {"view_path": "by_reaction_type/catalysis", "claim_count": 15},
    ],
}


def _patch_db(monkeypatch):
    """Monkeypatch all chemtree.db functions with deterministic test stubs."""
    import askchem.db as db_mod

    monkeypatch.setattr(db_mod, "get_stats", lambda: SAMPLE_STATS)
    monkeypatch.setattr(db_mod, "search_claims",
                        lambda *a, **kw: {"results": [SAMPLE_CLAIM], "total": 1})
    monkeypatch.setattr(db_mod, "search_by_structure",
                        lambda *a, **kw: {"results": [], "total": 0})
    monkeypatch.setattr(db_mod, "get_claim",
                        lambda cid: SAMPLE_CLAIM if cid == "abc123" else None)
    monkeypatch.setattr(db_mod, "get_claims_bulk",
                        lambda ids: [SAMPLE_CLAIM for i in ids if i == "abc123"])
    monkeypatch.setattr(db_mod, "list_views", lambda: SAMPLE_VIEWS)
    monkeypatch.setattr(db_mod, "get_tree_with_depth",
                        lambda vid, path='', depth=1: (
                            SAMPLE_TREE if vid != "nonexistent_view" else None))
    monkeypatch.setattr(db_mod, "get_claims_at_node",
                        lambda *a, **kw: {"claims": [SAMPLE_CLAIM], "total": 1})
    monkeypatch.setattr(db_mod, "search_tree_children",
                        lambda *a, **kw: [
                            {"path": "physical/conductivity",
                             "name": "conductivity", "claim_count": 41},
                        ])
    monkeypatch.setattr(db_mod, "search_papers",
                        lambda **kw: {"papers": [], "total": 0})
    monkeypatch.setattr(db_mod, "get_source",
                        lambda doi: SAMPLE_SOURCE if doi == "10.1234/test" else None)
    monkeypatch.setattr(db_mod, "get_claims_by_doi",
                        lambda doi: [SAMPLE_CLAIM] if doi == "10.1234/test" else [])
    monkeypatch.setattr(db_mod, "get_authors_for_doi",
                        lambda doi: [{"author_id": "A123", "name": "Test Author"}])
    monkeypatch.setattr(db_mod, "get_by_time_period",
                        lambda **kw: {"decades": ["2020s"]})
    monkeypatch.setattr(db_mod, "get_temporal_overlay",
                        lambda *a: {"years": {}})
    monkeypatch.setattr(db_mod, "get_evolution_timeline",
                        lambda *a: {"years": {}})
    monkeypatch.setattr(db_mod, "get_reading_list",
                        lambda *a, **kw: {"total_papers": 1, "tiers": [],
                                          "topic": "catalysis"})
    monkeypatch.setattr(db_mod, "get_discoveries_feed", lambda **kw: [])
    monkeypatch.setattr(
        db_mod, "add_subscription",
        lambda **kw: {"subscription_id": 1, "manage_token": "test-mtok"},
    )
    monkeypatch.setattr(db_mod, "get_user_subscriptions", lambda user_id: [])
    monkeypatch.setattr(
        db_mod, "cancel_user_subscription", lambda user_id, sub_id: None
    )
    monkeypatch.setattr(db_mod, "get_subscription_row",
                        lambda sid: {"id": sid, "manage_token": "test-mtok"} if sid == 1 else None)
    monkeypatch.setattr(db_mod, "get_notification_history",
                        lambda *a, **kw: [])
    monkeypatch.setattr(db_mod, "search_authors", lambda *a, **kw: [])
    monkeypatch.setattr(db_mod, "find_experts", lambda *a, **kw: [])
    monkeypatch.setattr(db_mod, "get_top_authors", lambda **kw: [])
    monkeypatch.setattr(db_mod, "get_author_profile",
                        lambda aid: SAMPLE_AUTHOR if aid == "A123" else None)
    monkeypatch.setattr(db_mod, "get_coauthor_network",
                        lambda *a, **kw: {"nodes": [], "edges": []})
    monkeypatch.setattr(db_mod, "add_submission", lambda *a, **kw: 1)
    monkeypatch.setattr(db_mod, "get_submission",
                        lambda sid: (
                            {"id": sid, "doi": "10.1234/test",
                             "status": "processing",
                             "submitted_at": "2024-01-01",
                             "submitter_name": "Test"}
                            if sid == 1 else None))
    monkeypatch.setattr(db_mod, "list_submissions", lambda **kw: [])
    monkeypatch.setattr(db_mod, "update_submission", lambda *a, **kw: None)
    monkeypatch.setattr(db_mod, "add_flag", lambda **kw: 1)
    monkeypatch.setattr(db_mod, "list_flags", lambda **kw: [])
    monkeypatch.setattr(db_mod, "get_flag_summary", lambda: {"total": 0})
    monkeypatch.setattr(db_mod, "get_flags_for_claim", lambda cid: [])
    monkeypatch.setattr(db_mod, "get_paper_validation",
                        lambda doi: (
                            {"crossref_verified": True, "is_retracted": False,
                             "journal": "Test J.", "publisher": "Test Pub",
                             "is_chemistry": True, "validation_score": 0.8,
                             "validated_at": "2024-01-01",
                             "validation_data": None}
                            if doi == "10.1234/test" else None))
    monkeypatch.setattr(db_mod, "export_claims",
                        lambda **kw: {"claims": [], "total": 0})
    monkeypatch.setattr(db_mod, "get_changelog",
                        lambda **kw: {"changes": [], "total": 0})
    monkeypatch.setattr(db_mod, "get_query_stats",
                        lambda **kw: {"top_queries": [], "daily_counts": {}})
    monkeypatch.setattr(db_mod, "create_api_key",
                        lambda **kw: {"api_key": "ac-test-key-123",
                                      "key_id": "k_123"})
    monkeypatch.setattr(db_mod, "log_query", lambda **kw: None)
    monkeypatch.setattr(db_mod, "log_security_event", lambda *a, **k: None)
    monkeypatch.setattr(db_mod, "get_key_rpd_today", lambda kid: 0)
    monkeypatch.setattr(db_mod, "record_authenticated_api_request", lambda kid: None)
    monkeypatch.setattr(db_mod, "get_api_key_usage_summary",
                        lambda kid, days=30: {"total_requests": 0, "daily_usage": []})
    monkeypatch.setattr(db_mod, "get_security_events", lambda **kw: [])
    monkeypatch.setattr(db_mod, "validate_api_key", lambda key: None)


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with all external dependencies mocked."""
    pytest.importorskip("fastapi")
    _patch_db(monkeypatch)

    from askchem import server

    monkeypatch.setattr(server, "ADMIN_TOKEN", "test-admin-token")
    server._cache.clear()
    server._rate_buckets.clear()
    server._anon_rpd.clear()

    async def _noop(*a, **kw):
        pass

    monkeypatch.setattr(server, "process_submission", _noop)

    from starlette.testclient import TestClient

    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def rate_client(monkeypatch):
    """TestClient with a low anonymous rate limit for rate-limit tests."""
    pytest.importorskip("fastapi")
    _patch_db(monkeypatch)

    from askchem import server

    monkeypatch.setattr(server, "ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setattr(server, "ANON_RATE_LIMIT", 3)
    server._cache.clear()
    server._rate_buckets.clear()
    server._anon_rpd.clear()

    async def _noop(*a, **kw):
        pass

    monkeypatch.setattr(server, "process_submission", _noop)

    from starlette.testclient import TestClient

    with TestClient(server.app) as c:
        yield c
