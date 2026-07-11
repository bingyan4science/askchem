"""Tests for rate limiting middleware in askchem.server."""

import pytest

pytest.importorskip("fastapi")


class TestRateLimit:
    def test_anonymous_rate_limit_triggers(self, rate_client):
        """With ANON_RATE_LIMIT=3, the 4th request should return 429."""
        for i in range(3):
            r = rate_client.get("/api/stats")
            assert r.status_code == 200, f"Request {i+1} should succeed"

        r = rate_client.get("/api/stats")
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["detail"]

    def test_rate_limit_headers_present(self, rate_client):
        """Non-exempt API responses should include OpenAI-style rate limit headers."""
        r = rate_client.get("/api/stats")
        assert r.status_code == 200
        assert "x-ratelimit-limit-requests" in r.headers
        assert "x-ratelimit-remaining-requests" in r.headers
        assert "x-ratelimit-reset-requests" in r.headers
        assert r.headers["x-ratelimit-limit-requests"] == "3"

    def test_rate_limit_remaining_decrements(self, rate_client):
        r1 = rate_client.get("/api/stats")
        r2 = rate_client.get("/api/stats")
        rem1 = int(r1.headers["x-ratelimit-remaining-requests"])
        rem2 = int(r2.headers["x-ratelimit-remaining-requests"])
        assert rem2 == rem1 - 1

    def test_exempt_path_not_rate_limited(self, rate_client):
        """Exempt paths like /api/health should never be rate-limited."""
        for _ in range(10):
            r = rate_client.get("/api/health")
            assert r.status_code == 200
        assert "x-ratelimit-limit-requests" not in r.headers

    def test_non_api_path_not_rate_limited(self, rate_client):
        """Non-API paths (e.g. /) should bypass rate limiting."""
        for _ in range(10):
            r = rate_client.get("/")
            assert r.status_code == 200
        assert "x-ratelimit-limit-requests" not in r.headers
