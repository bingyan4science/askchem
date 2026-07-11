"""Tests for the TTL cache helpers in askchem.server."""

import time

import pytest

pytest.importorskip("fastapi")

from askchem.server import _cached, _set_cache, _cache, CACHE_TTL


class TestTTLCache:
    def setup_method(self):
        _cache.clear()

    def test_cached_returns_none_for_missing_key(self):
        assert _cached("nonexistent_key") is None

    def test_set_then_get(self):
        _set_cache("test_key", {"data": 42})
        assert _cached("test_key") == {"data": 42}

    def test_expired_entry_returns_none(self):
        _cache["stale"] = (time.time() - CACHE_TTL - 10, {"data": "old"})
        assert _cached("stale") is None

    def test_fresh_entry_returns_value(self):
        _cache["fresh"] = (time.time(), {"data": "new"})
        assert _cached("fresh") == {"data": "new"}

    def test_custom_ttl(self):
        _cache["short"] = (time.time() - 5, "value")
        assert _cached("short", ttl=3) is None
        assert _cached("short", ttl=10) == "value"
