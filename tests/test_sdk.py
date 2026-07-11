"""Tests for the SDK client — exception hierarchy and retry logic."""

import importlib.util
import sys
from pathlib import Path

# Load SDK modules directly from file paths to avoid collision with src/askchem
_sdk_dir = Path(__file__).resolve().parent.parent / "sdk" / "askchem"


def _load_sdk_module(name: str, filepath: Path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(f"sdk_askchem.{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"sdk_askchem.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Load SDK models first (dependency of client)
sdk_models = _load_sdk_module("models", _sdk_dir / "models.py")

# Patch sys.modules so client.py can do `from .models import ...`
# We need a fake parent package
import types
_fake_pkg = types.ModuleType("sdk_askchem_pkg")
_fake_pkg.__path__ = [str(_sdk_dir)]
_fake_pkg.__package__ = "sdk_askchem_pkg"
sys.modules["sdk_askchem_pkg"] = _fake_pkg
sys.modules["sdk_askchem_pkg.models"] = sdk_models

# Now load client with the right parent package context
_client_spec = importlib.util.spec_from_file_location(
    "sdk_askchem_pkg.client", _sdk_dir / "client.py",
    submodule_search_locations=[])
_client_mod = importlib.util.module_from_spec(_client_spec)
_client_mod.__package__ = "sdk_askchem_pkg"
sys.modules["sdk_askchem_pkg.client"] = _client_mod
_client_spec.loader.exec_module(_client_mod)

AskChem = _client_mod.AskChem
AskChemError = _client_mod.AskChemError
NotFoundError = _client_mod.NotFoundError
RateLimitError = _client_mod.RateLimitError
ValidationError = _client_mod.ValidationError
ServerError = _client_mod.ServerError


class TestExceptionHierarchy:
    def test_not_found_is_askchem_error(self):
        assert issubclass(NotFoundError, AskChemError)

    def test_rate_limit_is_askchem_error(self):
        assert issubclass(RateLimitError, AskChemError)

    def test_validation_is_askchem_error(self):
        assert issubclass(ValidationError, AskChemError)

    def test_server_error_is_askchem_error(self):
        assert issubclass(ServerError, AskChemError)

    def test_askchem_error_has_status_code(self):
        e = AskChemError("test", status_code=500, response={"detail": "fail"})
        assert e.status_code == 500
        assert e.response == {"detail": "fail"}

    def test_rate_limit_has_retry_after(self):
        e = RateLimitError("too fast", retry_after=5.0)
        assert e.retry_after == 5.0
        assert e.status_code == 429


class TestClientInit:
    def test_default_base_url(self):
        ct = AskChem()
        assert ct.base_url == "https://askchem.org"
        ct.close()

    def test_custom_base_url(self):
        ct = AskChem(base_url="http://localhost:8080")
        assert ct.base_url == "http://localhost:8080"
        ct.close()

    def test_trailing_slash_stripped(self):
        ct = AskChem(base_url="http://localhost:8080/")
        assert ct.base_url == "http://localhost:8080"
        ct.close()

    def test_context_manager(self):
        with AskChem() as ct:
            assert ct.base_url == "https://askchem.org"

    def test_repr(self):
        ct = AskChem()
        assert "askchem.org" in repr(ct)
        ct.close()

    def test_max_retries_configurable(self):
        ct = AskChem(max_retries=5)
        assert ct.max_retries == 5
        ct.close()


class TestSDKModels:
    def test_claim_model_extra_allow(self):
        Claim = sdk_models.Claim
        c = Claim(claim_id="test", extra_field="allowed")
        assert c.claim_id == "test"

    def test_search_result_model(self):
        SearchResult = sdk_models.SearchResult
        sr = SearchResult(query="test", total=10, limit=50, offset=0)
        assert sr.total == 10

    def test_stats_result_model(self):
        StatsResult = sdk_models.StatsResult
        sr = StatsResult(total_claims=1000, total_sources=500)
        assert sr.total_claims == 1000
