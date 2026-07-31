"""
AskChem API client — OpenAI-style developer experience.

Usage:
    from askchem import AskChem

    ct = AskChem(base_url="https://askchem.org")

    # Search
    results = ct.search("Suzuki coupling", limit=20)

    # Browse
    node = ct.browse("by_reaction_type", path="catalysis/cross_coupling")

    # Bulk fetch claims
    claims = ct.claims.bulk(["id1", "id2", "id3"])

    # Feed
    discoveries = ct.feed(days=7)

    # Authors
    experts = ct.authors.search("John Hartwig")

    # Temporal
    timeline = ct.evolution("by_reaction_type", "catalysis/cross_coupling")

    # Context manager
    with AskChem() as ct:
        results = ct.search("MOF CO2")
"""

import os
import time
import logging
from typing import Optional

import httpx

from .models import (
    Claim, Source, SearchResult,
    BrowseResult, SourceResult, StatsResult, View,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://askchem.org"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ── Exceptions ────────────────────────────────────────────────────────────────

class AskChemError(Exception):
    """Base exception for all AskChem SDK errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class NotFoundError(AskChemError):
    """Resource not found (404)."""
    pass


class RateLimitError(AskChemError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = None, **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after


class ValidationError(AskChemError):
    """Request validation error (422)."""
    pass


class ServerError(AskChemError):
    """Server-side error (5xx)."""
    pass


# ── Namespaces ────────────────────────────────────────────────────────────────

class _ClaimsNamespace:
    """Namespace for claim-related operations: ct.claims.get(id)"""

    def __init__(self, client: "AskChem"):
        self._client = client

    def get(self, claim_id: str) -> Claim:
        """Get a specific claim by ID."""
        data = self._client._get(f"/api/claims/{claim_id}")
        return Claim(**data)

    def by_doi(self, doi: str) -> SourceResult:
        """Get all claims from a paper by DOI."""
        data = self._client._get(f"/api/sources/{doi}")
        return SourceResult(**data)

    def bulk(self, claim_ids: list[str]) -> list[Claim]:
        """Fetch up to 200 claims in a single request."""
        data = self._client._post("/api/claims/bulk", {"claim_ids": claim_ids})
        return [Claim(**c) for c in data.get("claims", [])]


class _SourcesNamespace:
    """Namespace for source-related operations: ct.sources.get(doi)"""

    def __init__(self, client: "AskChem"):
        self._client = client

    def get(self, doi: str) -> SourceResult:
        """Get all claims extracted from a paper."""
        data = self._client._get(f"/api/sources/{doi}")
        return SourceResult(**data)


class _AuthorsNamespace:
    """Namespace for author-related operations."""

    def __init__(self, client: "AskChem"):
        self._client = client

    def search(self, query: str, *, limit: int = 20) -> list[dict]:
        """Search authors by name."""
        data = self._client._get("/api/authors", {"q": query, "limit": limit})
        return data.get("authors", [])

    def experts(self, topic: str, *, limit: int = 20) -> list[dict]:
        """Find experts on a topic."""
        data = self._client._get("/api/authors", {"topic": topic, "limit": limit})
        return data.get("authors", [])

    def profile(self, author_id: str) -> dict:
        """Get full author profile."""
        return self._client._get(f"/api/authors/{author_id}")

    def network(self, author_id: str, *, depth: int = 1, limit: int = 30) -> dict:
        """Get co-authorship network."""
        return self._client._get(
            f"/api/authors/{author_id}/network",
            {"depth": depth, "limit": limit},
        )


# ── Main Client ───────────────────────────────────────────────────────────────

class AskChem:
    """
    AskChem API client.

    Args:
        api_key: API key for authentication. Falls back to ASKCHEM_API_KEY
            (or the legacy CHEMTREE_API_KEY) environment variable.
        base_url: Base URL of the AskChem API. Falls back to ASKCHEM_BASE_URL
            (or the legacy CHEMTREE_BASE_URL) environment variable.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts for transient errors (429, 5xx).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.api_key = (
            api_key
            or os.environ.get("ASKCHEM_API_KEY", "")
            or os.environ.get("CHEMTREE_API_KEY", "")
        )
        self.base_url = (
            base_url
            or os.environ.get("ASKCHEM_BASE_URL", "")
            or os.environ.get("CHEMTREE_BASE_URL", "")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        headers = {"User-Agent": "askchem-python/0.3.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

        self.claims = _ClaimsNamespace(self)
        self.sources = _SourcesNamespace(self)
        self.authors = _AuthorsNamespace(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the underlying HTTP client."""
        try:
            self._http.close()
        except Exception:
            pass

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with retry logic and error handling."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.request(method, path, **kwargs)

                if resp.status_code == 200:
                    return resp.json()

                body = {}
                try:
                    body = resp.json()
                except Exception:
                    pass
                detail = body.get("detail", resp.text[:200])

                if resp.status_code == 404:
                    raise NotFoundError(f"Not found: {path}", status_code=404, response=body)
                elif resp.status_code == 422:
                    raise ValidationError(f"Validation error: {detail}", status_code=422, response=body)
                elif resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", RETRY_BACKOFF * (2 ** attempt)))
                    if attempt < self.max_retries:
                        logger.info("Rate limited, retrying in %.1fs (attempt %d/%d)",
                                    retry_after, attempt + 1, self.max_retries)
                        time.sleep(retry_after)
                        continue
                    raise RateLimitError(
                        f"Rate limit exceeded: {detail}",
                        retry_after=retry_after, response=body,
                    )
                elif resp.status_code >= 500:
                    if attempt < self.max_retries:
                        wait = RETRY_BACKOFF * (2 ** attempt)
                        logger.info("Server error %d, retrying in %.1fs (attempt %d/%d)",
                                    resp.status_code, wait, attempt + 1, self.max_retries)
                        time.sleep(wait)
                        continue
                    raise ServerError(
                        f"Server error {resp.status_code}: {detail}",
                        status_code=resp.status_code, response=body,
                    )
                else:
                    raise AskChemError(
                        f"HTTP {resp.status_code}: {detail}",
                        status_code=resp.status_code, response=body,
                    )

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.info("Connection error, retrying in %.1fs (attempt %d/%d): %s",
                                wait, attempt + 1, self.max_retries, e)
                    time.sleep(wait)
                    continue
                raise AskChemError(f"Connection failed after {self.max_retries} retries: {e}") from e

        raise AskChemError(f"Request failed after {self.max_retries} retries") from last_exc

    def _get(self, path: str, params: dict = None) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json: dict = None) -> dict:
        return self._request("POST", path, json=json)

    # ── Core API methods ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        claim_type: Optional[str] = None,
        view: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SearchResult:
        """Search claims by text query."""
        params = {"q": query, "limit": limit, "offset": offset}
        if claim_type:
            params["claim_type"] = claim_type
        if view:
            params["view"] = view
        data = self._get("/api/search", params)
        return SearchResult(**data)

    def browse(
        self,
        view_id: str,
        *,
        path: str = "",
        depth: int = 1,
        limit: int = 50,
        offset: int = 0,
    ) -> BrowseResult:
        """Browse the knowledge tree with paginated claims."""
        if path:
            endpoint = f"/api/tree/{view_id}/{path}"
        else:
            endpoint = f"/api/tree/{view_id}"
        data = self._get(endpoint, {"depth": depth, "limit": limit, "offset": offset})
        return BrowseResult(**data)

    def views(self) -> list[View]:
        """List all available hierarchical views."""
        data = self._get("/api/views")
        return [View(**v) for v in data.get("views", [])]

    def stats(self) -> StatsResult:
        """Get index statistics."""
        data = self._get("/api/stats")
        return StatsResult(**data)

    def health(self) -> dict:
        """Check API health status."""
        return self._get("/api/health")

    def feed(self, *, limit: int = 20, days: int = 7) -> list[dict]:
        """Get the discoveries feed: recent high-impact claims."""
        data = self._get("/api/feed", {"limit": limit, "days": days})
        return data.get("discoveries", [])

    def temporal(self, view_id: str, path: str) -> dict:
        """Get year-by-year breakdown of claims at a tree node."""
        return self._get(f"/api/temporal/{view_id}/{path}")

    def evolution(self, view_id: str, path: str) -> dict:
        """Get rich evolution timeline for a tree node."""
        return self._get(f"/api/evolution/{view_id}/{path}")

    def time_browse(
        self,
        *,
        decade: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
    ) -> dict:
        """Browse claims by time period (decade -> year -> quarter)."""
        params = {}
        if decade:
            params["decade"] = decade
        if year:
            params["year"] = year
        if quarter:
            params["quarter"] = quarter
        return self._get("/api/time", params)

    def submit(
        self,
        doi: str,
        *,
        name: str = "",
        email: str = "",
        notes: str = "",
    ) -> dict:
        """Submit a paper for extraction."""
        return self._post("/api/submit", {
            "doi": doi, "name": name, "email": email, "notes": notes,
        })

    def submission_status(self, submission_id: int) -> dict:
        """Check the status of a paper submission."""
        return self._get(f"/api/submissions/{submission_id}")

    def subscribe(
        self,
        email: str,
        sub_type: str,
        target: str,
        *,
        frequency: str = "weekly",
    ) -> dict:
        """Subscribe to updates for a topic, author, or search query."""
        return self._post("/api/subscribe", {
            "email": email, "sub_type": sub_type,
            "target": target, "frequency": frequency,
        })

    def subscriptions(self, email: str, *, manage_token: str) -> list[dict]:
        """List all active subscriptions for an email (requires manage_token from subscribe)."""
        data = self._get("/api/subscriptions", {"email": email, "token": manage_token})
        return data.get("subscriptions", [])

    def cancel_subscription(self, sub_id: int, *, manage_token: str) -> dict:
        """Cancel a subscription."""
        return self._request(
            "DELETE", f"/api/subscriptions/{sub_id}", params={"token": manage_token}
        )

    def __repr__(self) -> str:
        return f"AskChem(base_url={self.base_url!r})"

    def __del__(self):
        self.close()
