"""Version-aware retrieval dispatcher.

Selects between the v1 (``all-MiniLM-L6-v2``, today's prod) and v2
(``mxbai-embed-large-v1`` + ``ms-marco-MiniLM-L-6-v2`` cross-encoder
rerank — Sprint-C winner) retrieval stacks at runtime.

Selection key (in priority order):
  1. The ``version=`` keyword argument to a public function.
  2. The ``CHEMTREE_RETRIEVER_VERSION`` env var (``v1`` | ``v2``).
  3. ``v1`` (default — preserves today's behaviour).

The v2 module silently falls back to "no-op" when its data files
(``data/claim_embeddings.v2.npz`` / ``.v2.faiss``) don't exist yet, so
flipping the env var is safe even before ``γ2`` finishes encoding.

Public API mirrors ``embeddings.py`` for drop-in use:

    embed_query(query, *, version=None)
    vector_search(query, top_k, min_score, *, version=None)
    semantic_rerank(query, candidate_ids, top_k, min_score, *, version=None)
    load_embeddings(*, version=None)
    is_loaded(*, version=None)
    active_version()              -> 'v1' | 'v2'
    cross_rerank_enabled()        -> bool

Cross-encoder rerank (γ1):
    cross_rerank(query, candidates, top_k=20)
This is exposed separately so callers (``askchem.search``) can wire
it in *after* their RRF / citation-boost stage rather than embed it
inside the dispatcher.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from . import embeddings as _v1
from . import embeddings_v2 as _v2
from . import cross_encoder_rerank as _ce
from . import remote_inference as _remote


def active_version(explicit: Optional[str] = None) -> str:
    """Resolve the active retriever version. Defaults to ``v1``."""
    if explicit:
        v = explicit.strip().lower()
        if v in {"v1", "v2"}:
            return v
    env = os.environ.get("CHEMTREE_RETRIEVER_VERSION", "").strip().lower()
    if env in {"v1", "v2"}:
        return env
    return "v1"


def _module(version: Optional[str]):
    return _v2 if active_version(version) == "v2" else _v1


# ---- public surface ------------------------------------------------------
def load_embeddings(*, version: Optional[str] = None) -> None:
    _module(version).load_embeddings()


def is_loaded(*, version: Optional[str] = None) -> bool:
    return _module(version).is_loaded()


def reload_embeddings(*, version: Optional[str] = None) -> None:
    _module(version).reload_embeddings()


def embed_query(query: str, *, version: Optional[str] = None) -> np.ndarray:
    """Embed a query, optionally via the remote Modal GPU endpoint.

    When ``CHEMTREE_REMOTE_EMBED_URL`` is set AND the v2 module is the
    selected dispatcher AND no cached vector exists for this query, the
    1024-d vector is fetched from the remote endpoint, Matryoshka-
    truncated to ``CHEMTREE_V2_DIM`` (if set), L2-renormalised, and
    written into the v2 module's query cache so subsequent local calls
    (vector_search / semantic_rerank) skip the round-trip. Any transport
    or HTTP error falls back to the local SentenceTransformer path.
    """
    mod = _module(version)
    if (
        mod is _v2
        and _remote.remote_embed_enabled()
        and query not in mod._query_cache
    ):
        try:
            vec, _ = _remote.remote_embed_query(query)
            # Apply Matryoshka truncation client-side so the cached
            # vector matches what local embed_query would have produced.
            if mod.QUERY_DIM and mod.QUERY_DIM != vec.shape[0]:
                vec = vec[: mod.QUERY_DIM]
                n = float(np.linalg.norm(vec))
                if n > 0:
                    vec = (vec / n).astype(np.float32)
            # Mirror the LRU semantics from mod.embed_query.
            if len(mod._query_cache) >= mod._QUERY_CACHE_MAX:
                oldest = next(iter(mod._query_cache))
                del mod._query_cache[oldest]
            mod._query_cache[query] = vec.astype(np.float32)
            return mod._query_cache[query]
        except Exception as exc:
            import sys as _sys
            print(
                f"[retrieval] remote embed failed: {exc!r}; falling back to local",
                file=_sys.stderr,
            )
    return mod.embed_query(query)


def vector_search(query: str, top_k: int = 200,
                  min_score: float = 0.20,
                  *, version: Optional[str] = None
                  ) -> list[tuple[str, float]]:
    return _module(version).vector_search(
        query, top_k=top_k, min_score=min_score
    )


def semantic_rerank(query: str, candidate_claim_ids: list[str],
                    top_k: int = 50, min_score: float = 0.25,
                    *, version: Optional[str] = None
                    ) -> list[tuple[str, float]]:
    return _module(version).semantic_rerank(
        query, candidate_claim_ids,
        top_k=top_k, min_score=min_score,
    )


# ---- cross-encoder rerank (γ1) ------------------------------------------
def cross_rerank_enabled() -> bool:
    """Cross-encoder rerank is opt-in: only on v2 *and* not env-disabled."""
    return active_version() == "v2" and _ce.is_available()


def cross_rerank(query: str,
                 candidates: Iterable[tuple[str, str]],
                 top_k: int = _ce.DEFAULT_TOP_K
                 ) -> list[tuple[str, float]]:
    """Rerank ``candidates`` (claim_id, text) pairs with the cross-encoder.

    When ``CHEMTREE_REMOTE_RERANK_URL`` is set, the (query, texts) batch
    is sent to the remote Modal GPU endpoint; any transport or HTTP
    error falls back to the local cross-encoder so search never silently
    degrades to "no rerank".
    """
    pairs = list(candidates)
    if _remote.remote_rerank_enabled() and pairs:
        try:
            out, _ = _remote.remote_rerank(query, pairs, top_k=top_k)
            return out
        except Exception as exc:
            import sys as _sys
            print(
                f"[retrieval] remote rerank failed: {exc!r}; falling back to local",
                file=_sys.stderr,
            )
    return _ce.rerank(query, pairs, top_k=top_k)


def warmup_cross_encoder() -> None:
    if cross_rerank_enabled():
        _ce.warmup()


__all__ = [
    "active_version",
    "load_embeddings",
    "reload_embeddings",
    "is_loaded",
    "embed_query",
    "vector_search",
    "semantic_rerank",
    "cross_rerank",
    "cross_rerank_enabled",
    "warmup_cross_encoder",
]
