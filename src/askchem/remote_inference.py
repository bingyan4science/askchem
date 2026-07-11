"""Optional remote GPU inference clients for the embed and rerank stages.

Both are env-var gated. When the corresponding URL is unset, the
``*_enabled`` helper returns ``False`` and callers stay on the local
CPU/MPS path. When the URL is set but the request fails (timeout, 5xx,
network), the call raises and the dispatcher in ``retrieval.py`` falls
back to local execution so search never silently degrades to "no
rerank" / "no semantic".

Env vars consumed:
  CHEMTREE_REMOTE_RERANK_URL   POST endpoint for cross-encoder rerank
  CHEMTREE_REMOTE_EMBED_URL    POST endpoint for query embedding
  CHEMTREE_REMOTE_AUTH_TOKEN   value sent as the X-Auth-Token header
  CHEMTREE_REMOTE_TIMEOUT_S    request timeout in seconds (default 5)
  CHEMTREE_REMOTE_TRACE        ``1`` prints per-call wire timings to stderr

The endpoint contract matches ``modal_app/search_gpu.py``:

  POST <embed_url>   body  = {"queries": [str, ...]}
                     reply = {"embeddings": [[float; 1024], ...], "dim": int,
                              "model": str}

  POST <rerank_url>  body  = {"query": str, "texts": [str, ...]}
                     reply = {"scores": [float, ...], "model": str}
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Iterable, Optional

import numpy as np
import requests


ENV_RERANK_URL = "CHEMTREE_REMOTE_RERANK_URL"
ENV_EMBED_URL = "CHEMTREE_REMOTE_EMBED_URL"
ENV_AUTH_TOKEN = "CHEMTREE_REMOTE_AUTH_TOKEN"
ENV_TIMEOUT = "CHEMTREE_REMOTE_TIMEOUT_S"
ENV_TRACE = "CHEMTREE_REMOTE_TRACE"


# ── shared session with TCP keep-alive ───────────────────────────────────
_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is None:
            s = requests.Session()
            # Keep-alive amortises TLS handshake across the burst of
            # remote calls within a single /api/search request.
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=8, max_retries=0,
            )
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            _session = s
    return _session


def _timeout_s() -> float:
    try:
        return float(os.environ.get(ENV_TIMEOUT, "5") or "5")
    except ValueError:
        return 5.0


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    tok = os.environ.get(ENV_AUTH_TOKEN, "")
    if tok:
        h["X-Auth-Token"] = tok
    return h


def _trace_enabled() -> bool:
    return os.environ.get(ENV_TRACE, "0") == "1"


# ── public API ───────────────────────────────────────────────────────────
def remote_rerank_enabled() -> bool:
    return bool(os.environ.get(ENV_RERANK_URL))


def remote_embed_enabled() -> bool:
    return bool(os.environ.get(ENV_EMBED_URL))


def remote_rerank(
    query: str,
    candidates: Iterable[tuple[str, str]],
    top_k: int = 20,
) -> tuple[list[tuple[str, float]], float]:
    """Send the (query, texts) batch to the Modal rerank endpoint.

    Returns ``(reordered_pairs, elapsed_ms)`` where ``reordered_pairs``
    is ``[(claim_id, score), ...]`` in descending score order truncated
    to ``top_k``. Raises on transport/HTTP error so the caller can fall
    back to local rerank.
    """
    url = os.environ.get(ENV_RERANK_URL, "")
    if not url:
        raise RuntimeError("CHEMTREE_REMOTE_RERANK_URL not set")

    ids: list[str] = []
    texts: list[str] = []
    for cid, text in candidates:
        if text:
            ids.append(cid)
            texts.append(text)
    if not ids:
        return [], 0.0

    t0 = time.monotonic()
    resp = _get_session().post(
        url,
        json={"query": query, "texts": texts},
        headers=_headers(),
        timeout=_timeout_s(),
    )
    resp.raise_for_status()
    scores = resp.json().get("scores") or []
    if len(scores) != len(ids):
        raise RuntimeError(
            f"remote rerank returned {len(scores)} scores for {len(ids)} texts"
        )
    elapsed_ms = (time.monotonic() - t0) * 1000

    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    out = [(ids[i], float(scores[i])) for i in order[: max(0, top_k)]]

    if _trace_enabled():
        print(
            f"[remote_rerank] n={len(ids)} -> top{top_k} in {elapsed_ms:.0f} ms",
            file=sys.stderr,
        )
    return out, elapsed_ms


def remote_embed_query(query: str) -> tuple[np.ndarray, float]:
    """Send a single query to the Modal embed endpoint.

    Returns ``(vector, elapsed_ms)`` where ``vector`` is the raw 1024-d
    L2-normalised mxbai output (no Matryoshka truncation; the caller is
    expected to apply ``CHEMTREE_V2_DIM`` truncation client-side so the
    behaviour matches the local ``embeddings_v2.embed_query`` path).
    Raises on transport/HTTP error.
    """
    url = os.environ.get(ENV_EMBED_URL, "")
    if not url:
        raise RuntimeError("CHEMTREE_REMOTE_EMBED_URL not set")

    t0 = time.monotonic()
    resp = _get_session().post(
        url,
        json={"queries": [query]},
        headers=_headers(),
        timeout=_timeout_s(),
    )
    resp.raise_for_status()
    payload = resp.json()
    embeddings = payload.get("embeddings") or []
    if not embeddings:
        raise RuntimeError("remote embed returned no embeddings")
    vec = np.asarray(embeddings[0], dtype=np.float32)
    elapsed_ms = (time.monotonic() - t0) * 1000

    if _trace_enabled():
        print(
            f"[remote_embed] dim={vec.shape[0]} in {elapsed_ms:.0f} ms",
            file=sys.stderr,
        )
    return vec, elapsed_ms


__all__ = [
    "remote_rerank_enabled",
    "remote_embed_enabled",
    "remote_rerank",
    "remote_embed_query",
]
