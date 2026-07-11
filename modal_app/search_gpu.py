"""Modal GPU search acceleration for AskChem.

Two endpoints wrapped on a single GPU container, both behind an
``X-Auth-Token`` header:

  POST /embed_v1   {"queries": [str, ...]}
                  -> {"embeddings": [[float; 1024], ...]}
                     mxbai-embed-large-v1, query prefix applied, L2-normalized.
                     DO-side handles Matryoshka truncation downstream.

  POST /rerank_v1  {"query": str, "texts": [str, ...]}
                  -> {"scores": [float, ...]}
                     cross-encoder/ms-marco-MiniLM-L-6-v2, scores in input order.

Image bakes both model weights at build time so cold-start avoids a 30 s
HF download. ``@modal.enter`` then loads them onto CUDA and runs a tiny
warmup forward pass.

Deploy:

    ~/modal-cli/bin/modal deploy modal_app/search_gpu.py

Auth secret is named ``askchem-modal-auth`` and exposes
``ASKCHEM_MODAL_TOKEN`` to the container. Clients must send
``X-Auth-Token: <token>`` on every request.
"""

from __future__ import annotations

import os
import sys

import modal
from fastapi import Header, HTTPException

# ── model identifiers (must match src/askchem/embeddings_v2.py and
#    src/askchem/cross_encoder_rerank.py) ───────────────────────────────
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MAX_LEN = 512


def _predownload_models() -> None:
    """Pulls model weights into the image at build time.

    Runs inside the Modal image build (CPU only), so we deliberately do
    not push the models onto a GPU here. ``@modal.enter`` does that.
    """
    from sentence_transformers import SentenceTransformer, CrossEncoder

    SentenceTransformer(EMBED_MODEL, device="cpu")
    CrossEncoder(RERANK_MODEL, device="cpu", max_length=RERANK_MAX_LEN)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.3.1",
        "sentence-transformers==4.0.0",
        "transformers>=4.41,<4.50",
        "numpy<2",
        "fastapi[standard]",
    )
    .run_function(_predownload_models)
)


app = modal.App("askchem-search-gpu", image=image)

auth_secret = modal.Secret.from_name("askchem-modal-auth")


def _shared_warmup(self, device: str) -> None:
    """Load both models on ``device`` and run a single forward pass each."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
    self.reranker = CrossEncoder(
        RERANK_MODEL, device=device, max_length=RERANK_MAX_LEN
    )
    self.token = os.environ.get("ASKCHEM_MODAL_TOKEN", "")

    self.embedder.encode(
        [QUERY_PREFIX + "warmup"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    self.reranker.predict(
        [("warmup", "warmup passage about chemistry.")],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    print(
        f"[search_{device}] ready  embed={EMBED_MODEL}  rerank={RERANK_MODEL}",
        file=sys.stderr,
    )


@app.cls(
    gpu="T4",
    min_containers=0,
    max_containers=2,
    scaledown_window=60,
    secrets=[auth_secret],
)
@modal.concurrent(max_inputs=8)
class SearchGPU:
    """GPU container (T4) running both models on CUDA."""

    @modal.enter()
    def warmup(self) -> None:
        _shared_warmup(self, "cuda")

    @modal.fastapi_endpoint(method="POST")
    def embed_v1(
        self,
        body: dict,
        x_auth_token: str = Header(default=""),  # noqa: B008
    ) -> dict:
        return _do_embed(self, body, x_auth_token)

    @modal.fastapi_endpoint(method="POST")
    def rerank_v1(
        self,
        body: dict,
        x_auth_token: str = Header(default=""),  # noqa: B008
    ) -> dict:
        return _do_rerank(self, body, x_auth_token)


def _check_auth(self, x_auth_token: str) -> None:
    if not self.token:
        return
    if x_auth_token != self.token:
        raise HTTPException(status_code=401, detail="bad token")


def _do_embed(self, body: dict, x_auth_token: str) -> dict:
    _check_auth(self, x_auth_token)
    queries = body.get("queries") or []
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise HTTPException(status_code=400, detail="queries must be List[str]")
    if not queries:
        return {"embeddings": [], "dim": 1024, "model": EMBED_MODEL}

    prefixed = [QUERY_PREFIX + q for q in queries]
    vectors = self.embedder.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=min(32, len(prefixed)),
        show_progress_bar=False,
    )
    return {
        "embeddings": vectors.astype("float32").tolist(),
        "dim": int(vectors.shape[-1]),
        "model": EMBED_MODEL,
    }


def _do_rerank(self, body: dict, x_auth_token: str) -> dict:
    _check_auth(self, x_auth_token)
    query = body.get("query")
    texts = body.get("texts") or []
    if not isinstance(query, str) or not isinstance(texts, list):
        raise HTTPException(
            status_code=400, detail="payload must be {query: str, texts: List[str]}"
        )
    if not query or not texts:
        return {"scores": [], "model": RERANK_MODEL}

    batch_size = int(body.get("batch_size", 64) or 64)
    pairs = [(query, t) for t in texts]
    scores = self.reranker.predict(
        pairs,
        batch_size=max(1, batch_size),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return {
        "scores": [float(s) for s in scores],
        "model": RERANK_MODEL,
    }


# ── CPU sibling: same models, no GPU. Sized with extra memory and more
#    cores than the DO box so we can isolate "was the win the GPU
#    specifically, or just better silicon / more cores than DO?"
@app.cls(
    cpu=4.0,
    memory=16384,
    min_containers=0,
    max_containers=2,
    scaledown_window=60,
    secrets=[auth_secret],
)
@modal.concurrent(max_inputs=4)
class SearchCPU:
    """CPU container (4 vCPU, 16 GB) running both models on CPU."""

    @modal.enter()
    def warmup(self) -> None:
        _shared_warmup(self, "cpu")

    @modal.fastapi_endpoint(method="POST")
    def embed_v1(
        self,
        body: dict,
        x_auth_token: str = Header(default=""),  # noqa: B008
    ) -> dict:
        return _do_embed(self, body, x_auth_token)

    @modal.fastapi_endpoint(method="POST")
    def rerank_v1(
        self,
        body: dict,
        x_auth_token: str = Header(default=""),  # noqa: B008
    ) -> dict:
        return _do_rerank(self, body, x_auth_token)
