"""Phase γ1 cross-encoder reranker — final stage of the v2 retrieval
pipeline.

The bake-off (``docs/plans/2026-05-04-sprint-c-rerank-results.md``)
selected ``cross-encoder/ms-marco-MiniLM-L-6-v2`` reranking the top-20
of the dense ANN candidates as the production config.  It lifts
nDCG@10 by **+0.022** over the dense-only mxbai baseline (and **+0.110**
over today's MiniLM-only retrieval) at p95 latency 150 ms on Apple-MPS,
well inside the 400 ms budget.

Public surface:

    rerank(query, candidates, top_k=20)
        candidates: list[(claim_id, claim_text)]
        returns:    list[(claim_id, score)]   reordered, len ≤ top_k

The cross-encoder is loaded lazily.  All callers should hydrate
``claim_text`` consistently with the indexing-side text builder
(``embeddings._claim_to_text``) so that retrieval and reranking see
the same document representation.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Iterable, Optional

# ---- model defaults (overridable via env) --------------------------------
DEFAULT_MODEL_ID = os.environ.get(
    "CHEMTREE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
DEFAULT_BATCH = int(os.environ.get("CHEMTREE_RERANK_BATCH", "128"))
DEFAULT_MAX_LEN = int(os.environ.get("CHEMTREE_RERANK_MAX_LEN", "512"))
DEFAULT_TOP_K = int(os.environ.get("CHEMTREE_RERANK_TOP_K", "20"))

_model = None
_model_id_loaded: Optional[str] = None
_MODEL_LOCK = threading.Lock()


def _get_device() -> str:
    forced = os.environ.get("CHEMTREE_RERANK_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    forced2 = os.environ.get("CHEMTREE_EMB_DEVICE", "").strip().lower()
    if forced2 in {"cpu", "mps", "cuda"}:
        return forced2
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _maybe_quantize(model):
    """Dynamically int8-quantize Linear layers when CHEMTREE_RERANK_QUANT=int8.

    Added for the May-14 ablation. PyTorch dynamic quantization replaces
    every nn.Linear with a QInt8 op at inference time. Works only on CPU
    (the QNNPACK / FBGEMM backends), so we also force the underlying
    torch model onto CPU regardless of _get_device().
    """
    quant = os.environ.get("CHEMTREE_RERANK_QUANT", "").strip().lower()
    if quant not in {"int8", "qint8"}:
        return model
    try:
        import torch
        import torch.nn as nn
        inner = getattr(model, "model", None) or getattr(model, "_target_device", model)
        target = getattr(model, "model", None)
        if target is None:
            return model
        target.to("cpu")
        quantized = torch.quantization.quantize_dynamic(
            target, {nn.Linear}, dtype=torch.qint8,
        )
        model.model = quantized
        try:
            model._target_device = torch.device("cpu")
        except Exception:
            pass
        print(
            "[cross_encoder_rerank] applied dynamic int8 quantization "
            "(Linear → QInt8)",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[cross_encoder_rerank] int8 quantization skipped: {exc!r}",
            file=sys.stderr,
        )
    return model


def _get_model(model_id: str = DEFAULT_MODEL_ID):
    """Lazy-load the cross-encoder.  Lock-guarded for multi-threaded callers."""
    global _model, _model_id_loaded
    if _model is not None and _model_id_loaded == model_id:
        return _model
    with _MODEL_LOCK:
        if _model is None or _model_id_loaded != model_id:
            from sentence_transformers import CrossEncoder
            quant = os.environ.get("CHEMTREE_RERANK_QUANT", "").strip().lower()
            # int8 dynamic quantization is CPU-only; honour it by forcing
            # CPU regardless of available MPS/CUDA.
            device = "cpu" if quant in {"int8", "qint8"} else _get_device()
            _model = CrossEncoder(
                model_id, device=device, max_length=DEFAULT_MAX_LEN
            )
            _model = _maybe_quantize(_model)
            _model_id_loaded = model_id
            print(
                f"[cross_encoder_rerank] Loaded {model_id} on {device}"
                + (f" (quant={quant})" if quant else ""),
                file=sys.stderr,
            )
    return _model


def warmup() -> None:
    """Pre-load the cross-encoder so the first request isn't slow."""
    model = _get_model()
    try:
        model.predict(
            [("warmup query", "warmup passage about chemistry.")],
            batch_size=DEFAULT_BATCH,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception:
        pass


def is_available() -> bool:
    """True when the reranker can be loaded.  Cheap — never imports torch."""
    flag = os.environ.get("CHEMTREE_RERANK_ENABLED", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def rerank(
    query: str,
    candidates: Iterable[tuple[str, str]],
    top_k: int = DEFAULT_TOP_K,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH,
) -> list[tuple[str, float]]:
    """Rerank ``candidates`` by (query, doc) cross-encoder scores.

    ``candidates`` is an iterable of ``(claim_id, claim_text)``.  The
    text should be built with ``embeddings._claim_to_text`` so it
    matches the bi-encoder's indexed representation.

    Returns up to ``top_k`` ``(claim_id, score)`` tuples in descending
    score order.
    """
    pairs: list[tuple[str, str]] = []
    ids: list[str] = []
    for cid, text in candidates:
        if not text:
            continue
        pairs.append((query, text))
        ids.append(cid)
    if not pairs:
        return []

    model = _get_model(model_id)
    t0 = time.monotonic()
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000

    order = sorted(
        range(len(scores)), key=lambda i: -float(scores[i])
    )
    out = [(ids[i], float(scores[i])) for i in order[:top_k]]

    if os.environ.get("CHEMTREE_RERANK_TRACE", "0") == "1":
        print(
            f"[cross_encoder_rerank] q='{query[:48]}…' "
            f"n={len(pairs)} -> top{top_k} in {elapsed_ms:.0f} ms",
            file=sys.stderr,
        )
    return out
