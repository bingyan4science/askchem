"""Phase γ2/γ3 retrieval module — `mxbai-embed-large-v1` dense ANN.

This is the *v2* counterpart to ``embeddings.py`` (which stays on the
v1 ``all-MiniLM-L6-v2`` encoder).  The two modules expose the same
public surface so that ``askchem.retrieval`` can dispatch between them
based on the ``CHEMTREE_RETRIEVER_VERSION`` env var without rewriting
callers.

Public surface (mirrors ``embeddings.py``):

    load_embeddings()      — load .v2.npz + .v2.faiss into module state
    is_loaded()            — bool
    embed_query(query)     — str → np.ndarray (1024-dim, mxbai prefix applied)
    vector_search(...)     — global ANN over claim corpus
    semantic_rerank(...)   — re-score a candidate id list
    reload_embeddings()    — drop caches, force re-load

The bake-off
(``docs/plans/2026-05-02-sprint-c-bakeoff-results.md``) ranked
``mxbai-embed-large-v1`` first by nDCG@10 (+0.088 vs MiniLM control on
the 10K pilot).  ``γ2`` re-embeds the full 2.34 M-claim corpus with
this encoder; this module loads that re-embedded corpus.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---- model / file layout (must match scripts/build_embeddings_v2.py) ----
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_DIM = 1024
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOC_PREFIX = ""  # mxbai prefixes queries only

_REPO_ROOT = Path(__file__).parent.parent.parent
# Canonical DB is askchem.db (renamed from chemtree.db); keep the legacy name as
# a fallback for hosts not yet migrated.
DB_PATH = (_REPO_ROOT / "askchem.db") if (_REPO_ROOT / "askchem.db").exists() \
    else (_REPO_ROOT / "chemtree.db")


def _resolve_path(env_var: str, default: Path) -> Path:
    """Honour an env-var override, otherwise fall back to ``default``.

    Useful for evals: point the dispatcher at a pilot ``.npz`` /
    ``.faiss`` pair without rewriting code.
    """
    override = os.environ.get(env_var, "").strip()
    if override:
        return Path(override).expanduser()
    return default


EMBEDDINGS_PATH = _resolve_path(
    "CHEMTREE_EMBEDDINGS_V2_NPZ",
    _REPO_ROOT / "data" / "claim_embeddings.v2.npz",
)
FAISS_INDEX_PATH = _resolve_path(
    "CHEMTREE_EMBEDDINGS_V2_FAISS",
    _REPO_ROOT / "data" / "claim_embeddings.v2.faiss",
)

# FAISS HNSW knobs — match v1 / build_embeddings_v2.py
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

_model = None
_raw_tokenizer = None
_raw_model = None
_raw_device: Optional[str] = None
_claim_ids: Optional[np.ndarray] = None
_embeddings: Optional[np.ndarray] = None
_faiss_index = None
_id_to_idx: Optional[dict[str, int]] = None
_MODEL_LOCK = threading.Lock()

# Pooling mode for query encoding. Must match the recipe used to encode
# the persisted document vectors (``data/claim_embeddings.v2.npz``).
#
# - ``"cls"`` (default): mxbai-native recipe used by sentence-transformers.
# - ``"mean"`` : matches the May-5 cluster encode (raw transformers,
#   mean-pool over attention mask, L2 norm).
#
# Override with ``CHEMTREE_V2_QUERY_POOLING=mean`` when the deployed
# corpus was encoded mean-pool.
QUERY_POOLING = os.environ.get(
    "CHEMTREE_V2_QUERY_POOLING", "cls"
).strip().lower()
if QUERY_POOLING not in {"cls", "mean"}:
    QUERY_POOLING = "cls"

# Matryoshka truncation dim. mxbai is trained as a Matryoshka encoder
# (a single 1024-d model that produces meaningful representations at
# 256 / 384 / 512 / 768 dims via simple prefix-slicing + renormalisation).
# Set ``CHEMTREE_V2_DIM=256`` to load the 256-d FAISS built by
# ``scripts/build_v2_truncated_flatip.py`` — 4x smaller storage so the
# index stays resident on a 16 GB Mac. Defaults to 0 = full 1024 d.
try:
    QUERY_DIM = int(os.environ.get("CHEMTREE_V2_DIM", "0"))
except ValueError:
    QUERY_DIM = 0
if QUERY_DIM and QUERY_DIM != EMBED_DIM:
    # Override file paths to the truncated artefacts unless the caller
    # has explicitly overridden them already.
    if "CHEMTREE_EMBEDDINGS_V2_NPZ" not in os.environ:
        EMBEDDINGS_PATH = _REPO_ROOT / "data" / f"claim_embeddings.v2_{QUERY_DIM}.npy"
    if "CHEMTREE_EMBEDDINGS_V2_FAISS" not in os.environ:
        FAISS_INDEX_PATH = _REPO_ROOT / "data" / f"claim_embeddings.v2_{QUERY_DIM}.faiss"


def _get_device() -> str:
    """Pick the best available device. Honors CHEMTREE_EMB_DEVICE."""
    forced = os.environ.get("CHEMTREE_EMB_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model():
    """Lazy-load the encoder. Lock-guarded for multi-threaded callers."""
    global _model
    if _model is not None:
        return _model
    with _MODEL_LOCK:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            device = _get_device()
            _model = SentenceTransformer(MODEL_NAME, device=device)
            print(f"[embeddings_v2] Loaded {MODEL_NAME} on {device}",
                  file=sys.stderr)
    return _model


def _get_raw_model():
    """Lazy-load raw HF transformers model + tokenizer for mean-pool path.

    Used when ``QUERY_POOLING == "mean"`` so query encoding matches the
    cluster's mean-pool recipe used to produce the deployed document
    vectors.
    """
    global _raw_tokenizer, _raw_model, _raw_device
    if _raw_model is not None:
        return _raw_tokenizer, _raw_model, _raw_device
    with _MODEL_LOCK:
        if _raw_model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            device = _get_device()
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            mdl = AutoModel.from_pretrained(MODEL_NAME,
                                            torch_dtype=torch.float32).eval()
            mdl.to(device)
            _raw_tokenizer = tok
            _raw_model = mdl
            _raw_device = device
            print(
                f"[embeddings_v2] Loaded raw {MODEL_NAME} on {device} "
                f"(mean-pool recipe)",
                file=sys.stderr,
            )
    return _raw_tokenizer, _raw_model, _raw_device


def is_loaded() -> bool:
    """True iff the v2 corpus is ready for search."""
    return _claim_ids is not None and (
        _embeddings is not None or _faiss_index is not None
    )


def load_embeddings() -> None:
    """Load v2 claim ids + FAISS index into module state.

    Memory plan:
    - On low-RAM hosts (default) we never even materialise the bulky
      ``embeddings`` array — FAISS reads vectors from its own file
      (mmap by default), and ``_gather_vectors`` reconstructs on demand
      for the rare ``semantic_rerank`` path. Loading ``data["embeddings"]``
      out of the 10 GB ``.npz`` archive briefly allocates a 9.5 GB
      transient array and on a 16 GB Mac that pushes us into swap before
      the array can be released.
    - We prefer the sidecar ``claim_embeddings.v2.claim_ids.npy`` written
      by ``scripts/build_v2_flatip.py`` — it is 600 MB and loads with
      mmap. Falls back to a one-time extraction from the npz.
    - Set ``CHEMTREE_KEEP_EMBEDDINGS=1`` to force the legacy behaviour
      (resident numpy embeddings — needed when no FAISS file exists or
      for offline analysis).
    """
    global _claim_ids, _embeddings, _faiss_index
    if _claim_ids is not None:
        return

    keep_embeddings = (
        os.environ.get("CHEMTREE_KEEP_EMBEDDINGS", "0") == "1"
    )

    # δ3 (May 11): on the prod VPS we deploy *only* the FAISS index +
    # claim-id sidecar (~3 GB) and skip the 2-10 GB matrix to keep the
    # 8 GB droplet honest. ``CHEMTREE_KEEP_EMBEDDINGS=0`` (the default)
    # already drops the matrix after FAISS is built; here we also
    # accept the matrix file being missing, as long as both the FAISS
    # index and the sidecar are present. Set
    # ``CHEMTREE_KEEP_EMBEDDINGS=1`` to require the matrix (only
    # needed for offline reranking / rebuilding the FAISS).
    ids_sidecar = EMBEDDINGS_PATH.with_suffix(".claim_ids.npy")
    has_matrix = EMBEDDINGS_PATH.exists()
    has_faiss = FAISS_INDEX_PATH.exists()
    has_sidecar = ids_sidecar.exists()

    if not has_matrix and not (has_faiss and has_sidecar):
        print(
            f"[embeddings_v2] WARNING: no v2 embeddings at "
            f"{EMBEDDINGS_PATH} and the FAISS+sidecar fallback "
            f"({FAISS_INDEX_PATH.name} + {ids_sidecar.name}) is "
            f"incomplete (faiss={has_faiss}, sidecar={has_sidecar}). "
            f"Run:\n"
            f"  python3 scripts/build_embeddings_v2.py merge\n"
            f"  python3 scripts/build_embeddings_v2.py faiss",
            file=sys.stderr,
        )
        return

    if keep_embeddings and not has_matrix:
        print(
            f"[embeddings_v2] ERROR: CHEMTREE_KEEP_EMBEDDINGS=1 but "
            f"matrix file {EMBEDDINGS_PATH} is missing.",
            file=sys.stderr,
        )
        return

    t0 = time.time()
    is_npz = EMBEDDINGS_PATH.suffix == ".npz"
    if has_sidecar and not keep_embeddings:
        _claim_ids = np.load(str(ids_sidecar), mmap_mode="r")
        print(
            f"[embeddings_v2] Loaded {len(_claim_ids):,} v2 claim ids "
            f"from sidecar in {time.time()-t0:.1f}s (mmap)",
            file=sys.stderr,
        )
    elif has_matrix and is_npz:
        data = np.load(str(EMBEDDINGS_PATH))
        _claim_ids = data["claim_ids"]
        if keep_embeddings:
            _embeddings = data["embeddings"]
        print(
            f"[embeddings_v2] Loaded {len(_claim_ids):,} v2 embeddings "
            f"in {time.time()-t0:.1f}s",
            file=sys.stderr,
        )
    else:
        # Truncated-matrix path with a missing sidecar is unrecoverable
        # because a plain .npy has no claim_ids column.
        print(
            f"[embeddings_v2] ERROR: matrix is .npy but sidecar "
            f"{ids_sidecar} is missing.  Rebuild with "
            "scripts/build_v2_truncated_flatip.py.",
            file=sys.stderr,
        )
        return

    _get_id_to_idx()
    _load_or_build_faiss_index()

    if _embeddings is not None and not keep_embeddings:
        _embeddings = None
        gc.collect()
        print(
            "[embeddings_v2] Released numpy embeddings; vectors served "
            "from FAISS.",
            file=sys.stderr,
        )


def reload_embeddings() -> None:
    global _claim_ids, _embeddings, _faiss_index, _id_to_idx
    _claim_ids = None
    _embeddings = None
    _faiss_index = None
    _id_to_idx = None
    load_embeddings()


def _load_or_build_faiss_index() -> None:
    """Load the FAISS HNSW index, or fall back to brute force."""
    global _faiss_index
    if not FAISS_INDEX_PATH.exists():
        print(
            f"[embeddings_v2] WARNING: no v2 FAISS index at "
            f"{FAISS_INDEX_PATH}.  Build it with:\n"
            f"  python3 scripts/build_embeddings_v2.py faiss\n"
            f"  Falling back to brute-force search until then.",
            file=sys.stderr,
        )
        return
    import faiss
    t0 = time.time()
    # Read with IO_FLAG_MMAP when explicitly requested — keeps the 9.6 GB
    # IndexFlatIP file paged from disk instead of resident in RAM, which
    # matters on the 16 GB dev box (otherwise FAISS + the 10 GB npz +
    # mxbai weights + OS exhaust RAM and Mach starts swapping).
    use_mmap = os.environ.get(
        "CHEMTREE_FAISS_MMAP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if use_mmap:
        _faiss_index = faiss.read_index(
            str(FAISS_INDEX_PATH), faiss.IO_FLAG_MMAP
        )
    else:
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    # ``efSearch`` only exists on graph-based indices (HNSW). Flat indices
    # (IndexFlatIP) are exact and have no equivalent knob.
    hnsw = getattr(_faiss_index, "hnsw", None)
    if hnsw is not None and hasattr(hnsw, "efSearch"):
        hnsw.efSearch = HNSW_EF_SEARCH
        kind = "HNSW"
    else:
        kind = type(_faiss_index).__name__
    # FAISS keeps its own OpenMP pool independent of OMP_NUM_THREADS that
    # the caller set for sentence-transformers init (we keep ST at 1 on
    # Apple-MPS to avoid the OMP collision that segfaults model load).
    # IndexFlatIP on the 2.34M x 1024 corpus is memory-bandwidth-bound,
    # not compute-bound: at 9.6 GB of fp32 vectors the matrix is ~400x
    # the M-series L3 cache. Isolated benchmarks on this Mac show
    # OMP=1, 2, 4, 8 all sit at ~570-600 ms / query (all DRAM-limited);
    # OMP=9 with mmap-in-server actually thrashed at 19 s/query due to
    # false-sharing under memory pressure. Default to 1 for predictable
    # per-thread footprint; honour CHEMTREE_FAISS_THREADS for tuning.
    try:
        ft = int(os.environ.get("CHEMTREE_FAISS_THREADS", "0"))
    except ValueError:
        ft = 0
    if ft <= 0:
        ft = 1
    try:
        faiss.omp_set_num_threads(ft)
        print(
            f"[embeddings_v2] FAISS OMP threads = {ft}  "
            f"mmap={use_mmap}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[embeddings_v2] faiss.omp_set_num_threads failed "
            f"({exc!r}); falling back to OMP_NUM_THREADS="
            f"{os.environ.get('OMP_NUM_THREADS','?')}",
            file=sys.stderr,
        )
    print(
        f"[embeddings_v2] Loaded v2 FAISS {kind} "
        f"({_faiss_index.ntotal:,} vectors) in {time.time()-t0:.1f}s",
        file=sys.stderr,
    )


def _get_id_to_idx() -> dict[str, int]:
    global _id_to_idx
    if _id_to_idx is None and _claim_ids is not None:
        _id_to_idx = {cid: i for i, cid in enumerate(_claim_ids)}
    return _id_to_idx or {}


def _gather_vectors(indices) -> np.ndarray:
    if _embeddings is not None:
        return _embeddings[indices]
    if _faiss_index is None:
        raise RuntimeError(
            "[embeddings_v2] neither numpy nor FAISS available."
        )
    out = np.empty((len(indices), EMBED_DIM), dtype=np.float32)
    for i, idx in enumerate(indices):
        _faiss_index.reconstruct(int(idx), out[i])
    return out


# ---- query embedding -----------------------------------------------------
_query_cache: dict[str, np.ndarray] = {}
_QUERY_CACHE_MAX = 512


def embed_query(query: str) -> np.ndarray:
    """Embed a search query.  Returns a 1-D float32 vector.  Cached.

    Honours :data:`QUERY_POOLING` (CLS or mean) so that query vectors
    sit in the same subspace as the deployed document vectors, and
    :data:`QUERY_DIM` for Matryoshka truncation — when ``CHEMTREE_V2_DIM``
    is set the encoder still produces 1024-d output (the model isn't
    re-loaded at a smaller size) and we slice + L2-renormalise to the
    target dim so the runtime query matches the truncated FAISS rows.
    """
    if query in _query_cache:
        return _query_cache[query]
    text = QUERY_PREFIX + query if QUERY_PREFIX else query

    if QUERY_POOLING == "mean":
        import torch
        tok, mdl, device = _get_raw_model()
        enc = tok([text], padding=True, truncation=True,
                  max_length=384, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = ((out * mask).sum(dim=1)
                      / mask.sum(dim=1).clamp(min=1))
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vec = pooled.float().cpu().numpy().astype(np.float32)[0]
    else:
        model = _get_model()
        vec = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)[0]

    if QUERY_DIM and QUERY_DIM != vec.shape[0]:
        vec = vec[: QUERY_DIM]
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = (vec / n).astype(np.float32)

    if len(_query_cache) >= _QUERY_CACHE_MAX:
        oldest = next(iter(_query_cache))
        del _query_cache[oldest]
    _query_cache[query] = vec
    return vec


def vector_search(query: str, top_k: int = 200,
                  min_score: float = 0.20) -> list[tuple[str, float]]:
    """Global ANN search over the v2 claim corpus.

    Returns a list of ``(claim_id, cosine_similarity)`` sorted by
    descending score, truncated at the first hit below ``min_score``.
    Falls back to brute-force numpy when the FAISS index is missing.
    """
    load_embeddings()
    if _claim_ids is None or (_embeddings is None and _faiss_index is None):
        return []

    query_vec = embed_query(query)

    if _faiss_index is not None:
        qv = query_vec.reshape(1, -1).astype(np.float32)
        scores_arr, indices_arr = _faiss_index.search(qv, top_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores_arr[0], indices_arr[0]):
            if idx < 0 or score < min_score:
                break
            results.append((_claim_ids[idx], float(score)))
        return results

    # Brute-force fallback (requires _embeddings still resident).
    scores = _embeddings @ query_vec  # type: ignore[operator]
    top_indices = np.argsort(-scores)[:top_k]
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            break
        results.append((_claim_ids[idx], score))
    return results


def semantic_rerank(query: str, candidate_claim_ids: list[str],
                    top_k: int = 50,
                    min_score: float = 0.25) -> list[tuple[str, float]]:
    """Re-score the given candidate ids by v2 dense similarity."""
    load_embeddings()
    if _claim_ids is None or (_embeddings is None and _faiss_index is None):
        return [(cid, 1.0) for cid in candidate_claim_ids[:top_k]]

    query_vec = embed_query(query)
    id_to_idx = _get_id_to_idx()

    valid_indices: list[int] = []
    valid_cids: list[str] = []
    for cid in candidate_claim_ids:
        idx = id_to_idx.get(cid)
        if idx is not None:
            valid_indices.append(idx)
            valid_cids.append(cid)

    if not valid_indices:
        return [(cid, 0.0) for cid in candidate_claim_ids[:top_k]]

    candidate_vecs = _gather_vectors(valid_indices)
    scores = candidate_vecs @ query_vec
    scored = list(zip(valid_cids, scores.tolist()))
    scored.sort(key=lambda t: -t[1])
    out = [(cid, float(s)) for cid, s in scored if float(s) >= min_score]
    return out[:top_k]
