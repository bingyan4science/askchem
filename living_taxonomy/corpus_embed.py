"""Reuse the precomputed corpus claim embeddings for placement at scale.

The corpus already has an mxbai v2 vector for every claim
(``data/claim_embeddings.v2.embeddings.npy`` + ``.claim_ids.npy`` sidecar, ~2.44M
rows). Re-encoding every leaf with the local SentenceTransformer costs ~1.5M
encodes at full scale; instead we memmap the matrix and pull the rows we need by
claim_id. Vectors live in the same mxbai space as the host-descriptor embeddings,
so cosine host-shortlisting stays consistent.

Falls back gracefully: if the artifacts are missing, `available()` is False and
callers should encode with placement._embed instead. Missing individual claim_ids
(not in the corpus) are simply absent from the returned dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"
_IDS_PATH = _DATA / "claim_embeddings.v2.claim_ids.npy"
_MAT_PATH = _DATA / "claim_embeddings.v2.embeddings.npy"
_NPZ_PATH = _DATA / "claim_embeddings.v2.npz"


def _aligned_ids(mat_rows):
    """Return an id array aligned row-for-row with the matrix, or None.

    The standalone ``.claim_ids.npy`` sidecar tracks the *current* FAISS corpus
    and can be a newer generation than the standalone ``.embeddings.npy`` matrix
    (they drift as the corpus grows). The self-consistent pairing lives in the
    ``.npz`` (its ``claim_ids`` + ``embeddings`` were saved together and the
    embeddings match ``.embeddings.npy`` row-for-row), so prefer the npz's ids
    when the standalone sidecar length disagrees with the matrix.
    """
    ids = np.load(str(_IDS_PATH), mmap_mode="r") if _IDS_PATH.exists() else None
    if ids is not None and len(ids) == mat_rows:
        return ids
    if _NPZ_PATH.exists():
        try:
            with np.load(str(_NPZ_PATH), mmap_mode="r") as z:
                cand = z["claim_ids"]
                if len(cand) == mat_rows:
                    return np.array(cand)  # materialize (~150 MB of <U64)
        except Exception:
            pass
    return ids

_MAT = None
_ID2ROW = None
_LOADED = None


def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _load() -> bool:
    global _MAT, _ID2ROW, _LOADED
    if _LOADED is not None:
        return _LOADED
    if not _MAT_PATH.exists():
        _LOADED = False
        return False
    _MAT = np.load(str(_MAT_PATH), mmap_mode="r")   # (N, dim) memmap, not resident
    ids = _aligned_ids(_MAT.shape[0])
    # If no id list aligns with the matrix rows, the row mapping is invalid ->
    # disable reuse and let callers fall back to mxbai encoding.
    if ids is None or len(ids) != _MAT.shape[0]:
        import sys
        sys.stderr.write(f"[corpus_embed] no id list aligns with matrix rows "
                         f"({_MAT.shape[0]}); disabling reuse, will mxbai-encode.\n")
        _MAT = None
        _LOADED = False
        return False
    _ID2ROW = {_dec(cid): i for i, cid in enumerate(ids)}
    import sys
    sys.stderr.write(f"[corpus_embed] reuse ON: {len(ids)} claim vectors "
                     f"({_MAT.shape[0]} rows)\n")
    _LOADED = True
    return True


def available() -> bool:
    return _load()


def vectors_for(claim_ids) -> dict:
    """Return {claim_id: L2-normalized float32 vector} for claim_ids present in
    the corpus. Empty dict if artifacts are unavailable."""
    if not _load():
        return {}
    out = {}
    for cid in claim_ids:
        r = _ID2ROW.get(cid)
        if r is None:
            continue
        v = np.asarray(_MAT[r], dtype=np.float32)
        n = float(np.linalg.norm(v))
        out[cid] = (v / n) if n > 0 else v
    return out
