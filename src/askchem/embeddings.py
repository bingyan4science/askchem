"""
Embedding-based semantic search for AskChem.

Pre-computes embeddings for all claims using a local sentence-transformer model,
then provides hybrid search: FTS5 for candidate retrieval + cosine similarity reranking.

Usage:
    # Full rebuild (~10 min for 876K claims on GPU)
    python -m askchem.embeddings build

    # Incremental update (only embeds new claims, fast)
    python -m askchem.embeddings update

    # Test search
    python -m askchem.embeddings search "CO2 electroreduction on copper"
"""

import gc
import json
import os
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
_REPO_ROOT = Path(__file__).parent.parent.parent
# Canonical DB is askchem.db (renamed from chemtree.db); legacy name is a fallback.
DB_PATH = (_REPO_ROOT / "askchem.db") if (_REPO_ROOT / "askchem.db").exists() \
    else (_REPO_ROOT / "chemtree.db")
EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "claim_embeddings.npz"
FAISS_INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "claim_embeddings.faiss"

HNSW_M = 32          # edges per node (higher = better recall, more RAM)
HNSW_EF_CONSTRUCTION = 200  # build-time search depth
HNSW_EF_SEARCH = 128        # query-time search depth (tunable)

_model = None
_claim_ids: Optional[np.ndarray] = None
_embeddings: Optional[np.ndarray] = None
_faiss_index = None
_id_to_idx: Optional[dict[str, int]] = None


def _get_device():
    """Pick the best available device: MPS (Apple GPU), CUDA, or CPU.

    Honors CHEMTREE_EMB_DEVICE for explicit override (set to 'cpu' from
    multi-threaded clients where MPS init isn't thread-safe).
    """
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


_MODEL_LOCK = __import__("threading").Lock()


def _get_model():
    """Lazy-load the encoder.  Wrapped in a lock so multi-threaded callers
    (e.g. the edge backfill runner) do not race to construct it — concurrent
    SentenceTransformer init on macOS MPS triggers a 'Cannot copy out of meta
    tensor' error and corrupts the model.
    """
    global _model
    if _model is not None:
        return _model
    with _MODEL_LOCK:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            device = _get_device()
            _model = SentenceTransformer(MODEL_NAME, device=device)
            print(f"Loaded {MODEL_NAME} on {device}")
    return _model


def _claim_to_text(claim: dict,
                   claim_contextualized: Optional[str] = None,
                   paper_summary: Optional[str] = None) -> str:
    """Build a searchable text representation of a claim for embedding.

    When the claim has a Sprint-1 ``claim_contextualized`` rewrite, that
    LLM-rewritten standalone sentence becomes the **primary** indexed
    text — it's already a one-sentence claim that names the system,
    method, and finding without copying the paper's prose. We append
    paper-level context (``paper_summary``) and the existing typed
    fields as secondary signal so that synonyms / abbreviations the
    rewrite didn't capture still match.

    Falls back cleanly to the legacy typed-field-and-verbatim text when
    no rewrite exists (e.g. abstract-only claims, or rows the
    contextualization batch hasn't reached yet).
    """
    parts: list[str] = []

    if claim_contextualized:
        parts.append(str(claim_contextualized))
    if paper_summary:
        parts.append(str(paper_summary)[:500])

    ct = claim.get('claim_type', '')
    title = claim.get('source_paper_title', '')
    if title:
        parts.append(title)

    if ct in ('reaction', 'scope_entry'):
        for key in ('reaction_type', 'subject'):
            if claim.get(key):
                parts.append(claim[key])
        for role in ('reactants', 'products'):
            items = claim.get(role) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get('name'):
                    parts.append(item['name'])
                elif isinstance(item, str):
                    parts.append(item)
    elif ct == 'property':
        for key in ('subject', 'property_name', 'value', 'unit'):
            if claim.get(key):
                parts.append(str(claim[key]))
    elif ct == 'method':
        for key in ('technique_name', 'what_it_achieves'):
            if claim.get(key):
                parts.append(claim[key])
    elif ct == 'mechanism':
        if claim.get('process_described'):
            parts.append(claim['process_described'])
    elif ct == 'comparison':
        if claim.get('comparison_result'):
            parts.append(claim['comparison_result'])
    elif ct == 'hypothesis':
        if claim.get('hypothesis_text'):
            parts.append(claim['hypothesis_text'])
    elif ct in ('conclusion', 'conclusions'):
        pass  # verbatim_quote covers it
    elif ct == 'limitation':
        if claim.get('limitation_text'):
            parts.append(claim['limitation_text'])
    elif ct == 'future_direction':
        if claim.get('direction_text'):
            parts.append(claim['direction_text'])
    elif ct == 'surprising_finding':
        for key in ('finding_text', 'why_surprising'):
            if claim.get(key):
                parts.append(claim[key])

    quote = claim.get('verbatim_quote', '')
    if quote:
        parts.append(str(quote)[:300])

    return ' '.join(str(p) for p in parts if p)


def _db_path() -> str:
    import os
    return os.environ.get("ASKCHEM_DB") or os.environ.get("CHEMTREE_DB", str(DB_PATH))


def build_embeddings():
    """Pre-compute embeddings for all claims and save to disk."""
    model = _get_model()

    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    # JOIN sources to pull paper_summary; LEFT JOIN so claims whose source
    # row is missing (rare) still get embedded with the legacy text path.
    rows = conn.execute(
        "SELECT c.claim_id, c.data, c.claim_contextualized, s.paper_summary "
        "FROM claims c "
        "LEFT JOIN sources s ON c.source_doi = s.doi "
        "ORDER BY c.claim_id"
    ).fetchall()
    conn.close()

    print(f"Building embeddings for {len(rows):,} claims using {MODEL_NAME}...")

    claim_ids = []
    texts = []
    for row in rows:
        claim_ids.append(row['claim_id'])
        claim = json.loads(row['data'])
        texts.append(_claim_to_text(
            claim,
            claim_contextualized=row['claim_contextualized'],
            paper_summary=row['paper_summary'],
        ))

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"Encoded {len(texts):,} claims in {elapsed:.1f}s ({len(texts)/elapsed:.0f} claims/s)")

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(EMBEDDINGS_PATH),
        claim_ids=np.array(claim_ids, dtype='U64'),
        embeddings=embeddings.astype(np.float32),
    )
    size_mb = EMBEDDINGS_PATH.stat().st_size / 1e6
    print(f"Saved to {EMBEDDINGS_PATH} ({size_mb:.1f} MB)")


def update_embeddings():
    """Incrementally embed only new claims and append to the existing file."""
    existing_ids = set()
    if EMBEDDINGS_PATH.exists():
        data = np.load(str(EMBEDDINGS_PATH))
        old_ids = data['claim_ids']
        old_vecs = data['embeddings']
        existing_ids = set(old_ids.tolist())
        print(f"Existing embeddings: {len(old_ids):,} claims")
    else:
        old_ids = np.array([], dtype='U64')
        old_vecs = np.empty((0, EMBED_DIM), dtype=np.float32)

    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT c.claim_id, c.data, c.claim_contextualized, s.paper_summary "
        "FROM claims c "
        "LEFT JOIN sources s ON c.source_doi = s.doi "
        "ORDER BY c.claim_id"
    ).fetchall()
    conn.close()

    new_rows = [
        (r['claim_id'], r['data'], r['claim_contextualized'], r['paper_summary'])
        for r in rows if r['claim_id'] not in existing_ids
    ]
    if not new_rows:
        print("All claims already have embeddings. Nothing to do.")
        return 0

    print(f"Embedding {len(new_rows):,} new claims (out of {len(rows):,} total)...")
    model = _get_model()

    new_ids = []
    new_texts = []
    for cid, data_str, ctx_text, paper_summary in new_rows:
        new_ids.append(cid)
        claim = json.loads(data_str)
        new_texts.append(_claim_to_text(
            claim,
            claim_contextualized=ctx_text,
            paper_summary=paper_summary,
        ))

    t0 = time.time()
    new_vecs = model.encode(
        new_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    elapsed = time.time() - t0
    print(f"Encoded {len(new_texts):,} new claims in {elapsed:.1f}s "
          f"({len(new_texts)/elapsed:.0f} claims/s)")

    merged_ids = np.concatenate([old_ids, np.array(new_ids, dtype='U64')])
    merged_vecs = np.concatenate([old_vecs, new_vecs], axis=0)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(EMBEDDINGS_PATH),
        claim_ids=merged_ids,
        embeddings=merged_vecs,
    )
    size_mb = EMBEDDINGS_PATH.stat().st_size / 1e6
    print(f"Saved {len(merged_ids):,} embeddings to {EMBEDDINGS_PATH} ({size_mb:.1f} MB)")
    return len(new_rows)


def reload_embeddings():
    """Force-reload embeddings from disk (call after update_embeddings)."""
    global _claim_ids, _embeddings, _faiss_index, _id_to_idx
    _claim_ids = None
    _embeddings = None
    _faiss_index = None
    _id_to_idx = None
    load_embeddings()


def load_embeddings():
    """Load pre-computed embeddings and FAISS index into memory.

    On low-RAM hosts (CHEMTREE_KEEP_EMBEDDINGS=0, the default) the bulky numpy
    array is dropped right after the FAISS HNSW index is loaded, since
    ``IndexHNSWFlat`` already stores the original vectors and supports
    ``reconstruct``.  This frees ~2.7 GB on a 1.77M × 384 corpus.
    """
    global _claim_ids, _embeddings, _faiss_index
    if _claim_ids is not None:
        return

    if not EMBEDDINGS_PATH.exists():
        print(f"WARNING: No embeddings file at {EMBEDDINGS_PATH}. Run: python -m askchem.embeddings build")
        return

    t0 = time.time()
    data = np.load(str(EMBEDDINGS_PATH))
    _claim_ids = data['claim_ids']
    _embeddings = data['embeddings']
    elapsed = time.time() - t0
    print(f"Loaded {len(_claim_ids):,} claim embeddings in {elapsed:.1f}s")

    _get_id_to_idx()  # pre-build the claim_id→index lookup

    _load_or_build_faiss_index()

    if _faiss_index is not None and os.environ.get("CHEMTREE_KEEP_EMBEDDINGS", "0") != "1":
        _embeddings = None
        gc.collect()
        print("Released numpy embeddings array; vectors served from FAISS HNSW.")


def _load_or_build_faiss_index():
    """Load FAISS HNSW index from disk, or build it if missing."""
    global _faiss_index
    import faiss

    if FAISS_INDEX_PATH.exists():
        t0 = time.time()
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        _faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        print(f"Loaded FAISS HNSW index ({_faiss_index.ntotal:,} vectors) in {time.time()-t0:.1f}s")
    else:
        build_faiss_index()


def build_faiss_index():
    """Build FAISS HNSW index from loaded embeddings and save to disk."""
    global _faiss_index
    import faiss

    if _embeddings is None:
        raise RuntimeError("Embeddings not loaded. Call load_embeddings() first.")

    n, d = _embeddings.shape
    print(f"Building FAISS HNSW index for {n:,} vectors (dim={d}, M={HNSW_M})...")
    t0 = time.time()

    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    batch_size = 100_000
    for i in range(0, n, batch_size):
        batch = _embeddings[i : i + batch_size]
        index.add(batch)
        if i + batch_size < n:
            print(f"  Added {min(i + batch_size, n):,}/{n:,} vectors...")

    elapsed = time.time() - t0
    print(f"Built HNSW index in {elapsed:.1f}s")

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    size_mb = FAISS_INDEX_PATH.stat().st_size / 1e6
    print(f"Saved FAISS index to {FAISS_INDEX_PATH} ({size_mb:.0f} MB)")

    _faiss_index = index


def is_loaded() -> bool:
    """Check if embeddings are loaded and ready for search."""
    return _claim_ids is not None and (
        _embeddings is not None or _faiss_index is not None
    )


def _gather_vectors(indices) -> np.ndarray:
    """Return the embedding vectors for the given row indices.

    Prefers the in-memory numpy array when it's still resident; otherwise
    reconstructs from the FAISS HNSW storage.  Both code paths return a
    contiguous ``float32`` array shaped ``(len(indices), EMBED_DIM)``.
    """
    if _embeddings is not None:
        return _embeddings[indices]
    if _faiss_index is None:
        raise RuntimeError("Neither numpy embeddings nor FAISS index is loaded.")
    out = np.empty((len(indices), EMBED_DIM), dtype=np.float32)
    for i, idx in enumerate(indices):
        _faiss_index.reconstruct(int(idx), out[i])
    return out


_query_cache: dict[str, np.ndarray] = {}
_QUERY_CACHE_MAX = 512


def embed_query(query: str) -> np.ndarray:
    """Embed a search query. Returns a 1-D float32 vector. Cached in memory."""
    if query in _query_cache:
        return _query_cache[query]
    model = _get_model()
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    if len(_query_cache) >= _QUERY_CACHE_MAX:
        oldest = next(iter(_query_cache))
        del _query_cache[oldest]
    _query_cache[query] = vec
    return vec


def vector_search(query: str, top_k: int = 200,
                  min_score: float = 0.20) -> list[tuple[str, float]]:
    """
    Global vector search over all claim embeddings using FAISS HNSW.

    Returns list of (claim_id, cosine_similarity) sorted by descending score.
    Falls back to brute-force numpy if the FAISS index isn't available.
    """
    load_embeddings()
    if _claim_ids is None or (_embeddings is None and _faiss_index is None):
        return []

    query_vec = embed_query(query)

    if _faiss_index is not None:
        qv = query_vec.reshape(1, -1).astype(np.float32)
        scores_arr, indices_arr = _faiss_index.search(qv, top_k)
        results = []
        for score, idx in zip(scores_arr[0], indices_arr[0]):
            if idx < 0 or score < min_score:
                break
            results.append((_claim_ids[idx], float(score)))
        return results

    # Brute-force fallback (only reachable when the numpy array is kept).
    scores = _embeddings @ query_vec
    top_indices = np.argsort(-scores)[:top_k]
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            break
        results.append((_claim_ids[idx], score))
    return results


def _get_id_to_idx() -> dict[str, int]:
    """Return a cached claim_id→index mapping (built once after load)."""
    global _id_to_idx
    if _id_to_idx is None and _claim_ids is not None:
        _id_to_idx = {cid: i for i, cid in enumerate(_claim_ids)}
    return _id_to_idx or {}


def semantic_rerank(query: str, candidate_claim_ids: list[str],
                    top_k: int = 50, min_score: float = 0.25) -> list[tuple[str, float]]:
    """
    Rerank candidate claims by semantic similarity to query.

    Returns list of (claim_id, score) sorted by descending similarity,
    filtered to score >= min_score.
    """
    load_embeddings()
    if _claim_ids is None or (_embeddings is None and _faiss_index is None):
        return [(cid, 1.0) for cid in candidate_claim_ids[:top_k]]

    query_vec = embed_query(query)

    id_to_idx = _get_id_to_idx()
    valid_indices = []
    valid_cids = []
    unembedded_cids = []
    for cid in candidate_claim_ids:
        idx = id_to_idx.get(cid)
        if idx is not None:
            valid_indices.append(idx)
            valid_cids.append(cid)
        else:
            unembedded_cids.append(cid)

    if not valid_indices:
        return [(cid, 0.0) for cid in candidate_claim_ids[:top_k]]

    candidate_vecs = _gather_vectors(valid_indices)
    scores = candidate_vecs @ query_vec

    scored = list(zip(valid_cids, scores.tolist()))
    scored.sort(key=lambda x: -x[1])

    results = [(cid, score) for cid, score in scored if score >= min_score]
    for cid in unembedded_cids:
        results.append((cid, 0.0))
    return results[:top_k]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("build", help="Full rebuild of all embeddings")
    sub.add_parser("update", help="Incrementally embed only new claims")
    sub.add_parser("build-index", help="Rebuild FAISS HNSW index from existing embeddings")
    search_p = sub.add_parser("search")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "build":
        build_embeddings()
    elif args.command == "update":
        update_embeddings()
    elif args.command == "build-index":
        load_embeddings()
        build_faiss_index()
    elif args.command == "search":
        load_embeddings()
        results = vector_search(args.query, top_k=args.top_k)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        for cid, score in results:
            row = conn.execute("SELECT data FROM claims WHERE claim_id=?", [cid]).fetchone()
            if row:
                claim = json.loads(row['data'])
                title = claim.get('source_paper_title', '')[:60]
                ct = claim.get('claim_type', '')
                quote = (claim.get('verbatim_quote', '') or '')[:80]
                print(f"  {score:.3f}  [{ct}] {title}")
                if quote:
                    print(f"         \"{quote}\"")
        conn.close()
    else:
        parser.print_help()
