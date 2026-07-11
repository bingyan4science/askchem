"""Build a FAISS IndexFlatIP from data/claim_embeddings.v2.npz.

Why Flat instead of HNSW
------------------------
- HNSW (M=32, efC=200) on 2.34 M × 1024-d takes hours on a 16 GB Mac
  because the 9.6 GB vector array swaps during graph construction.
- Flat is *exact* (no recall floor) and gives us the highest-fidelity
  ground truth for the live eval. At 2.34 M × 1024 the per-query cost
  is ~50 ms with Apple Accelerate BLAS — fine for AskChem.
- The FAISS file format is interchangeable: ``faiss.read_index`` returns
  an object that exposes the same ``.search(q, k)`` API regardless of
  index type, so ``embeddings_v2._load_or_build_faiss_index`` works
  without changes.

Memory plan
-----------
Naively ``IndexFlatIP.add(vecs)`` peaks at ~19 GB (source array + FAISS
internal storage). To stay within 16 GB:

1. Convert ``.npz['embeddings']`` to a flat ``.npy`` file once
   (writes ~9.6 GB; ~30 s).
2. ``np.load(..., mmap_mode='r')`` so vectors are paged from disk on
   demand.
3. Chunked ``index.add(...)`` (250 K rows at a time, contiguous copy).
   Peak resident memory: FAISS storage (~9.6 GB) + 1 GB chunk staging.
4. ``faiss.write_index`` to disk.

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONPATH=src \
        python3 scripts/build_v2_flatip.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
NPZ_PATH = DATA_DIR / "claim_embeddings.v2.npz"
NPY_PATH = DATA_DIR / "claim_embeddings.v2.embeddings.npy"
IDS_PATH = DATA_DIR / "claim_embeddings.v2.claim_ids.npy"
FAISS_PATH = DATA_DIR / "claim_embeddings.v2.faiss"
CHUNK = 250_000


def materialise_npy() -> None:
    """One-time extract embeddings + claim_ids from npz to standalone npy."""
    if NPY_PATH.exists() and IDS_PATH.exists():
        print(f"[skip] {NPY_PATH.name} and {IDS_PATH.name} already exist")
        return
    print(f"loading {NPZ_PATH.name} ({NPZ_PATH.stat().st_size / 1e9:.1f} GB)…",
          flush=True)
    t0 = time.monotonic()
    with np.load(str(NPZ_PATH)) as z:
        ids = z["claim_ids"]
        vecs = z["embeddings"]
        print(f"  shape={vecs.shape}  dtype={vecs.dtype}  "
              f"({time.monotonic()-t0:.1f}s)", flush=True)
        t1 = time.monotonic()
        np.save(str(IDS_PATH), ids)
        print(f"  wrote {IDS_PATH.name} "
              f"({IDS_PATH.stat().st_size / 1e6:.1f} MB)  "
              f"in {time.monotonic()-t1:.1f}s", flush=True)
        t1 = time.monotonic()
        np.save(str(NPY_PATH), vecs)
        print(f"  wrote {NPY_PATH.name} "
              f"({NPY_PATH.stat().st_size / 1e9:.2f} GB)  "
              f"in {time.monotonic()-t1:.1f}s", flush=True)
    gc.collect()


def build_flat_ip() -> None:
    import faiss

    print("\n--- building IndexFlatIP ---", flush=True)
    vecs = np.load(str(NPY_PATH), mmap_mode="r")
    n, d = vecs.shape
    print(f"  vectors  : {n:,} × {d}  (mmap'd from {NPY_PATH.name})",
          flush=True)
    print(f"  faiss out: {FAISS_PATH}", flush=True)
    print(f"  chunk    : {CHUNK:,}", flush=True)

    index = faiss.IndexFlatIP(d)
    t0 = time.monotonic()
    last_print = 0
    for i in range(0, n, CHUNK):
        chunk = np.ascontiguousarray(vecs[i:i + CHUNK], dtype=np.float32)
        index.add(chunk)
        del chunk
        if (i + CHUNK) - last_print >= CHUNK * 4 or i + CHUNK >= n:
            done = min(i + CHUNK, n)
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n - done) / rate if rate > 0 else 0.0
            print(f"  added {done:>10,}/{n:,}  "
                  f"({rate:>8,.0f} c/s, ETA {eta/60:.1f} min)",
                  flush=True)
            last_print = done

    print(f"\nadd() done in {(time.monotonic()-t0)/60:.1f} min "
          f"(ntotal={index.ntotal:,})", flush=True)

    print("writing index…", flush=True)
    t0 = time.monotonic()
    tmp = FAISS_PATH.with_suffix(".tmp.faiss")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, FAISS_PATH)
    size_gb = FAISS_PATH.stat().st_size / 1e9
    print(f"  wrote {FAISS_PATH} ({size_gb:.2f} GB) "
          f"in {time.monotonic()-t0:.1f}s", flush=True)


def main() -> None:
    print(f"=== build_v2_flatip ===", flush=True)
    print(f"  npz : {NPZ_PATH}", flush=True)
    print(f"  npy : {NPY_PATH}", flush=True)
    print(f"  ids : {IDS_PATH}", flush=True)
    print(f"  out : {FAISS_PATH}\n", flush=True)
    materialise_npy()
    build_flat_ip()
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
