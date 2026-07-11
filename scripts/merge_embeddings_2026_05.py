#!/usr/bin/env python3
"""Merge ``data/embeddings.2026_05.npz`` (from the torch-cluster H200 run)
into the existing ``claim_embeddings.v2_256.{faiss,claim_ids.npy}``.

The cluster output is 1024-dim mxbai-embed-large-v1 CLS pooled, L2
normalised, claim_ids aligned. We Matryoshka-truncate to 256 dim,
L2-renormalise, append to the existing IndexFlatIP, and update the
claim_ids sidecar.

Usage::

    python3 scripts/merge_embeddings_2026_05.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SRC_FAISS = DATA_DIR / "claim_embeddings.v2_256.faiss"
SRC_IDS = DATA_DIR / "claim_embeddings.v2_256.claim_ids.npy"
NEW_NPZ = DATA_DIR / "embeddings.2026_05.npz"

EMBED_DIM = 256


def main() -> int:
    for p in (SRC_FAISS, SRC_IDS, NEW_NPZ):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1

    import faiss

    print(f"loading {NEW_NPZ}")
    src = np.load(str(NEW_NPZ), allow_pickle=False)
    new_ids = src["claim_ids"]
    new_emb_full = src["embeddings"]
    print(f"  new claim_ids: {len(new_ids):,}")
    print(f"  new embeddings shape: {new_emb_full.shape}, dtype={new_emb_full.dtype}")

    if new_emb_full.shape[1] < EMBED_DIM:
        print(f"new dim {new_emb_full.shape[1]} < {EMBED_DIM}", file=sys.stderr)
        return 1

    new_emb = np.ascontiguousarray(
        new_emb_full[:, :EMBED_DIM].astype(np.float32, copy=True)
    )
    norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_emb /= norms
    print(f"  truncated to {EMBED_DIM} + L2 renormalised: {new_emb.shape}")

    print(f"loading existing ids from {SRC_IDS}")
    existing_ids = np.load(str(SRC_IDS), allow_pickle=False)
    existing_set = set(existing_ids.tolist())
    print(f"  existing claim_ids: {len(existing_ids):,}")

    print(f"loading existing FAISS {SRC_FAISS}")
    index = faiss.read_index(str(SRC_FAISS))
    print(f"  ntotal={index.ntotal:,} dim={index.d}")
    if index.d != EMBED_DIM:
        print(f"index dim {index.d} != {EMBED_DIM}")
        return 1

    # Filter out any new ids that are already in the existing set
    mask = np.array([cid not in existing_set for cid in new_ids], dtype=bool)
    n_dup = (~mask).sum()
    if n_dup:
        print(f"  skipping {n_dup} duplicate claim_ids")
    new_ids = new_ids[mask]
    new_emb = new_emb[mask]
    print(f"  appending {len(new_ids):,} new rows")

    index.add(new_emb)
    print(f"  new ntotal: {index.ntotal:,}")

    merged_ids = np.concatenate([existing_ids, new_ids])
    if len(merged_ids) != index.ntotal:
        print(f"!! id-count mismatch: {len(merged_ids)} vs {index.ntotal}")

    # Back up originals (idempotent)
    bak_faiss = SRC_FAISS.with_name(SRC_FAISS.name + ".pre_2026_05.bak")
    bak_ids = SRC_IDS.with_name(SRC_IDS.name + ".pre_2026_05.bak")
    if not bak_faiss.exists():
        os.replace(str(SRC_FAISS), str(bak_faiss))
        print(f"  backed up {bak_faiss.name}")
    else:
        SRC_FAISS.unlink(missing_ok=True)
    if not bak_ids.exists():
        os.replace(str(SRC_IDS), str(bak_ids))
        print(f"  backed up {bak_ids.name}")
    else:
        SRC_IDS.unlink(missing_ok=True)

    t0 = time.time()
    faiss.write_index(index, str(SRC_FAISS))
    np.save(str(SRC_IDS), merged_ids)
    print(f"wrote {SRC_FAISS} ({SRC_FAISS.stat().st_size / 1e9:.2f} GB) "
          f"+ {SRC_IDS} ({SRC_IDS.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
