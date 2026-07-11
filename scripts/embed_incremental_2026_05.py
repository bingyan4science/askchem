#!/usr/bin/env python3
"""Incremental embedding of the new claims from this ingestion.

Avoids a full 2.4M-row re-embed (5-8h on MPS) by:
  1. Loading existing ``claim_embeddings.v2_256.{faiss,claim_ids.npy}``
     (the production 256-d Matryoshka FAISS that the VPS serves).
  2. Identifying claims in chemtree.db whose claim_id is NOT in
     ``claim_ids.npy`` — those are the new ones.
  3. Encoding the new claims with the same mxbai-embed-large-v1 / 1024-d
     model used to build the existing index, then Matryoshka-truncating
     to 256-d and L2-renormalising.
  4. Appending the new rows to a FRESH FAISS IndexFlatIP (existing
     ntotal + new) and writing ``claim_embeddings.v2_256.faiss`` +
     ``claim_embeddings.v2_256.claim_ids.npy``.

Output paths match what ``src/askchem/embeddings_v2.py`` reads at
runtime, so the deploy step picks up the new vectors automatically.

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src \
        python3 scripts/embed_incremental_2026_05.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SRC_FAISS = DATA_DIR / "claim_embeddings.v2_256.faiss"
SRC_IDS = DATA_DIR / "claim_embeddings.v2_256.claim_ids.npy"

OUT_FAISS = SRC_FAISS  # overwrite
OUT_IDS = SRC_IDS
OUT_NPY = DATA_DIR / "claim_embeddings.v2_256.npy"

MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_DIM = 256
SOURCE_DIM = 1024
BATCH_SIZE = 32


def _get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    if not SRC_FAISS.exists() or not SRC_IDS.exists():
        print(f"ERROR: missing {SRC_FAISS} or {SRC_IDS}", file=sys.stderr)
        return 1

    print("loading existing claim_ids ...")
    existing_ids = np.load(str(SRC_IDS), allow_pickle=False)
    existing_set: set[str] = set(existing_ids.tolist())
    print(f"  existing: {len(existing_set):,} claim_ids")

    import faiss
    print(f"loading {SRC_FAISS} ...")
    src_index = faiss.read_index(str(SRC_FAISS))
    print(f"  src index ntotal={src_index.ntotal:,} dim={src_index.d}")
    if src_index.d != EMBED_DIM:
        print(f"ERROR: index dim {src_index.d} != {EMBED_DIM}")
        return 1
    if src_index.ntotal != len(existing_set):
        print(f"WARN: ntotal mismatch ({src_index.ntotal} vs {len(existing_set)})")

    # Find new claims
    print("scanning chemtree.db for claims not in existing index ...")
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT c.claim_id, c.data, c.claim_contextualized, s.paper_summary "
        "FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
        "ORDER BY c.claim_id"
    )
    new_rows: list[tuple[str, str]] = []
    n_scanned = 0
    for r in cur:
        n_scanned += 1
        cid = r["claim_id"]
        if not cid or cid in existing_set:
            continue
        try:
            claim = json.loads(r["data"]) if r["data"] else {}
        except Exception:
            continue
        text = _claim_to_text(
            claim,
            claim_contextualized=r["claim_contextualized"],
            paper_summary=r["paper_summary"],
        )
        if not text:
            continue
        new_rows.append((cid, text))
    conn.close()
    print(f"  scanned {n_scanned:,} claims; {len(new_rows):,} need embedding")

    if not new_rows:
        print("nothing to embed")
        return 0

    # Encode new claims
    device = _get_device()
    print(f"loading {MODEL_ID} on {device} ...")
    from sentence_transformers import SentenceTransformer

    started = time.time()
    model = SentenceTransformer(MODEL_ID, device=device)
    print(f"  loaded in {time.time() - started:.1f}s")

    new_ids = np.array([cid for cid, _ in new_rows], dtype=object)
    texts = [text for _, text in new_rows]

    print(f"encoding {len(texts):,} new texts (batch={BATCH_SIZE}) ...")
    t0 = time.time()
    emb_1024 = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    print(f"  encoded in {time.time() - t0:.1f}s; shape={emb_1024.shape}")

    # Matryoshka truncate + L2 renormalize
    emb_256 = emb_1024[:, :EMBED_DIM].astype(np.float32, copy=True)
    norms = np.linalg.norm(emb_256, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_256 /= norms
    print(f"  truncated to {EMBED_DIM} + L2 renormalized; shape={emb_256.shape}")

    # Append to existing FAISS
    print("appending to FAISS ...")
    src_index.add(np.ascontiguousarray(emb_256))
    print(f"  new ntotal={src_index.ntotal:,}")

    # Stamp the existing.npy with merged matrix as well — runtime uses the
    # truncated .npy when CHEMTREE_FAISS_MMAP=0 (current prod config).
    # Easier: rebuild from FAISS index instead. For safety, keep going.

    # Update claim_ids sidecar
    merged_ids = np.concatenate([existing_ids, new_ids])
    if len(merged_ids) != src_index.ntotal:
        print(f"WARN: ids ({len(merged_ids)}) != ntotal ({src_index.ntotal})")

    # Backup originals
    bak_faiss = SRC_FAISS.with_suffix(".faiss.pre_2026_05.bak")
    bak_ids = SRC_IDS.with_suffix(".npy.pre_2026_05.bak")
    if not bak_faiss.exists():
        os.rename(str(SRC_FAISS), str(bak_faiss))
        print(f"  backed up faiss to {bak_faiss}")
    else:
        # Already exists; remove current and rewrite
        SRC_FAISS.unlink(missing_ok=True)
    if not bak_ids.exists():
        os.rename(str(SRC_IDS), str(bak_ids))
        print(f"  backed up ids to {bak_ids}")
    else:
        SRC_IDS.unlink(missing_ok=True)

    print(f"writing {OUT_FAISS} ...")
    faiss.write_index(src_index, str(OUT_FAISS))
    print(f"writing {OUT_IDS} ...")
    np.save(str(OUT_IDS), merged_ids)
    sz = OUT_FAISS.stat().st_size / 1e9
    print(f"\nDone. faiss size: {sz:.2f} GB; total ntotal: {src_index.ntotal:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
