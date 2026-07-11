"""Build a truncated-dim (Matryoshka) IndexFlatIP from data/claim_embeddings.v2.npz.

Motivation (δ1 — local-dev latency fix, May 11)
================================================
The full 1024-dim ``claim_embeddings.v2.faiss`` is 9.6 GB; on a 16 GB
Mac it shares RAM with the mxbai weights (1.3 GB) plus FastAPI / SQLite
/ Python so the kernel page-evicts FAISS between queries and search
latency degrades from 0.6 s (isolated) to 40-45 s (in-server). mxbai is
trained as a Matryoshka encoder, so we can drop to 256 dims and recover
the recall budget at ~25 % of the storage (2.4 GB). The query path
truncates + L2-renormalises at runtime to match.

Usage::

    python3 scripts/build_v2_truncated_flatip.py --dim 256

Writes (alongside the v2 artefacts):
    data/claim_embeddings.v2_<dim>.npy           memory-mapped truncated matrix
    data/claim_embeddings.v2_<dim>.faiss         IndexFlatIP over the truncated rows
    data/claim_embeddings.v2_<dim>.claim_ids.npy sidecar id array (reuses v2's)

Memory plan
-----------
* The source ``claim_embeddings.v2.npz`` is opened with ``mmap_mode='r'``
  so we never load all 10 GB at once.
* The output is written in 100 K-row chunks via ``np.memmap`` so peak RSS
  stays under ~600 MB.
* FAISS is added in 200 K-row chunks against the new memmap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SRC_NPZ = DATA_DIR / "claim_embeddings.v2.npz"
SRC_IDS = DATA_DIR / "claim_embeddings.v2.claim_ids.npy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256,
                    help="Truncation dim (one of 256, 384, 512, 768, 1024).")
    ap.add_argument("--chunk", type=int, default=100_000,
                    help="Rows per write chunk (controls peak RAM).")
    ap.add_argument("--faiss-chunk", type=int, default=200_000,
                    help="Rows per FAISS add() chunk.")
    args = ap.parse_args()

    if args.dim not in (256, 384, 512, 768, 1024):
        print(f"WARN: --dim {args.dim} not in canonical mxbai grid "
              "(256/384/512/768/1024); recall may degrade.",
              file=sys.stderr)

    if not SRC_NPZ.exists():
        print(f"ERROR: source missing {SRC_NPZ}", file=sys.stderr)
        return 1

    out_mat = DATA_DIR / f"claim_embeddings.v2_{args.dim}.npy"
    out_faiss = DATA_DIR / f"claim_embeddings.v2_{args.dim}.faiss"
    out_ids = DATA_DIR / f"claim_embeddings.v2_{args.dim}.claim_ids.npy"

    print(f"Opening {SRC_NPZ} ...")
    src = np.load(str(SRC_NPZ), mmap_mode="r")
    embeddings = src["embeddings"]
    claim_ids = src["claim_ids"]
    n, d = embeddings.shape
    print(f"  source: n={n:,}  dim={d}  dtype={embeddings.dtype}")
    if d < args.dim:
        print(f"ERROR: source dim {d} < --dim {args.dim}", file=sys.stderr)
        return 1

    if not SRC_IDS.exists():
        print(f"Writing claim id sidecar {SRC_IDS} ...")
        np.save(str(SRC_IDS), claim_ids)
    if not out_ids.exists():
        # Truncated index uses the same claim_ids ordering as the source.
        # Symlink if possible; otherwise copy.
        try:
            out_ids.symlink_to(SRC_IDS.name)
        except (OSError, NotImplementedError):
            np.save(str(out_ids), claim_ids)

    print(f"Writing truncated matrix {out_mat} ({n:,} x {args.dim}, fp32) ...")
    out = np.lib.format.open_memmap(
        str(out_mat), mode="w+", dtype=np.float32, shape=(n, args.dim),
    )
    t0 = time.time()
    written = 0
    for i in range(0, n, args.chunk):
        j = min(i + args.chunk, n)
        rows = np.asarray(embeddings[i:j, : args.dim], dtype=np.float32)
        # Re-normalise (truncation breaks L2-norm preservation slightly;
        # mxbai's Matryoshka training keeps it close but the IndexFlatIP
        # cosine semantics rely on unit-norm vectors).
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        rows /= norms
        out[i:j] = rows
        written = j
        if i // args.chunk % 5 == 0:
            elapsed = time.time() - t0
            rate = written / max(elapsed, 1e-6)
            eta = (n - written) / max(rate, 1e-6)
            print(
                f"  [{written:>10,}/{n:>10,}] "
                f"{100*written/n:5.1f} %  "
                f"rate {rate/1e3:5.1f} K/s  eta {eta:5.0f}s"
            )
    out.flush()
    print(f"  matrix written in {time.time()-t0:.1f}s")

    print(f"Building FAISS IndexFlatIP at dim={args.dim} ...")
    import faiss
    index = faiss.IndexFlatIP(args.dim)
    t1 = time.time()
    for i in range(0, n, args.faiss_chunk):
        j = min(i + args.faiss_chunk, n)
        block = np.ascontiguousarray(out[i:j])
        index.add(block)
        if i // args.faiss_chunk % 2 == 0:
            print(f"  added {j:,}/{n:,}")
    print(f"  built FAISS in {time.time()-t1:.1f}s  ntotal={index.ntotal:,}")
    faiss.write_index(index, str(out_faiss))
    size_gb = out_faiss.stat().st_size / 1e9
    print(f"Wrote {out_faiss} ({size_gb:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
