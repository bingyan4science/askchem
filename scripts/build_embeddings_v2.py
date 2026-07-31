"""Phase γ2: full re-embed of the claim corpus with the bake-off winner.

Encoder: ``mixedbread-ai/mxbai-embed-large-v1`` (1024 dim, MTEB-leading,
Matryoshka-truncatable), selected by the encoder bake-off.

This script is intentionally **standalone** — it does not modify the
existing ``data/claim_embeddings.npz``. Output paths are ``*.v2.npz``
and ``*.v2.faiss`` so that the existing ``embeddings.py`` (still on
MiniLM) keeps serving traffic until we explicitly flip a switch.

Design notes
------------
* **Resumable**: the encoder writes one ``.npy`` shard per chunk so a
  crash or kill only loses at most ``--chunk-size`` rows of work.
* **Apple-MPS friendly**: ``KMP_DUPLICATE_LIB_OK=TRUE`` and a small
  ``--batch-size`` (default 32) match the values that produced 9-13 it/s
  in the bake-off without OOM / thermal throttling.
* **Memory**: 2.34 M × 1024 × float32 ≈ 9.5 GB, so we stream rows from
  the DB chunk-by-chunk and only assemble the final ``.npz`` at the end.

Usage
-----
::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \\
        python3 scripts/build_embeddings_v2.py encode \\
        --chunk-size 50000 --batch-size 32

    PYTHONPATH=src python3 scripts/build_embeddings_v2.py merge

    PYTHONPATH=src python3 scripts/build_embeddings_v2.py faiss
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402

# ---- output layout -------------------------------------------------------
# Paths can be overridden by env vars so that eval scripts can point this
# pipeline at a pilot ``.npz`` / ``.faiss`` without rewriting code.
DATA_DIR = REPO_ROOT / "data"
SHARD_DIR = Path(
    os.environ.get("CHEMTREE_EMBEDDINGS_V2_SHARD_DIR", "")
    or DATA_DIR / "claim_embeddings_v2_shards"
).expanduser()
NPZ_PATH = Path(
    os.environ.get("CHEMTREE_EMBEDDINGS_V2_NPZ", "")
    or DATA_DIR / "claim_embeddings.v2.npz"
).expanduser()
FAISS_PATH = Path(
    os.environ.get("CHEMTREE_EMBEDDINGS_V2_FAISS", "")
    or DATA_DIR / "claim_embeddings.v2.faiss"
).expanduser()

# ---- encoder config (must match the bake-off winner) ---------------------
MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_DIM = 1024
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)
DOC_PREFIX = ""  # mxbai's recipe: prefix on queries only.

# FAISS HNSW knobs (match production ``embeddings.py``).
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200


def _get_device(force: Optional[str] = None) -> str:
    if force:
        return force
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


def _iter_rows(conn: sqlite3.Connection, chunk_size: int):
    """Yield (claim_id, text) tuples chunk-by-chunk in claim_id order."""
    cur = conn.execute(
        "SELECT c.claim_id, c.data, c.claim_contextualized, s.paper_summary "
        "FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
        "ORDER BY c.claim_id"
    )
    cur.arraysize = chunk_size
    while True:
        rows = cur.fetchmany(chunk_size)
        if not rows:
            return
        out = []
        for r in rows:
            cid = r[0]
            try:
                claim = json.loads(r[1])
            except Exception:
                continue
            txt = _claim_to_text(
                claim,
                claim_contextualized=r[2],
                paper_summary=r[3],
            )
            if txt:
                out.append((cid, txt))
        if out:
            yield out


def cmd_encode(args: argparse.Namespace) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    done_shards = sorted(SHARD_DIR.glob("shard_*.npz"))
    done_count = len(done_shards)
    completed_ids: set[str] = set()
    if done_shards:
        for s in done_shards:
            with np.load(str(s)) as z:
                completed_ids.update(z["claim_ids"].tolist())
    print(f"=== build_embeddings_v2 encode ===")
    print(f"  model        : {MODEL_ID}")
    print(f"  shard dir    : {SHARD_DIR}")
    print(f"  resumed      : {done_count} shard(s) "
          f"covering {len(completed_ids):,} claims")

    db_path = get_db_path()
    print(f"  db           : {db_path}")
    total = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True).execute(
        "SELECT COUNT(*) FROM claims"
    ).fetchone()[0]
    remaining_est = total - len(completed_ids)
    print(f"  total claims : {total:,}")
    print(f"  remaining    : ~{remaining_est:,}\n")

    device = _get_device(args.device)
    print(f"loading {MODEL_ID} on {device}…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, device=device)

    try:
        import torch  # local import so the merge/faiss subcommands stay light
    except Exception:
        torch = None

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    shard_idx = done_count
    encoded = 0
    skipped = 0
    t_global = time.monotonic()
    for chunk in _iter_rows(conn, args.chunk_size):
        chunk = [(cid, txt) for cid, txt in chunk if cid not in completed_ids]
        skipped += args.chunk_size - len(chunk)
        if not chunk:
            continue
        ids = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]
        if DOC_PREFIX:
            texts = [DOC_PREFIX + t for t in texts]
        t0 = time.monotonic()
        vecs = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        dt = time.monotonic() - t0
        out_path = SHARD_DIR / f"shard_{shard_idx:05d}.npz"
        tmp_stem = SHARD_DIR / f".shard_{shard_idx:05d}.partial"
        tmp_npz = tmp_stem.with_suffix(".partial.npz")
        np.savez(
            str(tmp_stem),
            claim_ids=np.array(ids, dtype="U64"),
            embeddings=vecs,
        )
        os.replace(tmp_npz, out_path)
        encoded += len(ids)
        shard_idx += 1
        rate = len(ids) / dt if dt > 0 else 0.0
        elapsed_total = time.monotonic() - t_global
        print(
            f"  shard {shard_idx-1:05d}: {len(ids):>6} claims  "
            f"{dt:>6.1f}s  {rate:>6.1f} c/s  "
            f"(total {encoded:,} in {elapsed_total/60:.1f} min)",
            flush=True,
        )
        del vecs, ids, texts, chunk
        gc.collect()
        if torch is not None and device == "mps":
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception:
                pass
        elif torch is not None and device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if args.max_chunks and (shard_idx - done_count) >= args.max_chunks:
            print(f"  reached --max-chunks={args.max_chunks}; stopping.")
            break
    conn.close()
    print(f"\nencoded {encoded:,} new claims into {shard_idx} shard(s)")


def cmd_merge(args: argparse.Namespace) -> None:
    shards = sorted(SHARD_DIR.glob("shard_*.npz"))
    if not shards:
        print(f"no shards found in {SHARD_DIR}")
        return
    print(f"merging {len(shards)} shards → {NPZ_PATH}")
    ids_list = []
    vecs_list = []
    for s in shards:
        with np.load(str(s)) as z:
            ids_list.append(z["claim_ids"])
            vecs_list.append(z["embeddings"])
    ids = np.concatenate(ids_list)
    vecs = np.concatenate(vecs_list, axis=0)
    print(f"  total ids    : {len(ids):,}")
    print(f"  unique ids   : {len(set(ids.tolist())):,}")
    print(f"  vec shape    : {vecs.shape} dtype={vecs.dtype}")
    if len(ids) != len(set(ids.tolist())):
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        vecs = vecs[order]
        keep = np.ones(len(ids), dtype=bool)
        keep[1:] = ids[1:] != ids[:-1]
        ids = ids[keep]
        vecs = vecs[keep]
        print(f"  deduped to   : {len(ids):,}")
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(NPZ_PATH),
        claim_ids=ids,
        embeddings=vecs.astype(np.float32),
    )
    size_mb = NPZ_PATH.stat().st_size / 1e6
    print(f"wrote {NPZ_PATH} ({size_mb:.1f} MB)")


def cmd_faiss(args: argparse.Namespace) -> None:
    import faiss
    if not NPZ_PATH.exists():
        raise SystemExit(f"npz not found: {NPZ_PATH} — run merge first.")
    with np.load(str(NPZ_PATH)) as z:
        ids = z["claim_ids"]
        vecs = z["embeddings"].astype(np.float32)
    n, d = vecs.shape
    print(f"building HNSW index for {n:,} vectors (dim={d})")
    if not vecs.flags["C_CONTIGUOUS"]:
        vecs = np.ascontiguousarray(vecs)
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    t0 = time.monotonic()
    index.add(vecs)
    dt = time.monotonic() - t0
    print(f"  added {n:,} vectors in {dt/60:.1f} min")
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    size_mb = FAISS_PATH.stat().st_size / 1e6
    print(f"wrote {FAISS_PATH} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Encode claims into shard files.")
    pe.add_argument("--chunk-size", type=int, default=50_000)
    pe.add_argument("--batch-size", type=int, default=32)
    pe.add_argument("--device", default=None)
    pe.add_argument("--max-chunks", type=int, default=0,
                    help="Stop after N new shards (for sanity checks).")
    pe.set_defaults(func=cmd_encode)

    pm = sub.add_parser("merge",
                        help="Merge shard files → claim_embeddings.v2.npz.")
    pm.set_defaults(func=cmd_merge)

    pf = sub.add_parser("faiss",
                        help="Build FAISS HNSW index from the merged npz.")
    pf.set_defaults(func=cmd_faiss)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
