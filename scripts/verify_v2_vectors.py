"""Spot-check: do cluster-encoded vectors match locally-encoded ones?

The cluster used raw HuggingFace ``transformers`` (mean-pool + L2 norm,
bf16 inference) to produce ``data/claim_embeddings.v2.npz``.

The query side uses ``sentence_transformers.SentenceTransformer``
loading the same checkpoint. If the two encoders disagree (different
pooling, different prefix policy, different normalization, etc.) all
queries will be subtly off-axis from the doc vectors and retrieval
quality collapses at scale.

This script encodes a small sample of claims with the local production
encoder and computes cosine similarity vs the cluster-encoded vector
for the same id. Healthy alignment: cosine ~0.99+. Drift below 0.95
indicates a real recipe mismatch.

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/verify_v2_vectors.py --n 50
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402

NPZ_PATH = REPO_ROOT / "data" / "claim_embeddings.v2.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50,
                   help="How many claims to spot-check.")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print(f"loading {NPZ_PATH.name}…", flush=True)
    with np.load(str(NPZ_PATH)) as z:
        ids = z["claim_ids"]
        emb = z["embeddings"]
    print(f"  cluster-encoded: {emb.shape} {emb.dtype}", flush=True)

    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(ids), args.n, replace=False)
    sample_ids = ids[pick].tolist()
    cluster_vecs = emb[pick]

    db_path = get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    placeholder = ",".join("?" * len(sample_ids))
    rows = conn.execute(
        f"SELECT c.claim_id, c.data, c.claim_contextualized, "
        f"       s.paper_summary "
        f"FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
        f"WHERE c.claim_id IN ({placeholder})",
        sample_ids,
    ).fetchall()
    conn.close()

    text_by_id: dict[str, str] = {}
    for r in rows:
        try:
            claim = json.loads(r["data"])
        except Exception:
            continue
        txt = _claim_to_text(
            claim,
            claim_contextualized=r["claim_contextualized"],
            paper_summary=r["paper_summary"],
        )
        if txt:
            text_by_id[r["claim_id"]] = txt
    missing = [cid for cid in sample_ids if cid not in text_by_id]
    if missing:
        print(f"  WARN: {len(missing)} ids had no text (skipping)")

    keep = [(i, cid) for i, cid in enumerate(sample_ids)
            if cid in text_by_id]
    if not keep:
        raise SystemExit("no valid samples")
    keep_idx = np.array([i for i, _ in keep])
    keep_ids = [cid for _, cid in keep]
    cluster_vecs = cluster_vecs[keep_idx]
    texts = [text_by_id[cid] for cid in keep_ids]

    print(f"\nencoding {len(texts)} claims locally with sentence-transformers"
          f"…", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1",
        device="mps",
    )
    local_vecs = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    cos = (cluster_vecs * local_vecs).sum(axis=1)
    print()
    print(f"=== cluster ↔ local cosine (n={len(cos)}) ===")
    print(f"  mean   : {cos.mean():.4f}")
    print(f"  median : {np.median(cos):.4f}")
    print(f"  min    : {cos.min():.4f}")
    print(f"  max    : {cos.max():.4f}")
    print(f"  std    : {cos.std():.4f}")
    print()
    print("worst 5 ids (lowest cosine to local re-encode):")
    order = np.argsort(cos)
    for i in order[:5]:
        print(f"  {cos[i]:.4f}  {keep_ids[i]:<60s}")
        snippet = texts[i].replace("\n", " ")[:160]
        print(f"           {snippet!r}")

    print()
    if cos.mean() > 0.99:
        print("PASS: cluster and local encoders agree (mean cos > 0.99).")
    elif cos.mean() > 0.95:
        print("WARN: cluster vs local cosine 0.95-0.99 — mild drift.")
    else:
        print("FAIL: cluster ↔ local cosine < 0.95 — recipe mismatch likely.")


if __name__ == "__main__":
    main()
