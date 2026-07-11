"""Diagnose the cluster vs sentence-transformers encoding drift.

Re-encodes a small sample using **the exact cluster recipe** (raw
transformers, mean-pool, L2 norm) under three precisions:
``fp32``, ``fp16``, ``bf16``. Compares each against:

- the persisted cluster vector (the ground truth in the npz)
- a fresh sentence-transformers encode (what query-time uses)

This pins down whether the drift comes from
(a) precision (cluster ran bf16, local query runs fp32),
(b) tokenizer/max_length, or
(c) some recipe-level mismatch (pooling, prefix, etc.).

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/diagnose_v2_drift.py --n 20
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
MODEL = "mixedbread-ai/mxbai-embed-large-v1"


def load_sample(n: int, seed: int):
    print(f"loading {NPZ_PATH.name}…", flush=True)
    with np.load(str(NPZ_PATH)) as z:
        ids = z["claim_ids"]
        emb = z["embeddings"]
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(ids), n, replace=False)
    sample_ids = ids[pick].tolist()
    cluster_vecs = emb[pick].copy()

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
    keep = [(i, cid) for i, cid in enumerate(sample_ids)
            if cid in text_by_id]
    keep_idx = np.array([i for i, _ in keep])
    keep_ids = [cid for _, cid in keep]
    return keep_ids, cluster_vecs[keep_idx], [text_by_id[cid]
                                              for cid in keep_ids]


def encode_raw_transformers(texts: list[str], dtype_name: str,
                            max_len: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[dtype_name]
    print(f"  -> raw transformers  dtype={dtype_name}  "
          f"max_len={max_len}  device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=dtype).eval()
    model.to(device)
    enc = tok(texts, padding=True, truncation=True,
              max_length=max_len, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    vecs = pooled.float().cpu().numpy().astype(np.float32, copy=False)
    del model, tok
    return vecs


def encode_sentence_transformers(texts: list[str], device: str) -> np.ndarray:
    print(f"  -> sentence-transformers  device={device}", flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device=device)
    return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)


def cos_stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    c = (a * b).sum(axis=1)
    print(f"  {name:<32s}  mean={c.mean():.4f}  median={np.median(c):.4f}  "
          f"min={c.min():.4f}  max={c.max():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    keep_ids, cluster_vecs, texts = load_sample(args.n, args.seed)
    print(f"  sample size: {len(keep_ids)}\n", flush=True)

    print("=== local re-encodes ===")
    raw_fp32 = encode_raw_transformers(texts, "fp32", 384, args.device)
    raw_bf16 = encode_raw_transformers(texts, "bf16", 384, args.device)
    raw_fp32_512 = encode_raw_transformers(texts, "fp32", 512, args.device)
    st_fp32 = encode_sentence_transformers(texts, args.device)

    print("\n=== cosine vs persisted cluster vectors ===")
    cos_stats("cluster vs raw_fp32 (max=384)", cluster_vecs, raw_fp32)
    cos_stats("cluster vs raw_bf16 (max=384)", cluster_vecs, raw_bf16)
    cos_stats("cluster vs raw_fp32 (max=512)", cluster_vecs, raw_fp32_512)
    cos_stats("cluster vs sentence-trf",       cluster_vecs, st_fp32)

    print("\n=== cosine within local re-encodes ===")
    cos_stats("raw_fp32 vs raw_bf16",          raw_fp32, raw_bf16)
    cos_stats("raw_fp32 vs sentence-trf",      raw_fp32, st_fp32)
    cos_stats("raw_fp32_512 vs raw_fp32_384",  raw_fp32_512, raw_fp32)


if __name__ == "__main__":
    main()
