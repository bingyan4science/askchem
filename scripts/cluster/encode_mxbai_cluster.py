"""H200/H100 encoder for ``mxbai-embed-large-v1`` — no sentence_transformers.

Designed for one-shot batch encoding on NYU HPC. Reads a JSONL of
``{claim_id, text}`` rows produced by
``scripts/dump_claims_for_encoding.py`` and writes a single
``embeddings.v2.npz`` (claim_ids + L2-normalised float32 vectors).

Why direct ``transformers`` instead of ``sentence_transformers``:
- Cluster's ``torch_env`` already has ``transformers``+``torch``; we avoid
  installing extras (``sentence_transformers`` doesn't yet support
  transformers 5.x cleanly).
- mxbai-large is a plain BERT-style encoder; sentence-transformers'
  config (``1_Pooling/config.json``) selects **CLS-token pooling** —
  i.e. ``hidden_state[:, 0]`` — which matches the contrastive-training
  recipe on the model card. This is the *default* below
  (``--pooling cls``). The first cluster run (May 5) accidentally used
  mean pooling and the resulting vectors were ~0.04 cos off the
  sentence-transformers query subspace, halving Recall@20 at full
  corpus scale. See
  ``docs/plans/2026-05-03-phase-alpha-gamma-rollout.md`` (γ2 results).

Throughput envelope on a single H200 SXM (141 GB HBM3e, bf16):
- batch_size 256, seq_len 256 → ~5–8 k claims/s tokenize-bound
- batch_size 128, seq_len 512 → ~3–5 k claims/s
We default to bs 192 / seq 384 (chemistry claims are typically short).

Usage::

    python scripts/cluster/encode_mxbai_cluster.py \\
        --jsonl   /scratch/$USER/embed_job/data/claims.jsonl \\
        --out-npz /scratch/$USER/embed_job/out/embeddings.v2.npz \\
        --hf-cache /scratch/$USER/embed_job/hf_cache \\
        --pooling cls \\
        --batch-size 192 --max-len 384 --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# These are intentionally late-imported after env vars are set.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True,
                   help="Input JSONL (claim_id, text per line).")
    p.add_argument("--out-npz", type=Path, required=True,
                   help="Output NPZ path (claim_ids + embeddings).")
    p.add_argument("--hf-cache", type=Path, default=None,
                   help="HF cache dir; defaults to $HF_HOME or ~/.cache/hf.")
    p.add_argument("--model", default="mixedbread-ai/mxbai-embed-large-v1")
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--max-len", type=int, default=384,
                   help="Tokenizer truncation length (mxbai max=512).")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"),
                   default="bf16")
    p.add_argument("--pooling", choices=("cls", "mean"), default="cls",
                   help="Pooling strategy. mxbai-embed-large-v1's "
                        "sentence-transformers config selects CLS pooling, "
                        "so this MUST be 'cls' to match the deployed query "
                        "path. 'mean' is retained only for diagnostic "
                        "reproduction of the May-5 run.")
    p.add_argument("--save-fp16", action="store_true",
                   help="Save embeddings as float16 (halves disk, ~zero "
                        "retrieval impact for normalised vectors).")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N rows (debug). 0 = full.")
    p.add_argument("--print-every", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_cache:
        args.hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(args.hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.hf_cache / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(args.hf_cache / "hub")

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA not available — refusing to run on CPU.")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print("=== encode_mxbai_cluster ===", flush=True)
    print(f"  model      : {args.model}", flush=True)
    print(f"  jsonl      : {args.jsonl}", flush=True)
    print(f"  out-npz    : {args.out_npz}", flush=True)
    print(f"  device     : {device}  dtype={args.dtype}", flush=True)
    print(f"  batch-size : {args.batch_size}  max-len={args.max_len}",
          flush=True)
    print(f"  pooling    : {args.pooling}", flush=True)
    print(f"  gpu        : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  hf-cache   : {os.environ.get('HF_HOME', '?')}", flush=True)

    print("\nloading tokenizer + model…", flush=True)
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch_dtype)
    model.eval().to(device)
    print(f"  loaded in {time.monotonic()-t0:.1f}s", flush=True)
    print(f"  hidden size : {model.config.hidden_size}", flush=True)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    # First pass: count rows so we can pre-allocate the output array.
    print("\ncounting rows…", flush=True)
    t0 = time.monotonic()
    n_rows = 0
    with args.jsonl.open() as fh:
        for _ in fh:
            n_rows += 1
    if args.limit and args.limit < n_rows:
        n_rows = args.limit
    print(f"  rows = {n_rows:,}  ({time.monotonic()-t0:.1f}s)",
          flush=True)

    out_dim = model.config.hidden_size
    save_dtype = np.float16 if args.save_fp16 else np.float32
    embeddings = np.zeros((n_rows, out_dim), dtype=save_dtype)
    claim_ids = np.empty(n_rows, dtype="U64")

    written = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    t0 = time.monotonic()
    last_print = 0

    @torch.inference_mode()
    def flush_batch() -> None:
        nonlocal written, batch_ids, batch_texts, last_print
        if not batch_texts:
            return
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch_dtype,
                            enabled=(args.dtype != "fp32")):
            out = model(**enc).last_hidden_state  # (B, L, H)
        if args.pooling == "cls":
            # Sentence-transformers' mxbai config:
            #   pooling_mode_cls_token: true
            # i.e. take the first ([CLS]) token's hidden state. This is
            # the recipe the encoder was contrastively trained for and
            # the recipe the deployed query path uses.
            pooled = out[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs = pooled.float().cpu().numpy().astype(save_dtype, copy=False)
        end = written + len(batch_ids)
        embeddings[written:end] = vecs
        claim_ids[written:end] = batch_ids
        written = end
        if written - last_print >= args.print_every:
            elapsed = time.monotonic() - t0
            rate = written / elapsed if elapsed > 0 else 0.0
            eta_min = (n_rows - written) / rate / 60 if rate > 0 else 0.0
            print(f"  {written:>9,}/{n_rows:,}  "
                  f"({rate:>6,.0f} c/s, ETA {eta_min:>5.1f} min)",
                  flush=True)
            last_print = written
        batch_ids.clear()
        batch_texts.clear()

    print("\nencoding…", flush=True)
    with args.jsonl.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            cid = row.get("claim_id")
            txt = row.get("text") or ""
            if not cid or not txt:
                continue
            batch_ids.append(cid)
            batch_texts.append(txt)
            if len(batch_texts) >= args.batch_size:
                flush_batch()
            if args.limit and written >= args.limit:
                break
        flush_batch()

    elapsed = time.monotonic() - t0
    if written < n_rows:
        embeddings = embeddings[:written]
        claim_ids = claim_ids[:written]

    print(f"\nencoded {written:,} in {elapsed/60:.1f} min "
          f"({written/elapsed:,.0f} c/s)", flush=True)

    print("\nsaving npz…", flush=True)
    t0 = time.monotonic()
    tmp = args.out_npz.with_suffix(".partial.npz")
    np.savez(str(tmp.with_suffix("")),
             claim_ids=claim_ids, embeddings=embeddings)
    os.replace(tmp, args.out_npz)
    size_gb = args.out_npz.stat().st_size / 1e9
    print(f"  wrote {args.out_npz} ({size_gb:.2f} GB) "
          f"in {time.monotonic()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
