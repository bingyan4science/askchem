"""Encoder bake-off harness for Phase γ (Sprint C).

Two-step pipeline so a 30-min encode is never lost to a late crash:

    encode  — load model, encode pilot corpus, save .npz to disk
    search  — load .npz, build FAISS HNSW, run labelled probes,
              save rankings JSONL for ``eval_metrics.py`` to score

For the MiniLM control we don't re-encode — we filter the production
``data/claim_embeddings.npz`` down to the pilot ids:

    PYTHONPATH=src python3 scripts/encoder_bakeoff.py slice-prod \
        --corpus data/eval/sample_200k.jsonl \
        --out data/eval/vecs/pilot-minilm.npz

For every candidate model:

    PYTHONPATH=src python3 scripts/encoder_bakeoff.py encode \
        --model bge-large \
        --corpus data/eval/sample_200k.jsonl \
        --out data/eval/vecs/pilot-bge-large.npz

    PYTHONPATH=src python3 scripts/encoder_bakeoff.py search \
        --model bge-large \
        --vecs data/eval/vecs/pilot-bge-large.npz \
        --label pilot-bge-large

Then score (using the standard eval entry point):

    PYTHONPATH=src python3 scripts/eval_metrics.py \
        --run pilot-bge-large \
        --rankings data/eval/runs/pilot-bge-large.rankings.jsonl

This deliberately tests **dense retrieval only** — no FTS, no RRF, no
tree, no rerank. That isolates the encoder.
"""
from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402
from eval_common import PROBES_PATH, load_probes  # noqa: E402

EVAL_DIR = REPO_ROOT / "data" / "eval"
RUNS_DIR = EVAL_DIR / "runs"
VECS_DIR = EVAL_DIR / "vecs"
PROD_EMBEDDINGS_PATH = REPO_ROOT / "data" / "claim_embeddings.npz"


# ── Model registry ─────────────────────────────────────────────────────────


@dataclass
class EncoderConfig:
    name: str
    model_id: str
    dim: int
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False
    max_seq_len: Optional[int] = None
    batch_size: int = 64


MODELS: dict[str, EncoderConfig] = {
    "minilm": EncoderConfig(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        batch_size=256,
    ),
    "bge-large": EncoderConfig(
        name="bge-large",
        model_id="BAAI/bge-large-en-v1.5",
        dim=1024,
        batch_size=32,
    ),
    "bge-base": EncoderConfig(
        name="bge-base",
        model_id="BAAI/bge-base-en-v1.5",
        dim=768,
        batch_size=64,
    ),
    "e5-large": EncoderConfig(
        name="e5-large",
        model_id="intfloat/e5-large-v2",
        dim=1024,
        query_prefix="query: ",
        doc_prefix="passage: ",
        batch_size=32,
    ),
    "pubmedbert": EncoderConfig(
        name="pubmedbert",
        model_id="pritamdeka/S-PubMedBert-MS-MARCO",
        dim=768,
        batch_size=64,
    ),
    "mxbai-large": EncoderConfig(
        name="mxbai-large",
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        dim=1024,
        query_prefix=(
            "Represent this sentence for searching relevant passages: "
        ),
        batch_size=32,
    ),
    "nomic": EncoderConfig(
        name="nomic",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        trust_remote_code=True,
        batch_size=32,
    ),
}


# ── Corpus loading ─────────────────────────────────────────────────────────


def load_corpus_ids(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw)["claim_id"])
        except Exception:
            continue
    return out


def hydrate_texts(claim_ids: list[str]) -> tuple[list[str], list[str]]:
    db_path = get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        BATCH = 900
        order = {cid: i for i, cid in enumerate(claim_ids)}
        rows_by_id: dict[str, sqlite3.Row] = {}
        t0 = time.monotonic()
        for i in range(0, len(claim_ids), BATCH):
            chunk = claim_ids[i:i + BATCH]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT c.claim_id, c.data, c.claim_contextualized, "
                f"s.paper_summary "
                f"FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
                f"WHERE c.claim_id IN ({ph})",
                chunk,
            ).fetchall()
            for r in rows:
                rows_by_id[r["claim_id"]] = r
        kept_ids: list[str] = []
        kept_texts: list[str] = []
        for cid in sorted(rows_by_id.keys(), key=lambda c: order.get(c, 1 << 30)):
            r = rows_by_id[cid]
            try:
                claim = json.loads(r["data"])
            except Exception:
                continue
            text = _claim_to_text(
                claim,
                claim_contextualized=r["claim_contextualized"],
                paper_summary=r["paper_summary"],
            )
            if not text:
                continue
            kept_ids.append(cid)
            kept_texts.append(text)
        print(f"hydrated {len(kept_ids):,}/{len(claim_ids):,} claims "
              f"in {time.monotonic()-t0:.1f}s")
        return kept_ids, kept_texts
    finally:
        conn.close()


# ── Encoder loading ────────────────────────────────────────────────────────


def _get_device(force: Optional[str] = None) -> str:
    if force:
        return force
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_encoder(cfg: EncoderConfig, device: str):
    from sentence_transformers import SentenceTransformer
    kwargs = {}
    if cfg.trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(cfg.model_id, device=device, **kwargs)
    if cfg.max_seq_len:
        model.max_seq_length = cfg.max_seq_len
    return model


# ── Subcommand: slice production MiniLM into pilot ─────────────────────────


def cmd_slice_prod(args):
    print(f"loading production embeddings from {PROD_EMBEDDINGS_PATH}…")
    t0 = time.monotonic()
    data = np.load(str(PROD_EMBEDDINGS_PATH))
    prod_ids = data["claim_ids"]
    prod_vecs = data["embeddings"]
    print(f"  loaded {len(prod_ids):,} prod vecs ({prod_vecs.shape[1]}d) "
          f"in {time.monotonic()-t0:.1f}s")
    prod_idx = {cid: i for i, cid in enumerate(prod_ids.tolist())}

    pilot_ids = load_corpus_ids(args.corpus)
    print(f"  pilot corpus has {len(pilot_ids):,} ids")
    out_ids: list[str] = []
    out_idxs: list[int] = []
    missing = 0
    for cid in pilot_ids:
        idx = prod_idx.get(cid)
        if idx is None:
            missing += 1
            continue
        out_ids.append(cid)
        out_idxs.append(idx)
    print(f"  matched {len(out_ids):,}; missing {missing:,}")

    out_vecs = prod_vecs[np.asarray(out_idxs, dtype=np.int64)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.out),
        claim_ids=np.array(out_ids, dtype="U64"),
        embeddings=out_vecs.astype(np.float32),
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"  saved {args.out} ({size_mb:.1f} MB)")


# ── Subcommand: encode ─────────────────────────────────────────────────────


def cmd_encode(args):
    cfg = MODELS[args.model]
    device = _get_device(args.device)
    print(f"=== encode ({cfg.name}) ===")
    print(f"  model    : {cfg.model_id}")
    print(f"  device   : {device}")
    print(f"  corpus   : {args.corpus}")
    print(f"  out      : {args.out}\n")

    cids = load_corpus_ids(args.corpus)
    print(f"corpus ids: {len(cids):,}")
    kept_cids, texts = hydrate_texts(cids)

    if cfg.doc_prefix:
        texts = [cfg.doc_prefix + t for t in texts]

    print(f"\nloading encoder {cfg.model_id}…")
    model = load_encoder(cfg, device)

    print("\nencoding corpus…")
    bs = args.batch_size or cfg.batch_size
    t0 = time.monotonic()
    vecs = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.monotonic() - t0
    print(f"\nencoded {len(texts):,} in {elapsed:.1f}s "
          f"({len(texts)/max(elapsed,1):.0f} / s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nsaving vecs to {args.out} (uncompressed for speed)…")
    np.savez(
        str(args.out),
        claim_ids=np.array(kept_cids, dtype="U64"),
        embeddings=vecs,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"  saved ({size_mb:.0f} MB)")

    del vecs, texts
    gc.collect()


# ── Subcommand: search ─────────────────────────────────────────────────────


def cmd_search(args):
    cfg = MODELS[args.model]
    label = args.label or f"pilot-{args.model}"
    device = _get_device(args.device)
    print(f"=== search ({label}) ===")

    print(f"loading {args.vecs}…")
    t0 = time.monotonic()
    data = np.load(str(args.vecs), allow_pickle=False)
    cids = data["claim_ids"].tolist()
    vecs = data["embeddings"]
    print(f"  loaded {len(cids):,} vecs ({vecs.shape[1]}d) "
          f"in {time.monotonic()-t0:.1f}s")

    print("building FAISS HNSW index…")
    import faiss
    n, d = vecs.shape
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    t0 = time.monotonic()
    for i in range(0, n, 100_000):
        index.add(vecs[i:i + 100_000])
    print(f"  built ({d}d) in {time.monotonic()-t0:.1f}s")

    print(f"loading encoder {cfg.model_id} for queries…")
    model = load_encoder(cfg, device)

    probes = load_probes(PROBES_PATH)
    raw_qs = [(cfg.query_prefix + p.q) for p in probes]
    print(f"\nencoding {len(probes)} queries…")
    qvecs = model.encode(
        raw_qs,
        batch_size=min(32, cfg.batch_size),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print("running probes…")
    t0 = time.monotonic()
    _, idxs = index.search(qvecs, args.top_k)
    print(f"  searched {len(probes)} probes in {time.monotonic()-t0:.2f}s "
          f"(p95 ≈ {(time.monotonic()-t0)*1000/len(probes):.0f} ms / probe)")

    rankings: list[dict] = []
    for i, p in enumerate(probes):
        ranked = [cids[j] for j in idxs[i] if 0 <= j < len(cids)]
        rankings.append({"probe_id": p.id, "ranked_claim_ids": ranked})

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{label}.rankings.jsonl"
    with out_path.open("w") as fh:
        for row in rankings:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote rankings to {out_path}")
    print(f"\nNext: PYTHONPATH=src python3 scripts/eval_metrics.py "
          f"--run {label} --rankings {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Encode pilot with a candidate model")
    pe.add_argument("--model", required=True, choices=sorted(MODELS))
    pe.add_argument("--corpus", type=Path,
                    default=EVAL_DIR / "sample_200k.jsonl")
    pe.add_argument("--out", type=Path, required=True,
                    help=".npz output path for encoded vecs")
    pe.add_argument("--batch-size", type=int, default=None)
    pe.add_argument("--device", default=None)
    pe.set_defaults(func=cmd_encode)

    ps = sub.add_parser("search", help="Build FAISS, run probes, save rankings")
    ps.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="Used for query encoding (must match the vecs file)")
    ps.add_argument("--vecs", type=Path, required=True)
    ps.add_argument("--label", default=None)
    ps.add_argument("--top-k", type=int, default=20)
    ps.add_argument("--device", default=None)
    ps.set_defaults(func=cmd_search)

    sl = sub.add_parser("slice-prod",
                        help="Filter prod MiniLM embeddings down to the pilot")
    sl.add_argument("--corpus", type=Path,
                    default=EVAL_DIR / "sample_200k.jsonl")
    sl.add_argument("--out", type=Path, required=True)
    sl.set_defaults(func=cmd_slice_prod)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
