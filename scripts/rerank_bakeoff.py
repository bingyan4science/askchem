"""Cross-encoder rerank pilot for Phase γ1.

Takes the top-N rankings from a bi-encoder run (produced by
``encoder_bakeoff.py search --top-k 100``) and reranks each probe's
candidates with a cross-encoder, then writes a new rankings file that
``eval_metrics.py`` can score.

Cross-encoder candidates (registry below):
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (33 M, fastest)
  - BAAI/bge-reranker-base                (278 M, stronger)
  - mixedbread-ai/mxbai-rerank-large-v1   (435 M, top of leaderboard)

The rerank operates on the same ``_claim_to_text(claim)`` text the
bi-encoder indexed, so the comparison is apples-to-apples on the same
document representation.

Usage::

    PYTHONPATH=src python3 scripts/rerank_bakeoff.py \
        --reranker ms-marco-mini \
        --base-rankings data/eval/runs/pilot10-mxbai-large-top100.rankings.jsonl \
        --label rerank-mxbai-mxminI

Latency budget per probe (top-100 rerank): ≤ 400 ms p95 (per the
Sprint-C plan).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from askchem.db import get_db_path  # noqa: E402
from askchem.embeddings import _claim_to_text  # noqa: E402
from eval_common import PROBES_PATH, load_probes  # noqa: E402

EVAL_DIR = REPO_ROOT / "data" / "eval"
RUNS_DIR = EVAL_DIR / "runs"


@dataclass
class CrossEncoderConfig:
    name: str
    model_id: str
    batch_size: int = 64
    max_length: int = 512


CROSS_MODELS: dict[str, CrossEncoderConfig] = {
    "ms-marco-mini": CrossEncoderConfig(
        name="ms-marco-mini",
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=128,
    ),
    "bge-reranker-base": CrossEncoderConfig(
        name="bge-reranker-base",
        model_id="BAAI/bge-reranker-base",
        batch_size=64,
    ),
    "mxbai-rerank-large": CrossEncoderConfig(
        name="mxbai-rerank-large",
        model_id="mixedbread-ai/mxbai-rerank-large-v1",
        batch_size=16,
    ),
}


def load_rankings(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        d = json.loads(raw)
        out[d["probe_id"]] = list(d["ranked_claim_ids"])
    return out


def hydrate_text_for(claim_ids: set[str]) -> dict[str, str]:
    """Return {claim_id: _claim_to_text(...)} for every requested id."""
    if not claim_ids:
        return {}
    db_path = get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        BATCH = 900
        out: dict[str, str] = {}
        ids = list(claim_ids)
        for i in range(0, len(ids), BATCH):
            chunk = ids[i:i + BATCH]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT c.claim_id, c.data, c.claim_contextualized, "
                f"s.paper_summary "
                f"FROM claims c LEFT JOIN sources s ON c.source_doi = s.doi "
                f"WHERE c.claim_id IN ({ph})",
                chunk,
            ).fetchall()
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
                    out[r["claim_id"]] = txt
        return out
    finally:
        conn.close()


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reranker", required=True, choices=sorted(CROSS_MODELS))
    p.add_argument("--base-rankings", type=Path, required=True,
                   help="Top-N rankings JSONL from a bi-encoder run "
                        "(default top_k=100).")
    p.add_argument("--label", required=True,
                   help="Output rankings label, e.g. 'rerank-mxbai-mxminI'.")
    p.add_argument("--top-rerank", type=int, default=100,
                   help="Rerank only the first N candidates per probe.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = CROSS_MODELS[args.reranker]
    device = _get_device(args.device)
    print(f"=== rerank pilot: {args.label} ===")
    print(f"  reranker      : {cfg.model_id}")
    print(f"  base rankings : {args.base_rankings}")
    print(f"  top-N rerank  : {args.top_rerank}")
    print(f"  device        : {device}\n")

    base = load_rankings(args.base_rankings)
    probes = load_probes(PROBES_PATH)
    print(f"loaded {len(base)} ranked probes "
          f"({sum(len(v) for v in base.values()):,} candidate ids)")

    needed: set[str] = set()
    for cids in base.values():
        for cid in cids[: args.top_rerank]:
            needed.add(cid)
    print(f"hydrating text for {len(needed):,} unique candidate claims…")
    t0 = time.monotonic()
    text_by_id = hydrate_text_for(needed)
    print(f"  hydrated {len(text_by_id):,} in {time.monotonic()-t0:.1f}s")

    print(f"\nloading reranker {cfg.model_id}…")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(cfg.model_id, device=device, max_length=cfg.max_length)

    print("\nreranking…")
    rankings: list[dict] = []
    per_probe_ms: list[float] = []
    for i, p_ in enumerate(probes, 1):
        cands = base.get(p_.id, [])[: args.top_rerank]
        pairs: list[tuple[str, str]] = []
        kept_ids: list[str] = []
        for cid in cands:
            txt = text_by_id.get(cid)
            if not txt:
                continue
            pairs.append((p_.q, txt))
            kept_ids.append(cid)
        if not pairs:
            rankings.append({"probe_id": p_.id, "ranked_claim_ids": []})
            continue
        t0 = time.monotonic()
        scores = model.predict(
            pairs,
            batch_size=cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        per_probe_ms.append(elapsed_ms)
        order = sorted(range(len(scores)), key=lambda j: -float(scores[j]))
        ranked = [kept_ids[j] for j in order]
        rankings.append({"probe_id": p_.id, "ranked_claim_ids": ranked})
        if i <= 3 or i % 20 == 0:
            print(f"  [{i:>2}/{len(probes)}] {p_.id:<8} family={p_.family:<10} "
                  f"reranked {len(pairs):>3}  ({elapsed_ms:.0f} ms)")

    if per_probe_ms:
        per_probe_ms.sort()
        p50 = per_probe_ms[len(per_probe_ms) // 2]
        p95 = per_probe_ms[int(len(per_probe_ms) * 0.95)]
        print(f"\nlatency: p50={p50:.0f} ms  p95={p95:.0f} ms  "
              f"max={max(per_probe_ms):.0f} ms")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{args.label}.rankings.jsonl"
    with out_path.open("w") as fh:
        for row in rankings:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote rankings to {out_path}")
    print(f"\nNext: PYTHONPATH=src python3 scripts/eval_metrics.py "
          f"--run {args.label} --rankings {out_path}")


if __name__ == "__main__":
    main()
