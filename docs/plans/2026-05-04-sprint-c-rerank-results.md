# Sprint C — Phase γ1 cross-encoder rerank pilot

*Run date: 2026-05-04. 80-probe eval on 10 K pilot corpus.*

## TL;DR

Ship **`mxbai-embed-large` (dense, top-100) → `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank top-20**.

| stage                                                   | nDCG@10 | nDCG@20 | MRR@20 | p95 latency (MPS) |
|---------------------------------------------------------|--------:|--------:|-------:|------------------:|
| MiniLM dense (today's prod)                             |  0.797  |   —     |   —    |              <50 ms |
| **mxbai-large** dense (γ0 winner)                       |  0.885  |  0.864  |  0.924 |             <100 ms |
| mxbai → **ms-marco-mini** rerank **top-20**     **★**   |  0.907  |  0.871  |  0.937 |             **150 ms** |
| mxbai → ms-marco-mini rerank top-100                    |  0.912  |  0.897  |  0.927 |             4 600 ms |
| mxbai → bge-reranker-base rerank top-100                |  0.920  |  0.902  |  0.940 |            18 300 ms |

Compared to today's MiniLM-only baseline:
**Δ nDCG@10 = +0.110**  (0.797 → 0.907) — *14 % relative*.

The marginal lift from larger rerank windows / heavier rerankers is small relative to the latency cost on local Apple-MPS.  When we ship to a CUDA box, we can revisit `bge-reranker-base` top-100; until then top-20 is the right pick.

## Per-family breakdown (★ config)

| family    |   n | dense-only | rerank top-20 |  Δ nDCG@10 |
|-----------|----:|-----------:|--------------:|-----------:|
| homonym   |  10 |     0.888  |        0.887  |   −0.001   |
| material  |  10 |     0.862  |        0.897  |   +0.035   |
| multi     |  15 |     0.836  |        0.879  |   +0.043   |
| property  |  15 |     0.905  |        0.933  |   +0.028   |
| reaction  |  20 |     0.921  |        0.926  |   +0.005   |
| technique |  10 |     0.874  |        0.905  |   +0.031   |

The biggest wins are exactly where dense was weakest — **multi-step reasoning queries** and **technique queries** that mix instrument acronyms and what-it-measures language.  Reaction queries are already near-saturated (MRR@20 = 1.000) at the dense stage so the headroom is tiny.

## Method

* `scripts/encoder_bakeoff.py search --top-k 100` runs each candidate dense
  encoder against the 10 K pilot corpus, writing
  `data/eval/runs/pilot10-<model>-top100.rankings.jsonl`.
* `scripts/rerank_bakeoff.py` loads those rankings, hydrates the top-N
  candidates (`_claim_to_text(claim, claim_contextualized, paper_summary)`),
  and reranks each (query, doc) pair with the cross-encoder.
* `scripts/eval_metrics.py` scores against
  `data/eval/labels_v1.jsonl` (5 998 graded labels).

Reproducer (★ config):

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 scripts/encoder_bakeoff.py search \
    --model mxbai-large \
    --vecs data/eval/vecs/pilot10-mxbai-large.npz \
    --label pilot10-mxbai-large-top100 \
    --top-k 100

KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 scripts/rerank_bakeoff.py \
    --reranker ms-marco-mini \
    --base-rankings data/eval/runs/pilot10-mxbai-large-top100.rankings.jsonl \
    --top-rerank 20 \
    --label rerank-mxbai-msmarco-top20

PYTHONPATH=src python3 scripts/eval_metrics.py \
    --run rerank-mxbai-msmarco-top20 \
    --rankings data/eval/runs/rerank-mxbai-msmarco-top20.rankings.jsonl
```

## Latency notes

* `cross-encoder/ms-marco-MiniLM-L-6-v2` is 33 M params, FP32, on Apple-MPS.
  At top-20 the rerank step runs at p50=105 ms / p95=150 ms.
* Hydration cost for the unique candidate texts (≈1.5 K rows for 80 probes)
  is amortised across queries.
* For the prod retrieval path we'll co-locate the cross-encoder with the
  dense index and serve it with `torch.compile` or ONNX, which historically
  gets us another 1.5-2 × on CPU and much more on a small GPU.

## Decisions

1. **Production stack** (Phase γ3, after the full re-embed):

   ```
   query
     → MiniLM PAW classifier (intent / family routing)
     → mxbai-embed-large dense ANN (top-100)
     → BM25/FTS5 candidate (top-100)
     → RRF merge (top-50)
     → cross-encoder/ms-marco-MiniLM-L-6-v2 rerank (top-20)
     → final ranked list
   ```

2. **Rerank window**: 20.  Revisit at 50/100 when we have GPU
   inference; quality plateau is real but the latency tail is brutal on
   MPS.

3. **Bigger rerankers (bge-base, mxbai-large) deferred** until a CUDA
   inference box is available.  Numbers above are recorded so we can
   pick them up immediately.

4. **Same harness re-runs as γ3 acceptance gate**: nDCG@10 ≥ 0.90 and
   p95 latency ≤ 400 ms required before flipping the prod switch.

## Next

* γ2 — full re-embed (2.34 M claims) with `mxbai-embed-large`.
  Standalone script: `scripts/build_embeddings_v2.py`.  Outputs
  `data/claim_embeddings.v2.npz` + `data/claim_embeddings.v2.faiss`.
* γ3 — wire dense-v2 + rerank into `chemtree.embeddings` /
  `chemtree.search` behind `CHEMTREE_RETRIEVER_VERSION=v2` and re-run
  the 80-probe eval at full corpus to confirm the pilot gains hold.
* Final benchmark + GitHub / HuggingFace / server sync.
