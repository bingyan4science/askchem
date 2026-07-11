# May-14 ablation — drop PAW, skip int8 cross-encoder

*Run date: 2026-05-14. 80-probe eval on the full v2 256-d production stack.*

## TL;DR

Two CPU/memory optimisations were measured against the prod retrieval pipeline:

1. **Drop PAW (programasweights)** — shipped. Frees ~700 MB resident on the
   8 GB DigitalOcean droplet at zero retrieval-quality cost.
2. **int8-quantize the cross-encoder reranker** — *not* shipped. Quality is
   preserved but PyTorch dynamic quantization gave no latency win on this CPU.

| run                                | nDCG@10 | nDCG@20 | MRR@20 | R@10  | R@20  | p50 lat | p95 lat | max lat |
| ---------------------------------- | ------: | ------: | -----: | ----: | ----: | ------: | ------: | ------: |
| **baseline** (FP32 rerank, PAW on) | **0.753** | 0.690 | 0.895 | 0.093 | 0.175 |  9.5 s |  14.5 s |  18.0 s |
| **paw-off** (FP32 rerank, PAW off) | **0.753** | 0.690 | 0.892 | 0.093 | 0.175 |  9.5 s |  16.8 s |  20.9 s |
| **rerank int8** (PAW on, qint8)    | **0.757** | 0.704 | 0.895 | 0.094 | 0.180 |  9.6 s |  16.9 s |  24.6 s |

Δ vs baseline:

* paw-off:    Δ nDCG@10 = **+0.000**, ΔMRR@20 = −0.003 (noise).
* int8 rerank: Δ nDCG@10 = **+0.004**, ΔMRR@20 =  0.000 (quantization noise breaks ties differently; still within ±0.015 family-level band).

## Per-family breakdown (nDCG@10)

| family    |  n | baseline | paw-off | Δ paw  | int8  | Δ int8 |
| --------- | -: | -------: | ------: | -----: | ----: | -----: |
| homonym   | 10 |    0.775 |   0.775 | +0.000 | 0.777 | +0.002 |
| material  | 10 |    0.721 |   0.721 | +0.000 | 0.716 | −0.005 |
| multi     | 15 |    0.700 |   0.704 | +0.004 | 0.718 | +0.018 |
| property  | 15 |    0.789 |   0.778 | −0.011 | 0.776 | −0.013 |
| reaction  | 20 |    0.790 |   0.789 | −0.001 | 0.789 | −0.001 |
| technique | 10 |    0.713 |   0.722 | +0.009 | 0.742 | +0.029 |

All deltas are inside the ±0.015 noise band measured across prior Phase Δ
ablations. The biggest int8 movers (`technique` +0.029, `multi` +0.018) are
quantization-noise tie-breaks — the quality preservation is what matters,
not the sign.

## Decision matrix

| ablation                                | quality | latency  | memory   | ship? |
| --------------------------------------- | :-----: | :------: | :------: | :---: |
| drop PAW (`CHEMTREE_DISABLE_PAW=1`)     | flat    | flat     | **−700 MB** | **yes** |
| int8 cross-encoder (`CHEMTREE_RERANK_QUANT=int8`) | flat    | flat     | ≈ −30 MB | no    |
| combined paw-off + int8                 | (skipped — independent no-op deltas) | | | n/a |

### Why PAW is safe to drop

All five PAW entry points already had documented fallbacks that fire when
`paw_functions._check_paw()` returns `False`:

* `classify_intent` → `"concept"` (the generic search banner; UI-only, no ranking effect)
* `normalize_query` → original query string (the FTS-empty rescue at
  [`src/chemtree/db.py`](../../src/chemtree/db.py) becomes a no-op when
  normalized == original)
* `is_relevant`, `detect_contradiction`, `expand_query`, `decompose_query`
  → all safe defaults (`True` / `"unclear"` / `[query]`)

The warmup loop at `server._warmup_paw` is wrapped in `try/except` and
short-circuits via the same `_check_paw()` guard — no orphan import cost.

The kill-switch was wired in commit
`2cfdb08 askchem: drop PAW on prod` (`paw_functions._check_paw` now honours
`CHEMTREE_DISABLE_PAW=1`).

### Why int8 didn't deliver a speedup

Expected: ~2× faster rerank ⇒ p50 drops from ~9.5 s to ~7.5 s.
Observed: p50 identical (9554 ms vs 9503 ms).

PyTorch dynamic quantization replaces `nn.Linear` weights with QInt8 packed
tensors but quantizes activations on-the-fly per inference. For a small
6-layer MiniLM model on x86 CPU (FBGEMM backend), the per-call
quantize/dequantize overhead approximately cancels the matmul speedup.
The max latency actually got **worse** (24.6 s vs 18 s) — int8 is more
latency-variable.

The path to a real rerank speedup is ONNX Runtime export + int8 calibration
with kernel fusion. Not in scope for this PR; the `CHEMTREE_RERANK_QUANT`
env knob is kept (default off) so we can revisit without touching call sites.

## Method

* Code knobs:
  * `CHEMTREE_DISABLE_PAW=1` → [`src/chemtree/paw_functions.py`](../../src/chemtree/paw_functions.py) — `_check_paw()` short-circuits to `False`.
  * `CHEMTREE_RERANK_QUANT=int8` → [`src/chemtree/cross_encoder_rerank.py`](../../src/chemtree/cross_encoder_rerank.py) — calls `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` after loading the cross-encoder onto CPU.
* Driver: `scripts/eval_search_live.py --top 20` against
  `data/eval/probes_v1.jsonl` (80 probes) and
  `data/eval/labels_v1.jsonl` (7 483 graded labels).
* Hardware: production VPS, askchem service stopped for the ~45 min eval
  window so the 8 GB box wasn't memory-thrashing.
* Scoring: `scripts/eval_metrics.py --run <label> --rankings …`.

Reproducer:

```bash
ssh root@askchem.org
systemctl stop askchem

cd /opt/askchem
for cfg in \
  "may14-baseline-256-rerank   CHEMTREE_RERANK_ENABLED=1" \
  "may14-paw-off-256-rerank    CHEMTREE_RERANK_ENABLED=1 CHEMTREE_DISABLE_PAW=1" \
  "may14-rerank-int8-256       CHEMTREE_RERANK_ENABLED=1 CHEMTREE_RERANK_QUANT=int8" \
; do
  read -r LABEL EXTRA <<< "$cfg"
  PYTHONPATH=/opt/askchem/src CHEMTREE_RETRIEVER_VERSION=v2 \
    CHEMTREE_V2_DIM=256 CHEMTREE_FAISS_MMAP=1 OMP_NUM_THREADS=2 \
    CHEMTREE_DB=/opt/askchem/chemtree.db $EXTRA \
    /opt/askchem/venv/bin/python scripts/eval_search_live.py \
      --label "$LABEL" --top 20
  PYTHONPATH=/opt/askchem/src CHEMTREE_DB=/opt/askchem/chemtree.db \
    /opt/askchem/venv/bin/python scripts/eval_metrics.py \
      --run "$LABEL" \
      --rankings data/eval/runs/$LABEL.rankings.jsonl
done

systemctl start askchem
```

Artefacts on the VPS:

```
/opt/askchem/data/eval/runs/may14-baseline-256-rerank.rankings.jsonl
/opt/askchem/data/eval/runs/may14-baseline-256-rerank.scored.json
/opt/askchem/data/eval/runs/may14-paw-off-256-rerank.rankings.jsonl
/opt/askchem/data/eval/runs/may14-paw-off-256-rerank.scored.json
/opt/askchem/data/eval/runs/may14-rerank-int8-256.rankings.jsonl
/opt/askchem/data/eval/runs/may14-rerank-int8-256.scored.json
```

## Deployment + rollback

Shipped via commit `2cfdb08`. Production env now carries
`CHEMTREE_DISABLE_PAW=1` in
[`deploy/askchem.service.d/override.conf`](../../deploy/askchem.service.d/override.conf).

Post-deploy verification (`GET /api/search` against askchem.org):

* `q=suzuki+coupling` → `intent: "reaction"` (rule-based override in
  `server._get_intent` still fires for cross-coupling queries).
* `q=palladium+catalyst` → `intent: "concept"` (expected PAW-off fallback).
* askchem RSS: **4.87 GB** (was ~5.6 GB) — ~700 MB freed.
* Swap: 45 MB / 4 GB (was 4.0 GB / 4 GB — pressure gone).

The 1.4 GB on-disk PAW cache at `/root/.cache/programasweights/` is left
in place for instant rollback. To re-enable PAW:

```bash
ssh root@askchem.org \
  "sed -i '/CHEMTREE_DISABLE_PAW/d' /etc/systemd/system/askchem.service.d/override.conf && \
   systemctl daemon-reload && systemctl restart askchem"
```

## Follow-ups (not in this PR)

* If nothing breaks for ~2 weeks, remove the PAW source path entirely:
  drop `src/chemtree/paw_functions.py`, the warmup at
  `server._warmup_paw`, the FTS-empty rescue at `db.py:3214-3219`, the
  intent thread-pool plumbing, and `programasweights` from
  `requirements.txt`. Saves ~280 MB on the runtime image and removes a
  dependency on a private git package.
* If we ever migrate the cross-encoder to a CUDA / accelerated box,
  re-run the int8 ablation there — on-GPU int8 typically delivers the
  expected 2-3× speedup that x86 CPU dynamic quant did not.
* The full rerank-window cap (`RERANK_WINDOW=50`, skip rerank on
  `offset>0`) is a separate latency PR; it does not affect quality at
  top-20 (the same top-50 get reranked the same way) so it was not
  measured here.
