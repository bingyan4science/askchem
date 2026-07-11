#!/usr/bin/env bash
# Phase 1 attribution sweep (May-29).
#
# Runs the instrumented attribution harness under four wiring configs:
#
#   W0  baseline (PAW off)
#   W1  current PAW-on: V3_ft on FTS only
#   W2  PAW on FTS AND rerank input
#   W3  W2 with rerank window relaxed to 50
#
# Each writes data/eval/runs/attribution_<label>.jsonl. After all four,
# also runs the existing eval_metrics.py to compute nDCG@10 deltas, then
# the eval_attribution_summary.py aggregator.
#
# Usage: bash scripts/run_attribution_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=.venv-benchmark/bin/python

# Production retrieval config (matches docs/search-pipeline.md "currently
# shipped on prod"). Per-config knobs are layered on top.
export CHEMTREE_RETRIEVER_VERSION=v2
export CHEMTREE_V2_DIM=256
export CHEMTREE_FAISS_MMAP=1
export OMP_NUM_THREADS=4
export KMP_DUPLICATE_LIB_OK=TRUE
export CHEMTREE_DB=chemtree.db
export PYTHONPATH=src
export CHEMTREE_RERANK_ENABLED=1
export CHEMTREE_RERANK_MAX_LEN=128
export CHEMTREE_DISABLE_PRF=1
export CHEMTREE_DISABLE_TREE_RERANK=1
export GGML_NO_METAL=1

run_attr () {
  local label="$1"; shift
  echo
  echo "================================================================"
  echo "Attribution run: $label"
  echo "Extra env: $*"
  echo "================================================================"
  # shellcheck disable=SC2086
  env "$@" $PYTHON scripts/eval_search_attribution.py --label "$label" --top 20
}

# W0: prod baseline (PAW off, no rewrites, rerank window 30).
run_attr "W0_baseline" \
  CHEMTREE_DISABLE_PAW=1 \
  CHEMTREE_PAW_REWRITES=0 \
  CHEMTREE_PAW_REWRITES_RERANK=0 \
  CHEMTREE_RERANK_WINDOW=30

# W1: PAW on, expand only feeds FTS (current May-23 wiring).
run_attr "W1_paw_fts" \
  CHEMTREE_DISABLE_PAW=0 \
  CHEMTREE_PAW_REWRITES=1 \
  CHEMTREE_PAW_REWRITES_RERANK=0 \
  CHEMTREE_PAW_FT_IDS=data/paw_ft_program_ids.json \
  CHEMTREE_RERANK_WINDOW=30

# W2: PAW on, expand feeds both FTS and rerank input.
run_attr "W2_paw_rerank" \
  CHEMTREE_DISABLE_PAW=0 \
  CHEMTREE_PAW_REWRITES=1 \
  CHEMTREE_PAW_REWRITES_RERANK=1 \
  CHEMTREE_PAW_FT_IDS=data/paw_ft_program_ids.json \
  CHEMTREE_RERANK_WINDOW=30

# W3: W2 with rerank window relaxed to 50.
run_attr "W3_paw_rerank_w50" \
  CHEMTREE_DISABLE_PAW=0 \
  CHEMTREE_PAW_REWRITES=1 \
  CHEMTREE_PAW_REWRITES_RERANK=1 \
  CHEMTREE_PAW_FT_IDS=data/paw_ft_program_ids.json \
  CHEMTREE_RERANK_WINDOW=50

echo
echo "================================================================"
echo "Score each run with eval_metrics.py (reads ranked_claim_ids from"
echo "the same attribution_<label>.jsonl)"
echo "================================================================"
for label in W0_baseline W1_paw_fts W2_paw_rerank W3_paw_rerank_w50; do
  attr_path="data/eval/runs/attribution_${label}.jsonl"
  $PYTHON scripts/eval_metrics.py --run "attr_${label}" --rankings "$attr_path"
done

echo
echo "================================================================"
echo "Attribution summary across all 4 configs"
echo "================================================================"
$PYTHON scripts/eval_attribution_summary.py --labels W0_baseline W1_paw_fts W2_paw_rerank W3_paw_rerank_w50
