#!/usr/bin/env bash
# Phase 3 A/B for the PAW finetune (paw-ft-bs48-20260522).
#
# Runs three configs of db.search_claims against the 80-probe eval set,
# scores each via scripts/eval_metrics.py, and emits a side-by-side diff.
# Each config runs in its own python process so the lazy-singleton PAW
# program-ID overrides do not bleed between runs.
#
# Configs:
#   A: ft-A-baseline    — production today (PAW off, no rewrites)
#   B: ft-B-std-rewrites — standard-compiler PAW with the new rewrite
#                          wiring enabled
#   C: ft-C-ft-rewrites  — finetuned PAW (paw-ft-bs48-20260522) with
#                          rewrites enabled
#
# Usage:
#   bash scripts/run_paw_ft_ab.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=.venv-benchmark/bin/python

# Match production retrieval config (see docs/search-pipeline.md
# "Currently shipped on prod"). PAW + rewrites flags are set per-config
# below; everything here is shared.
export CHEMTREE_RETRIEVER_VERSION=v2
export CHEMTREE_V2_DIM=256
export CHEMTREE_FAISS_MMAP=1
export OMP_NUM_THREADS=4
export KMP_DUPLICATE_LIB_OK=TRUE
export CHEMTREE_DB=chemtree.db
export PYTHONPATH=src
export CHEMTREE_RERANK_ENABLED=1
export CHEMTREE_RERANK_WINDOW=30
export CHEMTREE_RERANK_MAX_LEN=128
export CHEMTREE_DISABLE_PRF=1
export CHEMTREE_DISABLE_TREE_RERANK=1

run_one () {
  local label="$1"; shift
  echo
  echo "================================================================"
  echo "Running: $label"
  echo "Extra env: $*"
  echo "================================================================"
  # shellcheck disable=SC2086
  env "$@" $PYTHON scripts/eval_search_live.py --label "$label" --top 20
  $PYTHON scripts/eval_metrics.py --run "$label" \
    --rankings "data/eval/runs/${label}.rankings.jsonl"
}

# A — baseline: PAW fully disabled, no rewrites.
run_one "ft-A-baseline" \
  CHEMTREE_DISABLE_PAW=1 \
  CHEMTREE_PAW_REWRITES=0

# B — standard-compiler PAW, rewrites wired.
run_one "ft-B-std-rewrites" \
  CHEMTREE_DISABLE_PAW=0 \
  CHEMTREE_PAW_REWRITES=1

# C — finetuned PAW, rewrites wired.
run_one "ft-C-ft-rewrites" \
  CHEMTREE_DISABLE_PAW=0 \
  CHEMTREE_PAW_REWRITES=1 \
  CHEMTREE_PAW_FT_IDS=data/paw_ft_program_ids.json

echo
echo "================================================================"
echo "Diffs"
echo "================================================================"
$PYTHON scripts/eval_metrics.py --compare ft-A-baseline ft-B-std-rewrites
$PYTHON scripts/eval_metrics.py --compare ft-A-baseline ft-C-ft-rewrites
$PYTHON scripts/eval_metrics.py --compare ft-B-std-rewrites ft-C-ft-rewrites
