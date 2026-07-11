#!/usr/bin/env bash
# Drive build_embeddings_v2.py in single-shard subprocess invocations.
#
# Each shard runs in its own Python process so MPS state is reset
# cleanly on exit (a long-running process hangs after the first shard
# on Apple-MPS, even with torch.mps.empty_cache()).
#
# Usage:
#     bash scripts/run_gamma2_overnight.sh [<n_max_shards>]
#
# n_max_shards defaults to 0 (run until all claims are encoded).
# Output is appended to /tmp/gamma2_encode.log.

cd "$(dirname "$0")/.."

CHUNK_SIZE=${CHUNK_SIZE:-5000}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_SHARDS=${1:-0}
LOG=${LOG:-/tmp/gamma2_encode.log}

count_shards() { ls data/claim_embeddings_v2_shards/shard_*.npz 2>/dev/null | wc -l | tr -d ' '; }

i=0
prev=$(count_shards)
echo "[$(date +%T)] driver start  shards=$prev  max=$MAX_SHARDS" >> "$LOG"

while true; do
    if [[ "$MAX_SHARDS" != "0" && "$i" -ge "$MAX_SHARDS" ]]; then
        echo "[$(date +%T)] driver hit MAX_SHARDS=$MAX_SHARDS, exiting" >> "$LOG"
        break
    fi

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python3 -u scripts/build_embeddings_v2.py encode \
            --chunk-size "$CHUNK_SIZE" \
            --batch-size "$BATCH_SIZE" \
            --max-chunks 1 \
        >> "$LOG" 2>&1
    rc=$?

    cur=$(count_shards)
    echo "[$(date +%T)] driver iter=$i rc=$rc shards=$prev->$cur" >> "$LOG"

    if [[ "$cur" -le "$prev" ]]; then
        echo "[$(date +%T)] driver no progress, exiting" >> "$LOG"
        break
    fi
    prev=$cur
    i=$((i + 1))
done

echo "[$(date +%T)] driver done  total_shards=$(count_shards)" >> "$LOG"
