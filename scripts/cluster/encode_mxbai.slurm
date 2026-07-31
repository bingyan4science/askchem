#!/bin/bash
#SBATCH --job-name=mxbai-encode
#SBATCH --account=torch_pr_375_general
#SBATCH --partition=h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/by2192/embed_job/logs/mxbai-%j.out
#SBATCH --error=/scratch/by2192/embed_job/logs/mxbai-%j.err

set -euo pipefail

JOB_DIR=/scratch/by2192/embed_job
PYTHON=/scratch/by2192/conda_envs/torch_env/bin/python

# Force unbuffered Python so progress lands in the log immediately.
export PYTHONUNBUFFERED=1
export STDBUF_O=L STDBUF_E=L

echo "[$(date +%T)] === node ==="
hostname
nvidia-smi -L
echo
echo "[$(date +%T)] === env probe ==="
$PYTHON -u --version
echo "[$(date +%T)] importing torch (first load can take 60-180 s on cold scratch)…"
$PYTHON -u -c "import sys, time; t=time.monotonic();
import torch;
print(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()} '
      f'({torch.version.cuda})  load={time.monotonic()-t:.1f}s', flush=True)
print(f'  device: {torch.cuda.get_device_name(0)}', flush=True)"
echo "[$(date +%T)] importing transformers + numpy…"
$PYTHON -u -c "import time; t=time.monotonic();
import transformers, numpy
print(f'  transformers {transformers.__version__}  numpy {numpy.__version__}  '
      f'load={time.monotonic()-t:.1f}s', flush=True)"

cd $JOB_DIR

echo "[$(date +%T)] === starting encode ==="
# IMPORTANT: --pooling cls. mxbai-embed-large-v1's sentence-transformers
# config selects CLS pooling and the deployed query path uses
# sentence-transformers, so docs must use the same recipe. The May-5
# run accidentally used mean pooling and the resulting vectors were
# ~0.04 cos off the query subspace, halving Recall@20 at full corpus
# scale.
$PYTHON -u code/encode_mxbai_cluster.py \
    --jsonl   data/claims.jsonl \
    --out-npz out/embeddings.v2.npz \
    --hf-cache hf_cache \
    --pooling cls \
    --batch-size 192 \
    --max-len 384 \
    --dtype bf16 \
    --print-every 50000

echo "[$(date +%T)] === done ==="
ls -la out/
