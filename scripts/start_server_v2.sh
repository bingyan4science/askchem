#!/bin/bash
# Local v2 dev server: mxbai dense retrieval, cross-encoder rerank disabled.
# (Rerank adds ~50 s/query on Apple-MPS with OMP=1; dense-only hits 0.908
#  nDCG@10 against the 7,483-judgment label pool — already +0.087 over v1.)
#
# δ1 latency tuning (May 11):
#   - v1 stack is no longer pre-warmed; CHEMTREE_RETRIEVER_VERSION=v2
#     routes the dispatcher through embeddings_v2 alone.
#   - CHEMTREE_V2_DIM=256 selects the Matryoshka-truncated 256-d FAISS
#     (2.4 GB). The 1024-d index (9.6 GB) does not fit comfortably in
#     16 GB alongside the mxbai weights, causing page-eviction-induced
#     40-s queries on this Mac. mxbai is trained at every Matryoshka dim
#     down to 256; recall loss vs 1024 d is ~5 nDCG points (per the
#     bake-off doc) and we re-verify in δ2 with the live eval harness.
#     Prod (more RAM) stays on 1024 d; unset this var to revert.
#   - CHEMTREE_FAISS_MMAP=0 + CHEMTREE_FAISS_THREADS=1: the search is
#     memory-bandwidth-bound, single-thread is at parity with multi-
#     thread (~570 ms isolated), and resident-in-RAM avoids first-query
#     page-fault cliffs.
#   - OMP_NUM_THREADS=1 keeps sentence-transformers + FAISS off the same
#     OpenMP pool, sidestepping the Apple-MPS segfault we hit earlier.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

export CHEMTREE_RETRIEVER_VERSION=v2
export CHEMTREE_RERANK_ENABLED=0
export CHEMTREE_V2_DIM=256
export CHEMTREE_FAISS_MMAP=0
export CHEMTREE_FAISS_THREADS=1
export CHEMTREE_SEARCH_PROFILE=${CHEMTREE_SEARCH_PROFILE:-0}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src

exec python3 -m uvicorn askchem.server:app --host 127.0.0.1 --port 8420
