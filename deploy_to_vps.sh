#!/usr/bin/env bash
# One-shot deploy script for askchem.org.
#
# Run this on your laptop after the HuggingFace upload of chemtree.db
# (and, for δ3, the v2 retrieval artefacts) has finished. The HF
# upload is driven by ``src/upload_to_hf.py`` — pass
# ``--include-v2-embeddings runtime`` to ship the FAISS + claim-id
# sidecar alongside the database.
#
# What it does:
#   1. SSH to the VPS, git pull the latest code, install runtime deps
#      (requirements.txt pins torch / transformers / faiss-cpu, see δ3).
#   2. Download fresh chemtree.db from HuggingFace (VPS bandwidth).
#   3. (δ3) Download the v2 retrieval artefacts from HuggingFace into
#      ``$APP_DIR/data/``. The 8 GB droplet ships the Matryoshka-
#      truncated 256-d FAISS (~2.2 GB) and the claim-id sidecar
#      (~598 MB); the npz and 1024-d FAISS only land on a larger
#      droplet when ``--with-v2-1024`` / ``--with-v2-npz`` is passed.
#   4. Install/refresh the systemd drop-in
#      ``/etc/systemd/system/askchem.service.d/override.conf`` from
#      the copy under ``deploy/askchem.service.d/override.conf`` so
#      the service has ``CHEMTREE_RETRIEVER_VERSION=v2`` and
#      ``CHEMTREE_V2_DIM=256``.
#   5. Stop askchem.service, atomic-mv the new DB + v2 artefacts into
#      place, create the ``..._256.claim_ids.npy`` symlink onto the
#      shared sidecar, reseed the ``contradictions`` table from
#      ``deploy/contradictions_seed_v1.json`` (the HF-distributed DB
#      carries an empty table; the seed is the source of truth),
#      daemon-reload, start.
#   6. Smoke-test the public URL: /api/stats + /api/search?q=suzuki...
#
# Idempotent: safe to re-run.
#
# Flags:
#   --skip-db          do not re-download chemtree.db (useful when only
#                      the v2 artefacts changed)
#   --skip-v2          do not deploy v2 retrieval artefacts (legacy v1
#                      deploy)
#   --with-v2-1024     additionally fetch the 1024-d FAISS (9.6 GB) —
#                      only safe on ≥ 16 GB droplets
#   --with-v2-npz      additionally fetch claim_embeddings.v2.npz
#                      (10 GB) for reproducibility; not needed at
#                      runtime
#   --skip-restart     install everything but leave the running service
#                      untouched (manual restart later)

set -euo pipefail

VPS="root@YOUR_VPS_HOST"
APP_DIR="/opt/askchem"
DATA_DIR="$APP_DIR/data"
# DB file renamed chemtree.db -> askchem.db (consistent with the askchem package
# / bing-yan/askchem HF dataset). LEGACY_DB is migrated in-place at swap time.
DB_PATH="$APP_DIR/askchem.db"
LEGACY_DB="$APP_DIR/chemtree.db"
NEW_DB="$DB_PATH.new"
HF_REPO="bing-yan/askchem"
DB_FILE="askchem.db"
# Pip + hf CLI live in the venv on the VPS; this matches the
# ExecStart path in /etc/systemd/system/askchem.service. Keep in sync.
VPS_PIP="$APP_DIR/venv/bin/pip"
VPS_HF="$APP_DIR/venv/bin/hf"

SKIP_DB=0
SKIP_V2=0
WITH_V2_1024=0
WITH_V2_NPZ=0
SKIP_RESTART=0
for arg in "$@"; do
  case "$arg" in
    --skip-db) SKIP_DB=1 ;;
    --skip-v2) SKIP_V2=1 ;;
    --with-v2-1024) WITH_V2_1024=1 ;;
    --with-v2-npz) WITH_V2_NPZ=1 ;;
    --skip-restart) SKIP_RESTART=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# Default 256-d-only runtime set. Stays in sync with
# src/upload_to_hf.py V2_ARTEFACTS_RUNTIME (subset thereof — we never
# pull the 1024-d index onto an 8 GB box).
V2_RUNTIME_FILES=("claim_embeddings.v2_256.faiss" "claim_embeddings.v2.claim_ids.npy")
EXTRA=()
[ "$WITH_V2_1024" = "1" ] && EXTRA+=("claim_embeddings.v2.faiss")
[ "$WITH_V2_NPZ"  = "1" ] && EXTRA+=("claim_embeddings.v2.npz")

echo "==> [1/6] git pull + ensure runtime deps on VPS"
ssh "$VPS" "cd $APP_DIR && git fetch origin main && git reset --hard origin/main && git log -1 --oneline"
# Refresh Python deps via the venv pip (pinned in requirements.txt:
# torch / transformers / faiss-cpu / sentence-transformers). The hf
# CLI ships with huggingface_hub so this also gets us /opt/askchem/
# venv/bin/hf if it ever goes missing.
ssh "$VPS" "cd $APP_DIR && \
  if [ -x $VPS_PIP ]; then \
    $VPS_PIP install --quiet --upgrade --requirement requirements.txt; \
    echo '  installed via venv pip'; \
  else \
    pip install --quiet --upgrade --requirement requirements.txt; \
    echo '  installed via system pip (no venv found at $VPS_PIP)'; \
  fi"
ssh "$VPS" "test -x $VPS_HF || { echo 'ERROR: $VPS_HF missing after pip install' >&2; exit 1; }"

if [ "$SKIP_DB" = "0" ]; then
  echo
  echo "==> [2/6] download fresh $DB_FILE from HF (VPS bandwidth)"
  ssh "$VPS" "cd $APP_DIR && \
    $VPS_HF download $HF_REPO $DB_FILE --type dataset --local-dir . && \
    test -f $DB_FILE && \
    mv $DB_FILE $NEW_DB && \
    ls -lh $NEW_DB"
else
  echo
  echo "==> [2/6] SKIP $DB_FILE download (--skip-db)"
fi

if [ "$SKIP_V2" = "0" ]; then
  echo
  echo "==> [3/6] download v2 retrieval artefacts from HF"
  # bash 5.2 + `set -u` treats an unset/empty array as "unbound"; the
  # default expansion (${EXTRA[@]:-}) avoids the bomb when no
  # ``--with-v2-*`` flags were passed.
  FILES=("${V2_RUNTIME_FILES[@]}" "${EXTRA[@]:-}")
  # Strip the empty fallback element when no extras were requested.
  FILES=("${FILES[@]/#}")  # no-op for non-empty entries
  CLEANED=()
  for f in "${FILES[@]}"; do
    [ -n "$f" ] && CLEANED+=("$f")
  done
  FILES=("${CLEANED[@]}")
  REMOTE_PATHS=""
  for f in "${FILES[@]}"; do
    echo "    + embeddings_v2/$f"
    REMOTE_PATHS+=" embeddings_v2/$f"
  done
  # Stage into data/.v2-new/ so we can atomic-rename after everything
  # arrives — otherwise an interrupted download would leave a half-
  # baked FAISS file that load_embeddings would happily mmap and serve
  # garbage from. The new ``hf download`` CLI lays files out under
  # ``<local-dir>/embeddings_v2/<f>`` so we collapse that one level
  # back up before the swap.
  ssh "$VPS" "set -e; mkdir -p $DATA_DIR && cd $DATA_DIR && \
    rm -rf .v2-new && mkdir .v2-new && \
    $VPS_HF download $HF_REPO ${REMOTE_PATHS} --type dataset --local-dir .v2-new && \
    mv .v2-new/embeddings_v2/* .v2-new/ && \
    rmdir .v2-new/embeddings_v2 && \
    ls -lh .v2-new/"
else
  echo
  echo "==> [3/6] SKIP v2 retrieval artefact download (--skip-v2)"
fi

echo
echo "==> [4/6] install systemd drop-in (CHEMTREE_RETRIEVER_VERSION=v2 …)"
ssh "$VPS" "mkdir -p /etc/systemd/system/askchem.service.d"
scp deploy/askchem.service.d/override.conf \
    "$VPS:/etc/systemd/system/askchem.service.d/override.conf"
ssh "$VPS" "systemctl daemon-reload && \
  systemctl cat askchem.service | grep -E '^Environment=' | head -30"

if [ "$SKIP_RESTART" = "1" ]; then
  echo
  echo "==> [5/6] SKIP restart (--skip-restart); inspect with"
  echo "     ssh $VPS 'systemctl status askchem.service'"
  echo "==> [6/6] SKIP smoke-test"
  exit 0
fi

echo
echo "==> [5/6] atomic swap + restart (brief downtime ~5 s)"
ssh "$VPS" "
  set -e
  systemctl stop askchem.service

  # Migrate the legacy filename in place: chemtree.db -> askchem.db. Runs once;
  # after this the canonical askchem.db is present and this block is a no-op.
  if [ -f $LEGACY_DB ] && [ ! -f $DB_PATH ]; then
    mv $LEGACY_DB $DB_PATH
    [ -f ${LEGACY_DB}-wal ] && mv ${LEGACY_DB}-wal ${DB_PATH}-wal || true
    [ -f ${LEGACY_DB}-shm ] && mv ${LEGACY_DB}-shm ${DB_PATH}-shm || true
    echo 'DB migrated: chemtree.db -> askchem.db'
  fi

  if [ '$SKIP_DB' = '0' ] && [ -f $NEW_DB ]; then
    if [ -f $DB_PATH ]; then
      mv $DB_PATH ${DB_PATH}.bak.\$(date +%Y%m%d_%H%M%S)
    fi
    mv $NEW_DB $DB_PATH
    rm -f ${DB_PATH}-shm ${DB_PATH}-wal
    echo 'askchem.db: swapped'
  fi

  # Reseed the contradictions table from the in-repo seed JSON. The
  # HF-distributed chemtree.db carries an empty contradictions table
  # (the detection pipeline runs separately, off-corpus); the source
  # of truth lives at deploy/contradictions_seed_v1.json, brought in
  # by the git pull at step [1/6]. We reseed every deploy regardless
  # of --skip-db so a code-only deploy still heals a prod table that
  # was accidentally wiped or hand-edited.
  if [ -f $APP_DIR/deploy/reseed_contradictions.py ] && \
     [ -f $APP_DIR/deploy/contradictions_seed_v1.json ]; then
    $APP_DIR/venv/bin/python3 $APP_DIR/deploy/reseed_contradictions.py \
      --db $DB_PATH \
      --seed $APP_DIR/deploy/contradictions_seed_v1.json
  else
    echo 'contradictions seed not present; skipping reseed' >&2
  fi

  # Living-taxonomy tables (taxonomy_nodes/edges/leaves/meta + paper_analysis)
  # are now baked INTO askchem.db as published on HF (v16+), so no separate SQL
  # apply is needed. The old taxonomy_v15.sql step was removed on 2026-07-07:
  # re-applying it would DROP the tables and downgrade the tree to v15 (and wipe
  # paper_analysis). If a stale copy lingers on the box, remove it so it can
  # never clobber the shipped DB.
  rm -f $APP_DIR/deploy/taxonomy_v15.sql

  if [ '$SKIP_V2' = '0' ] && [ -d $DATA_DIR/.v2-new ]; then
    for f in $DATA_DIR/.v2-new/* ; do
      base=\$(basename \$f)
      if [ -f $DATA_DIR/\$base ]; then
        mv $DATA_DIR/\$base $DATA_DIR/\$base.bak.\$(date +%Y%m%d_%H%M%S)
      fi
      mv \$f $DATA_DIR/\$base
    done
    rmdir $DATA_DIR/.v2-new || true

    # Symlink the 256-d claim-id sidecar onto the shared sidecar.
    # embeddings_v2.load_embeddings does EMBEDDINGS_PATH.with_suffix(
    # '.claim_ids.npy'), which for the 256-d matrix resolves to
    # claim_embeddings.v2_256.claim_ids.npy. The bytes are identical
    # between 1024-d and 256-d (same row ordering), so we link the
    # 256-d view onto the one file we actually downloaded.
    cd $DATA_DIR
    ln -sf claim_embeddings.v2.claim_ids.npy claim_embeddings.v2_256.claim_ids.npy

    echo 'v2 artefacts: swapped'
    ls -lh claim_embeddings.v2* 2>/dev/null
  fi

  systemctl start askchem.service
  sleep 8
  systemctl status askchem.service --no-pager -l | head -20
"

echo
echo "==> [6/6] smoke-test public URL"
sleep 6
echo "  /api/stats:"
curl -s "https://askchem.org/api/stats" | python3 -c "import json,sys; s=json.load(sys.stdin); print(f'    total_claims: {s.get(\"total_claims\",\"?\"):,}'); print(f'    total_sources: {s.get(\"total_sources\",\"?\"):,}')"
echo "  /api/search (suzuki coupling, top 5, expect Suzuki-Miyaura reaction claims):"
curl -s "https://askchem.org/api/search?q=suzuki+coupling&limit=5" | \
  python3 -c "
import json, sys, re
d = json.load(sys.stdin)
hits = d.get('results', [])
suzuki_re = re.compile(r'suzuki[\\s_-]*miyaura|suzuki[- ]coupling', re.I)
n_match = 0
for i, h in enumerate(hits[:5], 1):
    txt = (h.get('claim_contextualized') or h.get('verbatim_quote') or
           h.get('claim') or '')
    is_suzuki = bool(suzuki_re.search(txt))
    if is_suzuki:
        n_match += 1
    print(f'    [{i}] {\"OK\" if is_suzuki else \"  \"} '
          f'{txt[:120].replace(chr(10), \" \")}')
print(f'    Suzuki-Miyaura coverage: {n_match}/{len(hits)}')
"
echo "  /api/search/graph (suzuki coupling, top 5 nodes):"
curl -s "https://askchem.org/api/search/graph?q=suzuki+coupling&limit=200&expand=one_hop" | \
  python3 -c "
import json,sys
d = json.load(sys.stdin)
edges = d.get('edges',[])
nodes = d.get('nodes',[])
connected = set()
for e in edges:
    connected.add(e['from']); connected.add(e['to'])
print(f'    nodes: {len(nodes)} (connected: {len(connected)}), edges: {len(edges)}')
"
echo
echo "==> Deploy complete. Visit https://askchem.org and try a search."
