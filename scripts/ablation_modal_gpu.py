"""Run the 4 Modal-GPU A/B configs back-to-back and score them.

Phase 3 of the Modal GPU plan: compare CPU rerank (current prod) against
three remote-GPU variants on the canonical 80-probe eval set. Same
hardware everywhere (this script runs locally on the M-series Mac), so
absolute latency numbers are 2-3x faster than DO prod — but the
relative deltas transfer cleanly. nDCG is device-invariant.

Configs (label/env summary):

  M1-baseline    local MPS rerank + MPS embed, window=30   (matches May 15 composite)
  M2-gpu-rk30    remote rerank, MPS embed, window=30        (apples-to-apples vs M1)
  M3-gpu-rk100   remote rerank, MPS embed, window=100       (quality at no latency cost)
  M4-gpu-full    remote rerank + remote embed, window=100   (full hybrid)

The search-result LRU cache (CHEMTREE_SEARCH_CACHE) is *disabled* for
every run so we measure the cold path rather than collapsing to 0 ms on
the second probe.

Each remote config sends a single warmup request before the timed run
so Modal's first-probe cold-start doesn't poison the numbers (the
prior cold-start would otherwise add ~10-20 s to the first probe and
fail-over to local). After the run we re-score via eval_metrics.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "eval" / "runs"
TOKEN_FILE = Path("/tmp/askchem_modal_token.txt")

EMBED_URL = "https://by2192--askchem-search-gpu-searchgpu-embed-v1.modal.run"
RERANK_URL = "https://by2192--askchem-search-gpu-searchgpu-rerank-v1.modal.run"

# Shared env across every run — match the current prod composite from
# the May-15 ablation report.
SHARED_ENV = {
    "CHEMTREE_RETRIEVER_VERSION": "v2",
    "CHEMTREE_V2_DIM": "256",
    "CHEMTREE_RERANK_ENABLED": "1",
    "CHEMTREE_DISABLE_PRF": "1",
    "CHEMTREE_DISABLE_TREE_RERANK": "1",
    "CHEMTREE_DISABLE_PAW": "1",
    # Critically — DISABLE the result LRU cache so we always measure the
    # cold path. Otherwise the second probe with the same query would
    # collapse to 0 ms and we'd lose the rerank latency we are trying
    # to measure.
    "CHEMTREE_SEARCH_CACHE": "0",
    # Emit per-stage timing on stderr so the post-hoc aggregator can show
    # exactly how much the rerank and embed_query stages changed.
    "CHEMTREE_SEARCH_PROFILE": "1",
    "CHEMTREE_FAISS_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "PYTHONPATH": str(REPO_ROOT / "src"),
    # Generous remote timeout so cold-start (which we pre-warm anyway)
    # doesn't trip the fall-back.
    "CHEMTREE_REMOTE_TIMEOUT_S": "20",
}


def _remote_env() -> dict[str, str]:
    tok = TOKEN_FILE.read_text().strip()
    return {
        "CHEMTREE_REMOTE_AUTH_TOKEN": tok,
        "CHEMTREE_REMOTE_EMBED_URL": "",  # filled per-config
        "CHEMTREE_REMOTE_RERANK_URL": "",
    }


CONFIGS = [
    {
        "label": "modal-m1-baseline",
        "env": {"CHEMTREE_RERANK_WINDOW": "30"},
        "remote_rerank": False,
        "remote_embed": False,
    },
    {
        "label": "modal-m2-gpu-rk30",
        "env": {"CHEMTREE_RERANK_WINDOW": "30"},
        "remote_rerank": True,
        "remote_embed": False,
    },
    {
        "label": "modal-m3-gpu-rk100",
        "env": {"CHEMTREE_RERANK_WINDOW": "100"},
        "remote_rerank": True,
        "remote_embed": False,
    },
    {
        "label": "modal-m4-gpu-full",
        "env": {"CHEMTREE_RERANK_WINDOW": "100"},
        "remote_rerank": True,
        "remote_embed": True,
    },
]


def warm_modal(rerank: bool, embed: bool, token: str) -> None:
    """Send one tiny request to each enabled remote endpoint."""
    import requests
    headers = {"X-Auth-Token": token, "Content-Type": "application/json"}
    if embed:
        t0 = time.monotonic()
        r = requests.post(EMBED_URL, headers=headers,
                          json={"queries": ["warm"]}, timeout=60)
        r.raise_for_status()
        print(f"  [warmup] embed: {1000*(time.monotonic()-t0):.0f} ms")
    if rerank:
        t0 = time.monotonic()
        r = requests.post(RERANK_URL, headers=headers,
                          json={"query": "warm", "texts": ["warm passage"]},
                          timeout=60)
        r.raise_for_status()
        print(f"  [warmup] rerank: {1000*(time.monotonic()-t0):.0f} ms")


def run_config(cfg: dict) -> dict:
    label = cfg["label"]
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    env = os.environ.copy()
    env.update(SHARED_ENV)
    env.update(cfg["env"])
    if cfg["remote_rerank"] or cfg["remote_embed"]:
        env.update(_remote_env())
        if cfg["remote_rerank"]:
            env["CHEMTREE_REMOTE_RERANK_URL"] = RERANK_URL
        if cfg["remote_embed"]:
            env["CHEMTREE_REMOTE_EMBED_URL"] = EMBED_URL
        token = TOKEN_FILE.read_text().strip()
        print("warming Modal endpoints…")
        warm_modal(cfg["remote_rerank"], cfg["remote_embed"], token)

    print(f"env:")
    for k in sorted(("CHEMTREE_RERANK_WINDOW",
                     "CHEMTREE_REMOTE_RERANK_URL",
                     "CHEMTREE_REMOTE_EMBED_URL")):
        print(f"  {k}={env.get(k, '<unset>')[:60]}")

    # Capture stderr to a per-config file so we can later aggregate the
    # [search_profile] lines into a per-stage table.
    stderr_path = RUNS_DIR / f"{label}.stderr.log"
    t_run = time.monotonic()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_search_live.py"),
        "--label", label,
        "--top", "20",
    ]
    with stderr_path.open("w") as stderr_fh:
        res = subprocess.run(
            cmd, env=env, capture_output=False,
            stdout=subprocess.PIPE, stderr=stderr_fh, text=True,
        )
    wall = time.monotonic() - t_run
    if res.returncode != 0:
        print(f"STDERR captured at {stderr_path}; tail:")
        try:
            print(stderr_path.read_text()[-2000:])
        except Exception:
            pass
        raise RuntimeError(f"{label} failed (rc={res.returncode})")
    # Echo the harness's own latency summary line
    for line in res.stdout.splitlines():
        if line.startswith("latency:") or line.startswith("[rss_end]"):
            print(f"  {line}")
    print(f"  wall: {wall:.0f} s")

    # Score
    rk_path = RUNS_DIR / f"{label}.rankings.jsonl"
    score_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_metrics.py"),
        "--run", label,
        "--rankings", str(rk_path),
    ]
    sc = subprocess.run(score_cmd, env=env, capture_output=True, text=True)
    if sc.returncode != 0:
        print("score STDERR tail:", sc.stderr[-1500:])
        raise RuntimeError(f"{label} scoring failed (rc={sc.returncode})")
    scored_path = RUNS_DIR / f"{label}.scored.json"
    scored = json.loads(scored_path.read_text())
    agg = scored["aggregate"]
    print(f"  nDCG@10={agg['ndcg@10']:.4f}  "
          f"MRR@20={agg['mrr@20']:.4f}  "
          f"recall@10={agg.get('recall@10', float('nan')):.4f}  "
          f"recall@20={agg['recall@20']:.4f}")
    return {
        "label": label,
        "wall_s": wall,
        "stdout": res.stdout,
        "scored": agg,
    }


def main() -> None:
    print(f"Token file: {TOKEN_FILE}  (exists={TOKEN_FILE.exists()})")
    print(f"Runs dir:   {RUNS_DIR}")

    summaries = []
    for cfg in CONFIGS:
        summaries.append(run_config(cfg))

    print("\n" + "=" * 60)
    print(f"{'config':<22s} {'nDCG@10':>9s} {'MRR@20':>9s} "
          f"{'R@10':>7s} {'R@20':>7s} {'wall_s':>8s}")
    print("-" * 64)
    for s in summaries:
        sc = s["scored"]
        print(f"{s['label']:<22s} "
              f"{sc['ndcg@10']:>9.4f} "
              f"{sc['mrr@20']:>9.4f} "
              f"{sc.get('recall@10', 0):>7.4f} "
              f"{sc['recall@20']:>7.4f} "
              f"{s['wall_s']:>8.0f}")

    out = RUNS_DIR / "modal-gpu-ab-summary.json"
    out.write_text(json.dumps([{**s, "stdout": s["stdout"][-3000:]}
                                for s in summaries], indent=2))
    print(f"\nwrote summary to {out}")


if __name__ == "__main__":
    main()
