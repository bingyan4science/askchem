"""Workstream B: pre-compute paper intelligence so the UI is instant.

For every placed (view, host node, paper) we generate three analyses in ONE LLM
call (shared context) and store them in the `paper_analysis` table:

  * advisor       - grounded positioning questions (advisor.py logic)
  * critique      - are the paper's claims supported by its own evidence; is the
                    reasoning logically sound
  * contribution  - how it instantiates/extends/challenges its parent principle
                    and differs from its branch neighbors

Two execution paths:
  run      live parallel calls, write rows directly (fills the table now)
  prepare/submit/poll/collect   Gemini BATCH API (scale path; ~50% cheaper)

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/precompute_analysis.py run --workers 10 --only-missing
    # or, at scale:
    python3 living_taxonomy/precompute_analysis.py prepare
    python3 living_taxonomy/precompute_analysis.py submit
    python3 living_taxonomy/precompute_analysis.py poll
    python3 living_taxonomy/precompute_analysis.py collect
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
from concurrent.futures import ThreadPoolExecutor, as_completed

from askchem import advisor, db, ltree
from askchem.advisor import (ANALYSIS_SYS as _SYS, build_analysis_user as _build_user,
                             gather_context as _context, split_analysis as _split)
from classify_papers import GATEWAY, MODEL, PROVIDER, _curl_json

OUT = _HERE / "output"
BATCH = OUT / "batch_analysis"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _store(lock, view_id, node_id, doi, advisor_j, critique_j, contribution_j):
    with lock, db.get_conn(readonly=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO paper_analysis(view_id,node_id,doi,advisor_json,"
            "critique_json,contribution_json,generated_at) VALUES (?,?,?,?,?,?,?)",
            (view_id, node_id, doi, advisor_j, critique_j, contribution_j, _now()))
        conn.commit()


def _targets(views=None, only_missing=False, limit=0, seeds_only=False):
    with db.get_conn() as c:
        q = "SELECT DISTINCT view_id,node_id,doi FROM taxonomy_leaves"
        rows = c.execute(q).fetchall()
        tgt = [(r["view_id"], r["node_id"], r["doi"]) for r in rows]
        if views:
            tgt = [t for t in tgt if t[0] in views]
        if only_missing:
            done = {(r["view_id"], r["node_id"], r["doi"]) for r in
                    c.execute("SELECT view_id,node_id,doi FROM paper_analysis").fetchall()}
            tgt = [t for t in tgt if t not in done]
    if seeds_only:
        # At scale, precompute only the seed paper of each node (bounded to
        # ~#nodes-with-papers); everything else is served on-demand + cached.
        nodes = sorted({(v, n) for v, n, _ in tgt})
        seeds = set()
        for v, n in nodes:
            try:
                sd = ltree.influence(v, n, limit=1).get("seed_doi")
            except Exception:
                sd = None
            if sd:
                seeds.add((v, n, sd))
        tgt = [t for t in tgt if t in seeds]
    if limit:
        tgt = tgt[:limit]
    return tgt


# ── live parallel path ────────────────────────────────────────────────────────

def cmd_run(args):
    tgt = _targets(args.views, args.only_missing, args.limit,
                   seeds_only=getattr(args, "seeds_only", False))
    print(f"[precompute] {len(tgt)} targets; workers={args.workers}", flush=True)
    lock = threading.Lock()
    done = {"n": 0, "err": 0}
    t0 = time.time()

    def work(t):
        view_id, node_id, doi = t
        paper, node, branch, siblings = _context(view_id, node_id, doi)
        node["_title"] = paper.get("title")
        user = _build_user(paper, node, branch, siblings, doi)
        try:
            parsed = advisor._parse_json(advisor._gemini_chat(_SYS, user, max_time=120))
            a, cr, co = _split(view_id, node_id, doi, branch, node, siblings, parsed)
            _store(lock, view_id, node_id, doi, a, cr, co)
        except Exception as e:
            return ("err", str(e)[:80])
        return ("ok", None)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, t) for t in tgt]):
            status, msg = f.result()
            done["n"] += 1
            if status == "err":
                done["err"] += 1
            if done["n"] % 25 == 0 or done["n"] == len(tgt):
                rate = done["n"] / max(time.time() - t0, 1)
                print(f"[precompute] {done['n']}/{len(tgt)} "
                      f"(err {done['err']}, {rate:.1f}/s)", flush=True)
    print(f"[precompute] done: {done['n']} rows, {done['err']} errors")


# ── batch path (scale) ─────────────────────────────────────────────────────────

def cmd_prepare(args):
    tgt = _targets(args.views, args.only_missing, args.limit,
                   seeds_only=getattr(args, "seeds_only", False))
    BATCH.mkdir(parents=True, exist_ok=True)
    reqs, meta = [], {}
    for k, (view_id, node_id, doi) in enumerate(tgt):
        paper, node, branch, siblings = _context(view_id, node_id, doi)
        node["_title"] = paper.get("title")
        cid = f"pa{k}"
        reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                     "body": {"model": MODEL, "temperature": 0.2,
                              "messages": [{"role": "system", "content": _SYS},
                                           {"role": "user", "content": _build_user(
                                               paper, node, branch, siblings, doi)}],
                              "max_completion_tokens": 2048,
                              "response_format": {"type": "json_object"}}})
        meta[cid] = {"view_id": view_id, "node_id": node_id, "doi": doi,
                     "branch": branch, "title": paper.get("title"),
                     "proposed": bool(node.get("proposed")),
                     "siblings": [{"doi": s["doi"], "title": s["title"]} for s in siblings]}
    (BATCH / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in reqs) + "\n")
    (BATCH / "meta.json").write_text(json.dumps(meta))
    print(f"[precompute-batch] prepared {len(reqs)} requests -> {BATCH}/requests.jsonl")


def cmd_submit(args):
    up = _curl_json("POST", "/files", form_fields={
        "purpose": "batch", "provider_file_name": "ltree_analysis.jsonl",
        "provider_model": MODEL}, file_path=str(BATCH / "requests.jsonl"), max_time=600)
    fid = up.get("id")
    if not fid:
        print("[precompute-batch] upload failed:", str(up)[:200]); return
    br = _curl_json("POST", "/batches", data={
        "input_file_id": fid, "endpoint": "/v1/chat/completions",
        "completion_window": "24h", "model": MODEL})
    (BATCH / "tracker.json").write_text(json.dumps(
        {"file_id": fid, "batch_id": br.get("id"), "status": br.get("status")}, indent=2))
    print(f"[precompute-batch] submitted batch={br.get('id')} status={br.get('status')}")


def cmd_poll(args):
    t = json.loads((BATCH / "tracker.json").read_text())
    r = _curl_json("GET", f"/batches/{t['batch_id']}")
    print(f"[precompute-batch] status={r.get('status')} counts={r.get('request_counts')}")


def cmd_collect(args):
    t = json.loads((BATCH / "tracker.json").read_text())
    api = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-s", "--max-time", "600", "-X", "GET",
           "-H", f"x-portkey-api-key: {api}", "-H", f"x-portkey-provider: {PROVIDER}",
           f"{GATEWAY}/batches/{t['batch_id']}/output"]
    raw = subprocess.run(cmd, capture_output=True, text=True, timeout=660).stdout
    (BATCH / "output.jsonl").write_text(raw)
    meta = json.loads((BATCH / "meta.json").read_text())
    lock = threading.Lock()
    n = 0
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            m = meta.get(item.get("custom_id", ""))
            if not m:
                continue
            body = item.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            parsed = advisor._parse_json(text)
            node = {"proposed": m["proposed"], "_title": m["title"]}
            a, cr, co = _split(m["view_id"], m["node_id"], m["doi"], m["branch"],
                               node, m["siblings"], parsed)
            _store(lock, m["view_id"], m["node_id"], m["doi"], a, cr, co)
            n += 1
        except Exception:
            pass
    print(f"[precompute-batch] stored {n} rows into paper_analysis")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--views", nargs="*", default=None)
        p.add_argument("--only-missing", action="store_true")
        p.add_argument("--seeds-only", action="store_true",
                       help="precompute only each node's seed paper (bounded); "
                            "the rest is served on-demand + cached")
        p.add_argument("--limit", type=int, default=0)
        if name == "run":
            p.add_argument("--workers", type=int, default=10)
    for name in ("submit", "poll", "collect"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"run": cmd_run, "prepare": cmd_prepare, "submit": cmd_submit,
     "poll": cmd_poll, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
