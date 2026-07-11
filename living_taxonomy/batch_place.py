"""Gemini BATCH placement for the living taxonomy (scale path, Milestone 2/3).

Same stateless insert logic as grow_onto_scaffold.py, but the LLM calls go
through the Gemini batch API (prepare JSONL -> submit -> poll -> collect) instead
of live calls - ~50% cheaper and far higher throughput for thousands/millions of
papers. Reuses _build_prompt / _LLM_SYS / apply_decisions from grow_onto_scaffold
and the batch helpers (_curl_json, /files, /batches) from classify_papers.

Subcommands:
    prepare   build requests.jsonl + meta.json (embeddings done locally now)
    submit    upload + create the batch job
    poll      check batch status
    collect   download outputs, apply decisions, persist grown_views.json + html

Usage:
    python3 living_taxonomy/batch_place.py prepare --papers 3000
    python3 living_taxonomy/batch_place.py submit
    python3 living_taxonomy/batch_place.py poll
    python3 living_taxonomy/batch_place.py collect
    python3 living_taxonomy/apply_to_db.py        # then load into chemtree.db
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
import build_viz
import corpus_embed
import grow_onto_scaffold as g
import pilot_data
import placement as pm
import view_layers as vl
from classify_papers import GATEWAY, MODEL, PROVIDER, _curl_json

OUT = g.OUT
BATCH = OUT / "batch"
SHARD_SIZE = 4000   # requests per batch job (stays under file/job limits)


def _leaf_vecs(lvs, corpus):
    """Vectors for a paper's leaves: reuse corpus vectors by claim_id, encode the
    rest with mxbai. Keeps host-shortlisting in the same embedding space."""
    vecs = [None] * len(lvs)
    miss_txt, miss_idx = [], []
    for i, lf in enumerate(lvs):
        v = corpus.get(lf["claim_id"])
        if v is None:
            miss_txt.append(lf["text"]); miss_idx.append(i)
        else:
            vecs[i] = v
    if miss_txt:
        mv = pm._embed(miss_txt, is_query=True)
        for j, i in enumerate(miss_idx):
            vecs[i] = mv[j]
    return np.vstack(vecs)


def cmd_prepare(args):
    nodes, views, host_nodes, _ = vl.build_all_views(use_cache=True)
    dois = g.sample_fulltext_papers(args.papers, args.seed, args.min_claims,
                                    exclude_placed=getattr(args, "exclude_placed", False))
    BATCH.mkdir(parents=True, exist_ok=True)
    for old in BATCH.glob("requests_*.jsonl"):
        old.unlink()
    reuse = corpus_embed.available()
    print(f"[batch] corpus-embedding reuse: {'ON' if reuse else 'OFF (mxbai encode)'}",
          file=sys.stderr)
    reqs, meta = [], {}
    for view in g.GROW_VIEWS:
        descs = g.host_descs(view, nodes)
        names = list(host_nodes[view])
        host_block = "\n".join(f"- {n}: {descs.get(n,'')}" for n in names)
        host_vecs = pm._embed([f"{n}. {descs.get(n,'')}" for n in names], is_query=False)
        leaves = g.clean_leaves(view, pilot_data.load_leaves(
            view, dois, max_leaves=args.per_paper * len(dois) * 4,
            per_paper=args.per_paper * 4))
        corpus = corpus_embed.vectors_for([lf["claim_id"] for lf in leaves]) if reuse else {}
        by_doi = {}
        for lf in leaves:
            by_doi.setdefault(lf["doi"], []).append(lf)
        for k, (doi, lvs) in enumerate(by_doi.items()):
            lvs = lvs[:args.per_paper]
            lvecs = _leaf_vecs(lvs, corpus)
            sims = lvecs @ host_vecs.T
            top1s = [names[int(np.argmax(sims[i]))] for i in range(len(lvs))]
            cid = f"{view}__{k}"
            reqs.append({
                "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": MODEL, "messages": [
                    {"role": "system", "content": g._LLM_SYS},
                    {"role": "user", "content": g._build_prompt(view, host_block, lvs)}],
                    "max_completion_tokens": 4096,
                    "response_format": {"type": "json_object"}}})
            meta[cid] = {"view": view, "top1s": top1s, "lvs": [
                {"claim_id": lf["claim_id"], "label": lf["label"], "text": lf["text"],
                 "doi": lf["doi"], "year": lf.get("year", 0)} for lf in lvs]}

    # shard into multiple request files (one Gemini batch job per shard)
    shards = []
    for i in range(0, len(reqs), SHARD_SIZE):
        fn = f"requests_{i // SHARD_SIZE:03d}.jsonl"
        (BATCH / fn).write_text("\n".join(json.dumps(r) for r in reqs[i:i + SHARD_SIZE]) + "\n")
        shards.append(fn)
    (BATCH / "meta.json").write_text(json.dumps(meta))
    (BATCH / "manifest.json").write_text(json.dumps(
        {"shards": shards, "n_requests": len(reqs), "papers": len(dois),
         "views": g.GROW_VIEWS}, indent=2))
    print(f"[batch] prepared {len(reqs)} requests ({len(dois)} papers x "
          f"{len(g.GROW_VIEWS)} views) in {len(shards)} shard(s) -> {BATCH}")


def cmd_submit(args):
    manifest = json.loads((BATCH / "manifest.json").read_text())
    tp = BATCH / "tracker.json"
    tracker = json.loads(tp.read_text()) if tp.exists() else {"jobs": []}
    done = {j["shard"] for j in tracker["jobs"]}
    for shard in manifest["shards"]:
        if shard in done:
            print(f"[batch] {shard} already submitted; skip"); continue
        up = _curl_json("POST", "/files", form_fields={
            "purpose": "batch", "provider_file_name": shard,
            "provider_model": MODEL}, file_path=str(BATCH / shard), max_time=600)
        fid = up.get("id")
        if not fid:
            print(f"[batch] upload failed for {shard}:", str(up)[:200]); continue
        br = _curl_json("POST", "/batches", data={
            "input_file_id": fid, "endpoint": "/v1/chat/completions",
            "completion_window": "24h", "model": MODEL})
        tracker["jobs"].append({"shard": shard, "file_id": fid,
                                "batch_id": br.get("id"), "status": br.get("status")})
        tp.write_text(json.dumps(tracker, indent=2))
        print(f"[batch] submitted {shard} batch={br.get('id')} status={br.get('status')}")


def cmd_poll(args):
    tracker = json.loads((BATCH / "tracker.json").read_text())
    done = 0
    for j in tracker["jobs"]:
        r = _curl_json("GET", f"/batches/{j['batch_id']}")
        j["status"] = r.get("status")
        if j["status"] in ("completed", "ended"):
            done += 1
        print(f"[batch] {j['shard']} batch={j['batch_id']} status={j['status']} "
              f"counts={r.get('request_counts')}")
    (BATCH / "tracker.json").write_text(json.dumps(tracker, indent=2))
    print(f"[batch] {done}/{len(tracker['jobs'])} shards complete")


def _download(batch_id):
    api = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-s", "--max-time", "600", "-X", "GET",
           "-H", f"x-portkey-api-key: {api}", "-H", f"x-portkey-provider: {PROVIDER}",
           f"{GATEWAY}/batches/{batch_id}/output"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=660).stdout


def _dump_leaves(views, sub):
    """Storage: strip the heavy leaf `full` text (recoverable from claims) to keep
    grown_views.json compact, and emit a flat leaves.jsonl artifact. Leaves stay in
    the tree so the cleanup chain keeps its per-branch leaf awareness."""
    count = [0]
    out = open(OUT / "leaves.jsonl", "w")

    def walk(view_id, node):
        for c in node.get("children", []) or []:
            if c.get("kind") == "leaf":
                out.write(json.dumps({"view_id": view_id, "claim_id": c.get("claim_id"),
                                      "doi": c.get("doi"), "label": c.get("name"),
                                      "score": c.get("score", 0)}) + "\n")
                c.pop("full", None)
                count[0] += 1
            else:
                walk(view_id, c)
    for vid, root in views.items():
        walk(vid, root)
    out.close()
    return count[0]


def cmd_collect(args):
    tracker = json.loads((BATCH / "tracker.json").read_text())
    meta = json.loads((BATCH / "meta.json").read_text())
    amaps = {}
    for j in tracker["jobs"]:
        raw = _download(j["batch_id"])
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                body = item.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                text = choices[0].get("message", {}).get("content", "") if choices else ""
                assign = json.loads(text).get("assign", []) if text.strip() else []
                amaps[item.get("custom_id", "")] = {x.get("i"): x for x in assign}
            except Exception:
                pass

    nodes, views, host_nodes, _ = vl.build_all_views(use_cache=True)
    by_view = {v: [] for v in g.GROW_VIEWS}
    for cid, m in meta.items():
        amap = amaps.get(cid, {})
        for i, lf in enumerate(m["lvs"]):
            a = amap.get(i, {})
            by_view[m["view"]].append({"leaf": lf, "host": a.get("host", ""),
                                       "propose": a.get("propose") or {},
                                       "top1": m["top1s"][i]})
    for view in g.GROW_VIEWS:
        g.apply_decisions(view, host_nodes[view], views[view], by_view[view])
        vl._count(views[view])
    sub = f"scaffold + batch placement ({len(meta)} paper-views)"
    _dump_leaves(views, sub)
    (OUT / "grown_views.json").write_text(json.dumps({"views": views, "subtitle": sub}))
    build_viz.render_html(views["by_reaction_type"], "chemistry living tree (with leaves)",
                          sub, OUT / "scaffold_multiview.html", views=views)
    n_resp = sum(1 for a in amaps.values() if a)
    print(f"[batch] collected {n_resp}/{len(meta)} paper-views; wrote grown_views.json "
          f"+ leaves.jsonl + html")
    print("[batch] next: consolidate -> cleanup chain -> apply_to_db")


def _parsed_ok(raw, got):
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            body = item.get("response", {}).get("body", {})
            ch = body.get("choices", [])
            text = ch[0].get("message", {}).get("content", "") if ch else ""
            if text.strip() and json.loads(text).get("assign"):
                got.add(item.get("custom_id", ""))
        except Exception:
            pass


def cmd_retry(args):
    """Recover paper-views whose batch response was empty/unparseable: rebuild
    their requests from meta.json and add new retry shards for submit->collect."""
    meta = json.loads((BATCH / "meta.json").read_text())
    tracker = json.loads((BATCH / "tracker.json").read_text())
    got = set()
    for j in tracker["jobs"]:
        _parsed_ok(_download(j["batch_id"]), got)
    missing = [cid for cid in meta if cid not in got]
    print(f"[batch] {len(missing)} missing of {len(meta)} paper-views")
    if not missing:
        return
    nodes, _, host_nodes, _ = vl.build_all_views(use_cache=True)
    hostblocks = {}
    for view in g.GROW_VIEWS:
        descs = g.host_descs(view, nodes)
        hostblocks[view] = "\n".join(f"- {n}: {descs.get(n,'')}" for n in host_nodes[view])
    reqs = []
    for cid in missing:
        m = meta[cid]
        reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                     "body": {"model": MODEL, "messages": [
                         {"role": "system", "content": g._LLM_SYS},
                         {"role": "user", "content": g._build_prompt(m["view"], hostblocks[m["view"]], m["lvs"])}],
                         "max_completion_tokens": 4096,
                         "response_format": {"type": "json_object"}}})
    manifest = json.loads((BATCH / "manifest.json").read_text())
    n_existing = len([s for s in manifest["shards"] if s.startswith("requests_retry")])
    for i in range(0, len(reqs), SHARD_SIZE):
        fn = f"requests_retry_{n_existing + i // SHARD_SIZE:03d}.jsonl"
        (BATCH / fn).write_text("\n".join(json.dumps(r) for r in reqs[i:i + SHARD_SIZE]) + "\n")
        manifest["shards"].append(fn)
    (BATCH / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[batch] wrote {len(reqs)} retry requests in new shards; "
          f"run: submit -> poll -> collect")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--papers", type=int, default=3000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--per-paper", type=int, default=40)
    p.add_argument("--min-claims", type=int, default=25)
    p.add_argument("--exclude-placed", action="store_true",
                   help="incremental: skip DOIs already in taxonomy_leaves so only "
                        "new papers are placed (abstract-only expansion)")
    sub.add_parser("submit")
    sub.add_parser("poll")
    sub.add_parser("collect")
    sub.add_parser("retry")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "poll": cmd_poll,
     "collect": cmd_collect, "retry": cmd_retry}[args.cmd](args)


if __name__ == "__main__":
    main()
