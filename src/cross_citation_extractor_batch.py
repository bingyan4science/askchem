"""Batch-API version of cross_citation_extractor (Gemini Batch via PortKey/Vertex).

Same prompt, same validation, same edge_jobs/claim_edges schema as the synchronous
extractor (`src/cross_citation_extractor.py`).  Difference:

* Pairs are bundled into JSONL chunks (default 5K requests/chunk).
* Each chunk is uploaded to the NYU PortKey gateway's `/v1/files` endpoint
  (which writes to a Vertex AI batch GCS bucket) and submitted via `/v1/batches`.
* The Vertex Batch Prediction API runs all requests asynchronously at ~50%
  of synchronous list-price.  Wall time is typically 5-15 min/chunk regardless
  of chunk size up to ~10K requests.
* `ingest` downloads completed batch outputs, validates each response, and
  writes edges + edge_jobs the same way as the sync path.

Subcommands:
    prepare   Select pairs, build prompts, write chunks to data/batch_jobs/<tag>/.
    submit    For each chunk without a batch_id: upload + create batch.
    status    Per-chunk batch status + aggregate progress.
    ingest    Download finished batches, parse, validate, insert edges.
    run       prepare + submit + poll-until-done + ingest, all in one go.
    purge     Delete edges + jobs + on-disk chunks for one extractor tag.

State layout (per extractor):
    data/batch_jobs/<extractor>/
        chunks/chunk_NNN.jsonl       # input batch requests
        chunks/chunk_NNN.meta.json   # {chunk_idx, file_id, batch_id, ingested,
                                     #  pairs:[{row:int, citing:str, cited:str,
                                     #          a_claim_ids:[...], b_claim_ids:[...]}]}
        outputs/chunk_NNN.jsonl      # downloaded prediction file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem.models import CROSS_PAPER_EDGE_TYPES  # noqa: E402
from backfill_edges import (  # noqa: E402
    MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M,
    DEFAULT_MAX_TOKENS, DEFAULT_MAX_CLAIMS_PER_PAPER,
    _compact, validate_edges,
)
from cross_citation_extractor import (  # noqa: E402
    CROSS_CITATION_PROMPT, PAIR_MODE,
    DEFAULT_FULL_TAG,
    open_db, fetch_claims_for, fetch_paper_title,
    insert_edges, record_job, select_pairs, _ensure_schema,
)

# ── Gateway / batch configuration ───────────────────────────────────────────

GATEWAY_BASE = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
GCS_BUCKET = "kc119-batch-inference-research-workspace-w9yz"

# Vertex Batch Prediction is 50% of online list price.
BATCH_PRICE_IN_PER_M = PRICE_IN_PER_M * 0.5
BATCH_PRICE_OUT_PER_M = PRICE_OUT_PER_M * 0.5

DEFAULT_CHUNK_SIZE = 5000
DEFAULT_TAG = "cross_citation_v1_batch"


def _headers(*, with_bucket: bool = False) -> dict:
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    h = {
        "x-portkey-api-key": api_key,
        "x-portkey-provider": PROVIDER,
        "x-portkey-provider-model": MODEL,
    }
    if with_bucket:
        h["x-portkey-vertex-storage-bucket-name"] = GCS_BUCKET
    return h


def _state_dir(extractor: str) -> Path:
    p = REPO_ROOT / "data" / "batch_jobs" / extractor
    (p / "chunks").mkdir(parents=True, exist_ok=True)
    (p / "outputs").mkdir(parents=True, exist_ok=True)
    return p


def _chunk_paths(extractor: str, chunk_idx: int) -> tuple[Path, Path, Path]:
    base = _state_dir(extractor)
    return (
        base / "chunks" / f"chunk_{chunk_idx:03d}.jsonl",
        base / "chunks" / f"chunk_{chunk_idx:03d}.meta.json",
        base / "outputs" / f"chunk_{chunk_idx:03d}.jsonl",
    )


def _read_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def _write_meta(meta_path: Path, meta: dict) -> None:
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(meta_path)


def _all_chunks(extractor: str) -> list[int]:
    chunks_dir = _state_dir(extractor) / "chunks"
    out = []
    for p in sorted(chunks_dir.glob("chunk_*.meta.json")):
        # filename: chunk_NNN.meta.json; we want NNN
        token = p.name.split("_", 1)[1].split(".", 1)[0]
        out.append(int(token))
    return sorted(set(out))


def estimate_batch_cost(t_in: int, t_out: int) -> float:
    return t_in * (BATCH_PRICE_IN_PER_M / 1e6) + t_out * (BATCH_PRICE_OUT_PER_M / 1e6)


# ── prepare ─────────────────────────────────────────────────────────────────


def _build_request_body(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": DEFAULT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def cmd_prepare(args):
    _ensure_schema()
    extractor = args.extractor_tag or DEFAULT_TAG
    state = _state_dir(extractor)
    print(f"[prepare] extractor={extractor} state={state}")

    con = open_db()

    view, path = (args.subarea.split(":", 1) if args.subarea else (None, None))
    pairs = select_pairs(
        con, extractor=extractor, resume=args.resume,
        limit=args.limit, subarea_view=view, subarea_path=path,
        seed=args.seed,
    )
    if args.also_skip_done_under:
        skip_tags = [t.strip() for t in args.also_skip_done_under.split(",") if t.strip()]
        placeholders = ",".join("?" * len(skip_tags))
        done_elsewhere = {
            r["paper_doi"]
            for r in con.execute(
                f"SELECT paper_doi FROM edge_jobs "
                f" WHERE mode=? AND extractor IN ({placeholders}) AND status='done'",
                (PAIR_MODE, *skip_tags),
            ).fetchall()
        }
        before = len(pairs)
        pairs = [(c, t) for c, t in pairs if f"{c}|{t}" not in done_elsewhere]
        print(f"  also-skip-done-under {skip_tags}: "
              f"{before - len(pairs):,} pairs already done elsewhere; "
              f"{len(pairs):,} remain")
    if not pairs:
        print("[prepare] no pairs to prepare")
        return

    # Drop pairs that are already in chunks on disk (resumable prepare).
    existing_pairs: set[str] = set()
    for ci in _all_chunks(extractor):
        _, meta_path, _ = _chunk_paths(extractor, ci)
        m = _read_meta(meta_path)
        for p in m.get("pairs", []):
            existing_pairs.add(f"{p['citing']}|{p['cited']}")
    if existing_pairs:
        before = len(pairs)
        pairs = [(c, t) for c, t in pairs if f"{c}|{t}" not in existing_pairs]
        print(f"[prepare] {before - len(pairs)} pairs already prepared on disk; "
              f"{len(pairs)} new pairs to chunk")
    if not pairs:
        return

    # Determine starting chunk index.
    chunk_idx = (max(_all_chunks(extractor)) + 1) if _all_chunks(extractor) else 0

    chunk_size = args.chunk_size
    skipped_immediately = 0
    n_in_chunk = 0
    chunk_jsonl = []
    chunk_meta_pairs: list[dict] = []
    started = datetime.utcnow().isoformat() + "Z"
    t0 = time.monotonic()

    def flush_chunk():
        nonlocal chunk_idx, chunk_jsonl, chunk_meta_pairs, n_in_chunk
        if not chunk_jsonl:
            return
        jsonl_path, meta_path, _ = _chunk_paths(extractor, chunk_idx)
        with jsonl_path.open("w") as f:
            for line in chunk_jsonl:
                f.write(line)
                f.write("\n")
        meta = {
            "chunk_idx": chunk_idx,
            "extractor": extractor,
            "model": MODEL,
            "n_requests": n_in_chunk,
            "file_id": None,
            "batch_id": None,
            "batch_status": None,
            "output_file_id": None,
            "ingested": False,
            "created_at": started,
            "pairs": chunk_meta_pairs,
        }
        _write_meta(meta_path, meta)
        print(f"  wrote chunk {chunk_idx:03d}: {n_in_chunk} requests "
              f"-> {jsonl_path.name}")
        chunk_idx += 1
        chunk_jsonl = []
        chunk_meta_pairs = []
        n_in_chunk = 0

    for i, (citing, cited) in enumerate(pairs):
        a_claims = fetch_claims_for(con, citing, max_claims=args.max_claims)
        b_claims = fetch_claims_for(con, cited, max_claims=args.max_claims)
        if not a_claims or not b_claims:
            pair_key = f"{citing}|{cited}"
            record_job(
                con, pair_key=pair_key, extractor=extractor,
                status="skipped", edges_inserted=0,
                tokens_in=0, tokens_out=0,
                error=f"empty claims (a={len(a_claims)} b={len(b_claims)})",
                started_at=started,
            )
            skipped_immediately += 1
            continue

        a_title = fetch_paper_title(con, citing)
        b_title = fetch_paper_title(con, cited)
        a_compact = [_compact(c) for c in a_claims]
        b_compact = [_compact(c) for c in b_claims]
        prompt = CROSS_CITATION_PROMPT.format(
            a_title=a_title, a_doi=citing,
            a_claims_json=json.dumps(a_compact, indent=1),
            b_title=b_title, b_doi=cited,
            b_claims_json=json.dumps(b_compact, indent=1),
        )

        custom_id = f"c{chunk_idx:03d}_r{n_in_chunk:05d}"
        req = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _build_request_body(prompt),
        }
        chunk_jsonl.append(json.dumps(req))
        chunk_meta_pairs.append({
            "row": n_in_chunk,
            "custom_id": custom_id,
            "citing": citing,
            "cited": cited,
            "a_claim_ids": [c["claim_id"] for c in a_claims],
            "b_claim_ids": [c["claim_id"] for c in b_claims],
        })
        n_in_chunk += 1

        # Mark this pair "queued" so the sync extractor won't pick it up.
        record_job(
            con, pair_key=f"{citing}|{cited}", extractor=extractor,
            status="queued", edges_inserted=0,
            tokens_in=0, tokens_out=0, error=None, started_at=started,
        )

        if n_in_chunk >= chunk_size:
            flush_chunk()
        if (i + 1) % 1000 == 0:
            el = time.monotonic() - t0
            print(f"  prepared {i+1}/{len(pairs)} pairs "
                  f"(skipped={skipped_immediately}, {el:.1f}s)")

    flush_chunk()
    el = time.monotonic() - t0
    print(f"[prepare] done in {el:.1f}s. "
          f"chunks_on_disk={len(_all_chunks(extractor))}, "
          f"skipped_immediately={skipped_immediately}")


# ── submit ──────────────────────────────────────────────────────────────────


def _upload_file(jsonl_path: Path) -> str:
    """POST the chunk JSONL to /v1/files; return file_id (URL-encoded GCS path)."""
    headers = _headers(with_bucket=True)
    with jsonl_path.open("rb") as fh:
        files = {"file": (jsonl_path.name, fh, "application/jsonl")}
        data = {"purpose": "batch"}
        r = requests.post(
            f"{GATEWAY_BASE}/files", headers=headers, files=files, data=data,
            timeout=600,
        )
    if r.status_code != 200:
        raise RuntimeError(f"upload failed http {r.status_code}: {r.text[:300]}")
    return r.json()["id"]


def _create_batch(file_id: str) -> dict:
    headers = _headers(with_bucket=True)
    headers["Content-Type"] = "application/json"
    body = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "model": MODEL,
    }
    r = requests.post(
        f"{GATEWAY_BASE}/batches", headers=headers, json=body, timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"create batch failed http {r.status_code}: {r.text[:300]}")
    return r.json()


def cmd_submit(args):
    extractor = args.extractor_tag or DEFAULT_TAG
    chunk_idxs = _all_chunks(extractor)
    if not chunk_idxs:
        print(f"[submit] no chunks for {extractor}; run prepare first")
        return
    print(f"[submit] extractor={extractor}, chunks_on_disk={len(chunk_idxs)}")

    pending = []
    for ci in chunk_idxs:
        _, meta_path, _ = _chunk_paths(extractor, ci)
        m = _read_meta(meta_path)
        if m.get("batch_id"):
            continue
        pending.append(ci)
    print(f"[submit] {len(pending)} chunks pending submission")

    for ci in pending:
        jsonl_path, meta_path, _ = _chunk_paths(extractor, ci)
        m = _read_meta(meta_path)
        try:
            print(f"  chunk {ci:03d}: uploading {jsonl_path.name} ...")
            t0 = time.monotonic()
            file_id = m.get("file_id") or _upload_file(jsonl_path)
            m["file_id"] = file_id
            _write_meta(meta_path, m)
            print(f"    uploaded in {time.monotonic()-t0:.1f}s, file_id={file_id[:60]}...")
            print(f"  chunk {ci:03d}: creating batch ...")
            b = _create_batch(file_id)
            m["batch_id"] = b["id"]
            m["batch_status"] = b.get("status")
            _write_meta(meta_path, m)
            print(f"    batch_id={b['id']} status={b.get('status')}")
        except Exception as e:
            print(f"  ! chunk {ci:03d} submit failed: {e}")
            if not args.continue_on_error:
                raise


# ── status ─────────────────────────────────────────────────────────────────


def _get_batch(batch_id: str) -> dict:
    headers = _headers()
    r = requests.get(
        f"{GATEWAY_BASE}/batches/{batch_id}", headers=headers, timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _refresh_chunk_status(extractor: str, ci: int) -> dict:
    _, meta_path, _ = _chunk_paths(extractor, ci)
    m = _read_meta(meta_path)
    if not m.get("batch_id"):
        return m
    if m.get("batch_status") in ("completed", "failed", "cancelled", "expired"):
        return m
    try:
        b = _get_batch(m["batch_id"])
    except Exception as e:
        m["batch_error"] = str(e)[:300]
        _write_meta(meta_path, m)
        return m
    m["batch_status"] = b.get("status")
    if b.get("output_file_id"):
        m["output_file_id"] = b["output_file_id"]
    if b.get("error_file_id"):
        m["error_file_id"] = b["error_file_id"]
    rc = b.get("request_counts") or {}
    m["request_counts"] = rc
    _write_meta(meta_path, m)
    return m


def cmd_status(args):
    extractor = args.extractor_tag or DEFAULT_TAG
    chunk_idxs = _all_chunks(extractor)
    if not chunk_idxs:
        print(f"[status] no chunks for {extractor}")
        return
    print(f"[status] extractor={extractor}")
    print(f"  {'chunk':>5} {'reqs':>6} {'status':>14} {'done':>6} {'fail':>6} "
          f"{'ingested':>9}  batch_id")
    print("  " + "-" * 80)
    agg = {"validating": 0, "in_progress": 0, "completed": 0,
           "failed": 0, "cancelled": 0, "expired": 0, "unsubmitted": 0,
           "ingested_chunks": 0, "total_requests": 0,
           "completed_requests": 0, "failed_requests": 0}
    for ci in chunk_idxs:
        m = _refresh_chunk_status(extractor, ci)
        st = m.get("batch_status") or ("unsubmitted" if not m.get("batch_id") else "?")
        rc = m.get("request_counts") or {}
        n = m.get("n_requests", 0)
        completed = rc.get("completed") or 0
        failed = rc.get("failed") or 0
        ingested = m.get("ingested", False)
        bid = m.get("batch_id") or "-"
        print(f"  {ci:>5} {n:>6} {st:>14} {completed:>6} {failed:>6} "
              f"{'yes' if ingested else 'no':>9}  {bid}")
        agg.setdefault(st, 0)
        agg[st] += 1
        agg["total_requests"] += n
        agg["completed_requests"] += completed
        agg["failed_requests"] += failed
        if ingested:
            agg["ingested_chunks"] += 1

    print()
    print(f"  chunks: {len(chunk_idxs)} total, {agg['ingested_chunks']} ingested")
    keys = ("validating", "in_progress", "completed", "failed",
            "cancelled", "expired", "unsubmitted")
    print("  by-status: " + ", ".join(f"{k}={agg.get(k,0)}" for k in keys))
    print(f"  requests: {agg['completed_requests']}/{agg['total_requests']} "
          f"completed, {agg['failed_requests']} failed")


# ── ingest ─────────────────────────────────────────────────────────────────


def _download_file(file_id: str) -> bytes:
    headers = _headers(with_bucket=True)
    r = requests.get(
        f"{GATEWAY_BASE}/files/{file_id}/content",
        headers=headers, timeout=600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"download failed http {r.status_code}: {r.text[:300]}")
    return r.content


def _parse_response_text(resp_obj: dict) -> tuple[str, int, int]:
    """From a single output line's `response` field, extract the text content
    and (prompt_tokens, completion_tokens).

    Vertex Batch returns native Vertex shape:
      response.candidates[0].content.parts[*].text
      response.usageMetadata.{promptTokenCount, candidatesTokenCount}
    """
    cands = resp_obj.get("candidates") or []
    if not cands:
        return "", 0, 0
    content = cands[0].get("content") or {}
    parts = content.get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    usage = resp_obj.get("usageMetadata") or {}
    return text.strip(), int(usage.get("promptTokenCount") or 0), int(usage.get("candidatesTokenCount") or 0)


def _ingest_chunk(extractor: str, ci: int) -> dict:
    jsonl_path, meta_path, out_path = _chunk_paths(extractor, ci)
    m = _read_meta(meta_path)
    if m.get("ingested"):
        return {"chunk": ci, "skipped_already": True,
                "edges": 0, "tokens_in": 0, "tokens_out": 0,
                "done": 0, "failed": 0}
    if m.get("batch_status") != "completed":
        return {"chunk": ci, "skipped_not_completed": True,
                "edges": 0, "tokens_in": 0, "tokens_out": 0,
                "done": 0, "failed": 0}
    out_file = m.get("output_file_id")
    if not out_file:
        return {"chunk": ci, "skipped_no_output": True,
                "edges": 0, "tokens_in": 0, "tokens_out": 0,
                "done": 0, "failed": 0}

    if not out_path.exists():
        print(f"  chunk {ci:03d}: downloading output ...")
        blob = _download_file(out_file)
        out_path.write_bytes(blob)

    pair_lookup = {p["custom_id"]: p for p in m.get("pairs", [])}
    started = datetime.utcnow().isoformat() + "Z"
    con = open_db()

    edges_total = t_in_total = t_out_total = 0
    n_done = n_failed = 0
    n_lines = 0
    with out_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = rec.get("requestId") or rec.get("custom_id")
            pair = pair_lookup.get(cid)
            if not pair:
                continue
            citing = pair["citing"]
            cited = pair["cited"]
            pair_key = f"{citing}|{cited}"
            valid_from = set(pair.get("a_claim_ids", []))
            valid_to = set(pair.get("b_claim_ids", []))

            err = rec.get("error")
            resp_obj = rec.get("response")
            if err or not resp_obj:
                msg = json.dumps(err)[:500] if err else "no response"
                record_job(
                    con, pair_key=pair_key, extractor=extractor,
                    status="failed", edges_inserted=0,
                    tokens_in=0, tokens_out=0, error=msg,
                    started_at=started,
                )
                n_failed += 1
                continue

            text, t_in, t_out = _parse_response_text(resp_obj)
            t_in_total += t_in
            t_out_total += t_out
            if not text:
                record_job(
                    con, pair_key=pair_key, extractor=extractor,
                    status="failed", edges_inserted=0,
                    tokens_in=t_in, tokens_out=t_out,
                    error="empty content",
                    started_at=started,
                )
                n_failed += 1
                continue
            try:
                parsed = json.loads(text)
            except Exception as e:
                record_job(
                    con, pair_key=pair_key, extractor=extractor,
                    status="failed", edges_inserted=0,
                    tokens_in=t_in, tokens_out=t_out,
                    error=f"json: {e}: {text[:200]}",
                    started_at=started,
                )
                n_failed += 1
                continue

            # Most responses come back as {"edges": [...]}, but a small
            # fraction return either a bare [...] list of edges, or a single
            # edge dict with 'from'/'to'/'type'.  Be liberal in what we accept.
            if isinstance(parsed, list):
                raw_edges = parsed
            elif isinstance(parsed, dict):
                if isinstance(parsed.get("edges"), list):
                    raw_edges = parsed["edges"]
                elif {"from", "to", "type"} <= set(parsed.keys()):
                    raw_edges = [parsed]
                else:
                    raw_edges = []
            else:
                raw_edges = []

            try:
                edges, _problems = validate_edges(
                    raw_edges,
                    valid_from_ids=valid_from, valid_to_ids=valid_to,
                    allowed_types=CROSS_PAPER_EDGE_TYPES,
                    extractor=extractor, now=started,
                )
            except Exception as e:
                record_job(
                    con, pair_key=pair_key, extractor=extractor,
                    status="failed", edges_inserted=0,
                    tokens_in=t_in, tokens_out=t_out,
                    error=f"validate: {e}: {text[:200]}",
                    started_at=started,
                )
                n_failed += 1
                continue
            for e in edges:
                e.to_doi = cited
            inserted = insert_edges(con, edges)
            edges_total += inserted
            record_job(
                con, pair_key=pair_key, extractor=extractor,
                status="done", edges_inserted=inserted,
                tokens_in=t_in, tokens_out=t_out,
                error=None, started_at=started,
            )
            n_done += 1

    m["ingested"] = True
    m["ingested_at"] = datetime.utcnow().isoformat() + "Z"
    m["ingest_summary"] = {
        "lines": n_lines, "done": n_done, "failed": n_failed,
        "edges": edges_total, "tokens_in": t_in_total, "tokens_out": t_out_total,
    }
    _write_meta(meta_path, m)
    print(f"  chunk {ci:03d}: ingested {n_done} done, {n_failed} failed, "
          f"{edges_total} edges, "
          f"tokens={t_in_total:,}/{t_out_total:,}, "
          f"cost=${estimate_batch_cost(t_in_total, t_out_total):.2f}")
    return {"chunk": ci, "done": n_done, "failed": n_failed,
            "edges": edges_total, "tokens_in": t_in_total,
            "tokens_out": t_out_total}


def cmd_ingest(args):
    extractor = args.extractor_tag or DEFAULT_TAG
    chunk_idxs = _all_chunks(extractor)
    print(f"[ingest] extractor={extractor}, chunks={len(chunk_idxs)}")
    totals = {"edges": 0, "tokens_in": 0, "tokens_out": 0,
              "done": 0, "failed": 0, "ingested_chunks": 0}
    for ci in chunk_idxs:
        m = _refresh_chunk_status(extractor, ci)
        if m.get("ingested"):
            continue
        if m.get("batch_status") != "completed":
            continue
        try:
            r = _ingest_chunk(extractor, ci)
        except Exception as e:
            print(f"  ! chunk {ci:03d} ingest failed: {e}")
            continue
        totals["edges"] += r.get("edges", 0)
        totals["tokens_in"] += r.get("tokens_in", 0)
        totals["tokens_out"] += r.get("tokens_out", 0)
        totals["done"] += r.get("done", 0)
        totals["failed"] += r.get("failed", 0)
        totals["ingested_chunks"] += 1
    print()
    print(f"[ingest summary]")
    print(f"  newly ingested chunks: {totals['ingested_chunks']}")
    print(f"  pairs done: {totals['done']}, failed: {totals['failed']}")
    print(f"  edges inserted: {totals['edges']}")
    print(f"  tokens: in={totals['tokens_in']:,} out={totals['tokens_out']:,}")
    print(f"  cost (batch pricing): "
          f"${estimate_batch_cost(totals['tokens_in'], totals['tokens_out']):.2f}")


# ── run (orchestration) ────────────────────────────────────────────────────


def cmd_run(args):
    """prepare (if needed) -> submit -> poll -> ingest, all together."""
    if not args.skip_prepare:
        cmd_prepare(args)
    cmd_submit(args)

    extractor = args.extractor_tag or DEFAULT_TAG
    poll_every = args.poll_every
    print(f"[run] polling every {poll_every}s until all chunks complete ...")
    while True:
        chunk_idxs = _all_chunks(extractor)
        if not chunk_idxs:
            print("[run] no chunks; exiting")
            return
        outstanding = 0
        completed = 0
        for ci in chunk_idxs:
            m = _refresh_chunk_status(extractor, ci)
            st = m.get("batch_status")
            if st == "completed":
                completed += 1
            elif st in ("failed", "cancelled", "expired"):
                pass
            else:
                outstanding += 1
        print(f"[run] {completed}/{len(chunk_idxs)} completed, "
              f"{outstanding} still running")
        if outstanding == 0:
            break
        time.sleep(poll_every)

    cmd_ingest(args)


# ── purge ──────────────────────────────────────────────────────────────────


def cmd_purge(args):
    if not args.extractor_tag:
        print("--extractor-tag required for purge", file=sys.stderr)
        sys.exit(1)
    extractor = args.extractor_tag
    con = open_db()
    n_e = con.execute("DELETE FROM claim_edges WHERE extractor=?",
                      (extractor,)).rowcount
    n_j = con.execute("DELETE FROM edge_jobs WHERE extractor=? AND mode=?",
                      (extractor, PAIR_MODE)).rowcount
    con.commit()
    state = REPO_ROOT / "data" / "batch_jobs" / extractor
    if state.exists() and args.delete_disk:
        import shutil
        shutil.rmtree(state)
        print(f"deleted on-disk state at {state}")
    print(f"deleted {n_e} edges and {n_j} job rows for extractor={extractor}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_select(sp):
        sp.add_argument("--extractor-tag", type=str, default=None)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--resume", action="store_true",
                        help="exclude pairs already marked 'done' in edge_jobs")
        sp.add_argument("--max-claims", type=int,
                        default=DEFAULT_MAX_CLAIMS_PER_PAPER)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--subarea", type=str, default=None,
                        help="view_id:path filter (e.g. by_application:.../HER)")
        sp.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
        sp.add_argument("--also-skip-done-under", type=str, default=None,
                        help="comma-sep extractor tags whose 'done' pairs to "
                             "additionally exclude (e.g. cross_citation_v1_pilot)")

    sp_prep = sub.add_parser("prepare", help="Build chunked batch JSONL files")
    common_select(sp_prep)
    sp_prep.set_defaults(func=cmd_prepare)

    sp_sub = sub.add_parser("submit", help="Upload + create batch jobs for unsent chunks")
    sp_sub.add_argument("--extractor-tag", type=str, default=None)
    sp_sub.add_argument("--continue-on-error", action="store_true")
    sp_sub.set_defaults(func=cmd_submit)

    sp_st = sub.add_parser("status", help="Show batch status per chunk + aggregate")
    sp_st.add_argument("--extractor-tag", type=str, default=None)
    sp_st.set_defaults(func=cmd_status)

    sp_in = sub.add_parser("ingest", help="Download + parse + insert edges for completed chunks")
    sp_in.add_argument("--extractor-tag", type=str, default=None)
    sp_in.set_defaults(func=cmd_ingest)

    sp_run = sub.add_parser("run", help="prepare + submit + poll + ingest end-to-end")
    common_select(sp_run)
    sp_run.add_argument("--continue-on-error", action="store_true")
    sp_run.add_argument("--skip-prepare", action="store_true",
                        help="if chunks already exist, jump straight to submit")
    sp_run.add_argument("--poll-every", type=int, default=60,
                        help="seconds between status polls in the run loop")
    sp_run.set_defaults(func=cmd_run)

    sp_pg = sub.add_parser("purge",
                           help="Delete all edges + edge_jobs for an extractor tag")
    sp_pg.add_argument("--extractor-tag", required=True)
    sp_pg.add_argument("--delete-disk", action="store_true",
                       help="also rm -rf the on-disk chunk/output dirs")
    sp_pg.set_defaults(func=cmd_purge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
