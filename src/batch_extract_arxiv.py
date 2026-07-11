"""
Batch extraction of arXiv papers via Gemini (PortKey gateway).

Downloads PDFs, builds batch JSONL files with base64-encoded PDFs,
submits to the PortKey/Vertex AI batch API, polls, and collects results.

Workflow:
    python src/batch_extract_arxiv.py prepare --tier 1    # Download PDFs + build JSONL
    python src/batch_extract_arxiv.py submit  --tier 1    # Upload & submit batches
    python src/batch_extract_arxiv.py status  --tier 1    # Check batch statuses
    python src/batch_extract_arxiv.py collect --tier 1    # Download results
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import requests as http_req

DATA_DIR = Path(__file__).parent.parent / "data"
HARVEST_DIR = DATA_DIR / "arxiv_harvest"
RESULTS_DIR = DATA_DIR / "deep_results"
PDF_DIR = DATA_DIR / "papers_full"

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

MAX_BATCH_FILE_BYTES = 90 * 1024 * 1024   # 90 MB per JSONL file
MAX_PDF_BYTES = 30 * 1024 * 1024           # skip PDFs > 30 MB
PAPERS_PER_BATCH = 200                      # hard cap per file (Vertex limit)

EXTRACTION_PROMPT = """You are a chemistry expert performing EXHAUSTIVE knowledge extraction from a research paper.

Extract EVERY piece of structured knowledge. Target 20-50 claims per paper. Do NOT summarize — extract individual data points.

Return a JSON object with:
{
  "paper_knowledge": {
    "hypothesis": "The central hypothesis or research question",
    "experimental_design": "Brief description of the experimental approach",
    "conclusions": ["Main conclusion 1", "Main conclusion 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "future_directions": ["Future direction 1", "Future direction 2"],
    "surprising_findings": ["Any unexpected or counter-intuitive results"],
    "paper_type": "research_article|review|communication|computational_study|methods_paper",
    "subfield": "organic_synthesis|inorganic|materials|catalysis|physical_chemistry|biochemistry|computational|electrochemistry|photochemistry|polymer|environmental|analytical|other"
  },
  "claims": [
    {
      "claim_id": "sequential number",
      "claim_type": "reaction|property|method|mechanism|comparison|scope_entry|computational_result|structure|hypothesis|experimental_design|limitation|future_direction|surprising_finding",
      "confidence": "high|medium|low",
      "location_in_paper": "Table 1, entry 3",
      "verbatim_quote": "exact sentence(s) from paper supporting this claim"
    }
  ]
}

CRITICAL: Extract EVERY entry from substrate scope and optimization tables as separate claims.
Include control experiments and negative results. A typical paper should yield 20-50 claims."""


def custom_id_for_doi(doi):
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def pipeline_dir(tier):
    d = DATA_DIR / f"arxiv_batch_tier{tier}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _curl_json(method, path, data=None, form_fields=None, file_path=None):
    """Call the PortKey gateway via curl."""
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-s", "--max-time", "600", "-X", method]
    cmd += ["-H", "x-portkey-api-key: " + api_key]
    cmd += ["-H", "x-portkey-provider: " + PROVIDER]

    if data is not None:
        cmd += ["-H", "Content-Type: application/json"]
        cmd += ["-d", json.dumps(data)]
    elif form_fields or file_path:
        bucket = os.environ.get("GCS_BKT", "")
        if bucket:
            cmd += ["-H", "x-portkey-vertex-storage-bucket-name: " + bucket]
        if form_fields:
            for k, v in form_fields.items():
                if k == "provider_file_name":
                    cmd += ["-H", "x-portkey-provider-file-name: " + v]
                elif k == "provider_model":
                    cmd += ["-H", "x-portkey-provider-model: " + v]
                else:
                    cmd += ["--form", k + '="' + v + '"']
        if file_path:
            cmd += ["--form", "file=@" + file_path]

    cmd.append(GATEWAY + path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=660)
    if not result.stdout.strip():
        return {"error": "empty_response", "stderr": result.stderr[:200]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "parse_error", "raw": result.stdout[:500]}


# ── PDF download ──────────────────────────────────────────────────────────────

def download_pdf(url, dest):
    for attempt in range(3):
        try:
            resp = http_req.get(
                url, timeout=60,
                headers={"User-Agent": "AskChem/1.0 (research; mailto:askchem@nyu.edu)"},
            )
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except Exception:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return False


# ── Load papers from tier file ────────────────────────────────────────────────

def load_tier_papers(tier):
    tf = HARVEST_DIR / ("tier_%d.jsonl" % tier)
    if not tf.exists():
        print("Tier file not found: %s" % tf)
        return []
    papers = []
    with open(tf) as f:
        for line in f:
            try:
                p = json.loads(line)
                doi = p.get("doi") or ("10.48550/arXiv." + p["arxiv_id"])
                papers.append({
                    "doi": doi,
                    "title": p.get("title", ""),
                    "pdf_url": p.get("pdf_url", ""),
                    "arxiv_id": p.get("arxiv_id", ""),
                    "custom_id": custom_id_for_doi(doi),
                    "citation_count": p.get("citation_count", 0),
                })
            except (json.JSONDecodeError, KeyError):
                pass
    papers.sort(key=lambda x: -(x.get("citation_count") or 0))
    return papers


# ── PREPARE ───────────────────────────────────────────────────────────────────
#
# Stream-and-delete: download PDF -> base64 encode into JSONL -> delete PDF.
# Peak disk usage = only the JSONL batch files (no persistent PDF storage).

def _download_pdf_bytes(url):
    """Download PDF and return bytes, or None on failure."""
    for attempt in range(3):
        try:
            resp = http_req.get(
                url, timeout=60,
                headers={"User-Agent": "AskChem/1.0 (research; mailto:askchem@nyu.edu)"},
            )
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return None


def _make_batch_line(cid, pdf_bytes):
    """Build a single JSONL request line from raw PDF bytes."""
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    request = {
        "custom_id": cid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:application/pdf;base64," + pdf_b64}},
                ],
            }],
            "max_completion_tokens": 65536,
            "response_format": {"type": "json_object"},
        },
    }
    return json.dumps(request) + "\n"


def _flush_batch(pdir, tier, batch_idx, current_lines):
    """Write current batch lines to a JSONL file and return metadata."""
    fname = "extract_tier%d_%03d.jsonl" % (tier, batch_idx)
    fpath = pdir / fname
    with open(fpath, "w") as f:
        f.writelines(current_lines)
    size_mb = round(fpath.stat().st_size / 1e6, 1)
    print("  %s: %d papers, %.1f MB" % (fname, len(current_lines), size_mb), flush=True)
    return {"file": fname, "count": len(current_lines), "size_mb": size_mb}


def cmd_prepare(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tempfile

    tier = args.tier
    pdir = pipeline_dir(tier)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_tier_papers(tier)
    print("Tier %d papers: %d" % (tier, len(papers)))

    already_done = {f.stem for f in RESULTS_DIR.glob("*.json")} if RESULTS_DIR.exists() else set()

    already_in_batch = set()
    for bf in pdir.glob("extract_tier*_*.jsonl"):
        try:
            for line in open(bf):
                cid = json.loads(line).get("custom_id", "")
                if cid:
                    already_in_batch.add(cid)
        except Exception:
            pass
    if already_in_batch:
        print("Already in batch JSONL files: %d" % len(already_in_batch))

    skip_ids = already_done | already_in_batch
    papers = [p for p in papers if p["custom_id"] not in skip_ids]
    print("After skipping extracted + already batched: %d" % len(papers))

    if args.max:
        papers = papers[:args.max]
        print("Limited to: %d" % args.max)

    if not papers:
        print("Nothing to prepare.")
        return

    # We process in parallel: download PDF -> encode -> append to batch file.
    # To keep memory bounded, we use a producer/consumer pattern:
    #   - 10 threads download PDFs concurrently
    #   - Main thread collects results and writes batch JSONL files

    print("\nDownloading PDFs & building batch JSONL (10 threads, stream-and-delete)...")

    existing_batches = sorted(pdir.glob("extract_tier%d_*.jsonl" % tier))
    batch_idx = len(existing_batches)
    if batch_idx > 0:
        print("Resuming from batch index %d" % batch_idx)

    current_lines = []
    current_size = 0
    batch_files = []
    paper_doi_map = {}
    total_ok = 0
    total_fail = 0
    total_skip = 0
    total_too_big = 0
    total_requests = 0

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    def _fetch_paper(p):
        """Download or load cached PDF, return (paper, pdf_bytes_or_None, status)."""
        cid = p["custom_id"]
        cached = PDF_DIR / (cid + ".pdf")
        if cached.exists() and cached.stat().st_size > 0:
            return (p, cached.read_bytes(), "cached")
        url = p.get("pdf_url", "")
        if not url:
            return (p, None, "no_url")
        pdf_bytes = _download_pdf_bytes(url)
        if pdf_bytes is None:
            return (p, None, "fail")
        return (p, pdf_bytes, "ok")

    CHUNK = 500
    for chunk_start in range(0, len(papers), CHUNK):
        chunk = papers[chunk_start:chunk_start + CHUNK]
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_fetch_paper, p): p for p in chunk}
            for fut in as_completed(futures):
                p, pdf_bytes, status = fut.result()
                cid = p["custom_id"]

                if status in ("no_url", "fail"):
                    total_fail += 1
                    continue

                if status == "cached":
                    total_skip += 1
                else:
                    total_ok += 1

                if len(pdf_bytes) > MAX_PDF_BYTES or len(pdf_bytes) == 0:
                    total_too_big += 1
                    continue

                line = _make_batch_line(cid, pdf_bytes)
                line_bytes = len(line.encode("utf-8"))

                if (current_size + line_bytes > MAX_BATCH_FILE_BYTES) or (len(current_lines) >= PAPERS_PER_BATCH):
                    if current_lines:
                        batch_files.append(_flush_batch(pdir, tier, batch_idx, current_lines))
                        batch_idx += 1
                    current_lines = []
                    current_size = 0

                current_lines.append(line)
                current_size += line_bytes
                total_requests += 1
                paper_doi_map[cid] = p["doi"]

        done = total_ok + total_fail + total_skip + total_too_big
        print("  Progress: %d/%d  (ok=%d, cached=%d, fail=%d, too_big=%d, batches=%d)" % (
            done, len(papers), total_ok, total_skip, total_fail, total_too_big,
            len(batch_files)), flush=True)

    if current_lines:
        batch_files.append(_flush_batch(pdir, tier, batch_idx, current_lines))

    manifest_path = pdir / "manifest.json"
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())
        old_manifest["files"].extend(batch_files)
        old_manifest["paper_dois"].update(paper_doi_map)
        old_manifest["total_requests"] += total_requests
        old_manifest["resumed_at"] = datetime.now().isoformat()
        manifest = old_manifest
    else:
        manifest = {
            "tier": tier,
            "model": MODEL,
            "total_requests": total_requests,
            "skipped_no_pdf": total_fail,
            "skipped_too_big": total_too_big,
            "used_cached": total_skip,
            "downloaded_new": total_ok,
            "files": batch_files,
            "paper_dois": paper_doi_map,
            "created_at": datetime.now().isoformat(),
        }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_gb = sum(b["size_mb"] for b in batch_files) / 1000
    print("\n" + "=" * 60)
    print("PREPARE COMPLETE — Tier %d" % tier)
    print("=" * 60)
    print("  Total requests: %d" % total_requests)
    print("  Downloaded new: %d  |  Used cached: %d" % (total_ok, total_skip))
    print("  Failed: %d  |  Too big: %d" % (total_fail, total_too_big))
    print("  Batch files: %d (%.2f GB)" % (len(batch_files), total_gb))
    print("\nSubmit with: python src/batch_extract_arxiv.py submit --tier %d" % tier)


# ── SUBMIT ────────────────────────────────────────────────────────────────────

def _submit_one_file(pdir, fname, size_mb):
    """Upload one JSONL file and create a batch. Returns tracker entry dict."""
    fpath = pdir / fname
    if not fpath.exists():
        return {"status": "missing", "error": "file not found on disk"}

    upload_resp = _curl_json("POST", "/files",
        form_fields={
            "purpose": "batch",
            "provider_file_name": fname,
            "provider_model": MODEL,
        },
        file_path=str(fpath),
    )
    file_id = upload_resp.get("id")
    if not file_id:
        return {"status": "failed", "error": str(upload_resp)[:200]}

    batch_resp = _curl_json("POST", "/batches", data={
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "model": MODEL,
    })
    batch_id = batch_resp.get("id")
    status = batch_resp.get("status", "unknown")

    if batch_id and fpath.exists():
        fpath.unlink()

    return {
        "file_id": file_id,
        "batch_id": batch_id,
        "status": status,
        "submitted_at": datetime.now().isoformat(),
    }


def cmd_submit(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    tier = args.tier
    pdir = pipeline_dir(tier)
    manifest_path = pdir / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json. Run 'prepare --tier %d' first." % tier)
        return

    manifest = json.loads(manifest_path.read_text())
    tracker_path = pdir / "tracker.json"
    tracker = {}
    if tracker_path.exists():
        tracker = json.loads(tracker_path.read_text())

    to_submit = []
    for entry in manifest["files"]:
        fname = entry["file"]
        if fname in tracker and tracker[fname].get("status") not in ("failed", "missing"):
            continue
        to_submit.append(entry)

    print("To submit: %d files (%d already done)" % (len(to_submit), len(manifest["files"]) - len(to_submit)))
    if not to_submit:
        print("Nothing to submit.")
        return

    tracker_lock = threading.Lock()
    submitted = {"ok": 0, "fail": 0}

    def _do_submit(entry):
        fname = entry["file"]
        result = _submit_one_file(pdir, fname, entry["size_mb"])
        with tracker_lock:
            tracker[fname] = result
            if result.get("batch_id"):
                submitted["ok"] += 1
            else:
                submitted["fail"] += 1
            if (submitted["ok"] + submitted["fail"]) % 10 == 0:
                with open(tracker_path, "w") as f:
                    json.dump(tracker, f, indent=2)
                print("  Submitted: %d ok, %d fail / %d total" % (
                    submitted["ok"], submitted["fail"], len(to_submit)), flush=True)

    workers = min(3, len(to_submit))
    print("Submitting with %d parallel workers..." % workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do_submit, e) for e in to_submit]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print("  Worker error: %s" % exc, flush=True)

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    print("\nSubmit complete: %d ok, %d failed" % (submitted["ok"], submitted["fail"]))
    print("Tracker: %s" % tracker_path)


# ── STATUS ────────────────────────────────────────────────────────────────────

def cmd_status(args):
    tier = args.tier
    pdir = pipeline_dir(tier)
    tracker_path = pdir / "tracker.json"
    if not tracker_path.exists():
        print("No tracker.json. Run 'submit --tier %d' first." % tier)
        return

    tracker = json.loads(tracker_path.read_text())
    summary = {"validating": 0, "in_progress": 0, "completed": 0, "failed": 0, "other": 0}

    for fname, info in tracker.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            summary["other"] += 1
            continue

        resp = _curl_json("GET", "/batches/" + batch_id)
        new_status = resp.get("status", "unknown")
        counts = resp.get("request_counts", {})
        info["status"] = new_status
        info["request_counts"] = counts

        cat = new_status if new_status in summary else "other"
        summary[cat] += 1

        completed = counts.get("completed") or 0
        total = counts.get("total") or 0
        print("  %s: %s (%d/%d)" % (fname, new_status, completed, total))
        time.sleep(0.5)

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    print("\nSummary: %s" % json.dumps(summary))


# ── COLLECT ───────────────────────────────────────────────────────────────────

def _collect_one_batch(batch_id):
    """Download output for one batch. Returns raw text or None."""
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-s", "--max-time", "120", "-X", "GET",
           "-H", "x-portkey-api-key: " + api_key,
           "-H", "x-portkey-provider: " + PROVIDER,
           GATEWAY + "/batches/" + batch_id + "/output"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.stdout.strip():
            return result.stdout
    except Exception:
        pass
    return None


def _parse_one_output(raw_text, paper_dois):
    """Parse a single batch output into per-paper result files. Returns (saved, failed)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    failed = 0
    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            cid = item.get("custom_id", "")
            response = item.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                failed += 1
                continue
            content = choices[0].get("message", {}).get("content", "")
            parsed = json.loads(content)
            claims = parsed.get("claims", [])

            result_path = RESULTS_DIR / (cid + ".json")
            if result_path.exists():
                continue

            usage = body.get("usage", {})
            result_data = {
                "doi": paper_dois.get(cid, ""),
                "custom_id": cid,
                "num_claims": len(claims),
                "collected_at": datetime.now().isoformat(),
                "extraction_model": PROVIDER + "/" + MODEL,
                "extraction_method": "batch_gemini_image",
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "data": {
                    "paper_knowledge": parsed.get("paper_knowledge", {}),
                    "claims": claims,
                },
            }
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)
            saved += 1
        except Exception:
            failed += 1
    return saved, failed


def cmd_collect(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    tier = args.tier
    pdir = pipeline_dir(tier)
    tracker_path = pdir / "tracker.json"
    manifest_path = pdir / "manifest.json"
    if not tracker_path.exists():
        print("No tracker.json.")
        return

    tracker = json.loads(tracker_path.read_text())
    paper_dois = {}
    if manifest_path.exists():
        paper_dois = json.loads(manifest_path.read_text()).get("paper_dois", {})

    output_dir = pdir / "outputs"
    output_dir.mkdir(exist_ok=True)

    to_collect = [(fname, info) for fname, info in tracker.items()
                  if info.get("status") == "completed" and not info.get("collected") and info.get("batch_id")]
    print("To collect: %d batches" % len(to_collect))
    if not to_collect:
        print("Nothing to collect. Running parse on existing outputs...")
        _parse_results(output_dir, paper_dois)
        return

    tracker_lock = threading.Lock()
    stats = {"saved": 0, "failed": 0, "collected": 0, "download_fail": 0}

    def _do_collect(fname, info):
        batch_id = info["batch_id"]
        raw = _collect_one_batch(batch_id)
        if raw is None:
            with tracker_lock:
                stats["download_fail"] += 1
            return

        out_path = output_dir / fname
        with open(out_path, "w") as f:
            f.write(raw)

        s, fl = _parse_one_output(raw, paper_dois)

        with tracker_lock:
            info["collected"] = True
            info["collected_at"] = datetime.now().isoformat()
            stats["saved"] += s
            stats["failed"] += fl
            stats["collected"] += 1

            if stats["collected"] % 20 == 0:
                with open(tracker_path, "w") as tf:
                    json.dump(tracker, tf, indent=2)
                print("  Collected: %d/%d  (papers saved: %d, parse fail: %d, download fail: %d)" % (
                    stats["collected"], len(to_collect), stats["saved"], stats["failed"],
                    stats["download_fail"]), flush=True)

    workers = min(8, len(to_collect))
    print("Collecting with %d parallel workers..." % workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do_collect, fname, info) for fname, info in to_collect]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print("  Worker error: %s" % exc, flush=True)

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    print("\nCollect complete: %d collected, %d download failures" % (stats["collected"], stats["download_fail"]))
    print("Papers saved: %d  |  Parse failures: %d" % (stats["saved"], stats["failed"]))


def _parse_results(output_dir, paper_dois):
    """Parse batch outputs into per-paper deep_results/ JSON files (legacy, for manual re-parse)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    failed = 0
    for ofile in sorted(output_dir.glob("*.jsonl")):
        s, f = _parse_one_output(ofile.read_text(), paper_dois)
        saved += s
        failed += f
    print("\n  Saved: %d papers  |  Parse failures: %d" % (saved, failed))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch extraction of arXiv papers via Gemini")
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prepare", help="Download PDFs + build batch JSONL")
    p_prep.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4])
    p_prep.add_argument("--max", type=int, help="Limit papers to process")

    p_sub = sub.add_parser("submit", help="Upload & submit batch files")
    p_sub.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4])

    p_st = sub.add_parser("status", help="Check batch statuses")
    p_st.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4])

    p_col = sub.add_parser("collect", help="Download completed results")
    p_col.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4])

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
