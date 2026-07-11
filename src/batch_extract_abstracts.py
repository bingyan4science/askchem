"""Abstract-only Gemini Batch extractor (Stage 4b).

Sibling of [src/batch_extract_arxiv.py](src/batch_extract_arxiv.py). Same
Gemini 3.1 Pro Preview model via Portkey/Vertex Batch infrastructure;
only the request payload differs:

  * **4a (full PDF)**: ``{type: text, text: prompt} + {type: image_url, ...PDF base64}``
  * **4b (abstract-only)**: ``{type: text, text: prompt with title+abstract}``

Per-request cost is ~25x lower because the PDF base64 (~5 MB per paper)
is gone — we only send ~500-2000 input tokens per request.

Output: written to the same ``data/deep_results/<custom_id>.json`` sink
the full-PDF extractor uses, but tagged with
``extraction_version='deep_v1_abstract'`` so the apply step can filter,
the search UI can surface provenance, and downstream metrics can
distinguish full-text vs abstract-only evidence.

Workflow (CLI mirrors batch_extract_arxiv.py):

::

    python src/batch_extract_abstracts.py prepare --input data/abstract_jobs/<file>.jsonl
    python src/batch_extract_abstracts.py submit  --pipeline <tag>
    python src/batch_extract_abstracts.py status  --pipeline <tag>
    python src/batch_extract_abstracts.py collect --pipeline <tag>

The input JSONL contains one paper per line with at least ``doi`` and
``abstract`` (also picks up ``title``, ``authors``, ``venue``, ``year``
when present). Stage 4b expects callers to dump these from CrossRef /
ChemRxiv / arXiv-too-big-to-pdf via a small helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PIPELINE_BASE = DATA_DIR / "abstract_batch"
RESULTS_DIR = DATA_DIR / "deep_results"

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"

MAX_BATCH_FILE_BYTES = 90 * 1024 * 1024
PAPERS_PER_BATCH = 1500   # No PDF base64 → each request is small;
                          # 1500 / file keeps file under 5-10 MB.

# Prompt rules of engagement:
#   - Match the full-PDF extractor's JSON schema so downstream apply
#     code (integrate_deep.py, apply_incremental_2026_05.py) can ingest
#     the same shape with no special-case branches.
#   - Be explicit about NOT inventing specifics that aren't in the
#     abstract. This is the main risk vs. full-PDF extraction.
#   - Target 3-7 claims per paper (abstracts contain less ground truth
#     than full papers; over-extraction here is how Gemini hallucinates
#     yields / temperatures / mol% values).
ABSTRACT_EXTRACTION_PROMPT = """You are a chemistry expert performing structured knowledge extraction from a paper's ABSTRACT (no full text available).

Extract 3-7 claims that the abstract directly supports. Do NOT invent specifics — only emit a quantitative value (yield, temperature, mol%, voltage, current density, etc.) when the abstract explicitly states it. Prefer fewer, well-supported claims over many speculative ones.

Return a JSON object with:
{{
  "paper_knowledge": {{
    "hypothesis": "The central hypothesis or research question",
    "experimental_design": "Brief description of the experimental approach (or empty if not stated)",
    "conclusions": ["Main conclusion 1", "Main conclusion 2"],
    "limitations": ["Limitation 1"],
    "future_directions": ["Future direction 1"],
    "surprising_findings": ["Any unexpected or counter-intuitive results"],
    "paper_type": "research_article|review|communication|computational_study|methods_paper",
    "subfield": "organic_synthesis|inorganic|materials|catalysis|physical_chemistry|biochemistry|computational|electrochemistry|photochemistry|polymer|environmental|analytical|other"
  }},
  "claims": [
    {{
      "claim_id": "sequential number",
      "claim_type": "reaction|property|method|mechanism|comparison|scope_entry|computational_result|structure|hypothesis|experimental_design|limitation|future_direction|surprising_finding",
      "confidence": "high|medium|low",
      "location_in_paper": "abstract",
      "verbatim_quote": "EXACT sentence from the abstract that supports this claim — do NOT paraphrase"
    }}
  ]
}}

CRITICAL:
- ``verbatim_quote`` MUST be a literal substring of the abstract you were given. If you cannot find a literal substring, omit the claim entirely.
- Confidence MUST be 'low' for any claim where the abstract is imprecise.
- 3-7 claims total. Quality over quantity.

PAPER TITLE: {title}
{authors_line}{venue_line}{year_line}
ABSTRACT:
{abstract}"""


# ── helpers ──────────────────────────────────────────────────────────────────


def custom_id_for_doi(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def pipeline_dir(tag: str) -> Path:
    d = PIPELINE_BASE / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _curl_json(method, path, data=None, form_fields=None, file_path=None):
    """Call the PortKey gateway via curl (matches batch_extract_arxiv._curl_json)."""
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


def _build_prompt(paper: dict) -> str:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    authors = paper.get("authors") or []
    if authors and isinstance(authors[0], dict):
        author_names = [a.get("name", "") for a in authors[:8] if a.get("name")]
    else:
        author_names = [str(a) for a in authors[:8] if a]
    venue = (paper.get("venue") or "").strip()
    year = paper.get("year") or ""

    return ABSTRACT_EXTRACTION_PROMPT.format(
        title=title or "(no title)",
        authors_line=("AUTHORS: " + ", ".join(author_names) + "\n") if author_names else "",
        venue_line=("VENUE: " + venue + "\n") if venue else "",
        year_line=("YEAR: " + str(year) + "\n") if year else "",
        abstract=abstract or "(no abstract)",
    )


def _make_request_line(cid: str, paper: dict) -> str:
    return json.dumps({
        "custom_id": cid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": _build_prompt(paper)}],
            "max_completion_tokens": 8192,
            "response_format": {"type": "json_object"},
        },
    }) + "\n"


# ── PREPARE ──────────────────────────────────────────────────────────────────


def cmd_prepare(args):
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    tag = args.tag
    pdir = pipeline_dir(tag)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    already_done = {f.stem for f in RESULTS_DIR.glob("*.json")} if RESULTS_DIR.exists() else set()

    requests_lines: list[str] = []
    paper_dois: dict[str, str] = {}
    n_skip_no_abstract = 0
    n_skip_done = 0
    n_total = 0

    with inp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1

            doi = (paper.get("doi") or
                   (paper.get("externalIds") or {}).get("DOI") or "")
            if not doi:
                continue
            if not (paper.get("abstract") or "").strip():
                n_skip_no_abstract += 1
                continue
            cid = custom_id_for_doi(doi)
            if cid in already_done:
                n_skip_done += 1
                continue
            paper_dois[cid] = doi
            requests_lines.append(_make_request_line(cid, paper))
            if args.max and len(requests_lines) >= args.max:
                break

    print(f"Input: {inp} (total={n_total})")
    print(f"  skipped (no abstract): {n_skip_no_abstract}")
    print(f"  skipped (already extracted): {n_skip_done}")
    print(f"  to submit: {len(requests_lines)}")

    if not requests_lines:
        print("Nothing to prepare.")
        return

    # Split into chunks; abstract-only requests are tiny so we cap by
    # number of papers rather than file size.
    batch_files = []
    for chunk_idx, start in enumerate(range(0, len(requests_lines), PAPERS_PER_BATCH)):
        chunk = requests_lines[start:start + PAPERS_PER_BATCH]
        fname = f"extract_abstract_{chunk_idx:03d}.jsonl"
        fpath = pdir / fname
        fpath.write_text("".join(chunk))
        size_mb = round(fpath.stat().st_size / 1e6, 2)
        batch_files.append({"file": fname, "count": len(chunk), "size_mb": size_mb})
        print(f"  {fname}: {len(chunk)} requests, {size_mb} MB")

    manifest = {
        "tag": tag,
        "model": MODEL,
        "kind": "abstract_batch",
        "total_requests": len(requests_lines),
        "skipped_no_abstract": n_skip_no_abstract,
        "skipped_done": n_skip_done,
        "files": batch_files,
        "paper_dois": paper_dois,
        "created_at": datetime.now().isoformat(),
    }
    (pdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {pdir / 'manifest.json'}")
    print(f"Submit with: python src/batch_extract_abstracts.py submit --tag {tag}")


# ── SUBMIT / STATUS / COLLECT — share machinery with batch_extract_arxiv ─────


def _submit_one_file(pdir: Path, fname: str) -> dict:
    fpath = pdir / fname
    if not fpath.exists():
        return {"status": "missing", "error": "file not found"}
    upload_resp = _curl_json("POST", "/files", form_fields={
        "purpose": "batch",
        "provider_file_name": fname,
        "provider_model": MODEL,
    }, file_path=str(fpath))
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
    pdir = pipeline_dir(args.tag)
    manifest_path = pdir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.json at {manifest_path}; run prepare first")
    manifest = json.loads(manifest_path.read_text())
    tracker_path = pdir / "tracker.json"
    tracker = json.loads(tracker_path.read_text()) if tracker_path.exists() else {}

    to_submit = [e for e in manifest["files"]
                 if e["file"] not in tracker or
                 tracker[e["file"]].get("status") in ("failed", "missing")]
    print(f"To submit: {len(to_submit)} files (already done: {len(manifest['files']) - len(to_submit)})")
    if not to_submit:
        return
    for e in to_submit:
        rec = _submit_one_file(pdir, e["file"])
        tracker[e["file"]] = rec
        print(f"  {e['file']}: status={rec.get('status')} batch_id={rec.get('batch_id')}")
        tracker_path.write_text(json.dumps(tracker, indent=2))


def cmd_status(args):
    pdir = pipeline_dir(args.tag)
    tracker_path = pdir / "tracker.json"
    if not tracker_path.exists():
        raise SystemExit("no tracker.json")
    tracker = json.loads(tracker_path.read_text())
    summary: dict[str, int] = {}
    for fname, info in tracker.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            summary["no_batch_id"] = summary.get("no_batch_id", 0) + 1
            continue
        resp = _curl_json("GET", f"/batches/{batch_id}")
        new_status = resp.get("status", "unknown")
        counts = resp.get("request_counts") or {}
        info["status"] = new_status
        info["request_counts"] = counts
        info["output_file_id"] = resp.get("output_file_id")
        summary[new_status] = summary.get(new_status, 0) + 1
        completed = counts.get("completed") or 0
        total = counts.get("total") or 0
        print(f"  {fname}: {new_status} ({completed}/{total})")
        time.sleep(0.3)
    tracker_path.write_text(json.dumps(tracker, indent=2))
    print(f"\nSummary: {json.dumps(summary)}")


def _parse_vertex_line(line: dict, paper_dois: dict, n_404: int = 0) -> tuple[str, dict | None]:
    """Parse a /files-format Vertex predictions.jsonl line into a deep_results entry."""
    cid = line.get("requestId") or ""
    if not cid:
        return ("", None)
    resp = line.get("response") or {}
    cands = resp.get("candidates") or []
    if not cands:
        return (cid, None)
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        return (cid, None)
    text = parts[0].get("text", "")
    if not text:
        return (cid, None)
    try:
        parsed = json.loads(text)
    except Exception:
        return (cid, None)
    usage = resp.get("usageMetadata") or {}
    return (cid, {
        "doi": paper_dois.get(cid, ""),
        "num_claims": len(parsed.get("claims") or []),
        "extraction_model": "gemini-3.1-pro-preview",
        "extraction_version": "deep_v1_abstract",
        "collected_at": datetime.now().isoformat(),
        "usage": {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        },
        "data": parsed,
    })


def _parse_openai_line(line: dict, paper_dois: dict) -> tuple[str, dict | None]:
    """Parse a /batches/{id}/output OpenAI-format line."""
    cid = line.get("custom_id") or ""
    if not cid:
        return ("", None)
    response = line.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return (cid, None)
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return (cid, None)
    try:
        parsed = json.loads(content)
    except Exception:
        return (cid, None)
    usage = body.get("usage") or {}
    return (cid, {
        "doi": paper_dois.get(cid, ""),
        "num_claims": len(parsed.get("claims") or []),
        "extraction_model": "gemini-3.1-pro-preview",
        "extraction_version": "deep_v1_abstract",
        "collected_at": datetime.now().isoformat(),
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "data": parsed,
    })


def _looks_html(text: str) -> bool:
    head = (text or "")[:120].lstrip().lower()
    return head.startswith("<html") or head.startswith("<!doctype") or "bad gateway" in head


def _download_with_timeout(path: str, timeout_s: int = 300) -> str | None:
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-sS", "--max-time", str(timeout_s), "-X", "GET",
           "-H", f"x-portkey-api-key: {api_key}",
           "-H", f"x-portkey-provider: {PROVIDER}",
           f"{GATEWAY}{path}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s + 60)
        if proc.stdout.strip() and not _looks_html(proc.stdout):
            return proc.stdout
    except Exception:
        pass
    return None


def cmd_collect(args):
    pdir = pipeline_dir(args.tag)
    manifest_path = pdir / "manifest.json"
    tracker_path = pdir / "tracker.json"
    if not (manifest_path.exists() and tracker_path.exists()):
        raise SystemExit("missing manifest.json or tracker.json")
    manifest = json.loads(manifest_path.read_text())
    tracker = json.loads(tracker_path.read_text())
    paper_dois = manifest.get("paper_dois") or {}
    out_dir = pdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    to_collect = [(f, i) for f, i in tracker.items()
                  if i.get("status") == "completed" and not i.get("collected")]
    print(f"To collect: {len(to_collect)}")

    saved = parse_fail = dl_fail = 0
    for fname, info in to_collect:
        # Prefer /files endpoint (reliable Vertex format) if we have output_file_id
        ofid = info.get("output_file_id")
        if ofid:
            raw = _download_with_timeout(f"/files/{ofid}/content", timeout_s=600)
            parser = _parse_vertex_line
        else:
            raw = _download_with_timeout(
                f"/batches/{info['batch_id']}/output", timeout_s=300)
            parser = _parse_openai_line

        if not raw:
            dl_fail += 1
            print(f"  {fname}: download failed")
            continue
        (out_dir / fname).write_text(raw)

        for raw_line in raw.strip().splitlines():
            if not raw_line.strip():
                continue
            try:
                line = json.loads(raw_line)
            except Exception:
                parse_fail += 1
                continue
            cid, result = parser(line, paper_dois)
            if not cid or result is None:
                parse_fail += 1
                continue
            rp = RESULTS_DIR / f"{cid}.json"
            if rp.exists():
                continue
            rp.write_text(json.dumps(result, ensure_ascii=False))
            saved += 1

        info["collected"] = True
        info["collected_at"] = datetime.now().isoformat()
        tracker_path.write_text(json.dumps(tracker, indent=2))

    print(f"\nCollect complete: saved={saved}  parse_fail={parse_fail}  dl_fail={dl_fail}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Build Gemini Batch JSONL from a paper list")
    p_prep.add_argument("--input", required=True,
                        help="JSONL file with {doi, title, abstract, ...} per line")
    p_prep.add_argument("--tag", required=True,
                        help="Pipeline tag, e.g. 'recover_arxiv_oversize'")
    p_prep.add_argument("--max", type=int, default=0,
                        help="Cap number of papers (debug)")

    p_sub = sub.add_parser("submit", help="Upload + submit batches")
    p_sub.add_argument("--tag", required=True)

    p_st = sub.add_parser("status", help="Poll Vertex Batch status")
    p_st.add_argument("--tag", required=True)

    p_col = sub.add_parser("collect", help="Download outputs + write deep_results")
    p_col.add_argument("--tag", required=True)

    args = p.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
