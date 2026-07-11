"""Deep extraction via Gemini through the Portkey gateway.

Mirrors ``src/extract_tier_a.py``'s contract: writes one JSON result per
paper to ``data/deep_results/<custom_id>.json`` so ``src/integrate_deep.py``
can pick it up unchanged.

Differences from extract_tier_a.py:
  * Uses Gemini (default ``gemini-3.1-pro-preview``) via the Portkey
    gateway (``PORTKEY_API_KEY``) instead of OpenAI Batch.
  * Concurrent realtime calls (no async batch jobs).
  * Resumable — already-extracted custom_ids are skipped on restart.

Inputs:
  * A JSONL job file with at minimum ``doi`` per line (e.g.
    ``data/audits/to_deep_extract.jsonl``).  Other fields are ignored.
  * PDFs must already exist in ``data/papers_full/<sha256(doi)[:16]>.pdf``.

Usage:
    PORTKEY_API_KEY=... python src/extract_gemini_batch.py \
        --jobs data/audits/to_deep_extract.jsonl
    # restrict to first N papers:
    python src/extract_gemini_batch.py --jobs ... --limit 5
    # status only:
    python src/extract_gemini_batch.py --status
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers_full"
RESULTS_DIR = REPO_ROOT / "data" / "deep_results"
LOG_DIR = REPO_ROOT / "data" / "gemini_extract_logs"

# Portkey gateway configuration (matches scripts/gemini_verify_*)
GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

# These are reused verbatim from src/extract_tier_a.py so the schema is
# identical.  Kept in this file to avoid a fragile cross-module import.
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
      "claim_id": sequential number,
      "claim_type": "reaction|property|method|mechanism|comparison|scope_entry|computational_result|structure|hypothesis|experimental_design|limitation|future_direction|surprising_finding",
      "confidence": "high|medium|low",
      "location_in_paper": "Table 1, entry 3" or "Figure 2" or "Results, paragraph 4",

      // FOR REACTIONS (including each scope entry as a separate claim):
      "reaction_type": "e.g., Suzuki coupling, C-H activation, MOF synthesis",
      "reactants": [
        {"name": "...", "smiles": "... or null if not determinable", "role": "substrate|reagent|catalyst|ligand|additive"}
      ],
      "products": [
        {"name": "...", "smiles": "... or null", "role": "major|minor|byproduct"}
      ],
      "conditions": {
        "catalyst": "...", "ligand": "...", "solvent": "...",
        "temperature": "...", "time": "...", "atmosphere": "...",
        "additives": ["..."], "concentration": "...", "other": "..."
      },
      "outcomes": {
        "yield_percent": number or null,
        "ee_percent": number or null,
        "dr": "...",
        "selectivity": "...",
        "conversion_percent": number or null,
        "turnover_number": number or null
      },
      "is_key_result": true/false,

      // FOR PROPERTIES:
      "subject": "molecule/material name",
      "subject_smiles": "...",
      "property_name": "e.g., melting point, BET surface area, IC50",
      "property_category": "physical|chemical|biological|spectroscopic|electrochemical|mechanical|optical|thermal",
      "value": "numerical value with units",
      "measurement_method": "instrument/technique",

      // FOR MECHANISMS:
      "process_described": "what reaction/process",
      "steps": ["step 1", "step 2"],
      "key_intermediates": ["..."],
      "evidence": [{"type": "...", "description": "..."}],

      // FOR METHODS:
      "technique_name": "name",
      "what_it_achieves": "description",
      "key_innovation": "what's new",

      // FOR COMPARISONS:
      "compared_items": ["item A", "item B"],
      "metric": "what's being compared",
      "comparison_result": "A is better/worse/equal to B by X",

      // FOR HYPOTHESIS:
      "hypothesis_text": "The specific hypothesis being tested",

      // FOR LIMITATION:
      "limitation_text": "The specific limitation described",

      // FOR FUTURE_DIRECTION:
      "direction_text": "The specific future direction suggested",

      // FOR SURPRISING_FINDING:
      "finding_text": "The unexpected result",
      "why_surprising": "Why this is unexpected given prior knowledge",

      // FOR ALL:
      "verbatim_quote": "exact sentence(s) from paper supporting this claim"
    }
  ]
}

CRITICAL INSTRUCTIONS:
1. Extract EVERY entry from substrate scope tables — each row is a separate claim
2. Extract EVERY entry from optimization tables — each row is a separate claim
3. Extract ALL characterization data (NMR, IR, MS, XRD, etc.)
4. Extract ALL numerical results from figures where readable
5. Include control experiments and negative results
6. Extract hypotheses from the introduction
7. Extract limitations from the discussion/conclusion
8. Extract future directions from the conclusion
9. Flag any surprising or counter-intuitive findings
10. A typical paper should yield 20-50 claims. If you have fewer than 15, you are likely missing data.

Respond with ONLY the JSON object — no surrounding prose, no markdown fences."""


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def doi_to_filename(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


def looks_like_complete_pdf(path: Path) -> bool:
    """Quick header+trailer sanity check. Catches truncated downloads
    BEFORE we spend a Gemini call on them."""
    try:
        with path.open("rb") as f:
            head = f.read(8)
            if not head.startswith(b"%PDF"):
                return False
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - 1024))
            tail = f.read()
            return b"%%EOF" in tail
    except OSError:
        return False


_clients_lock = threading.Lock()
_clients: dict[int, object] = {}


def _client(api_key: str):
    """One OpenAI client per worker thread."""
    tid = threading.get_ident()
    if tid not in _clients:
        with _clients_lock:
            if tid not in _clients:
                from openai import OpenAI
                _clients[tid] = OpenAI(
                    base_url=GATEWAY,
                    api_key=api_key,
                    default_headers={"x-portkey-provider": PROVIDER},
                    timeout=600.0,
                )
    return _clients[tid]


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # ```json ... ```  or  ``` ... ```
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return s


def _parse_json_response(text: str) -> dict:
    """Tolerant JSON parser for slightly-malformed model output."""
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Greedy: find the largest balanced JSON object substring.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"could not parse JSON; first 200 chars: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Per-paper extraction
# --------------------------------------------------------------------------- #

def build_messages(pdf_path: Path) -> list[dict]:
    """Build the multimodal Gemini message containing the PDF inline.

    Portkey's OpenAI-compat shim routes ``{"type": "image_url", ...}`` blocks
    with a data URL into Gemini's ``inline_data`` field, including PDF MIME
    types.  This is the same pattern used for image inputs.
    """
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": EXTRACTION_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:application/pdf;base64,{b64}"}},
        ],
    }]


def extract_one(doi: str, pdf_path: Path, api_key: str, model: str,
                max_retries: int = 5) -> tuple[bool, dict | str]:
    """Returns (ok, payload).  payload is the parsed dict on success, an
    error string on failure."""
    client = _client(api_key)
    messages = build_messages(pdf_path)

    backoff = 5.0
    last_err = ""
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=32768,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - t0
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)

            try:
                parsed = _parse_json_response(text)
            except ValueError as e:
                last_err = f"parse: {e}"
                # Sometimes Gemini ignores response_format — retry without it.
                time.sleep(backoff); backoff = min(backoff * 2, 60); continue

            claims = parsed.get("claims", []) or []
            paper_knowledge = parsed.get("paper_knowledge", {}) or {}

            return True, {
                "doi": doi,
                "custom_id": doi_to_filename(doi),
                "num_claims": len(claims),
                "collected_at": datetime.now().isoformat(),
                "extraction_model": model,
                "extraction_method": "gemini_portkey_realtime",
                "model": model,
                "elapsed_s": round(elapsed, 1),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                },
                "data": {
                    "paper_knowledge": paper_knowledge,
                    "claims": claims,
                },
            }
        except Exception as e:
            err = str(e)
            last_err = err[:300]
            is_rate = "429" in err or "quota" in err.lower() or "rate" in err.lower()
            is_transient = is_rate or "timeout" in err.lower() or "connection" in err.lower() or "503" in err or "500" in err
            if attempt < max_retries - 1 and is_transient:
                wait = backoff * (2 if is_rate else 1)
                wait = min(wait, 120)
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
                continue
            return False, last_err
    return False, last_err or "unknown"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def load_jobs(jobs_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jobs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def cmd_run(args) -> int:
    api_key = os.environ.get("PORTKEY_API_KEY", "")
    if not api_key:
        print("ERROR: PORTKEY_API_KEY not set", file=sys.stderr)
        return 1

    if not args.jobs:
        print("ERROR: --jobs is required (path to a JSONL with doi entries)",
              file=sys.stderr)
        return 1

    jobs_path = Path(args.jobs)
    if not jobs_path.exists():
        print(f"ERROR: jobs file not found: {jobs_path}", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jobs(jobs_path)
    print(f"loaded {len(rows):,} jobs from {jobs_path}", flush=True)

    already_done = {f.stem for f in RESULTS_DIR.glob("*.json")}
    print(f"  already extracted (skip): {len(already_done):,}", flush=True)

    queue: list[tuple[str, Path]] = []
    missing_pdf = 0
    corrupt_pdf = 0
    corrupt_log = LOG_DIR / "corrupt_pdfs.jsonl"
    corrupt_log_fh = corrupt_log.open("a")
    for r in rows:
        doi = r.get("doi") or r.get("DOI")
        if not doi:
            continue
        cid = doi_to_filename(doi)
        if cid in already_done:
            continue
        pdf_path = PAPERS_DIR / f"{cid}.pdf"
        if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
            missing_pdf += 1
            continue
        if not looks_like_complete_pdf(pdf_path):
            corrupt_pdf += 1
            corrupt_log_fh.write(json.dumps({
                "doi": doi, "pdf": str(pdf_path),
                "size": pdf_path.stat().st_size,
                "ts": datetime.now().isoformat(),
            }) + "\n")
            continue
        queue.append((doi, pdf_path))
    corrupt_log_fh.close()

    print(f"  jobs with no PDF on disk : {missing_pdf:,}  (skipped)", flush=True)
    print(f"  jobs with corrupt PDF    : {corrupt_pdf:,}  (skipped, logged)", flush=True)
    print(f"  jobs to extract (pre-shard): {len(queue):,}", flush=True)

    if args.total > 1:
        queue = [q for i, q in enumerate(queue) if i % args.total == args.shard]
        print(f"  shard {args.shard}/{args.total} -> {len(queue):,} jobs", flush=True)

    if args.limit and args.limit > 0:
        queue = queue[: args.limit]
        print(f"  --limit applied → {len(queue):,}", flush=True)

    if not queue:
        print("nothing to do.", flush=True)
        return 0

    fail_log_path = LOG_DIR / f"failures_{int(time.time())}.jsonl"
    fail_log = fail_log_path.open("w")
    fail_lock = threading.Lock()

    progress = {"ok": 0, "fail": 0, "claims": 0, "started": time.time()}
    progress_lock = threading.Lock()

    def _do(doi_pdf: tuple[str, Path]) -> None:
        doi, pdf_path = doi_pdf
        ok, payload = extract_one(doi, pdf_path, api_key, args.model)
        if ok:
            cid = payload["custom_id"]
            (RESULTS_DIR / f"{cid}.json").write_text(json.dumps(payload, indent=2))
            with progress_lock:
                progress["ok"] += 1
                progress["claims"] += payload["num_claims"]
        else:
            with fail_lock:
                fail_log.write(json.dumps({
                    "doi": doi,
                    "pdf": str(pdf_path),
                    "error": payload,
                    "ts": datetime.now().isoformat(),
                }) + "\n")
                fail_log.flush()
            with progress_lock:
                progress["fail"] += 1

        with progress_lock:
            done = progress["ok"] + progress["fail"]
            if done % 10 == 0 or done == len(queue):
                el = time.time() - progress["started"]
                rate = done / max(el, 1)
                rem = (len(queue) - done) / max(rate, 0.01)
                print(
                    f"  [{done}/{len(queue)}] ok={progress['ok']} fail={progress['fail']} "
                    f"claims={progress['claims']:,} | {rate:.2f} pp/s | "
                    f"ETA {rem/60:.0f}m", flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_do, dp) for dp in queue]
        for _ in as_completed(futs):
            pass

    fail_log.close()

    el = time.time() - progress["started"]
    print(f"\nDONE in {el/60:.1f} min", flush=True)
    print(f"  succeeded: {progress['ok']:,}  ({progress['claims']:,} claims)", flush=True)
    print(f"  failed   : {progress['fail']:,}  -> {fail_log_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


def cmd_status(args) -> int:
    n = len(list(RESULTS_DIR.glob("*.json"))) if RESULTS_DIR.exists() else 0
    print(f"deep_results files: {n:,}")
    if LOG_DIR.exists():
        for f in sorted(LOG_DIR.glob("failures_*.jsonl")):
            try:
                n_fail = sum(1 for _ in f.open())
            except OSError:
                n_fail = -1
            print(f"  {f.relative_to(REPO_ROOT)}: {n_fail:,} failures")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Gemini-batch deep extraction")
    p.add_argument("--jobs", help="JSONL with at minimum a 'doi' field per line")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--workers", type=int, default=6,
                   help="Concurrent workers (default 6)")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of papers (0 = all)")
    p.add_argument("--shard", type=int, default=0,
                   help="This worker's shard index (0-based). Picks job N where "
                        "N % --total == --shard. Lets you run multiple "
                        "extractor processes safely without DOI collisions.")
    p.add_argument("--total", type=int, default=1,
                   help="Number of shards (default 1). Only effective with --shard.")
    p.add_argument("--status", action="store_true",
                   help="Print results-dir status and exit")
    args = p.parse_args()

    if args.status:
        return cmd_status(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
