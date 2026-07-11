"""
Extract claims from arXiv papers using Gemini via PortKey.

Downloads arXiv PDFs and uses the Gemini-first fallback chain:
  1. Gemini + image_url (PDF as base64) — 3 attempts, exponential backoff
  2. Gemini + text (extract text via GPT-mini, send to Gemini) — 1 attempt
  3. GPT-5.4 + file (native PDF support) — 1 attempt

Usage:
    python src/extract_arxiv.py                # Process all ingested arXiv papers
    python src/extract_arxiv.py --max 10       # Limit to 10 papers
    python src/extract_arxiv.py --skip-download # Skip PDF download (use existing)
"""

import argparse
import base64
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests as http_req
from portkey_ai import Portkey
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "deep_results"
PDF_DIR = DATA_DIR / "papers_full"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"

GEMINI_MODEL = "@vertexai-gemini-kc119-2/gemini-3.1-pro-preview"
GPT_MODEL = "gpt-5.4"

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


def get_db_path():
    return Path(os.environ.get("CHEMTREE_DB", str(DB_PATH)))


def get_portkey_client():
    return Portkey(
        base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
        api_key=os.environ["PORTKEY_API_KEY"],
    )


def get_openai_client():
    return OpenAI()


def custom_id_for_doi(doi: str) -> str:
    return hashlib.sha256(doi.encode()).hexdigest()[:16]


# ── PDF download ──────────────────────────────────────────────────────────────

def download_pdf(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            resp = http_req.get(
                url, timeout=60,
                headers={"User-Agent": "AskChem/1.0 (research; mailto:askchem@nyu.edu)"},
            )
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"    PDF download failed: {str(e)[:80]}")
    return False


# ── Extraction methods (same chain as resubmit_failed_bcd.py) ────────────────

def load_pdf_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def try_gemini_image(client, pdf_b64: str) -> Optional[dict]:
    delays = [5, 15, 45]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:application/pdf;base64,{pdf_b64}"}},
                    ],
                }],
                max_completion_tokens=65536,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            usage = resp.usage
            return {
                "data": parsed,
                "method": "gemini_image",
                "model": GEMINI_MODEL,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
            }
        except Exception as e:
            if attempt < 2:
                print(f"    gemini_image attempt {attempt+1} failed: "
                      f"{str(e)[:80]}, retry in {delays[attempt]}s")
                time.sleep(delays[attempt])
            else:
                print(f"    gemini_image failed after 3 attempts: {str(e)[:80]}")
    return None


def extract_text_via_gpt_mini(openai_client, pdf_b64: str) -> Optional[str]:
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Extract all text content from this PDF. "
                        "Return the full text, preserving tables and figures as best you can."
                    )},
                    {"type": "file", "file": {
                        "filename": "paper.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}"}},
                ],
            }],
            max_completion_tokens=32768,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"    text extraction failed: {str(e)[:80]}")
        return None


def try_gemini_text(client, openai_client, pdf_b64: str) -> Optional[dict]:
    text = extract_text_via_gpt_mini(openai_client, pdf_b64)
    if not text:
        return None
    try:
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{
                "role": "user",
                "content": f"{EXTRACTION_PROMPT}\n\n--- PAPER TEXT ---\n{text}",
            }],
            max_completion_tokens=65536,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        usage = resp.usage
        return {
            "data": parsed,
            "method": "gemini_text",
            "model": GEMINI_MODEL,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            },
        }
    except Exception as e:
        print(f"    gemini_text failed: {str(e)[:80]}")
        return None


def try_gpt_file(openai_client, pdf_b64: str) -> Optional[dict]:
    try:
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "file", "file": {
                        "filename": "paper.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}"}},
                ],
            }],
            max_completion_tokens=65536,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        usage = resp.usage
        return {
            "data": parsed,
            "method": "gpt_file",
            "model": GPT_MODEL,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            },
        }
    except Exception as e:
        print(f"    gpt_file failed: {str(e)[:80]}")
        return None


def process_paper(pdf_path: str, portkey_client, openai_client) -> Optional[dict]:
    pdf_b64 = load_pdf_b64(pdf_path)

    result = try_gemini_image(portkey_client, pdf_b64)
    if result:
        return result

    result = try_gemini_text(portkey_client, openai_client, pdf_b64)
    if result:
        return result

    return try_gpt_file(openai_client, pdf_b64)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_arxiv_papers(tier=None):
    """Load arXiv papers from DB, optionally filtered by tier."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT doi, title, open_access_url, data FROM sources "
        "WHERE venue LIKE 'arXiv:%'"
    ).fetchall()
    conn.close()

    papers = []
    for r in rows:
        data = {}
        if r["data"]:
            try:
                data = json.loads(r["data"])
            except json.JSONDecodeError:
                pass
        if tier is not None and data.get("tier") != tier:
            continue
        papers.append({
            "doi": r["doi"],
            "title": r["title"],
            "pdf_url": r["open_access_url"] or data.get("pdf_url", ""),
            "arxiv_id": data.get("arxiv_id", ""),
            "custom_id": custom_id_for_doi(r["doi"]),
            "tier": data.get("tier"),
            "citation_count": data.get("citationCount", 0),
        })

    # Within a tier, process highest-cited first
    papers.sort(key=lambda p: -(p.get("citation_count") or 0))
    return papers


def load_arxiv_papers_from_tier_file(tier: int):
    """Load papers from a per-tier JSONL file (faster than DB scan)."""
    tier_file = DATA_DIR / "arxiv_harvest" / f"tier_{tier}.jsonl"
    if not tier_file.exists():
        print(f"Tier file not found: {tier_file}")
        return []
    papers = []
    with open(tier_file) as f:
        for line in f:
            try:
                p = json.loads(line)
                doi = p.get("doi") or f"10.48550/arXiv.{p['arxiv_id']}"
                papers.append({
                    "doi": doi,
                    "title": p.get("title", ""),
                    "pdf_url": p.get("pdf_url", ""),
                    "arxiv_id": p.get("arxiv_id", ""),
                    "custom_id": custom_id_for_doi(doi),
                    "tier": p.get("tier", tier),
                    "citation_count": p.get("citation_count", 0),
                })
            except (json.JSONDecodeError, KeyError):
                pass
    papers.sort(key=lambda x: -(x.get("citation_count") or 0))
    return papers


def main():
    parser = argparse.ArgumentParser(description="Extract claims from arXiv papers via Gemini")
    parser.add_argument("--max", type=int, help="Limit number of papers")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                        help="Only process papers from this tier (reads tier file)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only process papers whose PDFs are already downloaded")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if args.tier:
        papers = load_arxiv_papers_from_tier_file(args.tier)
        print(f"Tier {args.tier} papers: {len(papers):,}")
    else:
        papers = load_arxiv_papers()
        print(f"arXiv papers in DB: {len(papers):,}")

    done = {f.stem for f in RESULTS_DIR.glob("*.json")}
    papers = [p for p in papers if p["custom_id"] not in done]
    print(f"After skipping already extracted: {len(papers):,}")

    if args.max:
        papers = papers[:args.max]
        print(f"Limited to: {args.max}")

    if not papers:
        print("Nothing to process.")
        return

    portkey_client = get_portkey_client()
    openai_client = get_openai_client()

    succeeded = 0
    failed = 0
    download_failed = 0
    method_counts: dict[str, int] = {}

    for i, paper in enumerate(papers):
        cid = paper["custom_id"]
        doi = paper["doi"]
        pdf_url = paper["pdf_url"]

        result_path = RESULTS_DIR / f"{cid}.json"
        if result_path.exists():
            continue

        pdf_path = PDF_DIR / f"{cid}.pdf"
        if not pdf_path.exists():
            if args.skip_download:
                continue
            if not pdf_url:
                print(f"[{i+1}/{len(papers)}] {cid} -> no PDF URL, skipping")
                download_failed += 1
                continue
            print(f"[{i+1}/{len(papers)}] {cid} downloading PDF...", end=" ", flush=True)
            if not download_pdf(pdf_url, pdf_path):
                download_failed += 1
                continue
            print("ok", flush=True)
            time.sleep(1)

        result = process_paper(str(pdf_path), portkey_client, openai_client)

        if result:
            parsed = result["data"]
            claims = parsed.get("claims", [])
            method = result["method"]
            method_counts[method] = method_counts.get(method, 0) + 1

            result_data = {
                "doi": doi,
                "custom_id": cid,
                "arxiv_id": paper.get("arxiv_id", ""),
                "num_claims": len(claims),
                "collected_at": datetime.now().isoformat(),
                "extraction_model": result["model"],
                "extraction_method": method,
                "usage": result["usage"],
                "data": {
                    "paper_knowledge": parsed.get("paper_knowledge", {}),
                    "claims": claims,
                },
            }
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

            print(f"[{i+1}/{len(papers)}] {cid} -> {method}: {len(claims)} claims")
            succeeded += 1
        else:
            print(f"[{i+1}/{len(papers)}] {cid} -> ALL METHODS FAILED")
            failed += 1

        time.sleep(3)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Download failed: {download_failed}")
    for method, count in sorted(method_counts.items()):
        print(f"    {method}: {count}")


if __name__ == "__main__":
    main()
