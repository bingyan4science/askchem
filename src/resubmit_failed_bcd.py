"""
Resubmit failed BCD paper extractions via Gemini (PortKey) with GPT fallback.

Fallback chain per paper:
  1. Gemini + image_url (PDF as base64) — 3 attempts, exponential backoff
  2. Gemini + text (extract text via GPT-mini, send to Gemini) — 1 attempt
  3. GPT-5.4 + file (native PDF support) — 1 attempt

Usage:
    python src/resubmit_failed_bcd.py              # Process all 488 failed papers
    python src/resubmit_failed_bcd.py --max 10      # Limit to 10 papers
    python src/resubmit_failed_bcd.py --resume      # Skip already-done papers
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from portkey_ai import Portkey
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
FAILED_FILE = DATA_DIR / "tiers_bcd_pipeline" / "failed_papers.json"
RESULTS_DIR = DATA_DIR / "deep_results"

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


def get_portkey_client():
    return Portkey(
        base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
        api_key=os.environ["PORTKEY_API_KEY"],
    )


def get_openai_client():
    return OpenAI()


def load_pdf_b64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def try_gemini_image(client, pdf_b64: str) -> Optional[dict]:
    """Attempt 1: Gemini with PDF as base64 image_url. 3 retries with exponential backoff."""
    delays = [5, 15, 45]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_b64}"}},
                    ],
                }],
                max_completion_tokens=65536,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            parsed = json.loads(text)
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
                print(f"    gemini_image attempt {attempt+1} failed: {str(e)[:80]}, retrying in {delays[attempt]}s")
                time.sleep(delays[attempt])
            else:
                print(f"    gemini_image failed after 3 attempts: {str(e)[:80]}")
    return None


def extract_text_via_gpt_mini(openai_client, pdf_b64: str) -> Optional[str]:
    """Use GPT-mini to extract text from a PDF for the text fallback."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text content from this PDF. Return the full text, preserving tables and figures as best you can."},
                    {"type": "file", "file": {"filename": "paper.pdf", "file_data": f"data:application/pdf;base64,{pdf_b64}"}},
                ],
            }],
            max_completion_tokens=32768,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"    text extraction failed: {str(e)[:80]}")
        return None


def try_gemini_text(client, openai_client, pdf_b64: str) -> Optional[dict]:
    """Attempt 2: Extract text via GPT-mini, then send text to Gemini."""
    text_content = extract_text_via_gpt_mini(openai_client, pdf_b64)
    if not text_content:
        return None

    try:
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{
                "role": "user",
                "content": f"{EXTRACTION_PROMPT}\n\n--- PAPER TEXT ---\n{text_content}",
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
    """Attempt 3: GPT-5.4 with native PDF file support."""
    try:
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "file", "file": {"filename": "paper.pdf", "file_data": f"data:application/pdf;base64,{pdf_b64}"}},
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


def process_paper(paper: dict, portkey_client, openai_client) -> Optional[dict]:
    pdf_path = paper["pdf_path"]
    if not Path(pdf_path).exists():
        abs_path = Path(__file__).parent.parent / pdf_path
        if abs_path.exists():
            pdf_path = str(abs_path)
        else:
            print(f"    PDF not found: {pdf_path}")
            return None

    pdf_b64 = load_pdf_b64(pdf_path)

    result = try_gemini_image(portkey_client, pdf_b64)
    if result:
        return result

    result = try_gemini_text(portkey_client, openai_client, pdf_b64)
    if result:
        return result

    result = try_gpt_file(openai_client, pdf_b64)
    return result


def main():
    parser = argparse.ArgumentParser(description="Resubmit failed BCD extractions")
    parser.add_argument("--max", type=int, help="Limit number of papers to process")
    parser.add_argument("--resume", action="store_true", help="Skip papers with existing results")
    args = parser.parse_args()

    with open(FAILED_FILE) as f:
        papers = json.load(f)
    print(f"Failed papers loaded: {len(papers)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done = {f.stem for f in RESULTS_DIR.glob("*.json")}
        papers = [p for p in papers if p["custom_id"] not in done]
        print(f"After skipping done: {len(papers)}")

    if args.max:
        papers = papers[:args.max]
        print(f"Limited to: {args.max}")

    portkey_client = get_portkey_client()
    openai_client = get_openai_client()

    succeeded = 0
    failed = 0
    method_counts = {}

    for i, paper in enumerate(papers):
        cid = paper["custom_id"]
        size_kb = paper.get("pdf_size_kb", 0)

        result_path = RESULTS_DIR / f"{cid}.json"
        if result_path.exists():
            print(f"[{i+1}/{len(papers)}] {cid} ({size_kb} KB) -> already done, skipping")
            continue

        result = process_paper(paper, portkey_client, openai_client)

        if result:
            parsed = result["data"]
            claims = parsed.get("claims", [])
            method = result["method"]
            method_counts[method] = method_counts.get(method, 0) + 1

            result_data = {
                "doi": paper["doi"],
                "custom_id": cid,
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

            print(f"[{i+1}/{len(papers)}] {cid} ({size_kb} KB) -> {method}: {len(claims)} claims")
            succeeded += 1
        else:
            print(f"[{i+1}/{len(papers)}] {cid} ({size_kb} KB) -> ALL METHODS FAILED")
            failed += 1

        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"RESUBMIT COMPLETE")
    print(f"{'='*60}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")


if __name__ == "__main__":
    main()
