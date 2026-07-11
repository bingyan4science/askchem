"""
Classify claims from deep_results/ using Gemini (PortKey) with GPT fallback.

Processes papers serially. For each claim, uses the full L1/L2/L3 taxonomy
in a single-call classification via FULL_CLASSIFICATION_SYSTEM_PROMPT.

Usage:
    python src/classify_bcd_claims.py               # Classify all papers
    python src/classify_bcd_claims.py --max 10       # Limit to 10 papers
    python src/classify_bcd_claims.py --resume       # Skip already-classified papers
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from portkey_ai import Portkey
from openai import OpenAI
from askchem.taxonomy import (
    FULL_CLASSIFICATION_SYSTEM_PROMPT,
    ALL_CONTENT_VIEWS,
    build_classification_prompt,
    normalize_path,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "deep_results"

GEMINI_MODEL = "@vertexai-gemini-kc119-2/gemini-3.1-pro-preview"
GPT_FALLBACK_MODEL = "gpt-5-mini"


def get_portkey_client():
    return Portkey(
        base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
        api_key=os.environ["PORTKEY_API_KEY"],
    )


def get_openai_client():
    return OpenAI()


def classify_claim(claim: dict, doi: str, portkey_client, openai_client) -> Optional[dict]:
    claim_type = claim.get("claim_type", "property")
    quote = (claim.get("verbatim_quote") or "")[:300]
    messages = [
        {"role": "system", "content": FULL_CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_classification_prompt(claim_type, quote, doi)},
    ]

    for attempt in range(3):
        try:
            resp = portkey_client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=messages,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            return _normalize_classification(parsed)
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"      Gemini failed: {str(e)[:60]}, trying GPT fallback")

    try:
        resp = openai_client.chat.completions.create(
            model=GPT_FALLBACK_MODEL,
            messages=messages,
            max_completion_tokens=2048,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return _normalize_classification(parsed)
    except Exception as e:
        print(f"      GPT fallback failed: {str(e)[:60]}")
        return None


def _normalize_classification(parsed: dict) -> dict:
    normalized = {}
    for view_id in ALL_CONTENT_VIEWS:
        raw_path = parsed.get(view_id)
        if raw_path and raw_path != ["not_applicable"]:
            normed = normalize_path(view_id, raw_path)
            if normed:
                normalized[view_id] = normed
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Classify BCD claims")
    parser.add_argument("--max", type=int, help="Limit number of papers to process")
    parser.add_argument("--resume", action="store_true", help="Skip already-classified papers")
    args = parser.parse_args()

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    print(f"Total result files: {len(result_files)}")

    papers = []
    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
            if data.get("num_claims", 0) > 0:
                papers.append((rf, data))
        except Exception:
            pass

    if args.resume:
        papers = [(rf, d) for rf, d in papers if "classified_at" not in d]
        print(f"After skipping classified: {len(papers)}")

    if args.max:
        papers = papers[:args.max]
        print(f"Limited to: {args.max}")

    print(f"Papers to classify: {len(papers)}")

    portkey_client = get_portkey_client()
    openai_client = get_openai_client()

    total_classified = 0

    for pi, (result_file, data) in enumerate(papers):
        cid = data.get("custom_id", result_file.stem)
        doi = data.get("doi", "")
        claims = data.get("data", {}).get("claims", [])
        classified_count = 0

        for claim in claims:
            if "classification" in claim:
                classified_count += 1
                continue

            classification = classify_claim(claim, doi, portkey_client, openai_client)
            if classification:
                claim["classification"] = classification
                classified_count += 1

            time.sleep(3)

        data["classified_at"] = datetime.now().isoformat()
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)

        total_classified += classified_count
        print(f"[{pi+1}/{len(papers)}] {cid}: classified {classified_count} claims")

    print(f"\n{'='*60}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Papers processed: {len(papers)}")
    print(f"Total claims classified: {total_classified}")


if __name__ == "__main__":
    main()
