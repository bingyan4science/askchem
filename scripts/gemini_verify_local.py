#!/usr/bin/env python3
"""Verify flagged contradiction pairs using Gemini (run locally).

Reads flagged pairs from JSON, verifies with Gemini via PortKey gateway,
and outputs verified results to JSON for upload to the server DB.
"""
import json
import os
import sys
import time
from pathlib import Path

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"


def main():
    flagged_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/paw_flagged_pairs.json")
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/gemini_verified.json")

    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        print("ERROR: PORTKEY_API_KEY not set")
        sys.exit(1)

    flagged = json.loads(flagged_file.read_text())
    print(f"Loaded {len(flagged)} flagged pairs from {flagged_file}", flush=True)

    import subprocess

    verified = []
    total = len(flagged)
    errors = 0

    for idx, cand in enumerate(flagged):
        prompt = (
            "You are verifying whether two chemistry claims contradict each other.\n\n"
            f"Claim A (DOI: {cand['doi_1']}): \"{cand['quote_1']}\"\n"
            f"Claim B (DOI: {cand['doi_2']}): \"{cand['quote_2']}\"\n\n"
            "Do these claims contradict each other? Respond with JSON only:\n"
            '{"verdict": "confirmed" or "rejected", "explanation": "one sentence why", '
            '"confidence": 0.0 to 1.0}'
        )

        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
        })

        for attempt in range(3):
            try:
                result = subprocess.run(
                    ["curl", "-s", "--max-time", "20",
                     "-X", "POST",
                     "-H", f"x-portkey-api-key: {api_key}",
                     "-H", f"x-portkey-provider: {PROVIDER}",
                     "-H", "Content-Type: application/json",
                     "-d", body,
                     f"{GATEWAY}/chat/completions"],
                    capture_output=True, text=True, timeout=25,
                )
                resp = json.loads(result.stdout)
                text = resp["choices"][0]["message"]["content"].strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(text)
                cand["gemini_verdict"] = parsed.get("verdict", "rejected")
                cand["gemini_explanation"] = parsed.get("explanation", "")
                cand["confidence"] = float(parsed.get("confidence", 0.5))
                verified.append(cand)
                break
            except json.JSONDecodeError:
                cand["gemini_verdict"] = "rejected"
                cand["gemini_explanation"] = f"Non-JSON: {result.stdout[:100] if 'result' in dir() else 'no response'}"
                cand["confidence"] = 0.0
                verified.append(cand)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(2 ** attempt * 5)
                    continue
                errors += 1
                if attempt == 2:
                    cand["gemini_verdict"] = "error"
                    cand["gemini_explanation"] = str(e)[:200]
                    cand["confidence"] = 0.0
                    verified.append(cand)
                else:
                    time.sleep(2)

        if (idx + 1) % 10 == 0:
            confirmed = sum(1 for v in verified if v.get("gemini_verdict") == "confirmed")
            print(f"  Gemini: {idx+1}/{total} verified, "
                  f"{confirmed} confirmed, {errors} errors", flush=True)

    confirmed = [v for v in verified if v.get("gemini_verdict") == "confirmed"]
    print(f"\nDone: {len(confirmed)} confirmed contradictions out of "
          f"{total} pairs ({errors} errors)", flush=True)

    output_file.write_text(json.dumps(verified, indent=2))
    print(f"Saved to {output_file}", flush=True)

    if confirmed:
        print(f"\nTop confirmed contradictions:", flush=True)
        for c in sorted(confirmed, key=lambda x: -x.get("confidence", 0))[:10]:
            print(f"  [{c.get('confidence', 0):.2f}] {c.get('subject', '')[:40]}:", flush=True)
            print(f"    A: {c['quote_1'][:80]}...", flush=True)
            print(f"    B: {c['quote_2'][:80]}...", flush=True)
            print(f"    → {c.get('gemini_explanation', '')}", flush=True)


if __name__ == "__main__":
    main()
