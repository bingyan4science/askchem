#!/usr/bin/env python3
"""Verify contradiction candidates with Gemini — batched & concurrent.

Reads candidate pairs from JSONL, sends batches of N pairs per Gemini call
using concurrent workers, outputs confirmed contradictions to JSON.

Usage:
    export PORTKEY_API_KEY=...
    python scripts/gemini_verify_batch.py [input.jsonl] [output.json]
    python scripts/gemini_verify_batch.py --resume  # continue from checkpoint
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-2.5-flash"

BATCH_SIZE = 5        # pairs per Gemini call
MAX_WORKERS = 4       # concurrent API calls
CHECKPOINT_EVERY = 50 # save checkpoint every N batches


def build_prompt(pairs: list[dict]) -> str:
    lines = [
        "You are a chemistry expert checking whether pairs of claims contradict "
        "each other. For each pair, decide if they make genuinely incompatible "
        "assertions about the same subject. Different measurements of different "
        "materials are NOT contradictions — only flag cases where two papers "
        "disagree about the same thing.\n"
    ]
    for i, p in enumerate(pairs, 1):
        lines.append(f"--- Pair {i} ---")
        lines.append(f"Claim A (DOI: {p['doi_1']}): \"{p['quote_1']}\"")
        lines.append(f"Claim B (DOI: {p['doi_2']}): \"{p['quote_2']}\"")
        lines.append("")

    lines.append(
        "For each pair, respond with a JSON array (one object per pair, in order):\n"
        '[{"pair": 1, "verdict": "confirmed"|"rejected", '
        '"explanation": "one sentence", "confidence": 0.0-1.0}, ...]\n'
        "Respond ONLY with the JSON array, no extra text."
    )
    return "\n".join(lines)


def _get_client(api_key: str):
    """Lazy-init thread-local OpenAI client."""
    import threading
    _local = threading.local()
    if not hasattr(_local, "client"):
        from openai import OpenAI
        _local.client = OpenAI(
            base_url=GATEWAY,
            api_key=api_key,
            default_headers={"x-portkey-provider": PROVIDER},
            timeout=30.0,
        )
    return _local.client


# Module-level client cache (one per thread via ThreadPoolExecutor)
_clients: dict[int, object] = {}
_clients_lock = __import__("threading").Lock()


def _client_for_thread(api_key: str):
    import threading
    tid = threading.get_ident()
    if tid not in _clients:
        with _clients_lock:
            if tid not in _clients:
                from openai import OpenAI
                _clients[tid] = OpenAI(
                    base_url=GATEWAY,
                    api_key=api_key,
                    default_headers={"x-portkey-provider": PROVIDER},
                    timeout=30.0,
                )
    return _clients[tid]


def _parse_results(text: str, n_pairs: int) -> list[dict]:
    """Robustly parse Gemini's JSON response."""
    import re
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try direct parse
    try:
        results = json.loads(text)
        if isinstance(results, dict):
            results = [results]
        return results
    except json.JSONDecodeError:
        pass

    # Try to extract individual JSON objects via regex
    results = []
    for m in re.finditer(r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text):
        try:
            results.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    return results


def call_gemini(pairs: list[dict], api_key: str, batch_idx: int) -> list[dict]:
    """Send a batch to Gemini and return annotated pairs."""
    client = _client_for_thread(api_key)
    prompt = build_prompt(pairs)

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000 + 300 * len(pairs),
            )
            text = resp.choices[0].message.content.strip()
            results = _parse_results(text, len(pairs))

            for i, p in enumerate(pairs):
                if i < len(results):
                    r = results[i]
                    p["gemini_verdict"] = r.get("verdict", "rejected")
                    p["gemini_explanation"] = r.get("explanation", "")
                    p["confidence"] = float(r.get("confidence", 0.5))
                else:
                    p["gemini_verdict"] = "error"
                    p["gemini_explanation"] = f"Missing from response ({len(results)}/{len(pairs)} parsed)"
                    p["confidence"] = 0.0
            return pairs

        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate" in err.lower() or "quota" in err.lower()
            is_conn = "connection" in err.lower() or "timeout" in err.lower()
            if attempt < 4:
                wait = 2 ** attempt * (10 if is_conn else 5 if is_rate else 2)
                wait = min(wait, 120)
                if attempt >= 2:
                    print(f"    Batch {batch_idx} attempt {attempt+1} error: {err[:80]}; "
                          f"retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            for p in pairs:
                p["gemini_verdict"] = "error"
                p["gemini_explanation"] = err[:200]
                p["confidence"] = 0.0
            return pairs

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="data/candidates_viewfree.jsonl")
    parser.add_argument("output", nargs="?", default="data/gemini_verified_viewfree.json")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="Max pairs to process (0=all)")
    args = parser.parse_args()

    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        print("ERROR: PORTKEY_API_KEY not set")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Load input
    pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs):,} candidate pairs from {input_path}", flush=True)

    if args.limit > 0:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs):,} pairs", flush=True)

    # Resume from checkpoint
    already_done = set()
    verified = []
    if args.resume and checkpoint_path.exists():
        verified = json.loads(checkpoint_path.read_text())
        already_done = {
            (v["claim_id_1"], v["claim_id_2"]) for v in verified
        }
        print(f"Resuming: {len(verified):,} already processed", flush=True)

    remaining = [
        p for p in pairs
        if (p["claim_id_1"], p["claim_id_2"]) not in already_done
    ]
    print(f"Remaining: {len(remaining):,} pairs to verify", flush=True)

    if not remaining:
        print("Nothing to do!")
        return

    # Create batches
    batches = []
    for i in range(0, len(remaining), args.batch_size):
        batches.append(remaining[i : i + args.batch_size])
    print(f"Batches: {len(batches):,} (size {args.batch_size}, {args.workers} workers)",
          flush=True)

    t0 = time.time()
    confirmed_count = 0
    error_count = 0
    batches_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for idx, batch in enumerate(batches):
            fut = pool.submit(call_gemini, batch, api_key, idx)
            futures[fut] = idx

        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                result_pairs = fut.result()
                verified.extend(result_pairs)
                for p in result_pairs:
                    if p.get("gemini_verdict") == "confirmed":
                        confirmed_count += 1
                    if p.get("gemini_verdict") == "error":
                        error_count += 1
            except Exception as e:
                print(f"  Batch {batch_idx} exception: {e}", flush=True)
                error_count += len(batches[batch_idx])

            batches_done += 1
            if batches_done % 10 == 0:
                elapsed = time.time() - t0
                rate = batches_done / elapsed * args.batch_size
                remaining_pairs = (len(batches) - batches_done) * args.batch_size
                eta_min = remaining_pairs / max(rate, 0.1) / 60
                print(
                    f"  Progress: {batches_done}/{len(batches)} batches, "
                    f"{confirmed_count} confirmed, {error_count} errors, "
                    f"{rate:.1f} pairs/s, ETA {eta_min:.0f}min",
                    flush=True,
                )

            if batches_done % CHECKPOINT_EVERY == 0:
                checkpoint_path.write_text(json.dumps(verified))

    # Save final results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verified, indent=2))
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - t0
    confirmed = [v for v in verified if v.get("gemini_verdict") == "confirmed"]
    errors = [v for v in verified if v.get("gemini_verdict") == "error"]

    print(f"\n{'='*60}", flush=True)
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Total verified: {len(verified):,}", flush=True)
    print(f"Confirmed contradictions: {len(confirmed):,}", flush=True)
    print(f"Rejected: {len(verified) - len(confirmed) - len(errors):,}", flush=True)
    print(f"Errors: {len(errors):,}", flush=True)

    if confirmed:
        print(f"\nTop confirmed contradictions (by confidence):", flush=True)
        for c in sorted(confirmed, key=lambda x: -x.get("confidence", 0))[:20]:
            print(
                f"  [{c.get('confidence', 0):.2f}] {c.get('subject', '')[:50]} "
                f"[{c.get('claim_type', '')}]:",
                flush=True,
            )
            print(f"    A ({c['doi_1'][:30]}): {c['quote_1'][:100]}...", flush=True)
            print(f"    B ({c['doi_2'][:30]}): {c['quote_2'][:100]}...", flush=True)
            print(f"    → {c.get('gemini_explanation', '')}", flush=True)


if __name__ == "__main__":
    main()
