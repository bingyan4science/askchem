"""Gemini-2.5-flash relevance judge for the Phase 0 eval pool.

Reads ``data/eval/candidates_v1.jsonl`` (built by
``build_eval_candidates.py``), renders each (query, claim) into the
same plaintext the spot-checker sees, and asks Gemini for a 0/1/2
relevance score with a one-sentence rationale.

Output: ``data/eval/labels_v1.jsonl`` — one JSON line per judgment.
The script is **idempotent**: rerunning it skips (probe, claim_id)
pairs already present in the labels file. So you can ctrl-C and
restart, or top up after extending the candidate pool.

Usage::

    export PORTKEY_API_KEY=...
    python scripts/llm_judge_eval.py
    python scripts/llm_judge_eval.py --workers 16 --model gemini-2.5-flash

Cost: ~$5–15 for the full 80-probe pool with gemini-2.5-flash.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_common import (  # noqa: E402
    CANDIDATES_PATH, LABELS_PATH, iter_jsonl, append_jsonl,
    open_claims_db, load_claims, render_claim_for_judge,
)


GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions"
PROVIDER = "@vertexai-gemini-kc119-2"

# Default to Pro for the v1 calibration pass. Pro hits κ ≈ 0.75-0.85
# against human adjudicators on hard relevance / homonym judgments;
# Flash sits at κ ≈ 0.55-0.65 — borderline-too-noisy for picking
# encoders later. Switch back to Flash via --model for incremental
# re-judging once the rubric is calibrated and the spot-check passes.
DEFAULT_MODEL = "gemini-3.1-pro-preview"

# USD per 1M tokens (≤200K context, 2026-Q1, sync tier).
# Flash:  in $0.075   out $0.30
# Pro:    in $2.00    out $12.00
# We pull `usage` off every response and bill from there; the constants
# below are only used for the running ETA.
PRICE_IN_PER_M = 2.00
PRICE_OUT_PER_M = 12.00

DEFAULT_TIMEOUT_S = 60
DEFAULT_RETRIES = 4

PROMPT = """You are a chemistry-domain relevance judge for a search index over chemistry papers.

You are given a search QUERY and a candidate CLAIM extracted from one paper. Output a single JSON object on one line:

{{"score": 0|1|2, "rationale": "<one short sentence>"}}

Rubric:

  2 = HIGHLY RELEVANT. The claim directly answers the query.
      • Query "Suzuki coupling" + claim about Pd-catalysed coupling of an aryl boronic acid with an aryl halide → 2.
      • Query "MOF surface area BET" + claim "ZIF-8 BET surface area = 1630 m²/g" → 2.
      • Query "DFT reaction mechanism" + claim using DFT to compute a transition state energy along a reaction coordinate → 2.

  1 = RELEVANT. Same chemistry sub-area as the query, but not the specific thing asked.
      • Query "Suzuki coupling" + claim about Negishi or Heck cross-coupling → 1.
      • Query "MOF surface area" + claim about MOF pore volume or gas uptake (related but different property) → 1.
      • Query "MoS2 hydrogen evolution overpotential" + claim about MoS2 synthesis or its electronic structure → 1.

  0 = IRRELEVANT. Different field, homonym in another physical-science context, off-topic, or chemistry sub-area unrelated to the query.
      • Query "Suzuki coupling" + claim about exciton-coupling, spin-orbit coupling, or Josephson coupling in physics → 0.
      • Query "spin coupling NMR scalar J" + claim about Heisenberg spin Hamiltonian / magnetic ordering in solids → 0.
      • Query "MOF surface area" + claim about MOF photoluminescence quantum yield → 0.

Be strict. When unsure between 1 and 2, prefer 1. When unsure between 0 and 1, prefer 0. Output JSON ONLY — no surrounding prose, no markdown fences.

QUERY: {query}

CLAIM:
{claim_text}
"""


# ── Gemini call ─────────────────────────────────────────────────────────────


def _parse_json_lenient(text: str) -> dict | None:
    """Parse JSON from a model response that may have prose around it.

    Pro occasionally emits a brief reasoning sentence before the JSON
    object even with response_format=json_object. We try strict
    parsing first, then strip ```json fences, then extract the first
    balanced {...} block. Returns None if nothing parses.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip("`").strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


def call_judge(model: str, query: str, claim_text: str,
               max_tokens: int = 1024,
               retries: int = DEFAULT_RETRIES,
               timeout: int = DEFAULT_TIMEOUT_S) -> dict:
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            query=query, claim_text=claim_text,
        )}],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "x-portkey-api-key": api_key,
        "x-portkey-provider": PROVIDER,
        "Content-Type": "application/json",
    }

    sess = _session()
    last_err = ""
    for attempt in range(retries):
        try:
            r = sess.post(GATEWAY, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            last_err = f"network: {e}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code != 200:
            last_err = f"http {r.status_code}: {r.text[:160]}"
            time.sleep(min(2 ** attempt, 30))
            continue
        try:
            resp = r.json()
        except Exception as e:
            last_err = f"json: {e}: {r.text[:160]}"
            time.sleep(min(2 ** attempt, 30))
            continue
        choices = resp.get("choices") or []
        if not choices:
            last_err = f"no choices: {json.dumps(resp)[:160]}"
            time.sleep(min(2 ** attempt, 30))
            continue
        msg = choices[0].get("message", {})
        content = (msg.get("content") or "").strip()
        usage = resp.get("usage") or {}
        if not content:
            last_err = f"empty content; usage={usage}"
            time.sleep(min(2 ** attempt, 30))
            continue
        parsed = _parse_json_lenient(content)
        if parsed is None:
            last_err = f"content not JSON: {content[:160]}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if not isinstance(parsed, dict) or "score" not in parsed:
            last_err = f"bad shape: {parsed!r}"
            time.sleep(min(2 ** attempt, 30))
            continue
        try:
            score = int(parsed["score"])
        except (TypeError, ValueError):
            last_err = f"score not int: {parsed.get('score')!r}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if score not in (0, 1, 2):
            last_err = f"score out of range: {score}"
            time.sleep(min(2 ** attempt, 30))
            continue
        return {
            "score": score,
            "rationale": str(parsed.get("rationale", ""))[:240],
            "usage": usage,
        }
    raise RuntimeError(f"Gemini judge failed after {retries} retries: {last_err}")


# ── Driver ──────────────────────────────────────────────────────────────────


def _key(probe_id: str, claim_id: str) -> str:
    return f"{probe_id}::{claim_id}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", default=str(CANDIDATES_PATH))
    p.add_argument("--out", default=str(LABELS_PATH))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-per-probe", type=int, default=None,
                   help="(debug) cap candidates per probe before judging")
    p.add_argument("--limit", type=int, default=None,
                   help="(debug) judge at most N pairs total then stop")
    args = p.parse_args()

    cand_path = Path(args.candidates)
    out_path = Path(args.out)

    if not cand_path.exists():
        print(f"ERROR: {cand_path} not found. Run build_eval_candidates.py first.",
              file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("PORTKEY_API_KEY"):
        print("ERROR: PORTKEY_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    candidate_rows = list(iter_jsonl(cand_path))
    print(f"Loaded {len(candidate_rows)} probes from {cand_path}")

    done_keys: set[str] = set()
    for r in iter_jsonl(out_path):
        done_keys.add(_key(r["probe_id"], r["claim_id"]))
    if done_keys:
        print(f"  resuming: {len(done_keys)} judgments already in {out_path}")

    pending: list[tuple[str, str, str, str]] = []  # (probe_id, family, q, cid)
    needed_cids: set[str] = set()
    for cand in candidate_rows:
        cids = cand["candidate_ids"]
        if args.max_per_probe:
            cids = cids[:args.max_per_probe]
        for cid in cids:
            if _key(cand["id"], cid) in done_keys:
                continue
            pending.append((cand["id"], cand["family"], cand["q"], cid))
            needed_cids.add(cid)
            if args.limit is not None and len(pending) >= args.limit:
                break
        if args.limit is not None and len(pending) >= args.limit:
            break

    if not pending:
        print("Nothing to do. All pairs already judged.")
        return

    print(f"  judging {len(pending)} new (probe, claim) pairs "
          f"with {args.model} via {args.workers} workers")

    print(f"  loading {len(needed_cids)} claims from DB...")
    conn = open_claims_db()
    try:
        claim_map = load_claims(sorted(needed_cids), conn)
    finally:
        conn.close()
    print(f"  loaded {len(claim_map)} / {len(needed_cids)} claims "
          f"({len(needed_cids) - len(claim_map)} missing)")

    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    counters = {
        "ok": 0, "fail": 0, "missing_claim": 0,
        "in_tok": 0, "out_tok": 0,
    }
    t0 = time.monotonic()

    def _do(item: tuple[str, str, str, str]):
        probe_id, family, q, cid = item
        claim = claim_map.get(cid)
        if claim is None:
            with progress_lock:
                counters["missing_claim"] += 1
            return None
        rendered = render_claim_for_judge(claim)
        try:
            verdict = call_judge(args.model, q, rendered)
        except Exception as e:
            with progress_lock:
                counters["fail"] += 1
            return {"error": str(e), "probe_id": probe_id, "claim_id": cid}
        usage = verdict.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        row = {
            "probe_id": probe_id,
            "family": family,
            "q": q,
            "claim_id": cid,
            "score": verdict["score"],
            "rationale": verdict["rationale"],
            "judge": args.model,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
        }
        with write_lock:
            append_jsonl(out_path, row)
        with progress_lock:
            counters["ok"] += 1
            counters["in_tok"] += in_tok
            counters["out_tok"] += out_tok
            n = counters["ok"] + counters["fail"]
            if n % 50 == 0 or n == len(pending):
                rate = n / max(1e-3, time.monotonic() - t0)
                cost = (counters["in_tok"]  / 1e6 * PRICE_IN_PER_M
                      + counters["out_tok"] / 1e6 * PRICE_OUT_PER_M)
                print(
                    f"    {n:>5}/{len(pending)}  ok={counters['ok']:>5} "
                    f"fail={counters['fail']:>3} miss={counters['missing_claim']:>3}  "
                    f"in_tok={counters['in_tok']:>7,} out_tok={counters['out_tok']:>6,}  "
                    f"~${cost:.3f}  ({rate:.1f}/s)",
                    flush=True,
                )
        return row

    failed: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do, item) for item in pending]
        for fut in as_completed(futs):
            res = fut.result()
            if res and "error" in res:
                failed.append(res)

    print()
    print(f"Done in {int(time.monotonic() - t0)}s")
    print(f"  judged ok:        {counters['ok']}")
    print(f"  failed (network): {counters['fail']}")
    print(f"  missing claim:    {counters['missing_claim']}")
    cost = (counters["in_tok"]  / 1e6 * PRICE_IN_PER_M
          + counters["out_tok"] / 1e6 * PRICE_OUT_PER_M)
    print(f"  tokens (in/out):  {counters['in_tok']:,} / {counters['out_tok']:,}")
    print(f"  est. cost:        ~${cost:.3f}")
    if failed:
        fail_path = out_path.with_suffix(".failures.jsonl")
        with fail_path.open("w") as f:
            for r in failed:
                f.write(json.dumps(r) + "\n")
        print(f"  failures dumped:  {fail_path}")
        print("  (rerun the script to retry — it skips done pairs.)")


if __name__ == "__main__":
    main()
