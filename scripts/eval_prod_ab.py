"""Lightweight retrieval-quality comparison between two AskChem endpoints.

This script compares a public or remote endpoint with an optional local
endpoint using a curated 10-probe set
spanning the 5 bake-off families (reaction, technique, substance,
property, paper) and records:

* per-probe top-5 ``claim_contextualized`` / ``verbatim_quote``
* relevance counts against a hand-coded regex per probe
* wall latency

Output goes to ``data/eval/final_v2_ab.json`` (gitignored sibling
of the benchmark JSON family).

Usage::

    python3 scripts/eval_prod_ab.py --prod https://askchem.org/api \\
        --label v2-prod-may11
    python3 scripts/eval_prod_ab.py --prod http://127.0.0.1:8420/api \\
        --label v2-local-may11
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise

PROBES: list[dict] = [
    {
        "id": "reac-suzuki",
        "family": "reaction",
        "q": "suzuki coupling",
        "rel_regex": r"suzuki[\s_-]*miyaura|suzuki[- ]coupling|miyaura",
    },
    {
        "id": "reac-heck",
        "family": "reaction",
        "q": "heck reaction palladium",
        "rel_regex": r"heck\b.*pd|palladium.*heck|heck.*palladium|heck.*reaction",
    },
    {
        "id": "tech-xrd",
        "family": "technique",
        "q": "powder X-ray diffraction characterization",
        "rel_regex": r"x[\s-]*ray|XRD|powder diffraction|diffraction",
    },
    {
        "id": "tech-dft",
        "family": "technique",
        "q": "DFT density functional theory calculation",
        "rel_regex": r"dft\b|density functional|b3lyp|m06|pbe|hybrid functional",
    },
    {
        "id": "subs-mof",
        "family": "substance",
        "q": "metal organic framework MOF",
        "rel_regex": r"\bmof\b|metal[\s-]organic framework|coordination polymer",
    },
    {
        "id": "subs-perovskite",
        "family": "substance",
        "q": "perovskite solar cell",
        "rel_regex": r"perovskite|solar cell|photovoltaic",
    },
    {
        "id": "prop-bandgap",
        "family": "property",
        "q": "band gap semiconductor",
        "rel_regex": r"band[\s-]*gap|bandgap|valence band|conduction band",
    },
    {
        "id": "prop-yield",
        "family": "property",
        "q": "catalytic yield turnover frequency",
        "rel_regex": r"yield|turnover|tof\b|tos\b|conversion",
    },
    {
        "id": "ctx-battery",
        "family": "property",
        "q": "lithium battery cathode capacity",
        "rel_regex": r"lithium|li[\s-]*ion|cathode|capacity|mah/g|battery",
    },
    {
        "id": "ctx-nmr",
        "family": "technique",
        "q": "NMR spectroscopy chemical shift",
        "rel_regex": r"nmr|chemical shift|ppm|1h|13c|2d cosy",
    },
]


def claim_text(claim: dict) -> str:
    return " ".join([
        claim.get("claim_contextualized") or "",
        claim.get("verbatim_quote") or "",
        claim.get("claim") or "",
        claim.get("source_paper_title") or "",
    ])


def run_probes(api_base: str, probes: Iterable[dict], *,
               limit: int = 5, top_k_eval: int = 5,
               warmup: int = 2,
               timeout: float = 60.0) -> dict:
    print(f"probing {api_base} ...", flush=True)
    # Warm up the model + faiss + page cache so the timings reflect
    # the steady-state path the next user hits.
    for i, p in enumerate(probes):
        if i >= warmup:
            break
        requests.get(f"{api_base}/search", params={"q": p["q"], "limit": limit},
                     timeout=timeout)
    results = []
    latencies = []
    for p in probes:
        t0 = time.monotonic()
        r = requests.get(f"{api_base}/search",
                         params={"q": p["q"], "limit": limit},
                         timeout=timeout)
        ms = (time.monotonic() - t0) * 1000
        latencies.append(ms)
        hits = r.json().get("results", []) if r.status_code == 200 else []
        rel = re.compile(p["rel_regex"], re.I)
        per_hit = []
        n_rel = 0
        for j, h in enumerate(hits[:top_k_eval]):
            txt = claim_text(h)
            is_rel = bool(rel.search(txt))
            n_rel += int(is_rel)
            per_hit.append({
                "rank": j + 1,
                "is_relevant": is_rel,
                "claim_id": h.get("claim_id"),
                "text": txt.strip()[:200],
            })
        results.append({
            "probe_id": p["id"],
            "family": p["family"],
            "q": p["q"],
            "latency_ms": round(ms, 1),
            "n_hits": len(hits),
            "n_relevant_top5": n_rel,
            "precision_at_5": round(n_rel / max(top_k_eval, 1), 3),
            "hits": per_hit,
        })
        print(f"  [{p['id']:<14}] {p['q']:<40} p@5={n_rel}/{top_k_eval} "
              f"latency={ms:>6.0f} ms",
              flush=True)
    families: dict[str, list[float]] = {}
    for row in results:
        families.setdefault(row["family"], []).append(row["precision_at_5"])
    return {
        "api_base": api_base,
        "n_probes": len(results),
        "macro_precision_at_5": round(
            statistics.mean(row["precision_at_5"] for row in results), 3),
        "family_precision_at_5": {
            fam: round(statistics.mean(scores), 3)
            for fam, scores in sorted(families.items())
        },
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 1),
        "latency_max_ms": round(max(latencies), 1),
        "probes": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prod", required=True,
                    help="API base (e.g. https://askchem.org/api or "
                         "http://127.0.0.1:8420/api).")
    ap.add_argument("--label", required=True,
                    help="Run label for the output JSON.")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--top-k-eval", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "data" / "eval" / "final_v2_ab.json")
    args = ap.parse_args()

    summary = run_probes(args.prod, PROBES,
                         limit=args.limit,
                         top_k_eval=args.top_k_eval)
    summary["label"] = args.label
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with out_path.open("r") as fh:
            data = json.load(fh)
    else:
        data = {"runs": {}}
    data["runs"][args.label] = summary
    with out_path.open("w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nmacro p@5 = {summary['macro_precision_at_5']:.3f}  "
          f"latency p50={summary['latency_p50_ms']:.0f} ms  "
          f"p95={summary['latency_p95_ms']:.0f} ms")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
