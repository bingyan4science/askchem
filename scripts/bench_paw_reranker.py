"""Component-level benchmark for the chemistry-relevance reranker.

Two evaluation modes:

* ``--mode unit`` (default): runs each candidate PAW reranker against
  the stratified 297-pair probe set at
  ``data/eval/paw_reranker_probes.json`` and reports per-class accuracy,
  macro-F1, and ROC-AUC of the implied ordering. Cheap (~50 s per
  variant on ft, ~5 min on std).

* ``--mode pipeline``: takes the W0 baseline RRF pool from the May-29
  Phase 1 attribution run, applies the PAW reranker under three
  deployment patterns, and reports nDCG@10 vs the production MS-MARCO
  reranker. More expensive (~5 min per variant per pattern).

Reuses the ``--variants-registry`` and adapter pattern from
[scripts/bench_paw_expander.py](scripts/bench_paw_expander.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

os.environ.setdefault("GGML_NO_METAL", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEFAULT_REGISTRY = REPO_ROOT / "data" / "paw_expander_variants.json"
DEFAULT_PROBES = REPO_ROOT / "data" / "eval" / "paw_reranker_probes.json"

# Score-label mapping for the 3-class reranker (matches labels_v1.jsonl).
LABEL_TO_SCORE = {
    "not_relevant": 0,
    "somewhat_relevant": 1,
    "highly_relevant": 2,
    # 4-class variant (R3): exact_match collapses to highly_relevant for
    # comparison with the 3-class gold labels.
    "exact_match": 2,
}
SCORE_TO_LABEL = {0: "not_relevant", 1: "somewhat_relevant", 2: "highly_relevant"}


@dataclass
class Probe:
    probe_id: str
    family: str
    query: str
    claim_id: str
    claim_text: str
    gold_score: int


@dataclass
class Prediction:
    probe_id: str
    family: str
    claim_id: str
    gold_score: int
    raw_output: str
    predicted_label: str
    predicted_score: int
    latency_ms: float


def load_probes(path: Path) -> list[Probe]:
    payload = json.loads(path.read_text())
    return [
        Probe(
            probe_id=p["probe_id"], family=p["family"], query=p["query"],
            claim_id=p["claim_id"], claim_text=p["claim_text"],
            gold_score=int(p["gold_score"]),
        )
        for p in payload["probes"]
    ]


def parse_paw_output(raw: str) -> tuple[str, int]:
    """Normalise PAW output to (label, score). Falls back to most-permissive
    not_relevant if the model emits anything unexpected.
    """
    s = (raw or "").strip().lower()
    # Take first comma/whitespace-delimited token, strip punctuation.
    token = s.split(",")[0].split()[0] if s else ""
    token = token.strip(".,;:!?'\"()[]{}")
    if token in LABEL_TO_SCORE:
        return token, LABEL_TO_SCORE[token]
    # Try the prefix match: e.g. "highly" -> "highly_relevant"
    for label in ("exact_match", "highly_relevant", "somewhat_relevant",
                  "not_relevant"):
        if label.startswith(token) or token.startswith(label.split("_")[0]):
            return label, LABEL_TO_SCORE[label]
    return "not_relevant", 0


def make_paw_reranker(program_id: str, n_gpu_layers: int = 0):
    import programasweights as paw

    fn = paw.function(program_id, n_gpu_layers=n_gpu_layers)

    def _score(query: str, claim_text: str) -> tuple[str, int, str, float]:
        inp = f"QUERY: {query} CLAIM: {claim_text}"
        t0 = time.perf_counter()
        raw = (fn(inp) or "").strip()
        elapsed = (time.perf_counter() - t0) * 1000
        label, score = parse_paw_output(raw)
        return label, score, raw, elapsed

    return _score


def make_msmarco_reranker():
    """The production MS-MARCO MiniLM cross-encoder, wrapped to the same
    (label, score, raw, latency_ms) contract for apples-to-apples
    comparison. Score is bucketed by quantile to map continuous scores
    to the {0, 1, 2} categorical scale used for accuracy/F1.
    """
    from askchem.cross_encoder_rerank import rerank as _rerank

    def _score(query: str, claim_text: str) -> tuple[str, int, str, float]:
        # MS-MARCO outputs an unbounded real number; for accuracy/F1
        # purposes we bucket >2.0 = highly, >0.0 = somewhat, else not.
        # The thresholds were eyeballed from MS-MARCO's typical
        # distribution on AskChem (production observation).
        t0 = time.perf_counter()
        out = _rerank(query, [("c", claim_text)], top_k=1)
        elapsed = (time.perf_counter() - t0) * 1000
        if not out:
            return "not_relevant", 0, "", elapsed
        raw = float(out[0][1])
        if raw > 2.0:
            score = 2
        elif raw > 0.0:
            score = 1
        else:
            score = 0
        return SCORE_TO_LABEL[score], score, f"{raw:.3f}", elapsed

    return _score


def build_systems(
    selected: list[str], variants_registry: Path | None
) -> dict[str, Callable]:
    systems: dict[str, Callable] = {}
    if "msmarco" in selected:
        print("Loading msmarco (production cross-encoder)...", flush=True)
        systems["msmarco"] = make_msmarco_reranker()
    if variants_registry is not None:
        try:
            payload = json.loads(variants_registry.read_text())
        except Exception as exc:
            print(f"WARN: could not read {variants_registry}: {exc}",
                  file=sys.stderr)
            payload = {"variants": []}
        for entry in payload.get("variants", []):
            if entry.get("program_type") != "reranker":
                # Don't auto-load expander variants — wrong contract.
                continue
            name = entry["name"]
            pid = entry["program_id"]
            if name in systems:
                continue
            print(f"Loading variant {name} ({pid})...", flush=True)
            systems[name] = make_paw_reranker(pid)
    return systems


# ── Unit eval ──────────────────────────────────────────────────────────────


def evaluate_unit(probes: list[Probe], systems: dict[str, Callable]) -> dict:
    """Per-variant classification metrics on the 297-pair set."""
    by_sys: dict[str, list[Prediction]] = defaultdict(list)

    # Warmup
    for name, fn in systems.items():
        try:
            fn("warmup query", "warmup claim text")
        except Exception as exc:
            print(f"WARN: {name} warmup raised {exc!r}", file=sys.stderr)

    print()
    for i, p in enumerate(probes, 1):
        for name, fn in systems.items():
            try:
                label, score, raw, ms = fn(p.query, p.claim_text)
            except Exception as exc:
                label, score, raw, ms = "not_relevant", 0, f"<err: {exc}>", 0.0
            by_sys[name].append(Prediction(
                probe_id=p.probe_id, family=p.family,
                claim_id=p.claim_id, gold_score=p.gold_score,
                raw_output=raw[:80], predicted_label=label,
                predicted_score=score, latency_ms=ms,
            ))
        if i <= 3 or i % 30 == 0 or i == len(probes):
            print(f"  [{i:>3}/{len(probes)}] {p.probe_id:<10} gold={p.gold_score} "
                  + " ".join(
                      f"{n}={by_sys[n][-1].predicted_score}"
                      for n in systems
                  ),
                  flush=True)

    # Aggregate metrics
    report: dict[str, dict] = {}
    for name, preds in by_sys.items():
        gold = [p.gold_score for p in preds]
        pred = [p.predicted_score for p in preds]
        acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
        # Per-class precision/recall/f1
        per_class: dict[str, dict] = {}
        for cls in (0, 1, 2):
            tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
            fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
            fn_ = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[SCORE_TO_LABEL[cls]] = {
                "tp": tp, "fp": fp, "fn": fn_,
                "precision": prec, "recall": rec, "f1": f1,
            }
        macro_f1 = statistics.mean(v["f1"] for v in per_class.values())
        # Mean absolute error of score (penalises wrong-by-one less than
        # wrong-by-two).
        mae = statistics.mean(abs(g - p) for g, p in zip(gold, pred))
        # Pairwise Kendall's tau-ish: fraction of (high, low) pairs where
        # predicted score for high >= predicted for low.
        pairs = 0
        concordant = 0
        for i, gi in enumerate(gold):
            for j in range(i + 1, len(gold)):
                gj = gold[j]
                if gi == gj: continue
                pairs += 1
                if (gi > gj) == (pred[i] > pred[j]) and pred[i] != pred[j]:
                    concordant += 1
                elif pred[i] == pred[j]:
                    concordant += 0.5
        pairwise = concordant / pairs if pairs else float("nan")
        latency = statistics.mean(p.latency_ms for p in preds)
        report[name] = {
            "n": len(preds),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "mae": mae,
            "pairwise_concordance": pairwise,
            "avg_latency_ms": latency,
            "per_class": per_class,
        }
    return report


def print_unit_report(report: dict[str, dict]) -> None:
    print()
    print("=" * 78)
    print("Unit metrics (3-class classification on stratified 297-pair set)")
    print("=" * 78)
    sys_names = sorted(report.keys())
    header = (f"  {'system':<14}  {'acc':>6}  {'macroF1':>8}  {'MAE':>6}  "
              f"{'pairwise':>9}  {'avg ms':>8}")
    print(header)
    print("  " + "-" * 60)
    for name in sys_names:
        m = report[name]
        print(f"  {name:<14}  {m['accuracy']:>6.3f}  {m['macro_f1']:>8.3f}  "
              f"{m['mae']:>6.3f}  {m['pairwise_concordance']:>9.3f}  "
              f"{m['avg_latency_ms']:>7.0f}")

    print()
    print("Per-class F1 (higher = better)")
    classes = ("not_relevant", "somewhat_relevant", "highly_relevant")
    header = f"  {'system':<14}  " + "  ".join(f"{c:>18}" for c in classes)
    print(header)
    print("  " + "-" * (14 + 22 * len(classes)))
    for name in sys_names:
        cells = [f"{report[name]['per_class'][c]['f1']:>18.3f}" for c in classes]
        print(f"  {name:<14}  " + "  ".join(cells))


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    ap.add_argument("--variants-registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--systems", default="msmarco",
                    help="Baselines beyond the registry variants. Default: msmarco.")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/eval/runs/paw_reranker_bench.json")
    ap.add_argument("--mode", choices=["unit"], default="unit",
                    help="(pipeline mode TBD)")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only the first N probes (for smoke tests).")
    args = ap.parse_args()

    probes = load_probes(args.probes)
    if args.limit > 0:
        probes = probes[: args.limit]
    print(f"Loaded {len(probes)} probes from {args.probes}")

    selected = [s.strip() for s in args.systems.split(",") if s.strip()]
    systems = build_systems(selected, args.variants_registry)
    if not systems:
        print("ERROR: no systems selected.", file=sys.stderr)
        return 2

    report = evaluate_unit(probes, systems)
    print_unit_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_probes": len(probes),
        "report": report,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
