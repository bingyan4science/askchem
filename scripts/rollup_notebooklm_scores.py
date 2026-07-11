#!/usr/bin/env python3
"""Aggregate the per-qid notebooklm scoring JSONs into one comparison report.

Reads every `data/eval/notebooklm_{qid}_scored.json` and joins it with the
cached bench results in `scripts/benchmark_results_gpt-5.5.json` so each
question shows NotebookLM alongside LLM-only, +AskChem, +Paperclip, and
Edison columns. Emits both a console table and a markdown report at
`data/eval/notebooklm_ca_tc_rollup_{date}.md`.

Usage::

    python scripts/rollup_notebooklm_scores.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics as _stats
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data" / "eval"
BENCH_PATH = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json"

# Metric keys that the leaderboard reads off `aggregate.<task>.<mode>` and
# the per-question `<mode>` cells. Must be a superset of what
# `web/index.html` looks up so the NotebookLM row renders consistently
# with the four existing systems.
_BENCH_METRIC_KEYS = (
    "doi_existence_rate",
    "doi_relevance_rate",
    "citation_density",
    "specificity_score",
    "grounded_specificity",
    "citation_count_mean",
    "citation_count_median",
    "high_impact_rate",
    "recent_high_impact_rate",
    "paper_relevance_mean",
    "paper_relevance_high_rate",
    "edison_overlap_rate",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return {'mean', 'std'} in the same shape the bench JSON uses."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = _stats.fmean(values)
    s = _stats.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": round(m, 3), "std": round(s, 3)}

# Order of qids to include; the script silently skips ones whose scored
# file doesn't exist yet, so it works mid-batch too.
QIDS = (
    [f"ca{i:02d}" for i in range(1, 11)]
    + [f"tc{i:02d}" for i in range(1, 11)]
    + [f"cs{i:02d}" for i in range(1, 11)]
)

# Metric keys that all five columns expose via `metrics`.
METRIC_KEYS = [
    ("dois_cited", "DOIs cited", False, 0),
    ("dois_exist", "DOIs verified", False, 0),
    ("doi_existence_rate", "DOI exist %", True, 1),
    ("citation_density", "Cites/answer", False, 1),
    ("grounded_specificity", "Grounded spec.", False, 1),
    ("citation_count_mean", "Avg cites/paper", False, 1),
    ("recent_high_impact_rate", "Recent impact %", True, 1),
    ("paper_relevance_mean", "Relevance (0-3)", False, 2),
    ("paper_relevance_high_rate", "On-topic ≥2 %", True, 1),
    ("edison_overlap_rate", "Edison overlap %", True, 1),
]

COLUMNS = [
    ("llm_alone", "LLM only"),
    ("unified", "+AskChem"),
    ("paperclip_unified", "+Paperclip"),
    ("edison_scientific", "Edison"),
    ("notebooklm", "NotebookLM"),
]


def _fmt(v, pct: bool, dp: int) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{dp}f}%"
    if isinstance(v, float):
        return f"{v:.{dp}f}"
    return str(v)


def _load_bench() -> dict[str, dict]:
    raw = json.loads(BENCH_PATH.read_text())
    out: dict[str, dict] = {}
    for q in raw.get("questions", []):
        out[q["id"]] = q
    return out


def _load_notebooklm(qid: str) -> dict | None:
    p = EVAL_DIR / f"notebooklm_{qid}_scored.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _row(qid: str, bench: dict, nlm: dict) -> dict[str, dict | None]:
    cells: dict[str, dict | None] = {}
    cells["llm_alone"] = bench.get("llm_alone")
    cells["unified"] = bench.get("unified")
    cells["paperclip_unified"] = bench.get("paperclip_unified")
    cells["edison_scientific"] = bench.get("edison_scientific")
    cells["notebooklm"] = {"metrics": nlm["metrics"]}
    return cells


def _cell_metric(cell: dict | None, key: str) -> float | None:
    if not isinstance(cell, dict):
        return None
    m = cell.get("metrics") or {}
    return m.get(key)


def _write_into_bench(rows: list[tuple[str, str, dict, dict]]) -> Path:
    """Merge per-task notebooklm cells back into benchmark_results_gpt-5.5.json.

    Adds:
        - `aggregate.<task>.notebooklm` block with `n_questions` plus a
          `{mean, std}` pair for every metric in `_BENCH_METRIC_KEYS`.
        - `questions[i].notebooklm.metrics` per scored question, so the
          per-question UI cells line up with the existing four columns.

    A `.bak.<timestamp>` snapshot is taken before the in-place rewrite so
    the previous JSON is recoverable without git.
    """
    bench_doc = json.loads(BENCH_PATH.read_text())
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = BENCH_PATH.with_suffix(f".pre_notebooklm_{ts}.bak")
    bak.write_text(json.dumps(bench_doc, indent=2, ensure_ascii=False))

    # Group rows by task.
    rows_by_task: dict[str, list[dict]] = {}
    for qid, _q, _cells, nlm in rows:
        task = nlm.get("task") or qid[:2].upper()
        rows_by_task.setdefault(task, []).append(nlm)

    bench_doc.setdefault("aggregate", {})
    for task, task_rows in rows_by_task.items():
        agg_block: dict[str, object] = {"n_questions": len(task_rows)}
        for key in _BENCH_METRIC_KEYS:
            vals = [r["metrics"].get(key) for r in task_rows]
            vals = [v for v in vals if isinstance(v, (int, float))]
            agg_block[key] = _mean_std(vals)
        bench_doc["aggregate"].setdefault(task, {})["notebooklm"] = agg_block

    # Per-question cell so the carousel and per-row APIs can find it.
    # Include `answer` so build_carousel_data.py can render the
    # NotebookLM pane in the comparison carousel alongside LLM,
    # +AskChem, +Paperclip, Edison.
    qmap = {q["id"]: q for q in bench_doc.get("questions", [])}
    for qid, _q, _cells, nlm in rows:
        q_entry = qmap.get(qid)
        if q_entry is None:
            continue
        q_entry["notebooklm"] = {
            "answer": nlm.get("answer", ""),
            "metrics": nlm.get("metrics", {}),
            "dois_cited": nlm.get("dois_cited", []),
            "dois_verified": nlm.get("dois_verified", {}),
            "answer_chars": nlm.get("answer_chars"),
        }

    BENCH_PATH.write_text(json.dumps(bench_doc, indent=2, ensure_ascii=False))
    return bak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write-into-bench",
        action="store_true",
        help=(
            "After printing the rollup, merge aggregate.<task>.notebooklm "
            "and per-question notebooklm cells into "
            "scripts/benchmark_results_gpt-5.5.json (backed up first). "
            "This is what surfaces the NotebookLM row on /api/benchmark."
        ),
    )
    args = ap.parse_args()

    bench = _load_bench()

    rows: list[tuple[str, str, dict[str, dict | None], dict]] = []
    for qid in QIDS:
        nlm = _load_notebooklm(qid)
        if nlm is None:
            continue
        if qid not in bench:
            print(f"  skip {qid}: not in bench results")
            continue
        rows.append((qid, bench[qid]["question"], _row(qid, bench[qid], nlm), nlm))

    if not rows:
        print("No notebooklm_*_scored.json files found yet.")
        return 1

    # ── Per-question table (mean relevance + DOI existence as the headline cells) ─
    print(f"\nPer-question summary ({len(rows)} questions):")
    header = f"{'qid':<6} {'task':<5}  " + "  ".join(
        f"{disp:>11s}" for _, disp in COLUMNS
    ) + f"  {'metric':<14}"
    headline_metrics = [
        ("doi_existence_rate", "DOI exist %", True, 1),
        ("paper_relevance_mean", "Relevance", False, 2),
    ]
    for key, label, pct, dp in headline_metrics:
        print()
        print(f"-- {label} --")
        print(header)
        print("-" * len(header))
        for qid, _q, cells, _nlm in rows:
            task = qid[:2].upper()
            vals = [
                _fmt(_cell_metric(cells[k], key), pct=pct, dp=dp)
                for k, _ in COLUMNS
            ]
            print(f"{qid:<6} {task:<5}  " + "  ".join(f"{v:>11s}" for v in vals))

    # ── Aggregate (mean over the questions we have, per column) ─────────
    print("\n\nAggregate means across scored questions:")
    print(f"{'metric':<22s}  " + "  ".join(f"{disp:>12s}" for _, disp in COLUMNS))
    print("-" * (22 + (12 + 2) * len(COLUMNS)))
    agg: dict[tuple[str, str], list[float]] = {}
    for key, label, pct, dp in METRIC_KEYS:
        for col_key, _disp in COLUMNS:
            vals = [
                _cell_metric(cells[col_key], key)
                for _qid, _q, cells, _n in rows
            ]
            vals = [v for v in vals if v is not None]
            agg[(key, col_key)] = vals
            avg = mean(vals) if vals else None
        line_vals = []
        for col_key, _disp in COLUMNS:
            vals = agg[(key, col_key)]
            avg = mean(vals) if vals else None
            line_vals.append(_fmt(avg, pct=pct, dp=dp))
        print(f"{label:<22s}  " + "  ".join(f"{v:>12s}" for v in line_vals))

    # ── Markdown report ────────────────────────────────────────────────
    today = _dt.date.today().isoformat()
    md_path = EVAL_DIR / f"notebooklm_ca_tc_rollup_{today}.md"
    lines: list[str] = []
    lines.append(f"# NotebookLM vs AskChem-Bench (CA + TC + CS) ({today})")
    lines.append("")
    lines.append(
        f"NotebookLM answers were collected manually into "
        f"[data/eval/notebooklm_answers.md](notebooklm_answers.md) "
        f"and scored through "
        f"[scripts/score_external_answer.py](../../scripts/score_external_answer.py). "
        f"Other columns are cached bench results from "
        f"[scripts/benchmark_results_gpt-5.5.json](../../scripts/benchmark_results_gpt-5.5.json). "
        f"Paper-grounded judge (gemini-3.1-pro-preview) across all systems "
        f"except AskChem unified, which uses the claim-grounded judge."
    )
    lines.append("")
    lines.append(f"Scored {len(rows)} questions: "
                 + ", ".join(qid for qid, _q, _c, _n in rows))
    lines.append("")

    # Aggregate means table.
    lines.append("## Aggregate means")
    lines.append("")
    head = "| metric | " + " | ".join(disp for _, disp in COLUMNS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(COLUMNS)) + "|"
    lines.append(head)
    lines.append(sep)
    for key, label, pct, dp in METRIC_KEYS:
        cells = []
        for col_key, _disp in COLUMNS:
            vals = agg[(key, col_key)]
            avg = mean(vals) if vals else None
            cells.append(_fmt(avg, pct=pct, dp=dp))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Headline per-question tables (DOI existence and Relevance).
    for key, label, pct, dp in headline_metrics:
        lines.append(f"## Per-question {label}")
        lines.append("")
        lines.append("| qid | task | " + " | ".join(disp for _, disp in COLUMNS) + " |")
        lines.append("|---|---|" + "|".join(["---:"] * len(COLUMNS)) + "|")
        for qid, _q, cells, _n in rows:
            task = qid[:2].upper()
            vals = [
                _fmt(_cell_metric(cells[k], key), pct=pct, dp=dp)
                for k, _ in COLUMNS
            ]
            lines.append(f"| `{qid}` | {task} | " + " | ".join(vals) + " |")
        lines.append("")

    # NotebookLM-specific per-question detail.
    lines.append("## NotebookLM per-question detail")
    lines.append("")
    lines.append(
        "| qid | task | chars | DOIs cited | DOIs verified | mean relevance | "
        "headline failure mode |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for qid, _q, _cells, nlm in rows:
        task = qid[:2].upper()
        m = nlm["metrics"]
        chars = nlm.get("answer_chars", 0)
        notes: list[str] = []
        for doi, info in nlm.get("dois_verified", {}).items():
            if not info.get("exists"):
                notes.append(f"`{doi}` 404")
        if m.get("dois_cited", 0) == 0:
            notes.append("no DOIs cited")
        rel = m.get("paper_relevance_mean")
        rel_s = _fmt(rel, pct=False, dp=2)
        lines.append(
            f"| `{qid}` | {task} | {chars:,} | "
            f"{m.get('dois_cited', 0)} | {m.get('dois_exist', 0)} | "
            f"{rel_s} | "
            + (", ".join(notes) if notes else "—")
            + " |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\nwrote {md_path}")

    if args.write_into_bench:
        bak = _write_into_bench(rows)
        print(f"merged notebooklm aggregate into {BENCH_PATH.relative_to(REPO_ROOT)}")
        print(f"  (previous JSON saved to {bak.relative_to(REPO_ROOT)})")
        print(
            "  restart askchem.service to invalidate the 1h /api/benchmark cache."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
