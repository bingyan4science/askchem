"""Generate web/benchmark_answers.json (carousel data) from a bench output.

The carousel on the website reads a flat list of:
    {id, task, domain, question,
     alone:        {answer, doi_pct, citations},
     unified:      {answer, doi_pct, citations},
     paperclip?:   {answer, doi_pct, citations},
     edison?:      {answer, doi_pct, citations},
     notebooklm?:  {answer, doi_pct, citations}}

Usage:
    .venv-benchmark/bin/python scripts/build_carousel_data.py \\
        [--input scripts/benchmark_results_gpt-5.5.json] \\
        [--output web/benchmark_answers.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(entry: dict, mode_key: str) -> dict | None:
    cell = entry.get(mode_key)
    if not isinstance(cell, dict) or not cell.get("answer"):
        return None
    dois_verified = cell.get("dois_verified", {}) or {}
    n_total = len(dois_verified)
    n_exists = sum(1 for v in dois_verified.values() if v.get("exists"))
    doi_pct = round(100.0 * n_exists / n_total) if n_total else 0
    return {
        "answer": cell["answer"],
        "doi_pct": doi_pct,
        "citations": n_exists,
    }


def build(bench_path: Path, out_path: Path) -> None:
    data = json.loads(bench_path.read_text())
    out: list[dict] = []
    for q in data.get("questions", []):
        item = {
            "id": q["id"],
            "task": q.get("task", ""),
            "domain": q.get("domain", ""),
            "question": q.get("question", ""),
        }
        alone = _cell(q, "alone") or _cell(q, "llm_alone")
        if alone:
            item["alone"] = alone

        unified = _cell(q, "unified")
        if unified:
            item["unified"] = unified
        else:
            # Back-compat: surface the strict_grounded cell as `unified`
            # when this question has not been re-run on the new pipeline
            # yet, so the carousel pane is never empty.
            legacy = (
                _cell(q, "grounded")
                or _cell(q, "strict_grounded")
                or _cell(q, "retrieval_assisted")
                or _cell(q, "llm_plus_askchem")
            )
            if legacy:
                item["unified"] = legacy

        paperclip = _cell(q, "paperclip_unified")
        if paperclip:
            item["paperclip"] = paperclip

        edison = _cell(q, "edison_scientific") or _cell(q, "edison")
        if edison:
            item["edison"] = edison

        notebooklm = _cell(q, "notebooklm")
        if notebooklm:
            item["notebooklm"] = notebooklm

        # Only emit rows with at least an `alone` and one AskChem cell so
        # the carousel can render the comparison panes.
        if "alone" in item and "unified" in item:
            out.append(item)

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path} ({len(out)} questions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default="scripts/benchmark_results_gpt-5.5.json",
        help="Bench output JSON to read (default: %(default)s).",
    )
    ap.add_argument(
        "--output",
        default="web/benchmark_answers.json",
        help="Carousel JSON to write (default: %(default)s).",
    )
    args = ap.parse_args()
    build(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
