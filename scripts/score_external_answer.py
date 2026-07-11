#!/usr/bin/env python3
"""Score an externally-produced answer against the AskChem-Bench metrics.

Useful for one-off comparisons against systems that don't yet expose an
API (e.g. NotebookLM, a custom prompt, a paper draft) so we can drop
their answer into the same scoring as our four bench modes.

Usage::

    PORTKEY_API_KEY=... python scripts/score_external_answer.py \\
        --qid ca02 \\
        --system notebooklm \\
        --answer-file data/eval/notebooklm_answers.md \\
        --label "NotebookLM"

The answer file can take one of two shapes:

- Consolidated append log (preferred): a single markdown file containing
  ``## qid: caXX — pasted YYYY-MM-DD`` headings followed by a
  ``**Answer:**`` marker. The script auto-detects this format and
  extracts only the section matching ``--qid``. This is what
  ``data/eval/notebooklm_answers.md`` uses.
- Single-answer file: any markdown file; an optional front matter
  block separated from the body by a ``\\n---\\n`` line is stripped.

In both cases the script extracts every DOI mention from the resulting
body and treats the union as "this system's cited DOIs".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_askchem import (  # noqa: E402
    extract_dois,
    verify_dois_in_text,
    score_paper_relevance,
    compute_metrics,
    _save_paper_relevance_cache,
)


BENCH_PATH = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json"


# Heading that delimits a single qid's entry in the consolidated NotebookLM log.
# Example line: "## qid: ca02 — pasted 2026-05-20"
_LOG_HEADING_RE = re.compile(r"^## qid:\s*(?P<qid>[A-Za-z0-9_-]+)\b", re.MULTILINE)
_LOG_ANSWER_MARKER_RE = re.compile(r"^\*\*Answer:\*\*\s*\n", re.MULTILINE)


def _extract_log_section(raw: str, qid: str) -> str:
    """Return just the `**Answer:**` body for `qid` from a multi-answer log."""
    needle = re.compile(
        rf"^## qid:\s*{re.escape(qid)}\b", re.IGNORECASE | re.MULTILINE
    )
    match = needle.search(raw)
    if not match:
        raise SystemExit(
            f"qid {qid!r} not found in consolidated log — expected a "
            f"'## qid: {qid}' heading"
        )
    next_match = _LOG_HEADING_RE.search(raw, pos=match.end())
    end = next_match.start() if next_match else len(raw)
    section = raw[match.end():end]
    answer_match = _LOG_ANSWER_MARKER_RE.search(section)
    if not answer_match:
        raise SystemExit(
            f"section for qid {qid!r} has no '**Answer:**' marker"
        )
    return section[answer_match.end():].strip()


def _read_answer(path: Path, qid: str | None = None) -> str:
    raw = path.read_text()
    # Consolidated NotebookLM log: when the file contains `## qid: caXX`
    # headings, treat each section as an independent answer and extract
    # only the one matching `qid`. This lets a single append-only file
    # serve as the canonical paste target for every benchmark question
    # without smearing DOIs from other qids into the scorer.
    if qid and _LOG_HEADING_RE.search(raw):
        return _extract_log_section(raw, qid)
    # Legacy single-answer file: strip an optional front-matter block
    # (everything up to the first '\n---\n' separator) so provenance
    # metadata doesn't pollute the scorer.
    parts = raw.split("\n---\n", 1)
    if len(parts) == 2 and len(parts[0]) < 3000:
        return parts[1].strip()
    return raw.strip()


def _fmt(v, pct: bool = False, dp: int = 2) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return f"{v:.{dp}f}"
    return str(v)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", required=True, help="Question id (e.g. ca01)")
    ap.add_argument("--system", required=True, help="Short system tag (e.g. notebooklm)")
    ap.add_argument("--answer-file", required=True)
    ap.add_argument("--label", default=None, help="Display label (defaults to --system)")
    ap.add_argument(
        "--judge-mode",
        choices=("paper", "claim"),
        default="paper",
        help=(
            "How to score relevance. 'paper' (default) uses paper "
            "title+abstract; 'claim' would require a claim_text field "
            "per DOI and is not supported for external systems yet."
        ),
    )
    args = ap.parse_args(argv)

    if args.judge_mode == "claim":
        raise SystemExit("--judge-mode=claim is not supported for external answers")

    label = args.label or args.system

    bench = json.loads(BENCH_PATH.read_text())
    try:
        q = next(x for x in bench["questions"] if x["id"] == args.qid)
    except StopIteration:
        raise SystemExit(f"qid {args.qid!r} not found in {BENCH_PATH}")

    question = q["question"]
    task = q["task"]

    answer = _read_answer(Path(args.answer_file), qid=args.qid)
    print(f"== {label} on {args.qid} ({task}) ==")
    print(f"  answer length: {len(answer):,} chars")

    dois = sorted({d.lower() for d in extract_dois(answer)})
    print(f"  unique DOIs cited: {len(dois)}")
    for d in dois:
        print(f"    - {d}")

    print("\n-- CrossRef verification --")
    dois_verified = verify_dois_in_text(answer, question, sleep_between=0.2)
    n_exists = sum(1 for v in dois_verified.values() if v.get("exists"))
    print(f"  {n_exists}/{len(dois_verified)} DOIs resolved against CrossRef")

    print("\n-- Gemini relevance judge (paper-grounded) --")
    judged = 0
    for doi, info in dois_verified.items():
        if not info.get("exists"):
            continue
        if not (info.get("crossref_title") or info.get("title")) and not info.get("abstract"):
            # doi.org-only verification path (arXiv DataCite, mEDRA,
            # legacy SICI) where CrossRef has no metadata. The DOI is
            # real but we have nothing for the judge to score against,
            # so we leave llm_relevance unset. The downstream metric
            # filters by isinstance(...), so this DOI contributes to
            # dois_exist but not to paper_relevance_mean.
            print(f"    {doi}: skip judge (verified via doi.org, no CrossRef metadata)")
            continue
        # Use a system-tagged qid so cached external runs don't collide
        # with the bench's own cache keys for the same (qid, doi).
        cache_qid = f"{args.qid}|ext:{args.system}"
        rec = score_paper_relevance(
            cache_qid, doi, question,
            info.get("crossref_title") or "",
            info.get("abstract") or "",
            use_cache=True,
        )
        if rec.get("score") is not None:
            info["llm_relevance"] = int(rec["score"])
            info["llm_relevance_rationale"] = rec.get("rationale", "")
            info["llm_relevance_judged_with_claim"] = False
            judged += 1
            print(f"    {doi}: score={rec['score']}  ({(rec.get('rationale') or '')[:90]})")
        else:
            print(f"    {doi}: SKIP (judge error: {rec.get('error')!r})")
    _save_paper_relevance_cache()
    print(f"  {judged} DOIs scored")

    edison_dois = {
        d.lower() for d, v in (q.get("edison_scientific") or {}).get("dois_verified", {}).items()
        if v.get("exists")
    }
    metrics = compute_metrics(answer, dois_verified, task, edison_dois=edison_dois)

    keys = [
        ("dois_cited", "DOIs cited", False, 0),
        ("dois_exist", "DOIs verified", False, 0),
        ("doi_existence_rate", "DOI %", True, 1),
        ("citation_density", "Cites/answer", False, 1),
        ("grounded_specificity", "Grounded spec.", False, 1),
        ("citation_count_mean", "Avg cites/paper", False, 1),
        ("recent_high_impact_rate", "Recent impact", True, 1),
        ("paper_relevance_mean", "Relevance (0-3)", False, 2),
        ("paper_relevance_high_rate", "On-topic (>=2)", True, 1),
        ("edison_overlap_rate", "Edison overlap", True, 1),
    ]

    bench_modes = [
        ("alone", "LLM only", q.get("llm_alone")),
        ("unified", "+ AskChem", q.get("unified")),
        ("paperclip_unified", "+ Paperclip", q.get("paperclip_unified")),
        ("edison_scientific", "Edison Sci.", q.get("edison_scientific")),
        ("external", label, {"metrics": metrics, "dois_verified": dois_verified, "answer": answer}),
    ]

    print(f"\n{'metric':24s}" + "".join(f"{name:>13s}" for _, name, _ in bench_modes))
    print("-" * (24 + 13 * len(bench_modes)))
    for key, name, pct, dp in keys:
        cells: list[str] = []
        for mode_key, _disp, cell in bench_modes:
            if not isinstance(cell, dict):
                cells.append("—")
                continue
            cell_metrics = cell.get("metrics") or {}
            v = cell_metrics.get(key)
            # Edison overlap is None for the Edison row itself.
            if v is None and key == "edison_overlap_rate" and mode_key == "edison_scientific":
                cells.append("—")
            else:
                cells.append(_fmt(v, pct=pct, dp=dp))
        row_label = f"{name}:"
        print(f"{row_label:24s}" + "".join(f"{c:>13s}" for c in cells))

    out_path = REPO_ROOT / "data" / "eval" / f"{args.system}_{args.qid}_scored.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "qid": args.qid,
        "system": args.system,
        "label": label,
        "task": task,
        "question": question,
        "answer": answer,  # full body so build_carousel_data.py can render it
        "answer_chars": len(answer),
        "dois_cited": dois,
        "dois_verified": dois_verified,
        "metrics": metrics,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
