"""Hand-grading tool for LLM-extracted claim edges.

Three subcommands let you ask one question: are these edges real?

    sample   Draw a stratified sample of 100 edges (80 intra + 20 cross) from
             the existing claim_edges table, with full self-contained context
             for each endpoint, and write to data/edge_grading_sample.jsonl.

    grade    Walk through the sample one edge at a time in the terminal.
             Resumable; each verdict is appended to the results file as soon
             as you press a key.

    report   Compute precision overall, per edge_type, per confidence, and
             per mode (intra vs cross) from the results file.

The grader writes nothing to the database — verdicts live in the JSONL only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "chemtree.db"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_SAMPLE_PATH = DATA_DIR / "edge_grading_sample.jsonl"
DEFAULT_RESULTS_PATH = DATA_DIR / "edge_grading_results.jsonl"

# Pools to draw from. The pilot extractor is intentionally excluded — it ran
# on abstract-only papers, so its edges are not representative of deep_v1 work.
INTRA_EXTRACTORS = (
    "intra_llm_gemini_preflight_v1",
    "intra_llm_gemini_v1",
)
CROSS_EXTRACTORS = (
    "cross_llm_gemini_preflight_v1",
)

# Stratification targets (per the approved plan).
INTRA_TARGETS = {
    # edge_type        (total, high, medium)
    "supports":         (14, 7, 7),
    "derives_from":     (14, 7, 7),
    "interprets":       (12, 6, 6),
    "sub_step_of":      (12, 6, 6),
    "bounded_by":       (12, 6, 6),
    "assumes":          (16, 8, 8),
}
CROSS_TARGETS = {
    # edge_type            total
    "cites_as_evidence":   4,
    "contradicts":         3,
    "extends":             4,
    "supersedes":          1,
    "uses_assumption_of":  5,
    "uses_method_of":      3,
}

# Fields we keep when summarizing a claim for the human grader. Anything not
# listed here is dropped to keep the screen readable.
CLAIM_DISPLAY_KEYS = (
    "claim_type",
    "verbatim_quote",
    "reaction_type", "subject", "subject_smiles", "property_name",
    "value", "unit", "measurement_method",
    "process_described", "steps", "key_intermediates",
    "technique_name", "what_it_achieves", "key_innovation", "limitations",
    "compared_items", "metric", "comparison_result",
    "hypothesis_text", "limitation_text", "direction_text",
    "finding_text", "why_surprising",
    "rationale", "evidence", "assumption", "epistemic_role",
    "conditions", "outcomes",
    "reactants", "products",
)


# ── DB helpers ───────────────────────────────────────────────────────────────


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def fetch_claim(con: sqlite3.Connection, claim_id: str) -> dict | None:
    r = con.execute(
        "SELECT data, source_doi FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if not r:
        return None
    try:
        d = json.loads(r["data"])
    except Exception:
        d = {}
    d.setdefault("claim_id", claim_id)
    d.setdefault("source_doi", r["source_doi"])
    return d


def fetch_paper_title(con: sqlite3.Connection, doi: str) -> str:
    r = con.execute("SELECT title FROM sources WHERE doi = ?", (doi,)).fetchone()
    return (r["title"] if r else "") or ""


def _claim_summary(c: dict) -> dict:
    """Trim a claim dict to the fields we want to show in the grader."""
    if not c:
        return {}
    out = {"claim_id": c.get("claim_id"), "source_doi": c.get("source_doi")}
    for k in CLAIM_DISPLAY_KEYS:
        v = c.get(k)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, str):
            v = v[:600]
        out[k] = v
    return out


# ── Sampling ────────────────────────────────────────────────────────────────


def _stratified_intra(con: sqlite3.Connection, rng: random.Random) -> list[dict]:
    """Pull one stratum at a time so we don't materialize 60k rows in memory."""
    ph = ",".join("?" * len(INTRA_EXTRACTORS))
    out: list[dict] = []
    for etype, (total, n_high, n_med) in INTRA_TARGETS.items():
        for conf, n_target in (("high", n_high), ("medium", n_med)):
            rows = con.execute(
                f"""SELECT id, from_claim_id, to_claim_id, edge_type, confidence,
                           evidence, extractor, extracted_at
                      FROM claim_edges
                     WHERE extractor IN ({ph})
                       AND edge_type = ?
                       AND confidence = ?
                       AND to_claim_id <> ''""",
                (*INTRA_EXTRACTORS, etype, conf),
            ).fetchall()
            rows = list(rows)
            rng.shuffle(rows)
            picked = rows[:n_target]
            if len(picked) < n_target:
                # Top up from the other confidence bucket if this one is short
                # (e.g. some types have very few medium edges).
                short = n_target - len(picked)
                fallback = con.execute(
                    f"""SELECT id, from_claim_id, to_claim_id, edge_type, confidence,
                               evidence, extractor, extracted_at
                          FROM claim_edges
                         WHERE extractor IN ({ph})
                           AND edge_type = ?
                           AND confidence <> ?
                           AND to_claim_id <> ''""",
                    (*INTRA_EXTRACTORS, etype, conf),
                ).fetchall()
                taken_ids = {r["id"] for r in picked}
                fallback = [r for r in fallback if r["id"] not in taken_ids]
                rng.shuffle(fallback)
                picked += fallback[:short]
            for r in picked:
                out.append(dict(r))
    return out


def _stratified_cross(con: sqlite3.Connection, rng: random.Random) -> list[dict]:
    ph = ",".join("?" * len(CROSS_EXTRACTORS))
    out: list[dict] = []
    for etype, n_target in CROSS_TARGETS.items():
        rows = con.execute(
            f"""SELECT id, from_claim_id, to_claim_id, edge_type, confidence,
                       evidence, extractor, extracted_at
                  FROM claim_edges
                 WHERE extractor IN ({ph})
                   AND edge_type = ?
                   AND to_claim_id <> ''""",
            (*CROSS_EXTRACTORS, etype),
        ).fetchall()
        rows = list(rows)
        rng.shuffle(rows)
        for r in rows[:n_target]:
            out.append(dict(r))
    return out


def _sample_extractor(con: sqlite3.Connection, extractor: str,
                      mode: str, rng: random.Random,
                      n: int, types_filter: list[str] | None = None) -> list[dict]:
    """Sample up to `n` edges for one extractor tag, with edge_type coverage:
    take floor(n / k_types) from each type first, then top up the remainder
    from the largest remaining pools (so dominant types still contribute when
    sparse types are exhausted)."""
    types = [r["edge_type"] for r in con.execute(
        "SELECT DISTINCT edge_type FROM claim_edges WHERE extractor = ?",
        (extractor,),
    ).fetchall()]
    if types_filter:
        types = [t for t in types if t in set(types_filter)]
    if not types:
        return []
    pools: dict[str, list[dict]] = {}
    for et in types:
        rows = con.execute(
            """SELECT id, from_claim_id, to_claim_id, edge_type, confidence,
                      evidence, extractor, extracted_at
                 FROM claim_edges
                WHERE extractor = ? AND edge_type = ?
                  AND to_claim_id <> ''""",
            (extractor, et),
        ).fetchall()
        pool = [dict(r) for r in rows]
        rng.shuffle(pool)
        pools[et] = pool

    per_type = max(1, n // len(types))
    out: list[dict] = []
    for et, pool in pools.items():
        for r in pool[:per_type]:
            r["_mode_hint"] = mode
            out.append(r)
        del pool[:per_type]

    # Top-up: round-robin over types with remaining edges (largest first) until
    # we hit the target n.
    while len(out) < n:
        remaining = [(et, p) for et, p in pools.items() if p]
        if not remaining:
            break
        remaining.sort(key=lambda kv: -len(kv[1]))
        et, pool = remaining[0]
        r = pool.pop(0)
        r["_mode_hint"] = mode
        out.append(r)

    rng.shuffle(out)
    return out[:n]


def cmd_sample(args):
    """Build the sample file with full endpoint context inlined."""
    rng = random.Random(args.seed)
    con = open_db()

    if args.extractor:
        # Pilot/single-extractor mode: pull only that extractor's edges.
        mode_hint = args.mode or ("cross" if "cross" in args.extractor else "intra")
        types_filter = (
            [t.strip() for t in args.types.split(",") if t.strip()]
            if args.types else None
        )
        picks = _sample_extractor(con, args.extractor, mode_hint, rng,
                                  args.n, types_filter=types_filter)
        intra = [e for e in picks if e.get("_mode_hint") == "intra"]
        cross = [e for e in picks if e.get("_mode_hint") == "cross"]
        print(f"[sample] extractor={args.extractor}: {len(picks)} edges "
              f"({len(intra)} intra + {len(cross)} cross)")
    else:
        intra = _stratified_intra(con, rng)
        cross = _stratified_cross(con, rng)
        print(f"selected {len(intra)} intra + {len(cross)} cross edges")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache claims and papers we touch — many edges share endpoints.
    claim_cache: dict[str, dict] = {}
    title_cache: dict[str, str] = {}

    def get_claim(cid: str) -> dict:
        if cid not in claim_cache:
            claim_cache[cid] = fetch_claim(con, cid) or {}
        return claim_cache[cid]

    def get_title(doi: str) -> str:
        if doi not in title_cache:
            title_cache[doi] = fetch_paper_title(con, doi)
        return title_cache[doi]

    rng.shuffle(intra)
    rng.shuffle(cross)
    all_edges = [("intra", e) for e in intra] + [("cross", e) for e in cross]
    rng.shuffle(all_edges)

    with out_path.open("w") as f:
        for idx, (mode, e) in enumerate(all_edges, 1):
            from_claim = get_claim(e["from_claim_id"])
            to_claim = get_claim(e["to_claim_id"])
            from_doi = from_claim.get("source_doi", "")
            to_doi = to_claim.get("source_doi", "")
            row = {
                "sample_idx": idx,
                "mode": mode,
                "edge_id": e["id"],
                "edge_type": e["edge_type"],
                "confidence": e["confidence"],
                "extractor": e["extractor"],
                "evidence": e["evidence"],
                "from_claim_id": e["from_claim_id"],
                "to_claim_id": e["to_claim_id"],
                "from_doi": from_doi,
                "to_doi": to_doi,
                "from_paper_title": get_title(from_doi) if from_doi else "",
                "to_paper_title": get_title(to_doi) if to_doi else "",
                "from_claim": _claim_summary(from_claim),
                "to_claim": _claim_summary(to_claim),
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(all_edges)} samples to {out_path}")


# ── Grading TUI ──────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _wrap(s: str, indent: str = "        ", width: int = 100) -> str:
    """Soft-wrap a long string for terminal display, preserving indent."""
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= width:
        return s
    out_lines = []
    line = ""
    for word in s.split():
        if len(line) + 1 + len(word) > width and line:
            out_lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out_lines.append(line)
    return ("\n" + indent).join(out_lines)


def _render_claim(c: dict, indent: str = "        ") -> str:
    """One-screen summary of a claim, with the verbatim quote front and centre."""
    if not c:
        return "(claim not found)"
    parts = []
    quote = c.get("verbatim_quote", "")
    if quote:
        parts.append(f'"{_wrap(quote, indent=indent)}"')
    meta_bits = []
    if c.get("claim_type"):
        meta_bits.append(f"type={c['claim_type']}")
    for k in (
        "process_described", "what_it_achieves", "finding_text",
        "hypothesis_text", "limitation_text", "comparison_result",
        "rationale", "assumption", "epistemic_role",
        "property_name", "value", "unit", "measurement_method",
        "reaction_type", "technique_name", "key_innovation",
    ):
        v = c.get(k)
        if v:
            v_str = ", ".join(v) if isinstance(v, list) else str(v)
            meta_bits.append(f"{k}={_wrap(v_str, indent=indent)[:200]}")
    if meta_bits:
        parts.append("; ".join(meta_bits))
    return ("\n" + indent).join(parts)


def _print_edge(idx: int, total: int, sample: dict) -> None:
    sep = "─" * 100
    print()
    print(sep)
    extr_short = sample["extractor"].replace("_llm_gemini", "").replace("_v1", "")
    print(
        f"[{idx}/{total}]  {sample['mode']} · {sample['edge_type']} · "
        f"confidence={sample['confidence']}  ({extr_short})"
    )
    if sample["mode"] == "cross":
        print(f"FROM paper: \"{sample['from_paper_title']}\"  ({sample['from_doi']})")
        print(f"TO   paper: \"{sample['to_paper_title']}\"  ({sample['to_doi']})")
    else:
        print(f"Paper:  \"{sample['from_paper_title']}\"  ({sample['from_doi']})")
    print()
    print(f"FROM ({sample['from_claim_id'][:12]}...):")
    print(f"        {_render_claim(sample['from_claim'])}")
    print()
    print(f"TO   ({sample['to_claim_id'][:12]}...):")
    print(f"        {_render_claim(sample['to_claim'])}")
    print()
    if sample.get("evidence"):
        print(f"LLM evidence: {_wrap(sample['evidence'], indent='              ')}")
    print(sep)


def cmd_grade(args):
    sample_path = Path(args.from_)
    results_path = Path(args.out)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    samples = _read_jsonl(sample_path)
    if not samples:
        print(f"no samples found at {sample_path}; run `sample` first", file=sys.stderr)
        sys.exit(1)

    existing = _read_jsonl(results_path)
    graded_ids = {r["edge_id"]: r for r in existing}
    print(f"loaded {len(samples)} samples; {len(graded_ids)} already graded")

    # Index by sample_idx so b (back) works naturally.
    samples_by_idx = {s["sample_idx"]: s for s in samples}
    order = sorted(samples_by_idx)
    total = len(order)

    # Resume position: first sample that is not yet graded.
    pos = 0
    while pos < total and samples_by_idx[order[pos]]["edge_id"] in graded_ids:
        pos += 1

    while 0 <= pos < total:
        idx = order[pos]
        s = samples_by_idx[idx]
        prior = graded_ids.get(s["edge_id"])
        os.system("clear" if os.name != "nt" else "cls")
        _print_edge(idx, total, s)
        if prior:
            print(f"prior verdict: {prior['verdict']}"
                  + (f" — {prior['comment']}" if prior.get("comment") else ""))
        prompt = "[y]es / [n]o / [s]kip / [b]ack / [c]omment / [q]uit-and-save  > "
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nsaved progress, exiting")
            return
        if choice == "q":
            print("saved progress, exiting")
            return
        if choice == "b":
            pos = max(0, pos - 1)
            continue
        if choice == "c":
            try:
                comment = input("comment > ").strip()
            except (EOFError, KeyboardInterrupt):
                comment = ""
            if prior:
                prior["comment"] = comment
                _rewrite_results(results_path, list(graded_ids.values()))
            else:
                # Don't advance until they actually verdict it.
                print("(comment recorded — now press y/n/s)")
                input("press enter to continue")
            continue
        verdict_map = {"y": "correct", "n": "wrong", "s": "skip"}
        if choice not in verdict_map:
            continue
        verdict = verdict_map[choice]
        record = {
            "edge_id": s["edge_id"],
            "sample_idx": idx,
            "mode": s["mode"],
            "edge_type": s["edge_type"],
            "confidence": s["confidence"],
            "extractor": s["extractor"],
            "verdict": verdict,
            "comment": prior.get("comment", "") if prior else "",
        }
        graded_ids[s["edge_id"]] = record
        # Append-only by default; only rewrite on edit/back-revisit.
        if prior:
            _rewrite_results(results_path, list(graded_ids.values()))
        else:
            with results_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        pos += 1

    print(f"\nall {total} samples graded. results: {results_path}")


def _rewrite_results(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Reporting ───────────────────────────────────────────────────────────────


def _precision(rows: list[dict]) -> tuple[int, int, float]:
    """(correct, evaluable, precision). 'skip' rows are excluded from the
    denominator since they represent 'I cannot tell', not 'wrong'."""
    evalable = [r for r in rows if r["verdict"] in ("correct", "wrong")]
    correct = sum(1 for r in evalable if r["verdict"] == "correct")
    n = len(evalable)
    return correct, n, (correct / n if n else 0.0)


def _line(label: str, rows: list[dict], width: int = 24) -> str:
    c, n, p = _precision(rows)
    skip = sum(1 for r in rows if r["verdict"] == "skip")
    skip_note = f"  ({skip} skipped)" if skip else ""
    return f"  {label:<{width}}{c:>3}/{n:<3}  ({p:.2f}){skip_note}"


def cmd_report(args):
    results_path = Path(args.from_)
    results = _read_jsonl(results_path)
    if not results:
        print(f"no results at {results_path}", file=sys.stderr)
        sys.exit(1)

    print(f"loaded {len(results)} graded edges from {results_path}")
    print()
    c, n, p = _precision(results)
    skipped_total = sum(1 for r in results if r["verdict"] == "skip")
    print(f"OVERALL                {c}/{n}  ({p:.2f})  [{skipped_total} skipped]")
    print()

    by_mode: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_conf_intra: dict[str, list[dict]] = defaultdict(list)
    by_conf_cross: dict[str, list[dict]] = defaultdict(list)
    by_type_conf: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
        by_type[r["edge_type"]].append(r)
        if r["mode"] == "intra":
            by_conf_intra[r["confidence"]].append(r)
        else:
            by_conf_cross[r["confidence"]].append(r)
        by_type_conf[(r["edge_type"], r["confidence"])].append(r)

    print("By mode:")
    for mode in sorted(by_mode):
        print(_line(mode, by_mode[mode]))
    print()

    print("By edge_type:")
    for etype in sorted(by_type, key=lambda k: -len(by_type[k])):
        print(_line(etype, by_type[etype]))
    print()

    if by_conf_intra:
        print("By confidence (intra):")
        for conf in ("high", "medium", "low"):
            if by_conf_intra.get(conf):
                print(_line(conf, by_conf_intra[conf]))
        print()
    if by_conf_cross:
        print("By confidence (cross):")
        for conf in ("high", "medium", "low"):
            if by_conf_cross.get(conf):
                print(_line(conf, by_conf_cross[conf]))
        print()

    # Wrong-edge surfaces — useful for prompt iteration.
    wrong = [r for r in results if r["verdict"] == "wrong"]
    if wrong:
        print(f"Wrong edges ({len(wrong)}) by (type, confidence):")
        bt: dict[tuple[str, str], int] = defaultdict(int)
        for r in wrong:
            bt[(r["edge_type"], r["confidence"])] += 1
        for (etype, conf), n in sorted(bt.items(), key=lambda kv: -kv[1]):
            print(f"  {etype:<24}{conf:<8}{n}")
        print()

    if any(r.get("comment") for r in results):
        print("Notable comments:")
        for r in results:
            c_ = (r.get("comment") or "").strip()
            if c_:
                tag = "OK" if r["verdict"] == "correct" else r["verdict"].upper()
                print(f"  [{tag}] {r['edge_type']} ({r['confidence']}): {c_}")


# ── Entry ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="draw a stratified sample of edges")
    sp.add_argument("--out", default=str(DEFAULT_SAMPLE_PATH))
    sp.add_argument("--seed", type=int, default=20260417)
    sp.add_argument("--extractor", type=str, default=None,
                    help="If given, sample ONLY edges from this extractor tag "
                         "(bypasses the stratified intra+cross plan)")
    sp.add_argument("--n", type=int, default=30,
                    help="Sample size when --extractor is given (default 30)")
    sp.add_argument("--mode", type=str, default=None,
                    help="Force mode label for --extractor edges "
                         "('intra' or 'cross'); default inferred from name")
    sp.add_argument("--types", type=str, default=None,
                    help="Comma-separated edge_type whitelist (e.g. "
                         "'contradicts,supersedes' for the spicy spot-check)")
    sp.set_defaults(func=cmd_sample)

    g = sub.add_parser("grade", help="interactive grader")
    g.add_argument("--from", dest="from_", default=str(DEFAULT_SAMPLE_PATH))
    g.add_argument("--out", default=str(DEFAULT_RESULTS_PATH))
    g.set_defaults(func=cmd_grade)

    r = sub.add_parser("report", help="precision report")
    r.add_argument("--from", dest="from_", default=str(DEFAULT_RESULTS_PATH))
    r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
