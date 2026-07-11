"""Aggregate per-stage timings from `[search_profile]` lines in ablation logs."""
import re
import statistics
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "ablation_logs"

# Each profile line: [search_profile] start+0ms entry+0ms imports+0ms ...
PROFILE_RE = re.compile(r"\[search_profile\]\s+(.+)$")
TOKEN_RE = re.compile(r"(\w+)\+(\d+)ms")

# Stages in order; we report median ms FROM ENTRY (subtract previous mark).
STAGES = [
    ("imports", "entry"),
    ("variants", "imports"),
    ("conn", "variants"),
    ("tree_recall", "conn"),
    ("author", "tree_recall"),
    ("paper_recall", "author"),
    ("fts", "paper_recall"),
    ("embed_query", "fts"),
    ("faiss_search", "embed_query"),
    ("pre_rerank", "faiss_search"),
    ("rerank", "pre_rerank"),
    ("done", "rerank"),
]


def parse_log(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        m = PROFILE_RE.search(line)
        if not m:
            continue
        marks = {k: int(v) for k, v in TOKEN_RE.findall(m.group(1))}
        if "done" not in marks:
            continue
        rows.append(marks)
    return rows


def summarize(rows):
    # Per-stage delta = mark[stage] - mark[prev]
    deltas = {s: [] for s, _ in STAGES}
    for r in rows:
        for stage, prev in STAGES:
            if stage in r and prev in r:
                deltas[stage].append(r[stage] - r[prev])
    return {s: (statistics.median(v) if v else 0) for s, v in deltas.items()}


def main():
    labels = {
        "baseline": "baseline.log",
        "a1-mat256": "a1-mat256.log",
        "a3-rerank30": "a3-rerank30.log",
        "a4-noprf": "a4-noprf.log",
        "a5-notreerk": "a5-notreerk.log",
        "a6-cache (cold)": "a6-cache.log",
    }
    print(f"{'stage':<14} " + " ".join(f"{k:>14}" for k in labels))
    print("-" * (14 + 15 * len(labels)))
    summaries = {k: summarize(parse_log(LOG_DIR / v)) for k, v in labels.items()}
    for stage, _ in STAGES:
        row = [stage] + [str(summaries[k][stage]) + "ms" for k in labels]
        print(f"{row[0]:<14} " + " ".join(f"{c:>14}" for c in row[1:]))
    print("\ntotal (sum of medians):")
    print(f"{'sum':<14} "
          + " ".join(
              f"{sum(summaries[k][s] for s, _ in STAGES):>12} ms"
              for k in labels
          ))


if __name__ == "__main__":
    main()
