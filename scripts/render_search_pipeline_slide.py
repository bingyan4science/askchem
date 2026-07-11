"""Render docs/search-pipeline-slide.png from a hand-laid layout.

The mermaid block in docs/search-pipeline.md is the authoritative
diagram for the doc; this script renders a slide-friendly horizontal
variant of the same flow for presentations. Re-run with:

    .venv-benchmark/bin/python scripts/render_search_pipeline_slide.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "search-pipeline-slide.png"


def _box(ax, x, y, w, h, text, face, edge, fontsize=11, fontweight="normal",
         text_color="#111827"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.4, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight,
        color=text_color, wrap=True,
    )
    return (x, y, w, h)


def _arrow(ax, src, dst, color="#9ca3af", lw=1.6, style="->",
           connectionstyle="arc3,rad=0.0"):
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    # Anchor to right edge of src and left edge of dst by default.
    p1 = (sx + sw, sy + sh / 2)
    p2 = (dx, dy + dh / 2)
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw,
        connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def _arrow_xy(ax, p1, p2, color="#9ca3af", lw=1.6,
              connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle="->", mutation_scale=12,
        color=color, linewidth=lw,
        connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.2, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title + subtitle
    ax.text(8, 8.45, "AskChem Search Pipeline",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="#0f172a")
    ax.text(8, 7.95,
            "5 recall channels  \u2192  RRF  \u2192  cross-encoder  \u2192  results",
            ha="center", va="center", fontsize=12, color="#475569")

    # ── Stage 0: query ──
    query = _box(ax, 0.3, 4.1, 1.4, 0.8, "query", "#f3f4f6", "#9ca3af",
                 fontsize=12, fontweight="bold")

    # ── Cache (diamond-ish) ──
    cache = _box(ax, 2.1, 4.05, 1.7, 0.9,
                 "result LRU\n(opt-in)",
                 "#fef3c7", "#d97706",
                 fontsize=10, fontweight="bold")

    # ── Recall fan-out (5 boxes stacked vertically) ──
    recall_x, recall_w, recall_h = 5.0, 2.2, 0.7
    recall_specs = [
        ("FTS5\n(claim text)",                "#dbeafe", "#1d4ed8"),
        ("Dense vector\n(mxbai 256-d FAISS)", "#dcfce7", "#15803d"),
        ("Tree BFS\n(taxonomy)",              "#ede9fe", "#6d28d9"),
        ("Paper-level\n(source_fts + claim-guided)", "#ffedd5", "#c2410c"),
        ("Author\n(name-triggered)",          "#fce7f3", "#be185d"),
    ]
    recall_ys = [6.5, 5.5, 4.5, 3.5, 2.5]
    recall_boxes = []
    for (label, face, edge), y in zip(recall_specs, recall_ys):
        recall_boxes.append(
            _box(ax, recall_x, y, recall_w, recall_h, label, face, edge,
                 fontsize=9.5, fontweight="bold",
                 text_color="#1f2937")
        )

    # Section header above the fan-out
    ax.text(recall_x + recall_w / 2, 7.5, "parallel recall",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#374151")

    # ── RRF merge ──
    rrf = _box(ax, 8.0, 4.05, 1.7, 0.9,
               "RRF merge", "#e0f2fe", "#0369a1",
               fontsize=12, fontweight="bold")

    # ── Cross-encoder ──
    rerank = _box(ax, 10.0, 4.0, 2.4, 1.0,
                  "cross-encoder\nrerank (top 30)",
                  "#ccfbf1", "#0f766e",
                  fontsize=11, fontweight="bold")

    # ── Diversify + filter ──
    div = _box(ax, 12.7, 4.05, 1.7, 0.9,
               "diversify\n+ filter",
               "#f3f4f6", "#6b7280",
               fontsize=11, fontweight="bold")

    # ── Results ──
    out = _box(ax, 14.6, 4.1, 1.2, 0.8,
               "results", "#dcfce7", "#15803d",
               fontsize=12, fontweight="bold")

    # ── Arrows ──
    grey = "#9ca3af"
    _arrow(ax, query, cache, color=grey)

    # Cache hit short-circuit (curved BELOW the recall fan-out, from the
    # bottom of the cache to the bottom of the results box).
    cx, cy, cw, ch = cache
    _arrow_xy(ax,
              (cx + cw / 2, cy),
              (out[0] + out[2] / 2, out[1]),
              color="#d97706",
              connectionstyle="arc3,rad=0.45")
    ax.text(8.5, 2.0, "cache hit \u2192 return",
            ha="center", va="center", fontsize=10,
            color="#b45309", style="italic", fontweight="bold")

    # Cache miss: cache -> each recall box
    cache_right = (cx + cw, cy + ch / 2)
    for rb in recall_boxes:
        _arrow_xy(ax, cache_right, (rb[0], rb[1] + rb[3] / 2),
                  color=grey, lw=1.2)

    # Each recall -> RRF
    rrf_left = (rrf[0], rrf[1] + rrf[3] / 2)
    for rb in recall_boxes:
        _arrow_xy(ax, (rb[0] + rb[2], rb[1] + rb[3] / 2), rrf_left,
                  color=grey, lw=1.2)

    # RRF -> rerank -> diversify -> results
    _arrow(ax, rrf, rerank, color=grey, lw=1.8)
    _arrow(ax, rerank, div, color=grey, lw=1.8)
    _arrow(ax, div, out, color=grey, lw=1.8)

    # ── Footer line: prod-shipped knobs ── placed BELOW the cache-hit
    # arc, which dips to about y=1.0 with rad=0.45.
    ax.text(8, 0.55,
            "prod (May 15, 2026): v2 retrieval \u00b7 Matryoshka 256-d \u00b7 "
            "rerank window 30 \u00b7 PRF off \u00b7 tree-rerank off \u00b7 "
            "result LRU on",
            ha="center", va="center", fontsize=10.5, color="#475569")
    ax.text(8, 0.18,
            "warm p50 0.2\u20130.3 s   \u00b7   cold p50 ~3 s   \u00b7   "
            "nDCG@10 = 0.783",
            ha="center", va="center", fontsize=10.5, color="#475569",
            fontweight="bold")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
