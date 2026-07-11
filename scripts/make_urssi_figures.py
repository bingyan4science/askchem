#!/usr/bin/env python3
"""Generate figures for the URSSI Early-Career Fellowship proposal (AskChem).

Three figures (for a 3-page proposal):

Figure 1 (fig1_flatvs) — conceptual: flat ranked document list (today's search)
            vs. AskChem's structured, multi-view, source-grounded claim index.

Figure 2 (fig2_architecture) — the AskChem pipeline as reusable software:
            extract -> structure (multi-view) -> serve (MCP/SDK/REST),
            with the components proposed for open release highlighted.

Figure 3 (fig3_bench) — preliminary evidence from AskChem-Bench: structured,
            source-grounded retrieval makes answers verifiable (100% existing
            DOIs vs. ~12% hallucinated for an unaugmented LLM). Real numbers
            from scripts/benchmark_results_gpt-5.5.json (aggregate.overall.<mode>).

Outputs PNG (300 dpi) + PDF into docs/proposals/figures/.

Usage::

    .venv-benchmark/bin/python scripts/make_urssi_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "scripts" / "benchmark_results_gpt-5.5.json"
OUT_DIR = REPO_ROOT / "docs" / "proposals" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# System display order + labels. Keys match aggregate.overall.<mode>.
SYSTEMS = [
    ("alone",             "LLM alone"),
    ("notebooklm",        "NotebookLM"),
    ("paperclip_unified", "+Paperclip"),
    ("edison_scientific", "Edison"),
    ("unified",           "+AskChem"),
]

# NYU violet brand palette.
NYU = "#57068C"          # primary brand accent
NYU_LIGHT = "#8b54b0"    # lighter violet
NYU_PALE = "#efe7f5"     # pale violet fill
GREY = "#9aa3ad"         # neutral / context

COLORS = {
    "alone": "#9aa3ad",
    "notebooklm": "#c9a13b",
    "paperclip_unified": "#3a8f86",
    "edison_scientific": "#5a7184",
    "unified": NYU,        # +AskChem highlighted in NYU violet
}


def _val(cell, key):
    v = cell.get(key)
    if isinstance(v, dict):
        return v.get("mean")
    return v


def figure1():
    d = json.loads(BENCH.read_text())
    overall = d["aggregate"]["overall"]

    # Four reliability-relevant panels. Each tuple: (metric_key, title,
    # normaliser, y-label). DOI existence and on-topic are rates in [0,1];
    # the other two are unbounded so we plot raw.
    panels = [
        ("doi_existence_rate",      "DOI existence rate",        "pct"),
        ("paper_relevance_high_rate", "On-topic rate (relevance >= high)", "pct"),
        ("citation_density",        "Citation density (per answer)", "raw"),
        ("grounded_specificity",    "Grounded specificity",      "raw"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (key, title, kind) in zip(axes, panels):
        xs = []
        ys = []
        cs = []
        for mode, label in SYSTEMS:
            cell = overall.get(mode, {})
            v = _val(cell, key)
            if v is None:
                continue
            if kind == "pct":
                v = v * 100.0
            xs.append(label)
            ys.append(v)
            cs.append(COLORS[mode])
        bars = ax.bar(xs, ys, color=cs, edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontweight="bold")
        if kind == "pct":
            ax.set_ylim(0, 105)
            ax.set_ylabel("%")
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + (1.5 if kind == "pct" else max(ys) * 0.02),
                    f"{y:.0f}" if kind == "pct" else f"{y:.1f}",
                    ha="center", va="bottom", fontsize=7)
        ax.tick_params(axis="x", rotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Preliminary evidence: structured, source-grounded retrieval makes answers verifiable "
        "(GPT-5.5 reader; 5 systems, n=30 chemistry questions)",
        fontsize=11, fontweight="bold", y=1.04,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig3_bench.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig3_bench.png'}")


def _box(ax, xy, w, h, text, fc, fontsize=8.5, fontweight="normal", tc="white"):
    box = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor="white", facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=tc, zorder=3,
            wrap=True)
    return (xy[0] + w / 2, xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color="#34495e", zorder=1,
        shrinkA=2, shrinkB=2,
    ))


def figure2():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    RELEASE = NYU      # proposed for open-source release
    CONTEXT = GREY     # current prototype infrastructure

    # Pipeline row (descriptive component names)
    ROW_Y = 2.5
    c_ingest = _box(ax, (0.3, ROW_Y), 1.7, 1.0,
                    "Ingestion\n(multi-source)", CONTEXT, fontsize=8.5)
    c_extract = _box(ax, (2.4, ROW_Y), 1.9, 1.0,
                     "Extraction\nengine", RELEASE, fontweight="bold")
    c_valid = _box(ax, (4.7, ROW_Y), 1.8, 1.0,
                   "Validation\ntools", RELEASE, fontweight="bold")
    c_index = _box(ax, (6.9, ROW_Y), 1.9, 1.0,
                   "Multi-view\nstructure", RELEASE, fontweight="bold")
    c_access = _box(ax, (9.2, ROW_Y), 2.5, 1.0,
                    "SDK + MCP server\n(agent / API access)",
                    RELEASE, fontweight="bold")

    for a, b in [(c_ingest, c_extract), (c_extract, c_valid),
                 (c_valid, c_index), (c_index, c_access)]:
        _arrow(ax, a, b)

    # evaluation harness below, spanning
    BENCH_Y = 0.5
    c_bench = _box(ax, (3.2, BENCH_Y), 5.6, 1.0,
                   "Evaluation harness  (askchem-bench)\n"
                   "DOI existence - grounding - relevance - on-topic",
                   RELEASE, fontweight="bold", fontsize=8.5)
    _arrow(ax, (c_index[0], ROW_Y), (c_bench[0] + 1.4, BENCH_Y + 1.0))
    _arrow(ax, (c_access[0], ROW_Y), (c_bench[0] + 2.8, BENCH_Y + 1.0))
    # feedback arrow back up to extraction (dashed)
    ax.add_patch(FancyArrowPatch(
        (c_bench[0] - 2.8, BENCH_Y + 0.5), (c_extract[0], ROW_Y),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
        color=NYU, linestyle=(0, (4, 2)), zorder=1,
        shrinkA=2, shrinkB=2,
    ))
    ax.text(1.9, 1.55, "quality\nfeedback", fontsize=7,
            color=NYU, ha="center", style="italic")

    # Legend (two categories)
    handles = [
        mpatches.Patch(color=RELEASE, label="Proposed for open-source release"),
        mpatches.Patch(color=CONTEXT, label="Current prototype infrastructure"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.05), frameon=False)

    ax.set_title("AskChem pipeline: ingest $\\rightarrow$ extract $\\rightarrow$ validate $\\rightarrow$ structure $\\rightarrow$ serve agent-first",
                 fontsize=10.5, fontweight="bold", pad=22)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig2_architecture.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig2_architecture.png'}")


def fig_flatvs():
    """Headline conceptual figure: flat ranked document list (today's search)
    vs. AskChem's structured, multi-view, source-grounded claim index.
    """
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    for ax in (axl, axr):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    GREY_BOX = "#9aa3ad"
    QUERY = NYU
    GREEN = "#2e8b57"      # semantic "verified" green
    VIEW = NYU_LIGHT
    CHIP = NYU_PALE

    # ---- Left: flat ranked list ----
    axl.set_title("Today: keyword / vector search", fontsize=11, fontweight="bold")
    _box(axl, (1.2, 8.7), 7.6, 0.9, "query:  \"Ni catalysts for C-N coupling\"",
         QUERY, fontsize=9, fontweight="bold")
    for i in range(5):
        y = 7.2 - i * 1.35
        _box(axl, (1.2, y), 7.6, 1.0,
             f"Paper {i+1}  (PDF, ranked by similarity)", GREY_BOX, fontsize=9)
    axl.text(5.0, 0.2, "flat list of documents -- read each one yourself",
             ha="center", fontsize=8.5, style="italic", color="#7f8c8d")

    # ---- Right: structured multi-view claim index ----
    axr.set_title("AskChem: atomic claims in multiple views",
                  fontsize=11, fontweight="bold")
    qx = _box(axr, (0.4, 8.7), 9.2, 0.9,
              "same query  ->  structured, source-grounded answer",
              QUERY, fontsize=9, fontweight="bold")

    views = [
        ("by_reaction_type", 6.7),
        ("by_mechanism",     4.1),
        ("by_application",   1.5),
    ]
    for name, vy in views:
        vc = _box(axr, (0.4, vy), 2.5, 1.0, name, VIEW, fontsize=8.5,
                  fontweight="bold")
        _arrow(axr, (qx[0], 8.7), (vc[0], vy + 1.0))
        # two claim chips per view
        for j in range(2):
            cx = 3.4
            cy = vy + 0.55 - j * 1.05
            chip = FancyBboxPatch(
                (cx, cy), 6.1, 0.85,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=0.8, edgecolor=GREEN, facecolor=CHIP, zorder=2)
            axr.add_patch(chip)
            axr.text(cx + 0.15, cy + 0.42,
                     "claim + conditions",
                     ha="left", va="center", fontsize=8, color="#1b2631", zorder=3)
            axr.text(cx + 5.95, cy + 0.42, "DOI verified",
                     ha="right", va="center", fontsize=7.5, color=GREEN,
                     fontweight="bold", zorder=3)
            _arrow(axr, (vc[0] + 1.25, vy + 0.5), (cx, cy + 0.42))

    axr.text(5.0, 0.15,
             "navigable hierarchy of grounded claims -- queryable by agents (MCP) and humans",
             ha="center", fontsize=8.5, style="italic", color="#7f8c8d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig1_flatvs.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig1_flatvs.png'}")


if __name__ == "__main__":
    fig_flatvs()      # Figure 1 (headline)
    figure2()         # Figure 2 (architecture / pipeline)
    figure1()         # Figure 3 (askchem-bench evidence; writes fig3_bench)
