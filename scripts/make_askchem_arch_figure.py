#!/usr/bin/env python3
"""Generate the AskChem platform architecture figure for the EMNLP demo paper.

Top-to-bottom layered diagram: literature sources -> live update pipeline
-> shared claim store -> stabilized faceted taxonomy + evidence graph +
exploratory Living Taxonomy -> interfaces -> chemists and AI agents.

Rendered near single-column width so text is legible without downscaling.
Outputs PDF + PNG into structure_the_universe_paper/figures/.

Usage:
    .venv-benchmark/bin/python scripts/make_askchem_arch_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "structure_the_universe_paper" / "figures"

# Cohesive, print-friendly palette (muted, layer-coded).
SRC = "#64748b"     # slate    — external sources (inputs)
PIPE = "#0e7490"    # teal     — live pipeline (process)
CLAIM = "#0f3d5e"   # navy     — claim store (data foundation / anchor)
VIEW = "#2e86c1"    # blue     — multi-view index (product)
GRAPH = "#14866d"   # green    — evidence graph (relational structure)
TREE = "#7e3f98"    # purple   — living taxonomy tree (product)
SERVE = "#b9770e"   # amber    — access layer (interfaces)
CONS = "#273240"    # charcoal — consumers (users)
ARROW = "#475569"   # slate-600 connectors
INK = "#1f2933"     # near-black text on light captions

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def _box(ax, x, y, w, h, text, fc, tc="white", fs=7.5, fw="bold"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, edgecolor="none", facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=3,
            linespacing=1.3)
    return (x + w / 2, y + h / 2)


def _arrow(ax, p0, p1, color=ARROW, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, zorder=1, shrinkA=3, shrinkB=3,
        capstyle="round"))


def main() -> None:
    # Slightly wider than one column so multi-line labels fit inside their boxes
    # (font size is fixed in points, so a wider canvas = more room per box).
    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(1.7, 14.2)      # crop to content (no dead space above/below)
    ax.axis("off")
    cx = 6.0
    # left/right column centres shared by the interface and consumer layers.
    lc, rc = cx - 2.9, cx + 2.9
    colw = 5.4

    # 1. Sources (where papers are discovered; CrossRef is metadata/citations
    #    only, not a discovery source, so it is not shown here)
    src_y = 12.95
    labels = ["arXiv", "ChemRxiv", "journals", "Semantic\nScholar"]
    sw, gap = 2.5, 0.3
    total = len(labels) * sw + (len(labels) - 1) * gap
    x0 = cx - total / 2
    src_pts = [_box(ax, x0 + i * (sw + gap), src_y, sw, 0.9, lb, SRC, fs=7.5)
               for i, lb in enumerate(labels)]
    ax.text(cx, src_y + 1.05, "Chemistry literature  ·  147K papers  ·  1925-2026",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)

    # 2. Pipeline
    pipe_y = 11.0
    pipe_top = pipe_y + 0.9
    p_center = _box(ax, cx - 5.6, pipe_y, 11.2, 0.9,
                    "Live pipeline:  extract $\\rightarrow$ classify $\\rightarrow$ "
                    "index $\\rightarrow$ embed $\\rightarrow$ sync",
                    PIPE, fs=7.5)
    for p in src_pts:
        _arrow(ax, (p[0], src_y), (p[0], pipe_top), lw=0.7)

    # 3. Claim store -- the single foundation everything else is built from
    core_y, core_h = 9.2, 1.05
    cw = 6.4
    c_claims = _box(ax, cx - cw / 2, core_y, cw, core_h,
                    "Claim store\n2.4M grounded claims", CLAIM, fs=8.0)
    _arrow(ax, (cx, pipe_y), (cx, core_y + core_h))

    # 4. Complementary structures over one shared claim store.
    der_y, der_h = 6.55, 1.55
    der_top = der_y + der_h
    dw, dgap = 3.55, 0.25
    dcenters = [cx - (dw + dgap), cx, cx + (dw + dgap)]
    _box(ax, dcenters[0] - dw / 2, der_y, dw, der_h,
         "Stabilized faceted\ntaxonomy\nwhat is it about?", VIEW, fs=6.8)
    _box(ax, dcenters[1] - dw / 2, der_y, dw, der_h,
         "Evidence graph\nhow are findings\nrelated?", GRAPH, fs=6.8)
    _box(ax, dcenters[2] - dw / 2, der_y, dw, der_h,
         "Living Taxonomy\nwhat principle\ngoverns it?", TREE, fs=6.8)
    for dc in dcenters:
        _arrow(ax, (cx, core_y), (dc, der_top))

    # 5. Access layer exposes the same claim identities and structures.
    acc_y, acc_h = 4.5, 0.9
    acc_top = acc_y + acc_h
    acc_labels = ["Web UI", "REST", "SDK", "MCP"]
    aw, agap = 2.6, 0.3
    atot = len(acc_labels) * aw + (len(acc_labels) - 1) * agap
    ax0 = cx - atot / 2
    acc_pts = [_box(ax, ax0 + i * (aw + agap), acc_y, aw, acc_h, lb, SERVE, fs=8.0)
               for i, lb in enumerate(acc_labels)]
    _arrow(ax, (dcenters[0], der_y), (acc_pts[0][0], acc_top))
    _arrow(ax, (dcenters[1], der_y), (cx, acc_top))
    _arrow(ax, (dcenters[2], der_y), (acc_pts[-1][0], acc_top))

    # 6. Consumers  (Web UI/REST -> chemists; SDK/MCP -> AI agents)
    con_y, con_h = 2.05, 1.45
    con_top = con_y + con_h
    _box(ax, lc - colw / 2, con_y, colw, con_h,
         "Chemists\nsearch $\\cdot$ browse $\\cdot$\nsubscription $\\cdot$ reading lists",
         CONS, fs=7.0)
    _box(ax, rc - colw / 2, con_y, colw, con_h,
         "AI agents\ngrounded answers $\\cdot$\nself-service keys",
         CONS, fs=7.0)
    for p in acc_pts[:2]:
        _arrow(ax, (p[0], acc_y), (lc, con_top), lw=0.7)
    for p in acc_pts[2:]:
        _arrow(ax, (p[0], acc_y), (rc, con_top), lw=0.7)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig1_platform.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig1_platform.pdf'}")


if __name__ == "__main__":
    main()
