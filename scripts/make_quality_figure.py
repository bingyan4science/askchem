#!/usr/bin/env python3
"""Data-quality overview figure for the AskChem EMNLP demo paper.

A compact dashboard mirroring askchem.org's Quality tab: headline KPIs plus
panels for extraction depth, coverage over time, chemistry subfield coverage,
and claim-type distribution. Data is pulled live from /api/quality.

Outputs PDF + PNG to structure_the_universe_paper/figures/fig_quality_overview.*.

Usage:
    .venv-benchmark/bin/python scripts/make_quality_figure.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "structure_the_universe_paper" / "figures"
API = "https://askchem.org/api/quality"

NAVY = "#0f3d5e"; TEAL = "#0e7490"; BLUE = "#2e86c1"; AMBER = "#b9770e"
PURPLE = "#7e3f98"; GREEN = "#147d64"; SLATE = "#5b7c99"; INK = "#1f2933"
GRID = "#e3e7ea"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def _fmt(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:.0f}"


LABEL_OVERRIDE = {
    "computational_result": "Computation",
    "experimental_design": "Experiments",
    "surprising_finding": "Surprises",
    "scope_entry": "Scope",
    "physical_chemistry": "Physical",
    "medicinal_chemistry": "Medicinal",
}


def _title(s: str) -> str:
    key = str(s).lower()
    if key in LABEL_OVERRIDE:
        return LABEL_OVERRIDE[key]
    return s.replace("_", " ").replace("|", " / ").title()


def _kpi_band(ax, data):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    dv = data.get("doi_verification", {})
    ed = data.get("extraction_depth", {})
    tot_ex = (ed.get("full_paper", 0) + ed.get("abstract_only", 0)) or 1
    fullpct = round(ed.get("full_paper", 0) / tot_ex * 100)
    yr = data.get("year_range", ["", ""])
    cards = [
        (_fmt(data.get("total_claims", 0)), "grounded claims", NAVY),
        (_fmt(data.get("total_sources", 0)), "source papers", TEAL),
        (f"{yr[0]}\u2013{yr[1]}", "year coverage", SLATE),
        (f"{dv.get('verification_rate', 0)}%", "DOI-verified (CrossRef)", GREEN),
        (f"{fullpct}%", "full-paper claims", AMBER),
    ]
    n = len(cards); gap = 0.012; w = (1 - gap * (n - 1)) / n
    for i, (big, lab, col) in enumerate(cards):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.04), w, 0.92,
                     boxstyle="round,pad=0.004,rounding_size=0.02",
                     linewidth=0, facecolor=col, transform=ax.transAxes, zorder=2))
        ax.text(x + w / 2, 0.62, big, ha="center", va="center", transform=ax.transAxes,
                fontsize=15, fontweight="bold", color="white", zorder=3)
        ax.text(x + w / 2, 0.22, lab, ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="white", zorder=3)


def _hbar(ax, pairs, color, title, note=None, topn=10):
    pairs = pairs[:topn][::-1]
    names = [_title(k) for k, _ in pairs]
    vals = [v for _, v in pairs]
    y = range(len(pairs))
    ax.barh(y, vals, color=color, height=0.8, zorder=2)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=6, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK, length=3)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _fmt(v) if v else "0"))
    ax.margins(y=0.02)
    mx = max(vals) if vals else 1
    ax.set_xlim(0, mx * 1.02)
    # Space-saving labels: name INSIDE long bars (white); to the RIGHT of short
    # bars (dark). Reclaims the whole left gutter so bars span full width.
    for i, (nm, v) in enumerate(zip(names, vals)):
        if v >= mx * 0.34:                                   # bar long enough for inside label
            # label + number grouped together at the bar's end
            ax.text(v - mx * 0.02, i, f"{nm}  {_fmt(v)}", va="center", ha="right",
                    color="white", fontsize=9, fontweight="bold")
        else:
            ax.text(v + mx * 0.02, i, f"{nm}  {_fmt(v)}", va="center", ha="left",
                    color=INK, fontsize=9)


def _donut(ax, data):
    """Extraction depth as two proportion bars -- distinguishing CLAIMS (by
    source) from PAPERS (full-text vs abstract-only). The full-paper minority of
    papers yields the majority of claims."""
    ed = data.get("extraction_depth", {})
    fc = ed.get("full_paper", 0); ac = ed.get("abstract_only", 0)
    tc = data.get("total_claims", fc + ac) or 1
    fp = ed.get("full_paper_papers", 0); tp = data.get("total_sources", 0) or 1
    ap = max(tp - fp, 0)
    ax.set_title("Extraction depth", fontsize=12.5, fontweight="bold", pad=6, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 2.35); ax.axis("off")

    rows = [("Claims", tc, fc, ac), ("Papers", tp, fp, ap)]
    ys, h = [1.45, 0.35], 0.46
    for (name, tot, full, ab), y in zip(rows, ys):
        ff = full / (full + ab or 1)
        ax.barh(y, ff, height=h, left=0, color=NAVY, zorder=2)
        ax.barh(y, 1 - ff, height=h, left=ff, color=AMBER, zorder=2)
        ax.text(0, y + h / 2 + 0.1, f"{name}  ({_fmt(tot)})", fontsize=9.5,
                fontweight="bold", color=INK, va="bottom")
        ax.text(ff / 2, y, f"{round(ff*100)}%", ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        ax.text(ff + (1 - ff) / 2, y, f"{round((1-ff)*100)}%", ha="center",
                va="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(ff / 2, y - h / 2 - 0.06, _fmt(full), ha="center", va="top",
                color=INK, fontsize=7.5)
        ax.text(ff + (1 - ff) / 2, y - h / 2 - 0.06, _fmt(ab), ha="center", va="top",
                color=INK, fontsize=7.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NAVY, label="full paper (Gemini 3.1 Pro)"),
                       Patch(color=AMBER, label="abstract only (GPT-5-mini)")],
              loc="lower center", ncol=2, fontsize=7.5, frameon=False,
              handlelength=1.1, columnspacing=1.0, bbox_to_anchor=(0.5, -0.05))


def _timeline(ax, data):
    yd = {int(k): v for k, v in data.get("year_distribution", {}).items() if str(k).isdigit()}
    yrs = sorted(y for y in yd if 1990 <= y <= 2026)
    vals = [yd[y] for y in yrs]
    ax.fill_between(yrs, vals, color=BLUE, alpha=0.28, zorder=1)
    ax.plot(yrs, vals, color=BLUE, lw=1.6, zorder=2)
    ax.set_title("Papers by year  (1990\u20132026)", fontsize=12.5, fontweight="bold",
                 pad=6, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(labelsize=8.5, colors=INK, length=3)
    ax.set_xlim(1990, 2026)
    ax.margins(y=0.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _fmt(v)))


def main() -> None:
    with urllib.request.urlopen(API, timeout=60) as r:
        data = json.loads(r.read())

    ct = sorted(data.get("claim_type_distribution", {}).items(), key=lambda x: -x[1])
    sf = sorted(data.get("subfield_coverage", {}).items(), key=lambda x: -x[1])
    dv = data.get("doi_verification", {})

    # Authored at ~2-column-span width so point-size fonts render true size in the
    # paper (no downscaling that would make text tiny).
    fig = plt.figure(figsize=(7.0, 6.4))
    gs = GridSpec(3, 2, height_ratios=[0.5, 1.0, 1.0], hspace=0.5, wspace=0.18,
                  left=0.04, right=0.985, top=0.9, bottom=0.065)

    _kpi_band(fig.add_subplot(gs[0, :]), data)
    _hbar(fig.add_subplot(gs[1, 0]), ct, TEAL, "Claim-type distribution", topn=10)
    _timeline(fig.add_subplot(gs[1, 1]), data)
    _hbar(fig.add_subplot(gs[2, 0]), sf, NAVY, "Subfield coverage", topn=10)
    _donut(fig.add_subplot(gs[2, 1]), data)

    fig.suptitle("AskChem data quality at a glance", fontsize=14, fontweight="bold",
                 x=0.04, ha="left", y=0.955)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_quality_overview.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_quality_overview.pdf'}")


if __name__ == "__main__":
    main()
