#!/usr/bin/env python3
"""Living-taxonomy SUBTREE figure for the AskChem EMNLP demo paper.

Renders one illustrative branch of the living taxonomy as a clean top-down tree
(NOT a UI screenshot) so it stays legible at print size and makes the structure
explicit:

    Principle  (Chemical kinetics)
        -> Theory     (Enzyme catalysis theory, with the Michaelis-Menten law)
            -> Mechanism x4 (each an LLM-written definition + governing
               equation where one exists, grounded in DOI-linked papers)

The abstraction ladder (principle / theory / mechanism) is labelled down the
left margin -- the whole point of the "living taxonomy" is that internal nodes
are real scientific concepts, enriched with definitions + equations, above the
leaf claims.

Data is read live from the taxonomy_* tables so the figure is reproducible.

Usage:
    .venv-benchmark/bin/python scripts/make_ltree_subtree_figure.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "structure_the_universe_paper" / "figures"
sys.path.insert(0, str(REPO_ROOT / "src"))

from askchem import db  # noqa: E402

VIEW = "by_reaction_type"
ROOT_NODE = "Enzyme catalysis theory"   # the "theory" node we feature

# palette: one hue family, darkening up the abstraction ladder
C_PRINCIPLE = "#1b3a4b"
C_THEORY = "#1f6f8b"
C_MECH = "#e8f1f4"
C_MECH_EDGE = "#1f6f8b"
INK = "#12222b"
CHIP = "#b9770e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def _mathify(eq: str) -> str | None:
    """Best-effort convert a stored LaTeX-ish equation to matplotlib mathtext."""
    if not eq:
        return None
    e = eq.strip().replace(r"\text{", r"\mathrm{").replace(r"\circ", r"\degree")
    return f"${e}$"


def _fetch():
    with db.get_conn() as c:
        node = {r[0]: dict(name=r[1], deff=r[2], eq=r[3], short=r[4])
                for r in c.execute("SELECT node_id,name,definition,equation,"
                                   "short_label FROM taxonomy_nodes").fetchall()}
        edges = c.execute("SELECT parent_id, child_id FROM taxonomy_edges "
                          "WHERE view_id=?", [VIEW]).fetchall()
        ch, parent = {}, {}
        for p, k in edges:
            ch.setdefault(p, []).append(k)
            parent[k] = p
        tid = next(n for n in node if node[n]["name"] == ROOT_NODE)

        def papers(nid):
            seen, stack = set(), [nid]
            while stack:
                x = stack.pop()
                for (d,) in c.execute("SELECT DISTINCT doi FROM taxonomy_leaves "
                                      "WHERE view_id=? AND node_id=?", [VIEW, x]):
                    seen.add(d)
                stack += ch.get(x, [])
            return len(seen)

        principle = node[parent[tid]]
        theory = node[tid]
        theory["papers"] = papers(tid)
        principle["papers"] = papers(parent[tid])
        mechs = []
        for k in ch.get(tid, []):
            m = dict(node[k]); m["papers"] = papers(k)
            mechs.append(m)
    return principle, theory, mechs


def _first_sentence(text: str, cap: int) -> str:
    if not text:
        return ""
    s = text.strip()
    dot = s.find(". ")
    if 0 < dot < cap:
        return s[:dot + 1]
    return (s[:cap].rsplit(" ", 1)[0] + "\u2026") if len(s) > cap else s


def _box(ax, cx, cy, w, h, *, title, definition, eq, papers,
         face, edge, title_color, def_color,
         title_fs, eq_fs, def_fs, def_wrap, def_cap, def_lines):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2))
    top = cy + h / 2
    y = top - 0.028
    ax.text(cx, y, title, ha="center", va="top", zorder=3,
            fontsize=title_fs, fontweight="bold", color=title_color)
    y -= 0.011 + title_fs * 0.0016
    m = _mathify(eq)
    if m:
        try:
            ax.text(cx, y, m, ha="center", va="top", fontsize=eq_fs,
                    color=title_color, zorder=3)
        except Exception:
            ax.text(cx, y, eq, ha="center", va="top", fontsize=eq_fs - 2,
                    color=title_color, zorder=3)
        y -= 0.016 + eq_fs * 0.0026
        if "\\frac" in eq:          # fractions render ~2 lines tall
            y -= 0.034
    if definition:
        lines = textwrap.wrap(_first_sentence(definition, def_cap), width=def_wrap)
        if len(lines) > def_lines:
            lines = lines[:def_lines]
            lines[-1] = lines[-1].rstrip(".") + "\u2026"
        ax.text(cx, y, "\n".join(lines), ha="center", va="top", fontsize=def_fs,
                color=def_color, zorder=3, linespacing=1.16)
    if papers:
        ax.text(cx, cy - h / 2 + 0.016, f"{papers:,} papers", ha="center", va="bottom",
                fontsize=7.6, fontweight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.28", fc=CHIP, ec="none"))


def _connect(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-", mutation_scale=1,
        linewidth=1.3, color="#93a4ad", zorder=1,
        connectionstyle="arc3,rad=0.0"))


def main():
    principle, theory, mechs = _fetch()
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # abstraction-ladder row labels (left margin)
    for ylab, txt in [(0.885, "PRINCIPLE"), (0.63, "THEORY"), (0.235, "MECHANISM")]:
        ax.text(0.006, ylab, txt, ha="left", va="center", rotation=90,
                fontsize=9, fontweight="bold", color="#5b7c99", zorder=3)

    # PRINCIPLE
    px, py, ph = 0.53, 0.885, 0.13
    _box(ax, px, py, 0.42, ph, title=principle["name"],
         definition=None, eq=principle["eq"], papers=None,
         face=C_PRINCIPLE, edge=C_PRINCIPLE, title_color="white", def_color="#dbe7ec",
         title_fs=12.5, eq_fs=11, def_fs=8, def_wrap=48, def_cap=0, def_lines=0)

    # THEORY
    tx, ty, th = 0.53, 0.63, 0.225
    _box(ax, tx, ty, 0.62, th, title=theory["name"],
         definition=theory["deff"], eq=theory["eq"], papers=theory["papers"],
         face=C_THEORY, edge=C_THEORY, title_color="white", def_color="#eef6f8",
         title_fs=12.5, eq_fs=11, def_fs=8, def_wrap=64, def_cap=88, def_lines=2)

    # MECHANISMS (row of 4)
    n = len(mechs)
    mw, mh, my = 0.222, 0.27, 0.30
    xs = [0.055 + (i + 0.5) / n * 0.89 for i in range(n)]
    _connect(ax, px, py - ph / 2, tx, ty + th / 2)   # principle -> theory
    for mx, m in zip(xs, mechs):
        _connect(ax, tx, ty - th / 2, mx, my + mh / 2)   # theory -> mechanism
        _box(ax, mx, my, mw, mh, title=m["short"] or m["name"],
             definition=m["deff"], eq=m["eq"], papers=m["papers"],
             face=C_MECH, edge=C_MECH_EDGE, title_color=INK, def_color="#2c4049",
             title_fs=9.6, eq_fs=7.3, def_fs=7.0, def_wrap=26, def_cap=80, def_lines=3)

    fig.suptitle("A living-taxonomy subtree: internal nodes are enriched "
                 "scientific concepts\n(definition + governing equation) grounded "
                 "in DOI-linked papers",
                 fontsize=12.5, fontweight="bold", y=1.015, color=INK)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_living_subtree.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_living_subtree.pdf'}")


if __name__ == "__main__":
    main()
