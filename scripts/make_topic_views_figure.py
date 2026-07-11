#!/usr/bin/env python3
"""Topic-across-views figure for the AskChem EMNLP demo paper.

Shows how one topic ("CO2 reduction") decomposes differently across AskChem's
NINE content views. For a sample of retrieved claims we aggregate, per view,
the leading categories:

  * 7 taxonomic views (reaction type, substance class, application, technique,
    mechanism, data, claim type) -> top-level (L1) node.
  * Author view -> most prolific authors on the topic (from paper metadata).
  * Network view -> dominant claim-to-claim relationship types (typed edges of
    the claim knowledge graph).

Outputs PDF + PNG to structure_the_universe_paper/figures/fig_topic_views.*.

Usage:
    .venv-benchmark/bin/python scripts/make_topic_views_figure.py
"""
from __future__ import annotations

import collections
import json
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "structure_the_universe_paper" / "figures"

BASE = "https://askchem.org/api"
TOPIC = "CO2 reduction"
TOPIC_DISPLAY = "CO$_2$ reduction"   # subscripted 2 for the figure title
LIMIT = 500
# Author whose coauthor ego-network is drawn for the Author panel (a prominent
# CO2-reduction electrochemist); resolved to an OpenAlex id at runtime.
AUTHOR_QUERY = "Koper"

# (kind, view_id, title, color). kind: "path" | "coauthor" | "claimnet"
PANELS = [
    ("path", "by_reaction_type", "Reaction type", "#1f6f8b"),
    ("path", "by_substance_class", "Substance class", "#147d64"),
    ("path", "by_application", "Application", "#b9770e"),
    ("path", "by_technique", "Technique", "#7e3f98"),
    ("path", "by_mechanism", "Mechanism", "#2e86c1"),
    ("path", "by_data", "Data", "#996515"),
    ("path", "by_claim_type", "Claim type", "#a93226"),
    ("coauthor", None, "Author  \u00b7  coauthor network", "#5b7c99"),
    ("time", None, "Time", "#2e86c1"),
    ("claimnet", None, "Network  \u00b7  claim graph", "#4c6b8a"),
]
# grid slots (4x3): center the 10th panel (Network) in the last row; hide 9 & 11.
PANEL_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
HIDDEN_SLOTS = [9, 11]
TIME_FILL = "#2e86c1"

# Node/edge colours copied verbatim from the website (web/index.html
# NODE_COLORS / EDGE_COLORS) so the Network panel matches askchem.org exactly.
CTYPE_COLOR = {
    "reaction": "#3b82f6", "method": "#10b981", "mechanism": "#a855f7",
    "property": "#f59e0b", "comparison": "#ec4899",
    "computational_result": "#06b6d4", "surprising_finding": "#ef4444",
}
_NODE_DEFAULT = "#94a3b8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=90) as r:
        return json.loads(r.read())


def _wrap(label: str, width: int = 42) -> str:
    """Full label, wrapped to <=2 lines (never truncated)."""
    s = str(label).replace("_", " ").strip()
    lines = textwrap.wrap(s, width=width, break_long_words=False) or [s]
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return "\n".join(lines)


def _path_counts(vid, results):
    c: collections.Counter = collections.Counter()
    for r in results:
        path = (r.get("view_paths") or {}).get(vid)
        if path:
            c[str(path[0])] += 1
    return c.most_common(4)[::-1]


def _bar_panel(ax, top, color):
    names = [t[0] for t in top]
    vals = [t[1] for t in top]
    step = 1.42
    ypos = [i * step for i in range(len(top))]
    ax.barh(ypos, vals, color=color, height=0.56, zorder=2)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c8ccd0")
    ax.tick_params(axis="x", labelsize=15, colors="black", length=3)
    maxv = max(vals) if vals else 1
    ax.set_xlim(0, maxv * 1.16)
    ax.set_ylim(-0.78, (len(top) - 1) * step + 1.02 if top else 1)
    for i, (nm, v) in enumerate(zip(names, vals)):
        ax.text(0, ypos[i] + 0.42, _wrap(nm), ha="left", va="bottom",
                fontsize=16, color="black")
        ax.text(v, ypos[i], f" {v}", va="center", fontsize=15, fontweight="bold",
                color="black")


def _last_name(name: str) -> str:
    parts = str(name).replace(".", "").split()
    return parts[-1] if parts else str(name)


def _author_bar_panel(ax, topic):
    """Top authors on the topic, by number of topic papers (bar chart)."""
    try:
        res = _get(f"authors?topic={urllib.parse.quote(topic)}&limit=8")
        auth = res.get("authors") or res.get("results") or []
    except Exception:
        auth = []
    rows = []
    for a in auth:
        nm = a.get("name", "")
        cnt = a.get("topic_papers") or a.get("matching_papers") or a.get("papers_in_index") or 0
        if nm and cnt:
            rows.append((nm, int(cnt)))
    rows = rows[:4][::-1]   # top 4, ascending so largest sits on top in barh
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "(no authors)", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#94a3b8")
        return
    _bar_panel(ax, rows, "#5b7c99")


def _coauthor_panel(ax, author_query, topic, show_title=True):
    """Ego coauthor network of a topic-prominent author."""
    ax.axis("off")
    try:
        res = _get(f"authors?q={urllib.parse.quote(author_query)}&topic={urllib.parse.quote(topic)}")
        cands = res.get("authors") or res.get("results") or []
        cands = [a for a in cands if (a.get("author_id") or a.get("id"))]
        cands.sort(key=lambda a: a.get("papers_in_index") or a.get("paper_count") or 0, reverse=True)
        aid = cands[0].get("author_id") or cands[0].get("id")
        center_name = cands[0].get("name", author_query)
        net = _get(f"authors/{aid}/network?depth=1&limit=12")
    except Exception:
        ax.text(0.5, 0.5, "(coauthor network unavailable)", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#94a3b8")
        return
    nodes = net.get("nodes", [])
    edges = net.get("edges", [])
    if len(nodes) <= 1:
        ax.text(0.5, 0.5, "(no coauthors indexed)", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#94a3b8")
        return
    G = nx.Graph()
    cid = None
    for nd in nodes:
        G.add_node(nd["id"], name=nd.get("name", ""), depth=nd.get("depth", 1))
        if nd.get("depth") == 0:
            cid = nd["id"]
    for e in edges:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"], w=e.get("weight", 1))
    # Ego star layout: hub in the middle, coauthors evenly spaced on a ring.
    import math
    others = [n for n in G if n != cid]
    R = 0.78                                       # shorter spokes -> smaller graph
    pos = {cid: (0.0, 0.0)} if cid is not None else {}
    ang_of = {}
    for j, node in enumerate(others):
        ang = 2 * math.pi * j / max(len(others), 1) + math.pi / 2
        ang_of[node] = ang
        pos[node] = (R * math.cos(ang), R * math.sin(ang))
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos, edge_color="#c2ccd6", width=1.2)
    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, nodelist=others, node_size=300,
                           node_color="#8fa8bf", edgecolors="white", linewidths=1.0)
    if cid is not None:
        nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, nodelist=[cid], node_size=1500,
                               node_color="#1f4e6b", edgecolors="white", linewidths=1.4)
    # coauthor names sit OUTSIDE their dots, pushed radially away from the hub so
    # they never collide with neighbouring dots; alignment follows the direction.
    R_lbl = R + 0.20
    for node in others:
        ang = ang_of[node]
        lx, ly = R_lbl * math.cos(ang), R_lbl * math.sin(ang)
        c = math.cos(ang)
        ha = "left" if c > 0.30 else ("right" if c < -0.30 else "center")
        ax.text(lx, ly, _last_name(G.nodes[node]["name"]), fontsize=15,
                color="black", ha=ha, va="center")
    if cid is not None:
        ax.text(0, 0, _last_name(G.nodes[cid]["name"]), fontsize=12,
                color="white", fontweight="bold", ha="center", va="center")
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.32, 1.32)
    if show_title:
        ax.set_title("Author", fontsize=18, fontweight="bold", pad=8, color="black")


EDGE_COLOR = {
    "supports": "#22c55e", "assumes": "#a855f7", "bounded_by": "#f59e0b",
    "interprets": "#06b6d4", "derives_from": "#3b82f6", "sub_step_of": "#64748b",
    "uses_method_of": "#0ea5e9", "uses_assumption_of": "#7c3aed",
    "extends": "#10b981", "supersedes": "#f97316", "contradicts": "#ef4444",
    "cites_as_evidence": "#94a3b8", "co_mention": "#cbd5e1",
}


def _claimnet_fetch(topic):
    """Fetch the topic claim graph once; return (components>=4 nodes, ntype, nhit)."""
    try:
        g = _get(f"search/graph?q={urllib.parse.quote(topic)}&limit=500&expand=one_hop")
    except Exception:
        return [], {}, {}
    ntype = {n["id"]: (n.get("claim_type") or "") for n in g.get("nodes", [])}
    nhit = {n["id"]: bool(n.get("in_search")) for n in g.get("nodes", [])}
    G = nx.DiGraph()
    for e in g.get("edges", []):
        if e.get("from") and e.get("to"):
            G.add_edge(e["from"], e["to"], type=e.get("type") or "")
    comps = [G.subgraph(c).copy()
             for c in sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)]
    comps = [c for c in comps if c.number_of_nodes() >= 4]
    return comps, ntype, nhit


def _claimnet_draw(ax, comp, ntype, nhit):
    """Draw one connected component as a hub-centered star; return edge types seen."""
    import math
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if comp is None:
        return []
    cxp, cyp, scale = 0.5, 0.6, 0.24
    deg = dict(comp.degree())
    hub = max(deg, key=deg.get)
    others = [n for n in comp.nodes if n != hub]
    pos = {hub: (cxp, cyp)}
    for j, node in enumerate(others):
        ang = 2 * math.pi * j / max(len(others), 1) - math.pi / 2
        pos[node] = (cxp + scale * math.cos(ang), cyp + scale * math.sin(ang))
    ecol = [EDGE_COLOR.get(comp.edges[e].get("type", ""), "#94a3b8") for e in comp.edges]
    nx.draw_networkx_edges(ax=ax, G=comp, pos=pos, edge_color=ecol, width=2.4,
                           arrowsize=16, arrowstyle="-|>", node_size=440, alpha=0.98)
    ncol = "#8fa8bf"
    ecolors = ["#f5b301" if nhit.get(n) else "#334155" for n in comp]
    lws = [3.0 if nhit.get(n) else 1.2 for n in comp]
    nx.draw_networkx_nodes(ax=ax, G=comp, pos=pos, node_size=440, node_color=ncol,
                           edgecolors=ecolors, linewidths=lws)
    present = []
    for e in comp.edges:
        t = comp.edges[e].get("type", "")
        if t and t not in present:
            present.append(t)
    return present


def _time_counts(results):
    """Paper counts per publication decade over ALL matched papers (uncapped),
    via /api/search/time -- not just the top-K retrieved sample. Oldest ->
    newest so the barh y-axis reads decades bottom-up."""
    try:
        d = _get(f"search/time?q={urllib.parse.quote(TOPIC)}").get("decades", {})
    except Exception:
        d = {}
    if not d:
        return []
    def _yr(k):
        try:
            return int(str(k).rstrip("s"))
        except ValueError:
            return -1
    # Fold sparse pre-2000 decades into one "pre-2000" bar so the panel stays a
    # clean 4-bar chart (matching its siblings) while still counting every paper.
    pre = sum(v for k, v in d.items() if 0 <= _yr(k) < 2000)
    out = [(k, v) for k, v in d.items() if _yr(k) >= 2000]
    out.sort(key=lambda kv: _yr(kv[0]))
    if pre:
        out.insert(0, ("pre-2000", pre))
    return out


# 8 horizontal-bar panels (7 content taxonomies + Time), 4 per row.
BAR_PANELS = [
    ("by_reaction_type", "Reaction type", "#1f6f8b"),
    ("by_substance_class", "Substance class", "#147d64"),
    ("by_application", "Application", "#b9770e"),
    ("by_technique", "Technique", "#7e3f98"),
    ("by_mechanism", "Mechanism", "#2e86c1"),
    ("by_data", "Data", "#996515"),
    ("by_claim_type", "Claim type", "#a93226"),
    ("__time__", "Time", "#2a9d8f"),
]


def main() -> None:
    results = _get(f"search?q={urllib.parse.quote(TOPIC)}&limit={LIMIT}").get("results", [])
    n = len(results)

    # Wide canvas so the 8 bar panels (top two rows) are large. The two graph
    # panels are placed as FIXED-width axes in the bottom band, so widening the
    # figure enlarges the bars only -- the graphs keep a constant width.
    W, H = 17.5, 10.0
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 4, hspace=0.40, wspace=0.22,
                          left=0.038, right=0.992, top=0.905, bottom=0.42)

    bar_axes = []
    for idx, (vid, title, color) in enumerate(BAR_PANELS):
        r, cc = divmod(idx, 4)
        ax = fig.add_subplot(gs[r, cc])
        bar_axes.append(ax)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=8, color="black")
        top = _time_counts(results) if vid == "__time__" else _path_counts(vid, results)
        if top:
            _bar_panel(ax, top, color)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#94a3b8")

    # Fixed-width graph panels (Author star + claim network), centered in the
    # bottom band. Width in figure fraction = constant inches / canvas width.
    # Bottom band: 4 panels across -- Author (bar + coauthor graph) and
    # Network (two claim-graph components). Fixed geometry in figure fractions.
    gb, gh = 0.08, 0.25
    pw = 0.205
    bar_box = bar_axes[0].get_position()
    abar_h = bar_box.height
    abar_w = bar_box.width
    # Vertically center the shorter bar chart against the full-height graphs so
    # the author bar chart and coauthor graph share the same visual center.
    abar_y = gb + (gh - abar_h) / 2

    # --- Author group: bar chart + coauthor ego-graph -----------------
    x_abar = 0.04
    x_agraph = x_abar + abar_w + 0.03
    ax_abar = fig.add_axes([x_abar, abar_y, abar_w, abar_h])
    _author_bar_panel(ax_abar, TOPIC)
    ax_agraph = fig.add_axes([x_agraph, gb, pw, gh])
    _coauthor_panel(ax_agraph, AUTHOR_QUERY, TOPIC, show_title=False)
    author_center = (x_abar + x_agraph + pw) / 2

    # --- Network group: two claim-graph components, placed close together ---
    comps, ntype, nhit = _claimnet_fetch(TOPIC)
    net_center = 0.76
    net_gap = 0.01
    net_pw = 0.17          # narrower than the bar/coauthor panels -> less flat
    x_n1 = net_center - net_pw - net_gap / 2
    x_n2 = net_center + net_gap / 2
    ax_n1 = fig.add_axes([x_n1, gb, net_pw, gh])
    present = _claimnet_draw(ax_n1, comps[0] if len(comps) > 0 else None, ntype, nhit)
    ax_n2 = fig.add_axes([x_n2, gb, net_pw, gh])
    present += [t for t in _claimnet_draw(ax_n2, comps[1] if len(comps) > 1 else None, ntype, nhit)
                if t not in present]

    # Group titles centered over each group, sitting just above the panels.
    title_y = gb + gh + 0.004
    fig.text(author_center, title_y, "Author", fontsize=18, fontweight="bold",
             ha="center", va="bottom", color="black")
    fig.text(net_center, title_y, "Network", fontsize=18, fontweight="bold",
             ha="center", va="bottom", color="black")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=EDGE_COLOR.get(t, "#aab4be"), lw=2.6,
                      label=t.replace("_", " ")) for t in present[:4]]
    if handles:
        leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                         fontsize=16, frameon=False, handlelength=1.3,
                         columnspacing=1.2, handletextpad=0.45,
                         bbox_to_anchor=(net_center, 0.075))
        for txt in leg.get_texts():
            txt.set_color("black")

    fig.suptitle(
        f"How \u201c{TOPIC_DISPLAY}\u201d claims decompose across AskChem's ten views",
        fontsize=20, fontweight="bold", y=0.978, color="black",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_topic_views.{ext}")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_topic_views.pdf'} (n={n} claims)")


if __name__ == "__main__":
    main()
