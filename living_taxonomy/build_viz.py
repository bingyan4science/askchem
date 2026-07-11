"""Build a standalone HTML radial tree from pilot placement results.

Reads ``output/placements_<view>.json`` + the seed tree, nests placed
leaves under their principle -> mechanism branch, adds an "Exceptions"
pseudo-branch for leaves that did not fit, and writes a self-contained
``output/tree_<view>.html`` (D3 radial tree, zoom/pan, collapsible).

Usage:
    python3 living_taxonomy/build_viz.py --view by_reaction_type
    open living_taxonomy/output/tree_by_reaction_type.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import seed_trees

OUT_DIR = _HERE / "output"


def _clone_structure(node):
    """Copy the seed tree as a hierarchy skeleton (no leaves yet)."""
    return {
        "name": node["name"],
        "kind": node.get("kind", "node"),
        "children": [_clone_structure(c) for c in node.get("children", [])],
        "leaves": [],
    }


def _find_branch(root, branch_name):
    """Find a node by name (branch leaf-host is unique by name in seed trees)."""
    if root["name"] == branch_name:
        return root
    for c in root["children"]:
        hit = _find_branch(c, branch_name)
        if hit:
            return hit
    return None


def _to_d3(node):
    """Convert skeleton + leaves into a D3 hierarchy dict."""
    children = [_to_d3(c) for c in node["children"]]
    for lf in node["leaves"]:
        children.append({
            "name": lf["label"],
            "kind": "leaf",
            "score": lf["score"],
            "doi": lf["doi"],
            "year": lf["year"],
            "full": lf["text"],
        })
    out = {"name": node["name"], "kind": node["kind"]}
    if children:
        out["children"] = children
    out["count"] = _count_leaves(node)
    return out


def _count_leaves(node):
    n = len(node["leaves"])
    for c in node["children"]:
        n += _count_leaves(c)
    return n


def build_hierarchy(view):
    data = json.loads((OUT_DIR / f"placements_{view}.json").read_text())
    skeleton = _clone_structure(seed_trees.PILOT_TREES[view])
    exceptions = {"name": "Exceptions (proposed new branches)",
                  "kind": "exception", "children": [], "leaves": []}

    for p in data["placements"]:
        label = p["text"][:48].replace("\n", " ")
        leaf = {"label": label, "text": p["text"], "doi": p["doi"],
                "year": p["year"], "score": p["score"]}
        if p["decision"] == "exception":
            exceptions["leaves"].append(leaf)
        else:
            branch_name = p["branch_path"][-1]
            target = _find_branch(skeleton, branch_name)
            (target or skeleton)["leaves"].append(leaf)

    root_d3 = _to_d3(skeleton)
    if exceptions["leaves"]:
        exc_d3 = _to_d3(exceptions)
        root_d3.setdefault("children", []).append(exc_d3)
    return root_d3, data


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Living tree — __TITLE__</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  html,body{margin:0;height:100%;background:#0f1721;color:#cbd5e1;
       font:13px system-ui,sans-serif;overflow:hidden}
  #hdr{position:fixed;top:0;left:0;right:0;padding:8px 14px;background:#0b121b;
       border-bottom:1px solid #1e293b;z-index:10;display:flex;align-items:center;gap:14px}
  #hdr b{color:#e2e8f0;font-size:15px}
  #hdr .sub{color:#64748b}
  #hdr select,#hdr button{background:#1e293b;color:#e2e8f0;border:1px solid #334155;
       border-radius:6px;padding:4px 8px;font:12px system-ui;cursor:pointer}
  .link{fill:none;stroke:#334155;stroke-width:1.2px}
  .node circle{stroke:#0f1721;stroke-width:1.5px;cursor:pointer}
  text.lbl{paint-order:stroke;stroke:#0b121b;stroke-width:3.5px;
       stroke-linejoin:round;font-weight:600;cursor:pointer}
  text.lbl .ct{fill:#64748b}
  .shared{stroke:#fbbf24 !important;stroke-width:2px !important;stroke-dasharray:2,2}
  .tip{position:fixed;pointer-events:none;background:#1e293b;border:1px solid #334155;
       padding:6px 9px;border-radius:6px;max-width:360px;color:#e2e8f0;font-size:12px;
       opacity:0;transition:opacity .1s;z-index:20}
  .legend{position:fixed;bottom:10px;left:14px;color:#64748b;z-index:10}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px;
       vertical-align:middle}
</style></head><body>
<div id="hdr">
  <b>__TITLE__</b>
  <select id="viewsel"></select>
  <button id="expand">expand all</button>
  <button id="collapse">collapse</button>
  <span class="sub">__SUB__</span>
</div>
<div class="legend">
  <i style="border:1px dashed #475569"></i>open root
  <i style="background:#fbbf24"></i>law <i style="background:#38bdf8"></i>framework
  <i style="background:#a78bfa"></i>theory <i style="background:#c4b5fd"></i>model
  <i style="background:#5eead4"></i>mechanism <i style="background:#34d399"></i>leaf
  <i style="background:#f87171"></i>exception
  <i style="border:1px dashed #fbbf24"></i>shared across views
  <i style="border:1px dashed #f59e0b"></i>proposed branch (new)</div>
<div class="tip" id="tip"></div>
<svg id="svg"></svg>
<script>
const VIEWS = __VIEWS__;            // {label: hierarchy}
const W = window.innerWidth, H = window.innerHeight;
const DX = 200, DY = 96, LBLW = 168;  // sibling gap, depth gap, label width
const color = {open_root:"#475569", root:"#e2e8f0",
               law:"#fbbf24", framework:"#38bdf8", theory:"#a78bfa",
               model:"#c4b5fd", principle:"#38bdf8",
               mechanism:"#5eead4", class:"#5eead4",
               leaf:"#34d399", exception:"#f87171"};
const fsize = {open_root:13, law:15, framework:15, theory:13, model:12,
               principle:14, mechanism:12.5, class:12.5, leaf:11, exception:13};
const ffill = {open_root:"#94a3b8", law:"#fcd34d", framework:"#7dd3fc",
               theory:"#c4b5fd", model:"#ddd6fe", principle:"#7dd3fc",
               mechanism:"#99f6e4", class:"#99f6e4", leaf:"#86efac",
               exception:"#fca5a5"};

const svg = d3.select("#svg").attr("width",W).attr("height",H);
const g = svg.append("g");
const gLink = g.append("g").attr("class","links");   // drawn first -> behind
const gNode = g.append("g").attr("class","nodes");   // drawn after -> in front
const zoom = d3.zoom().scaleExtent([0.12,3]).on("zoom",e=>g.attr("transform",e.transform));
svg.call(zoom);
const tip = d3.select("#tip");
let _tipFor = null;
const tree = d3.tree().nodeSize([DX,DY]).separation((a,b)=> a.parent===b.parent?1:1.3);

let root;
function nodeR(d){ return d.data.kind==="leaf"?3.2: Math.max(4,Math.sqrt((d.data.count||1))*1.4); }
function hiddenCount(d){ return d._children? d._children.length : 0; }
function wrapText(s, max){ const words=String(s).split(/\\s+/); const out=[]; let cur="";
  for(const w of words){ if((cur+" "+w).trim().length>max){ if(cur)out.push(cur); cur=w; }
    else cur=(cur+" "+w).trim(); }
  if(cur)out.push(cur);
  if(out.length>3){ out.length=3; out[2]=out[2].slice(0,max-1)+"\\u2026"; } return out; }
function hideTip(){ tip.style("opacity",0); _tipFor=null; }
function showTip(e,d){
  if(_tipFor===d){ hideTip(); return; }          // second click toggles off
  _tipFor=d;
  tip.style("opacity",1).style("left",(e.clientX+12)+"px")
  .style("top",(e.clientY+12)+"px")
  .html(`<b>${d.data.year||""}</b>${d.data.score?" &middot; sim "+d.data.score.toFixed(2):""}`
    +`<br>${d.data.full||d.data.name}<br><span style='color:#64748b'>${d.data.doi||""}</span>`); }
function nodeClick(e,d){ if(d.data.kind==="leaf"){ showTip(e,d); return; }
  if(!d.children && !d._children) return;
  if(d.children){d._children=d.children;d.children=null;}
  else{d.children=d._children;d._children=null;} update(); }

function loadView(name){
  root = d3.hierarchy(VIEWS[name]);
  root.descendants().forEach(d=>{                 // start collapsed below depth 2
    if(d.depth>=2 && d.children){ d._children=d.children; d.children=null; }
  });
  update(true);
}
function expandAll(){ root.each(d=>{ if(d._children){d.children=d._children;d._children=null;} }); update(); }
function collapseTo(depth){ root.descendants().forEach(d=>{
    if(d.depth>=depth && d.children){d._children=d.children;d.children=null;}
    else if(d.depth<depth && d._children){d.children=d._children;d._children=null;}
  }); update(); }

function update(recenter){
  tree(root);
  const nodes = root.descendants(), links = root.links();
  gLink.selectAll("path.link").data(links,d=>d.target.data.__id||(d.target.data.__id=d.target.data.name+d.target.depth+Math.random()))
    .join("path").attr("class","link")
    .attr("d", d3.linkVertical().x(d=>d.x).y(d=>d.y));
  const node = gNode.selectAll("g.node").data(nodes,d=>d.data.__id||(d.data.__id=d.data.name+d.depth+Math.random()))
    .join("g").attr("class","node").attr("transform",d=>`translate(${d.x},${d.y})`);

  node.selectAll("circle").data(d=>[d]).join("circle")
    .attr("r",nodeR)
    .attr("fill",d=> d.data.kind==="open_root"?"none":(color[d.data.kind]||"#64748b"))
    .attr("class",d=> d.data.shared?"shared":null)
    .attr("stroke",d=> d.data.proposed?"#f59e0b":(d.data.kind==="open_root"?"#475569":(d.data.shared?"#fbbf24":"#0f1721")))
    .style("stroke-dasharray",d=> (d.data.proposed||d.data.kind==="open_root")?"3,3":null)
    .style("stroke-width",d=> d.data.proposed?"2.5px":null)
    .style("cursor","pointer")
    .on("click",(e,d)=> nodeClick(e,d));

  const txt = node.selectAll("text.lbl").data(d=>[d]).join("text").attr("class","lbl")
    .attr("text-anchor","middle")
    .attr("font-size",d=> fsize[d.data.kind]||12)
    .attr("fill",d=> d.data.proposed?"#fbbf24":(ffill[d.data.kind]||"#e2e8f0"))
    .on("click",(e,d)=> nodeClick(e,d));
  txt.each(function(d){
    const sel=d3.select(this); sel.selectAll("tspan").remove();
    const raw = d.data.kind==="leaf"? (d.data.name||"").slice(0,70) : d.data.name;
    const lines = wrapText(raw, 24);
    const nkids = (d.children? d.children.length:0) + (d._children? d._children.length:0);
    const tail = nkids? "  ("+nkids+")" : "";
    const y0 = nodeR(d)+12;
    lines.forEach((ln,i)=> sel.append("tspan").attr("x",0)
        .attr("dy", i===0? y0 : 13)
        .text(ln + (i===lines.length-1? tail : "")));
  });

  if(recenter){
    svg.transition().duration(300).call(zoom.transform,
       d3.zoomIdentity.translate(W/2, 70).scale(0.85));
  }
}

// view selector
const sel = d3.select("#viewsel");
const keys = Object.keys(VIEWS);
keys.forEach(k=> sel.append("option").attr("value",k).text(k));
sel.style("display", keys.length>1?null:"none");
sel.on("change", e=> loadView(e.target.value));
d3.select("#expand").on("click",expandAll);
d3.select("#collapse").on("click",()=>collapseTo(2));
loadView(keys[0]);
</script></body></html>"""


def render_html(root_d3, title, subtitle, out_path, views=None):
    """Write a self-contained vertical-tree HTML.

    ``views`` (optional) maps a view label -> hierarchy dict; when given, a
    selector swaps between them over the shared trunk. Otherwise a single
    ``root_d3`` is shown.
    """
    if views is None:
        views = {title: root_d3}
    html = (_HTML
            .replace("__TITLE__", title)
            .replace("__SUB__", subtitle)
            .replace("__VIEWS__", json.dumps(views)))
    Path(out_path).write_text(html)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="by_reaction_type",
                    choices=list(seed_trees.PILOT_TREES.keys()))
    args = ap.parse_args()

    root_d3, data = build_hierarchy(args.view)
    out = OUT_DIR / f"tree_{args.view}.html"
    sub = f"{data['n_leaves']} leaves &middot; {data['n_papers']} papers"
    render_html(root_d3, args.view, sub, out)
    print(f"wrote {out}")
    print(f"open with:  open {out}")


if __name__ == "__main__":
    main()
