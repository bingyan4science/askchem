"""Multi-view scaffold: shared explanatory trunk + per-view host/leaf layers.

Guiding principle (user's call): accuracy over sharing. The shared trunk
(laws -> frameworks -> theories -> models) is reused by every view because it is
accurate for all of chemistry; each view diverges into its own host layer
attached UNDER the trunk node that most accurately governs it. The structure is
a DAG (a host may accurately sit under several trunk nodes / serve several
views), so hosts carry multiple parents and nodes carry a `views` membership.

Views:
  by_reaction_type  (leaf = reaction)            hosts = reaction mechanisms (from scaffold)
  by_mechanism      (leaf = mechanistic obs.)     hosts = same mechanism layer (accurate overlap)
  by_substance_class(leaf = molecule/material)    hosts = substance classes (LLM, under bonding/IMF/phase)
  by_technique      (leaf = measurement)          hosts = measurement principles (LLM, under QM/thermo/EM)

Outputs:
  output/view_layers/<view>_raw.json   cached LLM host enumerations
  output/view_layers/manifest.json     per-node view membership + cross-links + unattached
  output/scaffold_multiview.html       one HTML with a view selector over the shared trunk

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/view_layers.py            # build all views (cached trunk)
    python3 living_taxonomy/view_layers.py --cache     # reuse cached host layers too
    open living_taxonomy/output/scaffold_multiview.html
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_viz
import placement as pm
import scaffold_builder as sb
from incremental_build import _parse_json

OUT = _HERE / "output"
VL = OUT / "view_layers"
TRUNK_KINDS = {"law", "framework", "theory", "model"}

# Reaction-mechanism families missing from the organic-centric scaffold layer.
# Each attaches under the most accurate existing trunk node or mechanism host.
REACTION_HOST_SUPPLEMENT = [
    {"name": "Heterogeneous catalysis & hydrogenation",
     "parent": "Chemical kinetics & dynamics",
     "desc": "Surface-catalyzed transformations on solid catalysts: "
             "hydrogenation/dehydrogenation, thermal CO2 reduction "
             "(methanation, RWGS, CO2-to-methanol), oxidation."},
    {"name": "Electrocatalytic redox & electrodeposition",
     "parent": "Marcus Theory",
     "desc": "Electron-transfer half-reactions driven at an electrode: CO2/N "
             "reduction (CO2RR, NxRR), HER/OER, electrodeposition, and "
             "electrochemical ion intercalation/extraction."},
    {"name": "Olefin metathesis",
     "parent": "Organometallic Elementary Steps",
     "desc": "Metal-carbene-mediated redistribution of alkene/alkyne fragments "
             "(ring-closing, cross, and ring-opening metathesis)."},
    {"name": "Polymerization & polycondensation",
     "parent": "Covalent bonding theory",
     "desc": "Repeated bond formation building macromolecules: chain- and "
             "step-growth polymerization and condensation network formation "
             "(e.g. covalent organic frameworks)."},
    {"name": "Nucleation, crystal growth & solid-state synthesis",
     "parent": "Thermodynamics & statistical mechanics",
     "desc": "Solid formation from solution, melt or solid state: solvothermal/"
             "hydrothermal crystallization, sol-gel, precipitation, calcination "
             "and ceramic/solid-state synthesis."},
]

# Measurement/characterization principle hosts the thin technique seed lacked.
# Attach under accurate trunk frameworks/theories (never under other hosts).
TECHNIQUE_HOST_SUPPLEMENT = [
    {"name": "Optical & vibrational spectroscopy", "parent": "Quantum mechanics",
     "desc": "UV-Vis, IR, Raman and fluorescence - light-matter electronic and "
             "vibrational transitions."},
    {"name": "Magnetic resonance spectroscopy", "parent": "Quantum mechanics",
     "desc": "NMR and EPR - nuclear/electron spin transitions in a magnetic field."},
    {"name": "X-ray & photoelectron spectroscopy", "parent": "Quantum mechanics",
     "desc": "XPS, XAS, Auger - core-level electronic spectroscopy of composition "
             "and oxidation state."},
    {"name": "Diffraction & scattering", "parent": "Quantum mechanics",
     "desc": "X-ray/electron/neutron diffraction and small-angle scattering - "
             "structure from wave interference."},
    {"name": "Microscopy & imaging", "parent": "Electromagnetic (Coulomb) interaction",
     "desc": "SEM, TEM, AFM, STM - real-space imaging of morphology and surfaces."},
    {"name": "Mass spectrometry", "parent": "Electromagnetic (Coulomb) interaction",
     "desc": "Ionization and mass/charge separation for molecular mass and "
             "composition."},
    {"name": "Chromatographic & phase separation",
     "parent": "Thermodynamics & statistical mechanics",
     "desc": "GC, HPLC, electrophoresis - separation by partitioning or mobility."},
    {"name": "Electrochemical & thermal analysis",
     "parent": "Chemical kinetics & dynamics",
     "desc": "Voltammetry, impedance, TGA, DSC and calorimetry."},
    {"name": "Computational & simulation methods", "parent": "Quantum mechanics",
     "desc": "DFT, molecular dynamics/simulation, force-field and ML-potential "
             "modeling, QM/MM - in silico characterization and parameter learning."},
]

VIEW_SPECS = {
    "by_substance_class": {
        "leaf": "molecule or material", "host_kind": "class",
        "noun": "substance classes (kinds of molecules and materials)",
        "examples": "transition-metal complexes, conjugated organic molecules, "
                    "synthetic polymers, semiconductors, metal nanoparticles, "
                    "metal-organic frameworks, ionic solids",
    },
    "by_technique": {
        "leaf": "measurement or characterization result", "host_kind": "mechanism",
        "noun": "measurement and characterization principles",
        "examples": "absorption spectroscopy, NMR spectroscopy, mass spectrometry, "
                    "X-ray diffraction, voltammetry, chromatographic separation, "
                    "calorimetry",
    },
}


# ── trunk inventory + parent resolution ──────────────────────────────────────

def trunk_inventory(nodes, children, roots, anchor_norm):
    """Indented text of the shared trunk (law/framework/theory/model only)."""
    lines = []

    def walk(nm, depth):
        n = nodes[nm]
        if n["kind"] not in TRUNK_KINDS and nm not in anchor_norm:
            return
        lines.append("  " * depth + f"- {n['name']} [{n['kind']}]")
        for c in children.get(nm, []):
            if nodes[c]["kind"] in TRUNK_KINDS:
                walk(c, depth + 1)

    for a in roots:
        if a in anchor_norm:
            walk(a, 0)
    return "\n".join(lines)


def resolve_trunk(name, nodes, anchor_norms):
    """Resolve a host's stated parent NAME to an existing trunk node norm."""
    nm = sb._norm(name)
    if nm in nodes and nodes[nm]["kind"] in TRUNK_KINDS:
        return nm
    ptoks = set(nm.split())
    best, bo = None, 0
    for k, n in nodes.items():
        if n["kind"] not in TRUNK_KINDS:
            continue
        ov = len(ptoks & set(k.split()))
        if ov >= 2 and ov > bo:
            best, bo = k, ov
    if best:
        return best
    return sb._anchor_for(name, anchor_norms)


# ── per-view host enumeration (LLM) ──────────────────────────────────────────

_SYS = ("You extend a shared first-principles chemistry scaffold with the host "
        "categories for ONE view. You attach each host under the EXACT existing "
        "scaffold node that most accurately governs it. Accuracy matters more "
        "than sharing; if a host genuinely belongs under several scaffold nodes, "
        "list them all. Never output specific compounds, reactions or instruments "
        "(those are leaves).")


def _user(view_id, spec, inventory):
    return f"""SHARED SCAFFOLD NODES (attach hosts under these EXACT names):
{inventory}

VIEW: {view_id} — organize chemistry by {spec['noun']}; each leaf will be a
{spec['leaf']}. List the canonical HOST categories for this view (the level just
above individual leaves). Each host:
{{"name": "...",
  "kind": "{spec['host_kind']}",
  "parents": ["exact scaffold node name(s) that accurately govern this host"],
  "definition": "one concise sentence"}}

Examples of hosts: {spec['examples']}.
Rules:
- parents MUST be exact names from the scaffold list; choose the most fundamental
  accurate parent(s); include several only if genuinely accurate.
- ~15-30 hosts. Do NOT include specific compounds, reactions, or instruments.
Return ONLY JSON: {{"hosts": [ ... ]}}"""


def build_view_layer(view_id, inventory, use_cache):
    cache = VL / f"{view_id}_raw.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    spec = VIEW_SPECS[view_id]
    try:
        resp = pm._gemini_chat(_SYS, _user(view_id, spec, inventory), max_time=120)
        hosts = _parse_json(resp).get("hosts", [])
    except Exception as e:
        print(f"[views] {view_id}: ERROR {e}", file=sys.stderr)
        hosts = []
    VL.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(hosts, indent=2))
    print(f"[views] {view_id}: {len(hosts)} hosts", file=sys.stderr)
    return hosts


# ── assemble a per-view tree (shared trunk + hosts, pruned to host ancestors) ─

def build_view_tree(nodes, children, roots, anchor_norm, anchor_norms, hosts):
    """Return (d3_tree, unattached, xrefs). Trunk shared=True, hosts shared=False."""
    vn = {}
    for nm, n in nodes.items():
        if n["kind"] in TRUNK_KINDS:
            vn[nm] = {"name": n["name"], "kind": n["kind"], "count": 0,
                      "shared": True, "children": []}
    # trunk primary-parent links
    for nm, n in nodes.items():
        if n["kind"] in TRUNK_KINDS:
            p = n.get("_parent")
            if p in vn and p != nm:
                vn[p]["children"].append(vn[nm])
    # host nodes
    host_norm = {}
    for h in hosts:
        hn = sb._norm(h["name"])
        if not hn or hn in vn or hn in host_norm:
            continue
        host_norm[hn] = {"name": h["name"], "kind": h.get("kind", "mechanism"),
                         "count": 0, "shared": False, "children": [],
                         "parent_norms": h.get("parent_norms", [])}
        vn[hn] = host_norm[hn]
    # host links (primary parent + cross-links)
    unattached, xrefs = [], []
    for hn, hnode in host_norm.items():
        pn = [p for p in hnode["parent_norms"] if p in vn]
        if not pn:
            unattached.append(hnode["name"])
            continue
        vn[pn[0]]["children"].append(hnode)
        for extra in pn[1:]:
            xrefs.append({"host": hnode["name"], "also_under": vn[extra]["name"]})
        hnode.pop("parent_norms", None)
    top = {"name": "(unifying principle unknown)", "kind": "open_root",
           "count": 0, "shared": True,
           "children": [vn[a] for a in roots if a in anchor_norm and a in vn]}
    _prune(top)
    host_by_name = {n["name"]: n for n in host_norm.values()}
    return top, unattached, xrefs, host_by_name


def _prune(node):
    """Keep a node iff it is a host, the open root, or has a kept descendant."""
    node["children"] = [c for c in node.get("children", []) if _keep(c)]
    for c in node["children"]:
        _prune(c)


def _keep(node):
    node["children"] = [c for c in node.get("children", []) if _keep(c)]
    return (not node.get("shared")) or node["kind"] == "open_root" \
        or len(node["children"]) > 0


def _count(node):
    n = 0 if node.get("children") else (1 if node["kind"] in ("leaf",) else 0)
    for c in node.get("children", []):
        n += _count(c)
    node["count"] = sum(1 for _ in _hosts(node))
    return node["count"]


def _hosts(node):
    if node.get("shared") is False:
        yield node
    for c in node.get("children", []):
        yield from _hosts(c)


# ── audit: near-duplicate detection + optional missing-concept check ─────────

def audit(nodes, use_llm):
    names = [(nm, n["name"]) for nm, n in nodes.items()]
    dups = []
    for i in range(len(names)):
        ti = set(names[i][0].split())
        for j in range(i + 1, len(names)):
            tj = set(names[j][0].split())
            if not ti or not tj:
                continue
            jac = len(ti & tj) / len(ti | tj)
            if jac >= 0.6:
                dups.append([names[i][1], names[j][1], round(jac, 2)])
    missing = []
    if use_llm:
        inv = ", ".join(sorted({n["name"] for n in nodes.values()
                                if n["kind"] in ("law", "framework", "theory")}))
        try:
            resp = pm._gemini_chat(
                "You audit a chemistry concept scaffold for completeness.",
                "Current laws/frameworks/theories:\n" + inv +
                "\n\nList up to 15 MAJOR chemistry laws or theories that are "
                "missing from this list. Return ONLY JSON {\"missing\":[\"...\"]}.",
                max_time=90)
            missing = _parse_json(resp).get("missing", [])
        except Exception as e:
            missing = [f"(audit LLM error: {e})"]
    report = {"near_duplicates": dups, "missing_major_concepts": missing}
    (OUT / "scaffold_audit.json").write_text(json.dumps(report, indent=2))
    return report


# ── main ─────────────────────────────────────────────────────────────────────

def build_all_views(use_cache=True):
    """Build every view tree over the shared trunk.

    Returns (nodes, views, host_nodes_by_view, manifest). ``host_nodes_by_view``
    maps view_id -> {host_name: d3_node} so callers can attach leaves.
    """
    VL.mkdir(parents=True, exist_ok=True)
    raw = json.loads((OUT / "scaffold_raw.json").read_text())
    nodes, children, roots, anchor_norm = sb.merge(raw)
    anchor_norms = [sb._norm(n) for n, _ in sb.ANCHORS]
    inventory = trunk_inventory(nodes, children, roots, anchor_norm)

    # reaction / mechanism hosts = scaffold mechanism nodes (no LLM)
    rxn_hosts = [{"name": n["name"], "kind": "mechanism", "desc": n["desc"],
                  "parent_norms": [n["_parent"]] if n.get("_parent") else []}
                 for nm, n in nodes.items() if n["kind"] == "mechanism"]

    # curated supplement: reaction-mechanism families the organic-centric
    # scaffold layer was missing (catalysis / electrochem / materials).
    for h in REACTION_HOST_SUPPLEMENT:
        pn = sb._norm(h["parent"])
        if pn not in nodes:                 # parent may be trunk or a mechanism host
            pn = resolve_trunk(h["parent"], nodes, anchor_norms)
        rxn_hosts.append({"name": h["name"], "kind": "mechanism",
                          "desc": h["desc"], "parent_norms": [pn] if pn else []})

    views, host_nodes_by_view = {}, {}
    manifest = {"views": {}, "cross_links": {}, "unattached": {}}

    def add_view(view_id, hosts):
        top, unattached, xrefs, host_nodes = build_view_tree(
            nodes, children, roots, anchor_norm, anchor_norms, hosts)
        _count(top)
        views[view_id] = top
        host_nodes_by_view[view_id] = host_nodes
        manifest["cross_links"][view_id] = xrefs
        manifest["unattached"][view_id] = unattached
        print(f"[views] {view_id}: {sum(1 for _ in _hosts(top))} hosts attached, "
              f"{len(unattached)} unattached, {len(xrefs)} cross-links",
              file=sys.stderr)

    add_view("by_reaction_type", rxn_hosts)
    add_view("by_mechanism", rxn_hosts)
    for vid in VIEW_SPECS:
        hosts = build_view_layer(vid, inventory, use_cache)
        for h in hosts:                                 # resolve parents -> trunk
            h["parent_norms"] = []
            for p in h.get("parents", []):
                r = resolve_trunk(p, nodes, anchor_norms)
                if r and r not in h["parent_norms"]:
                    h["parent_norms"].append(r)
        if vid == "by_technique":                       # curated supplement hosts
            for s in TECHNIQUE_HOST_SUPPLEMENT:
                pn = sb._norm(s["parent"])
                if pn not in nodes:
                    pn = resolve_trunk(s["parent"], nodes, anchor_norms)
                hosts.append({"name": s["name"], "kind": "mechanism",
                              "desc": s["desc"], "parents": [s["parent"]],
                              "parent_norms": [pn] if pn else []})
        add_view(vid, hosts)

    membership = defaultdict(set)
    for nm, n in nodes.items():
        if n["kind"] in TRUNK_KINDS:
            membership[n["name"]] = set(views.keys())
    for vid, top in views.items():
        for h in _hosts(top):
            membership[h["name"]].add(vid)
    manifest["views"] = {k: sorted(v) for k, v in membership.items()}
    (VL / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return nodes, views, host_nodes_by_view, manifest


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true",
                    help="reuse cached per-view host layers (no LLM)")
    ap.add_argument("--audit-llm", action="store_true",
                    help="also ask the LLM for missing major concepts")
    args = ap.parse_args()

    nodes, views, _host_nodes, manifest = build_all_views(use_cache=args.cache)
    rep = audit(nodes, args.audit_llm)
    sub = ("shared trunk + per-view host layers &middot; "
           f"{len(views)} views &middot; "
           f"{len(rep['near_duplicates'])} near-dup pairs flagged")
    build_viz.render_html(views["by_reaction_type"], "chemistry living scaffold",
                          sub, OUT / "scaffold_multiview.html", views=views)
    print("\n=== MULTI-VIEW SUMMARY ===")
    for vid, top in views.items():
        print(f"  {vid:20s}: {sum(1 for _ in _hosts(top))} hosts")
    print(f"  near-duplicate pairs: {len(rep['near_duplicates'])}")
    if rep["missing_major_concepts"]:
        print(f"  missing (LLM): {rep['missing_major_concepts'][:8]}")
    print(f"[views] wrote {OUT/'scaffold_multiview.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
