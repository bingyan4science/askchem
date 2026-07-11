"""First-principles scaffold: the curated UPPER TRUNK of the living tree.

The starting point is not empty and not paper-derived. It is a small, curated
descent from fundamental physics toward chemistry, so that mechanisms and
specific observations extracted from papers attach UNDER genuine principles and
theories rather than spawning ad-hoc top-level nodes (which is what produced
"Hydrothermal Synthesis" as a principle in the first grow run).

Graded node kinds (most -> least fundamental):
  open_root : the single unifying principle is not claimed (faint root)
  law       : a fundamental physical law / interaction (conservation, Coulomb)
  framework : an overarching framework (quantum mechanics, thermodynamics, kinetics)
  theory    : an explanatory theory within a framework (bonding theory, TST, Marcus)
  model     : a concrete model under a theory (MO theory, VSEPR, band theory)
  mechanism : an elementary-step motif where paper reactions attach (host=True)
  leaf      : a specific paper-grounded entity (added during growth)

Depth is intentionally NON-uniform: physics -> QM -> bonding theory -> model is
deep, while a conservation law stays shallow. Conceptually this is a DAG (a
mechanism can derive from several principles); we keep a primary parent here and
leave cross-links for later.
"""

from __future__ import annotations

SCAFFOLD = {
    "name": "(unifying principle unknown)",
    "kind": "open_root",
    "desc": "No single principle unifying all of chemistry is claimed; the trunk "
            "descends from fundamental physics toward chemical reactivity.",
    "children": [
        {
            "name": "Conservation laws",
            "kind": "law",
            "desc": "Quantities conserved in every process; constrain stoichiometry, "
                    "energy and charge balance in all reactions.",
            "children": [
                {"name": "Conservation of mass", "kind": "theory",
                 "desc": "Mass/atoms are conserved — the basis of balanced "
                         "stoichiometry.", "children": []},
                {"name": "Conservation of energy", "kind": "theory",
                 "desc": "First law of thermodynamics; reaction energetics balance.",
                 "children": []},
                {"name": "Conservation of charge", "kind": "theory",
                 "desc": "Electron/charge balance — the basis of redox bookkeeping.",
                 "children": []},
            ],
        },
        {
            "name": "Electromagnetic (Coulomb) interaction",
            "kind": "law",
            "desc": "The electrostatic force between charges — the dominant "
                    "interaction governing chemical structure and reactivity.",
            "children": [
                {"name": "Electrostatics & polarization", "kind": "theory",
                 "desc": "Ionic interactions, dipoles, solvation and charge "
                         "polarization.", "children": []},
                {"name": "Intermolecular forces", "kind": "theory",
                 "desc": "Van der Waals, hydrogen bonding and pi-stacking "
                         "non-covalent interactions.",
                 "children": [
                     {"name": "Non-covalent association", "kind": "mechanism",
                      "host": True,
                      "desc": "Host-guest binding, adsorption and self-assembly "
                              "driven by non-covalent forces.", "children": []},
                 ]},
            ],
        },
        {
            "name": "Quantum mechanics",
            "kind": "framework",
            "desc": "The framework governing matter and electrons at the atomic "
                    "scale; the origin of structure, bonding and spectroscopy.",
            "children": [
                {"name": "Wavefunction & Schrodinger equation", "kind": "theory",
                 "desc": "Electron states described by wavefunctions and their "
                         "energies.",
                 "children": [
                     {"name": "Atomic orbitals", "kind": "model",
                      "desc": "One-electron atomic orbital solutions.", "children": []},
                     {"name": "Molecular orbitals", "kind": "model",
                      "desc": "Orbitals delocalized over a molecule.", "children": []},
                 ]},
                {"name": "Pauli exclusion principle", "kind": "theory",
                 "desc": "No two electrons share all quantum numbers; sets electron "
                         "configuration and shell structure.",
                 "children": [
                     {"name": "Electron configuration & Aufbau", "kind": "model",
                      "desc": "Filling of orbitals that underlies periodicity.",
                      "children": []},
                 ]},
                {"name": "Chemical bonding theory", "kind": "theory",
                 "desc": "How atoms share or transfer electrons to form bonds.",
                 "children": [
                     {"name": "Valence bond theory & hybridization", "kind": "model",
                      "desc": "Localized bonds from overlapping hybrid orbitals.",
                      "children": []},
                     {"name": "Molecular orbital theory", "kind": "model",
                      "desc": "Bonding from combination of atomic orbitals into MOs.",
                      "children": []},
                     {"name": "Ligand / crystal field theory", "kind": "model",
                      "desc": "d-orbital splitting in metal complexes.",
                      "children": [
                          {"name": "Metal-ligand coordination", "kind": "mechanism",
                           "host": True,
                           "desc": "Formation of coordination/organometallic complexes "
                                   "via dative metal-ligand bonds.", "children": []},
                      ]},
                     {"name": "Band theory (solids)", "kind": "model",
                      "desc": "Continuous electronic bands in extended solids; "
                              "semiconductors and conductivity.",
                      "children": [
                          {"name": "Semiconductor charge separation", "kind": "mechanism",
                           "host": True,
                           "desc": "Photogenerated electron-hole pairs and interfacial "
                                   "charge transfer (photocatalysis).", "children": []},
                      ]},
                     {"name": "VSEPR (molecular geometry)", "kind": "model",
                      "desc": "Electron-pair repulsion sets molecular shape.",
                      "children": []},
                 ]},
                {"name": "Periodic law", "kind": "theory",
                 "desc": "Periodic trends in electronegativity, size and ionization "
                         "energy across the elements.", "children": []},
            ],
        },
        {
            "name": "Thermodynamics & statistical mechanics",
            "kind": "framework",
            "desc": "Energy, entropy and the statistical behavior of many particles; "
                    "governs spontaneity and equilibrium.",
            "children": [
                {"name": "Gibbs free energy & spontaneity", "kind": "theory",
                 "desc": "Free-energy balance of enthalpy and entropy determines "
                         "reaction direction.", "children": []},
                {"name": "Chemical equilibrium", "kind": "theory",
                 "desc": "Equilibrium constants and Le Chatelier's principle.",
                 "children": [
                     {"name": "Acid-base proton transfer", "kind": "mechanism",
                      "host": True,
                      "desc": "Bronsted-Lowry proton transfer governed by pKa and "
                              "equilibrium.", "children": []},
                 ]},
                {"name": "Phase behavior & transitions", "kind": "theory",
                 "desc": "Phase equilibria, nucleation and crystallization.",
                 "children": [
                     {"name": "Nucleation & crystal growth", "kind": "mechanism",
                      "host": True,
                      "desc": "Solid formation from solution/melt (e.g. solvothermal, "
                              "sol-gel) — a mechanism, not a top-level principle.",
                      "children": []},
                 ]},
            ],
        },
        {
            "name": "Chemical kinetics & dynamics",
            "kind": "framework",
            "desc": "Rates and pathways of chemical change over time.",
            "children": [
                {"name": "Transition state theory", "kind": "theory",
                 "desc": "Reaction rate set by the free energy of the transition "
                         "state / activation barrier.",
                 "children": [
                     {"name": "Concerted pericyclic reorganization", "kind": "mechanism",
                      "host": True,
                      "desc": "Single cyclic transition state governed by orbital "
                              "symmetry (cycloaddition, sigmatropic).", "children": []},
                     {"name": "Polar two-electron bond reorganization", "kind": "mechanism",
                      "host": True,
                      "desc": "Nucleophile-electrophile pairing: substitution, "
                              "addition, elimination.", "children": []},
                 ]},
                {"name": "Catalysis", "kind": "theory",
                 "desc": "Lowering the activation barrier with a regenerated catalyst.",
                 "children": [
                     {"name": "Organometallic catalytic cycle", "kind": "mechanism",
                      "host": True,
                      "desc": "Oxidative addition / transmetalation / reductive "
                              "elimination and migratory insertion at a metal center "
                              "(cross-coupling, Heck).", "children": []},
                 ]},
                {"name": "Marcus theory of electron transfer", "kind": "theory",
                 "desc": "Rates of single-electron transfer from reorganization energy "
                         "and driving force.",
                 "children": [
                     {"name": "Single-electron transfer & radical chemistry",
                      "kind": "mechanism", "host": True,
                      "desc": "Homolytic, open-shell radical pathways; photoredox SET "
                              "and hydrogen-atom transfer.", "children": []},
                 ]},
            ],
        },
    ],
}


def iter_hosts(node, path=None):
    path = (path or []) + [node["name"]]
    if node.get("host"):
        yield path, node
    for c in node.get("children", []):
        yield from iter_hosts(c, path)


def to_d3(node):
    children = [to_d3(c) for c in node.get("children", [])]
    out = {"name": node["name"], "kind": node["kind"], "count": 0}
    if children:
        out["children"] = children
    return out


def stats(node):
    counts = {}

    def walk(n, d):
        counts[n["kind"]] = counts.get(n["kind"], 0) + 1
        for c in n.get("children", []):
            walk(c, d + 1)

    walk(node, 0)
    depths = []

    def maxd(n, d):
        depths.append(d)
        for c in n.get("children", []):
            maxd(c, d + 1)

    maxd(node, 0)
    return counts, max(depths)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    import build_viz

    counts, max_depth = stats(SCAFFOLD)
    n_hosts = len(list(iter_hosts(SCAFFOLD)))
    sub = (f"first-principles scaffold &middot; max depth {max_depth} &middot; "
           f"{n_hosts} mechanism hosts &middot; "
           + ", ".join(f"{k}:{v}" for k, v in counts.items() if k != "open_root"))
    out = _HERE / "output" / "scaffold.html"
    build_viz.render_html(to_d3(SCAFFOLD), "first-principles scaffold", sub, out)
    print(f"kinds: {counts}\nmax depth: {max_depth}, hosts: {n_hosts}")
    print(f"wrote {out}")
