"""
Normalize the by_mechanism taxonomy across all claims.

10 unified L1 categories organized by type of mechanistic process:
  1. bond_formation_and_breaking
  2. electron_transfer
  3. photophysics_and_excited_states
  4. catalytic_cycles
  5. adsorption_and_surface
  6. transport_and_diffusion
  7. self_assembly_and_phase
  8. molecular_recognition
  9. conformational_and_structural
 10. degradation_and_stability

Usage:
    python src/normalize_mechanism.py              # dry-run
    python src/normalize_mechanism.py --apply      # apply changes
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from askchem.display import smart_title

DB_PATH = Path(__file__).parent.parent / "chemtree.db"
VIEW_ID = "by_mechanism"

CANONICAL_L1 = [
    "bond_formation_and_breaking",
    "electron_transfer",
    "photophysics_and_excited_states",
    "catalytic_cycles",
    "adsorption_and_surface",
    "transport_and_diffusion",
    "self_assembly_and_phase",
    "molecular_recognition",
    "conformational_and_structural",
    "degradation_and_stability",
]

CANONICAL_SET = set(CANONICAL_L1)

# ── Direct L1 remapping ─────────────────────────────────────────────────────

DIRECT_L1_MAP = {
    # bond_formation_and_breaking
    "reaction_mechanisms_and_kinetics": "bond_formation_and_breaking",
    "bond_activation": "bond_formation_and_breaking",
    "radical_mechanism": "bond_formation_and_breaking",
    "reaction_kinetics": "bond_formation_and_breaking",
    "redox_reactions": "bond_formation_and_breaking",
    "acid_base_equilibrium": None,  # route by L2

    # electron_transfer
    "electrochemistry_and_charge_transfer": "electron_transfer",
    "electron_transfer": "electron_transfer",
    "electronic_structure_and_quantum_chemistry": "electron_transfer",
    "electronic_structure": "electron_transfer",

    # photophysics_and_excited_states
    "photophysics_and_excited_state_processes": "photophysics_and_excited_states",
    "photophysics": "photophysics_and_excited_states",
    "energy_transfer": "photophysics_and_excited_states",
    "aggregation_induced_emission": "photophysics_and_excited_states",

    # catalytic_cycles
    "catalysis_and_catalytic_mechanisms": "catalytic_cycles",
    "catalytic_cycle": "catalytic_cycles",
    "catalytic_cycles": "catalytic_cycles",

    # adsorption_and_surface
    "heterogeneous_and_surface_processes": "adsorption_and_surface",
    "adsorption_desorption": "adsorption_and_surface",

    # transport_and_diffusion
    "interfacial_and_transport_processes": "transport_and_diffusion",
    "diffusion": "transport_and_diffusion",
    "charge_transport": "transport_and_diffusion",

    # self_assembly_and_phase
    "self_assembly_nucleation_and_crystallization": "self_assembly_and_phase",
    "nucleation_growth": "self_assembly_and_phase",
    "phase_transition": "self_assembly_and_phase",
    "self_assembly_mechanisms": "self_assembly_and_phase",
    "assembly_mechanisms": "self_assembly_and_phase",
    "hydrolysis_and_condensation": "self_assembly_and_phase",

    # molecular_recognition
    "molecular_and_noncovalent_interactions": "molecular_recognition",
    "molecular_recognition": "molecular_recognition",
    "coordination_competition": "molecular_recognition",

    # conformational_and_structural
    "materials_structure_and_lattice_dynamics": "conformational_and_structural",

    # degradation_and_stability
    "stability_and_degradation_and_decomposition": "degradation_and_stability",

    # spectroscopy is a technique, not a mechanism — route by L2
    "spectroscopy_and_vibrational_dynamics": None,
    "thermodynamics_and_phase_behavior": None,
    "thermodynamics": None,
    "biological_and_enzymatic_mechanisms": None,

    # Junk / non-mechanism
    "not_applicable": None,
    "other": None,
    "by_mechanism": None,
    "predictive_models": None,
    "computational_and_electronic_structure": None,
    "structural_biology_and_bioassays": None,
}

# ── L2-keyword routing ──────────────────────────────────────────────────────
# For L1s mapped to None above, we route by L2 keywords.

BOND_KW = ["substitut", "eliminat", "addition", "pericyclic", "cycloaddit",
           "radical", "c_h_activ", "bond_activ", "bond_form", "bond_break",
           "metathesis", "rearrangement", "ring_open", "ring_clos",
           "nucleophil", "electrophil", "sn1", "sn2", "retro",
           "coupling", "cross_coupling", "insertion", "migrat",
           "polymeriz", "combustion", "oxidat", "reduct",
           "hydrolysis", "condensat", "acylat", "alkylat",
           "halogenat", "dehydrat", "decarboxylat", "aminat",
           "stereochem", "stereocontrol", "enantioselect",
           "reaction_mech", "reaction_kinet", "rate_law",
           "acid_base_cataly", "protonation", "deprotonation"]

ELECTRON_KW = ["electron_transfer", "charge_transfer", "redox",
               "marcus", "electrochemical", "electrode", "overpotential",
               "voltamm", "impedance", "faradaic", "galvanic",
               "band_structure", "band_gap", "orbital", "electronic_structure",
               "spin_orbit", "density_functional", "quantum_chem",
               "many_body", "hartree", "dft", "coupled_cluster",
               "magnetism", "spin_state"]

PHOTO_KW = ["photophys", "excited_state", "fluorescen", "phosphorescen",
            "luminescen", "fret", "forster", "dexter", "exciton",
            "intersystem", "internal_conversion", "radiative",
            "non_radiative", "nonradiative", "plasmon", "absorption_emission",
            "photoluminescen", "emission_spectr", "quantum_yield",
            "singlet", "triplet", "chromophore", "photosensit",
            "carrier_dynamic", "charge_recombinat"]

CATALYTIC_KW = ["catalytic_cycle", "catalytic_mechanism", "turnover",
                "active_site", "catalyst_design", "catalyst_select",
                "organocataly", "metal_cataly", "enzyme_cataly",
                "biocataly", "palladium_cataly", "transition_metal_cataly",
                "lewis_acid", "bronsted", "acid_cataly",
                "photocataly", "electrocataly"]

SURFACE_KW = ["adsorpt", "desorpt", "chemisorpt", "physisorpt",
              "surface_react", "surface_bind", "langmuir", "eley_rideal",
              "heterogeneous", "surface_activ", "surface_process"]

TRANSPORT_KW = ["transport", "diffusion", "permeation", "conductiv",
                "ion_migrat", "mass_transfer", "convect",
                "charge_carrier", "ionic_conduct", "proton_conduct",
                "membrane_transport"]

ASSEMBLY_KW = ["nucleation", "crystalliz", "self_assembl", "aggregat",
               "micelle", "vesicle", "phase_transit", "glass_transit",
               "polymorph", "precipitat", "gelat", "fibrillat",
               "supramolecular_assembl", "colloid"]

RECOGNITION_KW = ["molecular_recognit", "host_guest", "binding",
                  "noncovalent", "non_covalent", "hydrogen_bond",
                  "supramolecular", "receptor", "ligand_bind",
                  "protein_protein", "protein_ligand", "nucleic_acid",
                  "aptamer", "antibody_antigen", "lock_and_key",
                  "ph_control", "buffer"]

CONFORM_KW = ["conformational", "folding", "unfolding", "lattice_dynamic",
              "structural_rearrang", "deformat", "strain",
              "crystal_structure", "polymorphic_transform",
              "materials_structure", "elastic", "mechanical"]

DEGRAD_KW = ["degradat", "decomposit", "corrosion", "stability",
             "photodegrad", "thermal_decomp", "hydrolytic_degrad",
             "oxidative_degrad", "aging", "weathering", "erosion"]

# For spectroscopy_and_vibrational_dynamics — route vibrational dynamics
# to conformational_and_structural, spectroscopic technique claims get dropped
SPECTRO_MECH_KW = ["vibrational_dynamic", "vibrational_relaxat",
                   "vibrational_coupling", "phonon", "lattice_vibrat",
                   "ir_absorpt", "raman_scatter"]

# For thermodynamics — route phase behavior to self_assembly_and_phase,
# thermochemistry to bond_formation, statistical mech to conformational
THERMO_PHASE_KW = ["phase_behav", "phase_equilibr", "phase_diagram",
                   "phase_separat", "critical_behav", "equation_of_state"]
THERMO_CHEM_KW = ["thermochem", "enthalpy", "heat_of_react", "free_energy",
                  "entropy", "gibbs", "activation_energy", "arrhenius"]
THERMO_STAT_KW = ["statistical_thermo", "partition_function", "boltzmann",
                  "heat_transfer", "thermal_conduct"]
THERMO_BIO_KW = ["biomolecular_thermo", "protein_thermo", "binding_thermo"]

# For biological_and_enzymatic_mechanisms — split into relevant categories
BIO_ENZYME_KW = ["enzyme", "enzymatic", "michaelis", "km_value",
                 "turnover_number", "active_site", "allosteric",
                 "biocataly", "substrate_specific"]
BIO_SIGNAL_KW = ["signal_transduct", "receptor", "kinase", "phosphorylat",
                 "gene_regulat", "transcription", "translation",
                 "metabolic_pathway", "biosynthesis", "cellular_signal"]
BIO_FOLD_KW = ["protein_fold", "protein_unfold", "misfolding",
               "chaperone", "amyloid", "fibril", "denatur",
               "conformational_change", "structural_transition"]
BIO_BIND_KW = ["protein_ligand", "protein_protein", "receptor_bind",
               "antigen", "antibody", "aptamer", "nucleic_acid_bind",
               "dna_bind", "rna_bind", "molecular_recognit",
               "host_guest", "supramolecular"]
BIO_MEMBRANE_KW = ["membrane_transport", "ion_channel", "pump",
                   "endocytosis", "exocytosis", "vesicle_traffic",
                   "membrane_fusion", "permeabil"]
BIO_REDOX_KW = ["electron_transfer", "redox", "oxidoreduct",
                "cytochrome", "nadh", "fadh", "respiratory_chain",
                "photosynthetic_electron"]

L2_ROUTING_RULES = [
    (BOND_KW, "bond_formation_and_breaking"),
    (ELECTRON_KW, "electron_transfer"),
    (PHOTO_KW, "photophysics_and_excited_states"),
    (CATALYTIC_KW, "catalytic_cycles"),
    (SURFACE_KW, "adsorption_and_surface"),
    (TRANSPORT_KW, "transport_and_diffusion"),
    (ASSEMBLY_KW, "self_assembly_and_phase"),
    (RECOGNITION_KW, "molecular_recognition"),
    (CONFORM_KW, "conformational_and_structural"),
    (DEGRAD_KW, "degradation_and_stability"),
]

L2_ROUTED_L1S = {
    "acid_base_equilibrium",
    "spectroscopy_and_vibrational_dynamics",
    "thermodynamics_and_phase_behavior",
    "thermodynamics",
    "biological_and_enzymatic_mechanisms",
    "other",
    "not_applicable",
    "predictive_models",
    "computational_and_electronic_structure",
    "structural_biology_and_bioassays",
    "by_mechanism",
}


def _normalize_seg(seg: str) -> str:
    return seg.strip().lower().replace('-', '_').replace(' ', '_')


def _route_by_l2(old_l1: str, l2: str) -> str | None:
    l2_lower = l2.lower()

    if old_l1 == "biological_and_enzymatic_mechanisms":
        for kw in BIO_ENZYME_KW:
            if kw in l2_lower:
                return "catalytic_cycles"
        for kw in BIO_FOLD_KW:
            if kw in l2_lower:
                return "conformational_and_structural"
        for kw in BIO_BIND_KW:
            if kw in l2_lower:
                return "molecular_recognition"
        for kw in BIO_MEMBRANE_KW:
            if kw in l2_lower:
                return "transport_and_diffusion"
        for kw in BIO_REDOX_KW:
            if kw in l2_lower:
                return "electron_transfer"
        for kw in BIO_SIGNAL_KW:
            if kw in l2_lower:
                return "bond_formation_and_breaking"
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return "molecular_recognition"  # biological mechanisms default

    if old_l1 == "spectroscopy_and_vibrational_dynamics":
        for kw in SPECTRO_MECH_KW:
            if kw in l2_lower:
                return "conformational_and_structural"
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return None  # pure spectroscopy technique, drop

    if old_l1 in ("thermodynamics_and_phase_behavior", "thermodynamics"):
        for kw in THERMO_PHASE_KW:
            if kw in l2_lower:
                return "self_assembly_and_phase"
        for kw in THERMO_CHEM_KW:
            if kw in l2_lower:
                return "bond_formation_and_breaking"
        for kw in THERMO_STAT_KW:
            if kw in l2_lower:
                return "conformational_and_structural"
        for kw in THERMO_BIO_KW:
            if kw in l2_lower:
                return "molecular_recognition"
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return "bond_formation_and_breaking"  # thermodynamics default

    if old_l1 == "acid_base_equilibrium":
        for kw in ["acid_base_cataly"]:
            if kw in l2_lower:
                return "catalytic_cycles"
        for kw in ["protonation", "deprotonation"]:
            if kw in l2_lower:
                return "bond_formation_and_breaking"
        for kw in ["ph_control", "buffer"]:
            if kw in l2_lower:
                return "molecular_recognition"
        return "bond_formation_and_breaking"

    # Generic fallback for other, not_applicable, etc.
    for keywords, target in L2_ROUTING_RULES:
        for kw in keywords:
            if kw in l2_lower:
                return target
    return None


def remap_claim_path(old_path: list) -> list | None:
    if not old_path or not isinstance(old_path, list):
        return None

    old_l1 = _normalize_seg(old_path[0])

    if old_l1 in ('not_applicable', 'none', ''):
        return None

    if old_l1 in DIRECT_L1_MAP:
        new_l1 = DIRECT_L1_MAP[old_l1]
        if new_l1 is None and old_l1 not in L2_ROUTED_L1S:
            return None
        if new_l1 is not None:
            new_path = [new_l1] + [_normalize_seg(s) for s in old_path[1:]]
            return [s for s in new_path if s not in ('not_applicable', 'none')]

    if old_l1 in L2_ROUTED_L1S:
        l2 = _normalize_seg(old_path[1]) if len(old_path) >= 2 else ""
        new_l1 = _route_by_l2(old_l1, l2)
        if new_l1 is None:
            return None
        new_path = [new_l1] + [_normalize_seg(s) for s in old_path[1:]]
        return [s for s in new_path if s not in ('not_applicable', 'none')]

    if old_l1 in CANONICAL_SET:
        new_path = [_normalize_seg(s) for s in old_path]
        return [s for s in new_path if s not in ('not_applicable', 'none')]

    return None


# ── L2 fuzzy clustering ─────────────────────────────────────────────────────

NOISE_WORDS = {'and', 'or', 'the', 'of', 'in', 'for', 'with', 'based', 'type',
               'general', 'various', 'related', 'other', 'like', 'class'}


def _tokenize(slug: str) -> set[str]:
    return set(slug.replace('_', ' ').split())


def _token_similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_subset_name(a: str, b: str) -> bool:
    ta, tb = _tokenize(a), _tokenize(b)
    return ta <= tb or tb <= ta


def _normalize_slug(slug: str) -> str:
    tokens = slug.replace('_', ' ').split()
    tokens = [t for t in tokens if t not in NOISE_WORDS]
    return '_'.join(tokens)


def build_l2_merge_map(l2_counts: dict[str, Counter]) -> dict[str, dict[str, str]]:
    merge_map = {}
    total_merges = 0
    l1_list = list(l2_counts.items())

    for li, (l1, slugs) in enumerate(l1_list):
        if len(slugs) <= 1:
            continue

        sorted_slugs = sorted(slugs.items(), key=lambda x: -x[1])
        canonical_map = {}
        norm_index: dict[str, str] = {}
        singular_index: dict[str, str] = {}
        top_clusters: list[str] = []
        MAX_SIM_CANDIDATES = 500

        for slug, count in sorted_slugs:
            norm = _normalize_slug(slug)
            singular = norm.rstrip('s')

            if norm in norm_index:
                canonical_map[slug] = norm_index[norm]
                continue

            if singular in singular_index:
                canon = singular_index[singular]
                canon_norm = _normalize_slug(canon)
                if abs(len(norm) - len(canon_norm)) <= 1:
                    canonical_map[slug] = canon
                    norm_index[norm] = canon
                    continue

            merged = False
            for canonical in top_clusters[:MAX_SIM_CANDIDATES]:
                sim = _token_similarity(slug, canonical)
                if sim >= 0.7 and _is_subset_name(slug, canonical):
                    canonical_map[slug] = canonical
                    norm_index[norm] = canonical
                    singular_index[singular] = canonical
                    merged = True
                    break
                if sim >= 0.85:
                    canonical_map[slug] = canonical
                    norm_index[norm] = canonical
                    singular_index[singular] = canonical
                    merged = True
                    break

            if not merged:
                canonical_map[slug] = slug
                norm_index[norm] = slug
                singular_index[singular] = slug
                top_clusters.append(slug)

        actual = {k: v for k, v in canonical_map.items() if k != v}
        if actual:
            merge_map[l1] = actual
            total_merges += len(actual)

        if (li + 1) % 3 == 0 or li == len(l1_list) - 1:
            print(f"    Clustered {li+1}/{len(l1_list)} L1s "
                  f"({total_merges:,} merges so far)", flush=True)

    return merge_map


# ── Tree rebuild ─────────────────────────────────────────────────────────────

def rebuild_tree_nodes(conn, all_paths: list[tuple[str, list]]):
    c = conn.cursor()
    c.execute("DELETE FROM tree_nodes WHERE view_id = ?", (VIEW_ID,))

    node_data: dict[str, dict] = {}
    for claim_id, path in all_paths:
        for depth in range(len(path)):
            prefix = '/'.join(path[:depth + 1])
            if prefix not in node_data:
                node_data[prefix] = {'claim_count': 0, 'claim_ids': [], 'children': set()}
            if depth == len(path) - 1:
                node_data[prefix]['claim_count'] += 1
                if len(node_data[prefix]['claim_ids']) < 100:
                    node_data[prefix]['claim_ids'].append(claim_id)
            if depth > 0:
                parent = '/'.join(path[:depth])
                if parent in node_data:
                    node_data[parent]['children'].add(path[depth])

    root_children = set()
    total_claims = 0
    for claim_id, path in all_paths:
        if path:
            root_children.add(path[0])
            total_claims += 1

    sorted_paths = sorted(node_data.keys(), key=lambda p: -p.count('/'))
    for path_str in sorted_paths:
        parts = path_str.split('/')
        if len(parts) > 1:
            parent = '/'.join(parts[:-1])
            if parent in node_data:
                node_data[parent]['claim_count'] += node_data[path_str]['claim_count']

    batch = []
    for path_str, nd in node_data.items():
        level = path_str.count('/') + 1
        name = smart_title(path_str.split('/')[-1])
        children_list = sorted(nd['children'])
        data_json = {
            'view_id': VIEW_ID, 'path': path_str, 'name': name, 'level': level,
            'claim_count': nd['claim_count'], 'children': children_list,
            'claim_ids': nd['claim_ids'],
        }
        batch.append((
            VIEW_ID, path_str, name, level, nd['claim_count'],
            json.dumps(children_list), json.dumps(nd['claim_ids']), json.dumps(data_json),
        ))

    root_data = {
        'view_id': VIEW_ID, 'path': '', 'name': 'By Mechanism', 'level': 0,
        'claim_count': total_claims, 'children': sorted(root_children), 'claim_ids': [],
    }
    batch.append((
        VIEW_ID, '', 'By Mechanism', 0, total_claims,
        json.dumps(sorted(root_children)), json.dumps([]), json.dumps(root_data),
    ))

    chunk_size = 10000
    for i in range(0, len(batch), chunk_size):
        c.executemany(
            "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            batch[i:i + chunk_size]
        )
    conn.commit()
    return len(batch)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Normalize by_mechanism taxonomy")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.1f} GB)")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN (use --apply to write changes)'}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print("Phase 1: Reading claims and remapping L1...", flush=True)
    t0 = time.time()

    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"  Total claims: {total:,}")

    old_l1_counts = Counter()
    new_l1_counts = Counter()
    dropped_count = 0
    kept_count = 0
    no_path_count = 0
    updates = []
    l2_counts_per_l1 = defaultdict(Counter)

    batch_size = 50000
    offset = 0
    while offset < total:
        rows = conn.execute(
            "SELECT claim_id, data FROM claims LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break

        for row in rows:
            claim_id = row[0]
            data = json.loads(row[1])
            vp = data.get('view_paths', {})
            old_path = vp.get(VIEW_ID, [])

            if not old_path or not isinstance(old_path, list) or len(old_path) < 1:
                no_path_count += 1
                updates.append((claim_id, data, None))
                continue

            old_l1_counts[old_path[0]] += 1
            new_path = remap_claim_path(old_path)

            if new_path and len(new_path) >= 1:
                new_l1_counts[new_path[0]] += 1
                kept_count += 1
                if len(new_path) >= 2:
                    l2_counts_per_l1[new_path[0]][new_path[1]] += 1
            else:
                dropped_count += 1
                new_path = None

            if new_path is not None:
                vp[VIEW_ID] = new_path
            elif VIEW_ID in vp:
                del vp[VIEW_ID]
            data['view_paths'] = vp
            updates.append((claim_id, data, new_path))

        offset += batch_size
        elapsed = time.time() - t0
        print(f"  Processed {offset:,}/{total:,} ({elapsed:.0f}s)", flush=True)

    print(f"\n  L1 remapping complete: {kept_count:,} kept, {dropped_count:,} dropped, "
          f"{no_path_count:,} had no path")
    print(f"\n  Old L1 distribution:")
    for l1, c in old_l1_counts.most_common():
        print(f"    {l1:55s} {c:>8,}")
    print(f"\n  New L1 distribution:")
    for l1, c in new_l1_counts.most_common():
        print(f"    {l1:55s} {c:>8,}")

    print("\nPhase 2: L2 fuzzy clustering...", flush=True)
    t1 = time.time()

    total_l2_before = sum(len(slugs) for slugs in l2_counts_per_l1.values())
    print(f"  Total unique L2 slugs before clustering: {total_l2_before:,}")

    l2_merge = build_l2_merge_map(l2_counts_per_l1)
    total_merges = sum(len(m) for m in l2_merge.values())
    print(f"  L2 merges: {total_merges:,}")

    for l1 in CANONICAL_L1:
        if l1 in l2_merge:
            print(f"    {l1}: {len(l2_merge[l1]):,} merges")

    l2_applied = 0
    for i, (claim_id, data, path) in enumerate(updates):
        if path and len(path) >= 2:
            l1 = path[0]
            l2 = path[1]
            if l1 in l2_merge and l2 in l2_merge[l1]:
                path[1] = l2_merge[l1][l2]
                data['view_paths'][VIEW_ID] = path
                updates[i] = (claim_id, data, path)
                l2_applied += 1

    print(f"  L2 merges applied to {l2_applied:,} claims")

    final_l2_counts = defaultdict(Counter)
    for claim_id, data, path in updates:
        if path and len(path) >= 2:
            final_l2_counts[path[0]][path[1]] += 1
    total_l2_after = sum(len(slugs) for slugs in final_l2_counts.values())
    print(f"  Total unique L2 slugs after clustering: {total_l2_after:,}")
    print(f"  Reduction: {total_l2_before:,} -> {total_l2_after:,} "
          f"({100*(1-total_l2_after/max(total_l2_before,1)):.1f}%)")

    elapsed2 = time.time() - t1
    print(f"  Phase 2 done in {elapsed2:.0f}s")

    if not args.apply:
        print("\n  DRY-RUN complete. Use --apply to write changes.")
        print("\n  Sample L2 categories per L1:")
        for l1 in CANONICAL_L1:
            if l1 in final_l2_counts:
                top = final_l2_counts[l1].most_common(8)
                total_in_l1 = sum(final_l2_counts[l1].values())
                unique_in_l1 = len(final_l2_counts[l1])
                print(f"\n    {l1} ({total_in_l1:,} claims, {unique_in_l1:,} L2s):")
                for l2, c in top:
                    print(f"      {l2:55s} {c:>6,}")
        return

    print("\nPhase 3: Writing changes to database...", flush=True)
    t2 = time.time()

    update_batch = []
    for claim_id, data, path in updates:
        vp_json = json.dumps(data.get('view_paths', {}))
        data_json = json.dumps(data)
        update_batch.append((vp_json, data_json, claim_id))

        if len(update_batch) >= 10000:
            conn.executemany(
                "UPDATE claims SET view_paths = ?, data = ? WHERE claim_id = ?",
                update_batch
            )
            conn.commit()
            update_batch = []

    if update_batch:
        conn.executemany(
            "UPDATE claims SET view_paths = ?, data = ? WHERE claim_id = ?",
            update_batch
        )
        conn.commit()

    elapsed3 = time.time() - t2
    print(f"  Claims updated in {elapsed3:.0f}s")

    print("\nPhase 4: Rebuilding tree nodes...", flush=True)
    t3 = time.time()

    all_paths = [(cid, path) for cid, _, path in updates if path]
    node_count = rebuild_tree_nodes(conn, all_paths)

    elapsed4 = time.time() - t3
    print(f"  Tree rebuilt: {node_count:,} nodes in {elapsed4:.0f}s")

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE in {total_elapsed:.0f}s")
    print(f"  Claims with {VIEW_ID}: {kept_count:,} / {total:,}")
    print(f"  Claims dropped from view: {dropped_count:,}")
    print(f"  L1 categories: {len(new_l1_counts)}")
    print(f"  L2 categories: {total_l2_after:,} (was {total_l2_before:,})")
    print(f"  Tree nodes: {node_count:,}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
