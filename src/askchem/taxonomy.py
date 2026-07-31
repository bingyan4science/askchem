"""
Canonical taxonomy definitions for all views.

Single source of truth for L1, L2, and L3 categories. Used by:
  - process_submission (server.py) for classifying new claims
  - update_index.py for incremental updates
  - Any future ingestion pipeline

L1: fixed categories (claim must match exactly one)
L2: fixed subcategories under each L1 (claim must match exactly one)
L3: canonical subcategories for large L2 nodes (>= 5K claims); defined in canonical_l3.py
"""

import json
import os
from pathlib import Path

from askchem.canonical_l3 import CANONICAL_L3

# ── L1 definitions ───────────────────────────────────────────────────────────

CANONICAL_L1 = {
    "by_reaction_type": [
        "catalysis", "electrocatalysis", "photocatalysis",
        "oxidation", "reduction", "polymerization",
        "coupling", "substitution", "addition",
        "elimination", "rearrangement", "cycloaddition",
        "metathesis", "isomerization", "decomposition",
        "combustion", "surface_modification", "synthesis",
        "hydrolysis", "condensation", "self_assembly",
        "biochemistry", "adsorption", "radical",
        "thermochemistry", "degradation",
    ],
    "by_substance_class": [
        "biomolecules", "nanomaterials", "organic_compounds",
        "polymers", "inorganic_compounds", "coordination_compounds",
        "carbon_materials", "semiconductors", "composites",
        "solvents_and_gases",
    ],
    "by_technique": [
        "spectroscopy", "computational_modeling", "electrochemistry",
        "microscopy", "chromatography", "diffraction",
        "thermal_analysis", "mass_spectrometry", "synthesis",
        "biological_assay", "mechanical_testing", "surface_analysis",
        "kinetics", "machine_learning", "materials_processing",
    ],
    "by_application": [
        "energy", "biomedicine", "environmental", "sensing",
        "catalysis", "materials_science", "separations",
        "synthetic_chemistry", "electronics", "food_science",
        "coatings",
    ],
    "by_mechanism": [
        "bond_formation_and_breaking", "electron_transfer",
        "photophysics_and_excited_states", "catalytic_cycles",
        "adsorption_and_surface", "transport_and_diffusion",
        "self_assembly_and_phase", "molecular_recognition",
        "conformational_and_structural", "degradation_and_stability",
    ],
}

# ── L2 definitions ───────────────────────────────────────────────────────────
# Each L1 maps to its allowed L2 subcategories. Every list ends with "other".

CANONICAL_L2 = {
    # ── by_reaction_type ─────────────────────────────────────────────────────
    "by_reaction_type": {
        "catalysis": [
            "heterogeneous_catalysis", "homogeneous_catalysis",
            "asymmetric_catalysis", "organocatalysis", "biocatalysis",
            "tandem_catalysis", "single_atom_catalysis",
            "transition_metal_catalysis", "acid_base_catalysis",
            "catalyst_design_and_synthesis", "other",
        ],
        "electrocatalysis": [
            "oxygen_evolution", "hydrogen_evolution",
            "co2_reduction", "nitrogen_reduction",
            "organic_electrosynthesis", "fuel_cells",
            "electrochemical_sensing", "energy_storage",
            "electrodeposition", "corrosion", "other",
        ],
        "photocatalysis": [
            "photodegradation", "water_splitting",
            "co2_photoreduction", "organic_photocatalysis",
            "photopolymerization", "photosensitization",
            "photodynamic_therapy", "other",
        ],
        "oxidation": [
            "selective_oxidation", "atmospheric_oxidation",
            "radical_oxidation", "advanced_oxidation_processes",
            "photooxidation", "dehydrogenation",
            "electrooxidation", "other",
        ],
        "reduction": [
            "hydrogenation", "transfer_hydrogenation",
            "electroreduction", "photoreduction",
            "reductive_amination", "bioreduction",
            "co2_reduction", "nitrogen_reduction", "other",
        ],
        "polymerization": [
            "radical_polymerization", "controlled_radical_polymerization",
            "ring_opening_polymerization", "step_growth_polymerization",
            "condensation_polymerization", "coordination_polymerization",
            "electrochemical_polymerization", "copolymerization",
            "biosynthetic_polymerization", "polymer_modification", "other",
        ],
        "coupling": [
            "cross_coupling", "c_h_activation", "cycloaddition",
            "click_chemistry", "multicomponent_reactions",
            "radical_coupling", "bioconjugation",
            "heterocycle_assembly", "amide_bond_formation",
            "macrocyclization", "carbon_heteroatom_coupling", "other",
        ],
        "substitution": [
            "nucleophilic_substitution", "electrophilic_substitution",
            "aromatic_substitution", "radical_substitution",
            "c_h_functionalization", "halogenation",
            "late_stage_functionalization", "doping_and_substitution",
            "other",
        ],
        "addition": [
            "michael_addition", "aldol_addition",
            "conjugate_addition", "hydroamination",
            "hydroboration", "hydrosilylation",
            "radical_addition", "other",
        ],
        "elimination": [
            "beta_elimination", "dehydration",
            "decarboxylation", "retro_reactions", "other",
        ],
        "rearrangement": [
            "sigmatropic", "ring_expansion", "ring_contraction",
            "skeletal_rearrangement", "olefin_isomerization",
            "gold_catalyzed_rearrangements", "other",
        ],
        "cycloaddition": [
            "diels_alder", "dipolar_1_3", "two_plus_two",
            "hetero_diels_alder", "bioorthogonal_cycloaddition", "other",
        ],
        "metathesis": [
            "olefin_metathesis", "alkyne_metathesis",
            "ring_closing_metathesis", "cross_metathesis", "other",
        ],
        "isomerization": [
            "cis_trans", "tautomerization",
            "epimerization", "racemization", "other",
        ],
        "decomposition": [
            "thermal_decomposition", "photodecomposition",
            "catalytic_decomposition", "pyrolysis", "other",
        ],
        "combustion": [
            "fuel_combustion", "propellant_combustion",
            "biomass_combustion", "other",
        ],
        "surface_modification": [
            "surface_functionalization", "thin_film_deposition",
            "surface_coating", "surface_passivation",
            "post_synthetic_modification", "etching",
            "grafting", "other",
        ],
        "synthesis": [
            "materials_synthesis", "nanomaterials_synthesis",
            "device_fabrication", "membrane_synthesis",
            "total_synthesis", "coordination_synthesis",
            "sol_gel", "solvothermal", "vapor_deposition",
            "electrodeposition", "other",
        ],
        "hydrolysis": [
            "ester_hydrolysis", "peptide_hydrolysis",
            "silane_hydrolysis", "acid_catalyzed_hydrolysis",
            "polymer_hydrolysis", "other",
        ],
        "condensation": [
            "imine_formation", "knoevenagel_condensation",
            "aldol_condensation", "ester_condensation",
            "peptide_bond_formation", "boronate_ester_formation",
            "mannich_reaction", "sol_gel_condensation", "other",
        ],
        "self_assembly": [
            "crystallization", "supramolecular_assembly",
            "polymer_and_gel_assembly", "framework_assembly",
            "nanomaterial_assembly", "biological_assembly",
            "coordination_assembly", "other",
        ],
        "biochemistry": [
            "enzymatic_catalysis", "post_translational_modification",
            "metabolic_pathways", "protein_interactions",
            "signal_transduction", "protein_engineering",
            "protein_folding", "bioconjugation",
            "metabolic_engineering", "enzyme_inhibition", "other",
        ],
        "adsorption": [
            "surface_adsorption", "gas_sorption",
            "ion_exchange", "biosorption",
            "adsorption_separation", "other",
        ],
        "radical": [
            "radical_oxidation", "photochemical_radical",
            "aqueous_radical_chemistry", "atmospheric_radical_chemistry",
            "other",
        ],
        "thermochemistry": [
            "biomass_conversion", "combustion",
            "pyrolysis", "hydrogen_production",
            "chemical_looping", "co2_conversion",
            "energy_storage", "other",
        ],
        "degradation": [
            "thermal_decomposition", "environmental_degradation",
            "oxidative_degradation", "polymer_degradation",
            "battery_degradation", "biodegradation",
            "advanced_oxidation_processes", "photodegradation",
            "atmospheric_degradation", "other",
        ],
    },

    # ── by_substance_class ───────────────────────────────────────────────────
    "by_substance_class": {
        "biomolecules": [
            "proteins_and_peptides", "nucleic_acids", "enzymes",
            "lipids_and_membranes", "carbohydrates",
            "small_molecule_metabolites", "cells_and_microorganisms",
            "antibodies", "other",
        ],
        "nanomaterials": [
            "metal_nanoparticles", "metal_oxide_nanoparticles",
            "single_atom_catalysts", "quantum_dots",
            "two_dimensional_materials", "mxenes",
            "perovskite_nanocrystals", "nanoclusters",
            "core_shell_nanostructures", "other",
        ],
        "organic_compounds": [
            "small_molecule_drugs", "natural_products",
            "heterocycles", "fluorescent_probes",
            "organic_semiconductors", "volatile_organic_compounds",
            "dyes_and_chromophores", "ligands_and_reagents",
            "other",
        ],
        "polymers": [
            "synthetic_polymers", "biopolymers",
            "block_copolymers", "conjugated_polymers",
            "hydrogels", "crosslinked_networks",
            "fluorinated_polymers", "conducting_polymers",
            "other",
        ],
        "inorganic_compounds": [
            "metal_oxides", "perovskite_oxides",
            "metals_and_alloys", "electrode_materials",
            "battery_materials", "electrolytes",
            "heterogeneous_catalysts", "coordination_compounds",
            "zeolites", "other",
        ],
        "coordination_compounds": [
            "metal_organic_frameworks", "coordination_polymers",
            "transition_metal_complexes", "supramolecular_assemblies",
            "palladium_complexes", "ligands",
            "other",
        ],
        "carbon_materials": [
            "covalent_organic_frameworks", "graphene_and_derivatives",
            "carbon_nanotubes", "activated_carbon",
            "porous_organic_frameworks", "biochar",
            "fullerenes", "carbon_dots", "other",
        ],
        "semiconductors": [
            "perovskite_semiconductors", "two_dimensional_materials",
            "inorganic_semiconductors", "organic_semiconductors",
            "photocatalysts", "nanostructured_semiconductors",
            "photovoltaic_materials", "other",
        ],
        "composites": [
            "polymer_matrix_composites", "hybrid_perovskites",
            "metal_based_composites", "nanocomposites",
            "core_shell_structures", "hybrid_organic_inorganic",
            "other",
        ],
        "solvents_and_gases": [
            "atmospheric_gases", "greenhouse_gases",
            "electrolytes", "ionic_liquids",
            "hydrogen", "interstellar_gases",
            "aqueous_solutions", "other",
        ],
    },

    # ── by_technique ─────────────────────────────────────────────────────────
    "by_technique": {
        "spectroscopy": [
            "uv_vis_spectroscopy", "photoluminescence",
            "raman_spectroscopy", "nmr_spectroscopy",
            "infrared_spectroscopy", "x_ray_spectroscopy",
            "ultrafast_spectroscopy", "fluorescence_spectroscopy",
            "epr_spectroscopy", "other",
        ],
        "computational_modeling": [
            "density_functional_theory", "molecular_dynamics",
            "ab_initio_methods", "monte_carlo",
            "quantum_chemistry", "coarse_grained_modeling",
            "finite_element_methods", "other",
        ],
        "electrochemistry": [
            "voltammetry", "impedance_spectroscopy",
            "battery_testing", "electrocatalytic_testing",
            "rotating_disk_electrode", "galvanostatic_methods",
            "electrochemical_sensing", "fuel_cell_testing",
            "other",
        ],
        "microscopy": [
            "electron_microscopy", "optical_microscopy",
            "scanning_probe_microscopy", "cryo_electron_microscopy",
            "fluorescence_microscopy", "in_vivo_imaging",
            "confocal_microscopy", "other",
        ],
        "chromatography": [
            "hplc", "gas_chromatography",
            "lc_ms", "size_exclusion_chromatography",
            "ion_chromatography", "other",
        ],
        "diffraction": [
            "x_ray_diffraction", "single_crystal_diffraction",
            "saxs_waxs", "neutron_diffraction",
            "electron_diffraction", "other",
        ],
        "thermal_analysis": [
            "thermogravimetric_analysis", "differential_scanning_calorimetry",
            "calorimetry", "cure_kinetics", "other",
        ],
        "mass_spectrometry": [
            "aerosol_mass_spectrometry", "lc_ms",
            "isotope_ratio_ms", "chemical_ionization_ms",
            "imaging_mass_spectrometry", "proteomics_ms",
            "other",
        ],
        "synthesis": [
            "materials_synthesis", "organic_synthesis",
            "colloidal_synthesis", "solvothermal_synthesis",
            "polymer_synthesis", "reticular_synthesis",
            "solution_processing", "post_synthetic_modification",
            "other",
        ],
        "biological_assay": [
            "cell_based_assays", "biochemical_assays",
            "genomics", "enzymology",
            "high_throughput_screening", "in_vivo_models",
            "x_ray_crystallography", "binding_assays",
            "mutagenesis", "other",
        ],
        "mechanical_testing": [
            "rheology", "tensile_testing",
            "compression_testing", "nanoindentation",
            "other",
        ],
        "surface_analysis": [
            "xps", "physisorption_and_surface_area",
            "adsorption_isotherms", "contact_angle",
            "surface_characterization", "other",
        ],
        "kinetics": [
            "reaction_kinetics", "adsorption_kinetics",
            "enzyme_kinetics", "atmospheric_kinetics",
            "photocatalytic_kinetics", "mechanistic_probes",
            "other",
        ],
        "machine_learning": [
            "deep_learning", "graph_neural_networks",
            "generative_models", "nlp_models",
            "cheminformatics", "bioinformatics",
            "machine_learning_potentials", "other",
        ],
        "materials_processing": [
            "thin_film_deposition", "device_fabrication",
            "membrane_fabrication", "thermal_processing",
            "additive_manufacturing", "electrospinning",
            "composite_fabrication", "other",
        ],
    },

    # ── by_application ───────────────────────────────────────────────────────
    "by_application": {
        "energy": [
            "batteries", "solar_cells", "fuel_cells",
            "supercapacitors", "hydrogen_storage",
            "water_splitting", "thermoelectrics",
            "energy_harvesting", "other",
        ],
        "biomedicine": [
            "drug_delivery", "cancer_therapy",
            "diagnostics", "tissue_engineering",
            "antimicrobial", "bioimaging",
            "gene_therapy", "wound_healing",
            "neuroscience", "other",
        ],
        "environmental": [
            "water_treatment", "atmospheric_chemistry",
            "carbon_capture", "air_quality",
            "soil_remediation", "waste_management",
            "aerosol_science", "climate_modeling",
            "other",
        ],
        "sensing": [
            "biosensing", "chemical_sensing",
            "electrochemical_sensing", "optical_sensing",
            "gas_sensing", "imaging",
            "other",
        ],
        "catalysis": [
            "heterogeneous_catalysis", "homogeneous_catalysis",
            "photocatalysis", "electrocatalysis",
            "biocatalysis", "asymmetric_catalysis",
            "other",
        ],
        "materials_science": [
            "porous_materials", "optoelectronics",
            "electronic_materials", "nanomaterials",
            "surface_engineering", "materials_characterization",
            "functional_materials", "structural_materials",
            "other",
        ],
        "separations": [
            "membrane_separations", "adsorption_separations",
            "gas_separation", "chromatographic_separations",
            "water_purification", "other",
        ],
        "synthetic_chemistry": [
            "method_development", "reaction_optimization",
            "asymmetric_synthesis", "green_chemistry",
            "total_synthesis", "heterocycle_synthesis",
            "c_h_functionalization", "polymer_synthesis",
            "other",
        ],
        "electronics": [
            "transistors", "leds", "memory_devices",
            "flexible_electronics", "photovoltaics",
            "other",
        ],
        "food_science": [
            "food_packaging", "food_safety",
            "nutrition", "crop_improvement",
            "food_preservation", "other",
        ],
        "coatings": [
            "anticorrosion", "antifouling",
            "self_healing", "superhydrophobic",
            "optical_coatings", "other",
        ],
    },

    # ── by_mechanism ─────────────────────────────────────────────────────────
    "by_mechanism": {
        "bond_formation_and_breaking": [
            "radical_mechanisms", "c_h_activation",
            "c_x_bond_activation", "small_molecule_activation",
            "protonation_deprotonation", "atmospheric_radical_reactions",
            "hydroxyl_radical_chemistry", "reaction_kinetics",
            "other",
        ],
        "electron_transfer": [
            "band_structure", "proton_coupled_electron_transfer",
            "interfacial_charge_transfer", "photoinduced_electron_transfer",
            "charge_storage", "spin_and_magnetism",
            "orbital_interactions", "intercalation",
            "electrocatalytic_mechanisms", "other",
        ],
        "photophysics_and_excited_states": [
            "energy_transfer", "charge_separation",
            "absorption_and_emission", "plasmonics",
            "excited_state_dynamics", "carrier_dynamics",
            "fluorescence", "other",
        ],
        "catalytic_cycles": [
            "structure_activity_relationships", "enzyme_catalysis",
            "acid_base_catalysis", "cross_coupling_cycle",
            "c_h_activation_mechanisms", "organocatalysis",
            "enzyme_inhibition", "transition_metal_catalysis",
            "other",
        ],
        "adsorption_and_surface": [
            "adsorption_mechanisms", "physisorption_and_chemisorption",
            "surface_reactions", "metal_support_interactions",
            "ion_adsorption", "host_guest_interactions",
            "plasmonic_enhancement", "other",
        ],
        "transport_and_diffusion": [
            "ion_transport", "electronic_transport",
            "mass_diffusion", "atmospheric_transport",
            "charge_transport", "quantum_tunneling",
            "membrane_permeation", "other",
        ],
        "self_assembly_and_phase": [
            "nucleation_and_growth", "crystallization",
            "coordination_self_assembly", "glass_transition",
            "polymorphism", "protein_folding",
            "dynamic_covalent_chemistry", "aerosol_nucleation",
            "other",
        ],
        "molecular_recognition": [
            "protein_ligand_binding", "protein_protein_interactions",
            "nucleic_acid_interactions", "host_guest_recognition",
            "hydrogen_bonding", "electrostatic_interactions",
            "noncovalent_interactions", "cellular_recognition",
            "other",
        ],
        "conformational_and_structural": [
            "structure_property_relationships", "crystal_structure",
            "lattice_dynamics", "mechanical_properties",
            "porosity_and_pore_structure", "thermodynamics",
            "microstructure_and_morphology", "other",
        ],
        "degradation_and_stability": [
            "electrochemical_degradation", "thermal_stability",
            "catalyst_deactivation", "biodegradation",
            "chemical_stability", "photostability",
            "mechanical_degradation", "environmental_persistence",
            "other",
        ],
    },
}

CLAIM_TYPE_LABELS = {
    "reaction": "reaction",
    "scope_entry": "reaction",
    "property": "property",
    "structure": "property",
    "method": "method",
    "experimental_design": "method",
    "mechanism": "mechanism",
    "comparison": "comparison",
    "computational_result": "computational_result",
    "hypothesis": "hypothesis",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitation": "limitation",
    "future_direction": "future_direction",
    "surprising_finding": "surprising_finding",
}

_TAXONOMY_V2_PATH = Path(
    os.environ.get(
        "ASKCHEM_TAXONOMY_PATH",
        Path(__file__).with_name("taxonomy_v2.json"),
    )
)
TAXONOMY_VERSION = "v1"
if (_TAXONOMY_V2_PATH.exists()
        and os.environ.get("ASKCHEM_DISABLE_TAXONOMY_V2", "0") != "1"):
    _v2 = json.loads(_TAXONOMY_V2_PATH.read_text())
    TAXONOMY_VERSION = _v2["taxonomy_version"]
    CANONICAL_L1 = _v2["canonical_l1"]
    CANONICAL_L2 = _v2["canonical_l2"]
    CANONICAL_L3 = {
        view: {
            tuple(path.split("/", 1)): values
            for path, values in parents.items()
        }
        for view, parents in _v2["canonical_l3"].items()
    }

ALL_CONTENT_VIEWS = list(CANONICAL_L1.keys())


def _build_taxonomy_text() -> str:
    """Build the L1→L2 taxonomy text (no L3). Used for batch reclassification."""
    lines = []
    for view_id in ALL_CONTENT_VIEWS:
        lines.append(f"\n{view_id}:")
        l2_map = CANONICAL_L2.get(view_id, {})
        for l1 in CANONICAL_L1[view_id]:
            l2s = l2_map.get(l1, ["other"])
            lines.append(f"  {l1}: {', '.join(l2s)}")
    return "\n".join(lines)


def _build_full_taxonomy_text() -> str:
    """Build the L1→L2→L3 taxonomy text. Used for single-call classification."""
    lines = []
    for view_id in ALL_CONTENT_VIEWS:
        lines.append(f"\n{view_id}:")
        l2_map = CANONICAL_L2.get(view_id, {})
        l3_map = CANONICAL_L3.get(view_id, {})
        for l1 in CANONICAL_L1[view_id]:
            l2s = l2_map.get(l1, ["other"])
            lines.append(f"  {l1}: {', '.join(l2s)}")
            for l2 in l2s:
                l3_key = (l1, l2)
                if l3_key in l3_map:
                    lines.append(f"    {l2} → {', '.join(l3_map[l3_key])}")
    return "\n".join(lines)


_TAXONOMY_TEXT = _build_taxonomy_text()
_FULL_TAXONOMY_TEXT = _build_full_taxonomy_text()


def _return_shape(full: bool = False) -> str:
    path = ["l1", "l2", "l3"] if full else ["l1", "l2"]
    return json.dumps({view: path for view in ALL_CONTENT_VIEWS})


# Two-step prompts (for batch reclassification of existing claims)
CLASSIFICATION_SYSTEM_PROMPT = f"""Classify chemistry claims into each listed hierarchical view.

Rules:
- L1 MUST be one of the listed categories (exactly one per view).
- L2 MUST be one of the listed subcategories under that L1.
- L3 is NOT needed here — it will be assigned separately.
- Use lowercase_with_underscores.
- If the claim does not fit a view, use ["not_applicable"].

Canonical categories (L1 → allowed L2):
{_TAXONOMY_TEXT}

Return JSON with exactly these view keys and path shape:
{_return_shape()}"""


L3_ASSIGNMENT_SYSTEM_PROMPT = """Assign L3 subcategories to a chemistry claim that has already been classified at L1/L2.

For each view path given, pick the best L3 from the allowed list.
Use lowercase_with_underscores. Pick "other" only if none of the specific categories fit.

Return JSON with the same keys, each value being the chosen L3 string."""


# Single-call prompt (for new individual claims — includes L3 in one shot)
FULL_CLASSIFICATION_SYSTEM_PROMPT = f"""Classify chemistry claims into each listed hierarchical view.

Rules:
- L1 MUST be one of the listed categories (exactly one per view).
- L2 MUST be one of the listed subcategories under that L1.
- L3: If the L2 has listed L3 subcategories (shown as "L2 → L3a, L3b, ..."), you MUST pick one. If no L3 is listed for that L2, omit L3 (path is just [L1, L2]).
- Use lowercase_with_underscores.
- If the claim does not fit a view, use ["not_applicable"].

Canonical categories (L1 → L2, and L2 → L3 where defined):
{_FULL_TAXONOMY_TEXT}

Return JSON with exactly these view keys and paths of two or three segments:
{_return_shape(full=True)}"""


def build_classification_prompt(claim_type: str, quote: str, title: str) -> str:
    """Build the LLM user message for classification."""
    return f"Claim type: {claim_type}\nClaim: {quote}\nPaper: {title}"


def build_classification_messages(claim_type: str, quote: str, title: str) -> list[dict]:
    """Build messages for L1/L2-only classification (two-step mode for batch)."""
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_classification_prompt(claim_type, quote, title)},
    ]


def build_full_classification_messages(claim_type: str, quote: str, title: str) -> list[dict]:
    """Build messages for single-call L1/L2/L3 classification (for new claims).

    One API call returns all levels. Prompt is larger (~23K tokens) but avoids
    a second round-trip, saving latency and total thinking-token cost.
    """
    return [
        {"role": "system", "content": FULL_CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_classification_prompt(claim_type, quote, title)},
    ]


def build_l3_assignment_messages(
    claim_type: str, quote: str, title: str,
    view_paths: dict,
):
    """Build messages for L3 assignment given existing L1/L2 paths.

    Returns None if no views need L3 assignment.
    """
    l3_needed = {}
    for view_id, path in view_paths.items():
        if not isinstance(path, list) or len(path) < 2:
            continue
        l1, l2 = path[0], path[1]
        l3_cats = get_canonical_l3(view_id, l1, l2)
        if l3_cats is not None:
            l3_needed[view_id] = {
                "path": f"{l1}/{l2}",
                "allowed_l3": l3_cats,
            }

    if not l3_needed:
        return None

    lines = [f"Claim type: {claim_type}", f"Claim: {quote}", f"Paper: {title}", ""]
    lines.append("Assign L3 for each view:")
    for view_id, info in l3_needed.items():
        lines.append(f"  {view_id} ({info['path']}): {', '.join(info['allowed_l3'])}")

    return [
        {"role": "system", "content": L3_ASSIGNMENT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def normalize_path(view_id: str, path: list):
    """Normalize a view path: validate L1, L2, and L3 against canonical lists."""
    if not path or not isinstance(path, list):
        return None

    cleaned = []
    for seg in path:
        s = str(seg).strip().lower().replace('-', '_').replace(' ', '_')
        if s and s not in ('not_applicable', 'none', ''):
            cleaned.append(s)

    if not cleaned:
        return None

    # Resolve legacy/free-form aliases before validating canonical IDs.  This
    # prevents ingestion from recreating a deprecated taxonomy node.
    from askchem.taxonomy_aliases import resolve_tree_path
    resolved, _ = resolve_tree_path(view_id, "/".join(cleaned))
    cleaned = resolved.split("/") if resolved else []

    # Validate L1
    canonical_l1 = set(CANONICAL_L1.get(view_id, []))
    if canonical_l1 and cleaned[0] not in canonical_l1:
        return None

    # Validate L2 if present
    if len(cleaned) >= 2:
        l2_map = CANONICAL_L2.get(view_id, {})
        allowed_l2 = set(l2_map.get(cleaned[0], ["other"]))
        if allowed_l2 and cleaned[1] not in allowed_l2:
            cleaned[1] = "other"

    # Validate L3 against canonical definitions
    l3_map = CANONICAL_L3.get(view_id, {})
    if len(cleaned) >= 2:
        l3_key = (cleaned[0], cleaned[1])
        allowed_l3 = l3_map.get(l3_key)
        if allowed_l3 is not None:
            norm_to_canon = {a.strip().lower().replace('-', '_').replace(' ', '_'): a for a in allowed_l3}
            if len(cleaned) >= 3:
                if cleaned[2] not in norm_to_canon:
                    cleaned[2] = "other"
                else:
                    cleaned[2] = norm_to_canon[cleaned[2]]
            else:
                cleaned.append("other")
        else:
            cleaned = cleaned[:2]

    # Cap at 3 levels
    return cleaned[:3]


def has_canonical_l3(view_id: str, l1: str, l2: str) -> bool:
    """Check if a (view, L1, L2) triple has canonical L3 defined."""
    l3_map = CANONICAL_L3.get(view_id, {})
    return (l1, l2) in l3_map


def get_canonical_l3(view_id: str, l1: str, l2: str):
    """Get canonical L3 categories for a (view, L1, L2) triple, or None."""
    l3_map = CANONICAL_L3.get(view_id, {})
    return l3_map.get((l1, l2))


def build_claim_type_path(claim_type: str) -> list:
    """Build the by_claim_type view path from claim_type."""
    ct_l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
    if not ct_l1:
        ct_l1 = "other"
    return [ct_l1]
