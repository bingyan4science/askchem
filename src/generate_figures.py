"""
Generate publication-quality figures for the AskChem paper.

Figures:
1. Architecture overview (conceptual diagram)
2. Extraction pipeline comparison (v1 vs v2 vs scaled)
3. Claim type distribution (sunburst/treemap)
4. Multi-view hierarchy visualization
5. Canonical coverage heatmap
6. Frontier/gap analysis
7. Scale statistics
"""

import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from askchem.display import smart_title

PAPER_DIR = Path(__file__).parent.parent / "structure_the_universe_paper" / "figures"
DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
INDEX_DIR = Path(__file__).parent.parent / "chemtree_index"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

COLORS = {
    'primary': '#1a5276',
    'secondary': '#2980b9',
    'accent': '#e74c3c',
    'green': '#27ae60',
    'orange': '#f39c12',
    'purple': '#8e44ad',
    'teal': '#16a085',
    'gray': '#7f8c8d',
    'light_bg': '#f8f9fa',
}

CLAIM_COLORS = {
    'method': '#2980b9',
    'property': '#27ae60',
    'computational_result': '#8e44ad',
    'mechanism': '#f39c12',
    'comparison': '#e74c3c',
    'reaction': '#1a5276',
    'scope_entry': '#16a085',
    'observation': '#7f8c8d',
}


def load_data():
    """Load all data needed for figures, preferring pipeline results for 100K-scale data."""
    data = {}

    # V2 extraction results (for quality comparison)
    v2_path = EXPERIMENTS_DIR / "003_extraction_v2" / "results" / "all_extractions_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            data['v2_extractions'] = json.load(f)

    # Scaled extraction checkpoints (500-paper pilot)
    scale_dir = EXPERIMENTS_DIR / "005_scale_extraction" / "checkpoints"
    if scale_dir.exists():
        data['scaled_claims'] = []
        for batch_file in sorted(scale_dir.glob("batch_*.json")):
            with open(batch_file) as f:
                data['scaled_claims'].extend(json.load(f))

    # Canonical map
    canon_path = EXPERIMENTS_DIR / "007_topdown_canonical" / "raw" / "canonical_map.json"
    if canon_path.exists():
        with open(canon_path) as f:
            data['canonical_map'] = json.load(f)

    # Coverage analysis
    coverage_path = EXPERIMENTS_DIR / "007_topdown_canonical" / "results" / "coverage_analysis.json"
    if coverage_path.exists():
        with open(coverage_path) as f:
            data['coverage'] = json.load(f)

    # Corpus metadata (for year/citation info)
    corpus_path = DATA_DIR / "metadata" / "all_papers.json"
    if corpus_path.exists():
        with open(corpus_path) as f:
            all_papers = json.load(f)
        data['all_papers'] = all_papers
        data['paper_meta'] = {}
        for p in all_papers:
            doi = (p.get('externalIds') or {}).get('DOI', '')
            if doi:
                data['paper_meta'][doi] = p

    # Load 100K-scale extraction results from pipeline
    pipeline_dir = INDEX_DIR / "_pipeline"
    results_dir = pipeline_dir / "extraction_results"
    if results_dir.exists():
        print("  Loading 100K extraction results...", flush=True)
        extractions = {}
        for f in sorted(results_dir.glob("*.jsonl")):
            if f.name.startswith("errors_"):
                continue
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        doi = r.get('custom_id', '').replace('extract::', '')
                        body = r.get('response', {}).get('body', {})
                        choices = body.get('choices', [])
                        if choices:
                            content = choices[0].get('message', {}).get('content', '')
                            if content:
                                claims = json.loads(content).get('claims', [])
                                extractions[doi] = claims
                    except Exception:
                        pass
        data['batch_extractions'] = extractions
        print(f"  Loaded {len(extractions):,} papers, {sum(len(v) for v in extractions.values()):,} claims", flush=True)

    # Load 100K-scale classification results from pipeline
    class_dir = pipeline_dir / "classification_results"
    if class_dir.exists():
        print("  Loading 100K classification results...", flush=True)
        classifications = {}
        for f in sorted(class_dir.glob("*.jsonl")):
            if f.name.startswith("errors_"):
                continue
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        cid = r.get('custom_id', '').replace('classify::', '')
                        body = r.get('response', {}).get('body', {})
                        choices = body.get('choices', [])
                        if choices:
                            content = choices[0].get('message', {}).get('content', '')
                            if content:
                                classifications[cid] = json.loads(content)
                    except Exception:
                        pass
        data['batch_classifications'] = classifications
        print(f"  Loaded {len(classifications):,} classifications", flush=True)

    return data


def fig1_architecture(data):
    """Figure 1: AskChem architecture overview."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.7, 'AskChem Architecture', ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    papers_box = FancyBboxPatch((0.3, 0.3), 9.4, 1.0,
                                 boxstyle="round,pad=0.1", facecolor='#eaf2f8',
                                 edgecolor=COLORS['secondary'], linewidth=1)
    ax.add_patch(papers_box)
    ax.text(5, 0.8, 'Research Papers (104,249 from Semantic Scholar)',
            ha='center', va='center', fontsize=8, color=COLORS['primary'])
    ax.text(1.5, 0.45, '249K collected', ha='center', fontsize=6, color=COLORS['gray'])
    ax.text(5, 0.45, '14 subfields', ha='center', fontsize=6, color=COLORS['gray'])
    ax.text(8.5, 0.45, '1888–2026', ha='center', fontsize=6, color=COLORS['gray'])

    ax.annotate('', xy=(5, 2.0), xytext=(5, 1.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))
    ax.text(6.5, 1.65, 'GPT-5-mini (Batch API)\n7.8 claims/paper',
            ha='center', va='center', fontsize=6, style='italic', color=COLORS['accent'])

    claims_box = FancyBboxPatch((0.3, 2.0), 9.4, 1.0,
                                 boxstyle="round,pad=0.1", facecolor='#fef9e7',
                                 edgecolor=COLORS['orange'], linewidth=1)
    ax.add_patch(claims_box)
    ax.text(5, 2.5, 'Claim Store — 814,339 structured claims',
            ha='center', va='center', fontsize=8, fontweight='bold', color=COLORS['primary'])
    claim_types = ['reaction', 'property', 'method', 'mechanism', 'comparison', 'computational']
    for i, ct in enumerate(claim_types):
        x = 1.0 + i * 1.5
        ax.text(x, 2.15, ct, ha='center', fontsize=5.5, color=CLAIM_COLORS.get(ct, COLORS['gray']),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=CLAIM_COLORS.get(ct, COLORS['gray']), linewidth=0.5))

    for i, (x, name) in enumerate([
        (1.0, 'Reaction\nType'),
        (3.0, 'Substance\nClass'),
        (5.0, 'Application\nDomain'),
        (7.0, 'Technique/\nMethod'),
        (9.0, 'Mechanism/\nPhenomenon'),
    ]):
        ax.annotate('', xy=(x, 3.7), xytext=(5, 3.1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=0.8))

        view_box = FancyBboxPatch((x - 0.85, 3.7), 1.7, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=['#eaf2f8', '#e8f8f5', '#fef9e7', '#f4ecf7', '#fdedec'][i],
                                   edgecolor=COLORS['secondary'], linewidth=0.8)
        ax.add_patch(view_box)
        ax.text(x, 4.3, name, ha='center', va='center', fontsize=6.5, fontweight='bold',
                color=COLORS['primary'])

    ax.annotate('', xy=(5, 5.5), xytext=(5, 5.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5))

    api_box = FancyBboxPatch((1.5, 5.5), 7.0, 0.8,
                              boxstyle="round,pad=0.1", facecolor='#e8f8f5',
                              edgecolor=COLORS['green'], linewidth=1)
    ax.add_patch(api_box)
    ax.text(5, 5.9, 'REST API + MCP Server', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['green'])

    for x, label in [(2.5, 'AI Agents'), (5.0, 'Web Explorer'), (7.5, 'Python SDK')]:
        ax.text(x, 5.6, label, ha='center', fontsize=6, color=COLORS['gray'])

    fig.savefig(PAPER_DIR / 'fig1_architecture.pdf')
    fig.savefig(PAPER_DIR / 'fig1_architecture.png')
    plt.close()
    print("  Fig 1: Architecture overview", flush=True)


def fig2_extraction_comparison(data):
    """Figure 2: Extraction pipeline comparison across versions."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Panel A: Claims per paper across versions
    ax = axes[0]
    versions = ['v1\n(single)', 'v1\n(multi)', 'v2\n(two-stage)', 'v3 pilot\n(abstract)', 'v3 batch\n(104K)']
    claims_per_paper = [8.6, 11.9, 13.0, 6.1, 7.8]
    colors = [COLORS['gray'], COLORS['gray'], COLORS['secondary'], COLORS['teal'], COLORS['primary']]
    bars = ax.bar(versions, claims_per_paper, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Claims per paper')
    ax.set_title('a  Extraction yield', loc='left', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, claims_per_paper):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6.5)

    # Panel B: Claim type distribution (100K batch data)
    ax = axes[1]
    batch_ext = data.get('batch_extractions', {})
    if batch_ext:
        type_counts = Counter()
        for doi, claims in batch_ext.items():
            for claim in claims:
                type_counts[claim.get('claim_type', 'unknown')] += 1

        canonical_types = ['property', 'method', 'mechanism', 'comparison', 'computational_result', 'reaction']
        counts = [type_counts.get(t, 0) for t in canonical_types]
        colors_list = [CLAIM_COLORS.get(t, COLORS['gray']) for t in canonical_types]
        short_names = [t.replace('_', '\n') for t in canonical_types]

        ax.barh(short_names[::-1], counts[::-1], color=colors_list[::-1], height=0.6, edgecolor='white')
        ax.set_xlabel('Count')
        ax.set_title('b  Claim types (104K papers)', loc='left', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i, (cnt, name) in enumerate(zip(counts[::-1], short_names[::-1])):
            ax.text(cnt + max(counts)*0.02, i, f'{cnt:,}', va='center', fontsize=5.5, color=COLORS['gray'])

    # Panel C: Scale progression
    ax = axes[2]
    stages = ['Phase 0\n(10)', 'Phase 1\n(500)', 'Phase 2\n(104K)']
    papers = [10, 500, 104249]
    claims = [130, 3203, 814339]

    ax2 = ax.twinx()
    x = np.arange(len(stages))
    w = 0.35
    ax.bar(x - w/2, papers, w, color=COLORS['secondary'], label='Papers', alpha=0.8)
    ax2.bar(x + w/2, claims, w, color=COLORS['accent'], label='Claims', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Papers', color=COLORS['secondary'])
    ax2.set_ylabel('Claims', color=COLORS['accent'])
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_title('c  Scaling trajectory', loc='left', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / 'fig2_extraction.pdf')
    fig.savefig(PAPER_DIR / 'fig2_extraction.png')
    plt.close()
    print("  Fig 2: Extraction comparison", flush=True)


def fig3_multiview_hierarchy(data):
    """Figure 3: Multi-view hierarchy visualization built from 100K classification data."""
    classifications = data.get('batch_classifications', {})
    if not classifications:
        print("  Fig 3: SKIPPED (no classification data)", flush=True)
        return

    VIEW_META = [
        ('by_reaction_type', 'Reaction\nType', '#1a5276'),
        ('by_substance_class', 'Substance\nClass', '#27ae60'),
        ('by_application', 'Application\nDomain', '#f39c12'),
        ('by_technique', 'Technique/\nMethod', '#8e44ad'),
        ('by_mechanism', 'Mechanism/\nPhenomenon', '#e74c3c'),
    ]

    NUM_L1 = 8
    NUM_L2 = 3

    PRETTY_NAMES = {
        'catalysis': 'Catalysis',
        'materials_and_surface_processes': 'Materials & Surfaces',
        'redox_reactions': 'Redox',
        'synthetic_methods': 'Synthesis',
        'electrochemical_processes': 'Electrochemical',
        'coupling_and_bond_formation': 'Coupling & Bonds',
        'functional_group_transformations': 'Functional Groups',
        'biochemical_and_enzymatic_transformations': 'Biochemical',
        'decomposition_and_degradation': 'Decomposition',
        'hydrolysis_and_condensation': 'Hydrolysis',
        'thermochemical_and_energy_conversion': 'Thermochemical',
        'photochemical_processes': 'Photochemistry',
        'polymerization': 'Polymerization',
        'materials': 'Materials',
        'organic_compounds': 'Organic',
        'biomolecules_and_biological_materials': 'Biomolecules',
        'inorganic_compounds': 'Inorganic',
        'small_molecules_and_probes': 'Small Molecules',
        'surfaces_interfaces_and_molecular_systems': 'Surfaces & Interfaces',
        'polymers_and_biopolymers': 'Polymers',
        'catalysts': 'Catalysts',
        'nanomaterials': 'Nanomaterials',
        'coordination_and_organometallic_compounds': 'Coordination Chem.',
        'computational_and_electronic_structure': 'Computational',
        'solvents_and_gases': 'Solvents & Gases',
        'carbonaceous_and_porous_materials': 'Porous & Carbon',
        'materials_chemistry': 'Materials',
        'computational_and_theoretical_chemistry': 'Computational',
        'biological_and_biotechnological_chemistry': 'Biology & Biotech',
        'analytical_and_characterization': 'Analytical',
        'energy_and_renewables': 'Energy',
        'environmental_and_sustainable_chemistry': 'Environmental',
        'pharmaceuticals_and_drug_discovery': 'Pharma & Drug Disc.',
        'synthetic_chemistry': 'Synthesis',
        'separations_and_processes': 'Separations',
        'food_and_nutrition_chemistry': 'Food Science',
        'computational_modeling_and_theory': 'Computational',
        'characterization_and_analytical_methods': 'Characterization',
        'spectroscopy_and_spectroscopic_methods': 'Spectroscopy',
        'synthesis_and_reaction_methods': 'Synthesis Methods',
        'electrochemistry_and_electrochemical_methods': 'Electrochemistry',
        'catalysis_and_biocatalysis': 'Catalysis',
        'physical_methods_and_materials_processing': 'Physical Methods',
        'structural_biology_and_bioassays': 'Bioassays',
        'experimental_methods_and_techniques': 'Experimental',
        'microscopy_and_imaging': 'Microscopy',
        'kinetics_and_mechanistic_studies': 'Kinetics',
        'data_science_and_machine_learning': 'ML & Data Science',
        'catalysis_and_catalytic_mechanisms': 'Catalytic Mech.',
        'electronic_structure_and_quantum_chemistry': 'Electronic Structure',
        'reaction_mechanisms_and_kinetics': 'Reaction Kinetics',
        'materials_structure_and_lattice_dynamics': 'Materials/Lattice',
        'heterogeneous_and_surface_processes': 'Surface Processes',
        'molecular_and_noncovalent_interactions': 'Molecular Interact.',
        'photophysics_and_excited_state_processes': 'Photophysics',
        'biological_and_enzymatic_mechanisms': 'Biological Mech.',
        'interfacial_and_transport_processes': 'Transport',
        'electrochemistry_and_charge_transfer': 'Charge Transfer',
        'self_assembly_nucleation_and_crystallization': 'Self-Assembly',
        'stability_and_degradation_and_decomposition': 'Stability/Degrad.',
        'thermodynamics_and_phase_behavior': 'Thermodynamics',
        'spectroscopy_and_vibrational_dynamics': 'Vibr. Dynamics',
    }

    def pretty(s):
        if s in PRETTY_NAMES:
            return PRETTY_NAMES[s]
        return smart_title(s)

    view_data = {}
    for view_id, title, color in VIEW_META:
        l1_counter = Counter()
        l2_map = {}
        for cid, entry in classifications.items():
            paths = entry.get('paths', {})
            path = paths.get(view_id, [])
            if not path or path == ['not_applicable']:
                continue
            l1 = path[0]
            l1_counter[l1] += 1
            if len(path) >= 2:
                l2_map.setdefault(l1, Counter())[path[1]] += 1

        top_l1 = [(n, c) for n, c in l1_counter.most_common(NUM_L1 + 2)
                   if n != 'other'][:NUM_L1]
        l2_for_l1 = {}
        for l1_name, _ in top_l1:
            if l1_name in l2_map:
                l2_for_l1[l1_name] = [n for n, _ in l2_map[l1_name].most_common(NUM_L2)]

        view_data[title] = {
            'L1': top_l1,
            'L2': l2_for_l1,
            'color': color,
            'total_l1': len([n for n in l1_counter if n != 'other']),
        }

    fig, axes = plt.subplots(1, 5, figsize=(8.5, 5.0))

    for idx, (view_name, vd) in enumerate(view_data.items()):
        ax = axes[idx]
        ax.set_xlim(0, 5.5)
        ax.set_ylim(-1.0, 10)
        ax.axis('off')

        color = vd['color']
        ax.text(2.75, 9.7, view_name, ha='center', va='top', fontsize=7.5,
                fontweight='bold', color=color)

        root_x, root_y = 2.75, 8.8
        ax.plot(root_x, root_y, 'o', color=color, markersize=5, zorder=5)

        l1_items = vd['L1']
        n_l1 = len(l1_items)
        l1_spacing = min(1.2, 8.5 / max(n_l1, 1))

        for i, (l1_name, l1_count) in enumerate(l1_items):
            y = 8.0 - i * l1_spacing
            x = 0.3

            ax.plot([root_x, x + 0.3], [root_y, y], '-', color=color, linewidth=0.6, alpha=0.5)
            ax.plot(x + 0.3, y, 'o', color=color, markersize=3, zorder=5)

            label = pretty(l1_name)
            count_str = f'{l1_count:,}' if l1_count >= 1000 else str(l1_count)
            ax.text(x + 0.5, y, f'{label} ({count_str})', fontsize=4.8, va='center',
                    color='#2c3e50', fontweight='medium')

            if l1_name in vd['L2']:
                for j, l2_name in enumerate(vd['L2'][l1_name]):
                    cy = y - 0.30 * (j + 1)
                    cx = 3.0
                    ax.plot([x + 0.35, cx], [y - 0.05, cy], '-', color=color,
                            linewidth=0.3, alpha=0.3)
                    l2_label = pretty(l2_name)
                    if len(l2_label) > 22:
                        l2_label = l2_label[:21] + '…'
                    ax.text(cx + 0.1, cy, l2_label, fontsize=4.0, va='center',
                            color=COLORS['gray'])

        ax.text(2.75, -0.6, f'{vd["total_l1"]} categories', ha='center', va='top',
                fontsize=4.5, color=COLORS['gray'], style='italic')

    fig.suptitle('Five hierarchical views over 814,339 claims',
                 fontsize=10, fontweight='bold', color=COLORS['primary'], y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(PAPER_DIR / 'fig3_multiview.pdf')
    fig.savefig(PAPER_DIR / 'fig3_multiview.png')
    plt.close()
    print("  Fig 3: Multi-view hierarchy (from 100K data)", flush=True)


def fig4_canonical_coverage(data):
    """Figure 4: Canonical knowledge coverage heatmap."""
    if 'canonical_map' not in data:
        print("  Fig 4: SKIPPED (no canonical map data)", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), gridspec_kw={'width_ratios': [2, 1]})

    canon = data['canonical_map']
    entries = canon.get('canonical_entries', [])

    ax = axes[0]
    cat_counts = Counter(e.get('category', 'unknown') for e in entries)
    cats = sorted(cat_counts.keys(), key=lambda x: cat_counts[x], reverse=True)
    counts = [cat_counts[c] for c in cats]
    short_cats = [smart_title(c)[:20] for c in cats]

    colors_list = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    bars = ax.barh(short_cats[::-1], counts[::-1], color=colors_list[::-1], height=0.7, edgecolor='white')
    ax.set_xlabel('Number of canonical entries')
    ax.set_title('a  Canonical knowledge map (302 entries)', loc='left', fontweight='bold', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), ha='left', va='center', fontsize=6)

    ax = axes[1]
    if 'coverage' in data:
        cov = data['coverage']
        covered = cov.get('entries_with_landmark_papers', 47)
        gaps = cov.get('entries_without_papers', 255)
    else:
        covered = 47
        gaps = 255

    sizes = [covered, gaps]
    labels = [f'Covered\n({covered})', f'Gaps\n({gaps})']
    colors_pie = [COLORS['green'], '#f0f0f0']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       explode=explode, autopct='%1.0f%%',
                                       startangle=90, textprops={'fontsize': 7})
    autotexts[0].set_color('white')
    autotexts[0].set_fontweight('bold')
    ax.set_title('b  Coverage by\nlandmark papers', loc='center', fontweight='bold', fontsize=8)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / 'fig4_canonical_coverage.pdf')
    fig.savefig(PAPER_DIR / 'fig4_canonical_coverage.png')
    plt.close()
    print("  Fig 4: Canonical coverage", flush=True)


def fig5_frontier_analysis(data):
    """Figure 5: Frontier and gap analysis — claim distribution and temporal coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    batch_ext = data.get('batch_extractions', {})
    paper_meta = data.get('paper_meta', {})

    # Panel A: Claim type distribution as donut
    ax = axes[0]
    if batch_ext:
        type_counts = Counter()
        for doi, claims in batch_ext.items():
            for c in claims:
                type_counts[c.get('claim_type', 'unknown')] += 1

        canonical_types = ['property', 'method', 'mechanism', 'comparison', 'computational_result', 'reaction']
        other_count = sum(v for k, v in type_counts.items() if k not in canonical_types)
        plot_types = [smart_title(t) for t in canonical_types] + ['Other']
        plot_counts = [type_counts.get(t, 0) for t in canonical_types] + [other_count]
        plot_colors = [CLAIM_COLORS.get(t, COLORS['gray']) for t in canonical_types] + [COLORS['gray']]

        total = sum(plot_counts)
        wedges, texts, autotexts = ax.pie(
            plot_counts, labels=plot_types, colors=plot_colors,
            autopct=lambda pct: f'{pct:.0f}%' if pct > 4 else '',
            startangle=90, textprops={'fontsize': 6}, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=0.5)
        )
        for at in autotexts:
            at.set_fontsize(5.5)
            at.set_color('white')
            at.set_fontweight('bold')
        ax.set_title(f'a  Claim types (n={total:,})', loc='left', fontweight='bold', fontsize=8, x=-0.1)

    # Panel B: Claims per source paper
    ax = axes[1]
    if batch_ext:
        claims_per_paper = [len(claims) for claims in batch_ext.values()]
        bins = range(0, min(max(claims_per_paper) + 2, 25))
        ax.hist(claims_per_paper, bins=bins, color=COLORS['teal'], edgecolor='white', linewidth=0.3, alpha=0.8)
        mean_cpp = np.mean(claims_per_paper)
        ax.axvline(x=mean_cpp, color=COLORS['accent'], linestyle='--', linewidth=0.8)
        ax.text(mean_cpp + 0.3, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 100,
                f'mean={mean_cpp:.1f}', fontsize=6, color=COLORS['accent'])
        ax.set_xlabel('Claims per paper')
        ax.set_ylabel('Number of papers')
        ax.set_title('b  Extraction depth', loc='left', fontweight='bold', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel C: Temporal distribution
    ax = axes[2]
    if batch_ext and paper_meta:
        years = []
        for doi in batch_ext:
            pm = paper_meta.get(doi, {})
            y = pm.get('year')
            if y and isinstance(y, (int, float)) and 1990 < y < 2030:
                years.append(int(y))

        if years:
            year_counts = Counter(years)
            yr_range = range(min(year_counts.keys()), max(year_counts.keys()) + 1)
            yr_vals = [year_counts.get(y, 0) for y in yr_range]

            ax.fill_between(yr_range, yr_vals, alpha=0.3, color=COLORS['primary'])
            ax.plot(list(yr_range), yr_vals, color=COLORS['primary'], linewidth=1.2)
            ax.set_xlabel('Publication year')
            ax.set_ylabel('Papers')
            ax.set_title('c  Temporal coverage', loc='left', fontweight='bold', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / 'fig5_frontiers.pdf')
    fig.savefig(PAPER_DIR / 'fig5_frontiers.png')
    plt.close()
    print("  Fig 5: Frontier analysis", flush=True)


def fig6_topdown_vs_bottomup(data):
    """Figure 6: Top-down canonical map vs bottom-up extraction — the dual approach."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    td_box = FancyBboxPatch((0.2, 3.5), 4.3, 2.0,
                             boxstyle="round,pad=0.15", facecolor='#eaf2f8',
                             edgecolor=COLORS['primary'], linewidth=1.2)
    ax.add_patch(td_box)
    ax.text(2.35, 5.2, 'Top-Down', ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])
    ax.text(2.35, 4.7, '302 canonical entries', ha='center', fontsize=7, color=COLORS['primary'])
    ax.text(2.35, 4.3, '"What chemistry SHOULD contain"', ha='center', fontsize=6, style='italic', color=COLORS['gray'])
    td_items = ['Named reactions (68)', 'Techniques (31)', 'Biochem pathways (32)',
                'Materials classes (27)', 'Catalysis methods (20)']
    for i, item in enumerate(td_items):
        ax.text(1.0, 3.9 - i * 0.25, f'• {item}', fontsize=5.5, color=COLORS['secondary'])

    bu_box = FancyBboxPatch((5.5, 3.5), 4.3, 2.0,
                             boxstyle="round,pad=0.15", facecolor='#fef9e7',
                             edgecolor=COLORS['orange'], linewidth=1.2)
    ax.add_patch(bu_box)
    ax.text(7.65, 5.2, 'Bottom-Up', ha='center', fontsize=10, fontweight='bold', color=COLORS['orange'])
    ax.text(7.65, 4.7, '814,339 extracted claims', ha='center', fontsize=7, color=COLORS['orange'])
    ax.text(7.65, 4.3, '"What papers ACTUALLY report"', ha='center', fontsize=6, style='italic', color=COLORS['gray'])
    bu_items = ['104,249 papers processed', '7.8 claims/paper avg', '7 claim types',
                '14 chemistry subfields', '5 views']
    for i, item in enumerate(bu_items):
        ax.text(6.3, 3.9 - i * 0.25, f'• {item}', fontsize=5.5, color=COLORS['orange'])

    ax.annotate('', xy=(5, 1.8), xytext=(2.35, 3.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
    ax.annotate('', xy=(5, 1.8), xytext=(7.65, 3.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5))

    ct_box = FancyBboxPatch((2.5, 0.5), 5.0, 1.5,
                             boxstyle="round,pad=0.15", facecolor='#e8f8f5',
                             edgecolor=COLORS['green'], linewidth=1.5)
    ax.add_patch(ct_box)
    ax.text(5, 1.6, 'AskChem Index', ha='center', fontsize=10, fontweight='bold', color=COLORS['green'])
    ax.text(5, 1.2, 'Comprehensive + Grounded', ha='center', fontsize=7, color=COLORS['green'])
    ax.text(5, 0.8, 'Canonical skeleton filled with extracted evidence', ha='center', fontsize=6,
            style='italic', color=COLORS['gray'])

    ax.text(3.2, 2.8, 'Coverage\nguarantee', ha='center', fontsize=6, color=COLORS['primary'], rotation=35)
    ax.text(6.8, 2.8, 'Empirical\nevidence', ha='center', fontsize=6, color=COLORS['orange'], rotation=-35)

    fig.savefig(PAPER_DIR / 'fig6_dual_approach.pdf')
    fig.savefig(PAPER_DIR / 'fig6_dual_approach.png')
    plt.close()
    print("  Fig 6: Top-down vs bottom-up", flush=True)


def fig_ed1_claim_examples(data):
    """Extended Data Figure 1: Example claims from each type."""
    batch_ext = data.get('batch_extractions', {})
    if not batch_ext:
        print("  Fig ED1: SKIPPED (no claims)", flush=True)
        return

    examples = {}
    for doi, claims in batch_ext.items():
        for c in claims:
            ct = c.get('claim_type', 'unknown')
            if ct not in examples and ct in CLAIM_COLORS:
                c['_doi'] = doi
                examples[ct] = c
            if len(examples) >= 7:
                break
        if len(examples) >= 7:
            break

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.axis('off')

    headers = ['Claim Type', 'Source DOI', 'Verbatim Quote (truncated)']
    col_widths = [0.12, 0.30, 0.58]

    y = 0.95
    for i, h in enumerate(headers):
        x = sum(col_widths[:i]) + col_widths[i] / 2
        ax.text(x, y, h, ha='center', va='top', fontsize=7, fontweight='bold',
                color='white', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['primary'], edgecolor='none'))

    for idx, (ct, c) in enumerate(examples.items()):
        y = 0.88 - idx * 0.12
        bg = COLORS['light_bg'] if idx % 2 == 0 else 'white'

        ax.axhspan(y - 0.05, y + 0.05, facecolor=bg, transform=ax.transAxes, zorder=0)

        ct_display = smart_title(ct)
        color = CLAIM_COLORS.get(ct, COLORS['gray'])
        ax.text(col_widths[0] / 2, y, ct_display, ha='center', va='center',
                fontsize=6.5, fontweight='bold', color=color, transform=ax.transAxes)

        doi_str = c.get('_doi', '')[:45] + '...' if len(c.get('_doi', '')) > 45 else c.get('_doi', '')
        ax.text(col_widths[0] + col_widths[1] / 2, y, doi_str, ha='center', va='center',
                fontsize=5.5, color='#2c3e50', transform=ax.transAxes, style='italic')

        quote = c.get('verbatim_quote', '')[:90] + '...'
        ax.text(col_widths[0] + col_widths[1] + col_widths[2] / 2, y, quote,
                ha='center', va='center', fontsize=5, color=COLORS['gray'],
                transform=ax.transAxes, wrap=True)

    fig.suptitle('Extended Data Table 1: Example claims from each type',
                 fontsize=9, fontweight='bold', color=COLORS['primary'])
    fig.savefig(PAPER_DIR / 'fig_ed1_claim_examples.pdf')
    fig.savefig(PAPER_DIR / 'fig_ed1_claim_examples.png')
    plt.close()
    print("  Fig ED1: Claim examples", flush=True)


def fig_ed2_extraction_quality(data):
    """Extended Data Figure 2: Extraction quality — v2 deep vs v3 abstract at 100K scale."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: Claims per paper histogram for deep extraction
    ax = axes[0]
    v2 = data.get('v2_extractions', [])
    if v2:
        deep_counts = [p.get('num_claims', 0) or len(p.get('stage2', {}).get('result', {}).get('claims', [])) for p in v2]
        ax.bar(range(len(deep_counts)), sorted(deep_counts, reverse=True),
               color=COLORS['primary'], width=0.7, edgecolor='white')
        ax.set_xlabel('Paper rank')
        ax.set_ylabel('Claims extracted')
        ax.axhline(y=np.mean(deep_counts), color=COLORS['accent'], linestyle='--', linewidth=0.8)
        ax.text(len(deep_counts) - 1, np.mean(deep_counts) + 0.5,
                f'mean={np.mean(deep_counts):.1f}', fontsize=6, color=COLORS['accent'], ha='right')
        ax.set_title('a  Deep extraction (GPT-5.4, 10 PDFs)', loc='left', fontweight='bold', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel B: Claims per paper histogram for 100K abstract extraction
    ax = axes[1]
    batch_ext = data.get('batch_extractions', {})
    if batch_ext:
        abstract_counts = [len(claims) for claims in batch_ext.values()]
        bins = range(0, min(max(abstract_counts) + 2, 30))
        ax.hist(abstract_counts, bins=bins, color=COLORS['teal'], edgecolor='white', linewidth=0.3, alpha=0.8)
        mean_val = np.mean(abstract_counts)
        ax.axvline(x=mean_val, color=COLORS['accent'], linestyle='--', linewidth=0.8)
        ax.text(mean_val + 0.3, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 100,
                f'mean={mean_val:.1f}', fontsize=6, color=COLORS['accent'])
        ax.set_xlabel('Claims per paper')
        ax.set_ylabel('Number of papers')
        ax.set_title('b  Abstract extraction (GPT-5-mini, 104K papers)', loc='left', fontweight='bold', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / 'fig_ed2_quality.pdf')
    fig.savefig(PAPER_DIR / 'fig_ed2_quality.png')
    plt.close()
    print("  Fig ED2: Extraction quality", flush=True)


def fig7_case_studies(data):
    """Figure 7: Three case studies showing AskChem in action."""
    batch_ext = data.get('batch_extractions', {})
    paper_meta = data.get('paper_meta', {})
    if not batch_ext:
        print("  Fig 7: SKIPPED (no claims)", flush=True)
        return

    all_claims = []
    for doi, claims in batch_ext.items():
        for c in claims:
            c['_doi'] = doi
            all_claims.append(c)

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          height_ratios=[1, 1, 1])

    # Case Study 1: Suzuki coupling
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    suzuki = [c for c in all_claims if 'suzuki' in json.dumps(c).lower()]
    suzuki_types = Counter(c.get('claim_type', 'unknown') for c in suzuki)
    suzuki_dois = set(c['_doi'] for c in suzuki)

    types = sorted(suzuki_types.keys(), key=lambda x: suzuki_types[x], reverse=True)[:6]
    counts = [suzuki_types[t] for t in types]
    colors_list = [CLAIM_COLORS.get(t, COLORS['gray']) for t in types]
    short = [t.replace('_', '\n') for t in types]

    ax_left.barh(short[::-1], counts[::-1], color=colors_list[::-1], height=0.6, edgecolor='white')
    ax_left.set_xlabel('Claims')
    ax_left.set_title('Case Study 1: "What do we know\nabout Suzuki coupling?"',
                       loc='left', fontweight='bold', fontsize=7.5, color=COLORS['primary'])
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    for i, (cnt, name) in enumerate(zip(counts[::-1], short[::-1])):
        ax_left.text(cnt + max(counts)*0.02, i, str(cnt), va='center', fontsize=6, color=COLORS['gray'])

    suzuki_years = Counter()
    for c in suzuki:
        pm = paper_meta.get(c['_doi'], {})
        y = pm.get('year')
        if y and 2000 <= y <= 2026:
            suzuki_years[y] += 1

    if suzuki_years:
        yrs = sorted(suzuki_years.keys())
        vals = [suzuki_years[y] for y in yrs]
        ax_right.bar(yrs, vals, color=COLORS['secondary'], width=0.7, edgecolor='white')
        ax_right.set_xlabel('Year')
        ax_right.set_ylabel('Claims')
        ax_right.set_title(f'{len(suzuki):,} claims from {len(suzuki_dois):,} papers',
                           loc='left', fontsize=7, color=COLORS['gray'])
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)

    # Case Study 2: MOF + photocatalysis + CO2
    ax_left2 = fig.add_subplot(gs[1, :])

    mof_all = [c for c in all_claims if 'mof' in json.dumps(c).lower() or 'metal-organic framework' in json.dumps(c).lower()]
    photo_all = [c for c in all_claims if 'photocataly' in json.dumps(c).lower()]
    co2_all = [c for c in all_claims if 'co2' in json.dumps(c).lower() or 'carbon dioxide' in json.dumps(c).lower()]
    mof_photo = [c for c in all_claims if ('mof' in json.dumps(c).lower() or 'metal-organic framework' in json.dumps(c).lower()) and 'photocataly' in json.dumps(c).lower()]
    mof_co2 = [c for c in all_claims if ('mof' in json.dumps(c).lower() or 'metal-organic framework' in json.dumps(c).lower()) and ('co2' in json.dumps(c).lower() or 'carbon dioxide' in json.dumps(c).lower())]

    ax_left2.set_xlim(0, 10)
    ax_left2.set_ylim(0, 3)
    ax_left2.axis('off')
    ax_left2.set_title('Case Study 2: Cross-domain discovery — "Find MOF-based photocatalysts for CO$_2$ reduction"',
                        loc='left', fontweight='bold', fontsize=7.5, color=COLORS['primary'], y=1.0)

    steps = [
        ('Substance:\nMOFs', len(mof_all), COLORS['secondary']),
        ('Technique:\nPhotocatalysis', len(photo_all), COLORS['orange']),
        ('Reaction:\nCO$_2$ reduction', len(co2_all), COLORS['green']),
        ('MOF +\nPhotocatalysis', len(mof_photo), COLORS['purple']),
        ('MOF +\nCO$_2$', len(mof_co2), COLORS['accent']),
    ]

    max_count = max(s[1] for s in steps) if steps else 1
    x_positions = [0.5, 2.3, 4.1, 6.2, 8.3]
    for i, (label, count, color) in enumerate(steps):
        x = x_positions[i]
        box_h = min(2.0, max(0.4, count / max_count * 2.0))
        y_center = 1.5
        box = FancyBboxPatch((x - 0.7, y_center - box_h/2), 1.4, box_h,
                              boxstyle="round,pad=0.08", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1)
        ax_left2.add_patch(box)
        count_str = f'{count:,}'
        ax_left2.text(x, y_center + 0.1, count_str, ha='center', va='center',
                      fontsize=9, fontweight='bold', color=color)
        ax_left2.text(x, y_center - box_h/2 - 0.15, label, ha='center', va='top',
                      fontsize=5.5, color=color)

        if i < len(steps) - 1:
            ax_left2.annotate('', xy=(x_positions[i+1] - 0.75, y_center),
                             xytext=(x + 0.75, y_center),
                             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8))

    mof_co2_papers = len(set(c['_doi'] for c in mof_co2))
    ax_left2.text(5, 0.15, f'Agent narrows from {len(mof_all):,} MOF claims to {len(mof_co2):,} at the MOF+CO$_2$ intersection — '
                  f'from {mof_co2_papers:,} papers',
                  ha='center', fontsize=6, style='italic', color=COLORS['gray'])

    # Case Study 3: ML in chemistry
    ax_left3 = fig.add_subplot(gs[2, 0])
    ax_right3 = fig.add_subplot(gs[2, 1])

    ml_claims = [c for c in all_claims if any(w in json.dumps(c).lower() for w in
                 ['machine learning', 'neural network', 'deep learning', 'graph neural'])]
    ml_years = Counter()
    for c in ml_claims:
        pm = paper_meta.get(c['_doi'], {})
        y = pm.get('year')
        if y and 2015 <= y <= 2026:
            ml_years[y] += 1

    if ml_years:
        yrs = sorted(ml_years.keys())
        vals = [ml_years[y] for y in yrs]
        ax_left3.fill_between(yrs, vals, alpha=0.3, color=COLORS['purple'])
        ax_left3.plot(yrs, vals, color=COLORS['purple'], linewidth=1.5, marker='o', markersize=3)
        ax_left3.set_xlabel('Year')
        ax_left3.set_ylabel('Claims')
        ax_left3.set_title('Case Study 3: Frontier detection —\n"Where is ML in chemistry surging?"',
                           loc='left', fontweight='bold', fontsize=7.5, color=COLORS['primary'])
        ax_left3.spines['top'].set_visible(False)
        ax_left3.spines['right'].set_visible(False)

        peak_year = max(ml_years, key=ml_years.get)
        ax_left3.annotate(f'Peak: {peak_year}', xy=(peak_year, ml_years[peak_year]),
                         xytext=(peak_year - 1.5, ml_years[peak_year] * 0.7),
                         fontsize=6, color=COLORS['accent'],
                         arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=0.5))

    ml_types = Counter(c.get('claim_type', 'unknown') for c in ml_claims)
    types_ml = sorted(ml_types.keys(), key=lambda x: ml_types[x], reverse=True)[:6]
    counts_ml = [ml_types[t] for t in types_ml]
    colors_ml = [CLAIM_COLORS.get(t, COLORS['gray']) for t in types_ml]
    short_ml = [smart_title(t) for t in types_ml]

    if counts_ml:
        wedges, texts, autotexts = ax_right3.pie(
            counts_ml, labels=short_ml, colors=colors_ml,
            autopct=lambda pct: f'{pct:.0f}%' if pct > 8 else '',
            startangle=90, textprops={'fontsize': 5.5},
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=0.5)
        )
        for at in autotexts:
            at.set_fontsize(5)
            at.set_color('white')
            at.set_fontweight('bold')
    ml_dois = set(c['_doi'] for c in ml_claims)
    ax_right3.set_title(f'{len(ml_claims):,} claims from {len(ml_dois):,} papers',
                        loc='left', fontsize=7, color=COLORS['gray'])

    fig.savefig(PAPER_DIR / 'fig7_case_studies.pdf')
    fig.savefig(PAPER_DIR / 'fig7_case_studies.png')
    plt.close()
    print("  Fig 7: Case studies", flush=True)


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...", flush=True)
    data = load_data()

    print(f"\nData loaded:", flush=True)
    print(f"  V2 extractions: {len(data.get('v2_extractions', []))} papers", flush=True)
    print(f"  Scaled claims (pilot): {len(data.get('scaled_claims', []))} papers", flush=True)
    print(f"  Canonical entries: {len(data.get('canonical_map', {}).get('canonical_entries', []))}", flush=True)
    print(f"  Batch extractions: {len(data.get('batch_extractions', {})):,} papers", flush=True)
    print(f"  Batch classifications: {len(data.get('batch_classifications', {})):,} claims", flush=True)

    print("\nGenerating figures...", flush=True)
    fig1_architecture(data)
    fig2_extraction_comparison(data)
    fig3_multiview_hierarchy(data)
    fig4_canonical_coverage(data)
    fig5_frontier_analysis(data)
    fig6_topdown_vs_bottomup(data)
    fig7_case_studies(data)
    fig_ed1_claim_examples(data)
    fig_ed2_extraction_quality(data)

    print(f"\nAll figures saved to {PAPER_DIR}/", flush=True)
    for f in sorted(PAPER_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
