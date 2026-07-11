"""
Normalize the by_reaction_type taxonomy across all claims.

Replaces the fragmented dual-taxonomy (abstract + deep-PDF) with a single
unified 20-category L1 taxonomy, applies L2 fuzzy clustering, and rebuilds
the tree_nodes for by_reaction_type.

Usage:
    python src/normalize_reaction_type.py              # dry-run (stats only)
    python src/normalize_reaction_type.py --apply      # apply changes to DB
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from askchem.display import smart_title

DB_PATH = Path(__file__).parent.parent / "chemtree.db"
VIEW_ID = "by_reaction_type"

# ── Unified L1 taxonomy ─────────────────────────────────────────────────────

CANONICAL_L1 = [
    "catalysis",
    "electrocatalysis",
    "photocatalysis",
    "synthesis",
    "polymerization",
    "self_assembly",
    "oxidation",
    "reduction",
    "coupling",
    "degradation",
    "biochemistry",
    "adsorption",
    "surface_modification",
    "substitution",
    "condensation",
    "hydrolysis",
    "radical",
    "cycloaddition",
    "rearrangement",
    "thermochemistry",
]

CANONICAL_SET = set(CANONICAL_L1)

# ── Direct L1 remapping ─────────────────────────────────────────────────────
# Old L1 -> new L1 (or None to drop from this view)

DIRECT_L1_MAP = {
    # Deep-PDF L1s that stay as-is
    "electrocatalysis": "electrocatalysis",
    "photocatalysis": "photocatalysis",
    "oxidation": "oxidation",
    "reduction": "reduction",
    "coupling": "coupling",
    "self_assembly": "self_assembly",
    "adsorption": "adsorption",
    "degradation": "degradation",
    "radical": "radical",
    "substitution": "substitution",
    "cycloaddition": "cycloaddition",
    "rearrangement": "rearrangement",
    "condensation": "condensation",
    "enzymatic": "biochemistry",
    "acid_base": "catalysis",

    # Abstract L1s with direct mapping
    "coupling_and_bond_formation": "coupling",
    "decomposition_and_degradation": "degradation",
    "biochemical_and_enzymatic_transformations": "biochemistry",
    "polymerization": "polymerization",
    "synthetic_methods": "synthesis",
    "thermochemical_and_energy_conversion": "thermochemistry",
    "functional_group_transformations": "substitution",
    "hydrolysis_and_condensation": None,  # route by L2

    # Junk / not a reaction
    "not_applicable": None,
    "by_reaction_type": None,
    "polymers_and_biopolymers_not_allowed_error": None,
    "synthesis": "synthesis",
    "reaction": None,
    "photoswitchable_reaction": "photocatalysis",
    "photoswitchable_reactions": "photocatalysis",
}

# ── L2-keyword routing for mega-categories and ambiguous L1s ────────────────
# Each entry: (keyword_list, target_L1)
# Checked in order; first match wins. None = drop from view.

ASSEMBLY_KW = ["assembly", "self_assembl", "supramolecular", "coordination_network",
               "mof", "cof", "framework", "gelation", "gel_formation"]
ADSORPTION_KW = ["adsorpt", "sorption", "capture", "uptake", "ion_exchange"]
ELECTRO_KW = ["electro", "battery", "galvanic", "fuel_cell", "orr", "her", "oer", "co2rr"]
PHOTO_KW = ["photocataly", "photodegra", "solar_fuel", "photoredox", "photovoltai",
            "light_driven", "photochem", "photoisomer", "photoswitch"]
CATALYSIS_KW = ["catalys", "catalyt"]
POLYMER_KW = ["polymer", "crosslink", "curing", "copolymer"]
DEGRAD_KW = ["degrad", "decompos", "corros", "dissolut", "etch"]
SURFACE_KW = ["surface", "thin_film", "coating", "functionali", "grafting",
              "passivat", "deposition", "film_", "post_synthetic_modif"]
CRYSTAL_KW = ["crystal", "nucleat", "precipitat"]
SYNTHESIS_KW = ["synthesis", "fabricat", "prepar", "growth", "processing",
                "formation", "solvothermal", "hydrothermal"]
OXIDATION_KW = ["oxidat", "combustion", "ozonation"]
REDUCTION_KW = ["reduct", "hydrogenat"]
CONDENSATION_KW = ["condensat", "imine_form", "esterificat", "transesterif",
                   "schiff_base", "aldol", "knoevenagel", "sol_gel", "sol-gel",
                   "boronate_ester", "peptide_bond_form", "amide_form"]
HYDROLYSIS_KW = ["hydrolys", "hydrolyt", "saponif"]
BIOCHEM_KW = ["enzyme", "enzymat", "metabol", "biosynthes", "ferment",
              "protein", "biocataly", "gene_express", "nucleic_acid",
              "cell_signal", "proteolys", "bioconjug"]
RADICAL_KW = ["radical"]
COUPLING_KW = ["coupling", "c_c_bond", "c_n_bond", "cross_coupling",
               "suzuki", "heck", "sonogashira", "buchwald"]
SUBSTITUTION_KW = ["substitut", "nucleophil", "electrophil", "snar",
                   "functional_group"]
THERMO_KW = ["pyrolys", "thermochem", "thermal_decomp", "calcin"]
NON_REACTION_KW = ["characteriz", "spectroscop", "microscop", "computational",
                   "modeling", "simulation", "analysis", "measurement",
                   "imaging", "diffraction", "review", "benchmark", "method",
                   "study", "propert", "performance", "testing", "evaluation",
                   "literature", "cheminformat", "data_driven", "sensing",
                   "separation", "environmental_monitor", "instrumentat",
                   "sample_prep", "biological_activity", "structural_biology",
                   "atmospheric"]

L2_ROUTING_RULES = [
    (ASSEMBLY_KW, "self_assembly"),
    (ADSORPTION_KW, "adsorption"),
    (ELECTRO_KW, "electrocatalysis"),
    (PHOTO_KW, "photocatalysis"),
    (BIOCHEM_KW, "biochemistry"),
    (POLYMER_KW, "polymerization"),
    (DEGRAD_KW, "degradation"),
    (RADICAL_KW, "radical"),
    (COUPLING_KW, "coupling"),
    (SUBSTITUTION_KW, "substitution"),
    (OXIDATION_KW, "oxidation"),
    (REDUCTION_KW, "reduction"),
    (CONDENSATION_KW, "condensation"),
    (HYDROLYSIS_KW, "hydrolysis"),
    (THERMO_KW, "thermochemistry"),
    (CATALYSIS_KW, "catalysis"),
    (SURFACE_KW, "surface_modification"),
    (CRYSTAL_KW, "synthesis"),
    (SYNTHESIS_KW, "synthesis"),
    (NON_REACTION_KW, None),
]

# L1s that need L2-based routing
L2_ROUTED_L1S = {
    "materials_and_surface_processes",
    "catalysis",
    "electrochemical_processes",
    "photochemical_processes",
    "redox_reactions",
    "hydrolysis_and_condensation",
    "other",
}


def _normalize_seg(seg: str) -> str:
    return seg.strip().lower().replace('-', '_').replace(' ', '_')


def _route_by_l2(old_l1: str, l2: str) -> str | None:
    """Route a claim to a new L1 based on its L2 slug. Returns None to drop."""
    l2_lower = l2.lower()

    # Special handling for specific old L1s with known dominant targets
    if old_l1 == "electrochemical_processes":
        return "electrocatalysis"
    if old_l1 == "photochemical_processes":
        return "photocatalysis"
    if old_l1 == "redox_reactions":
        for kw in OXIDATION_KW:
            if kw in l2_lower:
                return "oxidation"
        for kw in REDUCTION_KW:
            if kw in l2_lower:
                return "reduction"
        for kw in ELECTRO_KW:
            if kw in l2_lower:
                return "electrocatalysis"
        for kw in PHOTO_KW:
            if kw in l2_lower:
                return "photocatalysis"
        return "reduction"  # default for redox
    if old_l1 == "hydrolysis_and_condensation":
        for kw in HYDROLYSIS_KW:
            if kw in l2_lower:
                return "hydrolysis"
        return "condensation"

    # General L2 keyword routing
    for keywords, target in L2_ROUTING_RULES:
        for kw in keywords:
            if kw in l2_lower:
                return target

    # Fallback for specific old L1s
    if old_l1 == "catalysis":
        return "catalysis"
    if old_l1 == "materials_and_surface_processes":
        return None  # unmatched materials claims aren't reactions
    if old_l1 == "other":
        return None  # drop unmatched "other" claims

    return None


def remap_claim_path(old_path: list) -> list | None:
    """Remap a single claim's by_reaction_type path to the new taxonomy.

    Returns the new path list, or None to remove this view from the claim.
    """
    if not old_path or not isinstance(old_path, list):
        return None

    old_l1 = _normalize_seg(old_path[0])

    if old_l1 in ('not_applicable', 'none', ''):
        return None

    # Direct mapping
    if old_l1 in DIRECT_L1_MAP:
        new_l1 = DIRECT_L1_MAP[old_l1]
        if new_l1 is None and old_l1 not in L2_ROUTED_L1S:
            return None
        if new_l1 is not None:
            new_path = [new_l1] + [_normalize_seg(s) for s in old_path[1:]]
            return [s for s in new_path if s not in ('not_applicable', 'none')]

    # L2-based routing
    if old_l1 in L2_ROUTED_L1S:
        l2 = _normalize_seg(old_path[1]) if len(old_path) >= 2 else ""
        new_l1 = _route_by_l2(old_l1, l2)
        if new_l1 is None:
            return None
        new_path = [new_l1] + [_normalize_seg(s) for s in old_path[1:]]
        return [s for s in new_path if s not in ('not_applicable', 'none')]

    # If old_l1 is already a canonical name, keep it
    if old_l1 in CANONICAL_SET:
        new_path = [_normalize_seg(s) for s in old_path]
        return [s for s in new_path if s not in ('not_applicable', 'none')]

    # Unknown L1 — drop
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
    """Build L2 merge map: {l1: {raw_l2: canonical_l2}} from claim-level L2 counts.

    Uses indexed lookups for exact/plural matches (O(1)) and only does expensive
    similarity checks against the top-N clusters by count.
    """
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

        if (li + 1) % 5 == 0 or li == len(l1_list) - 1:
            print(f"    Clustered {li+1}/{len(l1_list)} L1s "
                  f"({total_merges:,} merges so far)", flush=True)

    return merge_map


# ── Tree rebuild ─────────────────────────────────────────────────────────────

def rebuild_tree_nodes(conn, all_paths: list[tuple[str, list]]):
    """Rebuild tree_nodes for by_reaction_type from claim paths.

    all_paths: list of (claim_id, path_list) for claims that have a valid path.
    """
    c = conn.cursor()

    # Delete existing tree nodes for this view
    c.execute("DELETE FROM tree_nodes WHERE view_id = ?", (VIEW_ID,))

    # Aggregate: for each path prefix, count claims and collect claim_ids
    node_data: dict[str, dict] = {}  # path_str -> {claim_count, claim_ids, children}

    for claim_id, path in all_paths:
        for depth in range(len(path)):
            prefix = '/'.join(path[:depth + 1])
            if prefix not in node_data:
                node_data[prefix] = {'claim_count': 0, 'claim_ids': [], 'children': set()}
            # Only leaf nodes get claims
            if depth == len(path) - 1:
                node_data[prefix]['claim_count'] += 1
                if len(node_data[prefix]['claim_ids']) < 100:
                    node_data[prefix]['claim_ids'].append(claim_id)
            # Register as child of parent
            if depth > 0:
                parent = '/'.join(path[:depth])
                if parent in node_data:
                    node_data[parent]['children'].add(path[depth])

    # Also create root node
    root_children = set()
    total_claims = 0
    for claim_id, path in all_paths:
        if path:
            root_children.add(path[0])
            total_claims += 1

    # Propagate claim counts up the tree
    # Sort paths by depth (deepest first) to propagate upward
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
            'view_id': VIEW_ID,
            'path': path_str,
            'name': name,
            'level': level,
            'claim_count': nd['claim_count'],
            'children': children_list,
            'claim_ids': nd['claim_ids'],
        }
        batch.append((
            VIEW_ID, path_str, name, level,
            nd['claim_count'],
            json.dumps(children_list),
            json.dumps(nd['claim_ids']),
            json.dumps(data_json),
        ))

    # Root node
    root_data = {
        'view_id': VIEW_ID,
        'path': '',
        'name': 'By Reaction Type',
        'level': 0,
        'claim_count': total_claims,
        'children': sorted(root_children),
        'claim_ids': [],
    }
    batch.append((
        VIEW_ID, '', 'By Reaction Type', 0,
        total_claims,
        json.dumps(sorted(root_children)),
        json.dumps([]),
        json.dumps(root_data),
    ))

    # Insert in chunks
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
    parser = argparse.ArgumentParser(description="Normalize by_reaction_type taxonomy")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Path to chemtree.db")
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

    # ── Phase 1: Read all claims and remap L1 ───────────────────────────────

    print("Phase 1: Reading claims and remapping L1...", flush=True)
    t0 = time.time()

    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"  Total claims: {total:,}")

    old_l1_counts = Counter()
    new_l1_counts = Counter()
    dropped_count = 0
    kept_count = 0
    updates = []  # (claim_id, new_data_json, new_view_paths_json)
    l2_counts_per_l1 = defaultdict(Counter)  # new_l1 -> l2 -> count

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

            if old_path and isinstance(old_path, list) and len(old_path) >= 1:
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

            # Record the update
            if new_path is not None:
                vp[VIEW_ID] = new_path
            elif VIEW_ID in vp:
                del vp[VIEW_ID]
            data['view_paths'] = vp
            updates.append((claim_id, data, new_path))

        offset += batch_size
        elapsed = time.time() - t0
        print(f"  Processed {offset:,}/{total:,} ({elapsed:.0f}s)", flush=True)

    print(f"\n  L1 remapping complete: {kept_count:,} kept, {dropped_count:,} dropped")
    print(f"\n  Old L1 distribution:")
    for l1, c in old_l1_counts.most_common():
        print(f"    {l1:55s} {c:>8,}")
    print(f"\n  New L1 distribution:")
    for l1, c in new_l1_counts.most_common():
        print(f"    {l1:55s} {c:>8,}")

    # ── Phase 2: L2 fuzzy clustering ────────────────────────────────────────

    print("\nPhase 2: L2 fuzzy clustering...", flush=True)
    t1 = time.time()

    total_l2_before = sum(len(slugs) for slugs in l2_counts_per_l1.values())
    print(f"  Total unique L2 slugs before clustering: {total_l2_before:,}")

    l2_merge = build_l2_merge_map(l2_counts_per_l1)

    total_merges = sum(len(m) for m in l2_merge.values())
    print(f"  L2 merges: {total_merges:,}")

    # Show merge stats per L1
    for l1 in CANONICAL_L1:
        if l1 in l2_merge:
            print(f"    {l1}: {len(l2_merge[l1]):,} merges")

    # Apply L2 merges to updates
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

    # Count L2s after clustering
    final_l2_counts = defaultdict(Counter)
    for claim_id, data, path in updates:
        if path and len(path) >= 2:
            final_l2_counts[path[0]][path[1]] += 1
    total_l2_after = sum(len(slugs) for slugs in final_l2_counts.values())
    print(f"  Total unique L2 slugs after clustering: {total_l2_after:,}")
    print(f"  Reduction: {total_l2_before:,} -> {total_l2_after:,} ({100*(1-total_l2_after/max(total_l2_before,1)):.1f}%)")

    elapsed2 = time.time() - t1
    print(f"  Phase 2 done in {elapsed2:.0f}s")

    # ── Phase 3: Write to DB ────────────────────────────────────────────────

    if not args.apply:
        print("\n  DRY-RUN complete. Use --apply to write changes.")

        # Show sample L2s per L1
        print("\n  Sample L2 categories per L1:")
        for l1 in CANONICAL_L1:
            if l1 in final_l2_counts:
                top = final_l2_counts[l1].most_common(8)
                total_in_l1 = sum(final_l2_counts[l1].values())
                unique_in_l1 = len(final_l2_counts[l1])
                print(f"\n    {l1} ({total_in_l1:,} claims, {unique_in_l1:,} L2s):")
                for l2, c in top:
                    print(f"      {l2:50s} {c:>6,}")
        return

    print("\nPhase 3: Writing changes to database...", flush=True)
    t2 = time.time()

    # Update claims in batches
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
            print(f"  Updated {len(update_batch):,} claims...", flush=True)
            update_batch = []

    if update_batch:
        conn.executemany(
            "UPDATE claims SET view_paths = ?, data = ? WHERE claim_id = ?",
            update_batch
        )
        conn.commit()

    elapsed3 = time.time() - t2
    print(f"  Claims updated in {elapsed3:.0f}s")

    # ── Phase 4: Rebuild tree nodes ─────────────────────────────────────────

    print("\nPhase 4: Rebuilding tree nodes...", flush=True)
    t3 = time.time()

    all_paths = [(cid, path) for cid, _, path in updates if path]
    node_count = rebuild_tree_nodes(conn, all_paths)

    elapsed4 = time.time() - t3
    print(f"  Tree rebuilt: {node_count:,} nodes in {elapsed4:.0f}s")

    # ── Phase 5: Rebuild FTS ────────────────────────────────────────────────

    print("\nPhase 5: Rebuilding FTS index...", flush=True)
    t4 = time.time()

    from askchem.db import build_searchable_text

    conn.execute("DELETE FROM claims_fts")
    conn.commit()

    fts_batch = []
    offset = 0
    while True:
        rows = conn.execute(
            "SELECT data FROM claims LIMIT ? OFFSET ?",
            (50000, offset)
        ).fetchall()
        if not rows:
            break
        for row in rows:
            cdata = json.loads(row[0])
            searchable = build_searchable_text(cdata)
            fts_batch.append((
                cdata.get('claim_id', ''),
                cdata.get('claim_type', ''),
                cdata.get('source_paper_title', ''),
                cdata.get('verbatim_quote', ''),
                searchable,
            ))
        offset += 50000

    for i in range(0, len(fts_batch), 10000):
        conn.executemany(
            "INSERT INTO claims_fts (claim_id, claim_type, source_paper_title, verbatim_quote, searchable_text) VALUES (?,?,?,?,?)",
            fts_batch[i:i + 10000]
        )
        conn.commit()
        print(f"  FTS indexed: {min(i + 10000, len(fts_batch)):,}/{len(fts_batch):,}", flush=True)

    elapsed5 = time.time() - t4
    print(f"  FTS rebuilt in {elapsed5:.0f}s")

    # ── Summary ─────────────────────────────────────────────────────────────

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE in {total_elapsed:.0f}s")
    print(f"  Claims with by_reaction_type: {kept_count:,} / {total:,}")
    print(f"  Claims dropped from view: {dropped_count:,}")
    print(f"  L1 categories: {len(new_l1_counts)}")
    print(f"  L2 categories: {total_l2_after:,} (was {total_l2_before:,})")
    print(f"  Tree nodes: {node_count:,}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
