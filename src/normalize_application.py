"""
Normalize the by_application taxonomy across all claims.

Usage:
    python src/normalize_application.py              # dry-run
    python src/normalize_application.py --apply      # apply changes
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
VIEW_ID = "by_application"

CANONICAL_L1 = [
    "energy",
    "biomedicine",
    "environmental",
    "sensing",
    "catalysis",
    "materials_science",
    "separations",
    "synthetic_chemistry",
    "electronics",
    "food_science",
    "coatings",
]

CANONICAL_SET = set(CANONICAL_L1)

# ── Direct L1 remapping ─────────────────────────────────────────────────────

DIRECT_L1_MAP = {
    # Energy cluster
    "energy_and_renewables": "energy",
    "energy_conversion": "energy",
    "energy_storage": "energy",
    "energy": "energy",

    # Biomedicine cluster
    "biological_and_biotechnological_chemistry": "biomedicine",
    "biological_and_biotecnological_chemistry": "biomedicine",
    "pharmaceuticals_and_drug_discovery": "biomedicine",
    "medicine": "biomedicine",
    "pharmaceutical": "biomedicine",
    "biotechnology": "biomedicine",
    "biological_applications": "biomedicine",
    "structural_biology_and_bioassays": "biomedicine",

    # Environmental cluster
    "environmental_and_sustainable_chemistry": "environmental",
    "environmental": "environmental",
    "atmospheric_science": "environmental",

    # Direct mappings
    "catalysis": "catalysis",
    "catalytic": "catalysis",
    "materials_chemistry": "materials_science",
    "materials_science": "materials_science",
    "separations_and_processes": "separations",
    "separation": "separations",
    "synthetic_chemistry": "synthetic_chemistry",
    "synthesis_methodology": "synthetic_chemistry",
    "electronics": "electronics",
    "food_and_nutrition_chemistry": "food_science",
    "food_science": "food_science",
    "agriculture": "food_science",
    "coatings_and_surfaces": "coatings",
    "sensing": "sensing",

    # L1s needing L2-based routing
    "analytical_and_characterization": None,
    "computational_and_theoretical_chemistry": None,
    "computational_modeling": None,
    "structural_characterization": None,
    "other": None,

    # Junk
    "not_applicable": None,
    "by_application": None,
    "polymerization": None,
    "computational_chemistry": None,
    "microscopy_and_imaging": None,
}

# ── L2-keyword routing ──────────────────────────────────────────────────────

SENSING_KW = ["sensing", "sensor", "biosens", "chemosens", "detection",
              "fluorescent_prob", "fluorescence_sens", "electrochemical_sens",
              "optical_sens", "gas_sens", "immunosens", "aptasens",
              "colorimetr", "lateral_flow", "point_of_care", "diagnostic",
              "biomarker", "analyt"]

ENERGY_KW = ["energy", "battery", "solar", "photovoltaic", "fuel_cell",
             "supercapacitor", "hydrogen", "water_splitting", "electrolys",
             "thermoelectric", "piezoelectric", "grid", "renewable"]

BIOMEDICINE_KW = ["drug", "therapeut", "medicin", "biomed", "tissue_engineer",
                  "regenerat", "imaging", "theranost", "nanomedicin",
                  "antimicrob", "anticancer", "antitumor", "antivir",
                  "antibacter", "cytotox", "biocompat", "implant",
                  "gene_deliver", "gene_therap", "vaccine",
                  "enzymolog", "structural_biology", "biocataly",
                  "protein_engineer", "metabolic_engineer"]

ENVIRONMENTAL_KW = ["water_treatment", "remediat", "pollut", "wastewater",
                    "atmospheric", "air_quality", "carbon_capture",
                    "bioremediat", "green_chem", "sustainab",
                    "biodegradab", "recycl", "waste", "decontaminat",
                    "climate", "emission"]

CATALYSIS_KW = ["catalys", "catalyt"]

MATERIALS_KW = ["optoelectronic", "nanomaterial", "materials_synth",
                "functional_material", "polymer_material", "porous_material",
                "composite", "coating", "thin_film", "crystal_engineer",
                "materials_design", "materials_character"]

SEPARATIONS_KW = ["separation", "adsorption", "membrane", "chromatograph",
                  "extraction", "distillat", "filtrat", "purificat",
                  "gas_separat", "ion_exchange"]

SYNTH_KW = ["method_develop", "organic_synthesis", "asymmetric_synthesis",
            "total_synthesis", "natural_product_synthesis", "reaction_develop",
            "heterocycle_synthesis", "polymer_synthesis", "green_synthesis",
            "flow_chemistry"]

ELECTRONICS_KW = ["transistor", "dielectric", "memory", "led", "oled",
                  "display", "circuit", "photonic", "waveguide",
                  "quantum_comput"]

FOOD_KW = ["food", "nutrition", "nutraceutic", "packaging", "preservat",
           "crop", "soil", "agricultur", "fertili"]

NON_APPLICATION_KW = ["characteriz", "spectroscop", "microscop", "diffract",
                      "computational", "modeling", "simulation", "theory",
                      "density_functional", "molecular_dynamics",
                      "electronic_structure", "machine_learn", "cheminformat",
                      "method_benchmark", "literature_review", "publication",
                      "systematic_review", "metadata", "geochemist",
                      "astrochemist", "scholarly"]

L2_ROUTING_RULES = [
    (SENSING_KW, "sensing"),
    (ENERGY_KW, "energy"),
    (BIOMEDICINE_KW, "biomedicine"),
    (ENVIRONMENTAL_KW, "environmental"),
    (ELECTRONICS_KW, "electronics"),
    (FOOD_KW, "food_science"),
    (CATALYSIS_KW, "catalysis"),
    (SEPARATIONS_KW, "separations"),
    (SYNTH_KW, "synthetic_chemistry"),
    (MATERIALS_KW, "materials_science"),
    (NON_APPLICATION_KW, None),
]

L2_ROUTED_L1S = {
    "analytical_and_characterization",
    "computational_and_theoretical_chemistry",
    "computational_modeling",
    "structural_characterization",
    "other",
}


def _normalize_seg(seg: str) -> str:
    return seg.strip().lower().replace('-', '_').replace(' ', '_')


def _route_by_l2(old_l1: str, l2: str) -> str | None:
    l2_lower = l2.lower()

    if old_l1 == "computational_and_theoretical_chemistry":
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return None  # most are techniques, not applications

    if old_l1 == "computational_modeling":
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return None

    if old_l1 == "structural_characterization":
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return None

    # analytical_and_characterization: route sensing to sensing, drop the rest
    if old_l1 == "analytical_and_characterization":
        for keywords, target in L2_ROUTING_RULES:
            for kw in keywords:
                if kw in l2_lower:
                    return target
        return None  # characterization is a technique, not an application

    # other
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
        'view_id': VIEW_ID, 'path': '', 'name': 'By Application', 'level': 0,
        'claim_count': total_claims, 'children': sorted(root_children), 'claim_ids': [],
    }
    batch.append((
        VIEW_ID, '', 'By Application', 0, total_claims,
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
    parser = argparse.ArgumentParser(description="Normalize by_application taxonomy")
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
                    print(f"      {l2:50s} {c:>6,}")
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
