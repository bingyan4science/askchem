"""
Phase 1 of L3 cleanup: deterministic alias-snap.

For every claim where (L1, L2) has a CANONICAL_L3 whitelist but the L3 value
is off-whitelist, try to snap it to a canonical L3 using:
  1. Hand-curated alias map for well-known synonym families.
  2. Suffix strip / add (e.g. _coupling, _synthesis, _reaction(s)).
  3. Bidirectional substring containment, preferring the longest match
     and breaking ties by token Jaccard.
  4. If nothing matches, leave alone (Phase 2 / PAW will handle the rest).

Default is dry-run; pass --commit to actually update the DB and rebuild the
tree_nodes table.

Usage:
    python scripts/snap_l3_aliases.py                    # dry-run report
    python scripts/snap_l3_aliases.py --view by_reaction_type
    python scripts/snap_l3_aliases.py --commit           # apply + rebuild tree
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.taxonomy import CANONICAL_L3, get_canonical_l3, ALL_CONTENT_VIEWS

DB_PATH = ROOT / "chemtree.db"

# ── Hand-curated aliases for the highest-value clusters ────────────────────
# Format: (view_id, l1, l2) -> { off_whitelist_l3 -> canonical_l3 }
# Values MUST appear in the canonical L3 list for that parent.
CURATED_ALIASES: dict[tuple[str, str, str], dict[str, str]] = {
    ("by_reaction_type", "coupling", "cross_coupling"): {
        # All Suzuki variants → suzuki_miyaura (the canonical name)
        "suzuki_coupling":          "suzuki_miyaura",
        "suzuki_miyaura_coupling":  "suzuki_miyaura",
        "suzukimiyaura":            "suzuki_miyaura",
        "suzukimiyaura_coupling":   "suzuki_miyaura",
        "miyaura_borylation":       "suzuki_miyaura",
        # Sonogashira / Heck / Negishi / etc. — drop _coupling suffix
        "sonogashira_coupling":     "sonogashira",
        "heck_coupling":            "heck",
        "heck_reaction":            "heck",
        "mizoroki_heck":            "heck",
        "mizoroki_heck_reaction":   "heck",
        "negishi_coupling":         "negishi",
        "stille_coupling":          "stille",
        "hiyama_coupling":          "hiyama",
        "kumada_coupling":          "kumada",
        "ullmann_coupling":         "ullmann",
        "ullmann_reaction":         "ullmann",
        "buchwald_hartwig_amination": "buchwald_hartwig",
        "buchwald_hartwig_coupling":  "buchwald_hartwig",
        "decarboxylative_coupling":   "decarboxylative_cross_coupling",
        "dehydrogenative_coupling":   "dehydrogenative_cross_coupling",
        "photoredox_dual_catalysis":  "photoredox_metal_dual_catalysis",
        "photoredox_catalysis":       "photoredox_metal_dual_catalysis",
        "metallaphotoredox":          "photoredox_metal_dual_catalysis",
    },
    ("by_reaction_type", "catalysis", "heterogeneous_catalysis"): {
        "hydrogenation":             "hydrogenation_and_dehydrogenation",
        "dehydrogenation":           "hydrogenation_and_dehydrogenation",
        "selective_hydrogenation":   "hydrogenation_and_dehydrogenation",
        "transfer_hydrogenation":    "hydrogenation_and_dehydrogenation",
        "oxidation":                 "oxidation_reactions",
        "co_oxidation":              "oxidation_reactions",
        "selective_oxidation":       "oxidation_reactions",
        "partial_oxidation":         "oxidation_reactions",
        "advanced_oxidation":        "advanced_oxidant_activation_and_fenton_processes",
        "fenton_oxidation":          "advanced_oxidant_activation_and_fenton_processes",
        "peroxymonosulfate_activation": "advanced_oxidant_activation_and_fenton_processes",
        "persulfate_activation":     "advanced_oxidant_activation_and_fenton_processes",
        "co2_hydrogenation":         "co2_conversion_and_utilization",
        "co2_reduction":             "co2_conversion_and_utilization",
        "reverse_water_gas_shift":   "co2_conversion_and_utilization",
        "water_gas_shift":           "reforming_and_syngas_processes",
        "fischer_tropsch_synthesis": "reforming_and_syngas_processes",
        "fischer_tropsch":           "reforming_and_syngas_processes",
        "dry_reforming":             "reforming_and_syngas_processes",
        "steam_reforming":           "reforming_and_syngas_processes",
        "methane_reforming":         "reforming_and_syngas_processes",
        "ammonia_synthesis":         "nitrogen_related_reactions_and_ammonia_synthesis",
        "nitrogen_fixation":         "nitrogen_related_reactions_and_ammonia_synthesis",
        "ammonia_decomposition":     "nitrogen_related_reactions_and_ammonia_synthesis",
        "nox_reduction":             "selective_reduction_and_nox_control",
        "scr":                       "selective_reduction_and_nox_control",
        "selective_catalytic_reduction": "selective_reduction_and_nox_control",
        "methanol_to_olefins":       "c_c_coupling_and_hydrocarbon_synthesis",
        "methanol_to_hydrocarbons":  "c_c_coupling_and_hydrocarbon_synthesis",
        "methanol_synthesis":        "c_c_coupling_and_hydrocarbon_synthesis",
        "ethanol_synthesis":         "c_c_coupling_and_hydrocarbon_synthesis",
        "single_atom_catalysis":     "catalyst_design_and_active_site_engineering",
        "nanocatalysis":             "catalyst_design_and_active_site_engineering",
        "solid_acid_catalysis":      "catalyst_design_and_active_site_engineering",
        "zeolite_catalysis":         "catalyst_design_and_active_site_engineering",
        "supported_catalysis":       "catalyst_design_and_active_site_engineering",
        "active_site_engineering":   "catalyst_design_and_active_site_engineering",
        "photocatalysis":            "photocatalytic_reactions",
        "photocatalytic_oxidation":  "photocatalytic_reactions",
        "photocatalytic_reduction":  "photocatalytic_reactions",
        "electrocatalysis":          "electrocatalytic_reactions",
        "piezocatalysis":            "mechanochemical_and_piezocatalysis",
        "mechanocatalysis":          "mechanochemical_and_piezocatalysis",
    },
    ("by_reaction_type", "synthesis", "materials_synthesis"): {
        # MOF / COF / zeolite family → reticular_frameworks_and_zeolites
        "mof_synthesis":                   "reticular_frameworks_and_zeolites",
        "metal_organic_framework_synthesis": "reticular_frameworks_and_zeolites",
        "metal_organic_frameworks":        "reticular_frameworks_and_zeolites",
        "covalent_organic_frameworks":     "reticular_frameworks_and_zeolites",
        "cof_synthesis":                   "reticular_frameworks_and_zeolites",
        "zeolite_synthesis":               "reticular_frameworks_and_zeolites",
        # Thin-film / deposition family
        "thin_film_deposition":            "deposition_and_fabrication_methods",
        "thin_film_growth":                "deposition_and_fabrication_methods",
        "thin_film_fabrication":           "deposition_and_fabrication_methods",
        "atomic_layer_deposition":         "deposition_and_fabrication_methods",
        "chemical_vapor_deposition":       "deposition_and_fabrication_methods",
        "physical_vapor_deposition":       "deposition_and_fabrication_methods",
        "spin_coating":                    "deposition_and_fabrication_methods",
        "membrane_fabrication":            "deposition_and_fabrication_methods",
        "composite_fabrication":           "deposition_and_fabrication_methods",
        # Crystal / epitaxial growth → solution precipitation/sol-gel bucket
        "epitaxial_growth":                "solution_precipitation_and_sol-gel",
        "crystal_growth":                  "solution_precipitation_and_sol-gel",
        "crystallization":                 "solution_precipitation_and_sol-gel",
        "perovskite_crystallization":      "solution_precipitation_and_sol-gel",
        "perovskite_synthesis":            "solution_precipitation_and_sol-gel",
        "sol_gel_synthesis":               "solution_precipitation_and_sol-gel",
        "coprecipitation":                 "solution_precipitation_and_sol-gel",
        # Carbonization → chemical_activation_and_carbon_activation
        "carbonization":                   "chemical_activation_and_carbon_activation",
        "pyrolysis":                       "chemical_activation_and_carbon_activation",
        "carbon_activation":               "chemical_activation_and_carbon_activation",
        # Hydrothermal/solvothermal variants
        "hydrothermal_synthesis":          "hydrothermal_and_solvothermal",
        "solvothermal_synthesis":          "hydrothermal_and_solvothermal",
        # Generic / catch-all
        "general_synthesis":               "other",
        "scope_exploration":               "other",
        "nanocomposite_synthesis":         "other",
    },
    ("by_reaction_type", "self_assembly", "supramolecular_assembly"): {
        "host_guest_complexation":         "host_guest_complexes",
        "host_guest_chemistry":            "host_guest_complexes",
        "inclusion_complexation":          "host_guest_complexes",
        "vesicle_formation":               "amphiphilic_and_colloidal_assemblies",
        "micelle_formation":               "amphiphilic_and_colloidal_assemblies",
        "liposome_formation":              "amphiphilic_and_colloidal_assemblies",
        "colloidal_assembly":              "amphiphilic_and_colloidal_assemblies",
        "peptide_assembly":                "biomacromolecular_assemblies",
        "peptide_self_assembly":           "biomacromolecular_assemblies",
        "protein_assembly":                "biomacromolecular_assemblies",
        "dna_origami":                     "biomacromolecular_assemblies",
        "dna_assembly":                    "biomacromolecular_assemblies",
        "rotaxane_formation":              "mechanically_interlocked_molecules",
        "catenane_formation":              "mechanically_interlocked_molecules",
        "knot_synthesis":                  "mechanically_interlocked_molecules",
        "gelation":                        "dynamic_and_dissipative_assembly",
        "hydrogel_formation":              "dynamic_and_dissipative_assembly",
        "organogel_formation":             "dynamic_and_dissipative_assembly",
        "liquid_crystal_formation":        "functional_properties_and_emergent_behavior",
        "liquid_crystals":                 "functional_properties_and_emergent_behavior",
        "hydrogen_bonding":                "specific_non_covalent_interactions",
        "pi_stacking":                     "specific_non_covalent_interactions",
        "halogen_bonding":                 "specific_non_covalent_interactions",
        "metal_coordination":              "specific_non_covalent_interactions",
    },
    ("by_reaction_type", "synthesis", "total_synthesis"): {
        "natural_products":                "small_molecule_natural_products",
        "natural_product_synthesis":       "small_molecule_natural_products",
        "alkaloid_synthesis":              "alkaloids",
        "terpene_synthesis":               "terpenes_and_steroids",
        "steroid_synthesis":               "terpenes_and_steroids",
        "polyketide_synthesis":            "polyketides_and_macrolides",
        "macrolide_synthesis":             "polyketides_and_macrolides",
        "peptide_synthesis":               "peptides_and_peptidomimetics",
        "carbohydrate_synthesis":          "carbohydrates_and_glycosides",
        "glycoside_synthesis":             "carbohydrates_and_glycosides",
    },
    ("by_reaction_type", "oxidation", "atmospheric_oxidation"): {
        "radical_oxidation":               "hydroxyl_radical_oxidation",
        "oh_radical_oxidation":            "hydroxyl_radical_oxidation",
        "photooxidation":                  "photochemical_ozone_formation",
        "ozone_oxidation":                 "ozonolysis_and_criegee_intermediates",
        "no3_oxidation":                   "nitrate_radical_oxidation",
        "so2_oxidation":                   "sulfur_compound_oxidation_and_so2_to_sulfate",
        "aerosol_formation":               "secondary_organic_aerosol_formation",
        "soa_formation":                   "secondary_organic_aerosol_formation",
    },
}


def _norm(s: str) -> str:
    """Normalize a slug for comparison: lowercase, single-underscore, alpha-num."""
    s = (s or "").strip().lower()
    s = "".join(c if c.isalnum() else "_" for c in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


SUFFIXES_TO_TRY = (
    "_coupling", "_couplings",
    "_reaction", "_reactions",
    "_synthesis", "_syntheses",
    "_polymerization", "_polymerizations",
    "_process", "_processes",
    "_chemistry", "_method", "_methods",
    "_growth", "_formation",
)


def _strip_known_suffixes(s: str) -> list[str]:
    """Return progressively-stripped variants of s."""
    out = [s]
    for suf in SUFFIXES_TO_TRY:
        if s.endswith(suf) and len(s) > len(suf) + 2:
            out.append(s[: -len(suf)])
    return out


def _tokens(s: str) -> set[str]:
    return {t for t in s.split("_") if t and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def snap(off: str, canonical: list[str]) -> tuple[str | None, str]:
    """
    Try to map an off-whitelist L3 to a canonical L3.
    Returns (canonical_value or None, reason_tag).
    """
    off_n = _norm(off)
    canon_n = {_norm(c): c for c in canonical}

    # 0. exact equality (shouldn't happen — caller filters — but guard)
    if off_n in canon_n:
        return canon_n[off_n], "exact"

    # 1. suffix-strip on either side
    for v in _strip_known_suffixes(off_n):
        if v in canon_n:
            return canon_n[v], "suffix_strip_off"
    for c_n, c_orig in canon_n.items():
        for v in _strip_known_suffixes(c_n):
            if v == off_n:
                return c_orig, "suffix_strip_canon"

    # 2. bidirectional substring containment, prefer the longest unique match.
    # Skip "other" — it appears in every canonical list and would falsely
    # match anything containing the substring (e.g. "immunotherapy" contains
    # "other"). "other" can only be assigned via the curated map.
    contains_canon = []  # canon n inside off n  (e.g. suzuki_miyaura ⊂ suzuki_miyaura_coupling)
    inside_canon = []    # off n inside canon n  (e.g. hydrogenation ⊂ hydrogenation_and_dehydrogenation)
    off_tok = _tokens(off_n)
    for c_n, c_orig in canon_n.items():
        if c_n == "other":
            continue
        c_tok = _tokens(c_n)
        # Whole-token boundary check: require c_n to start at a word boundary
        # in off_n and end at a word boundary, so we don't catch e.g.
        # "other" inside "immunotherapy".
        if _whole_token_substring(c_n, off_n) and len(c_n) >= 5:
            contains_canon.append((c_orig, c_n, _jaccard(off_tok, c_tok)))
        elif _whole_token_substring(off_n, c_n) and len(off_n) >= 5:
            inside_canon.append((c_orig, c_n, _jaccard(off_tok, c_tok)))
    # Prefer "off contains canon" (canon name fully inside off name — the off
    # name is just a longer label for the same thing, e.g.
    # suzuki_miyaura_coupling ⊃ suzuki_miyaura). Require uniqueness: if
    # multiple distinct canonicals are inside the off name, refuse to guess.
    if contains_canon:
        contains_canon.sort(key=lambda t: (-len(t[1]), -t[2]))
        # If only one canonical matches, or the longest is strictly longer
        # than the runner-up, accept.
        unique = (len(contains_canon) == 1
                  or len(contains_canon[0][1]) > len(contains_canon[1][1]))
        if unique:
            return contains_canon[0][0], "contains_canon"
    if inside_canon:
        inside_canon.sort(key=lambda t: (-len(t[1]), -t[2]))
        # Refuse if ambiguous: more than one canonical contains the off name
        # (e.g. "polysaccharides" inside both structural_/storage_polysacch).
        if len(inside_canon) > 1:
            pass
        elif inside_canon[0][2] >= 0.4 or len(off_n) >= 7:
            return inside_canon[0][0], "inside_canon"

    # 3. high token Jaccard (also skip "other")
    best = None
    for c_n, c_orig in canon_n.items():
        if c_n == "other":
            continue
        j = _jaccard(off_tok, _tokens(c_n))
        if j >= 0.6 and (best is None or j > best[1]):
            best = (c_orig, j)
    if best:
        return best[0], "jaccard"

    return None, "no_match"


def _whole_token_substring(needle: str, haystack: str) -> bool:
    """
    True iff `needle` appears in `haystack` aligned to underscore boundaries
    (or at the start/end). E.g.:
      _whole_token_substring("suzuki_miyaura", "suzuki_miyaura_coupling") -> True
      _whole_token_substring("other", "immunotherapy")                    -> False
      _whole_token_substring("hydrogenation", "hydrogenation_and_dehydrogenation") -> True
    """
    if needle == haystack:
        return True
    if needle not in haystack:
        return False
    # Check every occurrence for word-boundary alignment.
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return False
        left_ok = (i == 0) or (haystack[i - 1] == "_")
        end = i + len(needle)
        right_ok = (end == len(haystack)) or (haystack[end] == "_")
        if left_ok and right_ok:
            return True
        start = i + 1


# ── DB sweep ────────────────────────────────────────────────────────────────

def build_snap_plan(views: list[str]) -> dict:
    """
    Returns:
      {
        view_id: {
          (l1, l2): {
            l3_off: { "to": l3_canonical, "reason": tag, "count": N }
            or { "to": None, "reason": "no_match", "count": N }
          }
        }
      }
    """
    plan: dict = defaultdict(lambda: defaultdict(dict))

    # Collect off-whitelist L3 counts per parent
    counts = defaultdict(Counter)  # (view, l1, l2) -> Counter(l3 -> N)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT view_paths FROM claims WHERE view_paths IS NOT NULL")
    for (vp_json,) in cur:
        try:
            vp = json.loads(vp_json)
        except Exception:
            continue
        for vid, p in vp.items():
            if vid not in views:
                continue
            if not isinstance(p, list) or len(p) < 3:
                continue
            l1, l2, l3 = p[0], p[1], p[2]
            allowed = get_canonical_l3(vid, l1, l2)
            if allowed is None or l3 in allowed:
                continue
            counts[(vid, l1, l2)][l3] += 1
    conn.close()

    for (vid, l1, l2), c in counts.items():
        canonical = get_canonical_l3(vid, l1, l2)
        curated = CURATED_ALIASES.get((vid, l1, l2), {})
        for l3_off, n in c.items():
            if l3_off in curated and curated[l3_off] in canonical:
                plan[vid][(l1, l2)][l3_off] = {
                    "to": curated[l3_off], "reason": "curated", "count": n,
                }
                continue
            mapped, reason = snap(l3_off, canonical)
            plan[vid][(l1, l2)][l3_off] = {
                "to": mapped, "reason": reason, "count": n,
            }
    return plan


def report(plan: dict) -> None:
    total_off = total_snapped = 0
    by_reason = Counter()
    for vid, parents in plan.items():
        v_off = v_snapped = 0
        for (l1, l2), m in parents.items():
            for l3_off, info in m.items():
                v_off += info["count"]
                if info["to"]:
                    v_snapped += info["count"]
                by_reason[info["reason"]] += info["count"]
        total_off += v_off
        total_snapped += v_snapped
        print(f"\n  {vid:24s}  off-whitelist {v_off:>9,}  →  snapped {v_snapped:>9,}  "
              f"({100 * v_snapped / max(v_off, 1):5.1f}%)")
    print("\nSnap reasons (by claim count):")
    for r, n in by_reason.most_common():
        print(f"    {r:25s}  {n:>9,}")
    print(f"\nTOTAL off-whitelist  : {total_off:,}")
    print(f"TOTAL snapped (Phase1): {total_snapped:,}  ({100 * total_snapped / max(total_off, 1):.1f}%)")
    print(f"REMAINING for Phase 2 : {total_off - total_snapped:,}")


def show_examples(plan: dict, limit: int = 30) -> None:
    print("\nTop snaps (Phase 1):")
    rows = []
    for vid, parents in plan.items():
        for (l1, l2), m in parents.items():
            for l3_off, info in m.items():
                if info["to"]:
                    rows.append((info["count"], vid, l1, l2, l3_off, info["to"], info["reason"]))
    rows.sort(reverse=True)
    print(f"  {'count':>7s}  {'view':22s}  {'L1/L2/L3_off':50s}  → {'L3_canon':40s}  ({'reason'})")
    for c, vid, l1, l2, off, to, r in rows[:limit]:
        path = f"{l1}/{l2}/{off}"
        print(f"  {c:>7,d}  {vid:22s}  {path[:50]:50s}  → {to[:40]:40s}  ({r})")

    print("\nTop unmatched (need Phase 2 / PAW):")
    rows = []
    for vid, parents in plan.items():
        for (l1, l2), m in parents.items():
            for l3_off, info in m.items():
                if not info["to"]:
                    rows.append((info["count"], vid, l1, l2, l3_off))
    rows.sort(reverse=True)
    print(f"  {'count':>7s}  {'view':22s}  {'L1/L2/L3_off'}")
    for c, vid, l1, l2, off in rows[:limit]:
        path = f"{l1}/{l2}/{off}"
        print(f"  {c:>7,d}  {vid:22s}  {path}")


def apply(plan: dict) -> None:
    print("\nApplying snaps to chemtree.db...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Build a flat mapping: (view, l1, l2, l3_off) -> l3_to
    flat: dict[tuple[str, str, str, str], str] = {}
    for vid, parents in plan.items():
        for (l1, l2), m in parents.items():
            for l3_off, info in m.items():
                if info["to"]:
                    flat[(vid, l1, l2, l3_off)] = info["to"]

    if not flat:
        print("  Nothing to apply.")
        conn.close()
        return

    cur = conn.execute("SELECT claim_id, view_paths, data FROM claims WHERE view_paths IS NOT NULL")
    updates: list[tuple[str, str, str]] = []
    BATCH = 5000
    n_changed = 0
    n_scanned = 0
    while True:
        rows = cur.fetchmany(20000)
        if not rows:
            break
        for row in rows:
            claim_id, vp_json, data_json = row[0], row[1], row[2]
            try:
                vp = json.loads(vp_json)
            except Exception:
                continue
            n_scanned += 1
            changed = False
            for vid, p in vp.items():
                if not isinstance(p, list) or len(p) < 3:
                    continue
                key = (vid, p[0], p[1], p[2])
                if key in flat:
                    p[2] = flat[key]
                    changed = True
            if changed:
                n_changed += 1
                new_vp = json.dumps(vp)
                # keep data JSON in sync
                try:
                    data = json.loads(data_json) if data_json else {}
                except Exception:
                    data = {}
                data["view_paths"] = vp
                updates.append((new_vp, json.dumps(data), claim_id))
                if len(updates) >= BATCH:
                    conn.executemany(
                        "UPDATE claims SET view_paths=?, data=? WHERE claim_id=?",
                        updates,
                    )
                    conn.commit()
                    updates = []
        print(f"  scanned {n_scanned:,}  changed {n_changed:,}")
    if updates:
        conn.executemany(
            "UPDATE claims SET view_paths=?, data=? WHERE claim_id=?",
            updates,
        )
        conn.commit()
    conn.close()
    print(f"\nDone. Updated {n_changed:,} claims.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="append",
                    help="restrict to one or more views (repeatable). default: all content views")
    ap.add_argument("--commit", action="store_true",
                    help="actually update DB (default: dry-run report only)")
    ap.add_argument("--rebuild-tree", action="store_true",
                    help="after --commit, rebuild tree_nodes from view_paths")
    ap.add_argument("--examples", type=int, default=30)
    args = ap.parse_args()

    views = args.view or list(ALL_CONTENT_VIEWS)
    print(f"Target views: {', '.join(views)}")
    print("Building snap plan...")
    plan = build_snap_plan(views)
    report(plan)
    show_examples(plan, args.examples)

    if args.commit:
        apply(plan)
        if args.rebuild_tree:
            print("\nRebuilding tree_nodes...")
            from reclassify_l3_batch import rebuild_tree
            rebuild_tree()
        else:
            print("\nNote: tree_nodes is now stale. Re-run with --rebuild-tree, or call "
                  "`python -c 'from reclassify_l3_batch import rebuild_tree; rebuild_tree()'`.")
    else:
        print("\n(dry-run — no changes written. Re-run with --commit to apply.)")


if __name__ == "__main__":
    main()
