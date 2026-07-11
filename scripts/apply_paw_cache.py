"""
Apply the PAW snap cache to the database, with a hand-curated denylist
that downgrades the most egregious wrong PAW snaps to "other" instead
of the wrong canonical L3.

Default is dry-run; pass --commit to apply + rebuild tree.

Usage:
    python scripts/apply_paw_cache.py
    python scripts/apply_paw_cache.py --commit --rebuild-tree
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

from askchem.taxonomy import get_canonical_l3, ALL_CONTENT_VIEWS  # noqa: E402

DB_PATH = ROOT / "chemtree.db"
CACHE_PATH = ROOT / "data" / "audits" / "l3" / "paw_snap_cache.json"

# Hand-curated denylist of (off_label, snapped_canonical) pairs that PAW got
# clearly wrong. These will be forced to "other" instead. Apply globally
# (across views and parents) since the off-label semantics don't change.
# Listed roughly in order of claim count (high-volume first).
DENYLIST_PAIRS: set[tuple[str, str]] = {
    ("rna", "dna"),
    ("asymmetric_synthesis", "cross_coupling_reactions"),
    ("electrochemical_characterization", "application_and_environmental_testing"),
    ("first_principles", "time_dependent_dft"),
    ("first_principles_calculations", "time_dependent_dft"),
    ("zeolitic_imidazolate_frameworks", "zirconium_based_mofs"),
    ("supported_metals", "single_atom_alloys"),
    ("black_phosphorus", "covalent_2d_frameworks_and_g_c3n4"),
    ("magnetotransport", "mixed_ionic_electronic_transport"),
    ("pump_probe_spectroscopy", "time_resolved_fluorescence"),
    ("aptamers", "nucleic_acid_modifications"),
    ("polysaccharides", "glycosaminoglycans"),
    ("borophene", "boron_nitride"),
    ("enzyme_inhibition", "small_molecule_binding"),
    ("glycoproteins", "membrane_proteins"),
    ("magnetic_alloys", "heavy_and_toxic_metals"),
    ("transition_metal_halides", "transition_metal_dichalcogenides"),
    ("lithium_sulfur_batteries", "application_and_environmental_testing"),
    ("hydrodesulfurization", "dehydrogenation_and_oxidative_dehydrogenation"),
    ("ion_and_metal_migration", "oxidative_thermal_degradation"),
    ("solid_solution_behavior", "carrier_mobility_and_diffusion"),
    ("catalytic_synthesis", "transition_metal_catalysis"),
    ("ammonium_sulfate", "ammonium_chloride"),  # different inorganic salts
    ("methane_activation", "methane_activation_and_conversion"),  # already exact in some views; safe
}

# Also force these "off labels" to always snap to "other" regardless of
# what PAW chose — useful for vague generic terms PAW guesses widely on.
DENYLIST_OFF_LABELS: set[str] = {
    "general_synthesis",
    "scope_exploration",
    "methodology_development",
    "methodology_focused_synthesis",  # canonical name elsewhere; here as off it's vague
    "general_methods",
    "miscellaneous",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--rebuild-tree", action="store_true")
    args = ap.parse_args()

    cache = json.load(open(CACHE_PATH))
    print(f"Loaded {len(cache):,} PAW snaps from cache.")

    # Build flat (view, l1, l2, off) -> mapped table with denylist applied
    flat: dict[tuple[str, str, str, str], str] = {}
    stats = Counter()

    for k, v in cache.items():
        try:
            view, l1, l2, off = k.split("|", 3)
        except ValueError:
            continue
        canon_set = set(get_canonical_l3(view, l1, l2) or [])

        # Apply denylist
        if off in DENYLIST_OFF_LABELS:
            mapped = "other" if "other" in canon_set else None
            stats["forced_to_other_offlabel"] += 1
        elif (off, v) in DENYLIST_PAIRS:
            mapped = "other" if "other" in canon_set else None
            stats["forced_to_other_pair"] += 1
        elif v in canon_set:
            mapped = v
            stats["accepted"] += 1
        else:
            # Cache value not in canonical list (shouldn't happen often); skip
            stats["invalid_cache_value"] += 1
            continue

        if mapped is None:
            stats["nothing_to_apply"] += 1
            continue
        flat[(view, l1, l2, off)] = mapped

    print(f"\nApply table built:")
    for k in sorted(stats):
        print(f"  {k:35s}  {stats[k]:>6,d}")
    print(f"  unique apply rules:                  {len(flat):>6,d}")

    if not args.commit:
        print("\n(dry-run — no DB changes. Re-run with --commit to apply.)")
        return

    # Apply to DB
    print("\nApplying to chemtree.db ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.execute("SELECT claim_id, view_paths, data FROM claims WHERE view_paths IS NOT NULL")
    BATCH = 5000
    updates: list[tuple[str, str, str]] = []
    n_changed = 0
    n_scanned = 0
    while True:
        rows = cur.fetchmany(20000)
        if not rows: break
        for claim_id, vp_json, data_json in rows:
            try: vp = json.loads(vp_json)
            except Exception: continue
            n_scanned += 1
            changed = False
            for vid, p in vp.items():
                if not isinstance(p, list) or len(p) < 3: continue
                key = (vid, p[0], p[1], p[2])
                if key in flat:
                    p[2] = flat[key]
                    changed = True
            if changed:
                n_changed += 1
                try: data = json.loads(data_json) if data_json else {}
                except Exception: data = {}
                data["view_paths"] = vp
                updates.append((json.dumps(vp), json.dumps(data), claim_id))
                if len(updates) >= BATCH:
                    conn.executemany("UPDATE claims SET view_paths=?, data=? WHERE claim_id=?", updates)
                    conn.commit()
                    updates = []
        print(f"  scanned {n_scanned:,}  changed {n_changed:,}", flush=True)
    if updates:
        conn.executemany("UPDATE claims SET view_paths=?, data=? WHERE claim_id=?", updates)
        conn.commit()
    conn.close()
    print(f"\nDone. Updated {n_changed:,} claims.")

    if args.rebuild_tree:
        print("\nRebuilding tree_nodes...")
        from reclassify_l3_batch import rebuild_tree
        rebuild_tree()


if __name__ == "__main__":
    main()
