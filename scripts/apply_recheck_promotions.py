"""
Patch gemini_validation_cache.json by promoting the 16 'demote_small_to_l3'
entries that the recheck pass flagged as 're_merge' → 'merge'.

These are pure synonym/plural cases that the first pass mis-tagged as demotes.
Idempotent: skip entries already at 'merge'.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = ROOT / "data/audits/l2/gemini_validation_cache.json"
RECHECK = ROOT / "data/audits/l2/gemini_recheck_demotes_cache.json"

primary = json.load(PRIMARY.open())
recheck = json.load(RECHECK.open())

n_promoted = 0
n_already = 0
n_skipped = 0
for k, rv in recheck.items():
    if rv.get("decision") != "re_merge":
        continue
    pv = primary.get(k)
    if not pv:
        n_skipped += 1
        continue
    if pv.get("decision") == "merge":
        n_already += 1
        continue
    pv["decision_orig"] = pv.get("decision")
    pv["decision"] = "merge"
    pv["recheck_promoted"] = True
    pv["recheck_reason"] = rv.get("reason", "")
    primary[k] = pv
    n_promoted += 1

backup = PRIMARY.with_suffix(".pre_recheck.json")
if not backup.exists():
    backup.write_text(json.dumps(json.load(PRIMARY.open()), indent=2))
PRIMARY.write_text(json.dumps(primary, indent=2))

print(f"Promoted: {n_promoted}")
print(f"Already merge: {n_already}")
print(f"Skipped (missing in primary): {n_skipped}")
print(f"Backup: {backup}")
