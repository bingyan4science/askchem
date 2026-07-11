#!/usr/bin/env python3
"""Generate view-independent contradiction candidate pairs.

Groups ALL claims by (normalized_subject, claim_type), pairs claims from
different DOIs within each group, and exports to JSONL for Gemini verification.

Runs on the server (needs DB access). Output is downloaded for local Gemini.
"""
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from askchem.db import get_db_path

JUNK_SUBJECTS = {
    "paper", "paper metadata", "this paper", "this study", "this review",
    "this review article", "this work", "publication metadata",
    "this paper (metadata)", "the paper", "the study", "the review",
    "study", "review", "article", "this article", "the article",
    "publication", "metadata", "research", "this research", "the research",
    "paper (metadata)",
}

CONTRADICTABLE_TYPES = {
    "property", "mechanism", "finding", "comparison", "performance",
    "observation", "outcome", "efficacy",
}

MAX_PAIRS_PER_GROUP = 20


def to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(s) for s in val if s)
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def main():
    output_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/candidates_viewfree.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    db_path = get_db_path()
    print(f"Database: {db_path}", flush=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA mmap_size=268435456")
    cur = conn.cursor()
    cur.execute(
        "SELECT claim_id, source_doi, claim_type, data FROM claims "
        "WHERE length(data) > 100"
    )

    # (subject_lower, claim_type) -> { doi -> [(claim_id, quote, subject_raw)] }
    groups: dict[tuple[str, str], dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    stats = defaultdict(int)
    batch_num = 0

    while True:
        rows = cur.fetchmany(10_000)
        if not rows:
            break
        batch_num += 1
        if batch_num % 10 == 0:
            n_groups = sum(1 for g in groups.values() if len(g) >= 2)
            print(
                f"  {batch_num * 10_000:,} rows, "
                f"{len(groups):,} groups ({n_groups:,} with >=2 DOIs)...",
                flush=True,
            )

        for cid, doi, ctype, data_str in rows:
            stats["total"] += 1
            if ctype not in CONTRADICTABLE_TYPES:
                stats["skip_type"] += 1
                continue

            try:
                d = json.loads(data_str)
            except Exception:
                stats["bad_json"] += 1
                continue

            subj = to_str(d.get("subject", "")).strip()
            subj_lower = subj.lower()
            if not subj_lower:
                stats["no_subject"] += 1
                continue
            if subj_lower in JUNK_SUBJECTS:
                stats["junk_subject"] += 1
                continue

            quote = to_str(d.get("verbatim_quote", ""))
            if len(quote) < 50:
                stats["short_quote"] += 1
                continue

            stats["usable"] += 1
            groups[(subj_lower, ctype)][doi].append((cid, quote[:500], subj))

    print(f"\n{'='*60}", flush=True)
    print(f"Scan complete. Stats:", flush=True)
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}", flush=True)

    # Generate pairs
    total_pairs = 0
    n_groups_with_pairs = 0

    with open(output_file, "w") as f:
        for (subj_lower, ctype), doi_map in groups.items():
            if len(doi_map) < 2:
                continue
            n_groups_with_pairs += 1
            doi_list = list(doi_map.keys())

            # Collect all cross-DOI pairs for this group
            group_pairs = []
            for i in range(len(doi_list)):
                for j in range(i + 1, len(doi_list)):
                    for cid1, q1, s1 in doi_map[doi_list[i]]:
                        for cid2, q2, s2 in doi_map[doi_list[j]]:
                            group_pairs.append((cid1, q1, s1, doi_list[i],
                                                cid2, q2, s2, doi_list[j]))

            # Cap: pick random sample if too many
            if len(group_pairs) > MAX_PAIRS_PER_GROUP:
                group_pairs = random.sample(group_pairs, MAX_PAIRS_PER_GROUP)

            for cid1, q1, s1, d1, cid2, q2, s2, d2 in group_pairs:
                rec = {
                    "claim_id_1": cid1,
                    "claim_id_2": cid2,
                    "quote_1": q1,
                    "quote_2": q2,
                    "doi_1": d1,
                    "doi_2": d2,
                    "subject": s1,
                    "claim_type": ctype,
                }
                f.write(json.dumps(rec) + "\n")
                total_pairs += 1

    print(f"\nGroups with >=2 DOIs: {n_groups_with_pairs:,}", flush=True)
    print(f"Total candidate pairs written: {total_pairs:,}", flush=True)
    print(f"Output: {output_file}", flush=True)


if __name__ == "__main__":
    main()
