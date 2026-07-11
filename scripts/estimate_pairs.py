#!/usr/bin/env python3
"""Estimate view-independent candidate pairs — streaming, low memory."""

import sqlite3, json
from collections import Counter

DB = "chemtree.db"

JUNK_SUBJECTS = {
    'paper', 'paper metadata', 'this paper', 'this study', 'this review',
    'this review article', 'this work', 'publication metadata',
    'this paper (metadata)', 'the paper', 'the study', 'the review',
    'study', 'review', 'article', 'this article', 'the article',
    'publication', 'metadata', 'research', 'this research', 'the research',
}

CONTRADICTABLE_TYPES = {
    'property', 'mechanism', 'finding', 'comparison', 'performance',
    'observation', 'outcome', 'efficacy',
}

def to_str(val):
    if val is None: return ''
    if isinstance(val, list): return ' '.join(str(s) for s in val)
    if isinstance(val, dict): return str(val)
    return str(val)

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Stream rows — don't load all at once
    cur.execute('SELECT claim_id, source_doi, claim_type, data FROM claims WHERE length(data) > 100')

    # subject_groups stores only IDs/DOIs/types — not full data blobs
    # Key: (subject, claim_type) -> list of (claim_id, doi)
    groups = {}  
    stats = Counter()
    batch = 0

    while True:
        rows = cur.fetchmany(10000)
        if not rows:
            break
        batch += 1
        if batch % 10 == 0:
            print(f"  processed {batch * 10000:,} rows, {len(groups):,} groups so far...", flush=True)

        for cid, doi, ctype, data_str in rows:
            stats['total'] += 1

            if ctype not in CONTRADICTABLE_TYPES:
                stats['non_contradictable_type'] += 1
                continue

            try:
                d = json.loads(data_str)
            except Exception:
                stats['bad_json'] += 1
                continue

            subj = to_str(d.get('subject', '')).strip().lower()
            if not subj:
                stats['no_subject'] += 1
                continue
            if subj in JUNK_SUBJECTS:
                stats['junk_subject'] += 1
                continue

            quote = to_str(d.get('verbatim_quote', ''))
            if len(quote) < 50:
                stats['short_quote'] += 1
                continue

            stats['usable'] += 1
            key = (subj, ctype)
            if key not in groups:
                groups[key] = {}
            groups[key].setdefault(doi, []).append(cid)

    print(f'\nTotal claims scanned: {stats["total"]:,}')
    print(f'\nFilter stats:')
    for k, v in stats.most_common():
        print(f'  {k}: {v:,}')

    print(f'\nUnique (subject, claim_type) groups: {len(groups):,}')

    MAX_CAP = 20
    total_capped = 0
    total_uncapped = 0
    groups_with_pairs = 0

    for key, doi_map in groups.items():
        if len(doi_map) < 2:
            continue
        groups_with_pairs += 1
        doi_list = list(doi_map.keys())
        gp = 0
        for i in range(len(doi_list)):
            for j in range(i + 1, len(doi_list)):
                gp += len(doi_map[doi_list[i]]) * len(doi_map[doi_list[j]])
        total_capped += min(gp, MAX_CAP)
        total_uncapped += gp

    print(f'\nGroups with >=2 DOIs (can form pairs): {groups_with_pairs:,}')
    print(f'Cross-DOI same-type pairs (uncapped): {total_uncapped:,}')
    print(f'Cross-DOI same-type pairs (capped {MAX_CAP}/group): {total_capped:,}')
    print(f'Estimated Gemini cost (~$0.001/pair): ${total_capped * 0.001:.0f}')

    # Top groups
    top = sorted(groups.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)[:20]
    print(f'\nTop 20 (subject, type) groups:')
    for (subj, ctype), doi_map in top:
        total_claims = sum(len(v) for v in doi_map.values())
        print(f'  "{subj[:60]}" [{ctype}]: {total_claims} claims, {len(doi_map)} DOIs')

if __name__ == '__main__':
    main()
