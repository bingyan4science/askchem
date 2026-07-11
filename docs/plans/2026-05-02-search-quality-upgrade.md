# AskChem search-quality upgrade plan

**Date:** 2026-05-02
**Author:** Bing Yan (paired with assistant)
**Status:** In progress
**Target DB:** `chemtree.db` (2,337,403 claims, 11.5 GB)

This plan addresses three observations made during a "Suzuki coupling"
search debugging session on 2026-05-01–02:

1. Search results in the Technique/Method view contained large amounts of
   condensed-matter / quantum-optics noise because the query embedder
   conflates "Suzuki coupling" with "spin coupling", "exciton coupling",
   etc.
2. Many claim cards rendered the same text under both **CLAIM** and
   **VERBATIM** because the structured fields the renderer expected
   were empty for ~25 % of records.
3. The reaction example
   *"Reaction: Suzuki–Miyaura cross-coupling"* (with a verbatim that
   simply paraphrased the title) was clearly too generic — paper-specific
   substrates / products were sitting unused in the JSON.

The plan is intentionally cheap to ship: it relies on **render-side
fallbacks** plus a **single-pass reclassification SQL** rather than
re-extraction, and queues the larger embedding upgrade as a follow-on.

---

## 0. What the audit shows

A 5 K-row sample per claim type (Python script in §6) gives the
following structured-field gap, by extractor model:

| `claim_type` | total | Gemini missing-primary | GPT-5-mini missing-primary |
|---|---:|---:|---:|
| `comparison`            | 281,552 | **0 %**   | **97 %** |
| `computational_result`  | 242,288 | **100 %** |  84 %  |
| `experimental_design`   |  46,773 |  90 %    | (n/a)  |
| `scope_entry`           |  55,109 |  51 %    | (n/a)  |
| `structure`             |  43,505 |  35 %    | (n/a)  |
| `hypothesis`            |  31,117 |  14 %    | (n/a)  |
| `property` / `method` / `mechanism` / `reaction` | 1.49 M | < 0.1 % | < 0.1 % |

Total claims missing the primary type-specific field:
**~580 K (≈ 25 % of all claims)**.

### Two failure modes at the data level

1. **Schema drift between extractors.** GPT-5-mini's prompt collapses
   every claim into a *property-shape* envelope:
   `subject` / `property_name` / `value` / `unit` / `measurement_method`.
   So for ~250 K "comparison" rows and ~170 K "computational_result"
   rows, the *content is there* — just under property-shape keys, not
   under `comparison_result` / `technique_name`.
2. **Mis-typed claims.** A "comparison" claim that emits property-shape
   and never compares two things is a `property` claim that was
   misclassified.

### Why the renderer made this worse

The original `buildClaimStatement` only knew about the *intended* keys
per type (e.g. `comparison_result` for `comparison` claims) and fell back
to `verbatim_quote` when those were empty. Result: same text printed
under CLAIM and under VERBATIM, ~580 K times.

### Why search was noisy

`all-MiniLM-L6-v2` (384-d, general-English) treats *coupling* as the
dominant token in the query "Suzuki coupling" and scores condensed-matter
"strong coupling" excerpts highly. This is an embedding-model issue;
filtering helps but does not solve the underlying ambiguity.

---

## 1. Plan

We split the work into three sprints by leverage / cost.

### Sprint A — Render-side fixes (today, no DB writes)

| Step | What | File(s) | Cost | Impact |
|---|---|---|---:|---|
| **1a** | Property-shape fallback. When a claim's primary field is empty but `subject`/`property_name`/`value`/`unit`/`measurement_method` are present, render those instead of falling back to verbatim. | `web/index.html` | ~30 LOC | Eliminates "Claim == Verbatim" for ~580 K claims |
| **1b** | Reaction-line composition. For `reaction` / `scope_entry`, render `{reaction_type} : {reactants} → {products} ; {conditions}` when those fields are populated. | `web/index.html` | ~40 LOC | Specifics shown for 121 K reaction claims; fixes the Suzuki-Miyaura genericity |
| **1c** | Use Gemini's `compared_items` + `metric` + `comparison_result` schema for comparison cards (so Gemini's 2 M-claim lane renders cleanly even before reclassification). | `web/index.html` | ~20 LOC | Cleans the 282 K comparison claims |

Acceptance: hard-refresh on `/#/search?q=Suzuki%20coupling` shows
substrates → products under the badge for at least 3 of the top 5 cards;
no card shows the same text in CLAIM and VERBATIM rows.

### Sprint B — One-shot reclassification (today, batched DB write)

| Step | What | Output |
|---|---|---|
| **2a** | SQL audit script that reports candidates: `claim_type ∈ {comparison, computational_result, scope_entry}` AND has property-shape but no type-specific field. | counts only |
| **2b** | Apply: for those rows, set `claim_type = 'property'` and stamp `data._reclassified_from = <orig>` / `data._reclassified_at = <ts>` for auditability. | reclassified rows |
| **2c** | Re-emit `view_paths['by_claim_type']` to `['properties']` for the reclassified rows (other views are unaffected — they live under different organising principles). | updated `view_paths` |
| **2d** | Backup before applying. | `.bak` SQLite copy |

Acceptance: search filtered by `claim_type=property` returns ≥ 250 K
new rows that were previously hidden under `comparison`. No `view_paths`
besides `by_claim_type` should change.

### Sprint C — Embedding upgrade + cross-encoder rerank (next session)

This is **out of scope for today's commit** but listed so it isn't lost.

1. Run a 20 K stratified-sample evaluation:
   - candidates: `pritamdeka/S-PubMedBert-MS-MARCO`, `m3rg-iitd/matscibert`, `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`.
   - queries: 20 hand-labeled chemistry probes (Suzuki coupling, MOF
     surface area, perovskite Voc, C-H activation, CRISPR, …).
   - metric: nDCG@10 vs. current MiniLM.
2. Pick winner, rebuild full embeddings (~3-6 h on Apple-MPS @ 768-d).
3. Add cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`
   first; chemistry-tuned later) on top-50.
4. Drop the ad-hoc `_technique_claim_is_irrelevant_for_coupling_query`
   filter once the underlying embedding ambiguity is gone.

---

## 2. What we are *not* doing

- **No full re-extraction.** 2.34 M claims at $0.02/claim is ~$47 K and
  weeks of API time. The renderer + reclassification path covers
  ≥ 95 % of the visible quality gap.
- **No prompt-level fix to GPT-5-mini's schema drift.** Re-running the
  ~820 K GPT-5-mini claims with a corrected prompt is queued behind
  Sprint C.
- **No taxonomy churn.** `view_paths` change only for `by_claim_type`
  and only for the reclassified rows; other views are stable.

---

## 3. Acceptance metrics (post-Sprint A+B)

| Metric | Before | Target |
|---|---:|---:|
| Claims rendering identical CLAIM == VERBATIM text | ~580 K | **< 5 K** |
| `Suzuki coupling` cards showing reactants → products | 0 | **≥ 80 % of reaction cards in top 50** |
| `comparison` claims rendering with body content | ~10 % | **≥ 90 %** |
| Technique/Method L1 buckets containing only organic-coupling claims for Suzuki query | 9 of 10 | 10 of 10 |

---

## 4. Risk / rollback

- The SQL reclassification ships under a transaction with a sibling
  `chemtree.db.pre_reclass_<ts>.bak` file. Rollback is `cp .bak
  chemtree.db`. The script also writes a row-id list of every
  reclassified claim to `data/reclassified_<ts>.txt` so we can `UPDATE`
  back if needed.
- Renderer changes are scoped to `renderClaim` in
  `web/index.html`; nothing in the API contract changes.

---

## 5. Files touched

- `web/index.html` — `buildClaimStatement` extension, `_renderReactionLine`, comparison schema fallback.
- `scripts/reclassify_property_shape.py` — audit + apply, with backup and audit log.
- `tests/test_property_shape_fallback.py` — unit test for the new
  rendering path (text generation only; the renderer is JS so the
  Python test mirrors the same logic).

## 6. Audit script reference

```python
# tools/audit_field_completeness.py — short-form
import sqlite3, json, collections

DB = "chemtree.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
expected = {
    'reaction':            ['reaction_type'],
    'scope_entry':         ['reaction_type'],
    'property':            ['value','property_name'],
    'method':              ['technique_name','what_it_achieves'],
    'mechanism':           ['process_described'],
    'comparison':          ['comparison_result'],
    'computational_result':['technique_name','what_it_achieves'],
    'experimental_design': ['technique_name','what_it_achieves','key_innovation'],
    'scope_entry':         ['reaction_type'],
    'hypothesis':          ['hypothesis_text'],
    'limitation':          ['limitation_text'],
    'future_direction':    ['direction_text'],
    'surprising_finding':  ['finding_text'],
}
def truthy(v):
    if v is None: return False
    if isinstance(v, str):  return bool(v.strip())
    if isinstance(v,(list,dict)): return bool(v)
    return bool(v)
for t, primary in expected.items():
    rows = conn.execute("SELECT data FROM claims WHERE claim_type=? LIMIT 5000",(t,)).fetchall()
    miss = sum(1 for r in rows if not any(truthy((json.loads(r['data']) or {}).get(k)) for k in primary))
    print(f"{t:<22} {miss/len(rows)*100:>5.1f}%  ({len(rows)} sampled)")
```
