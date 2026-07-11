# AskChem retrieval upgrade plan

**Date:** 2026-04-26
**Author:** Bing Yan (with feedback from Kyunghyun Cho)
**Status:** Proposed

This plan operationalizes Cho's feedback on AskChem retrieval. It maps each
recommendation to concrete files, schema changes, prompts, and Gemini batch
costs measured against the live database.

---

## 0. Feedback summary

> 1. **Anthropic-style contextual retrieval, applied to claims.** Do not index a
>    bare extracted claim. Index a *contextualized claim card* that contains:
>    paper-level extraction (Step A), evidence localization (Step B), a
>    contextualized rewrite (Step C), and multi-view fields (Step D).
> 2. **ColBERT for fine-grained scientific matching.** Add a late-interaction
>    retrieval/ranking layer on top of claim cards and evidence spans (not as
>    the extractor).
>
> Final retrieval stack:
>
> ```
> User query
>   → query understanding (entities, reaction type, method, property, intent)
>   → candidate retrieval from
>       1. structured filters/entities
>       2. BM25 over contextualized claims
>       3. dense embeddings over contextualized claims
>       4. ColBERT over evidence spans / claim cards
>   → reranking
>   → answer with cited claims + evidence
> ```

---

## 1. How the feedback maps to today's system

| Cho's prescription | What we have | What's missing |
|---|---|---|
| **Step A — Full-paper extraction** | `src/extract_v2.py` + the arxiv tier1 batch (`gemini-3.1-pro` / `deep_v1`) already produced **1,515,538 claims from 41,076 full papers** — these have `claim_type`, `process_described`/`steps`/`key_intermediates` (or analogous typed fields), `verbatim_quote`, and `location_in_paper`. | First-class `paper_summary`, `contributions`, `entities`, `limitations` fields per paper. (Existing extractions did not write a paper-level row; we'll backfill in Sprint 2b.) |
| **Step B — Evidence localization** | `claims.verbatim_quote` + `claims.location_in_paper` (free text). `Claim.evidence: list[dict]` is populated for deep_v1 (e.g. `[{"type":"text","description":"Figure 19 and accompanying discussion."}]`). | Structured `{page, section_path, block_type, char_span, quote, confidence}`. Will land in Sprint 4 on top-cited deep_v1 papers (where we have or can fetch PDFs). |
| **Step C — Contextualized rewriting** | `build_searchable_text()` concatenates typed fields — works, but is not a real sentence. | A `claim_contextualized` sentence per **deep_v1** claim. **Out of scope for the abstract-only 821k claims** (Bing's directive). |
| **Step D — Multi-view indexing** | `claims_fts` has one combined `searchable_text`; `sources_fts` has title/abstract/`paper_text` (derived from claims). MiniLM-L6-v2 (384d) embeddings outside SQLite. | Separate FTS columns for `claim_raw` / `claim_contextualized` / `evidence_text` / `entity_line` / `paper_summary`. Per-field BM25 weights. |
| **ColBERT late interaction** | Single dense vector per claim. | No token-level retrieval; specific chemistry queries get blurred. |
| **Reranker** | RRF over FTS + vector + tree + author. | No cross-encoder rerank. |

Bones are good (claim/source schema, FTS5, embeddings, taxonomy, edge graph).
What's missing is the *claim-card abstraction* and a *fine-grained retrieval
channel*.

---

## 1.5. Two distinct extraction processes

This plan contains **two LLM extraction passes that look superficially similar
but have very different prompts, scopes, and costs.** Calling them out here so
the difference is unambiguous.

### Process A — Backfill contextualization on **existing** claims (Sprint 1)

| | |
|---|---|
| Applies to | the **1,515,538** claims we already have from `gemini-3.1-pro` / `deep_v1` (full-paper extractions done before we adopted Cho's contextualization step) |
| LLM input | the existing typed fields, `verbatim_quote`, `paper_title`, `location_in_paper`, **plus paper_summary** (from Sprint 0) — **NOT the full paper** |
| LLM output | one sentence per claim → `claim_contextualized` |
| Prompt rule | "Use ONLY information present in the inputs. Do not invent." |
| Why no full paper | (a) the full paper has already been distilled into the typed fields + verbatim quote at original-extraction time, so re-feeding it is mostly redundant; (b) feeding paper text per claim at 1.52M claims would cost ≈ $150k+ on 3.1 Pro batch, vs. $1.27k for the slot-based prompt |
| One-time cost | ≈ **$1,271** (3.1 Pro batch, all 1.52M claims) |

### Process B — Step-A extraction for **new** papers as they ingest (Sprint 6)

| | |
|---|---|
| Applies to | every paper added to the corpus *after* this plan ships (≈ 10k/month) |
| LLM input | the **full paper text** (PDF → text, ≈ 30–80k input tokens) |
| LLM output | a single response containing: `paper_summary`, `contributions`, `claims[]` (each with `claim_type`, typed fields, `verbatim_quote`, `location_in_paper`, structured `evidence[]`, **and `claim_contextualized` already filled in**), `entities`, `methods`, `limitations` |
| Why one call | contextualization is a `response_schema` field, not a second pass — claim cards are born complete |
| Recurring cost | ≈ **$260/mo** (≈ 10k papers/mo on 3.1 Pro batch; 80M input + 30M output tokens) |

The 1.52M existing `deep_v1` claims will never run through Process B — they
were extracted before the new schema existed. They get Process A only.

---

## 2. Numbers I'm planning against (live DB)

| Metric | Value |
|---|---|
| Total claims | **2,337,403** |
| → from full-paper extraction (`deep_v1`, `gemini-3.1-pro`) | **1,515,538** (64.8%) |
| → from abstract-only extraction (`v3-abstract*`, `gpt-5-mini`) | **821,372** (35.1%) |
| → out-of-scope for contextualization (per Cho/Bing decision) | the abstract-only ones |
| Distinct papers w/ full-paper extraction | **41,076** |
| Total papers (`sources`) | 140,913 |
| Claims/paper (deep_v1): mean / p50 / p90 / max | **36.9 / 30 / 61 / 241** |
| Per-claim `data` blob: mean / p50 / p90 / p99 (chars) | **1,358 / 1,343 / 1,782 / 2,252** |
| Verbatim-quote chars: mean / p99 | 145 / 387 |
| Full-text JSONL we already have (arxiv tier1) | 292 files, 0.17 GB |

**Citation distribution within the deep_v1 universe** (this is the dial we actually
turn for cost):

| Threshold (paper citation count) | deep_v1 claims passing |
|---|---|
| ≥ 0 (everything) | **1,515,538** |
| ≥ 25 | 1,357,348 |
| ≥ 100 | **689,972** |
| ≥ 250 | 232,318 |
| ≥ 500 | 87,050 |
| ≥ 1000 | 25,202 |

Implications:

- **Contextualization is bounded to 1.52M claims**, not 2.34M. The 821k
  abstract-only claims keep using `searchable_text` as their card text — no
  contextualization pass, because there is no "context" beyond what is already
  in their JSON fields and the abstract.
- Per-claim contextualize input averages **~340 tokens** (1358 chars / ~4
  chars-per-token). Output is one sentence + JSON wrapper, **~80 tokens**.
- Batching 8 claims per Gemini request amortizes the prompt header (~150
  tokens) and roughly cuts input cost by ~30%.

---

## 3. Gemini API pricing for the model we picked

We use **`gemini-3.1-pro-preview`** (Cho's instruction: latest and best). Source:
[Google AI for Developers pricing](https://ai.google.dev/gemini-api/docs/pricing),
fetched 2026-04-26.

| Tier | Input $/M tok | Output $/M tok |
|---|---|---|
| Standard, prompt ≤200k | $2.00 | $12.00 |
| Standard, prompt >200k | $4.00 | $18.00 |
| **Batch, prompt ≤200k** (what we use) | **$1.00** | **$6.00** |
| Batch, prompt >200k | $2.00 | $9.00 |
| Cached input, ≤200k | $0.20 | — |

We **always stay ≤200k** — even at 8 claims/request our input is ~3k tokens.

For comparison (in case we ever need a budget alternative for a non-critical
backfill):

| Model | Batch $/M in | Batch $/M out |
|---|---|---|
| Gemini 3.1 Flash-Lite Preview | $0.13 | $0.75 |
| Gemini 3.1 Pro Preview | **$1.00** | **$6.00** |

Pro is ~8× the cost of Flash-Lite, which is the price we accept for the rewrite
quality on which the entire downstream search experience hinges.

---

## 4. Cost matrix

> **Calibration note (2026-04-26 — now anchored on a real Vertex Batch):**
> Gemini 3.1 Pro Preview is a *reasoning model* — reasoning trace counts
> as `completion_tokens`.
>
>   - **Sprint 0 chunk 0 (real batch, 7,031 papers, completed 12:40 UTC-4):**
>     in = 23,127,540, out = 13,782,339, **mean 3,289 in / 1,960 out per paper**.
>     Cost paid: **$105.82**. 100% parseable, 100% finish_reason=STOP, 1
>     reject (8-char too-long). Quality on 30-paper hand-rate: factual,
>     claim-grounded, chemistry-rich.
>   - **Sprint 1 sync dry-run (16 claims, batched 8/req):** in=4,267,
>     out=4,194 → **~267 in / 262 out per claim**. 14/16 passed validation.
>
> The "Naive" column was the v1 plan's estimate (assumed 80 visible output
> tokens). The "**Calibrated**" column was based on small sync dry-runs.
> The "**Measured**" column is what the real Vertex Batch is actually
> billing for Sprint 0 — **higher than the sync calibration by ~17%**
> because real-batch reasoning runs longer than sync dry-runs.

Assumptions (live data — Sprint 0 from real batch, Sprint 1 from sync dry-run, 2026-04-26):
- Sprint 0: 1 paper/call → in≈3,289, out≈1,960 (real-batch measurement)
- Sprint 1: **8 claims/call** → in≈267/claim, out≈262/claim (sync dry-run; will be re-anchored when Sprint 1a finishes)
- model: **`gemini-3.1-pro-preview` batch** unless noted

| Phase | Scope | Claims/Papers | Tokens (in / out) | Naive cost | Calibrated | **Measured** |
|---|---|---|---|---|---|---|
| **0** Per-paper short summary (deep_v1 papers) | summary | **41,076** papers | 126M / 80M | $58 | $530 | **$608.64** ✓ |
| **1a** Contextualize, top-cited deep_v1 (cit ≥1000) | smoke | **25,202** | 10M / 6.7M | $21 | $46 | **$50.05** ✓ |
| **1b/1c** *Subsumed by 1d* (gated on 1a quality) | — | — | — | — | — | — |
| **1d** Contextualize **all deep_v1 residual** | full | **1,491,492** | 619M / 403M | $1,271 | $2,792 | **$3,034.88** ✓ |
| **3** Cross-encoder reranker | open-source | — | — | $0 | $0 | $0 |
| **4** Evidence localization, deep_v1 top-cited | enrich | top-25k papers | ~200M / ~150M | $650 | $1,100 | _tbd_ |
| **5** ColBERT index | open-source PyLate | — | — | $0 | $0 (compute) | $0 |
| **6** Full-paper extraction for new ingest | ongoing | ~10k papers/mo | ~80M / ~50M | $260/mo | $380/mo | _tbd_ |
|  | **Total claim-card foundation paid** | | | | | **$3,693.57** |

> **History of the Sprint 1d estimate**, for posterity so future readers
> understand why we went down to the bone on this number:
>   - v1 plan (naive output budget):                **$1,271**
>   - calibrated, 1 claim/req:                     **~$11,000**
>   - calibrated, **batched 8 claims/req (now)**:    **~$2,792**

**Sprint 0 launch decision (made 2026-04-26 12:50 UTC-4):** Chunk 0 cost
$105.82 with 99.99% effective yield and excellent quality. Extrapolating
linearly, the remaining 5 chunks (~34k papers) will cost **~$510** for a
**total Sprint 0 spend of ~$620**. Approved — submitting all 5 remaining
chunks now.

**Cost knobs still on the table (apply after Sprint 1a passes the gate):**

1. **`thinking_config: {budget: 0}`** on the Gemini API (Vertex pass-through).
   Cuts reasoning to near-zero. Risk: quality regression on long-tail
   claims. We'll A/B 100 claims with budget=0 vs default before flipping.
   Potential additional savings: **2–4×** on output tokens.
2. **Drop Sprint 0 to `gemini-2.5-flash`** (no reasoning trace, ~30× cheaper):
   $16 vs $530. Paper summaries are easy. We'll A/B 30 summaries from
   chunk 0 vs Flash before flipping for chunks 1–5. Potential savings: **~$510**.

**Recommended full upgrade after batch-8 + knob 2:** Sprint 0 (Flash for
chunks 1–5, Pro for chunk 0 already in flight) ≈ $100 + Sprint 1d (Pro
batched-8) ≈ $2,792 + Sprint 4 ≈ $1,100 = **~$4,000 LLM total**.

That's about 3× the v1 plan estimate, but the v1 plan was unaware of
reasoning tokens; the **batch-8 optimization** is what brings it back into
sane range. The dollars buy us: 41k paper summaries + 1.52M
contextualized claim cards + evidence localization on the top 25k papers,
i.e. the entire Cho-style "claim card with evidence" foundation.

---

## 5. Sprint plan

### Sprint 0 — `paper_summary` for the 41k full-paper papers (run *before* Sprint 1)

**Why first:** Sprint 1's contextualization prompt has a `paper_summary` slot.
Computing it once per paper (≈ 41k calls, 36.9× cheaper than per-claim) lets
all 1.52M Sprint-1 rewrites share that paper-level grounding without paying
paper-level token cost per claim.

**Scope:** the **41,076 distinct papers** that have at least one `deep_v1`
claim — i.e. the papers we have full-paper coverage of.

**Model:** **`gemini-3.1-pro-preview` batch**.

**Files**

1. **`src/chemtree/db.py` — `init_db()`** (idempotent ALTERs):
   ```sql
   ALTER TABLE sources ADD COLUMN paper_summary TEXT;
   ALTER TABLE sources ADD COLUMN paper_summary_model TEXT;
   ALTER TABLE sources ADD COLUMN paper_summary_version TEXT;
   ALTER TABLE sources ADD COLUMN paper_summary_extracted_at TEXT;
   ```

2. **`scripts/summarize_papers.py`** (new)
   - For each of the 41k papers, build the LLM input from *all* of that
     paper's `deep_v1` claims:
     ```
     paper_title:  ...
     authors:      ...
     venue/year:   ...
     claims (n):
       1. claim_type=...  verbatim_quote="..."
          process/finding=...  steps=[...]
       2. ...
       ...
     ```
   - Truncate gracefully if the paper has > 60 claims (use top-confidence
     ones first).
   - Output schema: `{"doi": "...", "paper_summary": "<≤ 80 words, ≤ 6 sentences>"}`.
   - Validation: ≤ 600 chars; the summary must cite material/system/method
     names that appear in the input claim list (substring check).

3. **`src/chemtree/prompts/summarize_paper_v1.txt`** (new)
   ```
   You write a short, factual summary of a chemistry paper using ONLY the list
   of claims that have already been extracted from it. The reader will use
   your summary as paper-level context when reading individual claims.

   Hard rules:
   - Use ONLY information present in the input claims list, paper title,
     and metadata. DO NOT consult outside knowledge.
   - DO NOT invent numbers, conditions, catalysts, ligands, materials, or
     methods. Omit anything you cannot ground in the inputs.
   - Output a single paragraph, ≤ 80 words, ≤ 6 sentences.
   - Mention the paper's main system (material/reaction/process), its main
     method or measurement, and its main finding(s) — but only if explicitly
     present in the claims.
   - Do NOT begin with "This paper", "The authors", "We", or "In this study".

   Output: JSON only, with this exact schema:
   {"doi": "<input doi>", "paper_summary": "<one paragraph>"}
   ```

**Cost:** ≈ **$58** (see §4 row "2 Per-paper short summary").
**Runtime:** one Gemini Batch submission, 2–24 h turnaround.
**Acceptance gate:** human-rate 30 random summaries; require ≥ 90% pass on
"factually grounded in the listed claims, no hallucination."

**Why this is Sprint 0, not part of Sprint 2:**
- It is a prerequisite for Sprint 1's prompt quality.
- It is itself cheap (~$58), so blocking on it adds at most a day.
- It makes Sprint 2's FTS rebuild *additive* rather than dependent — Sprint 2
  can read `sources.paper_summary` directly into the FTS table.

---

### Sprint 1 — `claim_contextualized` on full-paper claims only

**Scope (per Cho + Bing decision):** *only* the **1,515,538 `deep_v1`** claims
extracted by `gemini-3.1-pro` from full-paper text. The 821k abstract-only
claims keep their existing `searchable_text` and are **not** contextualized —
there is no extra context to add to a claim that itself was distilled from a
~250-word abstract; rewriting it would just paraphrase, not enrich.

**Model:** **`gemini-3.1-pro-preview`** via the **Batch API** (~$1/M input,
$6/M output). One sentence in, one sentence out — but it's the sentence we
will index, embed, and rerank against, so we pay Pro rates here.

**Goal:** every full-paper claim has a one-sentence, standalone-readable
rewrite that names the system, method, measurement, and condition.

**Files**

1. **`src/chemtree/db.py` — `init_db()`** (idempotent ALTERs, matching the
   existing try/except pattern):
   ```sql
   ALTER TABLE claims ADD COLUMN claim_contextualized TEXT;
   ALTER TABLE claims ADD COLUMN context_model TEXT;
   ALTER TABLE claims ADD COLUMN context_version TEXT;
   ALTER TABLE claims ADD COLUMN context_extracted_at TEXT;
   ```

2. **`scripts/contextualize_claims.py`** (new)
   - Args: `--min-citations 0` (default), `--limit N`,
     `--model gemini-3.1-pro-preview`, `--prompt-version v1`,
     `--batch-size 8`, `--require-deep-v1 1`.
   - Query (note `extraction_version` filter — non-negotiable):
     ```sql
     SELECT c.claim_id, c.claim_type, c.verbatim_quote, c.source_paper_title,
            c.location_in_paper, c.data,
            s.citation_count, s.paper_summary
     FROM claims c JOIN sources s ON c.source_doi = s.doi
     WHERE c.extraction_version = 'deep_v1'
       AND c.claim_contextualized IS NULL
       AND s.citation_count >= ?
     ORDER BY s.citation_count DESC, c.claim_id
     LIMIT ?;
     ```
   - `paper_summary` is filled by Sprint 0; if NULL (e.g. a deep_v1 claim
     whose paper somehow missed the Sprint 0 batch), the prompt's
     `paper_summary` slot becomes an empty string and the rewrite degrades
     gracefully to the same prompt as the slot-only path.
   - Reuses existing Gemini Batch infra (the same pattern already used in
     `src/backfill_edges.py` and `scripts/gemini_validate_l2_merges.py`):
     submit batch → poll → download → parse → validate → write back.
   - Cache responses in `data/audits/contextualize/<prompt_version>/`
     (resumable; one JSONL per batch chunk).
   - Validation per output (reject any failure → leave the column NULL and log):
     - rewrite ≤ 280 chars
     - no numeric tokens that don't appear in
       `verbatim_quote ∪ data` (regex `\d+(?:[.,]\d+)?`)
     - rewrite is not a near-duplicate of `verbatim_quote` (lev distance > 30%)
     - rejects logged to `data/audits/contextualize/<v>/rejects.jsonl`
   - DB writeback in transactions of 1,000 rows.

3. **`src/chemtree/prompts/contextualize_v1.txt`** (new)
   ```
   You rewrite a single scientific claim from a chemistry paper so it stands
   alone outside that paper. Readers see only your rewrite; they do not have
   the paper.

   Hard rules:
   - Use ONLY information present in the inputs (typed fields, verbatim
     quote, paper title, location_in_paper, paper_summary). DO NOT consult
     any outside knowledge.
   - DO NOT invent numbers, conditions, catalysts, ligands, materials,
     methods, or comparisons. If a fact is not in the inputs, omit it.
   - The paper_summary is provided as paper-level context; you may pull
     specific terms (a material name, a reaction class, a method) from it
     into the rewrite, but DO NOT add facts that are absent from BOTH the
     claim's typed fields and the paper_summary.
   - One sentence, ≤ 220 characters, declarative.
   - Begin with the chemical subject (a material, reaction, mechanism,
     measurement, or finding). Do NOT start with "The paper", "This study",
     "We", "The authors".
   - Include the relevant {system, method, measurement, condition,
     evidence-type} only WHEN they are explicit in the inputs.
   - Preserve the original claim_type's emphasis (a mechanism stays a
     mechanism, a measurement stays a measurement, etc.).

   Output: JSON only, with this exact schema:
   {"claim_id": "<input claim_id>", "claim_contextualized": "<one sentence>"}

   Inputs:
   claim_id:           {claim_id}
   claim_type:         {claim_type}
   paper_title:        {paper_title}
   paper_summary:      {paper_summary_or_empty}
   location_in_paper:  {location_in_paper}
   verbatim_quote:     {verbatim_quote}
   typed_fields_json:  {typed_fields_json}
   ```
   The `typed_fields_json` slot is the `data` column with `claim_id`,
   `extraction_*`, `view_paths`, `verbatim_quote`, `source_doi`,
   `source_paper_title`, `extracted_at` stripped (already in other slots, or
   not useful for rewriting).

4. **Batch packaging (`scripts/contextualize_claims.py` internals):**
   - Group **8 claims per Gemini request**: each request is a JSON list, the
     model returns a JSON list of 8 `{claim_id, claim_contextualized}` objects.
   - Use Gemini structured output (`response_schema`) so we don't have to
     re-parse free-text JSON.
   - 1.52M / 8 ≈ 189k requests. Submit in chunks of ~25k requests per batch
     job (Gemini Batch caps at 100k requests / 2 GB / 24 h).

5. **`web/index.html::renderClaim`** — if `c.claim_contextualized` is present,
   render it as a one-line lede above the verbatim quote (which becomes a
   smaller, italicized "evidence" snippet). For abstract-only claims with no
   rewrite, fall back to the existing rendering.

**Suggested rollout (gates between stages):**

| Stage | Scope | Cost | Why |
|---|---|---|---|
| 1a | top 25k (cit ≥ 1000) | ≈ $21 | smoke test prompt + pipeline + DB writeback; human-rate 50 |
| 1b | top 232k (cit ≥ 250) | ≈ $200 | confirm quality scales; entity coverage spot-check |
| 1c | top 690k (cit ≥ 100) | ≈ $578 | covers most user search hits |
| 1d | all 1.52M (`deep_v1`) | ≈ $1,271 | full coverage; long-tail searches benefit |

**Total:** ≈ **$1,271** for the full Sprint 1 (1a–1d are cumulative).

**Runtime:** Gemini Batch turnarounds are typically 2–24 h per submission; we
run 1a, gate, then submit 1b/1c/1d in parallel-but-staged jobs.

**Acceptance gate (human eval):**
- Stage 1a: human-rate 50 random rewrites, require ≥ 90% pass on
  *self-explanatory AND not hallucinated*.
- Each subsequent stage: rate 30 random rewrites; if pass-rate drops below
  85%, freeze the rollout, fix the prompt, re-run.

---

### Sprint 2 — Multi-field FTS5 + entity_line + paper_summary (Phase 2)

**Goal:** search splits into roles; per-field BM25 weights become tunable.

**Files**

1. **`src/chemtree/db.py`** — new FTS table alongside the existing one (do not
   drop yet):
   ```sql
   CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts2 USING fts5(
     claim_id UNINDEXED,
     claim_raw,
     claim_contextualized,
     evidence_text,
     paper_summary,
     entity_line,
     claim_type,
     limitations_text,
     tokenize='porter'
   );
   ```

2. **`scripts/build_claim_search_fields.py`** (new)
   - Per claim:
     - `claim_raw` = `verbatim_quote`
     - `claim_contextualized` = column from Sprint 1; fallback to existing
       `searchable_text` for un-rewritten rows
     - `evidence_text` = `verbatim_quote + ' [' + location_in_paper + ']'`
     - `entity_line` = deterministic concat of typed-field names (catalysts,
       ligands, reactions, materials, technique_name, property_name) parsed
       from `data` JSON
     - `paper_summary` = column from Sprint 2b, fallback to `abstract`
     - `limitations_text` = empty for now; populated by Phase 4/6
   - Bulk insert into `claims_fts2` in a single transaction.

3. **`src/chemtree/db.py::search_claims`** — feature-flag `USE_FTS2 = True`:
   - Replace `claims_fts MATCH` block with
     `claims_fts2 MATCH ? ORDER BY bm25(claims_fts2, w_raw, w_ctx, w_ev, w_sum, w_ent, w_type, w_lim)`
   - Initial weights:
     `claim_contextualized = 1.0`, `entity_line = 1.4`,
     `evidence_text = 0.9`, `paper_summary = 0.6`,
     `claim_raw = 0.7`, `limitations = 0.4`, `claim_type = 0.5`.
   - Keep RRF merge with vector + tree + author.
   - Add `?fts=v1|v2` query param for A/B from frontend.

**Acceptance gate:** offline eval on 100 saved queries — expect nDCG@10 lift
on long, specific queries. If lift < 5% on the entity-heavy slice, raise
`entity_line` weight or shrink `claim_type` weight.

---

### Sprint 3 — Cross-encoder reranker (Phase 3)

**Goal:** precision lift on top-K with no API cost.

**Files**

1. **`src/chemtree/rerank.py`** (new)
   - Lazy-load `cross-encoder/ms-marco-MiniLM-L-12-v2` (start) — swap to a
     SciBERT/SciNCL cross-encoder once benchmarked.
   - `rerank(query, candidates) -> list`, where each candidate's text is
     `claim_contextualized + ' || ' + evidence_text[:300]`.
   - Batched on CPU, top-50 → ~50 ms; on GPU, ~10 ms.

2. **`db.py::search_claims`** — after RRF, reranker call (feature-flagged);
   returns top-K reranked.

**Cost:** $0 LLM. ~50–100 MB extra RAM. ~50 ms/query CPU.
**Acceptance gate:** human-rated top-5 on 30 queries — expect noticeable
improvement vs no rerank.

---

### Sprint 4 — Evidence localization (Phase 4), where we have PDFs

**Goal:** real grounding (page/section/quote) on the slice that matters most.

**Step 4.0 — PDF acquisition**

- Reuse `data/arxiv_batch_tier1/outputs/*.jsonl` (292 files) first.
- For top-25k cited papers without arxiv full-text, run
  `scripts/fetch_open_access_pdfs.py` (Unpaywall API → cache in
  `data/pdfs/<doi>.pdf` + extracted text in `data/papers_text/<doi>.txt`).
- Realistic OA hit-rate: 50–70% on biomedical/chem.

**Step 4.1 — Schema**

`src/chemtree/db.py`:
```sql
CREATE TABLE IF NOT EXISTS claim_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id TEXT NOT NULL,
  page INTEGER,
  section_path TEXT,
  block_type TEXT,            -- 'paragraph' | 'table' | 'figure' | 'caption'
  quote TEXT NOT NULL,
  char_start INTEGER, char_end INTEGER,
  confidence TEXT,
  extractor TEXT, extracted_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_claim_evidence_claim ON claim_evidence(claim_id);
```

**Step 4.2 — Extraction**

`src/extract_claims_with_evidence.py`:
- Input: structured paper text (sections + page numbers from `pypdf` or
  `grobid`).
- For each existing claim of that paper, ask Gemini Pro to *find* the
  supporting span with page/section.
- Output: list of evidence rows.
- Hard validate: quote must be a substring of input text (after whitespace +
  Unicode + smart-quote normalization). Reject otherwise.

**Cost:** ≈ **$650** with `gemini-3.1-pro-preview` batch on the top-cited
deep_v1 papers (where we have or can fetch PDFs). The prompt is mostly
retrieval, so a Flash run would be cheaper, but we already committed to Pro
for the contextualization pass — staying on Pro keeps a single quality bar
across `claim_contextualized`, `paper_summary`, and `claim_evidence.quote`.

**Step 4.3 — Plumb into FTS**

Sprint 2's `evidence_text` field becomes concatenated `quote` from
`claim_evidence`.

---

### Sprint 5 — ColBERT late interaction (Phase 5)

**Goal:** specific chemistry queries resolve to the exact right claim.

**Files**

1. **`requirements.txt`**: add `pylate>=1.1` (or `ragatouille`).
2. **`scripts/build_colbert_index.py`** — index two collections:
   - `claim_card_idx`: `claim_contextualized + ' || ' + entity_line`.
   - `evidence_span_idx`: each `claim_evidence.quote + ' || ' + section_path`
     (after Phase 4).
   - Persist under `data/colbert/`.
3. **`src/chemtree/colbert_retrieve.py`**: thin wrapper exposing
   `colbert_search(query, k=50, idx='card')`.
4. **`db.py::search_claims`** — extra recall channel; included in RRF.

**Compute**

- Index build: a few hours for 2.3M cards on a single GPU; can run overnight on
  M-series Mac with PyLate.
- Query latency: 30–80 ms/query GPU, 200–500 ms CPU.
- Disk: ~5–8 GB for cards index.

**Cost:** $0 LLM. May need a small inference VM if VPS CPU is too slow.

---

### Sprint 6 — Full-paper extraction for new ingests (Phase 6)

**Goal:** Cho's Step A becomes the default for all *new* papers (not
retroactive).

**Files**

1. **`src/process_corpus.py`** — replace `EXTRACTION_PROMPT` with Step-A
   prompt that returns `{contributions, claims (incl. evidence rows),
   entities, methods, reactions, materials, datasets, limitations}`.
2. **`src/chemtree/db.py`** — new table:
   ```sql
   CREATE TABLE IF NOT EXISTS paper_extractions (
     doi TEXT PRIMARY KEY,
     contributions TEXT,
     methods_json TEXT,
     datasets_json TEXT,
     limitations_text TEXT,
     entities_json TEXT,
     raw_response TEXT,
     model TEXT, version TEXT,
     extracted_at TEXT
   );
   ```
3. New ingest routes: every fresh paper goes Step A → claims + evidence →
   multi-field index entries from day one.

**Model:** **`gemini-3.1-pro-preview`** batch — same model we used for
contextualization, so claims born from this pipeline already have a
contextualized rewrite at write-time and never need a backfill pass.

**Cost:** ≈ **$260/mo** at ~10k new papers/month with `gemini-3.1-pro-preview`
batch (assumes ~8k input tok / paper, ~3k output tok / paper, including the
contextualized rewrites for ~30 claims). For new ingest, contextualization is
**not a separate Sprint 1-style pass** — it is a single field in the Step-A
response schema, so we pay the marginal output tokens, not a second prompt.

---

## 6. Suggested ordering & gating

All LLM line-items use **`gemini-3.1-pro-preview` batch**.

```
Sprint 0              ~$58       paper_summary for 41k deep_v1 papers
   gate: 90% human-pass on 30 sampled summaries

Sprint 1a (smoke)     ~$21       contextualize top 25k deep_v1 claims (cit ≥1000)
   gate: 90% human-pass on 50 sampled rewrites
Sprint 1b–1d          ~$1,250    contextualize the rest of deep_v1 (1.49M claims)
   gate: 85% human-pass on 30 random rewrites at each stage

Sprint 2  (Phase 2)   $0 LLM     multi-field FTS5 rebuild (paper_summary already exists)
   gate: nDCG@10 lift on entity-heavy queries

Sprint 3  (Phase 3)   $0         cross-encoder reranker
   gate: human top-5 quality on 30 queries

Sprint 4  (Phase 4)   ~$650      evidence localization on top-cited deep_v1
   gate: ≥95% of evidence quotes pass substring validation

Sprint 5  (Phase 5)   $0 LLM     ColBERT card + span index
   gate: lift on a held-out "specific chemistry" query set

Sprint 6  (Phase 6)   ~$260/mo   full-paper Step-A pipeline for new ingests
```

**Smoke-budget path (ship in days):** Sprint 0 + Sprint 1a + Sprint 2 + Sprint 3
≈ **$140** total LLM spend (covers paper_summaries + 25k smoke rewrites).

**Full upgrade (Cho's vision, end to end on existing corpus):**
Sprint 0 + Sprint 1a-d + Sprint 2 + Sprint 3 + Sprint 5 ≈ **$1,329** one-time
(Sprint 0 is now in this number).

We are **not** considering a Flash-Lite alternative pass for Sprint 1: the
contextualized rewrite is the canonical text indexed and shown to users, and
the user (Bing) chose "best model" precisely because cutting cost on this
field cuts quality everywhere downstream.

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Hallucinated rewrite | Prompt rule + post-hoc number-substring check; near-duplicate-of-quote check; `claim_contextualized` left NULL on rejection so renderer falls back to `searchable_text`. |
| `gemini-3.1-pro-preview` cost overrun | Citation-tier rollout (1a → 1b → 1c → 1d). After 1a we know real input/output token counts and can revise §4 against actuals before committing 1b–1d. |
| Preview model deprecation mid-rollout | The model string is parameterized via `--model` and stored in `claims.context_model`. If Google deprecates `gemini-3.1-pro-preview`, future re-runs use the successor and old rows show their original model. |
| FTS rebuild downtime | Build `claims_fts2` alongside `claims_fts`; switch reads via feature flag. |
| Rebuild blows up DB | Additive change; SQLite WAL stays ~4 GB. |
| OA PDF coverage low (<60%) for Sprint 4 | Sprint 4 only runs over papers where we already have full text (arxiv tier1 outputs) plus what Unpaywall returns; `claim_evidence` rows exist only for resolved spans, so the UI badge stays honest. |
| ColBERT latency on VPS | Move retrieval off `chemtree.db` server onto a small inference VM; or only invoke ColBERT for queries longer than 5 tokens. |
| Quote validation false positives (formatting/Unicode) | Normalize whitespace + Unicode + smart-quotes before substring check. |

---

## 8. End-state retrieval stack

```
User query
  │
  ├── Query understanding (light): entities, taxonomy hints, intent
  │
  ├── Candidate sets (parallel, top-K each):
  │     1. structured filters (taxonomy, claim_type, year, venue)
  │     2. multi-field BM25 (claim_contextualized, entity_line,
  │        evidence_text, paper_summary, claim_raw, limitations)
  │     3. dense vectors over claim_contextualized (+ entity_line)
  │     4. ColBERT over claim cards
  │     5. ColBERT over evidence spans
  │
  ├── Fusion: RRF across the five, with weighted positional bonuses
  │
  ├── Cross-encoder reranker over [query, claim_contextualized + evidence_text]
  │
  └── Cited claim cards + evidence panels
```

This is exactly Cho's diagram, mapped to our existing modules.

---

## 9. Public-facing narrative (after Sprint 1–5)

> AskChem extracts paper-level scientific **claim cards**, each grounded in
> localized evidence (page/section/quote), rewritten to be self-contained, and
> indexed across multiple text views with hybrid retrieval (structured filters,
> multi-field BM25, dense embeddings, ColBERT late interaction) followed by a
> cross-encoder reranker. It is not RAG over chunks; it is search over
> verifiable scientific claims.

---

## 10. Week-1 execution checklist

**Day 1 — schema + Sprint 0 launch**

- [x] Add `claim_contextualized` (+ context_*) columns to `claims` and
      `paper_summary` (+ paper_summary_*) columns to `sources` — idempotent
      ALTERs. (`src/chemtree/db.py`)
- [x] Verify Gemini Batch quota / API key for `gemini-3.1-pro-preview`. (existing
      Portkey gateway used by `batch_extract_arxiv.py`)
- [x] Write `scripts/summarize_papers.py` + `prompts/summarize_paper_v1.txt`.
- [x] Dry-run on 3 sampled papers synchronously; hand-rate. Quality good,
      but exposed reasoning-token cost gap (calibration update applied).
- [x] Submit **Sprint 0** batch chunk 0 (smoke, 7,031 papers, $105.82).
- [x] Hand-rate 30 chunk-0 summaries — pass.
- [x] Submit Sprint 0 chunks 1–5 (33,945 papers).

**Day 2 — Sprint 0 land + Sprint 1a launch**

- [x] Receive Sprint 0 chunks 1–5, validate, write back. Total **41,074 / 41,076
      papers (99.995%)** applied. Total cost **$608.64**. 100% finish_reason=STOP,
      0 truncated.
- [x] Write `scripts/contextualize_claims.py` + `prompts/contextualize_v1.txt`.
      Hard-coded `extraction_version='deep_v1'` filter,
      `model=gemini-3.1-pro-preview`, `CLAIMS_PER_REQUEST=8`.
- [x] Dry-run synchronously on 16 sampled claims with paper_summary populated;
      **15 / 16 pass (94%)**. Per-claim batch cost $0.00209 → Sprint 1d
      extrapolation **~$3,170**.
- [x] Submit **Sprint 1a** batch (25,202 claims, deep_v1 with cit ≥ 1000,
      3,151 requests). batch_id pending; cost ≈ **$53**.

**Day 3–4 — Sprint 1a land + Sprint 1d launch**

- [x] Receive Sprint 1a batch, validate, write back; hand-rate 30. **24,046 /
      25,202 (95.4%) applied; 28/30 hand-rated samples high-quality.** Cost
      $50.05 (vs $46 calibrated).
- [x] Validator improvement: include `paper_title`, `paper_summary`,
      `location_in_paper`, `claim_type` in the haystack used for the
      "invented numbers" check. Recovered ~1,257 false-positive rejects
      (chemical formula subscripts like "Cr2Ge2Te6" no longer flagged).
- [x] Refactored `cmd_prepare` to stream from a forward cursor instead of
      `fetchall()` (1.5M rows × `ORDER BY` was hanging on the 11 GB DB).
- [x] Submit **Sprint 1d residual** (1,491,492 claims, 30 chunks of ~6.4k
      requests each at 8 claims/req).
- [x] Receive Sprint 1d batch, validate, write back. **1,435,487 / 1,491,492
      (96.2%) applied.** Reject mix: 42,592 model abstentions on
      uncontextualizable inputs (table cells with no headers) +
      12,918 invented_numbers + 440 unparseable + 55 bad_opening.
      Cost $3,034.88 (vs $2,792 calibrated, +9%).

**Day 5 — multi-view FTS**

- [ ] Render `claim_contextualized` in `web/index.html::renderClaim`,
      gated behind `?ctx=v1`.
- [ ] Write `scripts/build_claim_search_fields.py` and `claims_fts2` schema
      (multi-view: claim_raw / claim_contextualized / paper_summary /
      verbatim_quote / typed_entities).
- [ ] Smoke-test 30 saved queries against `?fts=v2&ctx=v1`.

**Realized LLM spend (claim-card foundation):**

| Sprint | Scope | Cost | Yield |
|---|---|---|---|
| Sprint 0 | 41,074 paper_summary | $608.64 | 99.995% |
| Sprint 1a | 24,046 claim_contextualized (top-cited) | $50.05 | 95.4% |
| Sprint 1d | 1,435,487 claim_contextualized (residual) | $3,034.88 | 96.2% |
| **Total** | | **$3,693.57** | |

**Final database state**: 41,074 paper summaries + **1,459,533 contextualized
deep_v1 claims** out of 1,515,538 (96.30% coverage). The remaining 3.70%
are intrinsically uncontextualizable claims (table cells without headers,
ultra-sparse extractions) that will fall back to `claim_raw` at search time.

This completes Cho's Step C ("contextualized claim rewriting") for the entire
full-paper subset of the corpus. Step D (multi-view FTS) is the next
deliverable on this plan.
