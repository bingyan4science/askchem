# Sprint C — Chemistry-aware embeddings + cross-encoder rerank

**Date:** 2026-05-02
**Author:** Bing Yan (paired with assistant)
**Status:** Proposed
**Predecessor:** [`2026-05-02-search-quality-upgrade.md`](./2026-05-02-search-quality-upgrade.md) (Sprints A+B done)

This is the larger, higher-leverage piece deferred from the
search-quality plan: replace the general-English `all-MiniLM-L6-v2`
embedder with a chemistry-aware encoder, and add a cross-encoder
reranker as the final stage of `search_claims`. It is roughly a
1.5–2-week effort and 5–10 GB of new on-disk artifacts.

The end-state we are aiming for:

```
User query
  → query understanding (entities, intent)             [unchanged]
  → candidate retrieval
       1. FTS5 over claim text                          [unchanged]
       2. dense vectors over claim cards                [NEW encoder]
       3. tree / paper / author recall                  [unchanged]
  → RRF merge                                            [unchanged]
  → cross-encoder rerank on top-100                     [NEW]
  → final ranked + paginated results
```

When this lands we delete the ad-hoc filters that papered over the
"Suzuki coupling → spin coupling" embedding bug
(`_technique_claim_is_irrelevant_for_coupling_query`, the
`_TREE_WEAK_SINGLE_OVERLAP_STEMS` guard, the `query_signals_…` rule
override). Those are bandaids; the encoder is the actual fix.

---

## 0. Current state — facts on the ground

| Thing | Value |
|---|---|
| Encoder | `all-MiniLM-L6-v2` (384-d, 23 M params, general English) |
| Claims encoded | 2,337,403 |
| `data/claim_embeddings.npz` | 2.4 GB |
| `data/claim_embeddings.faiss` (HNSW, M=32) | 3.0 GB |
| Min semantic score in `search_claims` | 0.20 |
| Vector top-K per query | `max(200, limit*4)` |
| Cross-encoder rerank | none |
| Eval harness | `scripts/eval_search.py` — 39 queries, proxy metrics only (zero-hit, tier-A share, latency). **No labelled relevance.** |

What we built `_claim_to_text` to encode is essentially an
"contextualised claim card" before that term existed: title +
type-specific structured fields + first 300 chars of verbatim. That's
the right unit of indexing — the issue is the encoder, not the unit.

---

## 1. The problem in one paragraph

`all-MiniLM-L6-v2` was trained on web sentences and MS-MARCO; the
strongest signal it has for *coupling* in a chemistry context is the
literal token. So **"Suzuki coupling"** matches "spin–orbit
coupling" / "exciton coupling" / "Josephson coupling" with cosine ≥ 0.6
because both phrases have the same syntactic shape and the same
high-frequency token. Conversely, the model under-rates real chemistry
matches that use synonyms ("Suzuki–Miyaura", "Pd-catalysed boronic
ester coupling"). We've spent two weeks adding rule-based filters to
suppress the false positives; the right move is to embed in a space
where these phrases are not neighbours in the first place.

---

## 2. Plan — five phases

### Phase 0 — Build a labelled eval set (3-4 days)

**Without ground-truth labels we cannot rank candidate models, so this
is the gating dependency.**

1. Pick **80 probe queries** stratified across 6 families:
   - 20× named reactions (Suzuki, Heck, Sonogashira, Negishi, Kumada,
     Stille, Buchwald–Hartwig, Mitsunobu, Wittig, Diels–Alder,
     Mannich, click, ROMP, hydroformylation, hydroboration, C–H
     activation, oxidative addition, transmetalation, β-hydride
     elimination, photoredox).
   - 15× properties (MOF surface area, perovskite Voc, band gap of
     anatase TiO₂, HER overpotential of MoS₂, etc.).
   - 10× materials classes (MXene, ZIF-8, BiVO₄, g-C₃N₄, single-atom
     catalyst, COF, metallocene, NHC, ionic liquid, deep eutectic
     solvent).
   - 10× techniques (XRD, NMR, EPR, XAS, DFT, AIMD, TD-DFT, Raman,
     SEM, BET).
   - 10× hard / homonym / Tier-C target (the cases MiniLM fails on
     today: "Suzuki coupling", "strong coupling" ← negative example,
     "C–H activation iridium", "MOF water splitting", …).
   - 10× multi-concept ("Pd-catalysed coupling under mild conditions",
     "visible-light trifluoromethylation of arenes", …).

2. For each probe, hand-label the **top 20 results** of *any one
   reasonable system* (start with current MiniLM at top-20) on a
   3-point scale:
   - **2 = highly relevant** — the claim answers the query directly.
   - **1 = relevant** — same chemistry, weak match.
   - **0 = irrelevant** — different field / homonym / off-topic.

   80 queries × 20 docs × ~10 s/label ≈ 4–5 h of labelling. To make
   this scale, dump candidates to a JSONL file and label in a Cursor
   Canvas with hotkeys.

3. Persist labels at `data/eval/labels_v1.jsonl` (per-query entries:
   `{q, family, intent, judgments: [{claim_id, score, why}], updated_at}`).

4. Extend `scripts/eval_search.py`:
   - New mode `--labels data/eval/labels_v1.jsonl` that computes
     **nDCG@10, nDCG@20, MRR, Recall@20** in addition to the existing
     proxy metrics.
   - New `--encoder MODEL_NAME` flag that routes through a model
     registry (so re-encoding with a different model = a new run).
   - Output: `scripts/eval_results/<label>.json` (already implemented;
     extend the schema rather than replace).

**Deliverables:** `data/eval/labels_v1.jsonl` (~ 1 600 judgments),
`scripts/eval_search.py` extended with nDCG.

### Phase 1 — Pilot encoder bake-off (1 week)

Goal: pick the production encoder *before* paying the full re-embed
cost.

1. **Stratified pilot corpus.** Sample 200 K claims preserving the
   `claim_type` distribution (`scripts/sample_eval_corpus.py`). Save
   the sample's `claim_id`s alongside the labels so we can run the eval
   on a self-consistent slice.

2. **Encoder candidates** (priority order):

   | Model | Dim | Params | Notes | Reason |
   |---|---:|---:|---|---|
   | `all-MiniLM-L6-v2` | 384 | 23 M | current | baseline |
   | `BAAI/bge-large-en-v1.5` | 1024 | 335 M | top of MTEB-en | strong general |
   | `intfloat/e5-large-v2` | 1024 | 335 M | needs `query: ` / `passage: ` prefixes | strong general |
   | `pritamdeka/S-PubMedBert-MS-MARCO` | 768 | 110 M | biomed-tuned, MS-MARCO | closest to chemistry without fine-tune |
   | `m3rg-iitd/matscibert` | 768 | 110 M | mat-sci paper corpus | needs sentence-level adapter |
   | `mixedbread-ai/mxbai-embed-large-v1` | 1024 (Matryoshka → 256) | 335 M | top of leaderboard, **truncatable** | reduces index size 4× |
   | `nomic-ai/nomic-embed-text-v1.5` | 768 (Matryoshka → 64) | 137 M | 8 K context, **truncatable** | longest doc support |

   Models excluded for now:
   - `text-embedding-3-large` (OpenAI, $45 to encode the corpus, no
     offline iteration).
   - `Linq-Embed-Mistral` (7 B params, 4096-d — index size ~36 GB).

3. **Encode the 200 K sample with each model.** Disk: 7 candidates ×
   200 K × 1024d × 4 B ≈ 5.7 GB total. Wall-clock on Apple-MPS:
   ~30 min per 1024-d model, ~10 min per 768-d. Whole bake-off ≈ 3 h.

4. **Build a temporary FAISS index per encoder** (HNSW, same M=32).

5. **Run the labelled eval** for each:
   ```bash
   for m in mini bge-large e5 pubmedbert matscibert mxbai nomic; do
     python scripts/eval_search.py --run pilot-$m \
       --encoder $m --labels data/eval/labels_v1.jsonl \
       --corpus data/eval/sample_200k.json
   done
   python scripts/eval_search.py --compare pilot-mini --baseline pilot-bge-large
   ```

6. **Decision criteria** (in order):
   - **Primary:** ΔnDCG@10 ≥ +0.05 over MiniLM on the
     reaction + homonym + multi-concept families.
   - **Secondary:** zero-hit count must not regress on the topical
     family.
   - **Tertiary tiebreaker:** Matryoshka-truncatable + 256-d quality
     not noticeably worse → wins (smaller production index).

   If no model clears +0.05 nDCG@10, fall back to ensemble (mxbai +
   pubmedbert via score averaging — implementable but doubles inference
   cost).

**Deliverables:** `scripts/sample_eval_corpus.py`, per-model eval
artifacts in `scripts/eval_results/pilot-*.json`, a 1-page winner
write-up at `docs/plans/2026-05-02-sprint-c-bakeoff-results.md`.

### Phase 2 — Cross-encoder rerank pilot (3 days, in parallel with Phase 1)

This phase is independent of the embedder choice — a cross-encoder
operates on raw text, not embeddings.

1. **Reranker candidates:**

   | Model | Params | Notes |
   |---|---:|---|
   | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 33 M | fast, general-MSMARCO baseline (~5 ms / pair on MPS) |
   | `BAAI/bge-reranker-base` | 278 M | strong general, ~25 ms / pair |
   | `mixedbread-ai/mxbai-rerank-large-v1` | 435 M | top of leaderboard, ~40 ms / pair |
   | `jinaai/jina-reranker-v2-base-multilingual` | 278 M | similar to bge-reranker-base |

2. **Eval setup.** For each labelled query, take the **top-100
   candidates from current production search** (FTS+vector RRF), feed
   `(query, claim_text)` pairs into the cross-encoder, re-sort by score,
   recompute nDCG@10/@20.

3. **Latency budget.** Rerank top-100 must finish under **400 ms p95**
   (current search ~120 ms p50, so total ≤ 600 ms feels acceptable for
   interactive use). That fits MS-MARCO-MiniLM cleanly (~33 ms × 100
   batched on MPS) and is borderline for `mxbai-rerank-large` (~75 ms
   batched).

4. **Decision criteria:**
   - +0.05 nDCG@10 over RRF baseline on at least the homonym +
     multi-concept families.
   - p95 latency budget held.

**Deliverables:** `scripts/eval_rerank.py`, per-rerank
`scripts/eval_results/rerank-*.json`, a recommendation in
`bakeoff-results.md`.

### Phase 3 — Full re-embed and integration (1 week)

Once Phases 1 + 2 picks are in:

1. **Schema and config changes.**
   - In `src/chemtree/embeddings.py`, replace the global constants
     with a `EncoderConfig` class:
     ```python
     @dataclass
     class EncoderConfig:
         name: str
         model_id: str   # e.g. 'BAAI/bge-large-en-v1.5'
         dim: int
         query_prefix: str = ''   # e.g. 'query: ' for e5
         doc_prefix: str = ''     # e.g. 'passage: ' for e5
         normalize: bool = True
         max_seq_len: int = 512
     ```
   - Active config selected by `CHEMTREE_EMBED_VERSION` env var.
     Default: `v1` (current MiniLM). Production: `v2` (winner).
   - File paths become version-suffixed:
     `data/claim_embeddings.v2.npz`,
     `data/claim_embeddings.v2.faiss`.
   - **Both versions coexist on disk** until v2 is fully rolled out.
     Rollback = `unset CHEMTREE_EMBED_VERSION`.

2. **Re-embedding cost (full corpus, 2.34 M claims):**

   | Model | Dim | Wall (MPS) | NPZ size | FAISS size |
   |---|---:|---:|---:|---:|
   | bge-large-en-v1.5 | 1024 | ~4 h | 9.6 GB | ~12 GB |
   | bge-large + Matryoshka 256 | 256 | ~4 h | 2.4 GB | ~3.5 GB |
   | mxbai-embed-large + Matryoshka 256 | 256 | ~4 h | 2.4 GB | ~3.5 GB |
   | pubmedbert-msmarco | 768 | ~2 h | 7.2 GB | ~9 GB |

   Strong preference for a Matryoshka-truncatable model so we can
   ship the 256-d index in production and keep 1024-d on disk for
   future fine-tuning.

3. **Index strategy.** Stay with HNSW Flat (current). Avoid IVF-PQ for
   now: index size is acceptable at 256-d, and the 1-2 % recall hit
   from PQ is not worth the rerank-time complexity.

4. **`search_claims` integration.**
   - After the existing RRF merge produces `merged_ids` (top
     ~`limit*4`), pass the **top-100** candidates to a new
     `chemtree.rerank.cross_encoder_rerank(query, candidate_claim_ids)`.
   - That function fetches `_claim_to_text(claim)` for each candidate,
     batches `(query, doc_text)` pairs through the reranker, returns
     `(claim_id, score)` sorted descending.
   - Re-merge: replace the top-`limit` of `merged_ids` with the
     reranker's top-`limit`.
   - Cap rerank at `n=100` always; if RRF returns fewer, only those.
   - **Skip rerank** when `query_intent == "author"` (rerankers don't
     help author lookups).

5. **Drop the bandaid filters.** With the new encoder + rerank live
   and verified by the eval set:
   - Delete `_technique_claim_is_irrelevant_for_coupling_query` and
     its imports.
   - Delete `_TREE_WEAK_SINGLE_OVERLAP_STEMS` and the guard around it.
   - Keep `query_signals_organic_cross_coupling` as a narrow intent
     hint (it still helps `_get_intent`); but stop using it to filter
     results.

6. **Server warm-up.** Reranker init is ~3 s on MPS. Add a `warmup()`
   call in the FastAPI lifespan so the first user query isn't slow.

**Deliverables:** updated `src/chemtree/embeddings.py`,
`src/chemtree/rerank.py`, `scripts/build_embeddings_v2.py`, the v2
artifacts on disk, `tests/test_rerank.py`.

### Phase 4 — Evaluation, rollout, and rollback (2-3 days)

1. **Final A/B on the full system.** Run `scripts/eval_search.py
   --labels data/eval/labels_v1.jsonl --run prod-v2 --compare baseline`.
   Acceptance:
   - nDCG@10: **≥ +0.10** over baseline.
   - nDCG@10 on the homonym family: **≥ +0.20**.
   - Zero-hit count: **= 0** on topical / acronym families.
   - p95 latency: **≤ 600 ms** on the eval set.

2. **Manual smoke set.** 30 hand-curated queries (Suzuki, Heck,
   ZIF-8, BET surface area, MOF water splitting, …) — top-5 result
   inspected by hand. Sign-off requires zero "obviously wrong" first
   results.

3. **Rollout.** Set `CHEMTREE_EMBED_VERSION=v2` in the systemd /
   launchctl unit. Restart server.

4. **Rollback path.** `unset CHEMTREE_EMBED_VERSION`, restart.
   v1 artifacts stay on disk for 30 days post-rollout.

5. **Post-launch monitoring.** Add a `query_log` table that records
   `(ts, query, intent, top_claim_id, latency_ms, num_results,
   encoder_version, rerank_version)` so we can spot regressions
   without re-running the labelled eval. **No PII** is logged.

---

## 3. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| No candidate beats MiniLM on labelled eval | medium | Fall back to MiniLM + cross-encoder rerank only — that alone typically gives +0.05–0.08 nDCG@10. |
| Cross-encoder p95 latency exceeds budget | medium | Use `ms-marco-MiniLM-L-6-v2` (33 M) instead of bge-reranker (278 M); rerank top-50 instead of top-100. |
| Re-embed Apple-MPS process OOMs | low | Batch size auto-tunes downward; fallback to CPU-only takes ~24 h instead of 4 h. |
| 256-d Matryoshka quality regresses | medium | Keep 1024-d artifacts on disk; flip a config flag to use full-dim if needed. |
| Production storage now > 15 GB | high | Archive v1 artifacts to S3 after 30 days. Quote: 15 GB on a 1 TB SSD = 1.5 % of disk. |
| Embedding text drift (we change `_claim_to_text` later) | medium | Stamp the input-text hash into `EncoderConfig` and into the npz metadata; `update_embeddings` recomputes when the hash changes. |
| Rerank changes search semantics in subtle ways the user dislikes | medium | Keep a `?rerank=0` query-string toggle for power users + the eval suite. |

---

## 4. What this plan is **not**

- **Not a re-extraction pass.** Sprint A's renderer + Sprint B's
  reclassification handle the structured-data quality issue.
  Re-extraction (~$47 K, weeks of API time) is still queued behind
  Sprint C.
- **Not ColBERT or token-level retrieval.** That's the
  2026-04-26-retrieval-upgrade plan; ColBERT requires a separate
  index and a separate eval and is best taken on after Sprint C is
  in production.
- **Not domain fine-tuning.** We are evaluating off-the-shelf
  chemistry/biomed encoders. Domain fine-tune (contrastive on
  ChemTree's own claim pairs) is a follow-on if the off-the-shelf
  best-in-class still under-performs on the homonym family.
- **Not query rewriting via LLM.** Today's normalisation is
  paw-functions + bigram dict. LLM rewriting belongs to a separate
  query-understanding sprint.

---

## 5. Timeline (calendar)

| Phase | Calendar days | Owner | Status |
|---|---:|---|---|
| 0. Labelled eval set | 3 | Bing | not started |
| 1. Encoder bake-off | 7 | Bing | not started |
| 2. Reranker bake-off | 3 (parallel) | Bing | not started |
| 3. Full re-embed + integration | 7 | Bing | not started |
| 4. A/B + rollout | 3 | Bing | not started |
| **Total** | **~17 working days** | | |

---

## 6. Files we will touch

```
src/chemtree/embeddings.py       # EncoderConfig, version-aware paths
src/chemtree/rerank.py           # NEW — cross-encoder reranker
src/chemtree/db.py               # search_claims integration; delete bandaids
scripts/build_embeddings_v2.py   # NEW — full re-embed driver
scripts/sample_eval_corpus.py    # NEW — stratified 200 K sampler
scripts/eval_search.py           # extend: --encoder, --labels, nDCG
scripts/eval_rerank.py           # NEW — reranker A/B harness
data/eval/labels_v1.jsonl        # NEW — hand labels (1 600 judgments)
data/eval/sample_200k.json       # NEW — pilot corpus claim_id list
data/claim_embeddings.v2.npz     # NEW — production embeddings
data/claim_embeddings.v2.faiss   # NEW — production HNSW index
docs/plans/2026-05-02-sprint-c-bakeoff-results.md  # NEW — winner write-up
tests/test_rerank.py             # NEW
tests/test_encoder_config.py     # NEW
```

---

## 7. Open questions to resolve before kickoff

1. **Labelling scale.** 1 600 judgments by hand is a lot. Acceptable
   alternatives: (a) bootstrap labels from MiniLM top-20 + accept that
   pilot Recall measurements are upper-bounded by what MiniLM finds;
   (b) use GPT-5 as a labeller with sampling-audit. Option (b) costs
   ~$30 and finishes in 30 min but introduces LLM-judge bias; we'd
   want Bing to spot-check 100 of the 1 600.
2. **Should we bake in query-side LLM rewriting now?** Probably no —
   it's orthogonal and adds a confounder to the encoder eval.
   Defer to a follow-on sprint.
3. **Open-source vs OpenAI embeddings** — confirmed open-source-only
   for now; OpenAI's pricing makes iteration prohibitive.
4. **Multi-language?** Today's corpus is ~99 % English-language papers.
   Skip multilingual encoders unless this changes.
