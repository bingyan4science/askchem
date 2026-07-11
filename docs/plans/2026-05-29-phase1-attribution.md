# Phase 1 — Does keyword expansion translate to nDCG@10?

*Run date: 2026-05-29. Instrumented attribution + 4-config wiring sweep
on the 80-probe v1 eval set, via the
[../../../scripts/eval_search_attribution.py](../../scripts/eval_search_attribution.py)
harness wired through a new `_trace_into` kwarg in
`db.search_claims` and an env-gated `CHEMTREE_PAW_REWRITES_RERANK=1`
augmented-query path through `cross_rerank`.*

## TL;DR

**Answer: No. On the current pipeline, expansion does not lift
end-to-end search quality — and the deeper we wire PAW into the
ranking path, the more nDCG@10 *drops***.

| Config | wiring | nDCG@10 | Δ vs W0 |
| --- | --- | ---: | ---: |
| **W0 baseline** | PAW off, no rewrites | **0.792** | — |
| W1 PAW -> FTS only | `CHEMTREE_PAW_REWRITES=1` (May-23 wiring) | 0.787 | -0.005 |
| W2 PAW -> FTS + rerank input | + `CHEMTREE_PAW_REWRITES_RERANK=1` | 0.774 | **-0.018** |
| W3 W2 + rerank window 50 | + `CHEMTREE_RERANK_WINDOW=50` | 0.765 | **-0.027** |

The May-23 plan ([docs/plans/2026-05-23-paw-ft-rewrites.md](2026-05-23-paw-ft-rewrites.md))
had already observed Δ = -0.001 from W0 -> W1; today's deeper wiring
W2/W3 confirms the pattern is structural, not noise. **Per-probe**:
W2 worsens 37 / 80 probes, improves 20, and is unchanged on 23. Losses
outweigh wins on both count and magnitude.

This is the evidence we needed to **stop investing in expander spec
iteration** (V9 / V10 follow-ups from the May-28 plan are now firmly
de-prioritised) and **pivot to Phase 2/3** of the search roadmap.

## What the per-probe data shows

Top losses (W2 vs W0):

| Δ nDCG@10 | probe | family | query |
| ---: | --- | --- | --- |
| -0.206 | tech-03 | technique | EPR spectroscopy radical species detection |
| -0.199 | multi-06 | multi | enantioselective Mannich reaction proline organocatalyst |
| -0.192 | multi-01 | multi | visible-light photoredox trifluoromethylation arenes |
| **-0.181** | rxn-01 | reaction | **Suzuki-Miyaura cross-coupling** |
| -0.177 | multi-13 | multi | nickel-catalyzed cross-coupling aryl chloride |
| **-0.148** | rxn-10 | reaction | **Diels-Alder cycloaddition** |
| -0.122 | rxn-11 | reaction | Mannich reaction enantioselective |
| -0.109 | mat-04 | material | g-C3N4 visible light photocatalyst |

Top wins:

| Δ nDCG@10 | probe | family | query |
| ---: | --- | --- | --- |
| +0.275 | mat-05 | material | single atom catalyst oxygen reduction reaction |
| +0.221 | mat-03 | material | BiVO4 photocatalyst water oxidation |
| +0.180 | tech-02 | technique | solid-state NMR characterization MOF |
| +0.150 | hom-10 | homonym | phase transition liquid crystal smectic |
| +0.093 | tech-06 | technique | ab initio molecular dynamics simulation |
| +0.085 | hom-01 | homonym | Suzuki coupling |

## Why expansion hurts the rerank

The augmented query for `Suzuki-Miyaura cross-coupling` becomes:

```
Suzuki-Miyaura cross-coupling Suzuki-Miyaura cross-coupling palladium Pd boronic acid aryl halide SPhos XPhos
```

For most-relevant Suzuki claims, this is **worse** than the bare query
because:

1. The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was trained
   on MS-MARCO web search pairs — natural-language queries against
   passage text. Appending a comma-less keyword dump shifts the input
   distribution away from what the model was trained on. The cross-encoder
   under-scores everything.
2. The bare query already matches the canonical Suzuki claims at the
   top of the rerank window perfectly. There's no room for expansion to
   add lift; there's only room for it to subtract.
3. PAW's added terms (`Pd, SPhos, XPhos, boronic acid`) appear in the
   claim text of *some* Suzuki claims but not *all* — so the augmented
   query slightly favors the subset that lists ligands over the subset
   that doesn't. That shuffles top-10 in a way that often picks worse
   claims.

The wins are real but constrained to **vocabulary-gap queries** where
the bare query misses key terms (`single atom catalyst -> Pt-free, M-N-C,
Fe-N-C`; `BiVO4 photocatalyst -> photoanode, oxygen evolution`). On those
probes, expansion does its intended job — but they're outnumbered ~2:1 by
queries where bare-query relevance is already saturated and expansion
only adds noise.

## What the attribution categories say

The harness reports every single probe as "unaffected" across all 4
configs — meaning the *best* judged-positive claim always landed in
top-10. Two interpretations:

1. **There is no recall-bounded probe in the 80-probe set under v2-256
   + cross-encoder.** Every probe has a relevant claim recalled
   somewhere in top-10. The dense channel (mxbai-256) plus cross-encoder
   rerank is strong enough that recall is not the bottleneck. Expansion
   has no recall headroom to capture.
2. **The category logic is too lenient.** "Any positive in top-10" is a
   weak bar; the interesting case is "what fraction of the top-10
   highly-relevant claims actually make it to top-10". That refinement
   would discriminate better but doesn't change the conclusion — the
   nDCG@10 metric already captures the relevant signal.

Both conclusions point the same direction: the rerank is the bottleneck,
not recall.

## Implications for the rest of the roadmap

Phase 1's purpose was to gate whether more expander spec iteration was
worthwhile. The answer is **no**: with the current MS-MARCO cross-encoder
in the rerank slot, no amount of better expansion will lift nDCG@10 —
and may hurt it.

Two avenues remain open:

1. **Replace or supplement the cross-encoder with a PAW-based scorer**
   trained on the same `labels_v1.jsonl` judgements (Phase 3 of the
   roadmap). A PAW relevance scorer that scores `(query, claim)` pairs
   would: (a) see the bare query, avoiding the augmented-query bias;
   (b) be fit to AskChem's actual relevance distribution, not generic
   web search; (c) potentially handle the chemistry-specific
   vocabulary that MS-MARCO miss.
2. **Find PAW use cases outside the rerank-bounded segment**, such as
   within-paper claim selection or view prediction (Phase 2 audit).

The V9 / V10 expander spec candidates from the May-28 plan are
**de-prioritised** until a PAW-friendly reranker exists; without one,
better expansion has no path to nDCG@10.

## Method

* Instrumented [src/chemtree/db.py](../../src/chemtree/db.py)
  `search_claims` with a `_trace_into: dict | None` kwarg that records
  intermediate pools (FTS / vector / RRF / rerank in/out / final). No
  behaviour change when `_trace_into` is None.
* Added `CHEMTREE_PAW_REWRITES_RERANK=1` env-gated path in the rerank
  block: when set, builds an augmented query `bare + " ".join(paw_expand(bare)[:8])`
  and passes it to `cross_rerank` instead of the bare query. Default
  unset = no behaviour change, matches May-23 wiring.
* Driver: [scripts/run_attribution_sweep.sh](../../scripts/run_attribution_sweep.sh)
  runs 4 configs sequentially, each in its own process so PAW
  lazy-singletons don't bleed.
* Harness: [scripts/eval_search_attribution.py](../../scripts/eval_search_attribution.py)
  + [scripts/eval_attribution_summary.py](../../scripts/eval_attribution_summary.py).
* Scoring reuses [scripts/eval_metrics.py](../../scripts/eval_metrics.py)
  on the `attribution_<label>.jsonl` files (the harness writes
  `ranked_claim_ids` alongside the trace fields).

Reproducer:

```bash
bash scripts/run_attribution_sweep.sh
```

Total wall time: ~43 min (most of it the 4 × 80-probe runs; PAW expand
adds ~2.5 s/probe to W1/W2/W3).

Artefacts:

```
data/eval/runs/attribution_W0_baseline.jsonl
data/eval/runs/attribution_W1_paw_fts.jsonl
data/eval/runs/attribution_W2_paw_rerank.jsonl
data/eval/runs/attribution_W3_paw_rerank_w50.jsonl
data/eval/runs/attr_W{0,1,2,3}_*.scored.json
```

## Risks and follow-ups

* **The attribution category is too lenient** (everything `unaffected`).
  Future iterations should refine to "fraction of top-3 judged-high
  claims in top-3 final" to surface the rank-within-top-10 differences
  that drive the actual nDCG@10 movements. Not done in this PR because
  it doesn't change the W0 < W1 < W2 < W3 monotone direction.
* **MS-MARCO MiniLM is a specific failure mode**, not necessarily a
  universal one. A different reranker (PAW, custom-trained, or a
  larger general-purpose model) might handle augmented queries better.
  Phase 3 (PAW reranker) tests exactly this.
* **labels_v1 was Gemini-judged.** A PAW reranker fit to those labels
  inherits Gemini's relevance notion. Useful for ablation but ship
  decisions should ideally cross-check with human spot-checks.
