# Phase 3 — PAW reranker (Q3 from the search-system roadmap)

*Run date: 2026-05-29. Stratified 297-pair unit bench against
`labels_v1.jsonl` + 80-probe end-to-end nDCG@10 with the top-5 gate.
Following the Phase 1 finding ([2026-05-29-phase1-attribution.md](2026-05-29-phase1-attribution.md))
that rerank is the bottleneck.*

## TL;DR

* **Unit bench: PAW reranker beats MS-MARCO by +13.1 pp accuracy and
  +12.5 pp macro-F1.** R3_std (the 4-class spec on the std mapper
  compiler) is the clear unit winner.
* **End-to-end nDCG@10: flat.** PAW R3_std applied as a top-5 gate
  over MS-MARCO yields 0.787 vs 0.792 baseline (-0.005, within noise).
* **Diagnosis: the top-5 gate is a no-op.** PAW's 3-class output
  ({not, somewhat, highly}_relevant) collapses all 5 cross-encoder
  top candidates to `highly_relevant` -> ties -> MS-MARCO order preserved.
  29 / 80 probes saw any reordering; the wins and losses cancel.
* **Surprising secondary finding: ft variants are WORSE than std for
  the reranker task** — exactly the opposite of the expander task
  (where ft beat std on 9/9). Across R0-R4, std accuracy averaged
  0.589 vs ft 0.531. Same probe set, same specs, same context budget.

| System | Accuracy | Macro F1 | MAE | Pairwise | Avg ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| **R3_std (winner)** | **0.650** | 0.577 | 0.424 | 0.721 | 1248 |
| R1_std | 0.586 | 0.570 | 0.471 | **0.736** | 1279 |
| R2_std | 0.582 | 0.538 | 0.465 | 0.718 | 1301 |
| R4_std | 0.579 | 0.553 | 0.471 | 0.725 | 1194 |
| R0_std | 0.556 | 0.530 | 0.481 | 0.724 | 1437 |
| R0_ft | 0.542 | 0.499 | 0.492 | 0.697 | 1231 |
| R2_ft | 0.542 | 0.480 | 0.492 | 0.696 | 1172 |
| R1_ft | 0.539 | 0.469 | 0.529 | 0.657 | 1116 |
| R3_ft | 0.529 | 0.470 | 0.495 | 0.693 | 1054 |
| R4_ft | 0.502 | 0.468 | 0.545 | 0.666 | 926 |
| **MS-MARCO** baseline | 0.519 | 0.452 | 0.653 | 0.688 | 522 |

## Probe set and spec variants

* **Probes** ([data/eval/paw_reranker_probes.json](../../data/eval/paw_reranker_probes.json)):
  stratified 297 (query, claim) pairs from `labels_v1.jsonl`, balanced
  across the 80 probes and 3 gold scores (158 highly_relevant, 79
  somewhat_relevant, 60 not_relevant). Claim text taken from
  `claim_contextualized` with fallback to `verbatim_quote`, truncated
  to 800 chars.
* **Specs** at [data/paw_specs/reranker/](../../data/paw_specs/reranker/):
  - **R0**: baseline — task description + 15 hand-authored shots (5 per class).
  - **R1**: R0 + 5 negative-style counter-examples (Pd vs Ni, melting point
    vs Tg, GO vs GO-substrate, MOF vs zeolite, cross-coupling background
    rather than result). Per May-28 lesson "negatives over rules".
  - **R2**: R0 + chain-of-thought instruction "first identify the
    central topic, then judge".
  - **R3**: 4-class output (`exact_match` for the strongest matches,
    `highly_relevant`, `somewhat_relevant`, `not_relevant`). Map
    `exact_match -> 2` for scoring against the 3-class gold.
  - **R4**: R1 + R2 combined.
* **Few-shot examples are independent** of `labels_v1.jsonl` (hand-authored
  to avoid train/test leakage).

## Why ft was worse than std (unexpected)

The May-28 expander sweep found ft uniformly beat std (+0.245 mean
macro score on the quick suite). The reranker reverses this. Several
candidate explanations:

1. **The reranker has only 3-4 output tokens.** The finetune compiler
   trains a per-spec LoRA; with so little output entropy, the LoRA
   may be calibrating away from the natural distribution, hurting rare
   classes. The per-class F1 for `not_relevant` drops from std 0.469
   (R0) to ft 0.370 — a clear calibration regression.
2. **The reranker input is heterogeneous and the spec examples are
   not representative.** Expansion inputs are short chemistry queries
   (all ~5-10 tokens); reranker inputs are query+claim pairs where the
   claim varies wildly in topic, length, and style. The finetune may
   overfit to the few-shot example distribution and underperform on
   the held-out probes.
3. **Output token bias.** "highly_relevant" is a frequent token; the
   ft LoRA may push toward it (concentration of mass), again hurting
   rare-class F1.

Diagnostic data on this is light. Future iteration could re-spec with
more examples per class, longer examples spanning the claim-length
distribution, or test the "exact_match / highly_relevant /
somewhat_relevant / not_relevant" 4-class output specifically (R3) on
both compilers to see if the calibration story holds.

## End-to-end with the top-5 gate

Configuration: MS-MARCO cross-encoder ranks all 30 candidates as
before; PAW R3_std re-orders the top-5 only, with tiebreaker = original
cross-encoder score. Behind `CHEMTREE_PAW_RERANK_ID=<id>` and
`CHEMTREE_PAW_RERANK_TOPK=5`.

| Run | nDCG@10 | Δ vs W0 |
| --- | ---: | ---: |
| **W0 baseline** | **0.792** | — |
| W4 R3_std top-5 | 0.787 | -0.005 |
| W5 R1_std top-5 | 0.787 | -0.005 |

R3 and R1 produce **identical top-20** on all 80 probes. The reason is
that PAW's 3-class output collapses all 5 cross-encoder top candidates
to `highly_relevant` (they're already top-5 = high-confidence relevant
by MS-MARCO standards), so PAW's vote is tied across the gate and the
tiebreaker (MS-MARCO score) preserves the original order. The gate is
mostly a no-op.

Per-family deltas vs W0 (R3_std top-5):

| family | W0 | W4 | Δ |
| --- | ---: | ---: | ---: |
| homonym | 0.814 | 0.822 | +0.008 |
| material | 0.736 | 0.758 | +0.022 |
| multi | 0.733 | 0.717 | -0.016 |
| property | 0.823 | 0.832 | +0.009 |
| reaction | 0.849 | 0.822 | -0.027 |
| technique | 0.749 | 0.745 | -0.004 |

Mixed: material and homonym gain modestly, reaction regresses by the
same amount. Net flat.

## Why the unit lift doesn't translate

* **Output granularity mismatch.** Unit bench has 60 not_relevant probes
  that PAW correctly identifies (where MS-MARCO often gets them wrong
  too — both at ~0.47 F1 on this class). But end-to-end, those
  not_relevant claims rarely make it into the top-5 in the first place
  — MS-MARCO already excludes them. PAW's accuracy lift is on a class
  the rerank gate never sees.
* **Top-5 saturation.** By definition the cross-encoder's top-5 is its
  most confident predictions. PAW agrees they're all `highly_relevant`,
  so PAW adds no information.
* **The unit metric overstates the practical lift.** Pairwise
  concordance (0.721 R3_std vs 0.688 MS-MARCO = +0.033) is the more
  predictive proxy, and the gap is small.

## What would make this work

Two avenues that the current run did *not* test:

1. **Wider gate.** Top-10 or top-20 lets PAW see the boundary between
   `highly_relevant` and `somewhat_relevant` candidates. Latency cost:
   80 probes × 20 × 1.3 s = ~35 min for the end-to-end run, vs ~10 min
   for top-5. Tractable.
2. **Finer output classes.** R3 has 4 classes (`exact_match` +
   the original 3) but the ft version regressed. A spec with 5-7
   discrete ranks, plus enough examples per rank, might give PAW
   useful signal within the top-5. ~3 weeks of spec iteration cost.

Neither was attempted in this PR. Both are queued as follow-ups in the
Phase 4 synthesis ([2026-05-30-paw-search-roadmap-decision.md](2026-05-30-paw-search-roadmap-decision.md)).

## Method

* Compile: [scripts/compile_paw_reranker_sweep.py](../../scripts/compile_paw_reranker_sweep.py),
  same retry/sanity pattern as the expander sweep but with the
  `QUERY: <q> CLAIM: <c>` input + 3-probe sanity set.
* Unit bench: [scripts/bench_paw_reranker.py](../../scripts/bench_paw_reranker.py),
  reuses the adapter pattern from `bench_paw_expander.py`.
* End-to-end wiring: new `CHEMTREE_PAW_RERANK_ID` and
  `CHEMTREE_PAW_RERANK_TOPK` env-gated path in
  [src/chemtree/db.py](../../src/chemtree/db.py) `search_claims`. When
  the env vars are set, the top-K of the cross-encoder's output is
  re-ordered by PAW with the cross-encoder score as tiebreaker.
* End-to-end run: [scripts/eval_search_attribution.py](../../scripts/eval_search_attribution.py)
  + [scripts/eval_metrics.py](../../scripts/eval_metrics.py), same
  pattern as Phase 1.

Artefacts:

```
data/paw_specs/reranker/R{0..4}.txt
data/paw_expander_variants.json (10 reranker entries appended)
data/eval/paw_reranker_probes.json (297 stratified pairs)
data/eval/runs/paw_reranker_bench_std.json (unit, 5 std variants)
data/eval/runs/paw_reranker_bench_ft.json (unit, 5 ft variants)
data/eval/runs/attribution_W4_paw_rerank_R3std_top5.jsonl + .scored.json
data/eval/runs/attribution_W5_paw_rerank_R1std_top5.jsonl + .scored.json
```

Reproducer:

```bash
# Compile (std fast, ft slow)
.venv-benchmark/bin/python scripts/compile_paw_reranker_sweep.py \
    --specs R0 R1 R2 R3 R4 --compiler std
.venv-benchmark/bin/python scripts/compile_paw_reranker_sweep.py \
    --specs R0 R1 R2 R3 R4 --compiler ft

# Unit bench (~35 min std, ~30 min ft)
PYTHONPATH=src .venv-benchmark/bin/python scripts/bench_paw_reranker.py \
    --variants-registry data/paw_reranker_variants_ft.json \
    --out data/eval/runs/paw_reranker_bench_ft.json

# End-to-end (~20 min per config)
CHEMTREE_PAW_RERANK_ID=fec982ab5bf88e634f92 \
CHEMTREE_PAW_RERANK_TOPK=5 \
  PYTHONPATH=src .venv-benchmark/bin/python \
    scripts/eval_search_attribution.py --label W4_paw_rerank_R3std_top5
```

Total wall time for this Phase 3: ~3.5 hours (compile + bench + 2
end-to-end runs).
