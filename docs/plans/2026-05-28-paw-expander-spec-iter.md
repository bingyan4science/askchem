# May-28 PAW expander spec iteration

*Run date: 2026-05-28. Local Mac (CPU PAW via `GGML_NO_METAL=1`), two-stage compiler protocol on the May-27 component bench
([docs/plans/2026-05-27-paw-expander-bench.md](2026-05-27-paw-expander-bench.md)).*

## TL;DR

Authored 9 spec variants along three axes (examples, task style, output format), swept them on the fast std mapper compiler against the 12-probe quick suite, promoted the top 3 to the slow ft finetune compiler, and benched on the full 30-probe suite. **`V3_ft` (20 positive + 5 negative examples, no rule block, `paw-ft-bs48-20260522`) clears the ship gate cleanly with a macro score of +0.30 vs the May-23 `paw_ft` baseline at +0.19 — a +0.11 lift with no family regression greater than 0.02.** Shipped:

- `QUERY_EXPANDER_PROGRAM_ID` at [src/chemtree/paw_functions.py](../../src/chemtree/paw_functions.py) L20 updated to `23d74e49bcb1ff445a7d` (V3_ft).
- [data/paw_ft_program_ids.json](../../data/paw_ft_program_ids.json) `expand` entry updated to match.

`V7_ft` (V3 + a rule block) produced a bigger raw lift (+0.20 macro) but failed the per-family regression gate — its rules over-restricted the model on canonical queries (`graphene oxide reduction` dropped the canonical `rGO / Hummers method / thermal reduction` vocabulary; `EIS` replaced `Warburg` with a non-standard `Colebrook plot`). Documented as a "follow-up: rules need calibration."

| System | Macro score | Coverage | Pollution | Degeneracy | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **V3_ft (shipped)** | **+0.30** | 0.35 | 0.02 | 0.07 | 20 pos + 5 neg examples |
| V7_ft | +0.39 | 0.43 | 0.00 | 0.07 | 20 pos + 5 neg + rules; fails family-regression gate |
| V8_ft | +0.18 | 0.26 | 0.02 | 0.12 | 20 pos + chain-of-thought instruction |
| paw_ft (V0, prior baseline) | +0.19 | 0.30 | 0.08 | 0.08 | 20 pos (current production until this PR) |
| static (db dict) | +0.05 | 0.12 | 0.00 | 0.13 | floor baseline |

## The 9 variants

Specs in [data/paw_specs/expander/V0.txt](../../data/paw_specs/expander/V0.txt) ... `V8.txt`. Same comma-separated output format throughout, so the downstream FTS wiring in `db.expand_query_variants` is unchanged.

| ID | Examples | Style | Hypothesis |
| --- | --- | --- | --- |
| V0 | 20 pos (production) | current | Baseline |
| V1 | zero-shot | current | Do we need examples at all? |
| V2 | 5 pos (diverse) | current | Fewer shots = less bias |
| V3 | 20 pos + 5 neg | current | Targeted counter-examples for May-23 losers |
| V4 | 10 pos (balanced) | current | Rebalance examples to match probe stratification |
| V5 | 20 pos | with-rules | Add explicit "do not propose Pd for Ni queries" rules |
| V6 | zero-shot | brief | Minimal-prompt baseline |
| V7 | 20 pos + 5 neg | with-rules | V3 + V5 stacked |
| V8 | 20 pos | chain-of-thought | "Identify topic, then list synonyms" prompt |

Negative examples (for V3, V7) target the five May-23 / May-27 loser queries — Mannich, nickel-cross-coupling, Tg, Grubbs, spin coupling. Rule block (for V5, V7) names three failure-mode patterns explicitly. See the spec files for the exact text.

## Stage 1: std-mapper sweep (12-probe quick suite)

Compile time: ~5 min total (9 × ~30 s). Bench time: ~30 min (dominated by std-mapper degeneracy).

Artefacts: [data/eval/runs/paw_expander_sweep_std.json](../../data/eval/runs/paw_expander_sweep_std.json).

| Variant | Score | Cov | Pol | Deg | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **V7_std** | **+0.16** | 0.35 | 0.06 | 0.25 | 10,390 |
| **V3_std** | **+0.08** | 0.24 | 0.00 | 0.32 | 19,849 |
| static | +0.04 | 0.11 | 0.00 | 0.14 | 3 |
| V8_std | -0.05 | 0.14 | 0.04 | 0.31 | 9,805 |
| V0_std | -0.07 | 0.15 | 0.00 | 0.44 | 22,432 |
| V5_std | -0.07 | 0.18 | 0.09 | 0.32 | 10,303 |
| V4_std | -0.10 | 0.17 | 0.04 | 0.45 | 16,506 |
| V6_std | -0.15 | 0.07 | 0.00 | 0.43 | 5,310 |
| V2_std | -0.16 | 0.14 | 0.06 | 0.49 | 14,627 |
| V1_std | -0.22 | 0.09 | 0.00 | 0.62 | 31,249 |

Findings:

- **The `degeneracy < 0.20` hard gate from the plan is too strict for std-mapper output** — every std variant lands in 0.25-0.62. Applied a relaxed `deg < 0.35` gate consistent with the empirical floor and ranked by macro score.
- **Zero-shot (V1, V6) collapses.** Both have catastrophic degeneracy (0.62 / 0.43) and the lowest macro scores. The std mapper needs few-shot scaffolding to produce usable output.
- **Rules without examples (V5) ≈ V0 baseline.** Adding rules alone doesn't help; the std compiler doesn't reliably follow them.
- **Negative examples (V3, V7) are the clear discriminator.** V7 (+rules +neg) tops the table; V3 (just neg) is second. The Stage-1 finding directly motivated the Stage-2 promotion of V3, V7, V8 (chain-of-thought as the third axis).

Note: `V0_std` content-addressed to `d442088a6063deb9f42a` — bit-identical to the production `paw_std` shipped today.

## Stage 2: ft-finetune promotion (30-probe full suite)

Compile time: ~3 min × 3 variants ≈ 10 min. Bench time: ~3 min (no degeneracy).

Artefacts: [data/eval/runs/paw_expander_sweep_ft.json](../../data/eval/runs/paw_expander_sweep_ft.json), [data/paw_expander_variants.json](../../data/paw_expander_variants.json).

| System | Score | Cov | Pol | Deg |
| --- | ---: | ---: | ---: | ---: |
| **V7_ft** | +0.39 | 0.43 | 0.00 | 0.07 |
| **V3_ft** | +0.30 | 0.35 | 0.02 | 0.07 |
| paw_ft (V0_ft, prior baseline) | +0.19 | 0.30 | 0.08 | 0.08 |
| V8_ft | +0.18 | 0.26 | 0.02 | 0.12 |
| static | +0.05 | 0.12 | 0.00 | 0.13 |

### Ship-gate evaluation

| Gate criterion | V3_ft | V7_ft | V8_ft |
| --- | --- | --- | --- |
| Macro ≥ baseline + 0.05 (baseline +0.19) | +0.30, **PASS** (+0.11) | +0.39, **PASS** (+0.20) | +0.18, FAIL (-0.01) |
| No family regression > 0.10 | max -0.017 (material), **PASS** | neutral -0.178, technique -0.116, **FAIL** | neutral -0.271, technique -0.188, FAIL |
| Multi pollution ≤ 0.10 | 0.074, **PASS** | 0.030, PASS | 0.085, PASS |
| Degeneracy ≤ 0.15 | 0.07, **PASS** | 0.07, PASS | 0.12, PASS |
| **Overall** | **PASS — ship** | FAIL (regression gate) | FAIL (macro + regression) |

V3_ft is the only candidate that clears all four gates.

### Per-family breakdown vs prior `paw_ft` baseline

| family | n | paw_ft | **V3_ft** | Δ | V7_ft | Δ | V8_ft | Δ |
| --- | -: | -: | -: | -: | -: | -: | -: | -: |
| homonym | 4 | +0.012 | +0.022 | +0.010 | +0.291 | +0.279 | -0.015 | -0.027 |
| material | 5 | +0.225 | +0.213 | -0.012 | +0.208 | -0.017 | +0.219 | -0.006 |
| multi | 3 | -0.068 | +0.185 | **+0.253** | +0.525 | +0.593 | +0.028 | +0.096 |
| neutral | 4 | +0.640 | +0.640 | 0.000 | +0.462 | **-0.178** | +0.369 | **-0.271** |
| property | 4 | -0.009 | +0.225 | **+0.234** | +0.279 | +0.288 | +0.050 | +0.059 |
| reaction | 6 | +0.041 | +0.332 | **+0.291** | +0.608 | +0.567 | +0.344 | +0.303 |
| technique | 4 | +0.461 | +0.511 | +0.050 | +0.345 | **-0.116** | +0.273 | **-0.188** |

V3_ft is the only variant that lifts the May-23 problem families (`multi`, `property`, `reaction` all up by +0.23 to +0.29) **without** regressing the easy families. V7_ft and V8_ft both deliver bigger reaction/multi lifts but at the cost of dropping the canonical `neutral` and `technique` queries — the rule block in V7 and the chain-of-thought wrapper in V8 both make the model too conservative on well-trodden vocabulary.

### Biggest wins (V3_ft vs prior `paw_ft`):

| Δ score | probe | query |
| ---: | --- | --- |
| **+1.22** | prop-tg | polymer glass transition temperature Tg |
| **+1.21** | rxn-grubbs | olefin metathesis Grubbs catalyst |
| **+0.67** | multi-ni-cc-arcl | nickel-catalyzed cross-coupling aryl chloride |
| **+0.52** | rxn-mannich | Mannich reaction enantioselective |
| +0.30 | hom-spin-coupling | spin coupling magnetic material |

All five biggest wins are May-23 / May-27 documented losers. The Suzuki spam pattern is gone on these: V3_ft now produces `DSC, DMA, Fox equation` for Tg, `RCM, ROMP, ruthenium, NHC` for Grubbs, `Ni, Negishi, Kumada, organozinc` for Ni-cross-coupling.

### Biggest regressions (V3_ft vs prior `paw_ft`):

| Δ score | probe | query | severity |
| ---: | --- | --- | --- |
| -0.35 | hom-ch-activation | C-H activation | local; on a probe paw_ft was already weak on |
| -0.25 | prop-plqy-oled | OLED photoluminescence quantum yield | both still positive |
| -0.16 | rxn-transmetalation | transmetalation cross-coupling mechanism | both still positive |
| -0.07 | mat-des | deep eutectic solvent biomass dissolution | within noise |
| -0.07 | tech-tem-np | transmission electron microscopy nanoparticle morphology | within noise |

Largest regression (-0.35 on `hom-ch-activation`) is on a probe where paw_ft was at +0.222 and V3_ft at -0.125; both negative-or-low, no canonical vocabulary is at stake here. Acceptable trade vs the +1.21 / +1.22 wins on Grubbs and Tg.

## Why V7_ft failed despite the bigger lift

Spot-checked the V7_ft regression cases via [data/eval/runs/paw_expander_sweep_ft.json](../../data/eval/runs/paw_expander_sweep_ft.json):

* `neu-go-reduction` ("graphene oxide reduction"):
  * paw_ft output: `GO, rGO, reduced graphene oxide, Hummers method, thermal reduction, chemical reduction` — canonical.
  * V7_ft output: `GO, graphene oxide, redox, reduction, hydrolysis, water oxidation, oxidation, electron transfer` — dropped `rGO / Hummers / thermal reduction` entirely. The "do not introduce vocabulary from unrelated reaction classes" rule appears to be over-firing.
  * V3_ft output: identical to paw_ft.

* `tech-eis` ("electrochemical impedance spectroscopy"):
  * paw_ft: `EIS, Nyquist plot, charge transfer resistance, Warburg, equivalent circuit` — canonical.
  * V7_ft: `EIS, Nyquist plot, charge transfer resistance, Colebrook plot, impedance, frequency response` — "Colebrook plot" is fluid-mechanics jargon, not electrochemistry. The rules don't catch this.
  * V3_ft: identical to paw_ft.

V7's failure mode is: **rules nudge the model toward conservatism that costs canonical vocabulary on easy queries**. The rules help Mannich (huge +0.83 win) but hurt graphene-oxide-reduction (huge -0.81 loss). Net macro is still positive (+0.20) but the per-family regression gate (correctly) flags this as unsafe to ship — those neutral / technique families are exactly the families AskChem can't afford to regress on, because they have the most user traffic.

## Cross-stage compiler transfer (initial, top-3 only)

| variant | std score (quick) | ft score (full) | rank-preserved? |
| --- | ---: | ---: | --- |
| V7 | +0.16 (1st of std) | +0.39 (1st of ft) | yes |
| V3 | +0.08 (2nd of std) | +0.30 (2nd of ft) | yes |
| V8 | -0.05 (3rd of std) | +0.18 (3rd of ft) | yes |

The top-3 std rank held on ft, which would have been encouraging — but with only 3 data points, it doesn't generalise. See the addendum below for the full 9-variant rank comparison, which substantially weakens the rank-preservation claim.

The std → ft *delta* per variant is more robust: each ft variant lifts its std-stage score by +0.22 to +0.23. The finetune compiler adds a roughly constant quality bonus on top of whatever spec it's given, suggesting the spec design and the compiler are independent levers.

## Shipped

1. [src/chemtree/paw_functions.py](../../src/chemtree/paw_functions.py) L20: `QUERY_EXPANDER_PROGRAM_ID = "23d74e49bcb1ff445a7d"` (was `d442088a6063deb9f42a`).
2. [data/paw_ft_program_ids.json](../../data/paw_ft_program_ids.json) `expand` entry: `program_id` updated, `spec_file` / `spec_stem` / `notes` added.
3. Full sweep registry at [data/paw_expander_variants.json](../../data/paw_expander_variants.json) (12 entries: 9 std + 3 ft).
4. Spec files at [data/paw_specs/expander/V0.txt](../../data/paw_specs/expander/V0.txt) ... `V8.txt`.
5. Sweep driver at [scripts/compile_paw_expander_sweep.py](../../scripts/compile_paw_expander_sweep.py).
6. Bench extension at [scripts/bench_paw_expander.py](../../scripts/bench_paw_expander.py) (`--variants-registry` flag).
7. Quick probe suite at [data/eval/paw_expander_probes_quick.json](../../data/eval/paw_expander_probes_quick.json).

Tests still green: 115/115 (94 pre-existing + 21 from the May-27 bench harness).

## Follow-ups (out of scope for this PR)

1. **Calibrate V7's rules.** The "do not introduce vocabulary from unrelated reaction classes" rule is too aggressive for queries like graphene oxide reduction. A scoped version ("...unless the query is itself about reduction / oxidation chemistry") might recover the V7 macro lift while preserving the canonical vocabulary V3_ft already gets right. Next iteration would be V9 = V3 + scoped-rules.
2. **Augment the static dictionary** with the May-23 / May-27 gap entries (`transmetalation cross-coupling`, `nickel-catalyzed cross-coupling`, `polymer glass transition`, `olefin metathesis`) per the May-27 follow-up. The bench shows static at +0.05 macro — even with V3_ft shipped, the static path will continue to fire as a parallel signal in `expand_query_variants` and a more complete dict reduces hallucination risk on adjacent queries.
3. **Re-run end-to-end nDCG@10 only when the May-23 wiring is re-enabled** (it ships behind `CHEMTREE_PAW_REWRITES=1`, default off). The May-27 cross-check showed nDCG@10 disagrees with the unit metric on ~38 % of overlapping probes (rerank ceiling), so the unit-bench lift documented here may or may not translate end-to-end — but the rerank ceiling is a separate problem from the spec quality.
4. **Same two-stage protocol applies to decompose / normalize / contradiction** PAW programs. The bench harness, registry format, and sweep driver are already generic enough to support those tracks with a new probe-set JSON.

## Method (reproducer)

```bash
# Author specs at data/paw_specs/expander/V0..V8.txt.

# Stage 1 (std mapper, ~30 min including bench):
.venv-benchmark/bin/python scripts/compile_paw_expander_sweep.py \
    --specs V0 V1 V2 V3 V4 V5 V6 V7 V8 --compiler std
PYTHONPATH=src .venv-benchmark/bin/python scripts/bench_paw_expander.py \
    --probes data/eval/paw_expander_probes_quick.json \
    --systems static \
    --variants-registry data/paw_expander_variants.json \
    --out data/eval/runs/paw_expander_sweep_std.json

# Pick top 3 by macro score (V3, V7, V8).

# Stage 2 (ft finetune, ~13 min including bench):
.venv-benchmark/bin/python scripts/compile_paw_expander_sweep.py \
    --specs V3 V7 V8 --compiler ft
PYTHONPATH=src .venv-benchmark/bin/python scripts/bench_paw_expander.py \
    --probes data/eval/paw_expander_probes.json \
    --systems ft,static \
    --variants-registry data/paw_expander_variants_ft_stage2.json \
    --out data/eval/runs/paw_expander_sweep_ft.json

# Apply ship gate, update QUERY_EXPANDER_PROGRAM_ID.
```

Total wall time: ~1 hour, dominated by the std-mapper bench (degenerate output is slow to generate). Compare to ~30 min if the std stage were skipped and all 9 went straight to ft (with worse rank certainty).

## Addendum (same day): comprehensive ft sweep

After the initial 3-variant ft promotion, ran the remaining 6 variants on ft as well to test the rank-preservation hypothesis with more data points. Compile time ~10 min; bench time ~14 min on the full 30-probe suite.

Artefacts: [data/eval/runs/paw_expander_sweep_ft_full.json](../../data/eval/runs/paw_expander_sweep_ft_full.json), [data/paw_expander_variants_ft_all.json](../../data/paw_expander_variants_ft_all.json).

### Full ft ranking (all 9 variants)

| ft rank | variant | macro score | std rank | rank Δ | notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | V7_ft | **+0.388** | 1 | 0 | rules + neg + 20pos |
| 2 | **V4_ft** | **+0.321** | 6 | **+4** | **10-pos balanced — surprise winner** |
| 3 | **V3_ft (shipped)** | **+0.304** | 2 | -1 | 20pos + 5neg |
| 4 | V5_ft | +0.288 | 5 | +1 | 20pos + rules |
| 5 | V2_ft | +0.221 | 8 | +3 | 5pos diverse |
| 6 | V0_ft (paw_ft baseline) | +0.186 | 4 | -2 | 20pos current |
| 7 | V8_ft | +0.181 | 3 | -4 | chain-of-thought |
| 8 | V1_ft | +0.025 | 9 | +1 | zero-shot |
| 9 | V6_ft | -0.091 | 7 | -2 | minimal prompt |

Pairwise rank agreement std → ft: **25/36 (69 %)**. The original "top-3 perfectly preserved" finding was lucky. The std mapper proxy is informative but noisy at the variant level.

The std → ft *score* lift is robust though: average +0.27, std-dev 0.10. V4 had the largest jump (+0.42), V6 the smallest (+0.06). So the finetune compiler adds quality everywhere, but how much depends on the spec.

### Full ship-gate evaluation

| Variant | Macro Δ vs baseline | Family regressions (>0.10) | Multi pollution | Degeneracy | Ship gate |
| --- | ---: | --- | ---: | ---: | --- |
| **V3_ft** | **+0.118** | **none** | 0.074 | 0.065 | **PASS** |
| V4_ft | +0.135 | neutral -0.248 | 0.095 | 0.053 | FAIL |
| V5_ft | +0.102 | neutral -0.150 | **0.120** | 0.086 | FAIL (also pollution) |
| V7_ft | +0.202 | neutral -0.179, technique -0.116 | 0.030 | 0.069 | FAIL |

**V3_ft is the only variant of all 9 that passes the strict ship gate.** This is not coincidence — V3 keeps all 20 V0 training shots intact and just appends 5 negative examples, so the canonical vocabulary for the neutral-family probes (heavy metal adsorption, water splitting, graphene oxide reduction, CRISPR) is preserved exactly. V4 throws out half the V0 shots; V5/V7 add rules that nudge the model away from canonical vocab. V3's "extend, don't replace" pattern is what makes it neutral-safe.

V4_ft is a striking near-miss: macro +0.32 (best after V7), but the rebalanced 10-shot examples drop the neutral family canonical vocab (`neu-go-reduction` regresses from +0.713 to +0.075 vs baseline). The "balanced exemplars" hypothesis was correct that it improves family coverage in problem areas, but it overcorrects on the easy queries.

### Updated conclusions

* The ship decision (V3_ft, `23d74e49bcb1ff445a7d`) **is unchanged by the full sweep**. The decision was right.
* The two-stage protocol's rank-preservation claim should be **moderated to "useful filter, not authoritative rank"**. The std → ft scoring is informative (no variant ranked outside the top-5 in std jumped to ft-rank 1 or 2), but precise rank assignment changed substantially between stages.
* Future spec iterations should probably **compile all candidates on ft** if compile budget allows (~30 min for 9 variants is cheap relative to the value of not missing a V4-style surprise).
* The empirically robust design pattern is **"keep training shots intact, append corrections"**. V3 demonstrates this; future variants (V9 = V3 + scoped rules; V10 = V3 + V4's reaction-class additions) should follow it.

### Updated follow-ups

1. **V9 candidate: V3 + scoped rules.** The V7 rules were too aggressive on neutral; a scoped version ("don't propose Pd for Ni queries; don't propose melting point for Tg queries — but DO propose canonical vocabulary on training-shot queries") might capture V7's reaction/multi lift while preserving V3's neutral score. The existing 30-probe bench and `compile_paw_expander_sweep.py` driver let this iterate cheaply.
2. **V10 candidate: V3 + V4's additional shots.** V4 added Mannich, Grubbs, CO2-to-ethylene-Cu, Tg as positive shots; combining those with V3's full 20-shot retention and 5 negative examples (30-shot total) tests whether more positive coverage on reaction/property families closes the residual gap on V3. Spec length will be ~5kB, still within the 8kB context budget.
3. (Carried over) Augment the static dictionary with the May-23 / May-27 gap entries.
4. (Carried over) Apply the same two-stage protocol to decompose / normalize / contradiction PAW programs.
