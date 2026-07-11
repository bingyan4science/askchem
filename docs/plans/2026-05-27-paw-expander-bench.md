# May-27 PAW expander component benchmark

*Run date: 2026-05-27. Local Mac (CPU PAW via `GGML_NO_METAL=1`),
30 hand-curated probes at [data/eval/paw_expander_probes.json](../../data/eval/paw_expander_probes.json).*

## TL;DR

The May-23 A/B compared three configs on end-to-end nDCG@10 and found
flat quality with PAW on or off. That measurement was confounded by the
cross-encoder rerank (65 % of probes had identical top-10s in *any*
PAW state, so the expander's term list never reached the visible
window) and by the standard-compiler's degenerate looping (which
inflated latency 30× without changing recall).

This benchmark scores the expander **directly**, by gold-labelled
coverage / pollution / degeneracy on hand-curated chemistry queries.
It cleanly answers the two questions the end-to-end A/B could not:

1. **Is `paw_ft` (finetune `paw-ft-bs48-20260522`) better than `paw_std`
   (standard `paw-4b-qwen3-0.6b`) at producing chemistry expansion
   lists?** **Yes, substantially**: macro score +0.19 vs −0.04,
   coverage 1.7× higher, degeneracy 5.6× lower, latency 17× faster.
2. **Is either PAW better than the static dictionary?** **Yes, but only
   on the queries the dictionary doesn't already cover.** `paw_ft`
   beats `static` on technique / material / neutral families
   (training-shot-adjacent vocabulary) but **loses** to `static` on
   the May-23 failure cases (Mannich, Ni cross-coupling, Tg, Grubbs),
   because static does nothing on those queries and so emits zero
   forbidden terms.

| System | Coverage | Pollution | Degeneracy | Score | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| **paw_ft** (`fe558023bbc5acb6665b`) | **0.30** | 0.08 | **0.08** | **+0.19** | **1,071 ms** |
| paw_std (`d442088a6063deb9f42a`) | 0.18 | 0.00 | 0.45 | −0.04 | 18,031 ms |
| static (dictionary lookup) | 0.12 | 0.00 | 0.13 | +0.05 | 0 ms |

`paw_std`'s zero pollution is misleading: its output collapses to
`"Suzuki coupling catalysts"` repeated dozens of times for many queries
(`degeneracy = 0.45`), and a string that says nothing diverse cannot
pollute. The 18-second average latency is the cost of generating that
spam until the 2048-token context fills up.

## Per-family score

| family | n | paw_ft | paw_std | static |
| --- | -: | -: | -: | -: |
| homonym | 4 | **+0.012** | −0.211 | −0.012 |
| material | 5 | **+0.225** | −0.345 | +0.089 |
| multi | 3 | −0.068 | **+0.058** | −0.053 |
| neutral | 4 | **+0.640** | +0.534 | +0.273 |
| property | 4 | −0.009 | −0.065 | −0.028 |
| reaction | 6 | +0.041 | −0.196 | +0.099 |
| technique | 4 | **+0.461** | −0.088 | −0.010 |

`paw_ft` wins on `technique`, `material`, `neutral`, `homonym`; loses on
`multi` (Suzuki spam survives the finetune for Ni / CO2-Cu queries) and
ties `static` on `property` and `reaction`. The "multi" loss is the
clean expression of the Suzuki-prior failure mode the May-23 plan
documented.

## Biggest individual deltas

**`paw_ft` ≫ `paw_std`** (the finetune fixes degeneracy):

| Δscore | probe | query |
| ---: | --- | --- |
| +1.11 | tech-ftir | FTIR spectroscopy analysis |
| +1.10 | rxn-suzuki | Suzuki coupling |
| +0.88 | mat-perovskite | perovskite solar cell |
| +0.73 | mat-mof | MOF gas storage |
| +0.70 | tech-tem-np | transmission electron microscopy nanoparticle morphology |

These are all training-shot or training-shot-adjacent queries; the
finetune lifts them from "degenerate loop" to "clean comma-separated
list of in-domain vocabulary."

**`paw_std` ≫ `paw_ft`** (the cases where finetune leaks Suzuki vocab
that the standard compiler was too degenerate to emit):

| Δscore | probe | query |
| ---: | --- | --- |
| +0.49 | rxn-mannich | Mannich reaction enantioselective |
| +0.38 | multi-ni-cc-arcl | nickel-catalyzed cross-coupling aryl chloride |
| +0.30 | prop-tg | polymer glass transition temperature Tg |
| +0.22 | multi-co2-ethylene-cu | CO2 electroreduction to ethylene copper catalyst |

`paw_ft` actively produces wrong vocabulary on these queries; `paw_std`
was so degenerate (`Mannich, Mannich coupling, Mannich coupling
catalysts, Mannich coupling catalysts, …`) that it never got to the
Suzuki vocab and so escaped the pollution penalty. This is *not* a
recommendation to use `paw_std` — its degeneracy still loses overall.
It's a recommendation to spend a follow-up iteration on the expander
spec to add counter-examples for Mannich / Ni-cross-coupling / Tg.

**`static` ≫ `paw_ft`** (the May-23 losers confirmed):

| Δscore | probe | query |
| ---: | --- | --- |
| +0.68 | rxn-grubbs | olefin metathesis Grubbs catalyst |
| +0.60 | rxn-mannich | Mannich reaction enantioselective |
| +0.34 | multi-ni-cc-arcl | nickel-catalyzed cross-coupling aryl chloride |
| +0.33 | prop-tg | polymer glass transition temperature Tg |
| +0.30 | hom-spin-coupling | spin coupling magnetic material |

On these queries the dictionary contributes ≤1 added term and so emits
zero forbidden terms; `paw_ft` adds 8-12 terms of which several are
forbidden, dragging its score below static's near-zero baseline. The
gap is real expansion *harm*, not a measurement artefact.

## Cross-check vs the May-23 end-to-end nDCG@10

For the 21 probes that overlap between this bench's 30 and the May-23
80-probe set, the sign of the `paw_ft − static` unit-score delta agrees
with the sign of the May-23 `ft-C − ft-A` nDCG@10 delta on **13/21
queries (62 %)**. The disagreements are mostly probes where the unit
bench says `paw_ft` is good but end-to-end is flat or worse (e.g.
`tech-ssnmr-mof` agrees at +0.215 / +0.145, but `tech-tem-np` disagrees
at +0.333 / −0.065). That divergence is exactly what the rerank
ceiling predicts: even when `paw_ft` produces better terms, the
mxbai + MS-MARCO MiniLM rerank reshuffles the visible top-10 in ways
that don't preserve the recall improvement.

The unit metric is therefore an **earlier, less-confounded** signal of
expander quality than nDCG@10. It can tell us when the rate-limiter is
the rerank (`tech-tem-np`: bench up, end-to-end flat → fix the
reranker) vs when the rate-limiter is the expander (`rxn-mannich`:
bench down, end-to-end down → fix the spec).

## What this benchmark answers and what it doesn't

**Answers:**

- Is `paw_ft` better than `paw_std` at producing chemistry expansion
  lists? Yes, +0.23 macro score, with the gap driven by 5.6× lower
  degeneracy and 1.7× higher coverage.
- Is PAW better than the static dictionary? Net yes (+0.14 macro
  score) but the gap collapses or inverts on the queries the
  dictionary was already silent on — the failure mode is "PAW
  hallucinates Pd vocab when static would have done nothing."
- Where do the wins and losses live? Technique / material / neutral
  are PAW's territory; multi / specific-reaction / property-bleed are
  the static dictionary's territory.

**Doesn't answer:**

- Downstream nDCG@10 — that's the May-23 A/B
  ([docs/plans/2026-05-23-paw-ft-rewrites.md](2026-05-23-paw-ft-rewrites.md)).
- Decompose / normalize / contradiction expander quality — same
  benchmark shape applies; future work.

## Implications

A clear next-iteration plan:

1. **Re-spec the expander with counter-examples.** Add ~5 reaction-class
   negative examples to the prompt so the finetune learns:
   `Mannich → iminium / proline / organocatalysis (NOT Pd, SPhos)`,
   `Ni-catalyzed cross-coupling → Negishi / Kumada (NOT palladium)`,
   `Tg → DSC / DMA / amorphous (NOT melting point)`. Recompile and
   re-run this bench. Acceptance: paw_ft macro score ≥ +0.30 on the
   30-probe set without regressing the existing wins.
2. **Augment the static dictionary** with the high-confidence vocabulary
   the bench shows static missing. Specifically: add bigram entries for
   `transmetalation cross-coupling`, `nickel-catalyzed cross-coupling`,
   `polymer glass transition`, `olefin metathesis`. Deterministic, zero
   pollution by construction, near-zero latency. Re-run bench.
3. **Only after both of those** revisit the end-to-end A/B. The unit
   metric will tell us whether the rerank ceiling is still the
   rate-limiter; if so, latency and quality of the expander are
   irrelevant to nDCG@10 and we should park the work.

## Method

* Probe set: 30 hand-curated probes at
  [data/eval/paw_expander_probes.json](../../data/eval/paw_expander_probes.json)
  (6 reaction, 5 material, 4 technique, 4 property, 3 multi, 4 homonym,
  4 neutral). Authored anchored to the May-23 losers/winners.
* Harness: [scripts/bench_paw_expander.py](../../scripts/bench_paw_expander.py).
  Uniform `Callable[[str], list[str]]` adapter contract over all three
  systems; macro-averaged metrics so small families aren't drowned.
* Unit tests: [tests/test_bench_paw_expander.py](../../tests/test_bench_paw_expander.py)
  (21 tests) verify the matching rule (Pd matches `Pd-catalyzed` and
  `palladium (Pd)` but not `Padova`), the degeneracy primitive, the
  end-to-end scoring, and the static-adapter parity with
  `db.expand_query_variants`.
* Raw per-probe metrics:
  [data/eval/runs/paw_expander_bench.json](../../data/eval/runs/paw_expander_bench.json).

Reproducer:

```bash
PYTHONPATH=src .venv-benchmark/bin/python scripts/bench_paw_expander.py
```

Runtime: ~10 minutes wall (dominated by `paw_std`'s degenerate
generation; `paw_ft` is ~50 s, `static` is instant).
