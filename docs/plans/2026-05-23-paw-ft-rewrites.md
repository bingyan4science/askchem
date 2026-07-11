# May-23 PAW finetune for query rewrites

*Run date: 2026-05-23. Local Mac M-series CPU + MPS, 80-probe eval on
the v2 256-d production stack with cross-encoder rerank ON.*

## TL;DR

Three configs of `db.search_claims` evaluated against the standard
80-probe set (`data/eval/probes_v1.jsonl`, judged by
`data/eval/labels_v1.jsonl`):

| Run | nDCG@10 | nDCG@20 | MRR@20 | R@20 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ft-A-baseline** (PAW off, no rewrites — current prod) | **0.791** | 0.729 | 0.918 | 0.187 | **1.8 s** | 4.1 s |
| **ft-B-std-rewrites** (`paw-4b-qwen3-0.6b` + rewrites) | 0.789 | 0.718 | 0.911 | 0.182 | 31.5 s | 40.2 s |
| **ft-C-ft-rewrites** (`paw-ft-bs48-20260522` + rewrites) | 0.790 | 0.719 | 0.911 | 0.183 | 3.4 s | 8.1 s |

Δ vs baseline:

* `ft-B`: nDCG@10 = **−0.002**, MRR@20 = −0.006 (noise on quality,
  catastrophic on latency).
* `ft-C`: nDCG@10 = **−0.001**, MRR@20 = −0.006, p50 = **+1.6 s**.

Both PAW configs **fail the ship gate** defined in
[../../.cursor/plans/paw_finetune_for_askchem_9e82db28.plan.md](../../.cursor/plans/paw_finetune_for_askchem_9e82db28.plan.md)
(`Δ nDCG@10 ≥ +0.015` outside noise **and** `Δ p50 < +1.0 s`). The
finetuned compiler is clearly better than the standard compiler
(+0.017 on `homonym`, +0.011 on `material`, no degenerate-output
latency tail), but neither lifts overall quality above the
PAW-off baseline. **Recommendation: keep prod at
`CHEMTREE_DISABLE_PAW=1`**; the rewrite wiring and the ft-IDs override
both land behind kill-switches so the next compiler iteration can be
A/B'd with a single env-var flip.

## Per-family breakdown (nDCG@10)

| family | n | A baseline | B std PAW | C ft PAW | C − A | C − B |
| --- | -: | -: | -: | -: | -: | -: |
| homonym | 10 | 0.810 | 0.805 | **0.822** | +0.012 | **+0.017** |
| material | 10 | 0.730 | 0.723 | 0.734 | +0.004 | +0.011 |
| multi | 15 | 0.733 | 0.716 | 0.721 | −0.012 | +0.005 |
| property | 15 | 0.823 | 0.824 | 0.820 | −0.003 | −0.004 |
| reaction | 20 | 0.849 | 0.858 | 0.847 | −0.002 | −0.011 |
| technique | 10 | 0.754 | 0.757 | 0.759 | +0.005 | +0.002 |

The finetune helps where homonym disambiguation and material vocabulary
matter, but those gains are roughly cancelled by small `reaction` /
`property` regressions. The `multi` family — which the plan flagged as
the weak spot at 0.704 in the May-14 ablation — moved from 0.733 → 0.721
under `ft-C`, *worse* than baseline. The `decompose_query` rescue only
fires when the FTS cascade returns zero hits, which the baseline rarely
does on the v2-256 + mxbai stack; on this probe set there were essentially
no zero-hit cases for it to rescue.

## Decision matrix

| ablation | quality | latency | ship? |
| --- | :-: | :-: | :-: |
| Re-enable PAW with std compiler + rewrites | flat | **−30 s p50 regression** | **no** |
| Re-enable PAW with `paw-ft-bs48-20260522` + rewrites | flat | −1.6 s p50 regression | **no** |
| Keep PAW off, leave wiring + override behind kill-switches | (default) | (default) | **yes (no-op)** |

### Why the standard compiler explodes latency

`paw-4b-qwen3-0.6b` on the unmodified expander spec produces severely
degenerate output for many queries — e.g. `expand_query("Suzuki
coupling")` repeats `"Suzuki coupling catalysts, "` literally hundreds
of times until it hits the 2048-token context limit. The lazy-singleton
`paw_functions.expand_query` filter in
[src/chemtree/paw_functions.py](../../src/chemtree/paw_functions.py)
L260-271 already dedupes that, so the *retrieval* side stays sane, but
each call takes 5-30 s of token generation. The `paw-ft-bs48-20260522`
compile eliminates the looping entirely; that is the +28 s p50 gap
between configs B and C.

### Why neither config clears the +0.015 quality gate

The baseline is already 0.791 — close to the prod ceiling of 0.794 in
[docs/search-pipeline.md](../search-pipeline.md). Most 80-probe queries
are short and single-topic; the static dictionary in
`db.expand_query_variants` (`CHEMISTRY_SYNONYMS`, `CHEMISTRY_FORMULAS`,
`CHEMISTRY_BIGRAM_SYNONYMS`) already covers the high-volume vocabulary
gaps. The PAW expander adds noise on the long tail (homonyms /
materials) that approximately cancels small regressions elsewhere.

The May-14 ablation predicted this: dropping PAW was flat at nDCG@10
because the only PAW touch points reaching ranking were
`normalize_query`'s 0-hit fallback and `classify_intent`'s UI banner
(no ranking effect). Wiring `expand_query` into the FTS variants
finally gives PAW a path to influence ranking, but the static
dictionary was doing most of that work already.

## Method

Driver:
[`scripts/run_paw_ft_ab.sh`](../../scripts/run_paw_ft_ab.sh).
Compile driver: [`scripts/compile_paw_ft.py`](../../scripts/compile_paw_ft.py).

Wiring shipped in this PR (gated, default off):

* `db.expand_query_variants` — adds a PAW-expanded variant when
  `CHEMTREE_PAW_REWRITES=1` (no-op when the env var is unset, which is
  the prod default). Code: [src/chemtree/db.py](../../src/chemtree/db.py)
  search for `_paw_expand`.
* FTS recall block — fires `paw_functions.decompose_query` as a
  second-stage rescue if `normalize_query`'s rescue still returned zero
  FTS hits, again gated on `CHEMTREE_PAW_REWRITES=1`. Code:
  [src/chemtree/db.py](../../src/chemtree/db.py) search for
  `decompose_query`.
* Program-ID override — `CHEMTREE_PAW_FT_IDS=<json>` rewrites the
  per-function PAW program IDs in
  [src/chemtree/paw_functions.py](../../src/chemtree/paw_functions.py)
  from the JSON written by `scripts/compile_paw_ft.py`. Used in `ft-C`
  to flip the compiler without editing source.

Finetuned program IDs (`paw-ft-bs48-20260522`):

```
expand    fe558023bbc5acb6665b
decompose 4d83a4ee8681fb4c4620
normalize 1d5c371f410d870ef017
```

All three pass their per-function sanity probes (`heavy metal
adsorption` → `Pb, Cd, Cr, …`; `What electrocatalysts for CO2
reduction?` → 5 distinct sub-queries; `how does Suzuki coupling work` →
`Suzuki coupling`).

Reproducer:

```bash
# Compile (≈3 min/function on the PAW hosted compile API):
/Users/bingyan/miniconda3/envs/paw/bin/python scripts/compile_paw_ft.py

# A/B + score (≈45 min on local MPS):
bash scripts/run_paw_ft_ab.sh
```

Artefacts:

```
data/paw_ft_program_ids.json
data/eval/runs/ft-A-baseline.{rankings.jsonl,scored.json}
data/eval/runs/ft-B-std-rewrites.{rankings.jsonl,scored.json}
data/eval/runs/ft-C-ft-rewrites.{rankings.jsonl,scored.json}
```

## What we kept (kill-switched, default off)

| Change | Default | Knob | Rollback |
| --- | --- | --- | --- |
| PAW `expand_query` plumbed into `expand_query_variants` | off | `CHEMTREE_PAW_REWRITES=1` | unset env var |
| PAW `decompose_query` plumbed into 0-hit FTS rescue | off | `CHEMTREE_PAW_REWRITES=1` | unset env var |
| Per-function program-ID swap from JSON | off | `CHEMTREE_PAW_FT_IDS=<path>` | unset env var |
| Standard `CHEMTREE_DISABLE_PAW=1` continues to win | **on, prod** | `CHEMTREE_DISABLE_PAW` | (no change) |

Regression test coverage in
[tests/test_paw_rewrites.py](../../tests/test_paw_rewrites.py) verifies
that:

1. `CHEMTREE_DISABLE_PAW=1` short-circuits PAW even with
   `CHEMTREE_PAW_REWRITES=1` set (the documented composition).
2. `CHEMTREE_PAW_FT_IDS=<json>` swaps the program-ID constants on
   module import.
3. `CHEMTREE_PAW_REWRITES=0` (default) does not call into
   `paw_functions` from `db.expand_query_variants`.
4. `CHEMTREE_PAW_REWRITES=1` does call `expand_query` and appends a
   PAW variant.

## Follow-ups (not in this PR)

* **Don't re-run on this probe set without expanding the multi-family
  judgements.** The 80-probe set under-samples the multi-topic queries
  the rewrite path targets; the labelled pool has only 15 of them.
  Re-eval should wait until a multi-heavy probe set exists.
* **Try a domain-specific finetune spec.** The current expander spec
  was iterated against the standard compiler; the finetune may
  reward richer few-shot examples (more in-domain abbreviations, more
  multi-topic decomposition examples) than the standard compiler can
  use.
* **Re-eval the contradiction detector with `paw-ft-bs48-20260522`.**
  The plan flagged this as an independent track. The current v4c is
  75 % acc / 100 % precision on 20 pairs; the finetune is most likely
  to lift this since the contradiction task has high-ceiling
  classification structure that benefits from per-spec LoRA finetuning.
* **If we ever resize prod down again,** the disabled PAW path can be
  ripped out entirely per the May-14 follow-up list — the override
  scaffolding shipped here does not block that work.
