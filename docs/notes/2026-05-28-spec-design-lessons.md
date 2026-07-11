# Spec design lessons — May-28 expander sweep

*Distilled from the 9-variant × 2-compiler PAW expander sweep documented in
[../plans/2026-05-28-paw-expander-spec-iter.md](../plans/2026-05-28-paw-expander-spec-iter.md).
Treat these as defaults for future PAW programs, not absolutes — they came
from one task (chemistry query expansion) and may not generalise to e.g.
classification or relevance scoring.*

## Four robust patterns

### 1. Keep training shots intact; append corrections

Of the 9 expander variants, **V3 was the only one that didn't regress the
`neutral` family** (the 4 probes whose canonical vocabulary lives verbatim
in the V0 training shots: heavy metal adsorption, water splitting, graphene
oxide reduction, CRISPR). The reason is mechanical: V3 = V0's 20 positives
+ 5 negatives appended. The model still sees the canonical mapping for
`graphene oxide reduction -> GO, rGO, Hummers method, thermal reduction`
during finetuning, so it continues to emit it at inference.

Every other variant either dropped V0 shots (V4 went from 20 to 10), added
rules that nudged away from canonical vocabulary (V5, V7), or rewrote the
task prompt entirely (V8). All of them regressed the neutral family — V4
by -0.248, V7 by -0.179, V8 by -0.271.

**Rule:** when iterating on a working spec, prefer **extend** over **replace**.
Add new shots and counter-examples; don't remove existing ones unless
you have evidence they're causing the failure mode.

### 2. Few-shot scaffolding is essential

Zero-shot (V1, just the task description) and minimal-prompt (V6, one sentence)
collapsed even on the finetune compiler — V1 at +0.025 macro, V6 at -0.091.
Both have the highest degeneracy (0.41, 0.36 respectively); the model can't
infer the desired output shape without examples.

The std-mapper compiler also needs examples — V1_std produced "S-C coupling,
S-C bond, S-C linkage" inventions for the Suzuki sanity probe, V6_std looped
"coupling reaction, coupling, coupling reaction".

**Rule:** **always include at least 10-20 few-shot examples** unless the task
is trivial (1-class classification, etc.). The compile cost is identical;
the quality gap is large.

### 3. Rules without examples are useless; rules with examples are over-restrictive

V5 (V0 shots + a 3-rule block) scored +0.288 macro vs V0's +0.186 — modest
lift. V7 (V3's 25 shots + same rule block) scored +0.388, the highest of
all variants, **but failed the ship gate** because the rules nudged the
model away from canonical vocabulary on easy queries:

- `neu-go-reduction` ("graphene oxide reduction"): V7 dropped `rGO, Hummers
  method, thermal reduction` because the rule "don't introduce vocabulary
  from unrelated reaction classes" over-fired.
- `tech-eis` ("EIS"): V7 replaced `Warburg` (the canonical electrochemistry
  term) with `Colebrook plot` (fluid mechanics).

The rules saved Mannich and nickel-cross-coupling (the May-23 losers) but
broke things that worked.

**Rule:** prefer **negative examples** (a counter-shot that demonstrates the
correct behaviour) over **rules** (a prose constraint). Negatives are
mechanically the same shape as positives; rules are a different modality
the model interprets unevenly.

### 4. ft beats std uniformly, but rank ordering doesn't transfer cleanly

Apples-to-apples on the 12-probe quick suite:

| variant | std | ft | delta |
| --- | ---: | ---: | ---: |
| V7 | +0.160 | +0.554 | **+0.395** |
| V4 | -0.101 | +0.293 | +0.394 |
| V5 | -0.069 | +0.305 | +0.374 |
| V3 | +0.084 | +0.370 | +0.286 |
| V2 | -0.161 | +0.121 | +0.282 |
| V8 | -0.051 | +0.113 | +0.164 |
| V1 | -0.220 | -0.018 | +0.202 |
| V0 | -0.069 | +0.000 | +0.069 |
| V6 | -0.149 | -0.114 | +0.035 |

**ft beats std on 9/9** with a mean lift of +0.245 (median +0.282). But the
pairwise std → ft rank agreement was only 25/36 = 69%. V4 jumped from std
rank 6 to ft rank 2; V8 dropped from std rank 3 to ft rank 7.

**Rule:** use std as a **filter** (it cheaply rules out catastrophic specs
like V6 that even ft can't save), not a **ranker**. If compile budget allows
(~3 min × N specs), prefer compiling all candidates on ft and using std only
as a smoke check. The ~30-min wall for 9 ft compiles is cheap relative to the
cost of mis-promoting a Stage 1 winner.

## Anti-patterns observed

- **Rebalancing examples to match probe stratification (V4)** sounds principled
  but threw out the training shots that the easy queries needed. Rank 2 on ft
  but failed the ship gate by -0.248 on neutral.
- **Chain-of-thought wrappers (V8)** ranked 3rd on std but dropped to 7th on ft.
  The finetune compiler appears to prefer direct exemplars over meta-instructions.
- **Brief task descriptions (V6)** are the worst possible spec design — the
  model has no signal about output format or domain. V6 was the only ft variant
  with a negative macro score.

## Practical checklist for the next PAW spec

When authoring a new PAW spec (e.g. the reranker in Phase 3):

1. Start with **15-20 positive few-shot examples** spanning the input distribution.
2. Once you find failure modes, add **3-5 negative examples** (same input/output
   shape, demonstrating the desired correction).
3. **Don't add rule blocks** unless negatives fail to fix the failure mode after
   2-3 iterations.
4. **Don't drop existing shots** without evidence they're the cause.
5. **Compile on ft for the bench**; use std only as a cheap smoke check.
6. **Bench against a hand-curated probe set** with both `gold_expand` (or whatever
   the equivalent positive label is) and `gold_forbid` (negative-space terms the
   model must not emit). The May-27 plan describes the harness shape; reusable
   across PAW programs.

## What this doesn't tell you

- Whether unit-bench lifts translate to **end-to-end** retrieval quality (that's
  the Phase 1 question in [../plans/paw_search_system_roadmap_d7397da0.plan.md](../plans/)).
- Whether expansion is the **right** PAW use case at all (that's Phase 2).
- Whether the same patterns hold for non-expansion specs (relevance scoring,
  intent classification, etc.). Cross-check the spec for each new program type.
