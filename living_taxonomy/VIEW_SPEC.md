# Living Taxonomy — View Spec (Phase A)

Decides which of AskChem's current views become **living knowledge trees**
(internal nodes = principles / mechanisms / theories; leaves = specific
paper-grounded entities) versus which stay as flat **facets/overlays**.

A view earns "living tree" status only if (a) its internal nodes can be
genuine principles/mechanisms (not arbitrary buckets) and (b) it has a
coherent **leaf entity type** that a single paper instantiates concretely.

## Living trees (full treatment)

| view_id | internal nodes (principle/mechanism vocabulary) | leaf entity type |
|---|---|---|
| `by_reaction_type` | reaction principles & catalytic mechanisms (oxidative addition / transmetalation, radical/HAT, metathesis, polar acid-base, ...) | a **reaction** (named transformation: Suzuki coupling, RCM, ...) |
| `by_substance_class` | structural / bonding principles (metal-ligand coordination, conjugation, macromolecular chain, nanoscale/surface, framework porosity) | a **molecule / material** |
| `by_mechanism` | mechanistic theories (electron transfer, bond formation/breaking, excited-state, transport) | a **mechanistic observation** |

## Evaluate / reshape (living tree only if internal nodes are real principles)

| view_id | concern | leaf entity type (if kept) |
|---|---|---|
| `by_technique` | internal nodes risk being flat instrument buckets, not principles; keep only if organized by measurement principle | a **measurement / characterization result** |
| `by_application` | "application domain" is an organizing axis, not a principle; may be better as a facet | an **application / device** |

## Facets / overlays (NOT living trees)

| view_id | why | role |
|---|---|---|
| `by_claim_type` | deterministic epistemic buckets (reaction/property/method/...) | filter facet |
| `by_time_period` | chronological, already the time axis of every tree | temporal overlay (drives the "grow the tree" slider) |

## Pilot selection (Phase 0)

Pilot on **`by_reaction_type`** (leaf = reaction) first — cleanest leaf entity,
richest existing structured fields (reactants/products/conditions) — and
**`by_substance_class`** (leaf = molecule/material) second, since the molecular
structure leaf renders directly in the radial "tree of life" visualization.

Focused domain for the pilot: **catalytic cross-coupling & related catalysis**
(papers whose reaction claims sit under `coupling` / `catalysis`), so a small
hand-built seed tree can plausibly cover most leaves while leaving room for
genuine exceptions.
