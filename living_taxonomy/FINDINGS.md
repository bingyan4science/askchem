# Phase 0 Pilot — Findings

Ran the placement loop on ~30 focused papers per view, read-only on
`chemtree.db`. The core idea works end-to-end: candidate leaves are placed
under principle/mechanism branches by embedding similarity, and genuinely
off-tree content is flagged as exceptions. Three concrete calibration
findings, each actionable for the next phase.

## 1. The placement loop is viable
- `by_reaction_type`: 400 reaction leaves placed across 7 seed branches; the
  branch distribution is sensible (C–C coupling 120, C–N amination 70,
  C–O/heteroatom 61, ...).
- Out-of-domain probe (enzymatic biochemistry, dye adsorption reactions)
  scores materially lower than in-domain (mean **0.568 vs 0.657**); the
  lowest scorers are exactly the conceptually off-tree items (proteolytic
  cleavage, kinase phosphorylation, dye adsorption) → correctly exceptions.
  At calibrated thresholds (attach ≥ 0.65 / exc ≤ 0.58), **75/120**
  out-of-domain leaves become exceptions while in-domain stays ~0.

## 2. Thresholds must be calibrated; absolute cosine sits high
- mxbai cosine for chemistry text is high and narrow: in-domain best scores
  span 0.58–0.77 (mean 0.657). A 0.55 attach threshold admits everything.
- Working range from the pilot: **attach ≥ 0.62–0.65, exception ≤ 0.55–0.58**,
  with a gray zone between for LLM adjudication. Bands overlap, so absolute
  thresholds alone are not enough (see #3).

## 3. Branches need exemplar-centroid representations, not prose
- Best-vs-runner-up margin is tiny (**mean 0.016, p50 0.012**): a one-line
  prose description makes every branch ~equidistant, so *which* branch a leaf
  attaches to is weakly determined.
- Fix for next phase: represent each branch by the **centroid of exemplar
  leaf embeddings** (a handful of known reactions/molecules per branch),
  reusing the existing 2.4M-claim corpus vectors. This should sharpen the
  margin and make branch assignment robust.

## 4. Leaf entity quality matters (substance tree)
- Reusing reaction *products* as substance leaves yields opaque compound
  codes ("3aa", "3ab", ...) → 62% gray zone, noisy exceptions.
- The substance tree needs real entities: `subject` / `subject_smiles` from
  property/structure claims (which also feed the molecular-structure leaf
  thumbnails in the visualization). Entity extraction is view-specific.

## Incremental LLM-grown tree (30 random reaction papers, seed=7)

`incremental_build.py` grew a tree paper-by-paper via Gemini (NYU): the tree
starts empty, paper 1 seeds principles/mechanisms, each later paper attaches
leaves or adds branches. Result: **23 principles, 49 mechanisms, 191 leaves**.
Reuse emerged (steps 10/17/20 added 0 new nodes — papers fit existing tree).

Three findings:
1. **Depth stayed uniform (all leaves at depth 3)** even though variable depth
   is allowed. The LLM defaulted to principle->mechanism->leaf everywhere and
   never nested deeper or attached a leaf directly under a principle. Variable
   depth must be *actively elicited* (prompt + allow leaf attach at any level),
   not merely permitted.
2. **Principle vocabulary drifts to methods/techniques.** Top level contained
   `Hydrothermal Synthesis`, `Sol-Gel Process` (techniques) as peers of
   `Heterogeneous Catalysis`, `Photocatalysis`. Needs steering toward
   fundamental governing principles + a consolidation pass.
3. **23 top-level principles is too flat** — diverse random papers each spawn a
   new principle. Needs a periodic **consolidation/refinement** step (merge
   near-duplicate principles, re-parent) — the "living" refinement loop.
4. LLM `is_exception` flag is noisy (marks all leaves as exceptions whenever any
   node is created); use node-reuse as the reliable "fit existing tree" signal.

## Prompt lab — generating principles/mechanisms/theories correctly

`prompt_lab.py` A/B-tested 4 prompts x 5 diverse papers (incl. the cases that
mislabeled techniques as principles). All allow MULTIPLE per paper.

- **V1 paper-only, free**: vague; leaks non-fundamentals ("Green Chemistry" as
  a principle).
- **V2 claims-grounded, free**: most faithful to the paper's actual chemistry,
  but without definitions still regresses to techniques (Sol-Gel, Hydrothermal)
  and mistypes methods as principles.
- **V3 claims + definitions + anti-examples**: cleanest typing, zero technique
  leakage, but LOW RECALL (electrosynthesis paper -> only 2 items).
- **V4 claims + definitions + bottom-up two-level (mechanism -> abstract to
  principle)**: BEST. Grounded mechanisms PLUS the fundamental principles/
  theories they instantiate; 6-9 items/paper; correct typing; 0 technique
  flags. Its two-level output is exactly the principle->mechanism nesting a
  variable-depth tree needs.

Decision: adopt **V4** (claims-grounded + explicit principle/mechanism/theory
definitions + anti-examples + bottom-up two-level reasoning) for tree growth.
Next refinement: have V4 emit the explicit parent link (which principle each
mechanism sits under) so growth places mechanism-under-principle directly,
giving real variable depth. The defs+anti-examples also fix the technique-as-
principle failure from the 30-paper run.

## First-principles scaffold (the curated upper trunk)

`seed_scaffold.py` = small hand-authored trunk (physics -> QM -> bonding ->
mechanism hosts). `scaffold_builder.py` = comprehensive version compiled by
enumerating 6 subfields (general/physical/organic/inorganic/analytical/
biochemistry) via Gemini, then deterministically merged.

Result: **192 internal nodes** — 26 laws, 3 frameworks, 43 theories, 73 models,
47 mechanisms — under the 5-anchor physics trunk, **variable depth up to 6**.
Raw enumerations cached in `scaffold_raw.json` so the merge can be re-run with
`--cache` (no new LLM calls). Improved merge (keyword anchor resolution + fuzzy
parent matching) cut orphans 18 -> 0; formerly-orphaned items now re-parent
correctly (Schrodinger/DFT -> QM; thermo laws -> Thermodynamics; Marcus/
Catalytic Cycle -> Kinetics).

Caveats: it is a primary-parent TREE of a concept that is really a DAG (a
mechanism can derive from several principles); some near-duplicate concepts
across subfields may survive exact-name dedupe and want a light consolidation
pass. Growth policy (decided): papers may ADD theory/mechanism nodes and start a
branch on exception/contradiction, but be cautious challenging existing theories.

## Multi-view scaffold + vertical tree (implemented)

Viz rewritten ([build_viz.py](build_viz.py)) from radial to a **vertical,
collapsible, label-wrapped** d3 tree with zoom/pan, kind-graded colors, and a
**view selector** (keeps the shared trunk, swaps each view's host layer; shared
trunk nodes get a gold dashed ring).

Multi-view scaffold ([view_layers.py](view_layers.py)) built on the principle
**accuracy over sharing**: the shared explanatory trunk (laws/frameworks/
theories/models) is reused by all views; each view's host layer attaches under
the trunk node that most accurately governs it, pruned to the accurate
sub-trunk. It is a DAG — hosts carry multiple parents (cross-links recorded).

Result (`output/scaffold_multiview.html`, `view_layers/manifest.json`):
- by_reaction_type / by_mechanism: 47 mechanism hosts (from scaffold), depth 6.
- by_substance_class: 21 hosts, 26 cross-links, depth 6 (e.g. transition-metal
  complexes under both Ligand Field Theory and complexation-equilibrium; main-
  group molecules under VSEPR + hybridization).
- by_technique: 20 hosts, 16 cross-links, depth 7.
- 0 unattached (every host resolved to an accurate trunk parent).
Accuracy safeguards: parent-resolution validation (exact/fuzzy/keyword; unmatched
-> unattached, not force-attached), `scaffold_audit.json` (8 near-duplicate pairs
+ LLM-flagged missing concepts: gas laws, Heisenberg, Bohr, ...), and recorded
cross-links + per-node `views` membership in the manifest.

## Leaves inserted onto the scaffold (test populate)

`grow_onto_scaffold.py` attaches real paper leaves under the accurate hosts.
15 test papers -> 101 reaction leaves + 157 substance leaves, depth 7, content
now searchable. Placement = embedding nearest-host (LLM-free). Accuracy is
MIXED (e.g. "[3+3] annulation" mis-placed under SN2): the low-margin embedding
issue. LLM placement (V4 prompt, pick host by name) is the accurate upgrade and
the recommended next step before relying on placements. Viz label bug fixed
(SVG text + wrap + halo, replacing foreignObject).

## LLM placement on 30 papers (accuracy upgrade)

`grow_onto_scaffold.py --use-llm` places each leaf under a host BY NAME via
Gemini (one call/paper/view), refusing to force-fit.
- by_reaction_type: 28 placed, **102 exceptions** — the 30 random papers are
  materials/synthesis-heavy; their steps (COF condensation, CO2 hydrogenation,
  hydrothermal/sol-gel) genuinely don't map to the organic reaction-mechanism
  hosts -> correctly flagged as exceptions (missing-branch signal), where
  embeddings force-fit all 101 inaccurately.
- by_substance_class: 149 placed, 28 exceptions (substance classes cover
  materials well).
- Placements that DO land are accurate (SN2 -> substitution/alkylation; acyl
  substitution -> Zemplen deacetylation; aldol -> nucleophilic addition; ligand
  exchange -> ligand substitution) vs embedding's wrong picks (aldol under E1).
- A few over-strict misses (diene cross-metathesis -> exception though a
  metathesis host exists) suggest passing richer host context or a 2nd-chance
  nearest-host check.

Implication: LLM placement >> embeddings for accuracy; the large reaction
exception pile is real and is exactly the input to the branch-proposal
(refinement) loop.

## Exception fix (recall + coverage + proposal placement)

Diagnosed the 102 reaction "exceptions" as 3 causes (coverage gaps, over-strict
placement, flat root dump) and fixed all three:
- Recall: relaxed prompt (best-fit; propose only if family absent), full host
  defs, all-hosts (no shrinking shortlist), dedupe proposed-name vs existing host.
- Coverage: `REACTION_HOST_SUPPLEMENT` (heterogeneous catalysis/hydrogenation,
  electrocatalytic redox & electrodeposition, olefin metathesis, polymerization/
  polycondensation, nucleation/crystal-growth/solid-state) under accurate parents.
- Proposal placement: genuine misses become `proposed:true` branches under the
  most-related parent (logged to `view_layers/proposed_<view>.json`), styled
  dashed-amber in the viz; no more flat root Exceptions pile.

Result on 30 papers (seed 11): reaction 28->**103 placed**, exceptions 102->27
in just **10 proposed branches**; substance 144->133 placed, 6 proposed branches.
Verified: nitrite reduction -> Electrocatalytic redox; CO2 hydrogenation ->
Heterogeneous catalysis; diene cross-metathesis -> Olefin metathesis;
glycosylation -> SN1/SN2. Proposed branches are genuine new families (COFs,
C-H activation, electrochemical intercalation, thermal dehydrogenation), not
duplicates. (One substance paper hit a transient JSON parse error -> nearest-host
fallback.)

## Recommended next steps
1. Switch branch representation to exemplar-centroid embeddings (#3).
2. Add view-specific entity extraction; for substances use `subject_smiles`
   and named compounds, not reaction-product codes (#4).
3. Lock pilot thresholds (attach 0.65 / exc 0.58) and wire the optional
   Gemini gray-zone adjudication (`--use-llm`) to calibrate the gray band.
4. Then proceed to Phase 1 (versioned DB-backed taxonomy_nodes) seeded from
   these validated seed trees.
