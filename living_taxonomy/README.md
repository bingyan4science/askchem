# Living Taxonomy — Phase 0 Pilot

Isolated pilot for re-structuring AskChem into living knowledge trees
(internal nodes = principles/mechanisms; leaves = specific entities). It
validates the **placement** idea on a small focused paper set **before**
touching the production index.

Safety: reads `chemtree.db` strictly read-only (`immutable=1`). Writes
nothing to the DB; all output goes to `living_taxonomy/output/`.

## View the living tree locally

```bash
source ~/.bashrc                 # needed for PORTKEY_API_KEY (Advisor uses Gemini)
python3 living_taxonomy/serve_ltree.py    # lightweight server, http://127.0.0.1:8126
# open http://127.0.0.1:8126  ->  "Living Tree" tab
```

`serve_ltree.py` is a minimal server (no FAISS/FTS warmup) for the Living Tree
tab: skeleton outline, papers-as-leaves, claims panel, node-name search, and the
**Advisor** (`/api/ltree/{view}/node/{node}/advise?doi=`) which asks grounded
positioning questions about a paper using its branch siblings. The Advisor needs
`PORTKEY_API_KEY` in the environment (Gemini via NYU), so start with `source ~/.bashrc`.

## Run

Use the system Python (it has `faiss` / `sentence-transformers`; the conda
`python` does not):

```bash
python3 living_taxonomy/run_pilot.py --view by_reaction_type --papers 30
python3 living_taxonomy/run_pilot.py --view by_substance_class --papers 30
```

Optional Gemini (NYU gateway) adjudication of gray-zone placements:

```bash
export PORTKEY_API_KEY=...   # NYU AI gateway key
python3 living_taxonomy/run_pilot.py --view by_reaction_type --use-llm
```

## Scale pipeline (300 -> 3k -> full-PDF corpus)

Batch placement + cleanup at scale. Placement uses the Gemini **batch** API
(sharded); cleanup is a fixed chain. Run order:

```bash
# 1. PLACEMENT (Gemini batch, sharded, all 4 views)
python3 living_taxonomy/batch_place.py prepare --papers 40000 --per-paper 40
python3 living_taxonomy/batch_place.py submit          # -> 1 job per shard
python3 living_taxonomy/batch_place.py poll            # until N/N complete
python3 living_taxonomy/batch_place.py collect         # -> grown_views.json + leaves.jsonl

# 2. CLEANUP CHAIN (order matters)
python3 living_taxonomy/consolidate.py                 # merge/promote proposed; folds tiny (<3-member) branches w/o LLM
python3 living_taxonomy/audit_nodes.py                 # concrete node statements + renames
python3 living_taxonomy/enrich_nodes.py                # short_label + LaTeX equation
python3 living_taxonomy/audit_positioning.py fix       # roles + name-dedup + inversion/gap repair
python3 living_taxonomy/semantic_dedup.py              # merge same-concept paraphrases/acronyms (cross-parent)
python3 living_taxonomy/combine_nodes.py               # per-parent: merge special-cases + group related siblings under family nodes
python3 living_taxonomy/enrich_gaps.py                 # create missing governing theories, re-home danglers

# 3. PERSIST + SERVE
python3 living_taxonomy/apply_to_db.py --version vN    # bulk load into chemtree.db
python3 living_taxonomy/build_node_index.py            # node vectors for semantic search (node_index.npz)
```

Notes: placement re-embeds leaves via mxbai unless `corpus_embed` finds aligned
`data/claim_embeddings.v2.*` artifacts (matrix rows must equal the id sidecar).
Cleanup is `O(internal nodes)`; `semantic_dedup` and `concept_registry` guard
against duplicate explosion. Paper intelligence is served on-demand (or
`precompute_analysis.py run --seeds-only`).

## What it does

1. `pilot_data.py` — samples ~30 catalytic-coupling papers and their
   reaction claims as candidate leaves (read-only).
2. `seed_trees.py` — hand-built seed trees (principle -> mechanism nodes)
   for `by_reaction_type` and `by_substance_class`.
3. `placement.py` — embeds leaves + branches with `mxbai-embed-large-v1`
   (same encoder as prod), decides `attach_leaf` / `gray_zone` / `exception`
   by cosine threshold; optional Gemini adjudication for the gray zone.
4. `run_pilot.py` — orchestrates, prints a summary, writes
   `output/placements_<view>.json`.

## Files

- `VIEW_SPEC.md` — Phase A: which views become living trees + leaf types.
- `seed_trees.py`, `pilot_data.py`, `placement.py`, `run_pilot.py`.
