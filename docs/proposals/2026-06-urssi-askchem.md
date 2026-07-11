# AskChem: Open, Agent-First Structured Knowledge Infrastructure to Accelerate Chemical Discovery

**URSSI Early-Career Fellowship — Summer 2026**
**Track 1: AI/ML & Scientific Software** (with software-sustainability deliverables)
**PI:** Bing Yan, PhD student, New York University (US-based)
**Working prototype:** [askchem.org](https://askchem.org)

> Markdown mirror of the submission LaTeX at
> [structure_the_universe_paper/urssi_proposal.tex](../../structure_the_universe_paper/urssi_proposal.tex)
> (which is the authoritative, PDF-generating version). Figures live in
> [structure_the_universe_paper/figures/](../../structure_the_universe_paper/figures/)
> (`urssi_fig1_flatvs`, `urssi_fig2_architecture`, `urssi_fig3_bench`).

---

## 1. Project Goals and Objectives

**Track.** Most closely aligned with **Track 1 (AI/ML & Scientific Software)**: AskChem uses LLMs to build, structure, and serve a scientific knowledge base, delivered as open research software. Sustainability and reuse (Track 2) are built into the deliverables.

**The problem.** Scientific knowledge is inherently *structured* — reactions, properties, mechanisms, conditions, and their relationships — yet we search it with tools built for flat text. Keyword and vector search both return a ranked list of whole documents; the researcher (or AI agent) must then open each PDF and reconstruct the structured answer by hand. As LLM agents increasingly plan experiments and search the literature, this flat interface is the bottleneck: agents cannot browse a hierarchy, cross-reference organizational dimensions, or get source-grounded claims they can verify.

**What we propose to build.** **AskChem** — an open, agent-first knowledge infrastructure that uses LLMs to "segment" the chemistry literature into atomic, source-grounded *claims*, organizes them into multiple simultaneously-navigable hierarchies (by reaction type, mechanism, application, technique, substance class, and more), and serves them to AI agents and human researchers via an MCP server, a REST API, and a Python SDK. The goal: turn the literature from "a pile of documents you search" into "a queryable map of what is known" that drops into the daily research loop and into agent workflows.

**Measurable goals (6 months).**
1. Release the **structuring engine** — LLM extraction of atomic claims + a quality-control validator + a taxonomy-driven multi-view organizer — as documented, tested, installable software.
2. Release the **agent-first serving layer** — an MCP server and the `askchem` SDK — so any agent or researcher can query structured chemistry knowledge.
3. Release the **ingestion/update pipeline** so the index can be kept current and others can stand up their own structured index.
4. Demonstrate, with evidence, that structured + source-grounded delivery improves answer verifiability and is useful in real workflows.

**Objectives serving the URSSI mission:** convert a working research prototype into sustainable, reusable open-source infrastructure, and establish "structured, source-grounded, agent-queryable" as a practical pattern for scientific knowledge software.

## 2. Preliminary Results (feasibility already demonstrated)

A working prototype is live at [askchem.org](https://askchem.org): **2.44M source-grounded claims from 146K chemistry papers**, organized into **7 simultaneous views**, with hybrid retrieval (full-text + taxonomy + vector) and live REST, MCP, and SDK access. The full pipeline — multi-source ingestion, Gemini-based extraction, claim validation, taxonomy classification, contextualization, incremental update — already runs end to end (Fig. 2).

Structuring the literature into source-grounded claims already pays off on reliability: on AskChem-Bench (30 cross-paper chemistry questions, GPT-5.5 reader), an unaugmented LLM fabricates ~12% of its cited DOIs, whereas the same reader grounded in AskChem reaches **100% existing DOIs** and the highest on-topic rate among five systems tested (Fig. 3) — preliminary evidence the structured, agent-first design produces verifiable, trustworthy answers.

**Figures:** Fig. 1 (flat list vs. structured multi-view), Fig. 2 (the pipeline as reusable software), Fig. 3 (AskChem-Bench verifiability evidence).

## 3. Expected Impact on the Scientific Software Community

- **Direct:** chemists and chemistry-aware agents gain a structured, source-grounded alternative to flat search — queryable by hierarchy, verifiable by DOI.
- **Indirect:** the released pipeline (ingestion, extraction, validation, multi-view structuring, hybrid retrieval, MCP/SDK serving) is a reusable blueprint other groups can adopt to turn their corpora into agent-queryable knowledge bases.
- **Improving software development practices:** package a working-but-monolithic prototype into documented, tested, versioned components with CI; contribute an MCP-native pattern for exposing structured scientific knowledge to agents; and make source-grounded verification a default in LLM-built scientific software.

## 4. Implementation Plan

**Methods/approach.** Claims are extracted with an LLM and admitted only through a validation gate (schema/type checks, optional RDKit SMILES validation); organized by classifying each paper into a canonical L1/L2/L3 taxonomy across multiple views and contextualized into standalone statements; retrieval fuses full-text, taxonomy, and vector signals. Each component is lifted out of the monorepo into its own repo with a hatchling build, type hints, pytest suites, GitHub Actions CI, and MkDocs docs; the MCP server and SDK are the agent-first interface; quality is tracked continuously with `askchem-bench`.

**Phased activities** follow the month-by-month timeline in §7. **Check-in points:** monthly GitHub progress reports, bi-weekly cohort meetings, July 25 progress evaluation (target: structuring engine + SDK/MCP released).

## 5. Community Engagement Strategy

- **Open releases:** PyPI + GitHub for each component (MIT) with docs, a quickstart notebook, and CI.
- **Agent ecosystem:** ship the MCP server + SDK so Cursor/Claude-class agents query structured chemistry knowledge out of the box, with a worked agent-integration example.
- **Adoption:** recruit 2–3 pilot users (chem-informatics groups, RSEs); document how others can index their own corpus.
- **URSSI integration:** a newsletter post and a candidate URSSI-school module on building and evaluating LLM-driven scientific software.

## 6. Evaluation Metrics

**Quantitative:** components released to PyPI with docs + CI and test coverage ≥70%; index scale/coverage (papers/claims/views) maintained and reported; answer verifiability via `askchem-bench` (existing-DOI rate, on-topic rate) held at or above the preliminary 100%/90% with retrieval; ≥1 external adopter or agent integration. **Qualitative:** adoption signals (issues/PRs/stars from outside users), researcher feedback on workflow integration, positive review of the teaching module by URSSI school organizers.

## 7. Timeline and Deliverables

- **Month 1:** structuring engine (extraction, validation, multi-view organizer). → package on TestPyPI.
- **Month 2:** serving layer (MCP server + `askchem` SDK). → PyPI release; July 25 checkpoint.
- **Month 3:** ingestion + incremental updater. → pipeline package + "stand up your own index" guide.
- **Month 4:** verifiability + agent-usefulness study. → evaluation report + released `askchem-bench`.
- **Month 5:** external-user pilot + agent-integration tutorial. → adopter case study + tutorial.
- **Month 6:** dissemination. → v1.0 releases + URSSI-school module + newsletter post.

## Reusable software released (answers the "libraries useful to a broader community" question)

- Ingestion + incremental updater (`src/update_index.py`, `scripts/harvest_new_papers.py`)
- Extraction: `gemini-batch` (`src/askchem/gemini_batch.py`) + `validation` (`src/askchem/validation.py`)
- Multi-view structuring: `src/classify_papers.py`, `src/askchem/taxonomy.py`, `scripts/contextualize_claims.py`
- Hybrid retrieval: `src/askchem/retrieval.py`, `src/askchem/embeddings_v2.py`
- Agent-first serving: MCP (`src/askchem/mcp_server.py`), REST (`src/askchem/server.py`), SDK (`sdk/`)
- Quality: `askchem-bench` (`scripts/benchmark_chemtree.py`)

_Budget submitted separately per the application instructions; references (Brown et al. 2020; Boiko et al. 2023) live in the LaTeX bibliography._
