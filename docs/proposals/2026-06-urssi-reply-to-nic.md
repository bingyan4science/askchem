# Reply to Nic Weber (URSSI Early-Career Fellowship)

Subject: Re: AskChem — URSSI Early-Career Fellowship

Hi Nic,

Thank you — that's exactly the steer I needed, and yes, the answer is a clear yes: AskChem is backed by a full, modular software pipeline that we've built, several pieces of which are designed to be useful well beyond our own index. The proposal I'm putting together centers on building AskChem as open, agent-first research software — turning the chemistry literature from flat document search into a structured, multi-view, source-grounded index that agents and researchers can query — with the current live system as preliminary evidence that it works. The reusable software stack is:

- **Extraction + structuring** — **gemini-batch** (a domain-agnostic, restart-survivable wrapper over the Vertex AI batch API for large-scale LLM extraction), **askchem.validation** (a claim quality-control gate: schema/type checks + RDKit SMILES validation), and a taxonomy-driven multi-view organizer that turns papers into atomic claims arranged across multiple navigable hierarchies. Reusable by any group doing LLM-based extraction of structured scientific data.

- **Agent-first serving** — **askchem**, our Python SDK (already MIT-licensed and pip-packaged) plus an MCP server that exposes the index as native tools for AI agents (Cursor/Claude). This is the piece I think makes it broadly useful: any agent can query structured, DOI-verified chemistry knowledge out of the box.

- **Ingestion/update pipeline** — multi-source harvest + incremental updater that keep the index current and let another group stand up their own structured index from their corpus.

- **askchem-bench** — an evaluation harness that scores any system (not just ours) on source-grounding reliability (DOI existence, citation density, grounded specificity, relevance, on-topic). We've already run it against NotebookLM, Edison Scientific, and the GXL Paperclip retriever; it shows AskChem-grounded answers reach 100% existing DOIs vs. ~12% hallucinated for an unaugmented LLM.

The fellowship work is the sustainability/hardening step: extract these from our monorepo into separately documented, tested, versioned, installable libraries (with the MCP server + SDK as the agent interface), so the broader community can both use AskChem and build their own structured indices. The deliverable is reusable open-source software plus evidence — not features for a hosted product.

One logistical question: I saw the June 1 deadline and want to make sure I'm not too late given your note. Is there any flexibility, or a preferred way for me to submit given the timing?

Thanks again for the encouragement.

Best,
Bing Yan
NYU
