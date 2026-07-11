# Ingestion postmortem — 2026-05-20

First multi-source ingestion run since 2026-04-19. **End state: 142,534 sources / 2,386,325 claims on prod, +1,621 papers + 48,922 claims this run.** All from arXiv.

This doc captures the things that went sideways so the next operator does not waste a day rediscovering them.

## What we wanted vs what we got

Plan was: full multi-source ingestion (arXiv + ChemRxiv + journal RSS + Semantic Scholar bulk) for the 31-day delta since 2026-04-19.

What actually contributed:

| Source | Plan | Reality |
|---|---|---|
| arXiv OAI-PMH | 1-2k papers | **1,763** (all of the new sources we got) |
| ChemRxiv API | ~500 papers | **0** (Cloudflare 403 from NYU IPs) |
| Journal RSS (8 feeds) | ~200 papers | **30** (Angew Chem only; all already in DB; other feeds returned 0) |
| Semantic Scholar bulk | thousands | **0** (transient 403 + retry policy exhausted) |

So the run was effectively **arXiv-only**.

## Failures, root causes, fixes

### 1. ChemRxiv: Cloudflare bot wall, not a license issue

The existing [`src/update_index.py`](../../src/update_index.py) `discover_chemrxiv` hit `chemrxiv.org/engage/chemrxiv/public-api/v1/items` and got HTTP 403 (`Just a moment...` HTML — Cloudflare bot challenge). The newer endpoint `chemrxiv.org/engage/api-gateway/chemrxiv/public/v1/items` returns the **same** Cloudflare challenge — there is no API path past the bot wall from NYU egress IPs.

**Fix shipped**: rewrite `discover_chemrxiv` to use CrossRef's prefix endpoint `/prefixes/10.26434/works?filter=from-pub-date:<date>` instead. ChemRxiv DOIs are all in CrossRef (member 316, ACS-operated). 1,184 preprints since 2026-04-19, ~95% with abstracts. No publisher-site contact, fully NYU-library-policy compliant.

### 2. Semantic Scholar: NOT actually blocked

The harvest log showed 24 consecutive HTTP 403s on `https://api.semanticscholar.org/graph/v1/paper/search/bulk` and the loop bailed at 0 papers for every field of study. The natural read was "S2 key expired" or "NYU IP blocked".

**A fresh live probe shows S2 works fine** — 200 OK with 180,121 chemistry papers for `year=2026-`, with and without the key. So the run's 403 was either rate-limit transient or a backend hiccup. The old retry policy in `discover_s2` had two design bugs:

- Flat 10 s sleep on any exception → 5 attempts × 10 s = 50 s before giving up. Far less than S2's typical recovery window.
- 403 was not in the explicit retry list (only 429 was). So 403 hit `resp.raise_for_status()` → exception → 10 s wait → another 403 → … and we burned through the budget in under a minute.

**Fix shipped**: rewrote retry policy in `discover_s2` and `enrich_via_s2`. New behaviour treats 403/429/502/503 as transient with exponential backoff (15 / 30 / 60 / 120 / 240 s), honours `Retry-After`, and waits long enough to clear the typical rate-limit window. Factored out into `_s2_get_with_backoff` so both functions share the policy.

### 3. Journal RSS: structurally wrong tool

Even with all 8 feeds working, RSS only exposes the **most recent 20-50 items per journal**. For a 31-day catch-up window that's a tiny fraction of new chemistry papers. Plus: ACS feeds (JACS, ACS Catal, Chem Rev, ACS Nano) returned 0 items consistently, RSC's Chem Sci feed has an XML namespace mismatch that breaks lxml, only Wiley's Angew Chem returned items — and all 30 were already in the DB.

**Fix shipped**: replace `discover_rss` in the unified discovery flow with `discover_crossref` (per-publisher metadata via CrossRef's `member:<id>,from-pub-date:<date>` filter). Covers ACS (316), Wiley (311), Elsevier (78), RSC (81), Springer Nature (297), Taylor & Francis (301), Cell Press (320), Nature Research (1968), plus ChemRxiv via DOI prefix. No publisher-site scraping, fully metadata-only.

RSS is kept as an opt-in source (`--source rss`) for explicit ad-hoc runs but no longer fires from `--source all`.

### 4. NYU library policy: bulk PDF downloads off-limits

NYU library sent a warning on 2026-04-18 about automated downloads from Science.org under our NetID. Their policy explicitly forbids downloading entire journals or large amounts from one publisher in a 24-h window — that would get the institution blocked from the publisher.

This rules out **full-text** ingestion for closed-access journals. Our discovery and extraction must be **metadata-only** for everything except OA preprints.

**Fix shipped**: the CrossRef + S2 path is metadata-only (DOI + title + abstract + venue) — no publisher-site contact. The new `src/batch_extract_abstracts.py` (Stage 4b) is the Gemini Batch extractor that takes only the title + abstract and produces the same JSON output schema as the full-PDF extractor, tagged with `extraction_version='deep_v1_abstract'`. So closed-access papers now have a viable, compliant ingestion path.

### 5. Vertex Batch output download was the real bottleneck

Of 134 submitted Gemini Batch chunks, 90 (67%) returned an HTML 502 Bad Gateway from `/batches/{id}/output` even though the batches themselves had completed cleanly. The original `cmd_collect` in `batch_extract_arxiv.py` had a 120 s timeout and no HTML-response detection, so it wrote the 107-byte error pages into `outputs/` as if they were JSONL and then `_parse_one_output` got nothing from them.

**Fix shipped**: [`scripts/batch_collect_files.py`](../../scripts/batch_collect_files.py) — a parallel collector that uses the `/files/{output_file_id}/content` endpoint (more reliable, 5 min timeout, retries on HTML 502) and writes per-paper JSONs from the Vertex prediction format. After the switch we recovered all 1,508 papers cleanly.

The existing `batch_extract_arxiv.py cmd_collect` should be kept for back-compat but the parallel-files collector is now the default for production runs.

### 6. MPS embedding was thermally throttled; H200 cluster fixed it

The local MPS embedding run started at ~2 s/batch and degraded to ~15 s/batch over ~20 min — classic thermal throttling on a 16 GB M-series Mac. At that rate the 48,922 new claims would have taken ~6 h.

**Fix shipped**: SSH ControlMaster to NYU's `torch` cluster, `sbatch` job on `h200_cds` partition (0 pending queue depth at the time), encode in **0.7 min at 1,145 c/s**, scp results back. Code: [`scripts/embed_incremental_2026_05.py`](../../scripts/embed_incremental_2026_05.py) + [`scripts/dump_new_claims_for_encoding.py`](../../scripts/dump_new_claims_for_encoding.py) + the SLURM `encode_mxbai_*.slurm` files already on cluster from earlier runs.

### 7. The per-row `DELETE FROM claims_fts` in `upsert_claims_batch` was unworkable

[`src/chemtree/db.py`](../../src/chemtree/db.py) `upsert_claims_batch` deletes-then-inserts each row to keep the FTS5 virtual table consistent across upserts. Against a 2.3M-row FTS5 index, each per-claim DELETE took >0.5 s. With 49k new claims, the math worked out to **~50 hours** to flush — and the first apply attempt was killed after 1 hour with only 1 batch (1000 claims) committed.

**Fix shipped**: [`scripts/apply_incremental_2026_05.py`](../../scripts/apply_incremental_2026_05.py) bypasses the per-row DELETE for confirmed-new claims (already filtered against `existing_claim_ids`) and uses raw `INSERT INTO claims_fts(...)` without the DELETE round-trip. Same 49k claims in **~1 minute**.

This pattern should generalise: when `upsert_claims_batch` is called with claims that are known to be new, we should bypass the FTS delete-then-insert. Worth a follow-up in `db.py`.

## What still needs doing

- **Run the now-fixed multi-source harvest** for 2026-04-19 → today. Expected new content from CrossRef (closed-access journals) + ChemRxiv: 5-15k papers. Cost: $5-15 in Gemini Batch.
- **Stage 4b on the 135 dropped arXiv papers** from this run (49 too-big + 2 PDF download fail + ~84 lost in Vertex output). Submitted as job `recover_arxiv_dropped` at the end of this fix-up session.
- **Re-deploy** with the updated `chemtree.db` after the 4b ingestion lands.

## Reference

- Plan: [`/Users/bingyan/.cursor/plans/fix_multi-source_coverage_and_add_update_history_9e681273.plan.md`](../../.cursor/plans/fix_multi-source_coverage_and_add_update_history_9e681273.plan.md)
- Original 2026-05-20 ingestion logs: [`logs/`](../../logs/) (`harvest_2026_05.log`, `extract_*_2026_05.log`, `classify_*_2026_05.log`, `apply_2026_05*.log`, `embed_2026_05.log`, `upload_hf_2026_05.log`, `deploy_2026_05.log`)
- NYU library policy email: archived in operator inbox, dated 2026-04-18, ref Science.org NetID `by2192`.
