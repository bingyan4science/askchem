"""
Top-Down Canonical Chemistry Knowledge Map.

Instead of only building bottom-up from papers, this generates the "skeleton"
of what chemistry knowledge SHOULD contain — the reactions, concepts, and
phenomena that every chemist knows — then uses it to drive targeted paper
collection to ensure comprehensive coverage.

Three stages:
1. Generate canonical knowledge map using GPT-5.4
2. For each canonical entry, search Semantic Scholar for landmark papers
3. Identify gaps: canonical knowledge not yet covered by our index
"""

import json
import os
import time
import requests
from pathlib import Path
from datetime import datetime
from openai import OpenAI

client = OpenAI()

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments" / "007_topdown_canonical"
DATA_DIR = Path(__file__).parent.parent / "data"

S2_BULK_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,externalIds,isOpenAccess,openAccessPdf"

CANONICAL_MAP_PROMPTS = [
    {
        "branch": "organic_chemistry",
        "prompt": """You are a chemistry professor listing the essential knowledge in ORGANIC CHEMISTRY that any PhD chemist should know. List 40-60 specific named reactions, reaction classes, and concepts.

For each entry provide: name, subcategory, importance ("fundamental"/"major"/"notable"), search_query (for Semantic Scholar), expected_landmark_paper.

Return JSON: {{"canonical_entries": [{{"name": "Suzuki-Miyaura Coupling", "category": "organic_chemistry", "subcategory": "cross_coupling_reactions", "importance": "fundamental", "search_query": "Suzuki coupling palladium cross-coupling", "expected_landmark_paper": "Miyaura and Suzuki, 1979"}}, ...]}}

Include: named reactions (Grignard, Wittig, Diels-Alder, aldol, Heck, Suzuki, Sonogashira, Buchwald-Hartwig, olefin metathesis, click chemistry, etc.), reaction classes (SN1, SN2, E1, E2, electrophilic aromatic substitution, radical reactions, pericyclic, photochemical), functional group transformations, protecting group strategies, retrosynthetic analysis, asymmetric synthesis, C-H activation, photoredox catalysis. Be exhaustive.""",
    },
    {
        "branch": "inorganic_and_materials",
        "prompt": """You are a chemistry professor listing the essential knowledge in INORGANIC CHEMISTRY and MATERIALS CHEMISTRY. List 40-60 entries.

For each: name, category (use "inorganic_chemistry" or "materials_chemistry"), subcategory, importance, search_query, expected_landmark_paper.

Return JSON: {{"canonical_entries": [...]}}

Include: coordination chemistry (crystal field theory, ligand field, spectrochemical series), organometallics (18-electron rule, oxidative addition, reductive elimination), solid-state (band theory, crystal structures, defects), bioinorganic (hemoglobin, metalloenzymes), MOFs, zeolites, nanomaterials, quantum dots, perovskites, polymers (ATRP, RAFT, ROMP), conducting polymers, ceramics, semiconductors, supramolecular chemistry.""",
    },
    {
        "branch": "physical_and_analytical",
        "prompt": """You are a chemistry professor listing the essential knowledge in PHYSICAL CHEMISTRY and ANALYTICAL CHEMISTRY. List 40-60 entries.

For each: name, category (use "physical_chemistry" or "analytical_chemistry"), subcategory, importance, search_query, expected_landmark_paper.

Return JSON: {{"canonical_entries": [...]}}

Include: thermodynamics (laws, Gibbs free energy, phase diagrams, Clausius-Clapeyron), kinetics (Arrhenius, transition state theory, Michaelis-Menten, Marcus theory), quantum chemistry (Schrodinger equation, MO theory, DFT, Born-Oppenheimer), spectroscopy (NMR, IR, UV-Vis, Raman, mass spec, XPS, XRD), electrochemistry (Nernst equation, Butler-Volmer, cyclic voltammetry), chromatography (HPLC, GC, SEC), microscopy (SEM, TEM, AFM, STM), surface analysis (BET, contact angle).""",
    },
    {
        "branch": "biochem_catalysis_computational",
        "prompt": """You are a chemistry professor listing essential knowledge in BIOCHEMISTRY, CATALYSIS, and COMPUTATIONAL CHEMISTRY. List 40-60 entries.

For each: name, category (use "biochemistry", "catalysis", or "computational_chemistry"), subcategory, importance, search_query, expected_landmark_paper.

Return JSON: {{"canonical_entries": [...]}}

Include: BIOCHEMISTRY — enzyme classes, metabolic pathways (glycolysis, Krebs, oxidative phosphorylation), DNA/RNA chemistry, protein folding, lipid bilayers, CRISPR, PCR. CATALYSIS — homogeneous (Wilkinson, Grubbs, Jacobsen), heterogeneous (Haber-Bosch, Fischer-Tropsch, Ziegler-Natta, automotive catalytic converter), photocatalysis (TiO2, visible light), electrocatalysis (HER, OER, CO2 reduction), biocatalysis, asymmetric catalysis. COMPUTATIONAL — DFT, ab initio, molecular dynamics, Monte Carlo, force fields, machine learning potentials, QSAR.""",
    },
    {
        "branch": "applied_and_interdisciplinary",
        "prompt": """You are a chemistry professor listing essential knowledge in APPLIED and INTERDISCIPLINARY chemistry. List 30-50 entries.

For each: name, category (use "electrochemistry", "environmental_chemistry", "medicinal_chemistry", "industrial_chemistry", or "emerging_fields"), subcategory, importance, search_query, expected_landmark_paper.

Return JSON: {{"canonical_entries": [...]}}

Include: ELECTROCHEMISTRY — Li-ion batteries, solid-state batteries, fuel cells, supercapacitors, corrosion, electrodeposition. ENVIRONMENTAL — green chemistry 12 principles, CO2 capture, water treatment, biodegradable polymers. MEDICINAL — ADMET, SAR, drug classes, lead optimization, antibody-drug conjugates. INDUSTRIAL — petroleum refining, ammonia synthesis, polymerization processes. EMERGING — single-atom catalysis, covalent organic frameworks, perovskite solar cells, DNA nanotechnology, artificial photosynthesis, flow chemistry, mechanochemistry.""",
    },
]

def generate_canonical_map():
    """Use GPT-5.4 to generate the canonical chemistry knowledge map in chunks."""
    all_entries = []
    total_tokens = 0

    for i, chunk in enumerate(CANONICAL_MAP_PROMPTS):
        branch = chunk["branch"]
        prompt = chunk["prompt"]
        print(f"Generating chunk {i+1}/{len(CANONICAL_MAP_PROMPTS)}: {branch}...", flush=True)

        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                print(f"  WARNING: Empty response for {branch}, skipping", flush=True)
                continue

            result = json.loads(content)
            entries = result.get("canonical_entries", [])
            tokens = response.usage.total_tokens
            total_tokens += tokens
            all_entries.extend(entries)
            print(f"  -> {len(entries)} entries, {tokens} tokens", flush=True)

        except json.JSONDecodeError as e:
            print(f"  WARNING: JSON parse error for {branch}: {e}", flush=True)
        except Exception as e:
            print(f"  ERROR for {branch}: {e}", flush=True)

        time.sleep(3)

    # Count by category and importance
    by_category = {}
    by_importance = {}
    for e in all_entries:
        cat = e.get("category", "unknown")
        imp = e.get("importance", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
        by_importance[imp] = by_importance.get(imp, 0) + 1

    result = {
        "canonical_entries": all_entries,
        "total_count": len(all_entries),
        "branch_counts": by_category,
        "importance_counts": by_importance,
        "total_tokens": total_tokens,
        "generated_at": datetime.now().isoformat(),
    }

    return result


def search_landmark_papers(canonical_map, max_entries=None):
    """For each canonical entry, find landmark papers on Semantic Scholar."""
    entries = canonical_map["canonical_entries"]
    if max_entries:
        entries = entries[:max_entries]

    print(f"\nSearching for landmark papers for {len(entries)} canonical entries...", flush=True)

    results = []
    for i, entry in enumerate(entries):
        query = entry.get("search_query", entry["name"])
        name = entry["name"]

        try:
            params = {
                "query": query,
                "fields": S2_FIELDS,
                "limit": 5,
                "minCitationCount": 50,
            }
            resp = requests.get(S2_BULK_SEARCH, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                papers = data.get("data", [])
                entry_result = {
                    **entry,
                    "papers_found": len(papers),
                    "top_papers": [
                        {
                            "paperId": p.get("paperId"),
                            "title": p.get("title"),
                            "year": p.get("year"),
                            "citationCount": p.get("citationCount", 0),
                            "doi": (p.get("externalIds") or {}).get("DOI", ""),
                            "isOpenAccess": p.get("isOpenAccess", False),
                            "abstract": (p.get("abstract") or "")[:300],
                        }
                        for p in sorted(papers, key=lambda x: x.get("citationCount", 0) or 0, reverse=True)[:5]
                    ],
                }
                results.append(entry_result)

                top = entry_result["top_papers"][0] if entry_result["top_papers"] else None
                if top:
                    print(f"  [{i+1}/{len(entries)}] {name}: {len(papers)} papers, top={top['title'][:50]}... ({top['citationCount']} cites)", flush=True)
                else:
                    print(f"  [{i+1}/{len(entries)}] {name}: no papers found", flush=True)

            elif resp.status_code == 429:
                print(f"  [{i+1}/{len(entries)}] {name}: rate limited, sleeping 30s...", flush=True)
                time.sleep(30)
                results.append({**entry, "papers_found": 0, "top_papers": [], "error": "rate_limited"})
            else:
                print(f"  [{i+1}/{len(entries)}] {name}: HTTP {resp.status_code}", flush=True)
                results.append({**entry, "papers_found": 0, "top_papers": [], "error": f"HTTP {resp.status_code}"})

        except Exception as e:
            print(f"  [{i+1}/{len(entries)}] {name}: ERROR {e}", flush=True)
            results.append({**entry, "papers_found": 0, "top_papers": [], "error": str(e)})

        time.sleep(3)

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            with open(EXPERIMENTS_DIR / "raw" / f"landmark_search_checkpoint_{i+1}.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Checkpoint saved: {i+1} entries", flush=True)

    return results


def analyze_coverage(landmark_results, existing_index_dir=None):
    """Analyze which canonical entries are/aren't covered by our index."""
    total = len(landmark_results)
    with_papers = sum(1 for r in landmark_results if r.get("papers_found", 0) > 0)
    fundamental = [r for r in landmark_results if r.get("importance") == "fundamental"]
    fundamental_with_papers = sum(1 for r in fundamental if r.get("papers_found", 0) > 0)

    # Collect all unique papers found
    all_papers = {}
    for r in landmark_results:
        for p in r.get("top_papers", []):
            pid = p.get("paperId")
            if pid and pid not in all_papers:
                all_papers[pid] = p

    analysis = {
        "total_canonical_entries": total,
        "entries_with_landmark_papers": with_papers,
        "entries_without_papers": total - with_papers,
        "fundamental_entries": len(fundamental),
        "fundamental_with_papers": fundamental_with_papers,
        "unique_landmark_papers": len(all_papers),
        "open_access_papers": sum(1 for p in all_papers.values() if p.get("isOpenAccess")),
        "papers_by_decade": {},
        "top_cited_papers": sorted(all_papers.values(), key=lambda x: x.get("citationCount", 0) or 0, reverse=True)[:20],
        "gaps": [
            {"name": r["name"], "category": r.get("category"), "importance": r.get("importance")}
            for r in landmark_results
            if r.get("papers_found", 0) == 0
        ],
    }

    # Papers by decade
    for p in all_papers.values():
        year = p.get("year")
        if year:
            decade = f"{(year // 10) * 10}s"
            analysis["papers_by_decade"][decade] = analysis["papers_by_decade"].get(decade, 0) + 1

    return analysis


def main():
    os.makedirs(EXPERIMENTS_DIR / "raw", exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR / "results", exist_ok=True)

    print(f"Top-Down Canonical Knowledge Map - {datetime.now().isoformat()}", flush=True)

    # Stage 1: Generate canonical map
    canonical_map = generate_canonical_map()

    with open(EXPERIMENTS_DIR / "raw" / "canonical_map.json", "w") as f:
        json.dump(canonical_map, f, indent=2)

    print(f"\nCanonical map: {canonical_map['total_count']} entries", flush=True)
    print(f"By category: {canonical_map['branch_counts']}", flush=True)
    print(f"By importance: {canonical_map['importance_counts']}", flush=True)

    # Stage 2: Search for landmark papers
    landmark_results = search_landmark_papers(canonical_map)

    with open(EXPERIMENTS_DIR / "raw" / "landmark_search_results.json", "w") as f:
        json.dump(landmark_results, f, indent=2)

    # Stage 3: Analyze coverage
    analysis = analyze_coverage(landmark_results)

    with open(EXPERIMENTS_DIR / "results" / "coverage_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("CANONICAL KNOWLEDGE MAP COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Canonical entries: {analysis['total_canonical_entries']}", flush=True)
    print(f"With landmark papers: {analysis['entries_with_landmark_papers']}", flush=True)
    print(f"Gaps (no papers found): {analysis['entries_without_papers']}", flush=True)
    print(f"Fundamental entries: {analysis['fundamental_entries']} ({analysis['fundamental_with_papers']} with papers)", flush=True)
    print(f"Unique landmark papers found: {analysis['unique_landmark_papers']}", flush=True)
    print(f"Open access: {analysis['open_access_papers']}", flush=True)

    if analysis["gaps"]:
        print(f"\nTop gaps (canonical knowledge without papers):", flush=True)
        for g in analysis["gaps"][:15]:
            print(f"  - [{g['importance']}] {g['name']} ({g['category']})", flush=True)

    print(f"\nTop cited landmark papers:", flush=True)
    for p in analysis["top_cited_papers"][:10]:
        print(f"  {p['title'][:60]}... ({p['citationCount']} cites, {p['year']})", flush=True)


if __name__ == "__main__":
    main()
