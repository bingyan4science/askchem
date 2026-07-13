"""Prompt lab: A/B different prompts for extracting the PRINCIPLES,
MECHANISMS and THEORIES embodied in a paper (multiple per paper allowed).

The 30-paper grow run mislabeled techniques as principles ("Hydrothermal
Synthesis", "Sol-Gel Process"). This harness runs several prompt strategies
over the same diverse papers and writes a side-by-side report so we can pick
the variant that generates principles/mechanisms/theories *correctly*.

Variants differ along two axes:
  - INPUT:  paper (title+abstract+summary) vs CLAIMS (extracted reactions) vs both
  - FRAMING: free / defined+fewshot / bottom-up two-level

LLM = Gemini via the NYU gateway (sync; low volume).

Usage:
    export PORTKEY_API_KEY=...
    python3 living_taxonomy/prompt_lab.py
    open living_taxonomy/output/prompt_lab/report.md
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import pilot_data
import placement as pm
from incremental_build import _parse_json

DB = "file:" + str(_HERE.parent / "askchem.db") + "?immutable=1"
OUT = _HERE / "output" / "prompt_lab"

# Diverse, telling sample (incl. cases that previously produced bad principles).
SAMPLE_DOIS = [
    "10.30919/ESMM5F712",        # Ag/TiO2/rGO nanocomposite synthesis (technique trap)
    "10.1038/s41598-017-12786-6",# curcumin@ZIF MOF drug delivery
    "10.1038/s41467-022-33232-w",# electrosynthesis of formamide
    "10.1002/anie.201410322",    # Ni-catalyzed dehydrogenative cross-coupling (clean)
    "10.1016/J.SNB.2015.05.107", # polyaniline gas sensors (application trap)
]

# Words that signal a *technique/application*, not a principle (heuristic flag).
_TECHNIQUE_CUES = (
    "synthesis", "process", "fabrication", "preparation", "deposition",
    "sensing", "sensor", "characterization", "spectroscopy", "microscopy",
    "method", "technique", "application", "device", "coating", "imaging",
)


def fetch_paper(doi):
    conn = sqlite3.connect(DB, uri=True)
    row = conn.execute(
        "SELECT title, abstract, paper_summary, year FROM sources WHERE doi=?",
        (doi,),
    ).fetchone()
    title, abstract, summary, year = row or ("", "", "", 0)
    claims = []
    for ct, (data_json,) in [
        (ct, r) for ct in ("reaction", "mechanism", "method")
        for r in conn.execute(
            "SELECT data FROM claims WHERE source_doi=? AND claim_type=? LIMIT 8",
            (doi, ct))
    ]:
        try:
            d = json.loads(data_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if ct == "reaction":
            txt = pilot_data._reaction_leaf_text(d)
        else:
            txt = (d.get("process_described") or d.get("what_it_achieves")
                   or d.get("verbatim_quote") or "")
        if txt:
            claims.append(f"({ct}) {txt}")
    conn.close()
    return {"doi": doi, "title": title or "", "abstract": abstract or "",
            "summary": summary or "", "year": year or 0, "claims": claims[:12]}


# ── prompt variants ──────────────────────────────────────────────────────────

_DEFS = (
    "Definitions:\n"
    "- PRINCIPLE: a fundamental, general governing law/concept that holds across "
    "many systems (e.g. single-electron transfer, conservation of orbital "
    "symmetry, Bronsted acid-base catalysis, thermodynamic vs kinetic control, "
    "microscopic reversibility). NOT a technique, material, or application.\n"
    "- MECHANISM: the specific elementary-step pathway of a transformation "
    "(e.g. oxidative addition / transmetalation / reductive elimination; SN2 "
    "backside attack; radical-chain hydrogen-atom transfer).\n"
    "- THEORY: an explanatory framework (e.g. Marcus theory, transition-state "
    "theory, crystal-field/ligand-field theory, band theory).\n"
    "Do NOT return techniques/processes/applications such as 'hydrothermal "
    "synthesis', 'sol-gel process', 'polymer synthesis', 'gas sensing'."
)

_SCHEMA = (
    'Return ONLY JSON: {"items":[{"type":"principle|mechanism|theory",'
    '"name":"...","definition":"one sentence","evidence":"which reaction/'
    'finding in the paper invokes it"}]}. A paper usually embodies MULTIPLE '
    "principles and mechanisms; list all that genuinely apply."
)


def _paper_block(p, with_claims):
    b = f"Title: {p['title']}\nAbstract: {p['abstract'][:1500]}"
    if p["summary"]:
        b += f"\nSummary: {p['summary'][:800]}"
    if with_claims and p["claims"]:
        b += "\nExtracted claims:\n" + "\n".join(
            f"  [{i}] {c}" for i, c in enumerate(p["claims"]))
    return b


VARIANTS = {
    "V1_paper_free": {
        "sys": "You identify the chemical principles, mechanisms and theories a "
               "paper embodies.",
        "user": lambda p: (
            f"{_paper_block(p, with_claims=False)}\n\n"
            "List the chemical principles, mechanisms and theories this paper "
            f"embodies (multiple allowed).\n{_SCHEMA}"),
    },
    "V2_claims_grounded": {
        "sys": "You infer the governing principles and mechanisms behind a "
               "paper's specific experimental claims.",
        "user": lambda p: (
            f"{_paper_block(p, with_claims=True)}\n\n"
            "For the claims above, infer the underlying chemical principles, "
            f"mechanisms and theories that govern them (multiple allowed).\n"
            f"{_SCHEMA}"),
    },
    "V3_defined_fewshot": {
        "sys": "You are a chemistry taxonomist extracting fundamental "
               "principles, mechanisms and theories. Be strict about the "
               "definitions; never output techniques or applications.",
        "user": lambda p: (
            f"{_DEFS}\n\n{_paper_block(p, with_claims=True)}\n\n"
            "Extract every principle, mechanism and theory this paper genuinely "
            f"embodies.\n{_SCHEMA}"),
    },
    "V4_bottom_up": {
        "sys": "You reason bottom-up from concrete chemistry to general "
               "principles.",
        "user": lambda p: (
            f"{_DEFS}\n\n{_paper_block(p, with_claims=True)}\n\n"
            "Step 1: name the specific MECHANISM(s) at work in the claims. "
            "Step 2: abstract each mechanism to the fundamental PRINCIPLE(s)/"
            "THEORY it is an instance of. Return both levels.\n"
            f"{_SCHEMA}"),
    },
}


def _flag(name):
    low = name.lower()
    return " [!technique?]" if any(c in low for c in _TECHNIQUE_CUES) else ""


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    papers = [fetch_paper(d) for d in SAMPLE_DOIS]
    report = ["# Prompt Lab — principle/mechanism/theory extraction\n",
              "`[!technique?]` flags an item whose name looks like a "
              "technique/application (heuristic).\n"]
    results = {}

    for p in papers:
        report.append(f"\n## {p['title'][:80]}\n`{p['doi']}` ({p['year']}), "
                      f"{len(p['claims'])} claims\n")
        results[p["doi"]] = {}
        for vname, v in VARIANTS.items():
            try:
                raw = pm._gemini_chat(v["sys"], v["user"](p), max_time=120)
                parsed = _parse_json(raw)
                items = parsed.get("items", [])
            except Exception as e:
                report.append(f"\n**{vname}**: ERROR {e}\n")
                results[p["doi"]][vname] = {"error": str(e)}
                continue
            results[p["doi"]][vname] = items
            n_flag = sum(1 for it in items if _flag(it.get("name", "")))
            report.append(f"\n**{vname}** ({len(items)} items, "
                          f"{n_flag} technique-flagged):\n")
            for it in items:
                report.append(
                    f"- _{it.get('type','?')}_: **{it.get('name','?')}**"
                    f"{_flag(it.get('name',''))} — {it.get('definition','')}")
            print(f"[lab] {p['doi'][:24]:24s} {vname:20s} "
                  f"{len(items):2d} items, {n_flag} flagged", file=sys.stderr)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "report.md").write_text("\n".join(report))
    print(f"\n[lab] wrote {OUT/'report.md'}", file=sys.stderr)


if __name__ == "__main__":
    run()
