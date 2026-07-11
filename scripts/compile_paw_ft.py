"""Compile finetuned PAW rewrite functions for AskChem.

Recompiles the three rewrite specs (`expand_query`, `decompose_query`,
`normalize_query`) with the new finetune compiler
``paw-ft-bs48-20260522`` and saves the resulting program IDs to
``data/paw_ft_program_ids.json`` so the A/B harness in Phase 3 can swap
them in without losing the standard-compiler rollback path.

The specs are reused verbatim from the program metadata cached in
``~/.cache/programasweights/programs/<id>/meta.json`` (the same specs
that produced the IDs currently hard-coded in
``src/askchem/paw_functions.py``). Keeping the spec identical isolates
the comparison to the compiler change — any quality delta is purely
attributable to the finetune.

Run (in the PAW conda env so the SDK is on the right Python):

    /Users/bingyan/miniconda3/envs/paw/bin/python scripts/compile_paw_ft.py

Flags:

    --compiler <name>     Compiler snapshot. Default: paw-ft-bs48-20260522.
    --only expand|decompose|normalize
                          Compile a single function (default: all three).
    --output <path>       Where to write the IDs JSON (default:
                          data/paw_ft_program_ids.json).
    --skip-sanity         Skip per-function sanity probe after compile.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GGML_NO_METAL", "1")  # LoRA + Metal crashes (see paw_functions.py).

import programasweights as paw

DEFAULT_COMPILER = "paw-ft-bs48-20260522"
DEFAULT_OUTPUT = Path("data") / "paw_ft_program_ids.json"

EXPANDER_SPEC = """Given a chemistry search query, output a comma-separated list of synonyms, abbreviations, specific examples, and related terms that should also be searched.

Input: heavy metal adsorption
Output: Pb, Cd, Cr, Hg, Zn, Cu, Ni, lead, cadmium, chromium, mercury, removal, uptake, sorption

Input: water splitting
Output: HER, OER, hydrogen evolution, oxygen evolution, photocatalytic H2, electrolysis

Input: carbon nanotube synthesis
Output: CNT, MWCNT, SWCNT, nanotube growth, CVD, arc discharge, chemical vapor deposition

Input: Suzuki coupling
Output: Suzuki-Miyaura, cross-coupling, palladium, Pd, boronic acid, aryl halide, SPhos, XPhos

Input: CO2 reduction catalyst
Output: carbon dioxide, electrocatalysis, Faradaic efficiency, CO, formate, overpotential, current density

Input: drug delivery nanoparticle
Output: nanocarrier, controlled release, targeted delivery, liposome, polymer, encapsulation, drug release

Input: perovskite solar cell
Output: PSC, CH3NH3PbI3, MAPbI3, photovoltaic, power conversion efficiency, methylammonium

Input: MOF gas storage
Output: metal-organic framework, porous material, hydrogen storage, methane uptake, BET surface area

Input: lithium ion battery cathode
Output: LIB, NMC, NCA, LFP, LiCoO2, capacity, cycling stability, energy density

Input: FTIR spectroscopy analysis
Output: Fourier transform infrared, IR, vibrational, absorption band, wavenumber, ATR

Input: photocatalytic degradation dye
Output: photocatalysis, TiO2, ZnO, methylene blue, rhodamine, UV light, reactive oxygen species, ROS

Input: quantum dot fluorescence
Output: QD, semiconductor nanocrystal, photoluminescence, FRET, emission, CdSe, InP

Input: polymer electrolyte membrane
Output: PEM, Nafion, proton conductivity, fuel cell, solid electrolyte, ion exchange

Input: graphene oxide reduction
Output: GO, rGO, reduced graphene oxide, Hummers method, thermal reduction, chemical reduction

Input: zeolite catalysis
Output: ZSM-5, Y-zeolite, microporous, shape selectivity, acid sites, cracking, isomerization

Input: chitosan heavy metal removal
Output: biopolymer, biosorption, Cr, Pb, Cd, Hg, wastewater, chelation, adsorption capacity

Input: electrochemical impedance spectroscopy
Output: EIS, Nyquist plot, charge transfer resistance, Warburg, equivalent circuit

Input: rare earth luminescence
Output: lanthanide, Eu, Tb, Dy, phosphor, upconversion, down-conversion, emission

Input: CRISPR gene editing
Output: Cas9, guide RNA, sgRNA, genome engineering, gene knockout, HDR, NHEJ

Input: supercapacitor electrode material
Output: EDLC, pseudocapacitor, specific capacitance, MnO2, carbon, graphene, energy density"""


DECOMPOSER_SPEC = """Given a chemistry research question, output 4-5 comma-separated keyword search queries covering different sub-topics, materials, or techniques. Each query targets a specific sub-category. Include element symbols, material names, and technique names. Do NOT repeat terms within or across queries.

Examples:
Input: What electrocatalysts have been reported for CO2 reduction to CO or formate?
Output: CO2 electroreduction Au Ag gold silver nanoparticle selectivity, CO2 reduction formate Sn Bi In tin bismuth indium oxide, molecular CO2 catalyst metalloporphyrin phthalocyanine cobalt, oxide-derived Cu Zn CO2RR oxygen vacancy defect, MOF-derived carbon M-N-C Fe-N-C single atom CO2

Input: What adsorbent materials have been used for heavy metal removal from water?
Output: activated carbon biochar heavy metal Pb Cd Cr adsorption, zeolite bentonite clay mineral adsorbent wastewater, Fe3O4 magnetic nanoparticle graphene oxide heavy metal, chitosan cellulose biosorbent agricultural waste, MOF hydrogel polymer composite adsorbent

Input: How has perovskite degradation understanding evolved?
Output: CH3NH3PbI3 MAPbI3 moisture oxygen decomposition PbI2, halide perovskite ion migration vacancy defect, mixed-cation halide perovskite phase segregation, perovskite passivation encapsulation stability, operando characterization degradation GIWAXS XRD

Input: What catalysts for Suzuki-Miyaura coupling of aryl chlorides?
Output: Suzuki aryl chloride Pd PPh3 SPhos XPhos ligand, NHC N-heterocyclic carbene palladium Suzuki coupling, nickel catalyst NiCl2 dppp coupling aryl chloride, heterogeneous Pd nanoparticle Pd/C supported catalyst, Suzuki deactivated aryl chloride electron-rich phosphine

Input: Are silver nanoparticles toxic or safe for biomedical use?
Output: AgNP silver nanoparticle cytotoxicity cell viability ROS, silver nanoparticle antimicrobial antibacterial MIC zone inhibition, AgNP size shape coating PVP citrate toxicity, silver nanoparticle wound healing tissue engineering biocompatibility, Ag ion release dissolution environmental aquatic toxicity"""


NORMALIZER_SPEC = """Extract the core chemistry search terms from a natural language query.

Remove question framing words (what is, how does, why do, can you explain, etc.), filler words, and conversational phrasing. Keep only the essential chemistry/science terms that should be searched.

If the input contains a well-known chemistry abbreviation, expand it AND keep the abbreviation.

Output ONLY the cleaned search terms, nothing else. Do not add explanation.

Examples:
Input: what is the mechanism of L-BFGS?
Output: L-BFGS mechanism

Input: how does Suzuki coupling work
Output: Suzuki coupling

Input: can you explain DFT calculations for TiO2
Output: DFT density functional theory TiO2

Input: what are the applications of palladium catalysis in organic synthesis
Output: palladium catalysis organic synthesis

Input: why is benzene stable
Output: benzene stability

Input: tell me about metal oxide nanoparticles
Output: metal oxide nanoparticles

Input: machine learning for drug discovery
Output: machine learning drug discovery

Input: L-BFGS
Output: L-BFGS
"""


SANITY_PROBES = {
    "expand": [
        ("heavy metal adsorption", ["pb", "cd", "cr"]),
        ("Suzuki coupling", ["pd", "palladium"]),
        ("graphene oxide reduction", ["go", "rgo"]),
    ],
    "decompose": [
        ("What electrocatalysts have been reported for CO2 reduction?", 4),
        ("How has TiO2 photocatalysis mechanism evolved?", 4),
        ("Are silver nanoparticles toxic or safe for biomedical use?", 4),
    ],
    "normalize": [
        ("what is the mechanism of L-BFGS?", "l-bfgs"),
        ("how does Suzuki coupling work", "suzuki coupling"),
        ("tell me about metal oxide nanoparticles", "metal oxide nanoparticles"),
    ],
}


PROGRAMS = {
    "expand": ("QUERY_EXPANDER_PROGRAM_ID", EXPANDER_SPEC),
    "decompose": ("QUERY_DECOMPOSER_PROGRAM_ID", DECOMPOSER_SPEC),
    "normalize": ("NORMALIZER_PROGRAM_ID", NORMALIZER_SPEC),
}


def compile_with_retry(spec: str, compiler: str, max_attempts: int = 5) -> str:
    """Compile a spec with the configured compiler, retrying on transient 5xx errors.

    Mirrors the existing retry pattern in scripts/compile_paw_query_expander.py
    so finetune compiles (which can sit in a queue for several minutes) survive
    the same transient server hiccups.
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            t0 = time.time()
            result = paw.compile(spec, compiler=compiler)
            program_id = result.id if hasattr(result, "id") else str(result)
            elapsed = time.time() - t0
            print(f"    compiled in {elapsed:.1f}s, program_id={program_id}")
            return program_id
        except Exception as e:  # noqa: BLE001 — broad except matches existing scripts.
            last_error = e
            msg = str(e)
            msg_lower = msg.lower()
            transient = (
                "500" in msg
                or "503" in msg
                or "timeout" in msg_lower
                or "timed out" in msg_lower
                or "read operation" in msg_lower
                or "connection" in msg_lower
            )
            if not transient or attempt == max_attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    compile failed ({msg!r}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"compile retries exhausted: {last_error}")  # unreachable


def load_with_retry(program_id: str, max_attempts: int = 12):
    """Load a compiled program, retrying while assets propagate to disk."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return paw.function(program_id, n_gpu_layers=0)
        except Exception as e:  # noqa: BLE001 — broad except matches existing scripts.
            last_error = e
            wait = 5 * (attempt + 1)
            print(f"    load attempt {attempt + 1}/{max_attempts} failed: {e}; "
                  f"retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"load retries exhausted: {last_error}")


def sanity_check(kind: str, fn) -> tuple[int, int]:
    """Probe the compiled function with a small in-domain check.

    Returns (passed, total). For the expander/normalizer we look for
    keyword overlap; for the decomposer we count comma-separated parts.
    """
    probes = SANITY_PROBES[kind]
    passed = 0
    for idx, probe in enumerate(probes, 1):
        if kind == "decompose":
            inp, min_parts = probe
        else:
            inp, expected_any = probe
        t0 = time.time()
        raw = fn(inp).strip()
        elapsed = time.time() - t0
        snippet = raw.replace("\n", " ")[:120]
        if kind == "decompose":
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            ok = len(parts) >= min_parts
            note = f"got {len(parts)} parts (want ≥ {min_parts})"
        else:
            lower = raw.lower()
            hits = [t for t in expected_any if t.lower() in lower]
            ok = bool(hits)
            note = f"matched {hits}"
        mark = "OK" if ok else "MISS"
        passed += ok
        print(f"    [{mark}] ({elapsed:.2f}s) {inp!r}")
        print(f"          out: {snippet!r}")
        print(f"          note: {note}")
    return passed, len(probes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler", default=DEFAULT_COMPILER,
                    help="Compiler snapshot to use.")
    ap.add_argument("--only", choices=sorted(PROGRAMS.keys()), default=None,
                    help="Compile only this function (default: all).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="JSON file to receive the program IDs.")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="Skip the per-function probe after compile.")
    args = ap.parse_args()

    selected = [args.only] if args.only else list(PROGRAMS.keys())
    print(f"Compiler: {args.compiler}")
    print(f"Selected: {selected}")
    print(f"Output:   {args.output}")

    existing: dict = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except Exception:
            existing = {}
    results = dict(existing)
    # Track the compiler we used for the latest write so the integration
    # step can sanity-check that the ft IDs were actually produced by the
    # finetune snapshot.
    results.setdefault("_meta", {})
    results["_meta"]["compiler"] = args.compiler
    results["_meta"]["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    summary: list[tuple[str, str, str, str]] = []
    for kind in selected:
        const_name, spec = PROGRAMS[kind]
        print(f"\n=== {kind} ({const_name}) ===")
        print(f"  spec length: {len(spec)} chars")
        try:
            program_id = compile_with_retry(spec, args.compiler)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: compile failed: {e}")
            summary.append((kind, const_name, "FAILED", str(e)))
            continue

        results[kind] = {
            "program_id": program_id,
            "constant": const_name,
            "compiler": args.compiler,
        }

        # Persist after each compile so a late failure doesn't lose earlier wins.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.output}")

        if args.skip_sanity:
            summary.append((kind, const_name, "COMPILED", program_id))
            continue

        print("  loading program for sanity probe...")
        try:
            fn = load_with_retry(program_id)
            try:
                fn("warmup")
            except Exception:
                pass
            passed, total = sanity_check(kind, fn)
            summary.append((kind, const_name, f"{passed}/{total}", program_id))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: sanity probe failed to load: {e}")
            summary.append((kind, const_name, "LOAD_FAIL", program_id))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for kind, const, status, info in summary:
        print(f"  {kind:>10s}  {const:<30s}  {status:<10s}  {info}")

    print(f"\nIDs written to {args.output}")
    print("To preview the integration constants:")
    for kind in selected:
        if kind in results and isinstance(results[kind], dict):
            const = results[kind]["constant"]
            pid = results[kind]["program_id"]
            print(f'  {const} = "{pid}"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
