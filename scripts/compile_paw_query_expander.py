"""Compile PAW query expander for chemistry search.

Given a chemistry search query, output comma-separated synonyms,
abbreviations, specific examples, and related terms for vocabulary expansion.

Run: /Users/bingyan/miniconda3/envs/paw/bin/python scripts/compile_paw_query_expander.py
"""
import sys
import time

import programasweights as paw

SPEC = """Given a chemistry search query, output a comma-separated list of synonyms, abbreviations, specific examples, and related terms that should also be searched.

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


def main():
    print("Compiling PAW query expander...")
    print(f"Spec length: {len(SPEC)} chars")

    for attempt in range(5):
        try:
            result = paw.compile(SPEC, compiler="paw-4b-qwen3-0.6b")
            program_id = result.id if hasattr(result, "id") else str(result)
            print(f"\nSUCCESS! Program ID: {program_id}")

            print("\nWaiting for artifact to become loadable...")
            fn = None
            last_error = None
            for load_attempt in range(10):
                try:
                    fn = paw.function(program_id, n_gpu_layers=0)
                    break
                except Exception as e:
                    last_error = e
                    wait = 5 * (load_attempt + 1)
                    print(f"  Load attempt {load_attempt + 1}/10 failed: {e}. "
                          f"Retrying in {wait}s...")
                    time.sleep(wait)

            if fn is None:
                print(f"  Error: could not load compiled program: {last_error}")
                sys.exit(1)

            print("\nSanity check...")
            tests = [
                ("heavy metal adsorption",
                 ["Pb", "Cd", "Cr", "removal"]),
                ("water splitting",
                 ["HER", "OER", "hydrogen", "evolution"]),
                ("Suzuki coupling",
                 ["palladium", "Pd", "boronic"]),
                ("FTIR spectroscopy analysis",
                 ["infrared", "IR", "wavenumber"]),
                ("graphene oxide reduction",
                 ["GO", "rGO", "Hummers"]),
            ]
            passed = 0
            for inp, expected_any in tests:
                raw = fn(inp).strip()
                terms = [t.strip().lower() for t in raw.split(",")]
                found = [e for e in expected_any
                         if any(e.lower() in t for t in terms)]
                ok = len(found) >= 1
                passed += ok
                mark = "OK" if ok else "MISS"
                print(f"  {mark} Input: {inp}")
                print(f"       Output: {raw[:120]}")
                print(f"       Matched: {found}")

            print(f"\nPassed: {passed}/{len(tests)}")
            print(f"\nProgram ID for paw_functions.py:")
            print(f'  QUERY_EXPANDER_PROGRAM_ID = "{program_id}"')
            return

        except Exception as e:
            if "500" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Server error (attempt {attempt + 1}/5), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                sys.exit(1)

    print("Failed after 5 attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
