"""
Select 10 diverse papers for deep extraction experiment and download PDFs.
Hand-picked for diversity across subfields and paper types.
"""

import json
import os
import requests
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"

SELECTED_DOIS = [
    # Organic synthesis - actual reaction paper with yields
    "10.1038/s41467-018-06019-1",   # "A general deoxygenation approach for synthesis of ketones"
    # Organic synthesis - copper catalysis with experimental results
    "10.1038/s41586-018-0234-8",    # "Decarboxylative sp3 C-N Coupling via Dual Copper/Photoredox"
    # Inorganic - MOF drug delivery with synthesis and characterization
    "10.1002/anie.201915848",       # "Multivariate Modulation of Zr MOF UiO-66"
    # Inorganic - MOF colloids with synthesis procedures
    "10.1039/c9cs00472f",           # "Colloidal metal-organic framework particles: ZIF-8"
    # Catalysis - CO2 reduction with experimental data
    "10.1021/jacs.1c11253",         # "Accelerating CO2 Electroreduction to Multicarbon Products"
    # Catalysis - Ru/TiO2 with catalytic data
    "10.1038/s41467-021-27910-4",   # "Interfacial compatibility controls Ru/TiO2 metal-support"
    # Physical chem - cobalt complex with spectroscopic data
    "10.1126/science.aat7319",      # "A linear cobalt(II) complex with maximal orbital angular momentum"
    # Physical chem - hydrogen evolution kinetics
    "10.1021/jacsau.1c00281",       # "Cation- and pH-Dependent Hydrogen Evolution"
    # Biochemistry - colibactin structure elucidation
    "10.1126/science.aax2685",      # "Structure elucidation of colibactin and its DNA cross-links"
    # Computational - ML for molecular wavefunctions
    "10.1038/s41467-019-12875-2",   # "Unifying ML and quantum chemistry with deep neural network"
]


def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)

    with open(DATA_DIR / "metadata" / "open_access_papers.json") as f:
        all_papers = json.load(f)

    # Build DOI lookup
    doi_to_paper = {}
    for p in all_papers:
        ext = p.get("externalIds") or {}
        doi = ext.get("DOI", "")
        if doi:
            doi_to_paper[doi.lower()] = p

    selected = []
    for doi in SELECTED_DOIS:
        paper = doi_to_paper.get(doi.lower())
        if paper:
            selected.append(paper)
            print(f"Found: [{paper.get('citationCount',0)} cites] {paper['title'][:70]}")
        else:
            print(f"NOT FOUND: {doi}")

    # Save selected papers metadata
    with open(DATA_DIR / "metadata" / "selected_10_papers.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved {len(selected)} selected papers metadata")

    # Download PDFs
    print(f"\nDownloading PDFs...")
    for i, paper in enumerate(selected):
        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "")
        if not pdf_url:
            print(f"  [{i+1}] No PDF URL: {paper['title'][:50]}")
            continue

        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in paper["title"])[:60].strip()
        filename = f"{i+1:02d}_{safe_title}.pdf"
        filepath = PAPERS_DIR / filename

        if filepath.exists():
            print(f"  [{i+1}] Already downloaded: {filename}")
            continue

        print(f"  [{i+1}] Downloading: {filename}")
        print(f"       URL: {pdf_url[:80]}")
        try:
            resp = requests.get(pdf_url, timeout=60, headers={
                "User-Agent": "AskChem/1.0 (academic research; mailto:askchem@mit.edu)"
            })
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                print(f"       Saved ({len(resp.content)//1024} KB)")
            else:
                print(f"       Failed: status={resp.status_code}, size={len(resp.content)}")
        except Exception as e:
            print(f"       Error: {e}")
        time.sleep(2)

    # List downloaded files
    print(f"\nDownloaded PDFs:")
    for f in sorted(PAPERS_DIR.iterdir()):
        if f.suffix == ".pdf":
            print(f"  {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
