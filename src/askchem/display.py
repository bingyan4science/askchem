"""
Display name utilities for AskChem.

Converts underscore_separated identifiers to properly-cased display names,
preserving chemistry abbreviations (NMR, DFT, MOF, etc.).
"""

# Abbreviations that should be ALL-CAPS when they appear as a whole word.
# Includes common chemistry acronyms, techniques, and units.
_ABBREVIATIONS = {
    # Spectroscopy / Analytical (ir/cd removed — ambiguous with element symbols)
    "nmr", "uv", "ms", "xrd", "xps", "xrf", "xas", "xanes", "exafs",
    "epr", "esr", "esi", "maldi", "tof", "hplc", "gc", "lc", "icp",
    "aas", "sers", "ftir", "saxs", "waxs", "pxrd", "eds", "edx",
    "eels", "sem", "tem", "afm", "stm", "spm", "hrtem", "haadf",
    # Computational
    "dft", "tddft", "md", "mc", "qm", "mm", "ai", "ml", "gnn", "vae",
    "llm", "nlp", "gpt", "bert", "mpnn",
    # Materials / Chemistry
    "mof", "cof", "zif", "pom", "peg", "pvdf", "ptfe", "pdms",
    "bet", "ssa", "tga", "dsc", "dta",
    # Electrochemistry
    "rde", "rrde", "orr", "oer", "her", "co2rr", "n2rr", "ecsa",
    "cv", "eis", "lsv", "pcet", "set", "hat",
    # Catalysis
    "ton", "tof",
    # Selectivity / Stereochemistry
    "ee", "de", "dr",
    # Misc chemistry
    "sar", "qsar", "admet", "pka", "pkb", "ic50", "ec50", "ld50",
    "ph", "smiles", "inchi",
    # Units / Standards
    "rpm", "psi", "atm",
    # Reaction types
    "snr", "sn1", "sn2",
    # Biological
    "dna", "rna", "atp", "nad", "fad", "pcr",
    "h2", "o2", "n2",
    # Hybrid terms
    "sp2", "sp3", "sp",
    # Misc
    "api", "sdk", "mcp", "doi", "cas", "iupac",
    "2d", "3d", "1d",
}

# Element symbols: should be Title-Cased (Rh, Cu, Fe), not ALL-CAPS.
_ELEMENTS = {
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne",
    "na", "mg", "al", "si", "p", "s", "cl", "ar",
    "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn",
    "ga", "ge", "as", "se", "br", "kr",
    "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd",
    "in", "sn", "sb", "te", "i", "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy",
    "ho", "er", "tm", "yb", "lu",
    "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi",
    "po", "at", "rn",
}

# Words that should stay lowercase (articles, prepositions, conjunctions)
# unless they're the first word.
_LOWERCASE_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "by", "of", "up", "as", "is", "if",
    "via", "vs", "per",
}

# Bond patterns: sequences of single-letter words that should be joined with hyphens.
# Applied BEFORE word-level processing. Matches "c h" but not "ch" or "carbon h".
_BOND_PATTERNS = {
    "c h": "C–H",
    "c c": "C–C",
    "c n": "C–N",
    "c o": "C–O",
    "c s": "C–S",
    "c f": "C–F",
    "c cl": "C–Cl",
    "c br": "C–Br",
    "n h": "N–H",
    "o h": "O–H",
    "n n": "N–N",
    "n o": "N–O",
    "s s": "S–S",
}


import re as _re

# Well-known chemical formulas with canonical capitalization
_KNOWN_FORMULAS = {
    "co2": "CO2", "h2o": "H2O", "h2o2": "H2O2", "nh3": "NH3", "ch4": "CH4",
    "no2": "NO2", "so2": "SO2", "no": "NO", "co": "CO",
    "tio2": "TiO2", "sio2": "SiO2", "al2o3": "Al2O3", "zno": "ZnO",
    "ceo2": "CeO2", "fe2o3": "Fe2O3", "fe3o4": "Fe3O4", "mno2": "MnO2",
    "cuo": "CuO", "nio": "NiO", "zro2": "ZrO2", "v2o5": "V2O5",
    "wo3": "WO3", "sno2": "SnO2", "in2o3": "In2O3", "ga2o3": "Ga2O3",
    "lifepo4": "LiFePO4", "licoo2": "LiCoO2", "limno2": "LiMnO2",
    "nacl": "NaCl", "kcl": "KCl", "cacl2": "CaCl2", "mgcl2": "MgCl2",
    "naoh": "NaOH", "koh": "KOH", "hcl": "HCl", "h2so4": "H2SO4",
    "hno3": "HNO3", "batio3": "BaTiO3", "srceo3": "SrCeO3",
}

_FORMULA_RE = _re.compile(r'^([a-z]{1,2}\d*)+$')
_FORMULA_TOKEN_RE = _re.compile(r'([a-z]{1,2})(\d*)')


def _is_chemical_formula(word: str) -> bool:
    """Detect chemical formulas like tio2, sio2, h2o, fe3o4."""
    if word in _KNOWN_FORMULAS:
        return True
    if not _FORMULA_RE.match(word):
        return False
    tokens = _FORMULA_TOKEN_RE.findall(word)
    has_element = any(t[0] in _ELEMENTS for t in tokens)
    has_number = any(t[1] for t in tokens)
    return has_element and has_number and len(tokens) >= 2


def _format_formula(word: str) -> str:
    """Format a chemical formula with correct capitalization."""
    if word in _KNOWN_FORMULAS:
        return _KNOWN_FORMULAS[word]
    tokens = _FORMULA_TOKEN_RE.findall(word)
    parts = []
    for sym, num in tokens:
        if sym in _ELEMENTS:
            parts.append(sym.capitalize() + num)
        else:
            parts.append(sym.upper() + num)
    return ''.join(parts)


def smart_title(slug: str) -> str:
    """Convert an underscore_separated slug to a properly-cased display name.

    Examples:
        smart_title("rotating_disk_electrode") -> "Rotating Disk Electrode"
        smart_title("nmr_spectroscopy") -> "NMR Spectroscopy"
        smart_title("ml_for_drug_discovery") -> "ML for Drug Discovery"
        smart_title("c_h_activation") -> "C-H Activation"
        smart_title("dft_computational_methods") -> "DFT Computational Methods"
    """
    if not slug:
        return ""

    text = slug.replace("_", " ")

    # Apply bond patterns: replace sequences of single-letter words like "c h"
    # with bonded forms like "C–H". Must match as whole words.
    import re
    for pattern, replacement in _BOND_PATTERNS.items():
        regex = r'\b' + r'\b\s+\b'.join(re.escape(w) for w in pattern.split()) + r'\b'
        text = re.sub(regex, replacement, text, flags=re.IGNORECASE)

    words = text.split()
    result = []

    for i, word in enumerate(words):
        lower = word.lower()

        if lower in _ABBREVIATIONS:
            result.append(word.upper())
        elif lower in _KNOWN_FORMULAS:
            result.append(_KNOWN_FORMULAS[lower])
        elif _is_chemical_formula(lower):
            result.append(_format_formula(lower))
        elif lower in _ELEMENTS and len(lower) <= 3:
            result.append(lower.capitalize())
        elif "–" in word:
            result.append(word)
        elif i > 0 and lower in _LOWERCASE_WORDS:
            result.append(lower)
        elif word == word.upper() and len(word) > 1:
            result.append(word)
        else:
            result.append(word.capitalize())

    return " ".join(result)
