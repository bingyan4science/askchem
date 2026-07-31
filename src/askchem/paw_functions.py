"""PAW (ProgramAsWeights) neural functions for AskChem.

Provides locally-running fuzzy text functions compiled from natural language specs.
All functions lazy-load on first call and gracefully fall back if PAW is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import threading

os.environ.setdefault("GGML_NO_METAL", "1")  # LoRA + Metal crashes in llama-cpp-python 0.3.19

log = logging.getLogger(__name__)

INTENT_PROGRAM_ID = "32765bb3d684d7fa604d"
NORMALIZER_PROGRAM_ID = "a065bf28afb81bb1759c"
RELEVANCE_PROGRAM_ID = "fd876c540a35f4a52018"
CONTRADICTION_PROGRAM_ID = "2ad2222a366ba8c18de8"  # v4c: 75% acc, 100% precision on 20-pair bench
QUERY_EXPANDER_PROGRAM_ID = "23d74e49bcb1ff445a7d"  # V3_ft (May-28 spec sweep): 20pos + 5neg, paw-ft-bs48-20260522. Macro +0.30 on 30-probe bench vs +0.19 baseline.
QUERY_DECOMPOSER_PROGRAM_ID = "f8d7f07cacc416fa4280"


# ── Per-function program-ID override ─────────────────────────────────────────
#
# Setting ``CHEMTREE_PAW_FT_IDS=<path>`` swaps the constants above with the IDs
# in the JSON file written by ``scripts/compile_paw_ft.py``.  This is the lever
# the Phase 3 A/B harness uses to flip between the standard-compiler IDs (the
# defaults that have shipped) and the finetuned-compiler IDs from
# ``paw-ft-bs48-20260522`` without editing source between runs.
#
# Expected layout (extras are ignored):
#
#     {
#       "expand":    {"program_id": "fe55...", "constant": "QUERY_EXPANDER_PROGRAM_ID"},
#       "decompose": {"program_id": "4d83...", "constant": "QUERY_DECOMPOSER_PROGRAM_ID"},
#       "normalize": {"program_id": "1d5c...", "constant": "NORMALIZER_PROGRAM_ID"}
#     }
def _maybe_load_ft_ids() -> None:
    """If CHEMTREE_PAW_FT_IDS points at a JSON file, override the constants."""
    path = os.environ.get("CHEMTREE_PAW_FT_IDS", "").strip()
    if not path:
        return
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception:
        log.exception("CHEMTREE_PAW_FT_IDS=%r unreadable; using default program IDs", path)
        return

    globals_ = globals()
    swapped: list[str] = []
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        const = entry.get("constant")
        pid = entry.get("program_id")
        if const in globals_ and isinstance(pid, str) and pid:
            globals_[const] = pid
            swapped.append(f"{const}={pid}")
    if swapped:
        log.info("PAW ft overrides applied from %s: %s", path, ", ".join(swapped))


_maybe_load_ft_ids()

_lock = threading.Lock()
_intent_fn = None
_normalizer_fn = None
_relevance_fn = None
_contradiction_fn = None
_query_expander_fn = None
_query_decomposer_fn = None
_paw_available: bool | None = None


def _check_paw() -> bool:
    """Return True iff PAW (programasweights) is importable AND not disabled.

    Disabling PAW saves roughly 640 MB of resident memory because separate
    qwen3-0.6b programs are loaded for intent and normalization. Set
    CHEMTREE_DISABLE_PAW=1 to force classify_intent, normalize_query,
    detect_contradiction, and related calls through their documented
    fallback paths.
    """
    global _paw_available
    if os.environ.get("CHEMTREE_DISABLE_PAW", "0") == "1":
        return False
    if _paw_available is not None:
        return _paw_available
    try:
        import programasweights  # noqa: F401
        _paw_available = True
    except Exception:
        log.exception("programasweights unavailable; PAW functions disabled")
        _paw_available = False
    return _paw_available


def _load_fn(program_id: str):
    import programasweights as paw
    return paw.function(program_id, n_gpu_layers=0)


def _get_intent_fn():
    global _intent_fn
    if _intent_fn is None:
        with _lock:
            if _intent_fn is None:
                _intent_fn = _load_fn(INTENT_PROGRAM_ID)
    return _intent_fn


def _get_normalizer_fn():
    global _normalizer_fn
    if _normalizer_fn is None:
        with _lock:
            if _normalizer_fn is None:
                _normalizer_fn = _load_fn(NORMALIZER_PROGRAM_ID)
    return _normalizer_fn


def _get_relevance_fn():
    global _relevance_fn
    if _relevance_fn is None:
        with _lock:
            if _relevance_fn is None:
                _relevance_fn = _load_fn(RELEVANCE_PROGRAM_ID)
    return _relevance_fn


def _get_contradiction_fn():
    global _contradiction_fn
    if _contradiction_fn is None:
        with _lock:
            if _contradiction_fn is None:
                _contradiction_fn = _load_fn(CONTRADICTION_PROGRAM_ID)
    return _contradiction_fn


def _get_query_expander_fn():
    global _query_expander_fn
    if _query_expander_fn is None:
        with _lock:
            if _query_expander_fn is None:
                _query_expander_fn = _load_fn(QUERY_EXPANDER_PROGRAM_ID)
    return _query_expander_fn


def _get_query_decomposer_fn():
    global _query_decomposer_fn
    if _query_decomposer_fn is None:
        with _lock:
            if _query_decomposer_fn is None:
                _query_decomposer_fn = _load_fn(QUERY_DECOMPOSER_PROGRAM_ID)
    return _query_decomposer_fn


VALID_INTENTS = {"author", "substance", "method", "concept", "paper"}
VALID_CONTRADICTION_VERDICTS = {"contradicts", "compatible", "unclear"}


def _normalize_contradiction_verdict(raw: str) -> str:
    text = (raw or "").strip().strip('"\'').lower()
    if not text:
        return "unclear"

    token = text.split()[0].strip(".,:;!?()[]{}")
    if token.startswith("contradict"):
        return "contradicts"
    if token.startswith("compat"):
        return "compatible"
    if token.startswith("unclear"):
        return "unclear"
    return "unclear"


def classify_intent(query: str) -> str:
    """Classify a search query as author/substance/method/concept/paper.

    Falls back to 'concept' if PAW is unavailable or returns an unexpected value.
    """
    if not _check_paw():
        return "concept"
    try:
        fn = _get_intent_fn()
        result = fn(query).strip().lower()
        return result if result in VALID_INTENTS else "concept"
    except Exception:
        log.exception("PAW intent classification failed")
        return "concept"


def normalize_query(query: str) -> str:
    """Strip question framing and normalize chemistry terms for FTS.

    Falls back to the original query if PAW is unavailable.
    """
    if not _check_paw():
        return query
    try:
        fn = _get_normalizer_fn()
        result = fn(query).strip()
        return result if result else query
    except Exception:
        log.exception("PAW query normalization failed")
        return query


def is_relevant(query: str, claim_text: str) -> bool:
    """Check if a claim is relevant to a query.

    Falls back to True (assume relevant) if PAW is unavailable.
    """
    if not _check_paw():
        return True
    try:
        fn = _get_relevance_fn()
        inp = f"QUERY: {query} CLAIM: {claim_text}"
        result = fn(inp).strip().lower()
        return result != "not_relevant"
    except Exception:
        log.exception("PAW relevance check failed")
        return True


def detect_contradiction(claim_a: str, claim_b: str) -> str:
    """Determine if two claims contradict each other.

    Returns 'contradicts', 'compatible', or 'unclear'.
    Falls back to 'unclear' if PAW is unavailable.
    """
    if not _check_paw():
        return "unclear"
    try:
        fn = _get_contradiction_fn()
        inp = f"CLAIM_A: {claim_a} CLAIM_B: {claim_b}"
        result = _normalize_contradiction_verdict(fn(inp))
        return result if result in VALID_CONTRADICTION_VERDICTS else "unclear"
    except Exception:
        log.exception("PAW contradiction detection failed")
        return "unclear"


def decompose_query(query: str) -> list[str] | None:
    """Decompose a chemistry question into sub-topic search queries using PAW.

    Returns a list of keyword-rich search strings, or None if PAW is
    unavailable or produces low-quality output (looping, too few results).
    Callers should fall back to LLM decomposition when None is returned.
    """
    if not _check_paw():
        return None
    try:
        fn = _get_query_decomposer_fn()
        raw = fn(query).strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",")]
        seen: set[str] = set()
        unique: list[str] = []
        for p in parts:
            p_lower = p.lower().strip()
            if not p_lower or len(p_lower) < 5:
                continue
            if p_lower in seen:
                continue
            seen.add(p_lower)
            unique.append(p.strip())

        if len(unique) < 3:
            return None

        max_token = max(
            (p.split()[0].lower() for p in unique if p.split()),
            key=lambda t: sum(1 for p in unique if t in p.lower()),
            default="",
        )
        if max_token and sum(1 for p in unique if max_token in p.lower()) > 3:
            return None

        return unique[:5]
    except Exception:
        log.exception("PAW query decomposition failed")
        return None


def expand_query(query: str) -> list[str]:
    """Expand a chemistry search query into related terms using PAW.

    Returns a list of expansion terms (synonyms, abbreviations, specific
    examples).  Falls back to an empty list if PAW is unavailable.
    """
    if not _check_paw():
        return []
    try:
        fn = _get_query_expander_fn()
        raw = fn(query).strip()
        if not raw:
            return []
        terms = [t.strip() for t in raw.split(",")]
        # Filter out terms that are just the query repeated or empty
        q_lower = query.lower().strip()
        cleaned = []
        seen: set[str] = set()
        for t in terms:
            t_lower = t.lower().strip()
            if not t_lower or t_lower == q_lower or len(t_lower) < 2:
                continue
            if t_lower in seen:
                continue
            seen.add(t_lower)
            cleaned.append(t.strip())
        return cleaned[:20]
    except Exception:
        log.exception("PAW query expansion failed")
        return []
