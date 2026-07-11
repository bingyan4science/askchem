"""Compatibility alias for the benchmark implementation.

The AskChem-Bench implementation lives in ``benchmark_chemtree.py`` (the
pre-rebrand module name). Several scripts import it under the current
``benchmark_askchem`` name; this shim re-exports the public API so both
names resolve to the same code. ``import *`` skips underscore-prefixed
names, so the cache helper is re-exported explicitly.
"""
from benchmark_chemtree import *  # noqa: F401,F403
from benchmark_chemtree import (  # noqa: F401
    extract_dois,
    verify_dois_in_text,
    score_paper_relevance,
    compute_metrics,
    _save_paper_relevance_cache,
)
