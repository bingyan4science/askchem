"""
AskChem Python SDK — the structured knowledge API for chemistry.

Usage:
    from askchem import AskChem

    ct = AskChem()  # or AskChem(api_key="ac-...", base_url="https://askchem.org")

    results = ct.search("Suzuki coupling")
    for claim in results.claims:
        print(claim.claim_type, claim.verbatim_quote)
"""

from askchem.client import (
    AskChem,
    AskChemError, NotFoundError, RateLimitError, ValidationError, ServerError,
)
from askchem.models import (
    Claim, Source, TreeNode, View, SearchResult, BrowseResult,
    FeedItem, EvolutionTimeline,
)

__version__ = "0.3.0"
__all__ = [
    "AskChem",
    "AskChemError", "NotFoundError", "RateLimitError", "ValidationError", "ServerError",
    "Claim", "Source", "TreeNode", "View",
    "SearchResult", "BrowseResult", "FeedItem", "EvolutionTimeline",
]
