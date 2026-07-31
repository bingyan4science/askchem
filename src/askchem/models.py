"""
Core data models for AskChem.

Defines the fundamental types: Claim, Source, TreeNode, View.
All data is JSON-serializable for filesystem storage and API responses.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
import hashlib
import json


@dataclass
class Source:
    """A primary source (paper) from which claims are extracted."""
    doi: str
    title: str
    authors: list[str]
    year: int
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0
    open_access_url: str = ""
    semantic_scholar_id: str = ""
    fields_of_study: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        return self.doi.lower().replace("/", "_").replace(".", "-") if self.doi else self.semantic_scholar_id

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Source:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Claim:
    """
    An atomic unit of chemical knowledge extracted from a source.

    This is the fundamental building block of AskChem. Every claim:
    - Has a type (reaction, property, method, mechanism, comparison, etc.)
    - Is grounded in a specific source (paper) with a verbatim quote
    - Can appear in multiple views/hierarchies simultaneously
    - Has a confidence level and extraction metadata
    """
    claim_id: str
    claim_type: str = ""  # reaction, property, method, mechanism, comparison, scope_entry, computational_result, structure, hypothesis, experimental_design, limitation, future_direction, surprising_finding
    source_doi: str = ""
    source_paper_title: str = ""
    confidence: str = "high"  # high, medium, low
    location_in_paper: str = ""
    verbatim_quote: str = ""
    extraction_model: str = "gpt-5.4"
    extraction_version: str = "v2"
    extracted_at: str = ""

    # Reaction-specific fields
    reaction_type: str = ""
    reactants: list[dict] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    is_key_result: bool = False
    parent_reaction_id: Optional[str] = None

    # Property-specific fields
    subject: str = ""
    subject_smiles: str = ""
    property_name: str = ""
    property_category: str = ""
    value: str = ""
    unit: str = ""
    measurement_method: str = ""
    is_computed: bool = False

    # Mechanism-specific fields
    process_described: str = ""
    steps: list[str] = field(default_factory=list)
    key_intermediates: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    # Method-specific fields
    technique_name: str = ""
    what_it_achieves: str = ""
    key_innovation: str = ""
    limitations: str = ""

    # Comparison-specific fields
    compared_items: list[str] = field(default_factory=list)
    metric: str = ""
    comparison_result: str = ""

    # Hypothesis / limitation / future direction / surprising finding fields
    hypothesis_text: str = ""
    limitation_text: str = ""
    direction_text: str = ""
    finding_text: str = ""
    why_surprising: str = ""

    # Extraction provenance
    extraction_tier: str = ""  # "full_paper" or "abstract_only"

    # View assignments (populated during indexing)
    view_paths: dict[str, list[str]] = field(default_factory=dict)

    _REQUIRED_FIELDS = {'claim_id', 'claim_type', 'source_doi', 'source_paper_title',
                         'confidence', 'extraction_model', 'extraction_version'}

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v or v == 0 or v is False or k in self._REQUIRED_FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @staticmethod
    def generate_id(source_doi: str, claim_type: str, content_hash: str) -> str:
        raw = f"{source_doi}:{claim_type}:{content_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class TreeNode:
    """
    A node in the AskChem hierarchy.

    Each node represents a category at some level of the tree.
    Nodes are zoomable: at a high level they show summaries,
    zooming in reveals children and eventually individual claims.
    """
    node_id: str
    name: str
    path: list[str]  # Full path from root, e.g. ["reactions", "coupling", "cross_coupling"]
    view: str  # Which view this node belongs to
    description: str = ""
    level: int = 0

    # Children
    children: list[str] = field(default_factory=list)  # child node_ids

    # Claims at this node
    claim_ids: list[str] = field(default_factory=list)
    claim_count: int = 0  # Total claims in this subtree (including children)

    # Statistics
    source_count: int = 0  # Number of unique source papers
    year_range: tuple[int, int] = (0, 0)
    citation_stats: dict = field(default_factory=dict)

    # Frontier indicators
    is_sparse: bool = False  # Few claims relative to siblings
    has_contradictions: bool = False
    recent_surge: bool = False  # Significant increase in publications recently
    temporal_gap: bool = False  # No new papers in 5+ years

    # Auto-generated summary
    summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Keep path and children even when empty — they're structurally required
        keep_always = {"node_id", "name", "path", "view", "level", "children", "claim_ids", "claim_count"}
        return {k: v for k, v in d.items() if k in keep_always or v or v == 0 or v is False}

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        if "year_range" in filtered and isinstance(filtered["year_range"], list):
            filtered["year_range"] = tuple(filtered["year_range"])
        # Ensure required fields have defaults
        filtered.setdefault("path", [])
        filtered.setdefault("view", "")
        return cls(**filtered)


@dataclass
class View:
    """
    A hierarchical view over the claim store.

    Multiple views can exist over the same set of claims,
    each organizing them by a different principle.
    """
    view_id: str
    name: str
    description: str
    organizing_principle: str  # What dimension this view organizes by
    root_node_id: str = ""
    node_count: int = 0
    claim_count: int = 0
    max_depth: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> View:
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


EDGE_TYPES = {
    "supports",
    "assumes",
    "bounded_by",
    "interprets",
    "derives_from",
    "sub_step_of",
    "uses_method_of",
    "uses_assumption_of",
    "extends",
    "supersedes",
    "contradicts",
    "cites_as_evidence",
}


INTRA_PAPER_EDGE_TYPES = {
    "supports", "assumes", "bounded_by", "interprets", "derives_from", "sub_step_of",
}


CROSS_PAPER_EDGE_TYPES = {
    "uses_method_of", "uses_assumption_of", "extends", "supersedes",
    "contradicts", "cites_as_evidence",
}


@dataclass
class ClaimEdge:
    """A typed directed edge between two claims, or from a claim to an external DOI.

    Edges encode scientific dependencies that the flat claim store cannot:
    why we believe a claim (`supports`, `assumes`), what it rests on
    (`uses_method_of`, `derives_from`), what bounds it (`bounded_by`), and
    what disputes it (`contradicts`).

    Cross-paper edges may target a DOI (`to_doi`) when the cited paper's
    claims are not yet indexed; a later upgrade pass can rewrite them to
    `to_claim_id` once both endpoints are known.
    """

    from_claim_id: str
    edge_type: str
    to_claim_id: Optional[str] = None
    to_doi: Optional[str] = None
    confidence: str = "medium"
    evidence: str = ""
    extractor: str = ""
    extracted_at: str = ""

    def __post_init__(self):
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge_type: {self.edge_type!r}")
        if not self.to_claim_id and not self.to_doi:
            raise ValueError("ClaimEdge requires either to_claim_id or to_doi")
        if self.to_claim_id and self.to_doi:
            raise ValueError("ClaimEdge accepts to_claim_id OR to_doi, not both")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ClaimEdge:
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class PaperKnowledge:
    """
    Paper-level intellectual framing extracted from full text.

    Captures the high-level scientific narrative that abstracts omit:
    what the authors hypothesized, how they designed experiments,
    what they concluded, what limitations they acknowledged, and
    what they suggested for future work.
    """
    doi: str
    hypothesis: str = ""
    experimental_design: str = ""
    conclusions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    future_directions: list[str] = field(default_factory=list)
    surprising_findings: list[str] = field(default_factory=list)
    paper_type: str = ""  # research_article, review, communication, computational_study, methods_paper
    subfield: str = ""
    extraction_model: str = ""
    extraction_version: str = ""
    extracted_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PaperKnowledge:
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# Default views based on hierarchy discovery experiments
DEFAULT_VIEWS = [
    View(
        view_id="by_reaction_type",
        name="By Reaction Type",
        description="Organizes claims by the type of chemical transformation",
        organizing_principle="reaction_type",
    ),
    View(
        view_id="by_substance_class",
        name="Substance",
        description="Organizes claims by chemical substance class",
        organizing_principle="substance_class",
    ),
    View(
        view_id="by_application",
        name="By Application Domain",
        description="Organizes claims by practical application area",
        organizing_principle="application_domain",
    ),
    View(
        view_id="by_technique",
        name="By Technique/Method",
        description="Organizes claims by experimental or computational technique used",
        organizing_principle="technique",
    ),
    View(
        view_id="by_mechanism",
        name="By Phenomenon/Mechanism",
        description="Organizes claims by underlying physical/chemical mechanism",
        organizing_principle="mechanism",
    ),
    View(
        view_id="by_claim_type",
        name="By Claim Type",
        description="Organizes claims by epistemic role: what kind of scientific statement each claim represents",
        organizing_principle="claim_type",
    ),
    View(
        view_id="by_time_period",
        name="By Time Period",
        description="Organizes claims chronologically: decade > year > quarter",
        organizing_principle="publication_date",
    ),
]
