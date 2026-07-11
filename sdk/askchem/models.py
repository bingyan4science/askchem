"""
Pydantic response models for the AskChem SDK.
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field


class Claim(BaseModel):
    claim_id: str
    claim_type: str = ""
    source_doi: str = ""
    source_paper_title: str = ""
    confidence: str = "high"
    location_in_paper: str = ""
    verbatim_quote: str = ""
    extraction_model: str = ""
    extraction_version: str = ""

    reaction_type: str = ""
    subject: str = ""
    property_name: str = ""
    value: str = ""
    unit: str = ""
    technique_name: str = ""
    what_it_achieves: str = ""
    process_described: str = ""
    comparison_result: str = ""

    reactants: list[dict] = Field(default_factory=list)
    products: list[dict] = Field(default_factory=list)
    conditions: dict = Field(default_factory=dict)
    outcomes: dict = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    compared_items: list[str] = Field(default_factory=list)
    view_paths: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Source(BaseModel):
    doi: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0

    model_config = {"extra": "allow"}


class TreeNode(BaseModel):
    name: str = ""
    path: str = ""
    claim_count: int = 0
    children: list[TreeNode] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class View(BaseModel):
    view_id: str
    name: str = ""
    description: str = ""

    model_config = {"extra": "allow"}


class SearchResult(BaseModel):
    query: str = ""
    total: int = 0
    claims: list[Claim] = Field(default_factory=list)
    offset: int = 0
    limit: int = 50

    model_config = {"extra": "allow"}


class BrowseResult(BaseModel):
    view_id: str = ""
    path: list[str] = Field(default_factory=list)
    node: Optional[TreeNode] = None
    claims: list[Claim] = Field(default_factory=list)
    total_claims: int = 0

    model_config = {"extra": "allow"}


class SourceResult(BaseModel):
    doi: str = ""
    source: Optional[Source] = None
    claims: list[Claim] = Field(default_factory=list)
    count: int = 0

    model_config = {"extra": "allow"}


class StatsResult(BaseModel):
    total_claims: int = 0
    total_sources: int = 0
    total_views: int = 0
    total_nodes: int = 0
    claim_types: dict[str, int] = Field(default_factory=dict)
    year_distribution: dict[str, int] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class YearStats(BaseModel):
    claim_count: int = 0
    types: dict[str, int] = Field(default_factory=dict)
    is_surge: bool = False

    model_config = {"extra": "allow"}


class EvolutionTimeline(BaseModel):
    view_id: str = ""
    path: str = ""
    years: dict[int, YearStats] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class FeedItem(BaseModel):
    claim: Claim
    surprise_score: float = 0.0
    reason: str = ""

    model_config = {"extra": "allow"}
