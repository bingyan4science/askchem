"""
AskChem Store: Filesystem-based storage for claims, nodes, and views.

The store is organized as a filesystem-like hierarchy:
  chemtree_index/
    claims/                     # All claims, keyed by claim_id
      {claim_id}.json
    sources/                    # All source papers
      {source_id}.json
    views/
      by_reaction_type/
        _view.json              # View metadata
        _root.json              # Root node
        coupling/
          _node.json            # Node metadata
          cross_coupling/
            _node.json
            ...
      by_substance_class/
        ...
    metadata.json               # Global index metadata
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import Claim, Source, TreeNode, View, DEFAULT_VIEWS


class AskChemStore:
    """Filesystem-based store for the AskChem index."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.claims_dir = self.root / "claims"
        self.sources_dir = self.root / "sources"
        self.views_dir = self.root / "views"
        self._metadata_path = self.root / "metadata.json"

    def initialize(self):
        """Create the directory structure and default views."""
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.views_dir.mkdir(parents=True, exist_ok=True)

        for view in DEFAULT_VIEWS:
            view_dir = self.views_dir / view.view_id
            view_dir.mkdir(exist_ok=True)
            view.created_at = datetime.now().isoformat()
            view.updated_at = view.created_at
            self._write_json(view_dir / "_view.json", view.to_dict())

            root_node = TreeNode(
                node_id=f"{view.view_id}_root",
                name=view.name,
                path=[],
                view=view.view_id,
                description=view.description,
                level=0,
            )
            view.root_node_id = root_node.node_id
            self._write_json(view_dir / "_root.json", root_node.to_dict())
            self._write_json(view_dir / "_view.json", view.to_dict())

        metadata = {
            "name": "AskChem Index",
            "version": "0.1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "claim_count": 0,
            "source_count": 0,
            "views": [v.view_id for v in DEFAULT_VIEWS],
        }
        self._write_json(self._metadata_path, metadata)
        return metadata

    # --- Claim operations ---

    def add_claim(self, claim: Claim) -> str:
        """Add a claim to the store. Returns the claim_id."""
        path = self.claims_dir / f"{claim.claim_id}.json"
        self._write_json(path, claim.to_dict())
        self._update_metadata("claim_count", delta=1)
        return claim.claim_id

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID."""
        path = self.claims_dir / f"{claim_id}.json"
        if not path.exists():
            return None
        data = self._read_json(path)
        return Claim.from_dict(data)

    def list_claims(self, limit: int = 100, offset: int = 0) -> list[Claim]:
        """List claims with pagination."""
        files = sorted(self.claims_dir.glob("*.json"))
        claims = []
        for f in files[offset:offset + limit]:
            data = self._read_json(f)
            claims.append(Claim.from_dict(data))
        return claims

    def count_claims(self) -> int:
        return len(list(self.claims_dir.glob("*.json")))

    # --- Source operations ---

    def add_source(self, source: Source) -> str:
        """Add a source paper to the store."""
        path = self.sources_dir / f"{source.source_id}.json"
        self._write_json(path, source.to_dict())
        self._update_metadata("source_count", delta=1)
        return source.source_id

    def get_source(self, source_id: str) -> Optional[Source]:
        path = self.sources_dir / f"{source_id}.json"
        if not path.exists():
            return None
        return Source.from_dict(self._read_json(path))

    def get_source_by_doi(self, doi: str) -> Optional[Source]:
        source_id = doi.lower().replace("/", "_").replace(".", "-")
        return self.get_source(source_id)

    def get_claims_for_source(self, doi: str) -> list[Claim]:
        """Get all claims extracted from a given source paper."""
        claims = []
        for f in self.claims_dir.glob("*.json"):
            data = self._read_json(f)
            if data.get("source_doi", "").lower() == doi.lower():
                claims.append(Claim.from_dict(data))
        return claims

    # --- View / Tree operations ---

    def get_view(self, view_id: str) -> Optional[View]:
        path = self.views_dir / view_id / "_view.json"
        if not path.exists():
            return None
        return View.from_dict(self._read_json(path))

    def list_views(self) -> list[View]:
        views = []
        for d in sorted(self.views_dir.iterdir()):
            if d.is_dir():
                view_path = d / "_view.json"
                if view_path.exists():
                    views.append(View.from_dict(self._read_json(view_path)))
        return views

    def get_node(self, view_id: str, path: list[str]) -> Optional[TreeNode]:
        """Get a node by its path within a view."""
        node_dir = self.views_dir / view_id
        for segment in path:
            node_dir = node_dir / segment
        node_file = node_dir / "_node.json"
        if not node_file.exists():
            if (node_dir.parent / "_root.json").exists() and not path:
                return TreeNode.from_dict(self._read_json(node_dir / "_root.json"))
            return None
        return TreeNode.from_dict(self._read_json(node_file))

    def get_node_with_children(self, view_id: str, path: list[str], depth: int = 1) -> dict:
        """
        Get a node and its children up to a given depth.
        Returns a dict suitable for API response (zoomable).
        """
        node = self.get_node(view_id, path)
        if node is None:
            # Try root
            root_path = self.views_dir / view_id / "_root.json"
            if root_path.exists() and not path:
                node = TreeNode.from_dict(self._read_json(root_path))
            else:
                return None

        result = node.to_dict()

        if depth > 0:
            node_dir = self.views_dir / view_id
            for segment in path:
                node_dir = node_dir / segment

            children_data = []
            if node_dir.exists():
                for child_dir in sorted(node_dir.iterdir()):
                    if child_dir.is_dir():
                        child_node_file = child_dir / "_node.json"
                        if child_node_file.exists():
                            child_data = self._read_json(child_node_file)
                            if depth > 1:
                                child_path = path + [child_dir.name]
                                child_data = self.get_node_with_children(
                                    view_id, child_path, depth - 1
                                )
                            children_data.append(child_data)

            result["children_data"] = children_data

        return result

    def add_node(self, view_id: str, path: list[str], node: TreeNode):
        """Add or update a node in a view's hierarchy."""
        node_dir = self.views_dir / view_id
        for segment in path:
            node_dir = node_dir / segment
        node_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(node_dir / "_node.json", node.to_dict())

        # Update parent's children list
        if path:
            parent_path = path[:-1]
            parent_dir = self.views_dir / view_id
            for segment in parent_path:
                parent_dir = parent_dir / segment
            parent_file = parent_dir / "_node.json" if parent_path else parent_dir / "_root.json"
            if parent_file.exists():
                parent_data = self._read_json(parent_file)
                children = parent_data.get("children", [])
                if node.node_id not in children:
                    children.append(node.node_id)
                    parent_data["children"] = children
                    self._write_json(parent_file, parent_data)

    def assign_claim_to_node(self, view_id: str, path: list[str], claim_id: str):
        """Assign a claim to a node in a view."""
        node_dir = self.views_dir / view_id
        for segment in path:
            node_dir = node_dir / segment
        node_file = node_dir / "_node.json"
        if node_file.exists():
            data = self._read_json(node_file)
            claim_ids = data.get("claim_ids", [])
            if claim_id not in claim_ids:
                claim_ids.append(claim_id)
                data["claim_ids"] = claim_ids
                data["claim_count"] = data.get("claim_count", 0) + 1
                self._write_json(node_file, data)

        # Update claim's view_paths
        claim = self.get_claim(claim_id)
        if claim:
            claim.view_paths[view_id] = path
            self.add_claim(claim)

    # --- Search ---

    def search_claims(self, query: str, view: str = None, claim_type: str = None, limit: int = 50) -> list[Claim]:
        """Simple text search across claims."""
        results = []
        query_lower = query.lower()
        for f in self.claims_dir.glob("*.json"):
            data = self._read_json(f)
            if claim_type and data.get("claim_type") != claim_type:
                continue
            if view and view not in data.get("view_paths", {}):
                continue
            text = json.dumps(data).lower()
            if query_lower in text:
                results.append(Claim.from_dict(data))
                if len(results) >= limit:
                    break
        return results

    # --- Frontier detection ---

    def get_frontier_indicators(self, view_id: str, path: list[str]) -> dict:
        """Analyze a node for frontier indicators."""
        node = self.get_node(view_id, path)
        if not node:
            return {}

        claims = [self.get_claim(cid) for cid in node.claim_ids if self.get_claim(cid)]

        years = [int(c.extracted_at[:4]) for c in claims if c.extracted_at]
        sources = list(set(c.source_doi for c in claims if c.source_doi))

        indicators = {
            "claim_count": len(claims),
            "source_count": len(sources),
            "is_sparse": len(claims) < 3,
            "year_range": (min(years), max(years)) if years else None,
        }

        # Check for contradictions (claims with conflicting outcomes)
        if node.claim_count > 1:
            reaction_claims = [c for c in claims if c.claim_type == "reaction"]
            if len(reaction_claims) > 1:
                yields = []
                for c in reaction_claims:
                    y = c.outcomes.get("yield_percent")
                    if y is not None:
                        yields.append(float(y))
                if yields and max(yields) - min(yields) > 30:
                    indicators["has_contradictions"] = True
                    indicators["contradiction_detail"] = f"Yield range: {min(yields):.0f}%-{max(yields):.0f}%"

        return indicators

    # --- Metadata ---

    def get_metadata(self) -> dict:
        if self._metadata_path.exists():
            return self._read_json(self._metadata_path)
        return {}

    def _update_metadata(self, key: str, delta: int = 0, value=None):
        meta = self.get_metadata()
        if delta:
            meta[key] = meta.get(key, 0) + delta
        elif value is not None:
            meta[key] = value
        meta["updated_at"] = datetime.now().isoformat()
        self._write_json(self._metadata_path, meta)

    # --- File I/O ---

    @staticmethod
    def _write_json(path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path) as f:
            content = f.read()
        if not content.strip():
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
