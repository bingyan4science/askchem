"""Backward-compatible resolution of taxonomy-v2 public paths."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ALIASES_PATH = Path(__file__).with_name("taxonomy_path_aliases.json")


@lru_cache(maxsize=1)
def _load() -> tuple[str | None, dict[str, dict[str, str]]]:
    if not _ALIASES_PATH.exists():
        return None, {}
    payload = json.loads(_ALIASES_PATH.read_text())
    return payload.get("taxonomy_version"), payload.get("aliases", {})


def resolve_tree_path(view_id: str, path: str) -> tuple[str, bool]:
    """Return canonical path and whether an alias was followed."""
    _, aliases = _load()
    current = path.strip("/")
    original = current
    seen = set()
    view_aliases = aliases.get(view_id, {})
    while True:
        if current in seen:
            raise ValueError(f"taxonomy alias cycle: {view_id}/{current}")
        seen.add(current)
        if current in view_aliases:
            current = view_aliases[current]
            continue
        parts = current.split("/") if current else []
        resolved_prefix = False
        for depth in range(len(parts) - 1, 0, -1):
            prefix = "/".join(parts[:depth])
            target = view_aliases.get(prefix)
            if target is None:
                continue
            target_parts = target.split("/")
            suffix = parts[depth:] if len(target_parts) == depth else []
            current = "/".join([*target_parts, *suffix][:3])
            resolved_prefix = True
            break
        if resolved_prefix:
            continue
        break
    return current, current != original


def taxonomy_version() -> str | None:
    return _load()[0]
