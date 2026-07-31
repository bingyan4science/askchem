#!/usr/bin/env python3
"""Create a migrated AskChem DB from a reviewed full-path taxonomy registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from askchem.taxonomy import (  # noqa: E402
    ALL_CONTENT_VIEWS,
    CANONICAL_L1,
    CANONICAL_L2,
    CANONICAL_L3,
)
from askchem.taxonomy_semantics import (  # noqa: E402
    assert_formula_safe_alias,
    display_label,
)

SUBSTANCE_VIEW = "by_substance_class"
DEPRECATED_SUBSTANCE_VIEWS = {"by_composition", "by_material_form"}


def install_taxonomy_spec(path: Path) -> None:
    """Install an explicit candidate spec for migration-time validation."""
    global ALL_CONTENT_VIEWS, CANONICAL_L1, CANONICAL_L2, CANONICAL_L3
    payload = json.loads(path.read_text())
    CANONICAL_L1 = payload["canonical_l1"]
    CANONICAL_L2 = payload["canonical_l2"]
    CANONICAL_L3 = {
        view: {
            tuple(parent.split("/", 1)): values
            for parent, values in parents.items()
        }
        for view, parents in payload["canonical_l3"].items()
    }
    ALL_CONTENT_VIEWS = list(CANONICAL_L1)


def copy_database(source: Path, output: Path, filesystem_clone: bool) -> None:
    if filesystem_clone:
        if platform.system() != "Darwin":
            raise RuntimeError("--filesystem-clone currently requires macOS")
        subprocess.run(["cp", "-c", str(source), str(output)], check=True)
        return
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    output_conn = sqlite3.connect(output)
    try:
        source_conn.backup(output_conn, pages=16_384)
    finally:
        output_conn.close()
        source_conn.close()


def load_registry(
    path: Path,
) -> tuple[str, set[str], dict[tuple[str, tuple[str, ...]], list[str]]]:
    payload = json.loads(path.read_text())
    mapping = {}
    for record in payload["mappings"]:
        if record.get("status") != "approved":
            continue
        assert_formula_safe_alias(record["old_path"], record["new_path"])
        old = tuple(record["old_path"].split("/"))
        new = record["new_path"].split("/")
        mapping[(record["view"], old)] = new
    views = payload.get("views", "all")
    scoped_views = (
        set(ALL_CONTENT_VIEWS) if views == "all" else set(views)
    )
    return payload["taxonomy_version"], scoped_views, mapping


def validate_path(view: str, path: list[str]) -> list[str]:
    if not path or len(path) > 3:
        raise ValueError(f"{view}: invalid path depth {path!r}")
    l1 = path[0]
    if l1 not in CANONICAL_L1.get(view, []):
        raise ValueError(f"{view}: noncanonical L1 {l1!r}")
    if len(path) >= 2:
        l2 = path[1]
        if l2 not in CANONICAL_L2.get(view, {}).get(l1, []):
            raise ValueError(f"{view}: noncanonical L2 {l1}/{l2}")
        if len(path) == 3:
            allowed = CANONICAL_L3.get(view, {}).get((l1, l2))
            if allowed is None or path[2] not in allowed:
                raise ValueError(
                    f"{view}: noncanonical L3 {'/'.join(path)}"
                )
    return path


def canonicalize_path(view: str, path: list) -> list[str] | None:
    cleaned = [
        str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for value in path[:3]
        if str(value).strip()
    ]
    if not cleaned or cleaned[0] not in CANONICAL_L1.get(view, []):
        return None
    if len(cleaned) == 1:
        return cleaned
    l1 = cleaned[0]
    allowed_l2 = CANONICAL_L2.get(view, {}).get(l1, ["other"])
    if cleaned[1] not in allowed_l2:
        # Keep the canonical broad category and retain the rejected L2 in
        # migration metadata. This avoids turning every rare label into a
        # misleading "other" assignment.
        return [l1]
    allowed_l3 = CANONICAL_L3.get(view, {}).get((l1, cleaned[1]))
    if allowed_l3 is None:
        return cleaned[:2]
    if len(cleaned) < 3:
        return cleaned[:2]
    if cleaned[2] not in allowed_l3:
        # Preserve the reviewed parent and retain the original leaf in
        # taxonomy_migrations metadata instead of creating a catch-all spike.
        return cleaned[:2]
    return validate_path(view, cleaned[:3])


def mapped_path(
    view: str,
    path: list,
    mapping: dict[tuple[str, tuple[str, ...]], list[str]],
) -> list:
    """Resolve exact or longest-prefix mappings for legacy deep paths."""
    source = tuple(path)
    exact = mapping.get((view, source))
    if exact is not None:
        return list(exact)
    for depth in range(len(source) - 1, 0, -1):
        target = mapping.get((view, source[:depth]))
        if target is None:
            continue
        # Preserve as much descendant specificity as the three-level taxonomy
        # permits. Candidate validation will drop a leaf that is not canonical
        # under its new parent.
        return [*target, *source[depth:]][:3]
    return list(path)


def load_facet_mapping(path: Path | None) -> dict | None:
    """Load a clean-break mapping into the unified substance view.

    The existing ``old_view``/``new_views`` shape remains accepted for safe
    migrations, while the singular ``source_view``/``target_view`` spelling is
    supported by new unified mapping artifacts.
    """
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if not {"mappings", "taxonomy_version"} <= payload.keys():
        raise ValueError(f"invalid facet mapping artifact: {path}")
    old_view = payload.get("source_view", payload.get("old_view"))
    new_views = payload.get("new_views")
    target_view = payload.get("target_view", payload.get("new_view"))
    if new_views is None and target_view:
        new_views = [target_view]
    if not old_view or not isinstance(new_views, list) or not new_views:
        raise ValueError(f"invalid facet mapping artifact: {path}")
    if old_view in new_views:
        raise ValueError("facet mapping cannot retain its legacy view")
    exposed = DEPRECATED_SUBSTANCE_VIEWS.intersection(new_views)
    if exposed:
        raise ValueError(
            "facet mapping cannot expose deprecated views: "
            + ", ".join(sorted(exposed))
        )
    if old_view == "by_composition" and new_views != [SUBSTANCE_VIEW]:
        raise ValueError(
            "by_composition must map only to by_substance_class"
        )
    payload["old_view"] = old_view
    payload["new_views"] = new_views
    if len(new_views) == 1:
        payload["target_view"] = new_views[0]
    removed_views = set(payload.get("removed_views", []))
    if old_view in DEPRECATED_SUBSTANCE_VIEWS:
        removed_views.add(old_view)
    removed_views.add("by_material_form")
    payload["removed_views"] = sorted(removed_views)
    policy = payload.get("policy") or {}
    metadata_key = payload.get("material_form_metadata_key")
    if not metadata_key and (
        payload.get("preserve_material_form_metadata")
        or policy.get("preserve_material_form_metadata")
        or policy.get("material_form_is_metadata")
    ):
        metadata_key = "material_form"
    payload["material_form_metadata_key"] = metadata_key
    return payload


def mapped_facets(path: list, artifact: dict) -> dict[str, list[str] | None]:
    """Resolve an exact or longest-prefix legacy path into each new facet."""
    source = tuple(path)
    record = None
    matched_depth = 0
    for depth in range(len(source), 0, -1):
        record = artifact["mappings"].get("/".join(source[:depth]))
        if record is not None:
            matched_depth = depth
            break
    if record is None:
        return {view: None for view in artifact["new_views"]}
    suffix = list(source[matched_depth:])
    result = {}
    for view in artifact["new_views"]:
        if isinstance(record, str):
            target = record
        else:
            target = record.get(view)
            if target is None and view == artifact.get("target_view"):
                target = record.get("path")
        if target is None:
            result[view] = None
            continue
        # Descendants are preserved only when they fit the three-level target.
        result[view] = [*target.split("/"), *suffix][:3]
    return result


def migrate_claims(
    conn: sqlite3.Connection,
    taxonomy_version: str,
    scoped_views: set[str],
    mapping: dict[tuple[str, tuple[str, ...]], list[str]],
    split_assignments: dict[str, dict],
    batch_size: int,
    facet_mapping: dict | None = None,
) -> dict:
    cursor = conn.execute(
        "SELECT claim_id, view_paths, data FROM claims WHERE view_paths IS NOT NULL"
    )
    pending = []
    stats = Counter()
    changed_by_view = Counter()
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for claim_id, view_paths_raw, data_raw in rows:
            stats["claims_scanned"] += 1
            try:
                view_paths = json.loads(view_paths_raw)
                data = json.loads(data_raw)
            except (json.JSONDecodeError, TypeError):
                stats["invalid_json"] += 1
                continue
            changed = False
            previous = {}
            material_form = view_paths.get("by_material_form")
            metadata_key = (
                facet_mapping.get("material_form_metadata_key")
                if facet_mapping else None
            )
            if (
                metadata_key
                and isinstance(material_form, list)
                and material_form
                and material_form[0] != "other"
            ):
                if data.get(metadata_key) != material_form:
                    data[metadata_key] = material_form
                    stats["material_form_metadata_stored"] += 1
                    changed = True

            removed_views = set(DEPRECATED_SUBSTANCE_VIEWS)
            if facet_mapping:
                removed_views.update(facet_mapping["removed_views"])
            if facet_mapping:
                old_view = facet_mapping["old_view"]
                old_path = view_paths.get(old_view)
                if isinstance(old_path, list):
                    merged_old_path = mapped_path(old_view, old_path, mapping)
                    replacements = mapped_facets(
                        merged_old_path, facet_mapping,
                    )
                    previous[old_view] = old_path
                    view_paths.pop(old_view, None)
                    changed_by_view[old_view] += 1
                    for new_view, replacement in replacements.items():
                        canonical = (
                            canonicalize_path(new_view, replacement)
                            if replacement else None
                        )
                        if canonical:
                            view_paths[new_view] = canonical
                            changed_by_view[new_view] += 1
                        stats[
                            f"facet_{new_view}_"
                            f"{'mapped' if canonical else 'unmapped'}"
                        ] += 1
                    stats["facet_claims_migrated"] += 1
                    changed = True
            for removed_view in removed_views:
                removed_path = view_paths.pop(removed_view, None)
                if removed_path is None:
                    continue
                previous.setdefault(removed_view, removed_path)
                changed_by_view[removed_view] += 1
                stats["deprecated_view_paths_removed"] += 1
                changed = True
            for view in scoped_views:
                if view in removed_views:
                    continue
                path = view_paths.get(view)
                if not isinstance(path, list):
                    continue
                split = split_assignments.get(f"{view}:{claim_id}")
                if (
                    split
                    and split.get("view") == view
                    and path[:3] == split.get("old_path", "").split("/")
                ):
                    replacement = split["new_path"].split("/")
                    stats["split_assignments_applied"] += 1
                else:
                    replacement = mapped_path(view, path, mapping)
                replacement = canonicalize_path(view, replacement)
                if replacement is None:
                    previous[view] = path
                    view_paths.pop(view, None)
                    changed_by_view[view] += 1
                    changed = True
                    continue
                if replacement == path:
                    continue
                previous[view] = path
                view_paths[view] = list(replacement)
                changed_by_view[view] += 1
                changed = True
            mirror_changed = data.get("view_paths") != view_paths
            if not changed and not mirror_changed:
                continue
            if previous:
                # Full old→new provenance lives in the versioned registry and
                # split-assignment artifact. Embedding previous paths in every
                # claim adds several gigabytes and duplicates that evidence.
                data["taxonomy_version"] = taxonomy_version
            data["view_paths"] = view_paths
            pending.append((
                json.dumps(view_paths, ensure_ascii=False, sort_keys=True),
                json.dumps(data, ensure_ascii=False, sort_keys=True),
                claim_id,
            ))
            stats["claims_changed"] += 1
        if pending:
            conn.executemany(
                "UPDATE claims SET view_paths = ?, data = ? WHERE claim_id = ?",
                pending,
            )
            conn.commit()
            pending.clear()
    return {
        **stats,
        "changed_by_view": dict(changed_by_view),
    }


def rebuild_tree_staged(
    conn: sqlite3.Connection,
    scoped_views: set[str],
    removed_views: set[str] | None = None,
) -> dict:
    conn.execute("DROP TABLE IF EXISTS tree_nodes_v2_staging")
    conn.execute(
        "CREATE TABLE tree_nodes_v2_staging AS "
        "SELECT * FROM tree_nodes WHERE 0"
    )
    nodes_by_view: dict[str, dict] = {
        view: {} for view in scoped_views
    }
    cursor = conn.execute(
        "SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL"
    )
    while True:
        rows = cursor.fetchmany(20_000)
        if not rows:
            break
        for claim_id, raw in rows:
            try:
                view_paths = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for view in scoped_views:
                path = view_paths.get(view)
                if not isinstance(path, list) or not path:
                    continue
                nodes = nodes_by_view[view]
                for depth in range(1, len(path) + 1):
                    key = "/".join(path[:depth])
                    node = nodes.setdefault(
                        key, {"own": 0, "children": set(), "sample": []},
                    )
                    if depth == len(path):
                        node["own"] += 1
                        if len(node["sample"]) < 100:
                            node["sample"].append(claim_id)
                    if depth > 1:
                        parent = "/".join(path[:depth - 1])
                        nodes.setdefault(
                            parent,
                            {"own": 0, "children": set(), "sample": []},
                        )["children"].add(key)

    totals_by_view = {}
    inserts = []
    for view, nodes in nodes_by_view.items():
        totals = {}

        def total(path: str) -> int:
            if path in totals:
                return totals[path]
            value = nodes[path]["own"] + sum(
                total(child) for child in nodes[path]["children"]
            )
            totals[path] = value
            return value

        for path, node in nodes.items():
            parts = path.split("/")
            count = total(path)
            children = sorted(child.rsplit("/", 1)[-1]
                              for child in node["children"])
            name = display_label(parts[-1])
            data = {
                "node_id": f"{view}_{path}",
                "name": name,
                "claim_count": count,
            }
            inserts.append((
                view, path, name, len(parts), count,
                json.dumps(children), json.dumps(node["sample"]),
                json.dumps(data),
            ))
        totals_by_view[view] = len(nodes)
    conn.executemany(
        "INSERT INTO tree_nodes_v2_staging "
        "(view_id,path,name,level,claim_count,children,claim_ids,data) "
        "VALUES (?,?,?,?,?,?,?,?)",
        inserts,
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    replaced_views = scoped_views | (removed_views or set())
    conn.execute(
        "DELETE FROM tree_nodes WHERE view_id IN "
        f"({','.join('?' for _ in replaced_views)})",
        tuple(sorted(replaced_views)),
    )
    conn.execute(
        "INSERT INTO tree_nodes SELECT * FROM tree_nodes_v2_staging"
    )
    conn.execute("DROP TABLE tree_nodes_v2_staging")
    conn.commit()
    return totals_by_view


def rebuild_view_map_staged(conn: sqlite3.Connection, batch_size: int) -> int:
    conn.execute("DROP TABLE IF EXISTS claim_view_map_v2_staging")
    conn.execute(
        "CREATE TABLE claim_view_map_v2_staging "
        "(claim_id TEXT NOT NULL, view_id TEXT NOT NULL, path TEXT NOT NULL)"
    )
    cursor = conn.execute(
        "SELECT claim_id, view_paths FROM claims WHERE view_paths IS NOT NULL"
    )
    batch = []
    inserted = 0
    while True:
        rows = cursor.fetchmany(20_000)
        if not rows:
            break
        for claim_id, raw in rows:
            try:
                view_paths = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for view, path in view_paths.items():
                if not isinstance(path, list):
                    continue
                for depth in range(1, len(path) + 1):
                    batch.append((claim_id, view, "/".join(path[:depth])))
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO claim_view_map_v2_staging VALUES (?,?,?)",
                    batch,
                )
                inserted += len(batch)
                batch.clear()
                conn.commit()
    if batch:
        conn.executemany(
            "INSERT INTO claim_view_map_v2_staging VALUES (?,?,?)", batch,
        )
        inserted += len(batch)
        conn.commit()
    conn.execute(
        "CREATE INDEX idx_cvm_v2_staging_path "
        "ON claim_view_map_v2_staging(view_id, path)"
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM claim_view_map")
    conn.execute(
        "INSERT INTO claim_view_map "
        "SELECT claim_id, view_id, path FROM claim_view_map_v2_staging"
    )
    conn.execute("DROP TABLE claim_view_map_v2_staging")
    conn.commit()
    return inserted


def migrate_view_metadata(
    conn: sqlite3.Connection, facet_mapping: dict | None,
) -> None:
    """Publish one substance view and remove deprecated view definitions."""
    definitions = {
        SUBSTANCE_VIEW: (
            "Substance",
            "Organizes claims by chemical substance class",
            "substance_class",
        ),
    }
    conn.executemany(
        "DELETE FROM views WHERE view_id = ?",
        [(view,) for view in sorted(DEPRECATED_SUBSTANCE_VIEWS)],
    )
    published_views = {SUBSTANCE_VIEW}
    if facet_mapping:
        published_views.update(facet_mapping["new_views"])
    for view in published_views:
        definition = definitions.get(view)
        if definition is None:
            name = display_label(view.removeprefix("by_"))
            description = f"Organizes claims by {name.lower()}"
            principle = view.removeprefix("by_")
        else:
            name, description, principle = definition
        node_count = conn.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE view_id = ?", (view,),
        ).fetchone()[0]
        claim_count = conn.execute(
            "SELECT COALESCE(SUM(claim_count), 0) FROM tree_nodes "
            "WHERE view_id = ? AND level = 1",
            (view,),
        ).fetchone()[0]
        data = {
            "view_id": view,
            "name": name,
            "description": description,
            "organizing_principle": principle,
            "root_node_id": "",
            "node_count": node_count,
            "claim_count": claim_count,
            "max_depth": 3,
            "created_at": "",
            "updated_at": "",
        }
        conn.execute(
            "INSERT INTO views(view_id,name,description,data) VALUES (?,?,?,?) "
            "ON CONFLICT(view_id) DO UPDATE SET "
            "name=excluded.name,description=excluded.description,data=excluded.data",
            (view, name, description, json.dumps(data, sort_keys=True)),
        )
    conn.commit()


def validate_database(
    conn: sqlite3.Connection, scoped_views: set[str],
) -> dict:
    result = {
        "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
        "claims": conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
        "tree_nodes": conn.execute(
            "SELECT COUNT(*) FROM tree_nodes"
        ).fetchone()[0],
        "claim_view_map": conn.execute(
            "SELECT COUNT(*) FROM claim_view_map"
        ).fetchone()[0],
    }
    mismatches = 0
    invalid_depth = 0
    invalid_canonical = 0
    deprecated_paths = 0
    cursor = conn.execute(
        "SELECT view_paths, data FROM claims WHERE view_paths IS NOT NULL"
    )
    while True:
        rows = cursor.fetchmany(20_000)
        if not rows:
            break
        for view_paths_raw, data_raw in rows:
            try:
                view_paths = json.loads(view_paths_raw)
                data = json.loads(data_raw)
            except (json.JSONDecodeError, TypeError):
                mismatches += 1
                continue
            if data.get("view_paths") != view_paths:
                mismatches += 1
            deprecated_paths += len(
                DEPRECATED_SUBSTANCE_VIEWS.intersection(view_paths)
            )
            invalid_depth += sum(
                1 for view, path in view_paths.items()
                if view in scoped_views
                and isinstance(path, list)
                and len(path) > 3
            )
            for view, path in view_paths.items():
                if view not in scoped_views or not isinstance(path, list):
                    continue
                try:
                    validate_path(view, path)
                except ValueError:
                    invalid_canonical += 1
    result["view_path_mismatches"] = mismatches
    result["paths_over_depth_3"] = invalid_depth
    result["invalid_canonical_paths"] = invalid_canonical
    result["deprecated_view_paths"] = deprecated_paths
    invalid_tree_paths = 0
    for view, path in conn.execute(
        "SELECT view_id,path FROM tree_nodes "
        f"WHERE view_id IN ({','.join('?' for _ in scoped_views)})",
        tuple(sorted(scoped_views)),
    ):
        try:
            validate_path(view, path.split("/"))
        except ValueError:
            invalid_tree_paths += 1
    result["invalid_tree_paths"] = invalid_tree_paths
    result["deprecated_tree_nodes"] = conn.execute(
        "SELECT COUNT(*) FROM tree_nodes "
        f"WHERE view_id IN ({','.join('?' for _ in DEPRECATED_SUBSTANCE_VIEWS)})",
        tuple(sorted(DEPRECATED_SUBSTANCE_VIEWS)),
    ).fetchone()[0]
    result["deprecated_view_metadata"] = conn.execute(
        "SELECT COUNT(*) FROM views "
        f"WHERE view_id IN ({','.join('?' for _ in DEPRECATED_SUBSTANCE_VIEWS)})",
        tuple(sorted(DEPRECATED_SUBSTANCE_VIEWS)),
    ).fetchone()[0]
    result["deprecated_view_map_rows"] = conn.execute(
        "SELECT COUNT(*) FROM claim_view_map "
        f"WHERE view_id IN ({','.join('?' for _ in DEPRECATED_SUBSTANCE_VIEWS)})",
        tuple(sorted(DEPRECATED_SUBSTANCE_VIEWS)),
    ).fetchone()[0]
    if (
        result["quick_check"] != "ok"
        or mismatches
        or invalid_depth
        or invalid_canonical
        or invalid_tree_paths
        or deprecated_paths
        or result["deprecated_tree_nodes"]
        or result["deprecated_view_metadata"]
        or result["deprecated_view_map_rows"]
    ):
        raise RuntimeError(f"candidate DB validation failed: {result}")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--taxonomy-spec",
        type=Path,
        help="Candidate canonical JSON used to validate migrated paths",
    )
    parser.add_argument(
        "--split-assignments",
        type=Path,
        help="Gemini-reviewed per-claim assignments for approved split nodes",
    )
    parser.add_argument(
        "--facet-mapping", "--substance-mapping",
        dest="facet_mapping",
        type=Path,
        help=(
            "Clean-break mapping artifact; --substance-mapping is the "
            "preferred name for the unified substance contract"
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--filesystem-clone",
        action="store_true",
        help="Use macOS clonefile semantics; source WAL must already be empty",
    )
    parser.add_argument("--skip-derived", action="store_true")
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-view-map", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()

    if args.taxonomy_spec:
        install_taxonomy_spec(args.taxonomy_spec)
    if args.output_db.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_db)
        args.output_db.unlink()
    taxonomy_version, scoped_views, mapping = load_registry(args.registry)
    split_assignments = (
        json.loads(args.split_assignments.read_text())
        if args.split_assignments else {}
    )
    facet_mapping = load_facet_mapping(args.facet_mapping)
    scoped_views.difference_update(DEPRECATED_SUBSTANCE_VIEWS)
    if facet_mapping:
        scoped_views.update(facet_mapping["new_views"])
    started = time.monotonic()
    copy_database(args.source_db, args.output_db, args.filesystem_clone)
    conn = sqlite3.connect(args.output_db)
    try:
        # The candidate is disposable until all gates pass. WAL can otherwise
        # grow by many gigabytes while rewriting claims and derived tables.
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        migration = migrate_claims(
            conn, taxonomy_version, scoped_views, mapping,
            split_assignments, args.batch_size, facet_mapping,
        )
        derived = {}
        if not args.skip_derived and not args.skip_tree:
            derived["tree_nodes_by_view"] = rebuild_tree_staged(
                conn, scoped_views, DEPRECATED_SUBSTANCE_VIEWS,
            )
        if not args.skip_derived and not args.skip_view_map:
            derived["claim_view_map_rows"] = rebuild_view_map_staged(
                conn, args.batch_size,
            )
        if not args.skip_derived and not args.skip_tree:
            total_nodes = conn.execute(
                "SELECT COUNT(*) FROM tree_nodes"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES ('total_nodes', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(total_nodes),),
            )
            conn.commit()
            derived["total_nodes"] = total_nodes
        migrate_view_metadata(conn, facet_mapping)
        validation = validate_database(conn, scoped_views)
        conn.execute("PRAGMA journal_mode=DELETE")
    finally:
        conn.close()
    report = {
        "taxonomy_version": taxonomy_version,
        "source_db": str(args.source_db.resolve()),
        "output_db": str(args.output_db.resolve()),
        "registry": str(args.registry.resolve()),
        "approved_mappings": len(mapping),
        "facet_mapping": (
            str(args.facet_mapping.resolve()) if args.facet_mapping else None
        ),
        "scoped_views": sorted(scoped_views),
        "migration": migration,
        "derived": derived,
        "validation": validation,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_sha256": file_sha256(args.output_db),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
