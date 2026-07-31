#!/usr/bin/env python3
"""Build the `by_data` view from existing structured claim fields.

The `by_data` view surfaces the specific tables/numbers that scientists
need but are usually buried in paper figures and SI tables. It groups
data-bearing claims into a 3-level taxonomy:

    L1: <data_category>     e.g. electrochemical, optical, mechanical
L2: <canonical_metric>   e.g. ionic_conductivity, band_gap, bet_surface_area
L3: <context>            e.g. general, product_co, reaction_co2_reduction

A claim is eligible when it has a measurement-bearing claim type and a
non-empty `property_name`. Canonical categories are derived from the metric,
not trusted from noisy extraction output. Measurement-like claims may use
their subject as a conservative fallback when they carry an explicit value.

Usage:
    python scripts/build_data_view.py dryrun [--limit N]
    python scripts/build_data_view.py apply  [--limit N]
    python scripts/build_data_view.py rebuild-tree

The `apply` step writes by_data to the JSON `view_paths` column AND to
the JSON `data` column on each eligible claim (keeping the two in sync
with how every other view is stored). It then rebuilds tree_nodes for
`by_data` and inserts a row into the `views` table.

Idempotent: re-running on already-tagged claims overwrites by_data only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from askchem.db import get_db_path  # noqa: E402
from askchem.measurement_canonical import (  # noqa: E402
    METRIC_DISPLAY,
    canonicalize_measurement,
)


VIEW_ID = "by_data"
VIEW_NAME = "By Data / Measurements"
VIEW_DESCRIPTION = (
    "Specific numerical measurements, parameters, and data points — "
    "grouped by category (electrochemical, optical, mechanical, etc.) "
    "and measurement name (band gap, ionic conductivity, etc.). "
    "Built to surface the kind of concrete numbers and table values "
    "that are usually hard to retrieve from the literature."
)
# How many top-by-claim_count L2 leaves to pre-cache into each L1 node's
# data JSON. The L1 categories like "physical" hold ~55k leaves each;
# precomputing avoids a slow batched IN-fetch on every page load.
L1_TOP_CHILDREN_N = 200


# Display labels for L1 buckets in the UI / tree_nodes.name.
CATEGORY_DISPLAY = {
    "physical":           "Physical",
    "chemical":           "Chemical",
    "electrochemical":    "Electrochemical",
    "spectroscopic":      "Spectroscopic",
    "biological":         "Biological",
    "optical":            "Optical",
    "thermal":            "Thermal",
    "mechanical":         "Mechanical",
    "electrical":         "Electrical",
    "electronic":         "Electronic",
    "magnetic":           "Magnetic",
    "structural":         "Structural",
    "analytical":         "Analytical",
    "computational":      "Computational",
    "kinetic":            "Kinetic",
    "photochemical":      "Photochemical",
    "biochemical":        "Biochemical",
    "environmental":      "Environmental",
    "energetic":          "Energetic / Thermodynamic",
    "experimental":       "Experimental Measurements",
    "performance":        "Performance Metrics",
    "mathematical_model": "Equations / Models",
    "economic":           "Economic",
    "uncategorized":      "Uncategorized / Review Required",
    "other":              "Other",
}


# Eligible claim_types for by_data. The first set is "always inspect"
# (these usually carry numeric structured fields). Anything else only
# makes it in if it explicitly has a `value` field.
ALWAYS_INSPECT_TYPES = {
    "property", "measurement", "computational_result",
    "data_point", "parameter", "performance",
    "experimental_result", "equation",
}


_NUMERIC_RE = re.compile(r"\d")


def _coerce_str(raw) -> str | None:
    """Coerce LLM outputs that are sometimes list/dict back to a string."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, list):
        for item in raw:
            s = _coerce_str(item)
            if s:
                return s
        return None
    if isinstance(raw, dict):
        for k in ("name", "value", "text", "label"):
            if k in raw:
                return _coerce_str(raw[k])
    return None


def _has_value_signal(d: dict) -> bool:
    """Does this claim carry an explicit numeric signal?"""
    for k in ("value", "values", "magnitude", "quantity",
              "numeric_value", "measured_value"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, list) and v:
            return True
    quote = d.get("verbatim_quote") or ""
    if quote and _NUMERIC_RE.search(quote):
        return True
    return False


def derive_by_data_path(d: dict) -> tuple[list[str], str | None] | None:
    """Compute the canonical [L1, L2, L3] path and leaf display name.

    Returns (path, leaf_display_name) on success, or None if the claim is
    not eligible for by_data.
    """
    claim_type = d.get("claim_type") or ""
    if claim_type not in ALWAYS_INSPECT_TYPES and not _has_value_signal(d):
        return None

    raw_name = d.get("property_name")
    if not _coerce_str(raw_name):
        # Stricter fallback: only allow it for claim_types that almost
        # always carry a real measurement (and require an explicit
        # `value` field — narrative quotes alone don't count). We use
        # `subject` as a proxy L2 when present so the bucket stays
        # informative instead of becoming a junk drawer.
        if claim_type not in ("measurement", "data_point",
                              "experimental_result", "parameter",
                              "performance", "equation"):
            return None
        if not d.get("value"):
            return None
        subj = _coerce_str(d.get("subject"))
        if not subj:
            return None
        signature = canonicalize_measurement(
            subj, d.get("property_category") or claim_type
        )
    else:
        signature = canonicalize_measurement(d)

    if signature is None:
        return None
    # Unsupported open-vocabulary labels are retained on the claim and can be
    # audited, but they must not recreate the 197k-node public string zoo.
    if signature.quarantined:
        return None
    return list(signature.path), signature.display


# ─────────────────────────── DB plumbing ────────────────────────────

def _open_rw() -> sqlite3.Connection:
    path = get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA cache_size=-65536")  # 64MB
    return conn


def _open_ro() -> sqlite3.Connection:
    path = get_db_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA mmap_size=268435456")
    return conn


def _eligible_types_clause() -> tuple[str, list]:
    types = sorted(ALWAYS_INSPECT_TYPES)
    ph = ",".join("?" * len(types))
    return f"claim_type IN ({ph})", list(types)


def stream_eligible(conn: sqlite3.Connection, limit: int | None = None):
    """Yield (claim_id, claim_type, view_paths_json, data_json) rows.

    Filters at SQL layer to claim_types that *might* be eligible. Final
    eligibility (e.g. requires property_category) is checked in Python.
    """
    where_sql, params = _eligible_types_clause()
    sql = (
        f"SELECT claim_id, claim_type, view_paths, data "
        f"FROM claims WHERE {where_sql}"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur = conn.execute(sql, params)
    for row in cur:
        yield row


def cmd_dryrun(args):
    conn = _open_ro()
    n_seen = 0
    n_eligible = 0
    cat_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    name_examples: dict[tuple[str, str], list[str]] = {}
    t0 = time.monotonic()
    for row in stream_eligible(conn, args.limit):
        n_seen += 1
        try:
            d = json.loads(row["data"])
        except Exception:
            continue
        out = derive_by_data_path(d)
        if not out:
            continue
        n_eligible += 1
        path, display = out
        cat_counts[path[0]] += 1
        pair_counts[(path[0], path[1])] += 1
        if (path[0], path[1]) not in name_examples:
            v = d.get("value") or ""
            quote = (d.get("verbatim_quote") or "")[:120]
            name_examples[(path[0], path[1])] = [
                f"{display}  | value={v}  | quote={quote}"
            ]
        if n_seen % 100000 == 0:
            print(
                f"  scanned {n_seen:,}  eligible {n_eligible:,}  "
                f"({n_eligible/n_seen*100:.1f}%)  "
                f"{time.monotonic()-t0:.1f}s",
                flush=True,
            )

    print()
    print(f"=== Dry-run summary ({time.monotonic()-t0:.1f}s) ===")
    print(f"  Scanned {n_seen:,} claims (claim_type in eligible set)")
    print(f"  Eligible for by_data: {n_eligible:,} "
          f"({(n_eligible/max(n_seen,1))*100:.1f}%)")
    print()
    print(f"=== Top L1 categories (of {len(cat_counts)} total) ===")
    for cat, c in cat_counts.most_common(20):
        label = CATEGORY_DISPLAY.get(cat, cat)
        print(f"  {c:>9,d}  {cat:25s}  →  {label}")
    print()
    print(f"=== Top 30 (L1, L2) pairs (of {len(pair_counts):,} total) ===")
    for (cat, name), c in pair_counts.most_common(30):
        ex = name_examples.get((cat, name), [""])[0]
        print(f"  {c:>6,d}  {cat}/{name}")
        print(f"           {ex[:200]}")


# ─────────────────────────── apply ────────────────────────────

def cmd_apply(args):
    conn = _open_rw()
    cur = conn.cursor()
    n_seen = 0
    n_updated = 0
    n_skipped = 0
    cat_counts: Counter = Counter()
    metric_counts: Counter = Counter()
    leaf_counts: Counter = Counter()
    leaf_display: dict[tuple[str, str, str], str] = {}
    leaf_claim_ids: dict[tuple[str, str, str], list[str]] = {}
    t0 = time.monotonic()

    BATCH = 5000
    pending: list[tuple[str, str, str]] = []  # (view_paths, data, claim_id)

    where_sql, params = _eligible_types_clause()
    sql = (
        f"SELECT claim_id, claim_type, view_paths, data "
        f"FROM claims WHERE {where_sql}"
    )
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    for row in cur.execute(sql, params):
        n_seen += 1
        cid = row["claim_id"]
        try:
            d = json.loads(row["data"])
        except Exception:
            n_skipped += 1
            continue
        out = derive_by_data_path(d)
        if not out:
            # Remove placements created by the legacy open-vocabulary Data
            # tree while preserving the raw label for later registry review.
            try:
                vp_col = (
                    json.loads(row["view_paths"]) if row["view_paths"] else {}
                )
            except Exception:
                vp_col = {}
            if not isinstance(vp_col, dict):
                vp_col = {}
            d_vp = d.get("view_paths") or {}
            if not isinstance(d_vp, dict):
                d_vp = {}
            had_assignment = (
                vp_col.pop(VIEW_ID, None) is not None
                or d_vp.pop(VIEW_ID, None) is not None
            )
            if had_assignment:
                d["view_paths"] = d_vp
                d["measurement_quarantine"] = {
                    "property_name": _coerce_str(d.get("property_name")),
                    "reason": "unsupported_canonical_metric",
                }
                d.pop("measurement_signature", None)
                pending.append((
                    json.dumps(vp_col, separators=(",", ":")),
                    json.dumps(d, separators=(",", ":"), ensure_ascii=False),
                    cid,
                ))
                n_updated += 1
                if len(pending) >= BATCH:
                    _flush(conn, pending)
                    pending = []
            continue
        path, display = out

        # Update both view_paths column and data['view_paths'] (kept in
        # sync with how all other views are stored).
        try:
            vp_col = json.loads(row["view_paths"]) if row["view_paths"] else {}
        except Exception:
            vp_col = {}
        if not isinstance(vp_col, dict):
            vp_col = {}
        vp_col["by_data"] = path

        d_vp = d.get("view_paths") or {}
        if not isinstance(d_vp, dict):
            d_vp = {}
        d_vp["by_data"] = path
        d["view_paths"] = d_vp
        signature = canonicalize_measurement(d)
        if signature is not None:
            d["measurement_signature"] = {
                "id": signature.stable_id,
                "category": signature.category,
                "metric": signature.metric,
                "context": signature.context,
                "display": signature.display,
                "quarantined": signature.quarantined,
            }

        pending.append((
            json.dumps(vp_col, separators=(",", ":")),
            json.dumps(d, separators=(",", ":"), ensure_ascii=False),
            cid,
        ))
        cat_counts[path[0]] += 1
        metric_counts[(path[0], path[1])] += 1
        leaf_key = (path[0], path[1], path[2])
        leaf_counts[leaf_key] += 1
        leaf_display[leaf_key] = display
        # Record up to MAX_LEAF_IDS claim_ids per leaf for tree_nodes.claim_ids
        ids = leaf_claim_ids.setdefault(leaf_key, [])
        if len(ids) < args.max_leaf_ids:
            ids.append(cid)
        n_updated += 1

        if len(pending) >= BATCH:
            _flush(conn, pending)
            pending = []
            print(
                f"  updated {n_updated:,} (scanned {n_seen:,}, "
                f"{n_updated/max(time.monotonic()-t0,0.001):.0f}/s)",
                flush=True,
            )

    if pending:
        _flush(conn, pending)
    conn.commit()

    print()
    print(f"=== Pass 1 done ({time.monotonic()-t0:.1f}s) ===")
    print(f"  Scanned: {n_seen:,}    Updated: {n_updated:,}    "
          f"Skipped(parse): {n_skipped:,}")
    print(f"  L1 categories: {len(cat_counts)}    "
          f"L2 metrics: {len(metric_counts):,}    "
          f"L3 leaves: {len(leaf_counts):,}")

    print()
    print("=== Pass 2: building tree_nodes / views ===")
    _rebuild_tree_nodes(
        conn, cat_counts, metric_counts, leaf_counts,
        leaf_display, leaf_claim_ids,
    )
    _rebuild_claim_view_map(conn)
    _upsert_view_row(conn)
    conn.commit()
    print("Done.")


def _flush(conn: sqlite3.Connection,
           pending: list[tuple[str, str, str]]) -> None:
    cur = conn.cursor()
    cur.executemany(
        "UPDATE claims SET view_paths=?, data=? WHERE claim_id=?",
        pending,
    )
    conn.commit()


def _rebuild_claim_view_map(conn: sqlite3.Connection) -> None:
    """Replace denormalized lookup rows for the canonical Data view."""
    conn.execute("DELETE FROM claim_view_map WHERE view_id = ?", [VIEW_ID])
    rows: list[tuple[str, str, str]] = []
    inserted = 0
    cursor = conn.execute(
        "SELECT claim_id, view_paths FROM claims "
        "WHERE view_paths LIKE '%\"by_data\"%'"
    )
    for claim_id, raw_paths in cursor:
        try:
            path = json.loads(raw_paths or "{}").get(VIEW_ID)
        except (json.JSONDecodeError, AttributeError):
            continue
        if not isinstance(path, list) or not path:
            continue
        for depth in range(1, len(path) + 1):
            rows.append((claim_id, VIEW_ID, "/".join(path[:depth])))
        if len(rows) >= 5000:
            conn.executemany(
                "INSERT OR IGNORE INTO claim_view_map "
                "(claim_id, view_id, path) VALUES (?, ?, ?)",
                rows,
            )
            inserted += len(rows)
            rows = []
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO claim_view_map "
            "(claim_id, view_id, path) VALUES (?, ?, ?)",
            rows,
        )
        inserted += len(rows)
    print(f"  claim_view_map: rebuilt {inserted:,} Data rows")


def _build_top_children_for_l1(
    cat: str,
    metric_counts: Counter,
) -> list[dict]:
    """Pick the top-N canonical L2 metrics under L1 ``cat``.

    Stored inside the L1 row's ``data`` JSON as ``top_children`` so the API
    can serve a subcategory grid in one DB read instead of fetching tens of
    thousands of L2 metadata rows on every page load.
    """
    top: list[dict] = []
    for (c, name), count in metric_counts.most_common():
        if c != cat:
            continue
        display = METRIC_DISPLAY.get(name, name.replace("_", " ").title())
        top.append({
            "name": display,
            "path": f"{c}/{name}",
            "claim_count": count,
        })
        if len(top) >= L1_TOP_CHILDREN_N:
            break
    return top


def _rebuild_tree_nodes(
    conn: sqlite3.Connection,
    cat_counts: Counter,
    metric_counts: Counter,
    leaf_counts: Counter,
    leaf_display: dict,
    leaf_claim_ids: dict,
) -> None:
    cur = conn.cursor()
    # Wipe any existing by_data tree.
    cur.execute("DELETE FROM tree_nodes WHERE view_id = ?", [VIEW_ID])

    # Root node (level 0, path = '').
    total = sum(cat_counts.values())
    top_level = [c for c, _ in cat_counts.most_common()]
    root_top_children = [
        {
            "name": CATEGORY_DISPLAY.get(cat, cat.title()),
            "path": cat,
            "claim_count": cnt,
        }
        for cat, cnt in cat_counts.most_common()
    ]
    cur.execute(
        "INSERT INTO tree_nodes (view_id,path,name,level,claim_count,"
        "children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
        (VIEW_ID, "", VIEW_NAME, 0, total,
         json.dumps(top_level), json.dumps([]),
         json.dumps({"name": VIEW_NAME, "view_id": VIEW_ID,
                     "claim_count": total,
                     "top_children": root_top_children})),
    )

    # L1 nodes — pre-compute top_children so the API never has to
    # materialize the full child list (some L1 holds 50k+ leaves).
    for cat, count in cat_counts.most_common():
        children = [name for (c, name), _ in metric_counts.most_common()
                    if c == cat]
        display = CATEGORY_DISPLAY.get(cat, cat.title())
        top_children = _build_top_children_for_l1(cat, metric_counts)
        cur.execute(
            "INSERT INTO tree_nodes (view_id,path,name,level,claim_count,"
            "children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            (VIEW_ID, cat, display, 1, count,
             json.dumps(children), json.dumps([]),
             json.dumps({"name": display, "view_id": VIEW_ID,
                         "claim_count": count,
                         "top_children": top_children})),
        )

    # L2 canonical metric nodes.
    rows = []
    for (cat, name), count in metric_counts.most_common():
        path = f"{cat}/{name}"
        display = METRIC_DISPLAY.get(name, name.replace("_", " ").title())
        children = [
            context for (c, metric, context), _ in leaf_counts.most_common()
            if c == cat and metric == name
        ]
        top_children = [
            {
                "name": leaf_display.get(
                    (cat, name, context), context.replace("_", " ").title()
                ),
                "path": f"{path}/{context}",
                "claim_count": leaf_counts[(cat, name, context)],
            }
            for context in children[:L1_TOP_CHILDREN_N]
        ]
        rows.append((
            VIEW_ID, path, display, 2, count,
            json.dumps(children), json.dumps([]),
            json.dumps({"name": display, "view_id": VIEW_ID,
                        "claim_count": count,
                        "top_children": top_children}),
        ))
        if len(rows) >= 5000:
            cur.executemany(
                "INSERT INTO tree_nodes (view_id,path,name,level,"
                "claim_count,children,claim_ids,data) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            rows = []
    if rows:
        cur.executemany(
            "INSERT INTO tree_nodes (view_id,path,name,level,"
            "claim_count,children,claim_ids,data) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )

    # L3 context leaves own the claim IDs.
    rows = []
    for key, count in leaf_counts.most_common():
        cat, name, context = key
        path = f"{cat}/{name}/{context}"
        display = leaf_display.get(key, context.replace("_", " ").title())
        rows.append((
            VIEW_ID, path, display, 3, count,
            json.dumps([]), json.dumps(leaf_claim_ids.get(key, [])),
            json.dumps({"name": display, "view_id": VIEW_ID,
                        "claim_count": count}),
        ))
        if len(rows) >= 5000:
            cur.executemany(
                "INSERT INTO tree_nodes (view_id,path,name,level,"
                "claim_count,children,claim_ids,data) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            rows = []
    if rows:
        cur.executemany(
            "INSERT INTO tree_nodes (view_id,path,name,level,"
            "claim_count,children,claim_ids,data) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    conn.commit()
    print(f"  tree_nodes: 1 root + {len(cat_counts)} L1 + "
          f"{len(metric_counts):,} L2 + {len(leaf_counts):,} L3 = "
          f"{1 + len(cat_counts) + len(metric_counts) + len(leaf_counts):,} rows")


def _upsert_view_row(conn: sqlite3.Connection,
                     node_count: int | None = None,
                     claim_count: int | None = None) -> None:
    cur = conn.cursor()
    if node_count is None:
        node_count = cur.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE view_id = ?", [VIEW_ID]
        ).fetchone()[0]
    if claim_count is None:
        row = cur.execute(
            "SELECT claim_count FROM tree_nodes "
            "WHERE view_id = ? AND path = ''",
            [VIEW_ID],
        ).fetchone()
        claim_count = row[0] if row else 0
    # Match the JSON schema other views use (organizing_principle,
    # node_count, claim_count, max_depth, …) so the /api/views response is
    # self-consistent across views.
    data = {
        "view_id": VIEW_ID,
        "name": VIEW_NAME,
        "description": VIEW_DESCRIPTION,
        "organizing_principle": "measurement_category",
        "root_node_id": "",
        "node_count": node_count,
        "claim_count": claim_count,
        "max_depth": 3,
        "created_at": "",
        "updated_at": "",
    }
    cur.execute(
        "INSERT OR REPLACE INTO views (view_id, name, description, data) "
        "VALUES (?,?,?,?)",
        (VIEW_ID, VIEW_NAME, VIEW_DESCRIPTION, json.dumps(data)),
    )
    conn.commit()
    print(f"  views: upserted '{VIEW_ID}'  "
          f"(node_count={node_count:,}, claim_count={claim_count:,})")


def cmd_backfill_l1(args):
    """Cheap in-place backfill that updates only L1 (and root) data JSON.

    Reads the existing tree_nodes for ``by_data`` and rewrites each L1 row's
    ``data`` JSON to include a ``top_children`` array (top-N L2 leaves by
    claim_count). Also refreshes the views metadata. Doesn't touch L2 rows
    or claims, so it runs in seconds and is safe to re-run.
    """
    conn = _open_rw()
    cur = conn.cursor()

    cat_counts: Counter = Counter()
    l2_by_l1: dict[str, list[tuple[str, int, str]]] = {}
    t0 = time.monotonic()

    # Pull every L1 (level=1) and L2 (level=2) row in one pass.
    rows = cur.execute(
        "SELECT path, name, level, claim_count "
        "FROM tree_nodes WHERE view_id = ? AND level IN (1, 2)",
        [VIEW_ID],
    ).fetchall()
    for r in rows:
        path = r["path"]
        if r["level"] == 1:
            cat_counts[path] = r["claim_count"] or 0
            l2_by_l1.setdefault(path, [])
        elif r["level"] == 2:
            parts = path.split("/", 1)
            if len(parts) != 2:
                continue
            cat = parts[0]
            l2_by_l1.setdefault(cat, []).append((
                parts[1],
                r["claim_count"] or 0,
                r["name"] or parts[1].replace("_", " "),
            ))

    n_updated = 0
    for cat, leaves in l2_by_l1.items():
        leaves.sort(key=lambda t: t[1], reverse=True)
        top_children = [
            {
                "name": display,
                "path": f"{cat}/{seg}",
                "claim_count": cnt,
            }
            for seg, cnt, display in leaves[: args.max_top_children]
        ]
        display = CATEGORY_DISPLAY.get(cat, cat.title())
        new_data = {
            "name": display,
            "view_id": VIEW_ID,
            "claim_count": cat_counts.get(cat, 0),
            "top_children": top_children,
        }
        cur.execute(
            "UPDATE tree_nodes SET data = ? WHERE view_id = ? AND path = ?",
            [json.dumps(new_data), VIEW_ID, cat],
        )
        n_updated += 1

    # Root: top_children = the L1 categories themselves, ordered by count.
    root_top = [
        {
            "name": CATEGORY_DISPLAY.get(c, c.title()),
            "path": c,
            "claim_count": n,
        }
        for c, n in cat_counts.most_common()
    ]
    total = sum(cat_counts.values())
    cur.execute(
        "UPDATE tree_nodes SET data = ? WHERE view_id = ? AND path = ''",
        [
            json.dumps({
                "name": VIEW_NAME,
                "view_id": VIEW_ID,
                "claim_count": total,
                "top_children": root_top,
            }),
            VIEW_ID,
        ],
    )
    conn.commit()

    print(f"Backfilled {n_updated} L1 nodes + 1 root in "
          f"{time.monotonic() - t0:.1f}s")
    _upsert_view_row(conn, claim_count=total)
    conn.commit()


def cmd_rebuild_tree(args):
    """Re-derive tree_nodes purely from claims.view_paths (no claim writes).

    Useful if the claims table is already tagged but tree_nodes were lost.
    """
    conn = _open_rw()
    cat_counts: Counter = Counter()
    metric_counts: Counter = Counter()
    leaf_counts: Counter = Counter()
    leaf_display: dict[tuple[str, str, str], str] = {}
    leaf_claim_ids: dict[tuple[str, str, str], list[str]] = {}
    t0 = time.monotonic()
    n = 0
    for row in conn.execute(
        "SELECT claim_id, view_paths, data FROM claims "
        "WHERE view_paths LIKE '%\"by_data\"%'"
    ):
        n += 1
        try:
            vp = json.loads(row["view_paths"])
            path = vp.get("by_data")
        except Exception:
            continue
        if not path or len(path) < 3:
            continue
        cat, name, context = path[:3]
        key = (cat, name, context)
        cat_counts[cat] += 1
        metric_counts[(cat, name)] += 1
        leaf_counts[key] += 1
        if key not in leaf_display:
            try:
                d = json.loads(row["data"])
                signature = canonicalize_measurement(d)
                leaf_display[key] = (
                    signature.display if signature else
                    METRIC_DISPLAY.get(name, name.replace("_", " ").title())
                )
            except Exception:
                leaf_display[key] = METRIC_DISPLAY.get(
                    name, name.replace("_", " ").title()
                )
        ids = leaf_claim_ids.setdefault(key, [])
        if len(ids) < args.max_leaf_ids:
            ids.append(row["claim_id"])
        if n % 100000 == 0:
            print(f"  scanned {n:,}  ({time.monotonic()-t0:.1f}s)", flush=True)
    print(f"  scanned {n:,} tagged claims in {time.monotonic()-t0:.1f}s")
    _rebuild_tree_nodes(
        conn, cat_counts, metric_counts, leaf_counts,
        leaf_display, leaf_claim_ids,
    )
    _rebuild_claim_view_map(conn)
    _upsert_view_row(conn)
    conn.commit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dry = sub.add_parser("dryrun", help="Scan + report without writing.")
    p_dry.add_argument("--limit", type=int, default=None,
                       help="Cap rows scanned (default: all eligible).")
    p_dry.set_defaults(func=cmd_dryrun)

    p_apply = sub.add_parser("apply", help="Backfill view_paths + tree_nodes.")
    p_apply.add_argument("--limit", type=int, default=None,
                         help="Cap rows updated (default: all eligible).")
    p_apply.add_argument("--max-leaf-ids", type=int, default=5000,
                         help="Max claim_ids stored per L2 tree_node "
                              "(default 5000). The largest L2 in by_data "
                              "currently has ~2.6k claims so this gives "
                              "comfortable headroom.")
    p_apply.set_defaults(func=cmd_apply)

    p_tree = sub.add_parser("rebuild-tree",
                            help="Rebuild tree_nodes from already-tagged "
                                 "claims (no claim updates).")
    p_tree.add_argument("--max-leaf-ids", type=int, default=5000)
    p_tree.set_defaults(func=cmd_rebuild_tree)

    p_l1 = sub.add_parser(
        "backfill-l1",
        help="Update only the L1/root data JSON (top_children). Cheap, safe "
             "to re-run; doesn't touch claims or L2 rows.",
    )
    p_l1.add_argument("--max-top-children", type=int, default=L1_TOP_CHILDREN_N,
                      help=f"How many top-by-claim-count L2 entries to embed "
                           f"in each L1 row (default {L1_TOP_CHILDREN_N}).")
    p_l1.set_defaults(func=cmd_backfill_l1)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
