#!/usr/bin/env python3
"""Build the `by_data` view from existing structured claim fields.

The `by_data` view surfaces the specific tables/numbers that scientists
need but are usually buried in paper figures and SI tables. It groups
data-bearing claims into a 2-level taxonomy:

    L1: <data_category>     e.g. electrochemical, optical, mechanical
    L2: <measurement_name>  e.g. ionic_conductivity, band_gap, bet_surface_area

A claim is eligible if it has BOTH a non-empty `property_category` and
a non-empty `property_name` (this is the structured signal we already
get for free from extraction). For non-property claims we fall back to a
claim_type-derived category when explicit fields are missing but the
claim has a `value` field.

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


# Canonical L1 buckets. Map raw `property_category` values (lowercased,
# trimmed) to one of these buckets. Anything not listed falls into 'other'.
CATEGORY_MAP = {
    # Direct hits / no-op
    "physical":            "physical",
    "chemical":            "chemical",
    "electrochemical":     "electrochemical",
    "spectroscopic":       "spectroscopic",
    "biological":          "biological",
    "optical":             "optical",
    "thermal":             "thermal",
    "mechanical":          "mechanical",
    "electrical":          "electrical",
    "electronic":          "electronic",
    "magnetic":            "magnetic",
    "analytical":          "analytical",
    "computational":       "computational",
    "kinetic":             "kinetic",
    "photochemical":       "photochemical",
    "biochemical":         "biochemical",
    "environmental":       "environmental",
    # Synonyms / merges
    "structure":           "structural",
    "structural":          "structural",
    "energy":              "energetic",
    "energetic":           "energetic",
    "energetics":          "energetic",
    "thermodynamic":       "energetic",
    "thermochemical":      "energetic",
    "theoretical":         "computational",
    "mathematical":        "computational",
    "dynamic":             "kinetic",
    "kinetics":            "kinetic",
    "economic":            "economic",
}

# When property_category is missing, fall back from claim_type.
CLAIM_TYPE_TO_CATEGORY = {
    "computational_result": "computational",
    "measurement":          "experimental",
    "data_point":           "experimental",
    "parameter":            "experimental",
    "performance":          "performance",
    "experimental_result":  "experimental",
    "equation":             "mathematical_model",
}

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


def _normalize_category(raw, claim_type: str | None) -> str | None:
    raw_str = _coerce_str(raw)
    if raw_str:
        key = raw_str.strip().lower()
        if key in CATEGORY_MAP:
            return CATEGORY_MAP[key]
        # Best-effort: collapse common compound categories (split on any
        # non-letter and try the first token).
        head = re.split(r"[^a-z]+", key, maxsplit=1)[0]
        if head and head in CATEGORY_MAP:
            return CATEGORY_MAP[head]
    if claim_type and claim_type in CLAIM_TYPE_TO_CATEGORY:
        return CLAIM_TYPE_TO_CATEGORY[claim_type]
    return None


_PAREN_RE = re.compile(r"\([^)]*\)")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")


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


def _normalize_measurement_name(raw) -> str | None:
    """Normalize a property_name to a stable tree-path segment.

    Lowercase, strip parenthetical text, collapse non-alphanumeric runs
    to underscore, drop leading/trailing underscores. Returns None if
    the result is empty or trivially short.
    """
    raw_str = _coerce_str(raw)
    if not raw_str:
        return None
    s = raw_str.strip().lower()
    s = _PAREN_RE.sub(" ", s)
    s = _NONWORD_RE.sub("_", s).strip("_")
    if not s or len(s) < 2:
        return None
    # Cap segment length so weird LLM outputs don't blow up the tree.
    if len(s) > 60:
        s = s[:60].rstrip("_")
    return s


def _display_name(raw, normalized: str) -> str:
    raw_str = _coerce_str(raw)
    if raw_str and len(raw_str) <= 80:
        return raw_str.strip()
    return normalized.replace("_", " ").title()


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
    """Compute the [L1, L2] path + display name for L2.

    Returns (path, l2_display_name) on success, or None if the claim is
    not eligible for by_data.
    """
    claim_type = d.get("claim_type") or ""
    if claim_type not in ALWAYS_INSPECT_TYPES and not _has_value_signal(d):
        return None

    raw_cat = d.get("property_category")
    raw_name = d.get("property_name")

    cat = _normalize_category(raw_cat, claim_type)
    if not cat:
        return None

    name = _normalize_measurement_name(raw_name)
    if not name:
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
        name = (_normalize_measurement_name(subj)
                or _normalize_measurement_name(claim_type))
        if not name:
            return None
        return [cat, name], _display_name(subj, name)

    return [cat, name], _display_name(raw_name, name)


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
    pair_counts: Counter = Counter()
    pair_display: dict[tuple[str, str], str] = {}
    pair_claim_ids: dict[tuple[str, str], list[str]] = {}
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

        pending.append((
            json.dumps(vp_col, separators=(",", ":")),
            json.dumps(d, separators=(",", ":"), ensure_ascii=False),
            cid,
        ))
        cat_counts[path[0]] += 1
        pair_counts[(path[0], path[1])] += 1
        pair_display[(path[0], path[1])] = display
        # Record up to MAX_LEAF_IDS claim_ids per leaf for tree_nodes.claim_ids
        ids = pair_claim_ids.setdefault((path[0], path[1]), [])
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
          f"L2 leaves: {len(pair_counts):,}")

    print()
    print("=== Pass 2: building tree_nodes / views ===")
    _rebuild_tree_nodes(conn, cat_counts, pair_counts,
                        pair_display, pair_claim_ids)
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


def _build_top_children_for_l1(
    cat: str,
    pair_counts: Counter,
    pair_display: dict,
) -> list[dict]:
    """Pick the top-N L2 measurements (by claim_count) under L1 ``cat``.

    Stored inside the L1 row's ``data`` JSON as ``top_children`` so the API
    can serve a subcategory grid in one DB read instead of fetching tens of
    thousands of L2 metadata rows on every page load.
    """
    top: list[dict] = []
    for (c, name), count in pair_counts.most_common():
        if c != cat:
            continue
        display = pair_display.get((c, name), name.replace("_", " ").title())
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
    pair_counts: Counter,
    pair_display: dict,
    pair_claim_ids: dict,
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
        children = [name for (c, name), _ in pair_counts.most_common()
                    if c == cat]
        display = CATEGORY_DISPLAY.get(cat, cat.title())
        top_children = _build_top_children_for_l1(cat, pair_counts, pair_display)
        cur.execute(
            "INSERT INTO tree_nodes (view_id,path,name,level,claim_count,"
            "children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            (VIEW_ID, cat, display, 1, count,
             json.dumps(children), json.dumps([]),
             json.dumps({"name": display, "view_id": VIEW_ID,
                         "claim_count": count,
                         "top_children": top_children})),
        )

    # L2 nodes.
    rows = []
    for (cat, name), count in pair_counts.most_common():
        path = f"{cat}/{name}"
        display = pair_display.get((cat, name), name.replace("_", " ").title())
        leaf_ids = pair_claim_ids.get((cat, name), [])
        rows.append((
            VIEW_ID, path, display, 2, count,
            json.dumps([]), json.dumps(leaf_ids),
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
          f"{len(pair_counts):,} L2  =  "
          f"{1 + len(cat_counts) + len(pair_counts):,} rows")


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
        "max_depth": 2,
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
    pair_counts: Counter = Counter()
    pair_display: dict[tuple[str, str], str] = {}
    pair_claim_ids: dict[tuple[str, str], list[str]] = {}
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
        if not path or len(path) < 2:
            continue
        cat, name = path[0], path[1]
        cat_counts[cat] += 1
        pair_counts[(cat, name)] += 1
        if (cat, name) not in pair_display:
            try:
                d = json.loads(row["data"])
                raw_name = d.get("property_name")
                pair_display[(cat, name)] = _display_name(raw_name, name)
            except Exception:
                pair_display[(cat, name)] = name.replace("_", " ").title()
        ids = pair_claim_ids.setdefault((cat, name), [])
        if len(ids) < args.max_leaf_ids:
            ids.append(row["claim_id"])
        if n % 100000 == 0:
            print(f"  scanned {n:,}  ({time.monotonic()-t0:.1f}s)", flush=True)
    print(f"  scanned {n:,} tagged claims in {time.monotonic()-t0:.1f}s")
    _rebuild_tree_nodes(conn, cat_counts, pair_counts,
                        pair_display, pair_claim_ids)
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
