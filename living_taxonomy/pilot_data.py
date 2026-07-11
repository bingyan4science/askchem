"""Read-only sampling of pilot papers/claims from chemtree.db.

Opens the production DB with ``immutable=1`` so the pilot can NEVER write
to it. We pull a small focused set of papers (catalytic cross-coupling /
catalysis) and their reaction & substance claims as candidate leaves.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _db_path():
    """Canonical DB path via askchem.db (handles the chemtree.db -> askchem.db
    rename + env overrides); falls back to the legacy name if the package is
    unavailable."""
    try:
        from askchem import db
        return db.get_db_path()
    except Exception:
        c = _REPO_ROOT / "askchem.db"
        return c if c.exists() else _REPO_ROOT / "chemtree.db"


DB_PATH = _db_path()


def _connect():
    """Open the production DB strictly read-only (immutable)."""
    uri = f"file:{_db_path()}?immutable=1"
    return sqlite3.connect(uri, uri=True)


def sample_papers(n_papers=30, l1_filter=("coupling", "catalysis")):
    """Pick papers whose reaction claims sit under the focused-domain L1s.

    Returns a list of DOIs. We rank by number of reaction claims so each
    paper contributes several candidate leaves.
    """
    conn = _connect()
    like_clauses = " OR ".join(
        ["json_extract(view_paths, '$.by_reaction_type[0]') = ?" for _ in l1_filter]
    )
    sql = f"""
        SELECT source_doi, COUNT(*) AS n
        FROM claims
        WHERE claim_type = 'reaction' AND ({like_clauses})
        GROUP BY source_doi
        HAVING n >= 3
        ORDER BY n DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (*l1_filter, n_papers)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _s(v):
    """Coerce a claim field to a stripped string. Some extracted fields are
    list-valued (e.g. a claim with multiple subjects), which broke the old
    ``(d.get(x) or "").strip()`` pattern with AttributeError at scale."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v).strip()
    return str(v).strip()


def _reaction_leaf_text(d):
    """Human-readable descriptor for a reaction claim (the candidate leaf)."""
    rt = _s(d.get("reaction_type"))
    reactants = ", ".join(
        _s(r.get("name")) for r in (d.get("reactants") or [])
        if isinstance(r, dict) and r.get("name")
    )
    products = ", ".join(
        _s(p.get("name")) for p in (d.get("products") or [])
        if isinstance(p, dict) and p.get("name")
    )
    quote = _s(d.get("verbatim_quote"))
    parts = []
    if rt:
        parts.append(f"Reaction: {rt}.")
    if reactants:
        parts.append(f"Reactants: {reactants}.")
    if products:
        parts.append(f"Products: {products}.")
    if quote:
        parts.append(quote)
    return " ".join(parts).strip()


def _substance_leaf_text(d):
    """Descriptor for a substance leaf: a molecule/material identified by its
    subject / subject_smiles (property, structure, computational claims).

    Returns "" when the claim has no subject substance, so reaction claims
    without a subject are skipped (substances come from subject-bearing claims,
    not opaque reaction-product codes like "3aa")."""
    subj = _s(d.get("subject"))
    smiles = _s(d.get("subject_smiles"))
    if not subj and not smiles:
        return ""
    prop = _s(d.get("property_name"))
    quote = _s(d.get("verbatim_quote"))
    parts = [f"Substance: {subj or smiles}."]
    if smiles:
        parts.append(f"SMILES: {smiles}.")
    if prop:
        parts.append(f"Property: {prop}.")
    if quote:
        parts.append(quote)
    return " ".join(parts).strip()


def _mechanism_leaf_text(d):
    """Descriptor for a mechanistic-observation leaf (mechanism claims)."""
    proc = _s(d.get("process_described"))
    inter = [x for x in (d.get("key_intermediates") or []) if x]
    quote = _s(d.get("verbatim_quote"))
    if not proc and not quote:
        return ""
    parts = [f"Mechanism: {proc}."] if proc else []
    if inter:
        parts.append("Intermediates: " + ", ".join(str(x) for x in inter[:4]) + ".")
    if quote:
        parts.append(quote)
    return " ".join(parts).strip()


def _technique_leaf_text(d):
    """Descriptor for a measurement/technique leaf (method claims)."""
    tech = _s(d.get("technique_name"))
    ach = _s(d.get("what_it_achieves"))
    quote = _s(d.get("verbatim_quote"))
    if not tech and not ach and not quote:
        return ""
    parts = [f"Technique: {tech}."] if tech else []
    if ach:
        parts.append(ach + ".")
    if quote:
        parts.append(quote)
    return " ".join(parts).strip()


_LEAF_TEXT = {
    "by_reaction_type": _reaction_leaf_text,
    "by_substance_class": _substance_leaf_text,
    "by_mechanism": _mechanism_leaf_text,
    "by_technique": _technique_leaf_text,
}

# Which claim_type(s) provide candidate leaves for each view.
_LEAF_CLAIM_TYPE = {
    "by_reaction_type": ["reaction"],
    # substances live in subject-bearing claims, not reaction products
    "by_substance_class": ["property", "structure", "computational_result"],
    "by_mechanism": ["mechanism"],
    "by_technique": ["method"],
}


def load_leaves(view_id, dois, max_leaves=400, per_paper=None):
    """Load candidate leaves for a view from the given papers.

    Returns ``[{claim_id, doi, title, year, text, current_path}, ...]``.
    ``current_path`` is the paper's existing fixed-taxonomy path (for
    comparison against where the living tree places it).

    ``per_paper`` (when set) caps leaves *per DOI* so that at scale every paper
    gets a fair share instead of the first papers in arbitrary SQL order eating
    the whole ``max_leaves`` budget (the old single global cap starved late
    papers). ``max_leaves`` remains an overall safety ceiling.
    """
    if not dois:
        return []
    conn = _connect()
    claim_types = _LEAF_CLAIM_TYPE[view_id]
    text_fn = _LEAF_TEXT[view_id]
    placeholders = ",".join("?" for _ in dois)
    type_ph = ",".join("?" for _ in claim_types)
    sql = f"""
        SELECT c.claim_id, c.source_doi, c.data, c.view_paths,
               s.title, s.year
        FROM claims c
        LEFT JOIN sources s ON s.doi = c.source_doi
        WHERE c.claim_type IN ({type_ph}) AND c.source_doi IN ({placeholders})
    """
    leaves = []
    per_doi = {}
    for cid, doi, data_json, vp_json, title, year in conn.execute(
        sql, (*claim_types, *dois)
    ):
        if per_paper is not None and per_doi.get(doi, 0) >= per_paper:
            continue  # this paper already has its fair share
        try:
            d = json.loads(data_json)
        except (TypeError, json.JSONDecodeError):
            continue
        text = text_fn(d)
        if isinstance(text, (list, tuple)):        # some claim fields are list-valued
            text = " ".join(str(x) for x in text)
        elif text is not None and not isinstance(text, str):
            text = str(text)
        if not text:
            continue
        try:
            vp = json.loads(vp_json) if vp_json else {}
        except json.JSONDecodeError:
            vp = {}
        leaves.append({
            "claim_id": cid,
            "doi": doi,
            "title": title or "",
            "year": year or 0,
            "text": text,
            "current_path": vp.get(view_id, []),
        })
        per_doi[doi] = per_doi.get(doi, 0) + 1
        if len(leaves) >= max_leaves:
            break
    conn.close()
    return leaves
