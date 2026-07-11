"""Living-taxonomy query layer (served by /api/ltree/* in server.py).

Reads the taxonomy_nodes / taxonomy_edges / taxonomy_leaves tables (populated by
living_taxonomy/apply_to_db.py) and hydrates leaves from the claims table. Pure
read-only; reuses db.get_conn().
"""

from __future__ import annotations

import json
import os

from askchem import db

ROOT_ID = "__root__"


def _pretty(view_id: str) -> str:
    return view_id.replace("by_", "").replace("_", " ").title()


def list_views() -> list[dict]:
    with db.get_conn() as c:
        row = c.execute("SELECT value FROM taxonomy_meta WHERE key='views'").fetchone()
        views = json.loads(row[0]) if row else []
        out = []
        for v in views:
            n_claims = c.execute(
                "SELECT COUNT(*) FROM taxonomy_leaves WHERE view_id=?", (v,)).fetchone()[0]
            n_papers = c.execute(
                "SELECT COUNT(DISTINCT doi) FROM taxonomy_leaves WHERE view_id=?",
                (v,)).fetchone()[0]
            out.append({"view_id": v, "name": _pretty(v),
                        "n_papers": n_papers, "n_claims": n_claims,
                        "n_leaves": n_papers})   # n_leaves kept = papers (leaf = paper)
        ver = c.execute("SELECT value FROM taxonomy_meta WHERE key='version'").fetchone()
    return {"views": out, "version": ver[0] if ver else None}


def _child_counts(c, view_id, nid):
    """(child internal nodes, distinct papers) under a node. Leaf = paper."""
    nc = c.execute("SELECT COUNT(*) FROM taxonomy_edges WHERE view_id=? AND parent_id=?",
                   (view_id, nid)).fetchone()[0]
    npapers = c.execute("SELECT COUNT(DISTINCT doi) FROM taxonomy_leaves "
                        "WHERE view_id=? AND node_id=?", (view_id, nid)).fetchone()[0]
    return nc, npapers


def _node_children(c, view_id, nid, depth):
    kids = []
    rows = c.execute(
        "SELECT n.node_id,n.kind,n.name,n.proposed,n.definition,n.short_label,n.equation "
        "FROM taxonomy_edges e "
        "JOIN taxonomy_nodes n ON n.node_id=e.child_id "
        "WHERE e.view_id=? AND e.parent_id=? ORDER BY n.kind, n.name",
        (view_id, nid)).fetchall()
    for r in rows:
        nc, nl = _child_counts(c, view_id, r["node_id"])
        ch = {"node_id": r["node_id"], "kind": r["kind"], "name": r["name"],
              "proposed": bool(r["proposed"]), "definition": r["definition"] or "",
              "short_label": r["short_label"] or "", "equation": r["equation"] or "",
              "n_children": nc, "n_leaves": nl}
        if depth > 1 and nc:
            ch["children"] = _node_children(c, view_id, r["node_id"], depth - 1)
        kids.append(ch)
    return kids


def get_node(view_id: str, node_id: str, depth: int = 1) -> dict | None:
    with db.get_conn() as c:
        n = c.execute(
            "SELECT node_id,kind,name,definition,short_label,equation,proposed "
            "FROM taxonomy_nodes WHERE node_id=?", (node_id,)).fetchone()
        if n is None and node_id != ROOT_ID:
            return None
        kind = n["kind"] if n else "open_root"
        name = n["name"] if n else "(root)"
        children = _node_children(c, view_id, node_id, depth)
        _, nl = _child_counts(c, view_id, node_id)
        return {"view_id": view_id, "node_id": node_id, "kind": kind, "name": name,
                "definition": (n["definition"] if n else ""),
                "short_label": (n["short_label"] if n else "") or "",
                "equation": (n["equation"] if n else "") or "",
                "proposed": bool(n["proposed"]) if n else False,
                "n_leaves": nl, "children": children}


def get_path(view_id: str, node_id: str) -> list[dict]:
    """Ancestor chain (root..node) for breadcrumb / auto-expand."""
    out, cur, seen = [], node_id, set()
    with db.get_conn() as c:
        while cur and cur != ROOT_ID and cur not in seen:
            seen.add(cur)
            n = c.execute("SELECT name FROM taxonomy_nodes WHERE node_id=?",
                          (cur,)).fetchone()
            out.append({"node_id": cur, "name": n["name"] if n else cur})
            r = c.execute(
                "SELECT parent_id FROM taxonomy_edges WHERE view_id=? AND child_id=? "
                "LIMIT 1", (view_id, cur)).fetchone()
            cur = r["parent_id"] if r else None
    out.reverse()
    return out


def get_papers(view_id: str, node_id: str, limit: int = 50, offset: int = 0) -> dict:
    """Paper leaves under a node: distinct DOIs with claim counts (most-cited first)."""
    with db.get_conn() as c:
        total = c.execute(
            "SELECT COUNT(DISTINCT doi) FROM taxonomy_leaves WHERE view_id=? AND node_id=?",
            (view_id, node_id)).fetchone()[0]
        rows = c.execute(
            "SELECT l.doi, COUNT(*) nclaims, s.title, s.year, s.citation_count "
            "FROM taxonomy_leaves l LEFT JOIN sources s ON s.doi=l.doi "
            "WHERE l.view_id=? AND l.node_id=? GROUP BY l.doi "
            "ORDER BY s.citation_count DESC LIMIT ? OFFSET ?",
            (view_id, node_id, limit, offset)).fetchall()
    papers = [{"doi": r["doi"], "n_claims": r["nclaims"], "title": r["title"],
               "year": r["year"], "citations": r["citation_count"]} for r in rows]
    return {"view_id": view_id, "node_id": node_id, "total": total, "papers": papers}


def get_paper_claims(view_id: str, node_id: str, doi: str, limit: int = 100) -> dict:
    """The claims of ONE paper that are placed at THIS node (same tree path) -
    claims of the paper placed elsewhere are intentionally excluded."""
    with db.get_conn() as c:
        rows = c.execute(
            "SELECT l.claim_id, l.label, cl.claim_type, cl.verbatim_quote "
            "FROM taxonomy_leaves l LEFT JOIN claims cl ON cl.claim_id=l.claim_id "
            "WHERE l.view_id=? AND l.node_id=? AND l.doi=? LIMIT ?",
            (view_id, node_id, doi, limit)).fetchall()
        title = c.execute("SELECT title,year FROM sources WHERE doi=?", (doi,)).fetchone()
    return {"view_id": view_id, "node_id": node_id, "doi": doi,
            "title": title["title"] if title else "", "year": title["year"] if title else None,
            "claims": [{"claim_id": r["claim_id"], "label": r["label"],
                        "claim_type": r["claim_type"], "quote": r["verbatim_quote"]}
                       for r in rows]}


def influence(view_id: str, node_id: str, limit: int = 200) -> dict:
    """Rank the co-branch papers under a host by influence and flag the seed.

    Influence blends *intra-branch* citations (how many sibling papers cite it),
    global citation_count, and recency. The seed = an early, impactful paper that
    is cited by >=1 in-branch paper (the evolution-chart 'representative'); we fall
    back to earliest-and-most-cited when the branch has no intra-citations yet.
    Publication time backstops missing citations for ordering the lineage.
    """
    import math
    with db.get_conn() as c:
        rows = c.execute(
            "SELECT l.doi, COUNT(*) nclaims, s.title, s.year, s.citation_count "
            "FROM taxonomy_leaves l LEFT JOIN sources s ON s.doi=l.doi "
            "WHERE l.view_id=? AND l.node_id=? GROUP BY l.doi",
            (view_id, node_id)).fetchall()
        papers = {r["doi"]: {"doi": r["doi"], "n_claims": r["nclaims"],
                             "title": r["title"], "year": r["year"],
                             "citations": r["citation_count"] or 0,
                             "in_cited_by": 0, "in_cites": 0} for r in rows}
        dois = list(papers)
        edges = []
        # bound the intra-branch edge query for very large branches: restrict to
        # the top-N papers by citation_count (seeds/most-cited), else the IN(...)
        # clause over thousands of DOIs against the 2.29M-row citations table is slow.
        EDGE_CAP = 500
        edge_dois = (sorted(dois, key=lambda d: papers[d]["citations"], reverse=True)[:EDGE_CAP]
                     if len(dois) > EDGE_CAP else dois)
        if len(edge_dois) > 1:
            ph = ",".join("?" for _ in edge_dois)
            # intra-branch citation edges: both endpoints are branch papers
            for e in c.execute(
                f"SELECT citing_doi, cited_doi FROM citations "
                f"WHERE cited_doi IN ({ph}) AND citing_doi IN ({ph})",
                (*edge_dois, *edge_dois)).fetchall():
                cg, cd = e["citing_doi"], e["cited_doi"]
                if cg in papers and cd in papers and cg != cd:
                    papers[cd]["in_cited_by"] += 1
                    papers[cg]["in_cites"] += 1
                    edges.append({"citing": cg, "cited": cd})

    yrs = [p["year"] for p in papers.values() if p["year"]]
    ymin, ymax = (min(yrs), max(yrs)) if yrs else (0, 0)

    def _influence(p):
        return 3.0 * p["in_cited_by"] + math.log10(p["citations"] + 1)

    def _seed_score(p):
        recency = 0.0
        if ymax > ymin and p["year"]:
            # earlier = higher (seeds precede followers)
            recency = (ymax - p["year"]) / (ymax - ymin)
        return (2.0 * p["in_cited_by"] + math.log10(p["citations"] + 1) + recency
                + (0.5 if p["in_cited_by"] > 0 else 0.0))

    plist = list(papers.values())
    for p in plist:
        p["influence"] = round(_influence(p), 3)
    seed_doi = None
    if plist:
        cited = [p for p in plist if p["in_cited_by"] > 0]
        pool = cited or plist
        seed_doi = max(pool, key=_seed_score)["doi"]
    for p in plist:
        p["is_seed"] = (p["doi"] == seed_doi)

    plist.sort(key=lambda p: (p["is_seed"], p["influence"], p["citations"]), reverse=True)
    lineage = sorted([p for p in plist if p["year"]], key=lambda p: p["year"])
    return {"view_id": view_id, "node_id": node_id, "n_papers": len(plist),
            "n_intra_edges": len(edges), "seed_doi": seed_doi,
            "papers": plist[:limit], "edges": edges,
            "lineage": [{"doi": p["doi"], "year": p["year"], "title": p["title"],
                         "citations": p["citations"], "in_cited_by": p["in_cited_by"],
                         "is_seed": p["is_seed"]} for p in lineage]}


def get_analysis(view_id: str, node_id: str, doi: str) -> dict:
    """Pre-computed paper intelligence (advisor + critique + contribution) stored
    in paper_analysis. Returns parsed dicts; missing parts are None."""
    import json as _json
    with db.get_conn() as c:
        r = c.execute(
            "SELECT advisor_json,critique_json,contribution_json,generated_at "
            "FROM paper_analysis WHERE view_id=? AND node_id=? AND doi=?",
            (view_id, node_id, doi)).fetchone()
    if not r:
        return {"advisor": None, "critique": None, "contribution": None,
                "generated_at": None}

    def _load(s):
        try:
            return _json.loads(s) if s else None
        except Exception:
            return None
    return {"advisor": _load(r["advisor_json"]), "critique": _load(r["critique_json"]),
            "contribution": _load(r["contribution_json"]), "generated_at": r["generated_at"]}


# ── semantic query -> branch routing ─────────────────────────────────────────
# This is the routing PRIMITIVE: it maps a free-text query onto the tree by
# meaning (query vs node name+definition vectors), not just a name substring.
# A future tree-for-search step can reuse route() inside db.search_claims to
# route/scope a corpus query to a branch and map result DOIs -> taxonomy_leaves;
# it is intentionally decoupled so /api/search stays the full-corpus workhorse.

_NODE_INDEX = None


def _index_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "living_taxonomy" / "output"


def _load_node_index():
    """Lazy-load the node-vector index (built by living_taxonomy/build_node_index.py),
    grouped per view. Returns {} if the index is absent."""
    global _NODE_INDEX
    if _NODE_INDEX is not None:
        return _NODE_INDEX
    import numpy as np
    npz = _index_dir() / "node_index.npz"
    meta = _index_dir() / "node_index_meta.json"
    if not npz.exists() or not meta.exists():
        _NODE_INDEX = {}
        return _NODE_INDEX
    d = np.load(npz, allow_pickle=True)
    rows = json.loads(meta.read_text())
    vecs = d["vecs"]
    by_view = {}
    for i, row in enumerate(rows):
        bv = by_view.setdefault(row["view_id"], {"idx": [], "meta": []})
        bv["idx"].append(i)
        bv["meta"].append(row)
    for v, bv in by_view.items():
        bv["vecs"] = vecs[np.array(bv["idx"])]
    _NODE_INDEX = {"by_view": by_view}
    return _NODE_INDEX


def _embed_query(q: str):
    """Embed a query for branch routing.

    Prefer the resident main-search encoder (``askchem.retrieval`` -> the same
    ``mxbai-embed-large-v1`` already loaded for ``/api/search``) so we do NOT load
    a second SentenceTransformer on the box. Falls back to
    ``living_taxonomy.placement._embed`` only when the search embeddings are not
    loaded (e.g. the light dev server / offline tooling).

    Note the returned dim depends on the host: prod truncates to
    ``CHEMTREE_V2_DIM`` (256-d, Matryoshka) while local/full is 1024-d. ``route``
    aligns the node-vector dim to whatever this returns.
    """
    import numpy as np
    try:
        from askchem import retrieval as _r
        if _r.is_loaded():
            return np.asarray(_r.embed_query(q), dtype="float32")
    except Exception:
        pass
    import sys
    from pathlib import Path
    sys.path.insert(0, str(_index_dir().parent))
    import placement as pm
    return pm._embed([q], is_query=True)[0].astype("float32")


def route(view_id: str, q: str, k: int = 8) -> list[dict]:
    """Rank tree branches for a query by embedding similarity (+ a small boost for
    literal name/short-label matches). Returns concept + path + a representative
    (most-cited) paper for host nodes."""
    idx = _load_node_index()
    bv = idx.get("by_view", {}).get(view_id)
    if not bv:
        return []
    import numpy as np
    qv = _embed_query(q)
    mat = np.asarray(bv["vecs"], dtype="float32")
    # Align dims: the query encoder may return a Matryoshka-truncated vector
    # (256-d on prod) while node vectors are stored at full 1024-d. Slice both
    # to the common (smaller) dim and L2-renormalize so the dot product stays a
    # valid cosine similarity.
    D = min(qv.shape[0], mat.shape[1])
    if qv.shape[0] != D:
        qv = qv[:D]
        n = float(np.linalg.norm(qv))
        if n:
            qv = (qv / n).astype("float32")
    if mat.shape[1] != D:
        mat = mat[:, :D]
        nn = np.linalg.norm(mat, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        mat = (mat / nn).astype("float32")
    sims = mat @ qv
    ql = (q or "").lower().strip()
    scored = []
    for i, row in enumerate(bv["meta"]):
        sim = float(sims[i])
        name = (row.get("name") or "").lower()
        sl = (row.get("short_label") or "").lower()
        hit = bool(ql) and (ql in name or (sl and ql in sl))
        scored.append((sim + (0.15 if hit else 0.0), sim, hit, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, sim, hit, row in scored[:k]:
        nid = row["node_id"]
        entry = {
            "node_id": nid, "name": row.get("name", ""),
            "short_label": row.get("short_label", ""), "kind": row.get("kind", ""),
            "definition": row.get("definition", ""), "equation": row.get("equation", ""),
            "score": round(float(score), 3), "sim": round(sim, 3),
            "via": "name" if hit else "semantic",
            "path": get_path(view_id, nid),
        }
        # representative paper (most-cited under this node) + branch size
        try:
            pap = get_papers(view_id, nid, limit=1)
            entry["n_papers"] = pap.get("total", 0)
            if pap.get("papers"):
                p0 = pap["papers"][0]
                entry["seed"] = {"doi": p0["doi"], "title": p0["title"],
                                 "year": p0["year"], "citations": p0.get("citations")}
        except Exception:
            entry["n_papers"] = 0
        out.append(entry)
    return out


def search(view_id: str, q: str, limit: int = 30, k: int = 8) -> dict:
    """Search-to-node via semantic routing. Returns a ranked `branches` list plus
    back-compat `targets`/`path` (so the existing expand/center UI keeps working).
    On the full server it also folds in claim-hit nodes from the hybrid search."""
    branches = route(view_id, q, k=k)
    targets = {b["node_id"]: {"node_id": b["node_id"], "name": b["name"],
                              "via": b["via"], "sim": b["sim"], "path": b["path"]}
               for b in branches}

    # optional: claim recall via the hybrid engine, mapped to placement nodes
    # (skipped under LTREE_LIGHT to avoid loading the 9.8 GB FAISS index).
    cids = []
    if not os.environ.get("LTREE_LIGHT"):
        try:
            res = db.search_claims(q, view=None, limit=limit).get("results", [])
            cids = [r.get("claim_id") for r in res if r.get("claim_id")]
        except Exception:
            cids = []
    if cids:
        ph = ",".join("?" for _ in cids)
        with db.get_conn() as c:
            for r in c.execute(
                f"SELECT claim_id,node_id FROM taxonomy_leaves "
                f"WHERE view_id=? AND claim_id IN ({ph})", (view_id, *cids)).fetchall():
                t = targets.setdefault(r["node_id"],
                                       {"node_id": r["node_id"], "via": "claims",
                                        "path": get_path(view_id, r["node_id"])})
                t["claim_hits"] = t.get("claim_hits", 0) + 1

    return {"view_id": view_id, "query": q, "branches": branches,
            "targets": list(targets.values()), "n_claim_hits": len(cids)}
