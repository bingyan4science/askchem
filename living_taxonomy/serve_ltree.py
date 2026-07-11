"""Lightweight server for the Living Tree tab (no heavy warmup).

The full askchem server loads a 9.8 GB FAISS index + FTS page-ins at startup,
which is slow/OOM-prone on a laptop. The living-tree endpoints only need plain
DB reads, so this minimal app serves web/index.html + /api/ltree/* instantly.
A catch-all returns {} for any other /api call so the SPA's init() doesn't break.

Usage:
    python3 living_taxonomy/serve_ltree.py          # http://127.0.0.1:8126
    open http://127.0.0.1:8126   ->  "Living Tree" tab
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LTREE_LIGHT", "1")   # skip FAISS-backed claim recall
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from askchem import advisor, ltree

WEB = _REPO / "web"
app = FastAPI(title="AskChem Living Tree (light)")


@app.get("/api/ltree/views")
def views():
    return ltree.list_views()


@app.get("/api/ltree/search")
def search(view: str = Query(...), q: str = Query(...), limit: int = Query(30),
           k: int = Query(8)):
    return ltree.search(view, q, limit=limit, k=k)


@app.get("/api/ltree/{view_id}/root")
def root(view_id: str, depth: int = Query(1, ge=1, le=4)):
    n = ltree.get_node(view_id, ltree.ROOT_ID, depth=depth)
    if not n:
        raise HTTPException(404, "view not found")
    return n


@app.get("/api/ltree/{view_id}/node/{node_id}/papers")
def papers(view_id: str, node_id: str, limit: int = Query(50), offset: int = Query(0)):
    return ltree.get_papers(view_id, node_id, limit=limit, offset=offset)


@app.get("/api/ltree/{view_id}/node/{node_id}/paper-claims")
def paper_claims(view_id: str, node_id: str, doi: str = Query(...), limit: int = Query(100)):
    return ltree.get_paper_claims(view_id, node_id, doi, limit=limit)


@app.get("/api/ltree/{view_id}/node/{node_id}/advise")
def advise(view_id: str, node_id: str, doi: str = Query(...)):
    return advisor.advise(view_id, node_id, doi)


@app.get("/api/ltree/{view_id}/node/{node_id}/influence")
def influence(view_id: str, node_id: str, limit: int = Query(200)):
    return ltree.influence(view_id, node_id, limit=limit)


@app.get("/api/ltree/{view_id}/node/{node_id}/critique")
def critique(view_id: str, node_id: str, doi: str = Query(...)):
    return advisor.critique(view_id, node_id, doi)


@app.get("/api/ltree/{view_id}/node/{node_id}/contribution")
def contribution(view_id: str, node_id: str, doi: str = Query(...)):
    return advisor.contribution(view_id, node_id, doi)


@app.get("/api/ltree/{view_id}/node/{node_id}")
def node(view_id: str, node_id: str, depth: int = Query(1, ge=1, le=4)):
    n = ltree.get_node(view_id, node_id, depth=depth)
    if not n:
        raise HTTPException(404, "node not found")
    n["path"] = ltree.get_path(view_id, node_id)
    return n


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


# stubs so the SPA's other on-load calls don't error out
@app.get("/api/{rest:path}")
def _api_stub(rest: str):
    return JSONResponse({})


@app.post("/api/{rest:path}")
def _api_stub_post(rest: str):
    return JSONResponse({})


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8126, log_level="warning")
