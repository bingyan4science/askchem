#!/usr/bin/env python3
"""Paperclip client for AskChem-Bench (requires Python 3.10+).

The upstream SDK routes API-key auth to ``/papers``, while AskChem-compatible
instances accept API keys on ``/mcp``. This module adapts that behavior and
exposes search, lookup, and paper helpers for ``benchmark_chemtree.py``.

Run standalone smoke tests::

    PAPERCLIP=... python3.14 scripts/paperclip_bench_client.py smoke
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

PAPERCLIP_LIB = Path.home() / ".paperclip" / "lib"
if str(PAPERCLIP_LIB) not in sys.path:
    sys.path.insert(0, str(PAPERCLIP_LIB))

from gxl_paperclip import APIKeyAuth, PaperclipClient  # noqa: E402
from gxl_paperclip.client.models import ExecuteResult  # noqa: E402


def _api_key() -> str:
    pk = os.environ.get("PAPERCLIP") or os.environ.get("PAPERCLIP_API_KEY") or ""
    if not pk:
        raise RuntimeError("PAPERCLIP or PAPERCLIP_API_KEY not set")
    return pk


class BenchPaperclipClient(PaperclipClient):
    """API-key client that posts to ``/mcp`` (SDK 0.3.0 defaults to broken ``/papers``)."""

    def _post_mcp(self, full_command: str, *, timeout: float) -> ExecuteResult:
        self._mcp_call_id += 1
        url = f"{self._base_url}/mcp"
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "paperclip",
                "arguments": {
                    "command": full_command,
                    "description": full_command[:80],
                    "skip_truncation": True,
                },
            },
            "id": self._mcp_call_id,
        }
        resp = self._request(
            "POST",
            url,
            json=payload,
            headers=self._headers(),
            timeout=timeout,
        )
        self._check_response(resp)
        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", err) if isinstance(err, dict) else str(err)
            from gxl_paperclip.client.errors import ServerError

            raise ServerError(str(msg))
        result = data.get("result", {}) or {}
        texts: list[str] = []
        result_id = None
        for block in result.get("content", []) or []:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "") or ""
            texts.append(text)
            if result_id is None:
                m = re.search(r"\[(s_[a-f0-9]+)\]", text)
                if m:
                    result_id = m.group(1)
        return ExecuteResult(
            output="\n".join(texts),
            exit_code=0,
            result_id=result_id,
            raw=data,
        )


def make_client() -> BenchPaperclipClient:
    return BenchPaperclipClient(auth=APIKeyAuth(_api_key()))


def execute(command: str, args: list[str], *, timeout: float = 120) -> ExecuteResult:
    return make_client().execute(command, args, timeout=timeout)


_DOI_RE = re.compile(r"https?://(?:dx\.)?doi\.org/([^\s\]>]+)", re.I)
_PAPER_ID_RE = re.compile(
    r"^\s*(PMC\d+|bio_[a-f0-9]+|med_[a-f0-9]+|arx_[a-f0-9]+)\s*·",
    re.M,
)


def parse_search_output(output: str) -> list[dict[str, Any]]:
    """Parse human-readable ``paperclip search`` listing into paper dicts."""
    papers: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n+", output or "")
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("Found ") or block.startswith("💡"):
            continue
        m_num = re.match(r"^\s*(\d+)\.\s+(.+)$", block, re.M)
        if not m_num:
            continue
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        title = re.sub(r"^\d+\.\s+", "", lines[0]).strip()
        authors = lines[1].strip() if len(lines) > 1 else ""
        paper_id = ""
        journal = ""
        year = ""
        doi = ""
        abstract = ""
        id_m = _PAPER_ID_RE.search(block)
        if id_m:
            paper_id = id_m.group(1)
        for ln in lines:
            dm = _DOI_RE.search(ln)
            if dm:
                doi = dm.group(1).rstrip(".,)")
            if ln.startswith('"') and ln.endswith('"'):
                abstract = ln.strip('"')
        meta_line = lines[2] if len(lines) > 2 else ""
        if "·" in meta_line and paper_id in meta_line:
            parts = [p.strip() for p in meta_line.split("·")]
            if len(parts) >= 2:
                journal = parts[1] if parts[1] != paper_id else (parts[2] if len(parts) > 2 else "")
            date_m = re.search(r"(20\d{2})", meta_line)
            if date_m:
                year = date_m.group(1)
        papers.append({
            "title": title,
            "authors": authors,
            "source_authors": [a.strip() for a in authors.split(",") if a.strip()][:6],
            "source_venue": journal,
            "source_year": year,
            "source_doi": doi,
            "paper_id": paper_id,
            "snippet": abstract,
            "source_paper_title": title,
        })
    return papers


def search_with_flags(
    query: str,
    *,
    limit: int = 15,
    source: str = "pmc,arxiv",
    ranking: str = "hybrid",
    exact: bool = False,
    mode: str | None = None,
    sort: str | None = None,
    timeout: float = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args: list[str] = []
    if exact:
        args.append("-e")
    args.append(query)
    args += ["-n", str(limit), "-s", source, "--ranking", ranking]
    if mode:
        args += ["-m", mode]
    if sort:
        args += ["--sort", sort]
    result = execute("search", args, timeout=timeout)
    papers = parse_search_output(result.output or "")
    stat = {
        "ranking": ranking,
        "mode": mode or "any",
        "exact": exact,
        "source": source,
        "sort": sort,
        "result_id": result.result_id,
        "raw_count": len(papers),
    }
    return papers, stat


def lookup_field(
    field: str,
    value: str,
    *,
    limit: int = 15,
    timeout: float = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = execute("lookup", [field, value, "-n", str(limit)], timeout=timeout)
    papers = parse_search_output(result.output or "")
    return papers, {"lookup": f"{field}={value}", "result_id": result.result_id}


def paper_snippet(paper_id: str, *, lines: int = 40) -> str:
    if not paper_id:
        return ""
    path = f"{paper_id}/content.lines"
    try:
        result = make_client().papers.head(path, lines=lines)
        return (result.output or "")[:4000]
    except Exception:
        return ""


def smoke() -> None:
    client = make_client()
    print("auth ok, client", type(client).__name__)
    papers, stat = search_with_flags("Suzuki Miyaura coupling", limit=5)
    print("search", stat, "papers", len(papers))
    if papers:
        print(" first:", papers[0].get("title", "")[:80], papers[0].get("source_doi"))
        snip = paper_snippet(papers[0].get("paper_id", ""), lines=8)
        print(" snippet:", snip[:200])


def rpc_dispatch(payload: dict) -> dict:
    """JSON-RPC-style dispatch for subprocess calls from benchmark_chemtree."""
    fn = payload.get("fn")
    kw = payload.get("kwargs") or {}
    if fn == "search_with_flags":
        papers, stat = search_with_flags(**kw)
        return {"papers": papers, "stat": stat}
    if fn == "lookup_field":
        papers, stat = lookup_field(**kw)
        return {"papers": papers, "stat": stat}
    if fn == "paper_snippet":
        return {"snippet": paper_snippet(**kw)}
    raise ValueError(f"unknown fn: {fn}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    elif len(sys.argv) > 2 and sys.argv[1] == "rpc":
        import json as _json

        print(_json.dumps(rpc_dispatch(_json.loads(sys.argv[2]))))
    else:
        print("Usage: python3.14 scripts/paperclip_bench_client.py smoke|rpc '<json>'")
