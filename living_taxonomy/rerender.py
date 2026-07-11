"""Re-render the multi-view tree HTML from already-grown data (no LLM).

Lets us apply visualization/template tweaks in build_viz.py without repeating
the ~60 Gemini placement calls. Loads the grown views from
``output/grown_views.json`` if present; otherwise it one-time extracts the
embedded ``const VIEWS = {...}`` object from the existing
``output/scaffold_multiview.html`` and caches it to JSON.

Usage:
    python3 living_taxonomy/rerender.py
    open living_taxonomy/output/scaffold_multiview.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_viz

OUT = _HERE / "output"
JSON_PATH = OUT / "grown_views.json"
HTML_PATH = OUT / "scaffold_multiview.html"


def _extract_views_from_html(html):
    """Brace-match the embedded `const VIEWS = {...};` object."""
    marker = "const VIEWS = "
    i = html.index(marker) + len(marker)
    depth = 0
    for k in range(i, len(html)):
        if html[k] == "{":
            depth += 1
        elif html[k] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[i:k + 1])
    raise ValueError("could not find embedded VIEWS object")


def load_views():
    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text())
        return data["views"], data.get("subtitle", "")
    if not HTML_PATH.exists():
        raise SystemExit("No grown_views.json and no scaffold_multiview.html to "
                         "extract from. Run grow_onto_scaffold.py first.")
    views = _extract_views_from_html(HTML_PATH.read_text())
    JSON_PATH.write_text(json.dumps({"views": views, "subtitle": ""}, indent=2))
    print(f"[rerender] extracted {len(views)} views from HTML -> {JSON_PATH.name}",
          file=sys.stderr)
    return views, ""


def main():
    views, sub = load_views()
    if not sub:
        sub = f"shared trunk + per-view host layers &middot; {len(views)} views"
    first = next(iter(views))
    build_viz.render_html(views[first], "chemistry living tree (with leaves)",
                          sub, HTML_PATH, views=views)
    print(f"[rerender] wrote {HTML_PATH}")
    print(f"[rerender] open {HTML_PATH}")


if __name__ == "__main__":
    main()
