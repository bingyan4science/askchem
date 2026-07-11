"""
Paper-level classification for AskChem (Gemini Batch via Vertex/Portkey).

Instead of classifying each claim independently (which creates incoherent trees),
classify each PAPER once, then propagate paths to all its claims.

Uses a fixed L1 taxonomy to prevent category explosion. Submitted to
Vertex AI Gemini 3.1 Pro via the Portkey gateway batch API.

Pipeline:
    python src/classify_papers.py prepare    # Build batch JSONL (chunked)
    python src/classify_papers.py submit     # Upload + submit all batches
    python src/classify_papers.py poll       # Poll status of all batches
    python src/classify_papers.py collect    # Download outputs and merge
    python src/classify_papers.py rebuild    # Rebuild index with paper-level paths

Required env: PORTKEY_API_KEY, GCS_BKT
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import hashlib

sys.path.insert(0, str(Path(__file__).parent))
from askchem.display import smart_title

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "deep_results"
SOURCES_JSONL = Path(__file__).parent.parent / "askchem" / "sources.jsonl"
PIPELINE_DIR = DATA_DIR / "paper_classify"
DB_PATH = Path(__file__).parent.parent / "chemtree.db"

GATEWAY = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PROVIDER = "@vertexai-gemini-kc119-2"
MODEL = "gemini-3.1-pro-preview"
PAPERS_PER_FILE = 5_000


def _normalize_extraction_model(raw) -> str:
    """Normalise raw extraction_model strings to a clean public label.

    Inputs we have seen in deep_results JSON:
      - '@vertexai-gemini-kc119-2/gemini-3.1-pro-preview' (Portkey route)
      - 'gemini-3.1-pro-preview'
      - None (legacy)
    """
    if not raw:
        return "gemini-3.1-pro"
    s = str(raw).lower()
    if "gemini" in s:
        return "gemini-3.1-pro"
    if "gpt-5.4" in s:
        return "gpt-5.4"
    if "gpt-5-mini" in s or "gpt5mini" in s:
        return "gpt-5-mini"
    return raw


def _curl_json(method, path, data=None, form_fields=None, file_path=None, max_time=60):
    """Call the Portkey gateway via curl and return parsed JSON."""
    api_key = os.environ["PORTKEY_API_KEY"]
    cmd = ["curl", "-s", "--max-time", str(max_time), "-X", method]
    cmd += ["-H", f"x-portkey-api-key: {api_key}"]
    cmd += ["-H", f"x-portkey-provider: {PROVIDER}"]

    if data is not None:
        cmd += ["-H", "Content-Type: application/json"]
        cmd += ["-d", json.dumps(data)]
    elif form_fields or file_path:
        bucket = os.environ["GCS_BKT"]
        cmd += ["-H", f"x-portkey-vertex-storage-bucket-name: {bucket}"]
        if form_fields:
            for k, v in form_fields.items():
                if k == "provider_file_name":
                    cmd += ["-H", f"x-portkey-provider-file-name: {v}"]
                elif k == "provider_model":
                    cmd += ["-H", f"x-portkey-provider-model: {v}"]
                else:
                    cmd += ["--form", f'{k}="{v}"']
        if file_path:
            cmd += ["--form", f"file=@{file_path}"]

    cmd.append(f"{GATEWAY}{path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 30)
    if not result.stdout.strip():
        return {"error": "empty_response", "stderr": result.stderr[:200]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "parse_error", "raw": result.stdout[:500]}

# ── Fixed L1 Taxonomy ────────────────────────────────────────────────────────
# Designed from analysis of 54K claims. Each view has 10-20 canonical L1 categories.

from askchem.taxonomy import CANONICAL_L1 as _CANONICAL_L1

TAXONOMY = {view_id: {cat: cat for cat in cats} for view_id, cats in _CANONICAL_L1.items()}

PAPER_CLASSIFY_PROMPT = """You are classifying a chemistry research paper into a hierarchical index.
Assign this paper to ONE category in each of the 5 views below.

Paper:
Title: {title}
Abstract: {abstract}
Claim types found: {claim_types}
Key topics: {topics}

For each view, choose the BEST-FIT L1 category from the fixed list, then add 1-3 subcategory levels.
Return paths as ["l1_category", "l2_subcategory", "l3_detail"].

{taxonomy_text}

Rules:
- L1 MUST be exactly one of the listed categories (copy it exactly).
- Choose the SINGLE most representative category per view — do NOT split across multiple L1s.
- L2-L3 subcategories: use lowercase_with_underscores, standard chemistry terminology.
- If a view truly doesn't apply, use the "not_applicable" category.

Return JSON:
{{
  "by_reaction_type": ["l1", "l2", ...],
  "by_substance_class": ["l1", "l2", ...],
  "by_application": ["l1", "l2", ...],
  "by_technique": ["l1", "l2", ...],
  "by_mechanism": ["l1", "l2", ...]
}}"""


def _build_taxonomy_text():
    lines = []
    for vid, cats in TAXONOMY.items():
        lines.append(f"\n{vid}:")
        for cat_id, desc in cats.items():
            if cat_id == "not_applicable":
                continue
            lines.append(f"  - {cat_id}: {desc}")
        lines.append(f"  - not_applicable: Not applicable to this paper")
    return '\n'.join(lines)


def load_corpus_metadata() -> dict[str, dict]:
    """Load paper metadata from askchem/sources.jsonl (canonical source).

    Returns a dict keyed by lowercase DOI. Each value uses keys compatible
    with the legacy corpus_checkpoints/ format (title/abstract/authors/year/
    venue/citationCount/openAccessPdf) so the rebuild logic stays unchanged.
    """
    papers: dict[str, dict] = {}
    if not SOURCES_JSONL.exists():
        return papers
    with SOURCES_JSONL.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = (d.get('doi') or '').strip().lower()
            if not doi:
                continue
            authors = d.get('authors') or []
            if authors and isinstance(authors[0], str):
                authors = [{'name': a} for a in authors]
            papers[doi] = {
                'title': d.get('title', ''),
                'abstract': d.get('abstract', '') or '',
                'authors': authors,
                'year': d.get('year') or 0,
                'venue': d.get('venue', ''),
                'citationCount': d.get('citation_count', 0) or 0,
                'openAccessPdf': {'url': d.get('open_access_url', '') or ''},
                'externalIds': {'DOI': d.get('doi', '')},
            }
    return papers


def load_deep_results() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get('num_claims', 0) > 0:
                results.append(data)
        except Exception:
            pass
    return results


def _doi_to_custom_id(doi: str) -> str:
    """Custom IDs in batch responses must round-trip the original DOI.

    Vertex/Portkey accepts most ASCII safely; we keep DOIs as-is but make
    sure they fit a custom_id by hashing if absurdly long (rare).
    """
    if len(doi) <= 200:
        return doi
    h = hashlib.sha256(doi.encode()).hexdigest()[:16]
    return f"DOI__{h}__{doi[:150]}"


def cmd_prepare(args):
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy_text = _build_taxonomy_text()

    corpus = load_corpus_metadata()
    results = load_deep_results()
    print(f"Loaded {len(results)} papers, {len(corpus):,} corpus entries", flush=True)

    classified_file = PIPELINE_DIR / "paper_classifications.json"
    already = set()
    if classified_file.exists():
        already = set(json.loads(classified_file.read_text()).keys())

    requests: list[dict] = []
    custom_id_map: dict[str, str] = {}
    for result in results:
        doi = result.get('doi', '')
        if not doi or doi in already:
            continue

        cp = corpus.get(doi.lower(), {})
        title = cp.get('title', '')
        abstract = (cp.get('abstract', '') or '')[:500]

        pk = result.get('data', {}).get('paper_knowledge', {})
        claims = result.get('data', {}).get('claims', [])
        claim_types = Counter(c.get('claim_type', '') for c in claims)
        claim_type_str = ', '.join(f"{ct}({n})" for ct, n in claim_types.most_common(5))

        topics = []
        if pk.get('subfield'):
            topics.append(pk['subfield'])
        for c in claims[:5]:
            for key in ['reaction_type', 'subject', 'technique_name', 'process_described']:
                val = c.get(key)
                if val and val not in topics:
                    topics.append(val)
                    break
        topics_str = ', '.join(topics[:8])

        prompt = PAPER_CLASSIFY_PROMPT.format(
            title=title,
            abstract=abstract,
            claim_types=claim_type_str,
            topics=topics_str,
            taxonomy_text=taxonomy_text,
        )

        custom_id = _doi_to_custom_id(doi)
        custom_id_map[custom_id] = doi
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
        })

    print(f"Papers needing classification: {len(requests):,}", flush=True)
    if not requests:
        print("Nothing to do.")
        return

    num_files = (len(requests) + PAPERS_PER_FILE - 1) // PAPERS_PER_FILE
    manifest = []
    for fi in range(num_files):
        start = fi * PAPERS_PER_FILE
        end = min(start + PAPERS_PER_FILE, len(requests))
        chunk = requests[start:end]
        fname = f"classify_papers_part{fi:03d}.jsonl"
        fpath = PIPELINE_DIR / fname
        with open(fpath, "w") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")
        manifest.append({
            "file": fname,
            "count": len(chunk),
            "size_mb": round(fpath.stat().st_size / 1e6, 1),
        })
        print(f"  {fname}: {len(chunk)} requests, {manifest[-1]['size_mb']} MB", flush=True)

    (PIPELINE_DIR / "manifest.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "files": manifest}, indent=2)
    )
    (PIPELINE_DIR / "custom_id_map.json").write_text(json.dumps(custom_id_map))
    print(f"\nGenerated {num_files} batch files in {PIPELINE_DIR}/")
    print(f"Total requests: {len(requests):,}")


def cmd_submit(args):
    """Upload JSONL files to GCS and submit batch jobs to Vertex via Portkey."""
    manifest_path = PIPELINE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json found. Run 'prepare' first.")
        return

    manifest = json.loads(manifest_path.read_text())
    tracker_path = PIPELINE_DIR / "tracker.json"
    tracker = {}
    if tracker_path.exists():
        tracker = json.loads(tracker_path.read_text())

    ok = fail = 0
    total = len(manifest["files"])
    for i, entry in enumerate(manifest["files"]):
        fname = entry["file"]
        if fname in tracker and tracker[fname].get("status") not in ("failed",):
            print(f"  [{i+1}/{total}] {fname}: already submitted "
                  f"(status={tracker[fname].get('status')}), skipping")
            continue

        fpath = PIPELINE_DIR / fname
        size_mb = entry.get("size_mb", 0)
        upload_timeout = max(120, int(size_mb * 5))
        print(f"  [{i+1}/{total}] {fname} ({size_mb} MB)...", end=" ", flush=True)

        upload_resp = _curl_json(
            "POST", "/files",
            form_fields={
                "purpose": "batch",
                "provider_file_name": fname,
                "provider_model": MODEL,
            },
            file_path=str(fpath),
            max_time=upload_timeout,
        )
        file_id = upload_resp.get("id")
        if not file_id:
            print(f"UPLOAD FAILED: {str(upload_resp)[:80]}")
            fail += 1
            time.sleep(10)
            continue

        batch_resp = _curl_json("POST", "/batches", data={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": MODEL,
        })
        batch_id = batch_resp.get("id")
        status = batch_resp.get("status", "unknown")
        if batch_id:
            print(f"OK batch={batch_id[:20]}")
            ok += 1
        else:
            print(f"NO BATCH: {str(batch_resp)[:80]}")
            fail += 1

        tracker[fname] = {
            "file_id": file_id,
            "batch_id": batch_id,
            "status": status,
            "submitted_at": datetime.now().isoformat(),
        }
        with open(tracker_path, "w") as f:
            json.dump(tracker, f, indent=2)
        time.sleep(3)

    print(f"\nSubmit done: {ok} ok, {fail} fail out of {total}")


def cmd_poll(args):
    """Check status of all submitted batch jobs."""
    tracker_path = PIPELINE_DIR / "tracker.json"
    if not tracker_path.exists():
        print("No tracker.json found. Run 'submit' first.")
        return

    tracker = json.loads(tracker_path.read_text())
    summary = {"validating": 0, "in_progress": 0, "completed": 0, "failed": 0, "other": 0}

    for fname, info in tracker.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            continue
        resp = _curl_json("GET", f"/batches/{batch_id}")
        new_status = resp.get("status", "unknown")
        counts = resp.get("request_counts", {}) or {}
        info["status"] = new_status
        info["request_counts"] = counts
        cat = new_status if new_status in summary else "other"
        summary[cat] += 1
        completed = counts.get("completed") or 0
        total = counts.get("total") or 0
        print(f"  {fname}: {new_status} ({completed}/{total})")
        time.sleep(0.4)

    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)
    print(f"\nSummary: {json.dumps(summary)}")


def cmd_collect(args):
    """Download completed batch outputs and merge into paper_classifications.json."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tracker_path = PIPELINE_DIR / "tracker.json"
    if not tracker_path.exists():
        print("No tracker.json found.")
        return
    tracker = json.loads(tracker_path.read_text())
    output_dir = PIPELINE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    cid_map_path = PIPELINE_DIR / "custom_id_map.json"
    cid_map = json.loads(cid_map_path.read_text()) if cid_map_path.exists() else {}

    to_collect = [(k, v) for k, v in tracker.items()
                  if v.get("status") == "completed" and not v.get("collected") and v.get("batch_id")]
    print(f"Collecting {len(to_collect)} completed batches (8 workers)...")

    api_key = os.environ["PORTKEY_API_KEY"]
    lock = threading.Lock()
    stats = {"ok": 0, "empty": 0}

    def _dl_one(fname, info):
        batch_id = info["batch_id"]
        cmd = ["curl", "-s", "--max-time", "180", "-X", "GET",
               "-H", f"x-portkey-api-key: {api_key}",
               "-H", f"x-portkey-provider: {PROVIDER}",
               f"{GATEWAY}/batches/{batch_id}/output"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            raw = result.stdout.strip()
        except Exception:
            raw = ""
        if not raw or len(raw) < 10:
            with lock:
                stats["empty"] += 1
            return
        (output_dir / fname).write_text(result.stdout)
        with lock:
            info["collected"] = True
            info["collected_at"] = datetime.now().isoformat()
            stats["ok"] += 1
            if stats["ok"] % 5 == 0:
                with open(tracker_path, "w") as tf:
                    json.dump(tracker, tf, indent=2)
                print(f"  Collected: {stats['ok']}/{len(to_collect)} (empty: {stats['empty']})", flush=True)

    if to_collect:
        workers = min(8, len(to_collect))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_dl_one, k, v) for k, v in to_collect]
            for f in as_completed(futs):
                try: f.result()
                except Exception: pass
        with open(tracker_path, "w") as tf:
            json.dump(tracker, tf, indent=2)
        print(f"Collect done: {stats['ok']} ok, {stats['empty']} empty/failed")

    # Parse & merge into paper_classifications.json
    classifications = {}
    classified_file = PIPELINE_DIR / "paper_classifications.json"
    if classified_file.exists():
        classifications = json.loads(classified_file.read_text())

    errors = 0
    new_count = 0
    for ofile in output_dir.glob("*.jsonl"):
        for line in ofile.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cid = item.get("custom_id", "")
                doi = cid_map.get(cid, cid)
                response = item.get("response", {})
                body = response.get("body", {})
                if response.get("status_code") not in (200, None):
                    errors += 1
                    continue
                choices = body.get("choices", [])
                if not choices:
                    errors += 1
                    continue
                text = choices[0].get("message", {}).get("content", "") or ""
                if not text.strip():
                    errors += 1
                    continue
                paths = json.loads(text)
                # Validate / normalize L1s against fixed taxonomy
                for vid, taxonomy in TAXONOMY.items():
                    p = paths.get(vid, [])
                    if p and isinstance(p, list) and len(p) > 0:
                        if isinstance(p[0], list):
                            p = p[0]
                            paths[vid] = p
                        l1 = p[0] if p else ''
                        if l1 not in taxonomy:
                            paths[vid] = ['not_applicable']
                if doi not in classifications:
                    new_count += 1
                classifications[doi] = paths
            except Exception:
                errors += 1

    classified_file.write_text(json.dumps(classifications, indent=2))
    print(f"\nPaper classifications: {len(classifications):,} total ({new_count} new, {errors} errors)")

    for vid in TAXONOMY:
        l1_counts = Counter()
        for _doi, paths in classifications.items():
            p = paths.get(vid, [])
            if p:
                l1_counts[p[0]] += 1
        print(f"\n  {vid}:")
        for l1, n in l1_counts.most_common():
            print(f"    {l1:35s} {n:6d}")


def _tokenize(slug: str) -> set[str]:
    """Split a slug into word tokens for fuzzy matching."""
    return set(slug.replace('_', ' ').split())


def _token_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_subset_name(a: str, b: str) -> bool:
    """Check if a's tokens are a subset of b's (or vice versa)."""
    ta, tb = _tokenize(a), _tokenize(b)
    return ta <= tb or tb <= ta


NOISE_WORDS = {'and', 'or', 'the', 'of', 'in', 'for', 'with', 'based', 'type',
               'general', 'various', 'related', 'other', 'like', 'class'}

PLURAL_PAIRS = [
    ('s', ''),  # catalysts → catalyst
]


def _normalize_slug(slug: str) -> str:
    """Normalize a slug for comparison: strip noise words, singularize."""
    tokens = slug.replace('_', ' ').split()
    tokens = [t for t in tokens if t not in NOISE_WORDS]
    return '_'.join(tokens)


def build_l2_merge_map(paper_paths: dict, views: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    """
    Build a mapping: {view_id: {l1: {raw_l2_slug: canonical_l2_slug}}}
    
    Strategy:
    1. Collect all L2 slugs per (view, L1) with paper counts
    2. Cluster by token similarity (threshold 0.6) or subset relationship
    3. Pick highest-count variant as canonical
    """
    # Collect L2 → paper count per (view, L1)
    l2_counts = defaultdict(lambda: defaultdict(lambda: Counter()))
    for doi, paths in paper_paths.items():
        for vid in views:
            p = paths.get(vid, [])
            if len(p) >= 2:
                l1 = p[0]
                l2 = p[1].strip().lower().replace('-', '_').replace(' ', '_')
                l2_counts[vid][l1][l2] += 1

    merge_map = {}
    total_merges = 0

    for vid in views:
        merge_map[vid] = {}
        for l1, slugs in l2_counts[vid].items():
            if len(slugs) <= 1:
                continue

            # Sort by count descending — canonical = most popular
            sorted_slugs = sorted(slugs.items(), key=lambda x: -x[1])
            canonical_map = {}
            clusters = []  # list of (canonical, [members])

            for slug, count in sorted_slugs:
                norm = _normalize_slug(slug)
                merged = False

                for canonical, members in clusters:
                    canon_norm = _normalize_slug(canonical)
                    
                    # Exact match after normalization
                    if norm == canon_norm:
                        canonical_map[slug] = canonical
                        members.append(slug)
                        merged = True
                        break
                    
                    # Singular/plural match
                    if (norm.rstrip('s') == canon_norm.rstrip('s') and
                            abs(len(norm) - len(canon_norm)) <= 1):
                        canonical_map[slug] = canonical
                        members.append(slug)
                        merged = True
                        break
                    
                    # High token similarity (≥0.7) AND one is subset of other
                    sim = _token_similarity(slug, canonical)
                    if sim >= 0.7 and _is_subset_name(slug, canonical):
                        canonical_map[slug] = canonical
                        members.append(slug)
                        merged = True
                        break
                    
                    # Very high similarity (≥0.85) — almost certainly the same
                    if sim >= 0.85:
                        canonical_map[slug] = canonical
                        members.append(slug)
                        merged = True
                        break

                if not merged:
                    clusters.append((slug, [slug]))
                    canonical_map[slug] = slug

            # Only store entries that actually change
            actual = {k: v for k, v in canonical_map.items() if k != v}
            if actual:
                merge_map[vid][l1] = actual
                total_merges += len(actual)

    print(f"L2 merge map: {total_merges} slugs merged across all views")
    for vid in views:
        vid_merges = sum(len(m) for m in merge_map[vid].values())
        if vid_merges:
            print(f"  {vid}: {vid_merges} merges")
            for l1, merges in sorted(merge_map[vid].items()):
                for old, new in sorted(merges.items()):
                    print(f"    {l1}/{old} → {l1}/{new}")
    return merge_map


def cmd_rebuild(args):
    """Rebuild the SQLite index using paper-level classifications."""
    import sqlite3
    from askchem.models import Claim, DEFAULT_VIEWS
    from askchem.taxonomy import CLAIM_TYPE_LABELS

    classified_file = PIPELINE_DIR / "paper_classifications.json"
    if not classified_file.exists():
        print("No paper classifications. Run collect first.")
        return

    paper_paths = json.loads(classified_file.read_text())
    print(f"Loaded {len(paper_paths)} paper classifications", flush=True)

    ALL_CONTENT_VIEWS = ['by_reaction_type', 'by_substance_class', 'by_application',
                         'by_technique', 'by_mechanism']

    # Build L2 merge map before loading heavy data
    l2_merge = build_l2_merge_map(paper_paths, ALL_CONTENT_VIEWS)

    corpus = load_corpus_metadata()
    results = load_deep_results()
    print(f"Loaded {len(results)} deep results, {len(corpus):,} corpus entries", flush=True)

    # Fresh DB
    if DB_PATH.exists():
        DB_PATH.unlink()

    from askchem import db
    db.init_db()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")

    # Insert views
    for view in DEFAULT_VIEWS:
        c.execute(
            "INSERT OR REPLACE INTO views (view_id, name, description, data) VALUES (?,?,?,?)",
            (view.view_id, view.name, view.description, json.dumps(view.to_dict()))
        )
    conn.commit()

    def _normalize_seg(seg: str) -> str:
        """Normalize a path segment: lowercase, hyphens→underscores, strip whitespace."""
        return seg.strip().lower().replace('-', '_').replace(' ', '_')

    source_batch = []
    claim_batch = []
    fts_batch = []
    node_counts = defaultdict(lambda: defaultdict(int))
    node_claims = defaultdict(lambda: defaultdict(list))
    total_claims = 0
    total_sources = 0
    sources_seen = set()
    claim_ids_seen: set[str] = set()

    # ── Phase 1: legacy abstract claims from askchem/claims.jsonl ──
    # These already have view_paths; preserve them so we don't lose
    # the ~875k abstract claims when rebuilding.
    abstract_file = Path(__file__).parent.parent / "askchem" / "claims.jsonl"
    abstract_count = 0
    abstract_classified = 0
    if abstract_file.exists():
        print("Phase 1: Loading existing abstract claims...", flush=True)
        with open(abstract_file) as f:
            for li, line in enumerate(f):
                try:
                    claim = json.loads(line)
                except json.JSONDecodeError:
                    continue
                claim_id = claim.get('claim_id', '')
                if not claim_id:
                    continue
                claim_ids_seen.add(claim_id)
                claim_type = claim.get('claim_type', 'unknown')
                doi = (claim.get('source_doi') or '').strip()
                source_title = claim.get('source_paper_title', '')

                if doi and doi.lower() not in sources_seen:
                    sources_seen.add(doi.lower())
                    cp = corpus.get(doi.lower(), {})
                    authors = [a.get('name', '') for a in (cp.get('authors') or [])[:20]]
                    source_data = {
                        'doi': doi,
                        'title': cp.get('title', '') or source_title,
                        'authors': authors,
                        'year': cp.get('year') or 0,
                        'venue': cp.get('venue', ''),
                        'abstract': cp.get('abstract', ''),
                        'citation_count': cp.get('citationCount', 0) or 0,
                        'open_access_url': (cp.get('openAccessPdf') or {}).get('url', ''),
                    }
                    source_batch.append((
                        doi, source_data['title'], json.dumps(authors),
                        source_data['year'], source_data['venue'], source_data['abstract'],
                        source_data['citation_count'], source_data['open_access_url'],
                        json.dumps(source_data),
                    ))
                    total_sources += 1

                view_paths = claim.get('view_paths', {}) or {}
                if view_paths and any(v for v in view_paths.values() if v):
                    abstract_classified += 1

                for vid, segs in view_paths.items():
                    if not segs:
                        continue
                    full_path = '/'.join(str(s) for s in segs)
                    for depth in range(len(segs)):
                        partial = '/'.join(str(s) for s in segs[: depth + 1])
                        node_counts[vid][partial] += 1
                    node_claims[vid][full_path].append(claim_id)

                # Searchable text (subset of fields for abstract claims).
                # Some fields can be lists (e.g. multi-subject claims), so flatten.
                parts: list[str] = []
                for key in ('claim_type', 'verbatim_quote', 'subject'):
                    v = claim.get(key, '')
                    if isinstance(v, list):
                        parts.extend(str(x) for x in v if x)
                    elif v:
                        parts.append(str(v))
                searchable = ' '.join(parts)

                claim_batch.append((
                    claim_id, claim_type, doi, source_title,
                    claim.get('confidence', 'high'),
                    claim.get('location_in_paper', ''),
                    claim.get('verbatim_quote', ''),
                    claim.get('extraction_model', 'gpt-5-mini'),
                    claim.get('extraction_version', 'v3-abstract-batch'),
                    claim.get('extracted_at', ''),
                    json.dumps(view_paths),
                    json.dumps(claim),
                ))
                fts_batch.append((
                    claim_id, claim_type, source_title,
                    claim.get('verbatim_quote', ''), searchable,
                ))
                total_claims += 1
                abstract_count += 1

                if (li + 1) % 50000 == 0:
                    _flush(c, conn, source_batch, claim_batch, fts_batch)
                    source_batch.clear(); claim_batch.clear(); fts_batch.clear()
                    print(f"  Phase 1: {li+1:,} claims, {total_sources:,} sources", flush=True)

        _flush(c, conn, source_batch, claim_batch, fts_batch)
        source_batch.clear(); claim_batch.clear(); fts_batch.clear()
        print(f"  Phase 1 done: {abstract_count:,} abstract claims "
              f"({abstract_classified:,} classified)", flush=True)

    # ── Phase 2: deep claims with paper-level Gemini classifications ──
    print("Phase 2: Loading deep extraction results with paper-level paths...", flush=True)
    deep_dupes = 0
    for ri, result in enumerate(results):
        doi = result.get('doi', '')
        if not doi:
            continue

        # Source
        if doi.lower() not in sources_seen:
            sources_seen.add(doi.lower())
            cp = corpus.get(doi.lower(), {})
            authors = [a.get('name', '') for a in (cp.get('authors') or [])[:20]]
            source_data = {
                'doi': doi,
                'title': cp.get('title', ''),
                'authors': authors,
                'year': cp.get('year') or 0,
                'venue': cp.get('venue', ''),
                'abstract': cp.get('abstract', ''),
                'citation_count': cp.get('citationCount', 0) or 0,
                'open_access_url': (cp.get('openAccessPdf') or {}).get('url', ''),
            }
            source_batch.append((
                doi, source_data['title'], json.dumps(authors),
                source_data['year'], source_data['venue'], source_data['abstract'],
                source_data['citation_count'], source_data['open_access_url'],
                json.dumps(source_data),
            ))
            total_sources += 1

        # Paper-level paths (same for all claims in this paper)
        pp = paper_paths.get(doi, {})
        paper_view_paths = {}
        for vid in ALL_CONTENT_VIEWS:
            p = pp.get(vid, ['not_applicable'])
            if isinstance(p, list) and p and isinstance(p[0], list):
                p = p[0]
            p = [_normalize_seg(str(s)) for s in p if isinstance(s, (str, int, float))]
            if p and p != ['not_applicable']:
                # Apply L2 merge map
                if len(p) >= 2 and vid in l2_merge:
                    l1_merges = l2_merge[vid].get(p[0], {})
                    if p[1] in l1_merges:
                        p[1] = l1_merges[p[1]]
                paper_view_paths[vid] = p

        paper_knowledge = result.get('data', {}).get('paper_knowledge', {})
        subfield = paper_knowledge.get('subfield', '')

        for raw_claim in result.get('data', {}).get('claims', []):
            claim_type = raw_claim.get('claim_type', 'unknown')
            content_hash = hashlib.sha256(
                json.dumps(raw_claim, sort_keys=True).encode()
            ).hexdigest()[:12]
            claim_id = Claim.generate_id(doi, claim_type, content_hash)

            if claim_id in claim_ids_seen:
                deep_dupes += 1
                continue
            claim_ids_seen.add(claim_id)

            view_paths = dict(paper_view_paths)

            # Plus deterministic by_claim_type
            ct_l1 = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
            ct_path = [ct_l1]
            if subfield:
                ct_path.append(subfield.lower().replace(' ', '_'))
            view_paths['by_claim_type'] = ct_path

            # Track tree nodes
            for vid, segs in view_paths.items():
                full_path = '/'.join(segs)
                for depth in range(len(segs)):
                    partial = '/'.join(segs[:depth + 1])
                    node_counts[vid][partial] += 1
                node_claims[vid][full_path].append(claim_id)

            # Searchable text
            parts = [
                raw_claim.get('claim_type', ''),
                raw_claim.get('verbatim_quote', ''),
                raw_claim.get('subject', ''),
                raw_claim.get('property_name', ''),
                raw_claim.get('reaction_type', ''),
                raw_claim.get('technique_name', ''),
                raw_claim.get('process_described', ''),
                raw_claim.get('hypothesis_text', ''),
                raw_claim.get('finding_text', ''),
                raw_claim.get('limitation_text', ''),
                raw_claim.get('direction_text', ''),
                raw_claim.get('comparison_result', ''),
                raw_claim.get('what_it_achieves', ''),
            ]
            for key in ['reactants', 'products']:
                for item in raw_claim.get(key, []):
                    if isinstance(item, dict):
                        parts.append(item.get('name', ''))
            searchable = ' '.join(p for p in parts if p)

            source_title = corpus.get(doi.lower(), {}).get('title', '')
            ext_model = _normalize_extraction_model(
                result.get('extraction_model') or raw_claim.get('extraction_model')
            )
            claim_data = dict(raw_claim)
            claim_data.update({
                'claim_id': claim_id, 'source_doi': doi,
                'source_paper_title': source_title,
                'extraction_model': ext_model, 'extraction_version': 'deep_v1',
                'view_paths': view_paths,
            })

            claim_batch.append((
                claim_id, claim_type, doi, source_title,
                raw_claim.get('confidence', 'high'),
                raw_claim.get('location_in_paper', ''),
                raw_claim.get('verbatim_quote', ''),
                ext_model, 'deep_v1',
                result.get('collected_at', datetime.now().isoformat()),
                json.dumps(view_paths), json.dumps(claim_data),
            ))
            fts_batch.append((
                claim_id, claim_type, source_title,
                raw_claim.get('verbatim_quote', ''), searchable,
            ))
            total_claims += 1

        if (ri + 1) % 100 == 0:
            _flush(c, conn, source_batch, claim_batch, fts_batch)
            source_batch.clear(); claim_batch.clear(); fts_batch.clear()
            print(f"  {ri+1}/{len(results)} papers, {total_claims:,} claims", flush=True)

    _flush(c, conn, source_batch, claim_batch, fts_batch)
    deep_count = total_claims - abstract_count
    print(f"\n  Sources: {total_sources:,}", flush=True)
    print(f"  Claims:  {total_claims:,} "
          f"(abstract: {abstract_count:,}, deep: {deep_count:,}, "
          f"dupes skipped: {deep_dupes:,})", flush=True)

    # Build tree nodes
    print("Building tree nodes...", flush=True)
    node_batch = []
    for vid, paths in node_counts.items():
        children_map = defaultdict(set)
        for path_str in paths:
            segs = path_str.split('/')
            if len(segs) > 1:
                parent = '/'.join(segs[:-1])
                children_map[parent].add(segs[-1])

        for path_str, count in paths.items():
            segs = path_str.split('/')
            level = len(segs)
            name = smart_title(segs[-1])
            children = sorted(children_map.get(path_str, set()))
            claim_ids = node_claims[vid].get(path_str, [])
            node_data = {
                'view_id': vid, 'path': path_str, 'name': name,
                'level': level, 'claim_count': count,
                'children': children, 'claim_ids': claim_ids[:2000],
            }
            node_batch.append((
                vid, path_str, name, level, count,
                json.dumps(children), json.dumps(claim_ids[:2000]),
                json.dumps(node_data),
            ))

    c.executemany(
        "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
        node_batch)
    conn.commit()

    total_nodes = len(node_batch)
    nodes_per_view = Counter(n[0] for n in node_batch)
    print(f"  Tree nodes: {total_nodes:,}")
    for v, cnt in nodes_per_view.most_common():
        print(f"    {v}: {cnt:,}")

    # Root nodes
    for vid in nodes_per_view:
        l1 = c.execute(
            "SELECT path, claim_count FROM tree_nodes WHERE view_id=? AND level=1 ORDER BY claim_count DESC",
            (vid,)).fetchall()
        children = [r[0] for r in l1]
        total = sum(r[1] for r in l1)
        root_data = {'view_id': vid, 'path': '', 'name': vid, 'level': 0,
                     'claim_count': total, 'children': children, 'claim_ids': []}
        c.execute(
            "INSERT OR REPLACE INTO tree_nodes (view_id,path,name,level,claim_count,children,claim_ids,data) VALUES (?,?,?,?,?,?,?,?)",
            (vid, '', vid, 0, total, json.dumps(children), json.dumps([]), json.dumps(root_data)))
    conn.commit()

    # Metadata
    view_count = len(nodes_per_view) + 1
    for k, v in [
        ('total_claims', str(total_claims)),
        ('total_sources', str(total_sources)),
        ('total_nodes', str(total_nodes)),
        ('total_views', str(view_count)),
        ('version', '3.0.0'),
        ('built_at', datetime.now().isoformat()),
    ]:
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)", (k, v))
    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"INDEX REBUILT (paper-level classification)")
    print(f"{'='*60}")
    print(f"Sources:    {total_sources:,}")
    print(f"Claims:     {total_claims:,}")
    print(f"Tree nodes: {total_nodes:,}")
    for v, paths in node_counts.items():
        l1_count = len([p for p in paths if '/' not in p])
        print(f"  {v}: {l1_count} L1 categories")


def _flush(c, conn, source_batch, claim_batch, fts_batch):
    if source_batch:
        c.executemany(
            "INSERT OR REPLACE INTO sources (doi,title,authors,year,venue,abstract,citation_count,open_access_url,data) VALUES (?,?,?,?,?,?,?,?,?)",
            source_batch)
    if claim_batch:
        c.executemany(
            "INSERT OR REPLACE INTO claims (claim_id,claim_type,source_doi,source_paper_title,confidence,location_in_paper,verbatim_quote,extraction_model,extraction_version,extracted_at,view_paths,data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            claim_batch)
    if fts_batch:
        c.executemany(
            "INSERT OR REPLACE INTO claims_fts (claim_id,claim_type,source_paper_title,verbatim_quote,searchable_text) VALUES (?,?,?,?,?)",
            fts_batch)
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prepare"); sub.add_parser("submit"); sub.add_parser("poll")
    sub.add_parser("collect"); sub.add_parser("rebuild")
    args = parser.parse_args()
    cmd_map = {'prepare': cmd_prepare, 'submit': cmd_submit, 'poll': cmd_poll,
               'collect': cmd_collect, 'rebuild': cmd_rebuild}
    if args.command in cmd_map:
        cmd_map[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
