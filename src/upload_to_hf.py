"""
Upload the FULL AskChem index to HuggingFace as a dataset.

Publishes both abstract-extracted and deep full-paper extraction claims
(plus the SQLite database file itself for fast local serving).

Reads from the SQLite database and packages into clean files:
  - chemtree.db           — full SQLite database (LFS, ~10 GB)
  - claims.jsonl          — every claim, one JSON object per line
  - sources.jsonl         — every source paper, one JSON object per line
  - hierarchy/            — per-view tree structure (flattened nodes)
  - paper_classifications.json — paper-level Gemini path map (optional)
  - metadata.json         — index stats

Use ``--abstract-only`` to publish only the abstract-extracted slice
(legacy behaviour) and ``--no-db`` to skip uploading the SQLite blob.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from huggingface_hub import HfApi, create_repo

sys.path.insert(0, str(Path(__file__).parent))

REPO_ROOT = Path(__file__).parent.parent
# Canonical DB is askchem.db (renamed from chemtree.db); legacy name is fallback.
DB_PATH = (REPO_ROOT / "askchem.db") if (REPO_ROOT / "askchem.db").exists() \
    else (REPO_ROOT / "chemtree.db")
DB_FILE = "askchem.db"
REPO_ID = "bing-yan/askchem"

# v2 retrieval artefacts (γ2 / δ3). Four files cover the prod runtime
# across both droplet sizes we currently care about:
#   • the 1024-d FAISS IndexFlatIP — only practical on hosts with ≥
#     16 GB free RAM (laptop dev box & future bigger droplets);
#   • the 256-d Matryoshka FAISS — 4× smaller, the one we actually
#     ship to the current 8 GB DigitalOcean droplet (deploy_to_vps.sh
#     sets CHEMTREE_V2_DIM=256 to select it);
#   • the claim-id sidecar that ``embeddings_v2.load_embeddings`` reads
#     with mmap so the deploy avoids materialising the 10 GB npz
#     (works for both dims; deploy_to_vps.sh symlinks the
#     ``_256.claim_ids.npy`` view); and
#   • the source npz, optional but uploaded for reproducibility so
#     anyone can rebuild a different FAISS variant (HNSW, IVF-PQ,
#     other Matryoshka dims, etc.) from the same vectors that the
#     live index was built from.
V2_FAISS_PATH        = REPO_ROOT / "data" / "claim_embeddings.v2.faiss"
V2_FAISS_256_PATH    = REPO_ROOT / "data" / "claim_embeddings.v2_256.faiss"
V2_IDS_PATH          = REPO_ROOT / "data" / "claim_embeddings.v2.claim_ids.npy"
V2_NPZ_PATH          = REPO_ROOT / "data" / "claim_embeddings.v2.npz"
V2_ARTEFACTS_RUNTIME = (V2_FAISS_PATH, V2_FAISS_256_PATH, V2_IDS_PATH)
V2_ARTEFACTS_FULL    = (V2_FAISS_PATH, V2_FAISS_256_PATH, V2_IDS_PATH, V2_NPZ_PATH)

ABSTRACT_VERSIONS = ("v3-abstract", "v3-abstract-batch")
PAPER_CLASS_PATH = (
    Path(__file__).parent.parent / "data" / "paper_classify" / "paper_classifications.json"
)

CLAIM_TYPES = [
    "reaction", "property", "method", "mechanism", "comparison",
    "computational_result",
]

VIEWS = [
    ("by_reaction_type", "Chemical transformation type"),
    ("by_substance_class", "Molecules/materials involved"),
    ("by_application", "Practical application domain"),
    ("by_technique", "Experimental/computational method"),
    ("by_mechanism", "Underlying mechanism/phenomenon"),
    ("by_claim_type", "Epistemic role of the claim"),
    ("by_data", "Extracted numerical measurements"),
    ("by_time_period", "Chronological organization"),
]


def package_dataset(output_dir: Path, abstract_only: bool = False,
                    include_db: bool = True,
                    include_v2_embeddings: Optional[str] = None):
    """Package the index into HuggingFace-friendly files.

    abstract_only=False (default): publish ALL claims + the chemtree.db file.
    abstract_only=True:           publish only v3-abstract* claims (legacy).
    include_db=True (default):    also copy chemtree.db into the upload dir.
    include_v2_embeddings:        None to skip; ``"runtime"`` to ship only
                                  the two files the prod server reads
                                  (FAISS + claim-ids sidecar, ≈ 9.5 GB);
                                  ``"full"`` to additionally ship the
                                  10 GB npz source-of-truth for
                                  reproducibility (≈ 19.5 GB).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Claims -> JSONL
    if abstract_only:
        print("Packaging abstract-extracted claims...", flush=True)
        claim_iter = conn.execute(
            "SELECT data, source_doi FROM claims "
            "WHERE extraction_version IN (?, ?) ORDER BY claim_id",
            ABSTRACT_VERSIONS,
        )
    else:
        print("Packaging ALL claims (abstract + deep)...", flush=True)
        claim_iter = conn.execute(
            "SELECT data, source_doi FROM claims ORDER BY claim_id"
        )

    claim_count = 0
    source_dois: set[str] = set()
    with open(output_dir / "claims.jsonl", "w") as out:
        for row in claim_iter:
            out.write(row["data"] + "\n")
            source_dois.add(row["source_doi"])
            claim_count += 1
    print(f"  {claim_count:,} claims from {len(source_dois):,} papers", flush=True)

    # Sources -> JSONL (only papers that actually have included claims)
    print("Packaging sources...", flush=True)
    source_count = 0
    with open(output_dir / "sources.jsonl", "w") as out:
        for row in conn.execute("SELECT data, doi FROM sources ORDER BY doi"):
            if row["doi"] in source_dois:
                out.write(row["data"] + "\n")
                source_count += 1
    print(f"  {source_count:,} sources", flush=True)

    # Hierarchy -> one JSON per view
    print("Packaging hierarchy...", flush=True)
    hier_dir = output_dir / "hierarchy"
    hier_dir.mkdir(exist_ok=True)

    views = conn.execute("SELECT * FROM views ORDER BY view_id").fetchall()
    for view_row in views:
        view_id = view_row["view_id"]
        nodes = []
        for nrow in conn.execute(
            "SELECT data FROM tree_nodes WHERE view_id = ? ORDER BY path", (view_id,)
        ):
            nodes.append(json.loads(nrow["data"]))

        view_data = {
            "view_id": view_id,
            "name": view_row["name"],
            "description": view_row["description"],
            "node_count": len(nodes),
            "nodes": nodes,
        }
        with open(hier_dir / f"{view_id}.json", "w") as f:
            json.dump(view_data, f)
        print(f"  {view_id}: {len(nodes):,} nodes", flush=True)

    # Metadata
    total_nodes = conn.execute("SELECT COUNT(*) FROM tree_nodes").fetchone()[0]
    view_count = len(views)

    if abstract_only:
        scope = "abstract-only"
        ext_model = "gpt-5-mini (abstract)"
        cls_model = "gpt-5-mini"
    else:
        scope = "full (abstract + deep full-paper)"
        ext_model = "gpt-5-mini (abstract) + gemini-3.1-pro (deep full-paper)"
        cls_model = "gemini-3.1-pro batch (paper-level + claim-level via Vertex AI)"

    meta = {
        "dataset_version": datetime.now().strftime("%Y%m%d"),
        "claim_count": claim_count,
        "source_count": source_count,
        "view_count": view_count,
        "node_count": total_nodes,
        "views": [v[0] for v in VIEWS],
        "claim_types": CLAIM_TYPES,
        "extraction_scope": scope,
        "extraction_model": ext_model,
        "classification_model": cls_model,
        "description": (
            "AskChem: A structured, hierarchical, multi-view knowledge index "
            "for chemistry research. Claims are extracted from "
            f"{('paper abstracts' if abstract_only else 'paper abstracts AND full PDFs')} "
            "and classified into 5 content views with a canonical L1/L2 taxonomy "
            "(plus by_claim_type and by_time_period). The full SQLite database "
            "(askchem.db) is provided for fast local serving via the "
            "AskChem reference server. Live API: https://askchem.org."
        ),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    conn.close()

    # Optionally copy the SQLite database file itself for fast local serving.
    if include_db:
        print(f"\nCopying {DB_FILE} ({DB_PATH.stat().st_size/1e9:.2f} GB) "
              f"into upload dir (hardlink if possible)...", flush=True)
        dest_db = output_dir / DB_FILE
        try:
            os.link(DB_PATH, dest_db)
        except OSError:
            shutil.copy2(DB_PATH, dest_db)

    # Optionally copy the v2 retrieval artefacts. Staged via os.link so
    # we don't double the on-disk footprint of a 10 GB FAISS index on
    # the dev machine; falls back to copy across filesystems.
    if include_v2_embeddings:
        if include_v2_embeddings == "runtime":
            v2_files = V2_ARTEFACTS_RUNTIME
        elif include_v2_embeddings == "full":
            v2_files = V2_ARTEFACTS_FULL
        else:
            raise ValueError(
                f"include_v2_embeddings must be 'runtime', 'full', or None; "
                f"got {include_v2_embeddings!r}"
            )
        missing = [p for p in v2_files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing v2 retrieval artefact(s); cannot package "
                f"(set include_v2_embeddings=None to skip): "
                + ", ".join(str(p) for p in missing)
            )
        v2_dir = output_dir / "embeddings_v2"
        v2_dir.mkdir(exist_ok=True)
        total_v2 = 0
        for src in v2_files:
            dest = v2_dir / src.name
            size = src.stat().st_size
            total_v2 += size
            print(f"  staging {src.name} ({size/1e9:.2f} GB) ...",
                  flush=True)
            # Resolve symlinks first — claim_embeddings.v2_256.claim_ids.npy
            # is a symlink to claim_embeddings.v2.claim_ids.npy locally so
            # we don't store 600 MB twice on disk; but HF's upload_folder
            # follows the link target name, which produces a confusing
            # artefact tree on the remote. Hardlink against the resolved
            # real path so the destination filename matches the planned
            # remote name.
            real_src = src.resolve()
            try:
                os.link(real_src, dest)
            except OSError:
                shutil.copy2(real_src, dest)
        print(f"v2 retrieval artefacts staged: "
              f"{len(v2_files)} files, {total_v2/1e9:.2f} GB",
              flush=True)

    # Optionally include the paper-level Gemini classification map.
    if PAPER_CLASS_PATH.exists():
        shutil.copy2(PAPER_CLASS_PATH, output_dir / "paper_classifications.json")
        print(f"Copied paper_classifications.json "
              f"({PAPER_CLASS_PATH.stat().st_size/1e6:.1f} MB)", flush=True)

    edition_title = ("Abstract Edition" if abstract_only
                     else "Full Edition (Abstract + Deep Full-Paper)")
    edition_blurb = (
        "extracted from paper abstracts using gpt-5-mini"
        if abstract_only
        else "extracted from both paper abstracts (gpt-5-mini) and full PDFs "
             "(Gemini 3.1 Pro via Vertex AI Batch)"
    ) + f" and classified into {view_count} simultaneous hierarchical views."
    db_size_gb = f"{DB_PATH.stat().st_size / 1e9:.2f} GB" if include_db else "n/a"
    v2_section = ""
    if include_v2_embeddings:
        faiss_gb = V2_FAISS_PATH.stat().st_size / 1e9
        faiss256_gb = V2_FAISS_256_PATH.stat().st_size / 1e9
        ids_mb = V2_IDS_PATH.stat().st_size / 1e6
        v2_section += (
            f"- `embeddings_v2/claim_embeddings.v2.faiss` -- "
            f"FAISS IndexFlatIP over mxbai-embed-large-v1 CLS embeddings "
            f"at the native 1024-d "
            f"({faiss_gb:.1f} GB; LFS) -- best quality, requires ≥ 16 GB RAM "
            f"to keep resident\n"
            f"- `embeddings_v2/claim_embeddings.v2_256.faiss` -- "
            f"Matryoshka-truncated 256-d FAISS IndexFlatIP "
            f"({faiss256_gb:.1f} GB; LFS) -- ~5 nDCG@10 point recall loss, "
            f"4x smaller, what we ship to the 8 GB VPS\n"
            f"- `embeddings_v2/claim_embeddings.v2.claim_ids.npy` -- "
            f"Row-aligned claim-id sidecar shared by both indices "
            f"(memory-mapped at load time, {ids_mb:.0f} MB)\n"
        )
        if include_v2_embeddings == "full":
            npz_gb = V2_NPZ_PATH.stat().st_size / 1e9
            v2_section += (
                f"- `embeddings_v2/claim_embeddings.v2.npz` -- "
                f"Source npz (CLS-pooled, fp32, 1024-d) used to rebuild "
                f"alternative FAISS indices (Matryoshka 256/384/512/768, "
                f"HNSW, etc.) -- {npz_gb:.1f} GB; LFS\n"
            )
    files_section = (
        "- `askchem.db` -- Full SQLite database "
        f"(~{db_size_gb}, includes FTS5 indexes; LFS)\n" if include_db else ""
    ) + (
        "- `claims.jsonl` -- Every claim, one JSON object per line\n"
        "- `sources.jsonl` -- Source paper metadata\n"
        "- `hierarchy/` -- Per-view tree structure (flattened nodes)\n"
        "- `metadata.json` -- Dataset statistics\n"
    ) + (
        "- `paper_classifications.json` -- Paper-level Gemini view-path "
        "assignments (used to build the trees)\n"
        if PAPER_CLASS_PATH.exists() else ""
    ) + v2_section

    extra_tag = "  - abstract-extraction" if abstract_only else "  - full-paper-extraction"
    size_cat = "100K<n<1M" if abstract_only else "1M<n<10M"
    note_blurb = (
        "> **Full-paper extraction** -- with additional claim types (limitations, "
        "surprising findings, hypotheses, scope entries, future directions) -- is "
        "available through the [AskChem API](https://askchem.org)."
        if abstract_only else
        "> The downloadable `askchem.db` is the same SQLite file that powers the "
        "live [AskChem API](https://askchem.org), minus user-generated tables."
    )
    extra_claim_types = "" if abstract_only else (
        "- **limitation** -- Acknowledged limitations and caveats\n"
        "- **hypothesis** -- Research hypotheses and theoretical predictions\n"
        "- **surprising_finding** -- Unexpected or counterintuitive results\n"
        "- **scope_entry** -- Individual entries from substrate scope tables\n"
        "- **future_direction** -- Suggested future research directions\n"
        "- **experimental_design** -- Experimental design rationale\n"
        "- **structure** -- Structural characterization data"
    )
    readme = f"""---
license: cc-by-4.0
task_categories:
  - text-classification
  - question-answering
language:
  - en
pretty_name: AskChem
tags:
  - chemistry
  - knowledge-graph
  - scientific-claims
  - hierarchical-index
  - multi-view
{extra_tag}
size_categories:
  - {size_cat}
---

# AskChem: Structured Chemistry Knowledge Index ({edition_title})

A hierarchical, multi-view knowledge index for chemistry research.
Each entry is an **atomic knowledge claim** {edition_blurb}

{note_blurb}

## Dataset Statistics

| Metric | Count |
|--------|-------|
| Claims | {claim_count:,} |
| Source papers | {source_count:,} |
| Hierarchical views | {view_count} |
| Tree nodes | {total_nodes:,} |
| Extraction model | {ext_model} |
| Classification model | {cls_model} |

## Claim Types

- **reaction** -- Chemical transformations with reactants, products, conditions, outcomes
- **property** -- Measured or computed properties of substances
- **method** -- Experimental or computational techniques
- **mechanism** -- Mechanistic pathways and processes
- **comparison** -- Comparisons between methods, materials, or results
- **computational_result** -- Computational chemistry results
{extra_claim_types}

## Views

1. **by_reaction_type** -- {VIEWS[0][1]}
2. **by_substance_class** -- {VIEWS[1][1]}
3. **by_application** -- {VIEWS[2][1]}
4. **by_technique** -- {VIEWS[3][1]}
5. **by_mechanism** -- {VIEWS[4][1]}
6. **by_claim_type** -- {VIEWS[5][1]}
7. **by_data** -- {VIEWS[6][1]}
8. **by_time_period** -- {VIEWS[7][1]}

## Files

{files_section}

## Usage

```python
import json

# Load claims
claims = []
with open("claims.jsonl") as f:
    for line in f:
        claims.append(json.loads(line))

# Find all reaction claims in catalysis
reactions = [c for c in claims if c.get("claim_type") == "reaction"]

# Find claims about a specific molecule
suzuki = [c for c in claims if "suzuki" in c.get("verbatim_quote", "").lower()]
```

## AskChem API

For programmatic access to the full index (including deep full-paper claims),
use the REST API:

```bash
curl "https://askchem.org/api/search?q=suzuki+coupling&limit=5"
```

See [askchem.org](https://askchem.org) for full API documentation and
an MCP server for AI agent integration.

## Citation

```
@dataset{{askchem2026,
  title={{AskChem: An Open, Agent-Native Platform for Structured Chemical Knowledge}},
  author={{Yan, Bing and others}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{REPO_ID}}}
}}
```

If you use AskChem in academic work, please also cite the accompanying
system-demonstration paper (EMNLP 2026; details to follow) and the code
repository at <https://github.com/bingyan4science/structure_the_universe>.

```text
Note: the software is MIT-licensed; this dataset is released under CC-BY-4.0.
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    print(f"\nDataset packaged: {output_dir}", flush=True)
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB", flush=True)


def upload(output_dir: Path, abstract_only: bool,
           include_v2_embeddings: Optional[str] = None):
    api = HfApi()

    print(f"\nCreating/updating repo: {REPO_ID}", flush=True)
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    label = "abstract-only" if abstract_only else "full (abstract + deep)"
    if include_v2_embeddings:
        label += f" + v2-embeddings ({include_v2_embeddings})"
    print("Uploading (this can take a while for the full edition)...", flush=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=(
            f"Update AskChem index ({label}) -- "
            f"{datetime.now().strftime('%Y-%m-%d')}"
        ),
        # Large blobs (chemtree.db ~10 GB and v2 FAISS/npz ~10 GB each)
        # are pushed as LFS automatically.
    )
    print(f"\nUploaded to https://huggingface.co/datasets/{REPO_ID}", flush=True)


def sync_to_hf(abstract_only: bool, include_db: bool,
               include_v2_embeddings: Optional[str] = None,
               output_dir=None):
    """Package and upload the current index to HuggingFace."""
    print("\nSyncing to HuggingFace...", flush=True)
    cleanup = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="askchem_hf_"))
        cleanup = True
    try:
        package_dataset(output_dir, abstract_only=abstract_only,
                        include_db=include_db,
                        include_v2_embeddings=include_v2_embeddings)
        upload(output_dir, abstract_only=abstract_only,
               include_v2_embeddings=include_v2_embeddings)
    finally:
        if cleanup:
            shutil.rmtree(output_dir, ignore_errors=True)
    print("HuggingFace sync complete.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstract-only", action="store_true",
                    help="Publish only v3-abstract* claims (legacy behaviour)")
    ap.add_argument("--no-db", action="store_true",
                    help="Don't include chemtree.db in the upload "
                         "(JSONL exports only)")
    ap.add_argument(
        "--include-v2-embeddings",
        choices=("runtime", "full"),
        default=None,
        help=(
            "Stage and upload the v2 retrieval artefacts under "
            "embeddings_v2/. 'runtime' ships only the FAISS index + "
            "claim-id sidecar (~9.5 GB; what prod actually needs). "
            "'full' additionally ships the 10 GB source npz for "
            "reproducibility (~19.5 GB total)."
        ),
    )
    ap.add_argument("--package-only", metavar="DIR",
                    help="Just package to DIR; don't upload")
    ap.add_argument("--upload-only", metavar="DIR",
                    help="Skip packaging; upload an existing directory")
    args = ap.parse_args()

    print(f"{'='*60}", flush=True)
    label = "abstract-only" if args.abstract_only else "FULL (abstract + deep)"
    if args.include_v2_embeddings:
        label += f" + v2-embeddings ({args.include_v2_embeddings})"
    print(f"AskChem -> HuggingFace Upload [{label}]", flush=True)
    print(f"{'='*60}\n", flush=True)

    if args.upload_only:
        upload(Path(args.upload_only),
               abstract_only=args.abstract_only,
               include_v2_embeddings=args.include_v2_embeddings)
    elif args.package_only:
        package_dataset(Path(args.package_only),
                        abstract_only=args.abstract_only,
                        include_db=not args.no_db,
                        include_v2_embeddings=args.include_v2_embeddings)
    else:
        sync_to_hf(abstract_only=args.abstract_only,
                   include_db=not args.no_db,
                   include_v2_embeddings=args.include_v2_embeddings)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
