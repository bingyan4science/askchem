"""Compile a sweep of chemistry-relevance reranker spec variants.

Same shape as
[scripts/compile_paw_expander_sweep.py](scripts/compile_paw_expander_sweep.py)
but targets the reranker specs at ``data/paw_specs/reranker/R*.txt``.
Writes to the same
[data/paw_expander_variants.json](data/paw_expander_variants.json)
registry (entries are namespaced by ``name`` so they don't collide with
expander variants).

The reranker contract is::

    QUERY: <query>  CLAIM: <claim text>

with output one of {not_relevant, somewhat_relevant, highly_relevant}
(or four labels for the R3 variant).

Usage::

    .venv-benchmark/bin/python scripts/compile_paw_reranker_sweep.py \\
        --specs R0 R1 R2 R3 R4 --compiler std
    .venv-benchmark/bin/python scripts/compile_paw_reranker_sweep.py \\
        --specs <top-3 from std> --compiler ft
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GGML_NO_METAL", "1")

import programasweights as paw

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC_DIR = REPO_ROOT / "data" / "paw_specs" / "reranker"
DEFAULT_REGISTRY = REPO_ROOT / "data" / "paw_expander_variants.json"

COMPILER_ALIASES = {
    "std": "paw-4b-qwen3-0.6b",
    "ft": "paw-ft-bs48-20260522",
}


def _load_registry(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"variants": []}
    return {"variants": []}


def _save_registry(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def compile_with_retry(spec: str, compiler: str, max_attempts: int = 5) -> str:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            t0 = time.time()
            result = paw.compile(spec, compiler=compiler)
            program_id = result.id if hasattr(result, "id") else str(result)
            elapsed = time.time() - t0
            print(f"    compiled in {elapsed:.1f}s, program_id={program_id}")
            return program_id
        except Exception as e:  # noqa: BLE001
            last_error = e
            msg = str(e)
            ml = msg.lower()
            transient = (
                "500" in msg or "503" in msg
                or "timeout" in ml or "timed out" in ml
                or "read operation" in ml or "connection" in ml
            )
            if not transient or attempt == max_attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    compile failed ({msg!r}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"compile retries exhausted: {last_error}")


def load_with_retry(program_id: str, max_attempts: int = 12):
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return paw.function(program_id, n_gpu_layers=0)
        except Exception as e:  # noqa: BLE001
            last_error = e
            wait = 5 * (attempt + 1)
            print(f"    load attempt {attempt + 1}/{max_attempts} failed: {e}; "
                  f"retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"load retries exhausted: {last_error}")


SANITY_PROBES = [
    # (input, valid_outputs)
    (
        "QUERY: Suzuki coupling palladium "
        "CLAIM: Pd(PPh3)4 catalyzed the Suzuki-Miyaura cross-coupling of aryl iodide "
        "with phenylboronic acid in 92% yield.",
        {"highly_relevant", "exact_match"},
    ),
    (
        "QUERY: Mannich reaction enantioselective "
        "CLAIM: The crystal structure of beta-galactosidase was determined at 2.1 "
        "Angstrom resolution.",
        {"not_relevant"},
    ),
    (
        "QUERY: MOF gas storage "
        "CLAIM: Metal-organic frameworks are crystalline porous materials with high "
        "surface area.",
        {"somewhat_relevant"},
    ),
]


def sanity_check(fn) -> tuple[int, int, list[str]]:
    """Return (passed, total, outputs) on the 3-probe sanity set."""
    passed = 0
    outputs: list[str] = []
    for inp, valid in SANITY_PROBES:
        raw = (fn(inp) or "").strip().lower()
        # Take the first comma/whitespace-bounded token to handle
        # occasional trailing text.
        token = raw.split(",")[0].split()[0] if raw else ""
        outputs.append(token)
        if token in valid:
            passed += 1
    return passed, len(SANITY_PROBES), outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--compiler", choices=sorted(COMPILER_ALIASES.keys()),
                    required=True)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    compiler_name = COMPILER_ALIASES[args.compiler]
    registry = _load_registry(args.registry)
    existing = {(v["name"], v["compiler"]): v for v in registry.get("variants", [])}

    print(f"Compiler:  {args.compiler} -> {compiler_name}")
    print(f"Spec dir:  {args.spec_dir}")
    print(f"Registry:  {args.registry}")
    print(f"Variants:  {args.specs}")
    print()

    summary: list[tuple[str, str, str]] = []
    for stem in args.specs:
        name = f"{stem}_{args.compiler}"
        spec_path = args.spec_dir / f"{stem}.txt"
        if not spec_path.exists():
            print(f"!! {name}: spec file {spec_path} not found")
            summary.append((name, "MISSING", str(spec_path)))
            continue

        if not args.force and (name, compiler_name) in existing:
            entry = existing[(name, compiler_name)]
            print(f"== {name}: already in registry as {entry['program_id']}")
            summary.append((name, "CACHED", entry["program_id"]))
            continue

        spec = spec_path.read_text()
        print(f"== {name}  ({len(spec)} chars)")

        try:
            program_id = compile_with_retry(spec, compiler_name)
        except Exception as e:
            print(f"   COMPILE FAILED: {e}")
            summary.append((name, "FAILED", str(e)[:80]))
            continue

        entry = {
            "name": name,
            "spec_stem": stem,
            "spec_file": str(spec_path.relative_to(REPO_ROOT)),
            "compiler": compiler_name,
            "compiler_alias": args.compiler,
            "program_id": program_id,
            "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "program_type": "reranker",
        }

        new_variants = [
            v for v in registry.get("variants", [])
            if not (v["name"] == name and v["compiler"] == compiler_name)
        ]
        new_variants.append(entry)
        registry["variants"] = new_variants
        _save_registry(args.registry, registry)

        if args.skip_sanity:
            summary.append((name, "OK", program_id))
            continue

        print("   loading for sanity probe...")
        try:
            fn = load_with_retry(program_id)
            try:
                fn("QUERY: warmup CLAIM: warmup")
            except Exception:
                pass
            passed, total, outputs = sanity_check(fn)
            status = f"{passed}/{total}"
            summary.append((name, status, program_id))
            print(f"   sanity {status} | outputs: {outputs}")
        except Exception as e:
            print(f"   load failed: {e}")
            summary.append((name, "LOAD_FAIL", program_id))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, status, info in summary:
        print(f"  {name:<14} {status:<10} {info}")

    print(f"\nRegistry written to {args.registry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
