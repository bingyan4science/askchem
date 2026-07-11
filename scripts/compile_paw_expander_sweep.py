"""Compile a sweep of chemistry expander spec variants.

Each variant lives as a plain-text spec file under
``data/paw_specs/expander/V*.txt``. This driver takes a list of
variant names (the file stems) and a compiler alias, compiles each on
the PAW server, polls for asset readiness, runs a 1-shot sanity probe,
and appends the result to a JSON registry that the benchmark harness
([scripts/bench_paw_expander.py](scripts/bench_paw_expander.py))
consumes via ``--variants-registry``.

The default registry path is ``data/paw_expander_variants.json``. The
file is read on start, the new entries are appended, and the file is
re-written after every successful compile so a late failure doesn't
lose earlier wins (same pattern as
[scripts/compile_paw_ft.py](scripts/compile_paw_ft.py)).

Compiler aliases:

* ``std`` -> ``paw-4b-qwen3-0.6b`` (mapper LoRA, ~30 s per compile;
  fast iteration loop for Stage 1 of the spec-iteration plan).
* ``ft``  -> ``paw-ft-bs48-20260522`` (finetune LoRA, ~3 min per
  compile; promotion compiler for Stage 2).

Usage::

    .venv-benchmark/bin/python scripts/compile_paw_expander_sweep.py \\
        --specs V0 V1 V2 V3 V4 V5 V6 V7 V8 \\
        --compiler std
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
DEFAULT_SPEC_DIR = REPO_ROOT / "data" / "paw_specs" / "expander"
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
    """Mirror of scripts/compile_paw_ft.py compile_with_retry.

    Compiles a spec with the configured compiler, retrying on transient
    HTTP errors. Caches the program ID so an exception during the load
    poll doesn't trigger a second compile (which would generate a fresh
    LoRA on the server).
    """
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
            msg_lower = msg.lower()
            transient = (
                "500" in msg
                or "503" in msg
                or "timeout" in msg_lower
                or "timed out" in msg_lower
                or "read operation" in msg_lower
                or "connection" in msg_lower
            )
            if not transient or attempt == max_attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    compile failed ({msg!r}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"compile retries exhausted: {last_error}")


def load_with_retry(program_id: str, max_attempts: int = 12):
    """Mirror of scripts/compile_paw_ft.py load_with_retry."""
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


def sanity_check(fn) -> tuple[bool, str]:
    """1-shot probe on a high-signal query.

    Returns (ok, raw_output). ``ok`` is True if the output is
    comma-separated, has at least 3 unique terms, and contains at least
    one of {Pd, palladium, Suzuki, cross-coupling} (since the prompt
    asks about Suzuki coupling). This catches catastrophic failures
    (empty output, immediate EOS) without enforcing per-spec quality
    which is the bench's job.
    """
    raw = (fn("Suzuki coupling") or "").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    uniq = {p.lower() for p in parts}
    has_signal = any(
        any(k in p.lower() for k in ("pd", "palladium", "suzuki", "cross"))
        for p in parts
    )
    ok = len(uniq) >= 3 and has_signal
    return ok, raw[:160]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", nargs="+", required=True,
                    help="Variant names (file stems under "
                    "data/paw_specs/expander/, e.g. V0 V1 V2).")
    ap.add_argument("--compiler", choices=sorted(COMPILER_ALIASES.keys()),
                    required=True, help="Compiler alias.")
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Recompile even if (name, compiler) is already in "
                    "the registry.")
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
            print(f"!! {name}: spec file {spec_path} not found, skipping")
            summary.append((name, "MISSING", str(spec_path)))
            continue

        if not args.force and (name, compiler_name) in existing:
            entry = existing[(name, compiler_name)]
            print(f"== {name}: already in registry as {entry['program_id']}, "
                  f"use --force to recompile")
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
        }

        # Persist BEFORE the sanity probe so a slow-loading program is
        # still captured; the bench will surface any load issues.
        # Replace any pre-existing same-name entry (covers --force).
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
                fn("warmup")
            except Exception:
                pass
            ok, snippet = sanity_check(fn)
            status = "OK" if ok else "WARN"
            summary.append((name, status, program_id))
            print(f"   [{status}] sanity output: {snippet!r}")
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
