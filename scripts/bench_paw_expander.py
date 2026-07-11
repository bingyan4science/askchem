"""Component-level benchmark for chemistry query expansion.

Scores three expander systems against the hand-curated probe set at
``data/eval/paw_expander_probes.json``:

* ``paw_std``  — standard-compiler PAW expander, program
  ``d442088a6063deb9f42a`` ([src/askchem/paw_functions.py](src/askchem/paw_functions.py) L20).
* ``paw_ft``   — finetune-compiler expander from the May-23 compile,
  program ``fe558023bbc5acb6665b`` ([data/paw_ft_program_ids.json](data/paw_ft_program_ids.json)).
* ``static``   — pure-static dictionary lookup, replicating the
  ``CHEMISTRY_BIGRAM_SYNONYMS`` / ``CHEMISTRY_SYNONYMS`` /
  ``CHEMISTRY_FORMULAS`` path inside
  ``db.expand_query_variants`` ([src/askchem/db.py](src/askchem/db.py) L2633)
  with the original query tokens subtracted.

Each system is wrapped to a uniform ``Callable[[str], list[str]]``
contract. Outputs are scored with four metrics anchored to the
``gold_expand`` / ``gold_forbid`` sets:

* ``coverage``    — fraction of gold_expand hit by the output.
* ``pollution``   — fraction of output items containing any
  gold_forbid term.
* ``degeneracy``  — 1 - unique_tokens / total_tokens. Catches the
  Suzuki-spam looping the standard compiler exhibits.
* ``score``       — ``coverage - pollution - 0.5 * degeneracy``.
  Higher is better.

Matching is case-insensitive with non-alphanumeric boundary on both
sides so gold ``Pd`` hits ``Pd-catalyzed`` and ``palladium (Pd)`` but
not ``Padova`` (see ``data/eval/paw_expander_probes.json``'s
``match_rule`` field).

Usage:

    .venv-benchmark/bin/python scripts/bench_paw_expander.py \\
        --probes data/eval/paw_expander_probes.json \\
        --systems std,ft,static \\
        --out data/eval/runs/paw_expander_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

os.environ.setdefault("GGML_NO_METAL", "1")  # LoRA + Metal crashes; see paw_functions.py.

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── PAW program IDs ────────────────────────────────────────────────────────

# Match the standard-compiler IDs in src/askchem/paw_functions.py and the
# finetune-compiler IDs in data/paw_ft_program_ids.json. Both are loaded
# eagerly here so a typo surfaces as an ImportError, not a runtime PAW
# 404.
STD_EXPANDER_ID = "d442088a6063deb9f42a"
FT_EXPANDER_ID_DEFAULT = "fe558023bbc5acb6665b"


# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class Probe:
    id: str
    query: str
    family: str
    gold_expand: list[str]
    gold_forbid: list[str]
    notes: str = ""


@dataclass
class Metrics:
    coverage: float
    pollution: float
    degeneracy: float
    score: float
    n_output: int
    matched_expand: list[str] = field(default_factory=list)
    matched_forbid: list[str] = field(default_factory=list)


def load_probes(path: Path) -> list[Probe]:
    payload = json.loads(path.read_text())
    probes = payload["probes"]
    return [
        Probe(
            id=p["id"],
            query=p["query"],
            family=p["family"],
            gold_expand=list(p.get("gold_expand", [])),
            gold_forbid=list(p.get("gold_forbid", [])),
            notes=p.get("notes", ""),
        )
        for p in probes
    ]


# ── Matching ───────────────────────────────────────────────────────────────


_BOUNDARY = r"(?:^|[^a-z0-9])"
_END_BOUNDARY = r"(?:$|[^a-z0-9])"


def _term_pattern(gold: str) -> re.Pattern:
    """Compile a non-alphanumeric-boundary pattern for ``gold``.

    The boundary is intentionally looser than ``\\b`` so that punctuated
    chemistry tokens (``Pd-catalyzed``, ``MOF-5``, ``[4+2]``, ``Pd(II)``)
    count as boundaries.
    """
    return re.compile(
        _BOUNDARY + re.escape(gold.lower()) + _END_BOUNDARY,
        re.IGNORECASE,
    )


def _term_matches_any(output_term: str, gold_terms: list[str]) -> list[str]:
    """Return the gold terms that match ``output_term`` (case-insensitive)."""
    out = output_term.lower().strip()
    if not out:
        return []
    hits: list[str] = []
    for g in gold_terms:
        g_norm = g.lower().strip()
        if not g_norm:
            continue
        if g_norm == out:
            hits.append(g)
            continue
        if _term_pattern(g_norm).search(out):
            hits.append(g)
    return hits


def _output_contains_gold(output: list[str], gold: str) -> bool:
    return any(_term_matches_any(o, [gold]) for o in output)


# ── Metrics ────────────────────────────────────────────────────────────────


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(s)]


def _degeneracy(output: list[str]) -> float:
    """Lexical repetition score: 1 - unique tokens / total tokens.

    Computed over the full output (all items concatenated). A clean
    expansion list has degeneracy 0; the standard-compiler Suzuki spam
    has degeneracy ≈ 0.9.
    """
    toks: list[str] = []
    for item in output:
        toks.extend(_tokens(item))
    if not toks:
        return 0.0
    uniq = len(set(toks))
    return 1.0 - uniq / len(toks)


def score(probe: Probe, output: list[str]) -> Metrics:
    # coverage: which gold_expand terms appear anywhere in the output?
    matched_expand: list[str] = []
    for g in probe.gold_expand:
        if _output_contains_gold(output, g):
            matched_expand.append(g)
    coverage = (
        len(matched_expand) / len(probe.gold_expand) if probe.gold_expand else 0.0
    )

    # pollution: which output items contain any gold_forbid term?
    polluted_items: list[str] = []
    for item in output:
        if _term_matches_any(item, probe.gold_forbid):
            polluted_items.append(item)
    pollution = (
        len(polluted_items) / len(output) if output else 0.0
    )

    deg = _degeneracy(output)
    final = coverage - pollution - 0.5 * deg

    return Metrics(
        coverage=coverage,
        pollution=pollution,
        degeneracy=deg,
        score=final,
        n_output=len(output),
        matched_expand=matched_expand,
        matched_forbid=polluted_items[:8],
    )


# ── Adapters ───────────────────────────────────────────────────────────────


def _parse_paw_terms(raw: str) -> list[str]:
    """Parse a PAW expander output string into a deduplicated list of terms.

    Mirrors the cleaning ``paw_functions.expand_query`` does, but without
    the per-query dedup against the original query (so the bench can
    measure raw model output).
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = p.lower().strip()
        if not key or len(key) < 2:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
        if len(out) >= 30:  # safety cap for degenerate outputs
            break
    return out


def make_paw_adapter(program_id: str, n_gpu_layers: int = 0):
    import programasweights as paw

    fn = paw.function(program_id, n_gpu_layers=n_gpu_layers)

    def _expand(query: str) -> list[str]:
        raw = fn(query) or ""
        return _parse_paw_terms(raw.strip())

    return _expand


def make_static_adapter():
    """Replicate the static-dict expansion path from ``db.expand_query_variants``.

    Returns the raw list of *added* terms (one per dict entry) rather
    than a concatenated query string, so per-item metrics are
    apples-to-apples with the PAW adapters. Logic mirrors
    [src/askchem/db.py](../src/askchem/db.py) L2654-2670.
    """
    from askchem.db import (
        CHEMISTRY_BIGRAM_SYNONYMS,
        CHEMISTRY_SYNONYMS,
        CHEMISTRY_FORMULAS,
        _normalize_query_text,
        _PUNCT_STRIP,
    )

    def _expand(query: str) -> list[str]:
        norm = _normalize_query_text(query).strip().lower()
        words = [w.strip(_PUNCT_STRIP) for w in norm.split() if w.strip(_PUNCT_STRIP)]
        seen = {w for w in words}
        out: list[str] = []

        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            for t in CHEMISTRY_BIGRAM_SYNONYMS.get(bigram, [])[:8]:
                k = t.lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    out.append(t)

        for w in words:
            for t in CHEMISTRY_SYNONYMS.get(w, [])[:3]:
                k = t.lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    out.append(t)
            for t in CHEMISTRY_FORMULAS.get(w, [])[:3]:
                k = t.lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    out.append(t)
        return out

    return _expand


def build_systems(
    selected: list[str], ft_program_id: str,
    variants_registry: Path | None = None,
) -> dict[str, Callable[[str], list[str]]]:
    systems: dict[str, Callable[[str], list[str]]] = {}
    if "std" in selected:
        print(f"Loading paw_std ({STD_EXPANDER_ID})...", flush=True)
        systems["paw_std"] = make_paw_adapter(STD_EXPANDER_ID)
    if "ft" in selected:
        print(f"Loading paw_ft  ({ft_program_id})...", flush=True)
        systems["paw_ft"] = make_paw_adapter(ft_program_id)
    if "static" in selected:
        print("Loading static  (db.expand_query_variants)...", flush=True)
        systems["static"] = make_static_adapter()

    if variants_registry is not None:
        # data/paw_expander_variants.json contributes a system per
        # registry entry (name -> program_id). Used by Stage 1 and
        # Stage 2 of the spec-iteration plan to bench many variants in
        # one run alongside the static + ft baselines.
        try:
            payload = json.loads(variants_registry.read_text())
        except Exception as exc:
            print(f"WARN: could not read variants registry {variants_registry}: "
                  f"{exc}", file=sys.stderr)
            payload = {"variants": []}
        for entry in payload.get("variants", []):
            name = entry.get("name")
            pid = entry.get("program_id")
            if not name or not pid:
                continue
            if name in systems:
                print(f"WARN: variant {name!r} clashes with an existing "
                      "system; skipping", file=sys.stderr)
                continue
            print(f"Loading variant {name} ({pid})...", flush=True)
            systems[name] = make_paw_adapter(pid)
    return systems


# ── Driver ─────────────────────────────────────────────────────────────────


@dataclass
class ProbeRow:
    probe_id: str
    family: str
    query: str
    system: str
    output: list[str]
    metrics: Metrics
    latency_ms: float


def run(probes: list[Probe],
        systems: dict[str, Callable[[str], list[str]]],
        warmup: bool = True) -> list[ProbeRow]:
    if warmup:
        # PAW programs need a warm pass before timing is meaningful; the
        # llama-cpp prefix-cache load on the first call is ~0.3-1 s on
        # CPU. Static expander is a no-op here but stays in the loop for
        # symmetry.
        for name, fn in systems.items():
            try:
                fn("warmup query")
            except Exception as exc:
                print(f"  WARN: {name} warmup raised {exc!r}", file=sys.stderr)

    rows: list[ProbeRow] = []
    for i, probe in enumerate(probes, 1):
        for name, fn in systems.items():
            t0 = time.perf_counter()
            try:
                out = fn(probe.query)
            except Exception as exc:
                print(f"  [{i:>2}/{len(probes)}] {probe.id:<22} {name:<8} "
                      f"FAILED: {exc!r}", file=sys.stderr)
                out = []
            latency = (time.perf_counter() - t0) * 1000
            m = score(probe, out)
            rows.append(ProbeRow(
                probe_id=probe.id, family=probe.family, query=probe.query,
                system=name, output=out, metrics=m, latency_ms=latency,
            ))
        if i <= 3 or i % 5 == 0 or i == len(probes):
            print(f"  [{i:>2}/{len(probes)}] {probe.id:<22} {probe.family:<10}  "
                  + " ".join(
                      f"{r.system}={r.metrics.score:+.2f}"
                      for r in rows[-len(systems):]
                  ),
                  flush=True)
    return rows


# ── Reporting ──────────────────────────────────────────────────────────────


def _fmt_pct(x: float) -> str:
    return f"{x:5.2f}"


def report(rows: list[ProbeRow]) -> None:
    by_system: dict[str, list[ProbeRow]] = defaultdict(list)
    for r in rows:
        by_system[r.system].append(r)

    families = sorted({r.family for r in rows})
    systems = sorted(by_system.keys())

    print()
    print("=" * 80)
    print("Overall (macro-averaged across families)")
    print("=" * 80)
    print(f"  {'system':<10} {'cov':>7} {'pol':>7} {'deg':>7} {'score':>7}  "
          f"{'avg ms':>8}")
    overall_for_compare: dict[str, dict[str, float]] = {}
    for sysname in systems:
        sysrows = by_system[sysname]
        # Macro-average over families: avoids small families being drowned.
        by_fam: dict[str, list[ProbeRow]] = defaultdict(list)
        for r in sysrows:
            by_fam[r.family].append(r)

        def _macro(metric):
            fam_means = [
                statistics.mean(metric(r) for r in rs)
                for rs in by_fam.values()
                if rs
            ]
            return statistics.mean(fam_means) if fam_means else 0.0

        cov = _macro(lambda r: r.metrics.coverage)
        pol = _macro(lambda r: r.metrics.pollution)
        deg = _macro(lambda r: r.metrics.degeneracy)
        sc = _macro(lambda r: r.metrics.score)
        ms = statistics.mean(r.latency_ms for r in sysrows)
        overall_for_compare[sysname] = {
            "coverage": cov, "pollution": pol, "degeneracy": deg, "score": sc,
            "latency_ms": ms,
        }
        print(f"  {sysname:<10} "
              f"{_fmt_pct(cov):>7} {_fmt_pct(pol):>7} {_fmt_pct(deg):>7} "
              f"{_fmt_pct(sc):>7}  {ms:>7.0f} ")

    print()
    print("=" * 80)
    print("Per-family score (higher = better)")
    print("=" * 80)
    header = f"  {'family':<10} {'n':>4}  " + "  ".join(
        f"{sysname:>10}" for sysname in systems
    )
    print(header)
    for fam in families:
        cells = []
        n = 0
        for sysname in systems:
            sysrows = [r for r in by_system[sysname] if r.family == fam]
            n = len(sysrows)
            if not sysrows:
                cells.append("       n/a")
            else:
                m = statistics.mean(r.metrics.score for r in sysrows)
                cells.append(f"{m:>+10.3f}")
        print(f"  {fam:<10} {n:>4}  " + "  ".join(cells))

    # Per-metric per-family for the coverage/pollution split.
    for metric_name, getter in [
        ("coverage", lambda r: r.metrics.coverage),
        ("pollution", lambda r: r.metrics.pollution),
        ("degeneracy", lambda r: r.metrics.degeneracy),
    ]:
        print()
        print("=" * 80)
        print(f"Per-family {metric_name} (cov: higher better; pol/deg: lower better)")
        print("=" * 80)
        print(header)
        for fam in families:
            cells = []
            n = 0
            for sysname in systems:
                sysrows = [r for r in by_system[sysname] if r.family == fam]
                n = len(sysrows)
                if not sysrows:
                    cells.append("       n/a")
                else:
                    m = statistics.mean(getter(r) for r in sysrows)
                    cells.append(f"{m:>10.3f}")
            print(f"  {fam:<10} {n:>4}  " + "  ".join(cells))

    # Biggest-delta queries: for each system pair, the 5 probes with the
    # largest score gap (positive = first system better).
    if len(systems) >= 2:
        print()
        print("=" * 80)
        print("Biggest score deltas (per system pair)")
        print("=" * 80)
        # Group rows by probe_id so we can diff aligned.
        by_pid: dict[str, dict[str, ProbeRow]] = defaultdict(dict)
        for r in rows:
            by_pid[r.probe_id][r.system] = r
        for a_idx, a in enumerate(systems):
            for b in systems[a_idx + 1:]:
                deltas: list[tuple[float, str, str, str]] = []
                for pid, m in by_pid.items():
                    if a in m and b in m:
                        d = m[a].metrics.score - m[b].metrics.score
                        deltas.append((d, pid, m[a].query, m[a].family))
                deltas.sort(reverse=True)
                print()
                print(f"  Top probes where {a} > {b}")
                for d, pid, q, fam in deltas[:5]:
                    if d <= 0:
                        break
                    print(f"    {d:+.3f}  {pid:<22} {fam:<10} {q!r}")
                print(f"  Top probes where {b} > {a}")
                for d, pid, q, fam in deltas[-5:][::-1]:
                    if d >= 0:
                        break
                    print(f"    {-d:+.3f}  {pid:<22} {fam:<10} {q!r}")


def write_json(rows: list[ProbeRow], out_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": [
            {
                "probe_id": r.probe_id,
                "family": r.family,
                "query": r.query,
                "system": r.system,
                "output": r.output,
                "metrics": asdict(r.metrics),
                "latency_ms": round(r.latency_ms, 2),
            }
            for r in rows
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote per-probe metrics to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", type=Path,
                    default=REPO_ROOT / "data/eval/paw_expander_probes.json")
    ap.add_argument("--systems", default="std,ft,static",
                    help="Comma-separated subset of {std, ft, static}.")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/eval/runs/paw_expander_bench.json")
    ap.add_argument("--ft-program-id", default=FT_EXPANDER_ID_DEFAULT,
                    help="PAW finetune-compiler program id (default: the "
                    "May-23 compile).")
    ap.add_argument("--variants-registry", type=Path, default=None,
                    help="Path to a variants registry written by "
                    "scripts/compile_paw_expander_sweep.py. Each entry "
                    "becomes a system in the run; baselines requested "
                    "via --systems are still loaded too.")
    args = ap.parse_args()

    probes = load_probes(args.probes)
    print(f"Loaded {len(probes)} probes from {args.probes}")

    selected = [s.strip() for s in args.systems.split(",") if s.strip()]
    invalid = [s for s in selected if s not in {"std", "ft", "static"}]
    if invalid:
        print(f"ERROR: unknown systems: {invalid}", file=sys.stderr)
        return 2

    systems = build_systems(
        selected, args.ft_program_id,
        variants_registry=args.variants_registry,
    )
    if not systems:
        print("ERROR: no systems selected.", file=sys.stderr)
        return 2

    rows = run(probes, systems)
    report(rows)
    write_json(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
