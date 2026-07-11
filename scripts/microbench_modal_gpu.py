"""Stage-level micro-benchmark: CPU/MPS vs Modal GPU for embed and rerank.

The full eval_search_live.py harness is dominated by cold-cache SQLite
reads on a 16 GB Mac (13 GB chemtree.db barely fits in OS page cache).
This script bypasses SQLite entirely and times only the two stages that
GPU offload affects.

Procedure
---------
For 20 representative queries from data/eval/probes_v1.jsonl:

  embed_query(): 3 runs each, report median latency for CPU/MPS path
                 vs Modal /embed path. Verify ranks (cosine sims) match.

  cross_rerank(): pick 30 random claims as candidates per query, run
                  3 trials each, report median latency for CPU/MPS path
                  vs Modal /rerank path. Verify resulting score ordering
                  matches to within Kendall tau >= 0.99.

Output
------
Pretty-printed table to stdout, plus
data/eval/runs/modal-gpu-microbench.json with full per-query numbers.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EMBED_URL_GPU = "https://by2192--askchem-search-gpu-searchgpu-embed-v1.modal.run"
RERANK_URL_GPU = "https://by2192--askchem-search-gpu-searchgpu-rerank-v1.modal.run"
EMBED_URL_CPU = "https://by2192--askchem-search-gpu-searchcpu-embed-v1.modal.run"
RERANK_URL_CPU = "https://by2192--askchem-search-gpu-searchcpu-rerank-v1.modal.run"
TOKEN_FILE = Path("/tmp/askchem_modal_token.txt")

# 20 representative queries from the canonical probe set, mixed across
# reaction / material / property / homonym / technique / multi families.
PROBE_QUERIES = [
    "Suzuki-Miyaura cross-coupling",
    "Buchwald-Hartwig amination",
    "olefin metathesis",
    "click chemistry",
    "asymmetric hydrogenation",
    "MOF synthesis",
    "perovskite solar cell efficiency",
    "graphene oxide reduction",
    "lithium ion battery cathode",
    "CO2 electrochemical reduction",
    "band gap engineering",
    "thermal conductivity 2D materials",
    "photocatalytic water splitting",
    "DFT calculations band gap",
    "Suzuki coupling mechanism",  # homonym-adjacent
    "spin coupling NMR",
    "Pd catalyst loading optimization",
    "aryl chloride amination",
    "boronic acid coupling partner",
    "heteroaryl chloride",
]

CANDIDATE_CLAIM_TEXTS = [
    "Pd(PPh3)4 catalyses the Suzuki-Miyaura coupling of aryl bromides with arylboronic acids in dioxane at 80 °C, delivering biaryl products in 85-95% yield.",
    "Nickel-catalysed amination of heteroaryl chlorides with primary amines using a bulky NHC ligand affords secondary amines at 130 °C in 78-92% yield.",
    "Grubbs second-generation catalyst performs ring-closing metathesis of dienes at 0.5 mol% loading in CH2Cl2 at 40 °C with 90% conversion.",
    "Copper-catalysed azide-alkyne cycloaddition (CuAAC) at room temperature in t-BuOH/H2O 1:1 gives 1,4-disubstituted triazoles in 95% yield.",
    "An iridium-PNP catalyst hydrogenates ketones at 0.05 mol% loading and 50 bar H2 in iPrOH at 60 °C, delivering chiral alcohols in 99% ee.",
    "ZIF-8 metal-organic framework forms in methanol at room temperature from Zn(NO3)2 and 2-methylimidazole; BET surface area 1450 m2/g.",
    "Mixed-cation perovskite (FAPbI3/MAPbBr3) solar cells with SnO2 electron transport layer achieve 24.6% power conversion efficiency under AM1.5G illumination.",
    "Hydrazine-reduced graphene oxide displays sheet resistance of 8.4 ohm/sq at 90% transparency after thermal annealing at 250 °C in argon.",
    "LiNi0.8Co0.1Mn0.1O2 (NMC811) cathode retains 87% of initial capacity after 500 cycles at C/3 between 2.5-4.3 V in 1 M LiPF6 EC/DMC electrolyte.",
    "Copper foil electrodes reduce CO2 to ethylene at -0.95 V vs RHE with 45% Faradaic efficiency in 0.1 M KHCO3 saturated with CO2.",
    "TiO2 anatase phase has an indirect band gap of 3.20 eV at 0 K, narrowing to 2.95 eV upon nitrogen doping at 5 mol%.",
    "Single-layer MoS2 exhibits in-plane thermal conductivity of 34.5 W/m/K measured by Raman thermometry on suspended flakes.",
    "Cocatalysed TiO2 with 1 wt% Pt produces H2 at 1.8 mmol/h/g under 365 nm UV irradiation in methanol/water solution.",
    "HSE06 hybrid functional reproduces experimental band gap of GaAs (1.42 eV) within 0.05 eV; PBE underestimates by 1.0 eV.",
    "Standard textbook description of the Suzuki coupling: Pd(0) inserts into the aryl halide C-X bond, transmetalation by Ar-B(OH)2 in the presence of base.",
    "Hyperfine coupling constants in EPR spectra of TEMPO radicals show aN = 15.5 G in nonpolar solvents.",
    "Optimised Pd-BippyPhos catalyst loading of 0.5 mol% with K3PO4 (3 equiv) in toluene at 100 °C affords aminated heteroaryl chlorides in 84-95% yield.",
    "Aryl chloride amination with morpholine using Pd2(dba)3/BippyPhos in toluene at 100 °C overnight (22 h) affords 84% isolated yield.",
    "Pinacol boronic ester partners (ArBpin) are typically more stable than free boronic acids for Suzuki couplings of electron-poor aryl chlorides.",
    "Heteroaryl chlorides bearing a 3-pyridyl group undergo amination with NHCs at 130 °C in 1,4-dioxane with K2CO3 base.",
    # 20 more for the 30-candidate batch
    "BippyPhos as a single-component ligand for Pd-catalysed amination at 50 °C in toluene with K3PO4 base.",
    "Hydrothermal carbonisation of glucose at 180 °C for 12 hours produces porous carbon with surface area 750 m2/g.",
    "Solid-state Li7La3Zr2O12 garnet electrolyte exhibits ionic conductivity of 0.5 mS/cm at 25 °C.",
    "Quantum dot light-emitting diodes (QLED) achieve 28% external quantum efficiency at 540 nm using CdSe/CdS core-shell.",
    "Cu(II)/PTABS phosphaadamantane in water enables Buchwald-Hartwig amination at room temperature with 84-99% yields.",
    "Nickel(II)/silane reduction system catalyses C-N coupling of chloropyridines at 130 °C with PMHS (silane reductant) and NaOtBu base.",
    "Ru-PyBOX hydrogenation of ketones at 80 °C and 30 bar H2 with NaOiPr base yields chiral alcohols at 90-94% ee.",
    "Photocatalytic Z-scheme using BiVO4 and Rh-doped SrTiO3 produces O2 and H2 at apparent quantum yield of 33% at 419 nm.",
    "Pd(OAc)2/XPhos with K3PO4 in 1,4-dioxane at 100 °C accomplishes Buchwald-Hartwig amination of chloropyrimidines.",
    "Crawford et al. (2013) reports BippyPhos as a superior monodentate phosphine for amination of 2-chloropyridines at 50 °C.",
]

random.seed(0)


def _percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[idx]


def _bench_embed():
    print("\n=== Embed benchmark (20 queries x 3 trials) ===")
    from askchem import retrieval

    import numpy as np
    from askchem import embeddings_v2 as _ev2

    def _run_path(label, url):
        if url:
            os.environ["CHEMTREE_REMOTE_EMBED_URL"] = url
            os.environ["CHEMTREE_REMOTE_AUTH_TOKEN"] = TOKEN_FILE.read_text().strip()
            os.environ["CHEMTREE_REMOTE_TIMEOUT_S"] = "30"
        else:
            os.environ.pop("CHEMTREE_REMOTE_EMBED_URL", None)
        _ev2._query_cache.clear()
        lats, vecs = [], {}
        for q in PROBE_QUERIES:
            for _ in range(3):
                t0 = time.monotonic()
                retrieval.embed_query(q)
                lats.append(1000 * (time.monotonic() - t0))
                _ev2._query_cache.clear()
            vecs[q] = retrieval.embed_query(q).copy()
            _ev2._query_cache.clear()
        return lats, vecs

    local_lats, local_vecs = _run_path("local", None)
    gpu_lats, gpu_vecs = _run_path("gpu", EMBED_URL_GPU)
    cpu_lats, cpu_vecs = _run_path("cpu", EMBED_URL_CPU)

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_gpu = [_cos(local_vecs[q], gpu_vecs[q]) for q in PROBE_QUERIES]
    cos_cpu = [_cos(local_vecs[q], cpu_vecs[q]) for q in PROBE_QUERIES]

    def _line(label, lats):
        print(f"  {label:<13s} med={statistics.median(lats):>5.0f} ms  "
              f"p95={_percentile(lats, 95):>5.0f} ms  "
              f"min={min(lats):>5.0f} ms")

    _line("local CPU", local_lats)
    _line("Modal CPU", cpu_lats)
    _line("Modal GPU", gpu_lats)
    print(f"  speedup vs local CPU: "
          f"Modal CPU={statistics.median(local_lats)/max(statistics.median(cpu_lats),1e-9):.2f}x  "
          f"Modal GPU={statistics.median(local_lats)/max(statistics.median(gpu_lats),1e-9):.2f}x")
    print(f"  quality: cosine(local, Modal-GPU) mean={statistics.mean(cos_gpu):.4f} min={min(cos_gpu):.4f}")
    print(f"           cosine(local, Modal-CPU) mean={statistics.mean(cos_cpu):.4f} min={min(cos_cpu):.4f}")

    return {
        "local_lats_ms": local_lats,
        "modal_cpu_lats_ms": cpu_lats,
        "modal_gpu_lats_ms": gpu_lats,
        "cosines_gpu": cos_gpu,
        "cosines_cpu": cos_cpu,
    }


def _bench_rerank():
    print("\n=== Rerank benchmark (20 queries x 30 candidates x 3 trials) ===")
    from askchem import retrieval

    candidates_30 = CANDIDATE_CLAIM_TEXTS[:30]
    pairs = [(f"c{i:02d}", t) for i, t in enumerate(candidates_30)]

    def _run_path(url):
        if url:
            os.environ["CHEMTREE_REMOTE_RERANK_URL"] = url
            os.environ["CHEMTREE_REMOTE_AUTH_TOKEN"] = TOKEN_FILE.read_text().strip()
            os.environ["CHEMTREE_REMOTE_TIMEOUT_S"] = "30"
        else:
            os.environ.pop("CHEMTREE_REMOTE_RERANK_URL", None)
        lats, orderings = [], {}
        for q in PROBE_QUERIES:
            for _ in range(3):
                t0 = time.monotonic()
                ranked = retrieval.cross_rerank(q, pairs, top_k=30)
                lats.append(1000 * (time.monotonic() - t0))
            orderings[q] = [cid for cid, _ in ranked]
        return lats, orderings

    local_lats, local_ord = _run_path(None)
    gpu_lats, gpu_ord = _run_path(RERANK_URL_GPU)
    cpu_lats, cpu_ord = _run_path(RERANK_URL_CPU)

    from itertools import combinations

    def kendall_tau(a, b):
        idx = {c: i for i, c in enumerate(b)}
        pairs_ = list(combinations(a, 2))
        if not pairs_:
            return 1.0
        concord = 0
        for x, y in pairs_:
            if idx[x] < idx[y]:
                concord += 1
        return 2 * concord / len(pairs_) - 1

    taus_gpu = [kendall_tau(local_ord[q], gpu_ord[q]) for q in PROBE_QUERIES]
    taus_cpu = [kendall_tau(local_ord[q], cpu_ord[q]) for q in PROBE_QUERIES]
    top5_gpu = [len(set(local_ord[q][:5]) & set(gpu_ord[q][:5])) / 5 for q in PROBE_QUERIES]
    top5_cpu = [len(set(local_ord[q][:5]) & set(cpu_ord[q][:5])) / 5 for q in PROBE_QUERIES]

    def _line(label, lats):
        print(f"  {label:<13s} med={statistics.median(lats):>5.0f} ms  "
              f"p95={_percentile(lats, 95):>5.0f} ms  "
              f"min={min(lats):>5.0f} ms")

    _line("local CPU", local_lats)
    _line("Modal CPU", cpu_lats)
    _line("Modal GPU", gpu_lats)
    print(f"  speedup vs local CPU: "
          f"Modal CPU={statistics.median(local_lats)/max(statistics.median(cpu_lats),1e-9):.2f}x  "
          f"Modal GPU={statistics.median(local_lats)/max(statistics.median(gpu_lats),1e-9):.2f}x")
    print(f"  quality: Kendall tau vs local "
          f"Modal-GPU={statistics.mean(taus_gpu):.3f} (min {min(taus_gpu):.3f})  "
          f"Modal-CPU={statistics.mean(taus_cpu):.3f} (min {min(taus_cpu):.3f})")
    print(f"           top5 overlap            "
          f"Modal-GPU={statistics.mean(top5_gpu):.3f}                "
          f"Modal-CPU={statistics.mean(top5_cpu):.3f}")
    return {
        "local_lats_ms": local_lats,
        "modal_cpu_lats_ms": cpu_lats,
        "modal_gpu_lats_ms": gpu_lats,
        "kendall_taus_gpu": taus_gpu,
        "kendall_taus_cpu": taus_cpu,
        "top5_overlaps_gpu": top5_gpu,
        "top5_overlaps_cpu": top5_cpu,
    }


def main() -> None:
    os.environ["CHEMTREE_RETRIEVER_VERSION"] = "v2"
    os.environ["CHEMTREE_V2_DIM"] = "256"

    # Warm every Modal container so the timed loop never pays cold-start.
    import requests
    headers = {
        "X-Auth-Token": TOKEN_FILE.read_text().strip(),
        "Content-Type": "application/json",
    }
    print("warming Modal endpoints (4 cold starts max)...")
    for url, body in [
        (EMBED_URL_GPU, {"queries": ["warmup"]}),
        (RERANK_URL_GPU, {"query": "warm", "texts": ["warm"]}),
        (EMBED_URL_CPU, {"queries": ["warmup"]}),
        (RERANK_URL_CPU, {"query": "warm", "texts": ["warm"]}),
    ]:
        t0 = time.monotonic()
        r = requests.post(url, json=body, headers=headers, timeout=120)
        r.raise_for_status()
        print(f"  warmup {url.split('--')[1][:40]:<40s} {1000*(time.monotonic()-t0):.0f} ms")

    emb = _bench_embed()
    rrk = _bench_rerank()

    out_path = REPO_ROOT / "data" / "eval" / "runs" / "modal-gpu-microbench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"embed": emb, "rerank": rrk}, indent=2))
    print(f"\nwrote raw timings + quality data to {out_path}")


if __name__ == "__main__":
    main()
