#!/usr/bin/env python3
"""Diagnose exactly why Metal + LoRA crashes on this machine.

Tests each combination independently in subprocesses so one crash
doesn't kill the others. Narrows down whether the crash is:
  A) Metal + base model (no LoRA)
  B) Metal + LoRA load
  C) Metal + LoRA apply
  D) Metal + LoRA eval
  E) Metal + LoRA generation

Usage:
    /opt/homebrew/bin/python3.10 scripts/diagnose_metal.py
"""
import json
import os
import signal
import subprocess
import sys

PYTHON = sys.executable
MODEL = os.path.expanduser("~/.cache/programasweights/base_models/qwen3-0.6b-q6_k.gguf")
ADAPTER = os.path.expanduser("~/.cache/programasweights/programs/e1ae405a57318a3fb9db/adapter.gguf")


def run_test(label: str, code: str, timeout: int = 60) -> dict:
    """Run a test in a subprocess and capture result."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )

    stdout = result.stdout.strip()
    stderr_lines = result.stderr.strip().split("\n")
    # Filter out the noisy LoRA fallback warnings
    stderr_filtered = [
        l for l in stderr_lines
        if "CPU_REPACK" not in l and "n_ctx_seq" not in l and "DeprecationWarning" not in l and l.strip()
    ]

    if result.returncode == 0:
        print(f"  PASS (exit 0)")
        if stdout:
            for line in stdout.split("\n"):
                print(f"  {line}")
    elif result.returncode < 0:
        sig = -result.returncode
        signame = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
        print(f"  CRASH: signal {signame} ({sig})")
    else:
        print(f"  FAIL: exit code {result.returncode}")

    if stderr_filtered:
        print(f"  STDERR ({len(stderr_filtered)} lines):")
        for line in stderr_filtered[-10:]:
            print(f"    {line}")

    return {
        "label": label,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr_key": "\n".join(stderr_filtered[-5:]),
    }


def main():
    print(f"Python: {PYTHON}")
    print(f"Model: {MODEL} ({os.path.getsize(MODEL)/1e6:.0f} MB)")
    print(f"Adapter: {ADAPTER} ({os.path.getsize(ADAPTER)/1e6:.0f} MB)")

    results = []

    # Test A: Metal + base model only, no LoRA
    results.append(run_test(
        "A: Metal base model load + eval (NO LoRA)",
        f"""
import time
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
tokens = llm.tokenize(b"Hello world", add_bos=True)
t0 = time.time()
llm.eval(tokens)
tok = llm.sample(temp=0)
print(f"OK: eval+sample in {{time.time()-t0:.3f}}s, token={{tok}}")
""",
    ))

    # Test B: Metal + LoRA load (no apply, no eval)
    results.append(run_test(
        "B: Metal base model + LoRA load (no apply)",
        f"""
import llama_cpp
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
adapter = llama_cpp.llama_adapter_lora_init(llm.model, b"{ADAPTER}")
print(f"OK: adapter loaded, ptr={{adapter is not None}}")
""",
    ))

    # Test C: Metal + LoRA apply (old API)
    results.append(run_test(
        "C: Metal + LoRA apply (old API, no eval)",
        f"""
import llama_cpp
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
adapter = llama_cpp.llama_adapter_lora_init(llm.model, b"{ADAPTER}")
llama_cpp.llama_set_adapter_lora(llm.ctx, adapter, 1.0)
print("OK: adapter applied")
""",
    ))

    # Test D: Metal + LoRA apply + tokenize (no eval)
    results.append(run_test(
        "D: Metal + LoRA + tokenize (no eval)",
        f"""
import llama_cpp
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
adapter = llama_cpp.llama_adapter_lora_init(llm.model, b"{ADAPTER}")
llama_cpp.llama_set_adapter_lora(llm.ctx, adapter, 1.0)
tokens = llm.tokenize(b"Hello world", add_bos=True)
print(f"OK: tokenized {{len(tokens)}} tokens")
""",
    ))

    # Test E: Metal + LoRA + eval (1 token)
    results.append(run_test(
        "E: Metal + LoRA + eval 1 token",
        f"""
import llama_cpp
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
adapter = llama_cpp.llama_adapter_lora_init(llm.model, b"{ADAPTER}")
llama_cpp.llama_set_adapter_lora(llm.ctx, adapter, 1.0)
tokens = llm.tokenize(b"Hi", add_bos=True)
llm.eval(tokens)
print("OK: eval succeeded")
""",
    ))

    # Test F: Metal + LoRA + eval many tokens
    results.append(run_test(
        "F: Metal + LoRA + eval 50 tokens",
        f"""
import llama_cpp
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
adapter = llama_cpp.llama_adapter_lora_init(llm.model, b"{ADAPTER}")
llama_cpp.llama_set_adapter_lora(llm.ctx, adapter, 1.0)
text = "This is a longer input to test whether the crash depends on input length " * 3
tokens = llm.tokenize(text.encode(), add_bos=True)
print(f"Evaluating {{len(tokens)}} tokens...")
llm.eval(tokens)
print("OK: eval succeeded")
""",
    ))

    # Test G: Metal with partial GPU layers
    for layers in [1, 5, 14, 28]:
        results.append(run_test(
            f"G: Metal + LoRA + eval, n_gpu_layers={layers}",
            f"""
import llama_cpp
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers={layers}, verbose=False)
adapter = llama_cpp.llama_adapter_lora_init(llm.model, b"{ADAPTER}")
llama_cpp.llama_set_adapter_lora(llm.ctx, adapter, 1.0)
tokens = llm.tokenize(b"Hello world test", add_bos=True)
llm.eval(tokens)
tok = llm.sample(temp=0)
print(f"OK: eval+sample succeeded, token={{tok}}")
""",
        ))

    # Test H: Metal base model (no LoRA) with eval of many tokens — speed check
    results.append(run_test(
        "H: Metal base model speed (NO LoRA, 50 tokens)",
        f"""
import time
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=512, n_gpu_layers=-1, verbose=False)
text = "This is a longer input to test eval speed without any LoRA adapter applied " * 3
tokens = llm.tokenize(text.encode(), add_bos=True)
t0 = time.time()
llm.eval(tokens)
dt = time.time() - t0
print(f"OK: {{len(tokens)}} tokens in {{dt:.3f}}s = {{len(tokens)/dt:.1f}} tok/s")
""",
    ))

    # Summary
    print(f"\n\n{'#'*60}")
    print("SUMMARY")
    print(f"{'#'*60}")
    for r in results:
        if r["returncode"] == 0:
            status = "PASS"
        elif r["returncode"] < 0:
            sig = -r["returncode"]
            signame = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
            status = f"CRASH ({signame})"
        else:
            status = f"FAIL (exit {r['returncode']})"
        print(f"  {status:20s}  {r['label']}")


if __name__ == "__main__":
    main()
