#!/usr/bin/env python3
"""Diagnose PAW / llama-cpp-python issues layer by layer.

Tests each component independently using raw llama-cpp-python calls.
No PAW wrapper is used — this isolates llama-cpp from the PAW SDK.

Usage:
    /opt/homebrew/bin/python3.10 scripts/diagnose_paw.py
"""
import ctypes
import os
import sys
import time

MODEL_PATH = os.path.expanduser(
    "~/.cache/programasweights/base_models/qwen3-0.6b-q6_k.gguf"
)
ADAPTER_PATH = os.path.expanduser(
    "~/.cache/programasweights/programs/e1ae405a57318a3fb9db/adapter.gguf"
)
TEMPLATE_PATH = os.path.expanduser(
    "~/.cache/programasweights/programs/e1ae405a57318a3fb9db/prompt_template.txt"
)
KV_CACHE_PATH = "/tmp/diagnose_paw_kv_cache.bin"

def check_files():
    print("=" * 60)
    print("TEST 0: File existence and sizes")
    print("=" * 60)
    for label, path in [("Model", MODEL_PATH), ("Adapter", ADAPTER_PATH), ("Template", TEMPLATE_PATH)]:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = f"{size / 1e6:.1f} MB" if exists else "MISSING"
        print(f"  {label}: {status}  ({path})")
        if not exists:
            print(f"  FATAL: {label} not found")
            sys.exit(1)
    print()


def test_base_model_load(n_gpu_layers):
    print("=" * 60)
    print(f"TEST 1: Load base model (n_gpu_layers={n_gpu_layers})")
    print("=" * 60)
    from llama_cpp import Llama
    t0 = time.time()
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=n_gpu_layers, verbose=False)
        print(f"  OK: loaded in {time.time()-t0:.2f}s")
        return llm
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def test_lora_load(llm):
    print("=" * 60)
    print("TEST 2: Load LoRA adapter")
    print("=" * 60)
    import llama_cpp
    t0 = time.time()
    try:
        adapter = llama_cpp.llama_adapter_lora_init(
            llm.model, ADAPTER_PATH.encode("utf-8")
        )
        print(f"  OK: adapter={adapter is not None}, loaded in {time.time()-t0:.2f}s")
        return adapter
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def test_lora_apply_old_api(llm, adapter):
    print("=" * 60)
    print("TEST 3a: Apply LoRA — old API (llama_set_adapter_lora)")
    print("=" * 60)
    import llama_cpp
    try:
        llama_cpp.llama_set_adapter_lora(llm.ctx, adapter, 1.0)
        print("  OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_lora_apply_new_api(llm, adapter):
    print("=" * 60)
    print("TEST 3b: Apply LoRA — new API (llama_set_adapters_lora)")
    print("=" * 60)
    import llama_cpp
    if not hasattr(llama_cpp, "llama_set_adapters_lora"):
        print("  SKIP: llama_set_adapters_lora not available")
        return None
    try:
        adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(adapter)
        scales = (ctypes.c_float * 1)(1.0)
        llama_cpp.llama_set_adapters_lora(llm.ctx, adapters, 1, scales)
        print("  OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_prefix_eval(llm):
    print("=" * 60)
    print("TEST 4: Tokenize + eval prefix")
    print("=" * 60)
    template = open(TEMPLATE_PATH).read()
    prefix_text = template.split("{INPUT_PLACEHOLDER}")[0]
    suffix_text = template.split("{INPUT_PLACEHOLDER}")[1]
    prefix_tokens = llm.tokenize(prefix_text.encode("utf-8"), add_bos=False, special=True)
    print(f"  Prefix: {len(prefix_tokens)} tokens")
    t0 = time.time()
    try:
        llm.eval(prefix_tokens)
        print(f"  OK: prefix eval in {time.time()-t0:.2f}s")
        return prefix_tokens, suffix_text
    except Exception as e:
        print(f"  FAIL: {e}")
        return None, None


def test_kv_cache_save(llm, prefix_tokens):
    print("=" * 60)
    print("TEST 5a: Save prefix KV cache to disk")
    print("=" * 60)
    import llama_cpp
    try:
        token_array = (llama_cpp.llama_token * len(prefix_tokens))(*prefix_tokens)
        result = llama_cpp.llama_state_seq_save_file(
            llm.ctx,
            KV_CACHE_PATH.encode("utf-8"),
            0,
            token_array,
            len(prefix_tokens),
        )
        size = os.path.getsize(KV_CACHE_PATH) if os.path.exists(KV_CACHE_PATH) else 0
        print(f"  OK: result={result}, file={size/1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_kv_cache_load(llm, prefix_tokens):
    print("=" * 60)
    print("TEST 5b: Load prefix KV cache from disk")
    print("=" * 60)
    import llama_cpp
    if not os.path.exists(KV_CACHE_PATH):
        print("  SKIP: no cache file from 5a")
        return None
    try:
        token_array = (llama_cpp.llama_token * len(prefix_tokens))(*prefix_tokens)
        n_token_count = ctypes.c_size_t(0)
        n_loaded = llama_cpp.llama_state_seq_load_file(
            llm.ctx,
            KV_CACHE_PATH.encode("utf-8"),
            0,
            token_array,
            len(prefix_tokens),
            ctypes.byref(n_token_count),
        )
        print(f"  OK: n_loaded={n_loaded}, n_token_count={n_token_count.value}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_inference(llm, prefix_tokens, suffix_text):
    print("=" * 60)
    print("TEST 6: Warm inference (prefix cached in memory)")
    print("=" * 60)
    n_prefix = len(prefix_tokens)
    pairs = [
        ("NCA has high cost", "NCA is low cost"),
        ("MOFs have high surface area", "MOFs have tunable pore sizes"),
        ("LMBs offer long lifetime", "LMBs suffer from short lifetime"),
    ]
    for i, (a, b) in enumerate(pairs, 1):
        llm.n_tokens = n_prefix
        inp = f"CLAIM_A: {a} CLAIM_B: {b}" + suffix_text
        input_tokens = llm.tokenize(inp.encode("utf-8"), add_bos=False, special=True)
        t0 = time.time()
        llm.eval(input_tokens)
        t_eval = time.time() - t0
        output_tokens = []
        for _ in range(20):
            token = llm.sample(temp=0)
            if token == llm.token_eos():
                break
            output_tokens.append(token)
            llm.eval([token])
        result = llm.detokenize(output_tokens).decode("utf-8", errors="replace").strip()
        total = time.time() - t0
        print(f"  {i}. {result:12s} eval={t_eval:.2f}s total={total:.2f}s ({len(input_tokens)} input tokens)")
    print()


def run_suite(gpu_layers):
    """Run all tests for a given n_gpu_layers setting."""
    label = "GPU (Metal)" if gpu_layers == -1 else "CPU only"
    print(f"\n{'#' * 60}")
    print(f"# Running all tests with {label} (n_gpu_layers={gpu_layers})")
    print(f"{'#' * 60}\n")

    llm = test_base_model_load(gpu_layers)
    if llm is None:
        print(f"  Cannot continue with n_gpu_layers={gpu_layers}\n")
        return

    adapter = test_lora_load(llm)
    if adapter is None:
        print("  Cannot continue without adapter\n")
        del llm
        return

    ok_old = test_lora_apply_old_api(llm, adapter)
    ok_new = test_lora_apply_new_api(llm, adapter)

    if not (ok_old or ok_new):
        print("  Cannot continue without adapter applied\n")
        del llm
        return

    prefix_tokens, suffix_text = test_prefix_eval(llm)
    if prefix_tokens is None:
        print("  Cannot continue without prefix eval\n")
        del llm
        return

    test_kv_cache_save(llm, prefix_tokens)
    test_kv_cache_load(llm, prefix_tokens)
    test_inference(llm, prefix_tokens, suffix_text)

    del llm
    if os.path.exists(KV_CACHE_PATH):
        os.unlink(KV_CACHE_PATH)


def main():
    import subprocess
    import llama_cpp
    print(f"llama-cpp-python: {llama_cpp.__version__}")
    try:
        import programasweights
        print(f"programasweights: {programasweights.__version__}")
    except Exception:
        print("programasweights: not importable")
    print(f"GGML_NO_METAL: {os.environ.get('GGML_NO_METAL', 'not set')}")
    print()

    check_files()

    if "--gpu-only" in sys.argv:
        run_suite(-1)
        return
    if "--cpu-only" in sys.argv:
        run_suite(0)
        return

    # Run CPU tests in this process first (safe)
    run_suite(0)

    # Run GPU tests in a subprocess so a crash doesn't kill CPU results
    print("\n" + "=" * 60)
    print("Launching GPU test in subprocess (crash-safe)...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, __file__, "--gpu-only"],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    if result.returncode != 0:
        print(f"GPU subprocess exited with code {result.returncode}")
        if result.returncode < 0:
            import signal
            sig = -result.returncode
            signame = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
            print(f"  Killed by signal {signame} ({sig})")


if __name__ == "__main__":
    main()
