"""
worker.py

Single-sample worker script. Invoked as a subprocess by evaluate.py.
Accepts a sample index via command-line argument, runs plain and FHE
inference, and prints a JSON result to stdout.

Usage:
    python worker.py <sample_index>

Memory methodology:
  - Plaintext : tracemalloc (Python heap peak, in bytes)
  - FHE steps : cumulative RSS above a shared pre-FHE baseline.
                Each step's value = RSS_after_step - RSS_before_FHE,
                so all four columns share the same baseline and are
                directly comparable (showing the growing footprint).
                mem_fhe_decrypt is therefore the peak/total FHE memory.
"""

import sys
import os
import json
import time
import tracemalloc
import psutil
import numpy as np
import pandas as pd
import math

EPSILON = 1e-10

def next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def get_rss():
    return psutil.Process(os.getpid()).memory_info().rss

def main():
    sample_idx = int(sys.argv[1])

    # ── Load data & model ──────────────────────────────────────────
    test_data = pd.read_csv("../data/dass42_test.csv")
    x = test_data.iloc[sample_idx].astype(float).tolist()
    n = len(x)

    weights = np.loadtxt("../model/weights.txt").tolist()
    bias    = float(np.loadtxt("../model/bias.txt"))

    result = {
        "sample_id": sample_idx,
        # accuracy
        "result_plain": None,
        "result_fhe":   None,
        # time
        "time_plain":         None,
        "time_fhe_setup":     None,
        "time_fhe_encrypt":   None,
        "time_fhe_inference": None,
        "time_fhe_decrypt":   None,
        # memory (bytes)
        "mem_plain":          None,   # tracemalloc peak (Python heap)
        "mem_fhe_setup":      None,   # cumulative RSS above pre-FHE baseline
        "mem_fhe_encrypt":    None,   # cumulative RSS above pre-FHE baseline
        "mem_fhe_inference":  None,   # cumulative RSS above pre-FHE baseline
        "mem_fhe_decrypt":    None,   # cumulative RSS above pre-FHE baseline (= peak total)
        "error": None,
    }

    # ── INFERENCE (PLAIN) ──────────────────────────────────────────
    try:
        tracemalloc.start()
        t0 = time.perf_counter()

        multiplied     = [x[i] * weights[i] for i in range(n)]
        multiplied_sum = sum(multiplied)
        result_plain   = multiplied_sum + bias

        t1 = time.perf_counter()
        _, mem_plain_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result["result_plain"] = float(result_plain)
        result["time_plain"]   = t1 - t0
        result["mem_plain"]    = mem_plain_peak   # peak bytes on Python heap

    except Exception as e:
        result["error"] = f"PLAIN ERROR: {e}"
        print(json.dumps(result))
        sys.exit(0)   # valid JSON was produced; let evaluate.py's data.get("error") branch handle it

    # ── INFERENCE (ENCRYPTED) ──────────────────────────────────────
    try:
        from openfhe import (
            CCParamsCKKSRNS, GenCryptoContext,
            PKESchemeFeature, ReleaseAllContexts,
        )

        batchSize = next_power_of_two(n)
        rotations = [2**i for i in range(int(math.log2(batchSize // 2)) + 1)]

        # shared baseline — RSS of the process before any FHE allocation
        fhe_baseline = get_rss()

        # ── 1. Setup ───────────────────────────────────────────────
        t0 = time.perf_counter()

        multDepth = 1
        scaleModSize = 50

        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(multDepth)
        params.SetScalingModSize(scaleModSize)
        params.SetBatchSize(batchSize)
        cryptoContext = GenCryptoContext(params)

        cryptoContext.Enable(PKESchemeFeature.PKE)
        cryptoContext.Enable(PKESchemeFeature.KEYSWITCH)
        cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)
        cryptoContext.Enable(PKESchemeFeature.ADVANCEDSHE)

        keypair   = cryptoContext.KeyGen()
        publicKey = keypair.publicKey
        secretKey = keypair.secretKey

        cryptoContext.EvalRotateKeyGen(secretKey, rotations)

        weights_pt = cryptoContext.MakeCKKSPackedPlaintext(weights)
        bias_pt    = cryptoContext.MakeCKKSPackedPlaintext([bias] * n)

        t1 = time.perf_counter()
        result["time_fhe_setup"] = t1 - t0
        # cumulative: RSS after setup − pre-FHE baseline
        result["mem_fhe_setup"]  = max(0, get_rss() - fhe_baseline)

        # ── 2. Encryption ──────────────────────────────────────────
        t0 = time.perf_counter()

        pt = cryptoContext.MakeCKKSPackedPlaintext(x)
        ct = cryptoContext.Encrypt(publicKey, pt)

        t1 = time.perf_counter()
        result["time_fhe_encrypt"] = t1 - t0
        # cumulative: RSS after encryption − pre-FHE baseline
        result["mem_fhe_encrypt"]  = max(0, get_rss() - fhe_baseline)

        # ── 3. Inference ───────────────────────────────────────────
        t0 = time.perf_counter()

        result_ct = cryptoContext.EvalInnerProduct(ct, weights_pt, batchSize)
        result_ct = cryptoContext.EvalAdd(result_ct, bias_pt)

        t1 = time.perf_counter()
        result["time_fhe_inference"] = t1 - t0
        # cumulative: RSS after inference − pre-FHE baseline
        result["mem_fhe_inference"]  = max(0, get_rss() - fhe_baseline)

        # ── 4. Decryption ──────────────────────────────────────────
        t0 = time.perf_counter()

        result_fhe_pt = cryptoContext.Decrypt(result_ct, secretKey)
        result_fhe    = result_fhe_pt.GetRealPackedValue()[0]

        t1 = time.perf_counter()
        result["time_fhe_decrypt"] = t1 - t0
        # cumulative: RSS after decryption − pre-FHE baseline (= peak total)
        result["mem_fhe_decrypt"]  = max(0, get_rss() - fhe_baseline)

        result["result_fhe"] = float(result_fhe)

        # ── Cleanup ────────────────────────────────────────────────
        cryptoContext.ClearEvalAutomorphismKeys()
        ReleaseAllContexts()

    except Exception as e:
        result["error"] = f"FHE ERROR: {e}"
        print(json.dumps(result))
        sys.exit(0)   # valid JSON was produced; let evaluate.py's data.get("error") branch handle it

    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()