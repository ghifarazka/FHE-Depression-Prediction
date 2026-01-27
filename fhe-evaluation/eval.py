"""
eval.py

Difference from base.py: this code performs evaluation not only on a 
single sample but goes through the whole test dataset. In addition, 
this code compares plain vs encrypted inference based on 4 metrics
(numerical accuracy, computation time, memory usage, and data size).
"""

from openfhe import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import psutil
import pickle

# ========================= HELPER FUNCTIONS =========================

def next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

process = psutil.Process(os.getpid())
def get_mem_mb():
    return process.memory_info().rss / (1024 ** 2)

# ====================== INFERENCE FUNCTION ==========================

def run_plain_inference(x, weights, bias):

    # Time-Memory baseline [inference]
    mem_baseline = get_mem_mb()
    peak_mem = mem_baseline
    t0 = time.time()

    # Inference (plain)
    multiplied = [x[i] * weights[i] for i in range(len(x))]
    peak_mem = max(peak_mem, get_mem_mb())
    result_plain = sum(multiplied) + bias
    peak_mem = max(peak_mem, get_mem_mb())

    # Time-Memory overhead [inference]
    time_plain = time.time() - t0
    mem_plain = peak_mem - mem_baseline  # in MB

    # Datasize calculation (input data)
    x_serialized = pickle.dumps(x)
    datasize_plain = len(x_serialized)  # in bytes

    return {
        "result_plain": result_plain,
        "time_plain": time_plain,
        "mem_plain": mem_plain,
        "datasize_plain": datasize_plain,
    }


def run_fhe_inference(x, ckks_params, weights, bias):

    # Time-Memory baseline [setup]
    mem_baseline = get_mem_mb()
    peak_mem = mem_baseline
    t0 = time.time()

    # Set up - CryptoContext
    multDepth = ckks_params["multDepth"]
    peak_mem = max(peak_mem, get_mem_mb())
    scaleModSize = ckks_params["scaleModSize"]
    peak_mem = max(peak_mem, get_mem_mb())
    batchSize = ckks_params["batchSize"]
    peak_mem = max(peak_mem, get_mem_mb())
    rotations = ckks_params["rotations"]
    peak_mem = max(peak_mem, get_mem_mb())

    params = CCParamsCKKSRNS()
    peak_mem = max(peak_mem, get_mem_mb())
    params.SetMultiplicativeDepth(multDepth)
    peak_mem = max(peak_mem, get_mem_mb())
    params.SetScalingModSize(scaleModSize)
    peak_mem = max(peak_mem, get_mem_mb())
    params.SetBatchSize(batchSize)
    peak_mem = max(peak_mem, get_mem_mb())

    cryptoContext = GenCryptoContext(params)
    peak_mem = max(peak_mem, get_mem_mb())
    cryptoContext.Enable(PKESchemeFeature.PKE)
    peak_mem = max(peak_mem, get_mem_mb())
    cryptoContext.Enable(PKESchemeFeature.KEYSWITCH)
    peak_mem = max(peak_mem, get_mem_mb())
    cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)
    peak_mem = max(peak_mem, get_mem_mb())
    cryptoContext.Enable(PKESchemeFeature.ADVANCEDSHE)
    peak_mem = max(peak_mem, get_mem_mb())

    # Set up - key generation
    keypair = cryptoContext.KeyGen()
    peak_mem = max(peak_mem, get_mem_mb())
    publicKey = keypair.publicKey
    peak_mem = max(peak_mem, get_mem_mb())
    secretKey = keypair.secretKey
    peak_mem = max(peak_mem, get_mem_mb())

    cryptoContext.EvalMultKeyGen(secretKey)
    peak_mem = max(peak_mem, get_mem_mb())
    cryptoContext.EvalRotateKeyGen(secretKey, rotations)
    peak_mem = max(peak_mem, get_mem_mb())

    # Set up - model encoding
    weights_pt = cryptoContext.MakeCKKSPackedPlaintext(weights)
    peak_mem = max(peak_mem, get_mem_mb())
    bias_pt = cryptoContext.MakeCKKSPackedPlaintext([bias] * len(x))
    peak_mem = max(peak_mem, get_mem_mb())

    # Time-Memory overhead [setup]
    time_fhe_setup = time.time() - t0
    mem_fhe_setup = peak_mem - mem_baseline  # in MB

    # ----------------------------------------------------------------

    # Time-Memory baseline [encrypt]
    mem_baseline = get_mem_mb()
    peak_mem = mem_baseline
    t0 = time.time()

    # Encrypt
    pt = cryptoContext.MakeCKKSPackedPlaintext(x)
    peak_mem = max(peak_mem, get_mem_mb())
    ct = cryptoContext.Encrypt(publicKey, pt)
    peak_mem = max(peak_mem, get_mem_mb())

    # Time-Memory overhead [encrypt]
    time_fhe_encrypt = time.time() - t0
    mem_fhe_encrypt = peak_mem - mem_baseline  # in MB

    # ----------------------------------------------------------------

    # Time-Memory baseline [inference]
    mem_baseline = get_mem_mb()
    peak_mem = mem_baseline
    t0 = time.time()

    # Inference (encrypted)
    result_ct = cryptoContext.EvalInnerProduct(ct, weights_pt, len(x))
    peak_mem = max(peak_mem, get_mem_mb())
    result_ct = cryptoContext.EvalAdd(result_ct, bias_pt)
    peak_mem = max(peak_mem, get_mem_mb())

    # Time-Memory overhead [encrypt]
    time_fhe_inference = time.time() - t0
    mem_fhe_inference = peak_mem - mem_baseline  # in MB

    # ----------------------------------------------------------------

    # Time-Memory baseline [decrypt]
    mem_baseline = get_mem_mb()
    peak_mem = mem_baseline
    t0 = time.time()

    # Decrypt
    result_fhe = cryptoContext.Decrypt(result_ct, secretKey)
    peak_mem = max(peak_mem, get_mem_mb())
    result_fhe = result_fhe.GetRealPackedValue()[0]
    peak_mem = max(peak_mem, get_mem_mb())

    # Time-Memory overhead [decrypt]
    time_fhe_decrypt = time.time() - t0
    mem_fhe_decrypt = peak_mem - mem_baseline  # in MB

    # ----------------------------------------------------------------

    # Calculate total Time-Memory overhead
    time_fhe_total = (time_fhe_setup + time_fhe_encrypt + time_fhe_inference + time_fhe_decrypt)
    mem_fhe_total = (mem_fhe_setup + mem_fhe_encrypt + mem_fhe_inference + mem_fhe_decrypt)

    # Serialize OpenFHE objects for datasize calculation

    pk_serialized = Serialize(publicKey, BINARY)
    datasize_fhe_pk = len(pk_serialized)  # in bytes

    evalmult_serialized = SerializeEvalMultKeyString(BINARY, "")
    datasize_fhe_evalmult = len(evalmult_serialized)  # in bytes

    evalauto_serialized = SerializeEvalAutomorphismKeyString(BINARY, "")
    datasize_fhe_evalauto = len(evalauto_serialized)  # in bytes

    ct_serialized = Serialize(ct, BINARY)
    datasize_fhe_ct = len(ct_serialized)  # in bytes

    result_ct_serialized = Serialize(result_ct, BINARY)
    datasize_fhe_result_ct = len(result_ct_serialized)  # in bytes

    # "Clean" CryptoContext and serialize it
    ClearEvalMultKeys()
    cryptoContext.ClearEvalAutomorphismKeys()
    ReleaseAllContexts()
    cc_serialized = Serialize(cryptoContext, BINARY)
    datasize_fhe_cc = len(cc_serialized)  # in bytes

    # Calculate total datasize
    datasize_fhe_total = (datasize_fhe_cc + datasize_fhe_pk +
                          datasize_fhe_evalmult + datasize_fhe_evalauto +
                          datasize_fhe_ct + datasize_fhe_result_ct)  # in bytes


    return {
        "result_fhe": result_fhe,
        "time_fhe_setup": time_fhe_setup,
        "time_fhe_encrypt": time_fhe_encrypt,
        "time_fhe_inference": time_fhe_inference,
        "time_fhe_decrypt": time_fhe_decrypt,
        "time_fhe_total": time_fhe_total,
        "mem_fhe_setup": mem_fhe_setup,
        "mem_fhe_encrypt": mem_fhe_encrypt,
        "mem_fhe_inference": mem_fhe_inference,
        "mem_fhe_decrypt": mem_fhe_decrypt,
        "mem_fhe_total": mem_fhe_total,
        "datasize_fhe_cc": datasize_fhe_cc,
        "datasize_fhe_pk": datasize_fhe_pk,
        "datasize_fhe_evalmult": datasize_fhe_evalmult,
        "datasize_fhe_evalauto": datasize_fhe_evalauto,
        "datasize_fhe_ct": datasize_fhe_ct,
        "datasize_fhe_result_ct": datasize_fhe_result_ct,
        "datasize_fhe_total": datasize_fhe_total,
}

# ======================== MAIN APP =========================

TIME_BEGIN = time.time()

test_data = pd.read_csv("data/dass42_test.csv")
# test_data = pd.read_csv("data/dass42_test.csv").head(50)

weights = np.loadtxt("model/weights.txt").tolist()
bias = float(np.loadtxt("model/bias.txt"))
n = len(weights)

ckks_params = {
    "multDepth": 2,
    "scaleModSize": 50,
    "batchSize": next_power_of_two(n),
    "rotations": [1, 2, 4, 8, 16, 32, 64]
}

# Collect results for each sample
results = []

# Go through each sample
for i in range(len(test_data)):
    row = test_data.iloc[i]
    x = list(row)

    print(f"--- Sample {i+1}/{len(test_data)} ---")
    metadata = {"sample_id": i}

    # Perform inference (plain & FHE)
    res_plain = run_plain_inference(x, weights, bias)
    res_fhe = run_fhe_inference(x, ckks_params, weights, bias)

    # Calculate numerical error
    result_abs_error = abs(res_plain["result_plain"] - res_fhe["result_fhe"])
    result_rel_error = result_abs_error / abs(res_plain["result_plain"])

    # Calculate time overhead
    time_overhead = res_fhe["time_fhe_total"] / res_plain["time_plain"]

    # Calculate memory overhead
    mem_overhead = res_fhe["mem_fhe_total"] - res_plain["mem_plain"]

    # Calculate datasize overhead
    datasize_overhead = res_fhe["datasize_fhe_total"] / res_plain["datasize_plain"]

    res_metrics = {
        "result_abs_error": result_abs_error,
        "result_rel_error": result_rel_error,
        "time_overhead": time_overhead,
        "mem_overhead": mem_overhead,
        "datasize_overhead": datasize_overhead
    }
    
    combined_res = {**metadata, **res_plain, **res_fhe, **res_metrics}
    results.append(combined_res)

df = pd.DataFrame(results)
df.to_csv("results/eval_results.csv", index=False)
print(f"The data has been saved to results/eval_results.csv")

TIME_END = time.time()
print(f"\nTotal evaluation time: {TIME_END - TIME_BEGIN:.2f} seconds")
