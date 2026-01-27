"""
eval_ckks_params.py

Evaluate CKKS parameters (multDepth, scaleModSize)
based on numerical accuracy and computation time
for FHE inference.
"""

from openfhe import *
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

TIME_BEGIN = time.time()

# ========================= HELPER FUNCTIONS =========================

def next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

# ======================== LOAD DATA & MODEL =========================

test_data_singular = pd.read_csv("data/dass42_test_singular.csv")
x = test_data_singular.iloc[0].astype(float).tolist()
n = len(x)

weights = np.loadtxt("model/weights.txt").tolist()
bias = float(np.loadtxt("model/bias.txt"))

# ======================== PLAIN INFERENCE ===========================

t0 = time.time()
multiplied = [x[i] * weights[i] for i in range(n)]
result_plain = sum(multiplied) + bias
plain_time = time.time() - t0

print(f"Plain inference result: {result_plain}")
print(f"Plain inference time  : {plain_time:.6f}s")

# ======================== FHE FUNCTION ==============================

def run_fhe_inference(multDepth, scaleModSize, x, weights, bias,
                      batchSize, rotations, result_plain, plain_time):

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(multDepth)
    params.SetScalingModSize(scaleModSize)
    params.SetBatchSize(batchSize)

    cryptoContext = GenCryptoContext(params)
    cryptoContext.Enable(PKESchemeFeature.PKE)
    cryptoContext.Enable(PKESchemeFeature.KEYSWITCH)
    cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)
    cryptoContext.Enable(PKESchemeFeature.ADVANCEDSHE)

    # Keys
    keypair = cryptoContext.KeyGen()
    publicKey = keypair.publicKey
    secretKey = keypair.secretKey

    cryptoContext.EvalMultKeyGen(secretKey)
    cryptoContext.EvalRotateKeyGen(secretKey, rotations)

    # Encrypt
    pt = cryptoContext.MakeCKKSPackedPlaintext(x)
    ct = cryptoContext.Encrypt(publicKey, pt)

    weights_pt = cryptoContext.MakeCKKSPackedPlaintext(weights)
    bias_pt = cryptoContext.MakeCKKSPackedPlaintext([bias] * len(x))

    # Encrypted inference
    t1 = time.time()
    result_ct = cryptoContext.EvalInnerProduct(ct, weights_pt, len(x))
    result_ct = cryptoContext.EvalAdd(result_ct, bias_pt)
    fhe_time = time.time() - t1

    # Decrypt
    result_fhe = cryptoContext.Decrypt(result_ct, secretKey)
    result_fhe = result_fhe.GetRealPackedValue()[0]

    # Clean CryptoContext
    ClearEvalMultKeys()
    cryptoContext.ClearEvalAutomorphismKeys()
    ReleaseAllContexts()

    # Errors
    abs_error = abs(result_fhe - result_plain)
    rel_error = abs_error / abs(result_plain)

    return {
        "multDepth": multDepth,
        "scaleModSize": scaleModSize,
        "plain_result": result_plain,
        "fhe_result": result_fhe,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "plain_time_sec": plain_time,
        "fhe_time_sec": fhe_time
    }

# ====================== EXPERIMENT SETTINGS =========================

batchSize = next_power_of_two(n)
rotations = [1, 2, 4, 8, 16, 32, 64]

# ====================== EXPERIMENT A ================================
# multDepth sweep (scaleModSize fixed)

multDepth_list = range(1, 51)
scale_fixed = 50

results_md = []

print("\n=== Running multDepth sweep ===")

for md in multDepth_list:
    try:
        print(f"multDepth={md}, scaleModSize={scale_fixed}")
        res = run_fhe_inference(
            multDepth=md,
            scaleModSize=scale_fixed,
            x=x,
            weights=weights,
            bias=bias,
            batchSize=batchSize,
            rotations=rotations,
            result_plain=result_plain,
            plain_time=plain_time
        )
        results_md.append(res)
    except Exception as e:
        print(f"Error with multDepth={md}: {e}")
        results_md.append({
            "multDepth": md,
            "fhe_time_sec": "ERROR",
            "abs_error": "ERROR"
        })

df_md = pd.DataFrame(results_md)
print("\n=== multDepth sweep results ===\n")
print(df_md)

# ====================== PLOTS (multDepth) ===========================

# Ensure the "figures" directory exists
os.makedirs("figures", exist_ok=True)

# Plot and save FHE Inference Time vs Multiplicative Depth
plt.figure()
for _, row in df_md.iterrows():
    if row["fhe_time_sec"] == "ERROR":
        plt.scatter(row["multDepth"], 0, color="red", marker="x", label="ERROR" if "ERROR" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(row["multDepth"], row["fhe_time_sec"], color="blue")
plt.xlabel("Multiplicative Depth")
plt.ylabel("FHE Inference Time (seconds)")
plt.title("Effect of Multiplicative Depth on FHE Computation Time")
plt.legend()
plt.grid(True)
plt.savefig("figures/fhe_time_vs_multdepth.png")
plt.show()

# Plot and save Absolute Error vs Multiplicative Depth
plt.figure()
for _, row in df_md.iterrows():
    if row["abs_error"] == "ERROR":
        plt.scatter(row["multDepth"], 1e-10, color="red", marker="x", label="ERROR" if "ERROR" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(row["multDepth"], row["abs_error"], color="blue")
plt.yscale("log")
plt.xlabel("Multiplicative Depth")
plt.ylabel("Absolute Error (log scale)")
plt.title("Effect of Multiplicative Depth on Numerical Error")
plt.legend()
plt.grid(True)
plt.savefig("figures/abs_error_vs_multdepth.png")
plt.show()

# ====================== EXPERIMENT B ================================
# scaleModSize sweep (multDepth fixed)

scale_list = range(1, 61)
multDepth_fixed = 2

results_sm = []

print("\n=== Running scaleModSize sweep ===")

for sm in scale_list:
    try:
        print(f"multDepth={multDepth_fixed}, scaleModSize={sm}")
        res = run_fhe_inference(
            multDepth=multDepth_fixed,
            scaleModSize=sm,
            x=x,
            weights=weights,
            bias=bias,
            batchSize=batchSize,
            rotations=rotations,
            result_plain=result_plain,
            plain_time=plain_time
        )
        results_sm.append(res)
    except Exception as e:
        print(f"Error with scaleModSize={sm}: {e}")
        results_sm.append({
            "scaleModSize": sm,
            "fhe_time_sec": "ERROR",
            "abs_error": "ERROR"
        })

df_sm = pd.DataFrame(results_sm)
print("\n=== scaleModSize sweep results ===\n")
print(df_sm)

# ====================== PLOTS (scaleModSize) ========================

# Plot and save FHE Inference Time vs Scaling Modulus Size
plt.figure()
for _, row in df_sm.iterrows():
    if row["fhe_time_sec"] == "ERROR":
        plt.scatter(row["scaleModSize"], 0, color="red", marker="x", label="ERROR" if "ERROR" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(row["scaleModSize"], row["fhe_time_sec"], color="blue")
plt.xlabel("Scaling Modulus Size (bits)")
plt.ylabel("FHE Inference Time (seconds)")
plt.title("Effect of Scaling Modulus Size on FHE Computation Time")
plt.legend()
plt.grid(True)
plt.savefig("figures/fhe_time_vs_scalemodsize.png")
plt.show()

# Plot and save Absolute Error vs Scaling Modulus Size
plt.figure()
for _, row in df_sm.iterrows():
    if row["abs_error"] == "ERROR":
        plt.scatter(row["scaleModSize"], 1e-10, color="red", marker="x", label="ERROR" if "ERROR" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(row["scaleModSize"], row["abs_error"], color="blue")
plt.yscale("log")
plt.xlabel("Scaling Modulus Size (bits)")
plt.ylabel("Absolute Error (log scale)")
plt.title("Effect of Scaling Modulus Size on Numerical Error")
plt.legend()
plt.grid(True)
plt.savefig("figures/abs_error_vs_scalemodsize.png")
plt.show()

# ====================== SAVE RESULTS ================================

df_md.to_csv("results/ckks_multdepth_sweep.csv", index=False)
df_sm.to_csv("results/ckks_scalemodsize_sweep.csv", index=False)

TIME_END = time.time()
print(f"\nTotal program running time: {TIME_END - TIME_BEGIN:.2f} s")

print("\nAll experiments completed.")
