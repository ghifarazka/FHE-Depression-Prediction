"""
measure_payload_composition.py

Measures the size of each component of the communication-cost payload
(CryptoContext, RotateKey, Ciphertext) INDIVIDUALLY -- both as raw
binary and as base64 text -- without touching the network at all.

Why this doesn't need comm_client.py / comm_server.py or a repeated
10-trial run: payload size is fixed entirely by the CKKS scheme
parameters (ring dimension, multiplicative depth, number of rotation
indices) and the input length n -- NOT by network conditions, which
host it's sent to, or which trial it is. This has already been
confirmed empirically: bytes_sent came out bit-for-bit identical
(22,030,794 bytes) across every trial in every previous run, regardless
of host or session. So this script runs the same setup + encryption
steps as comm_client.py, once, purely locally, and reports the
composition directly -- no Flask, no requests, no remote host required.

Usage:
    python measure_payload_composition.py
"""

import base64
import math
import os

import pandas as pd

from openfhe import (
    BINARY,
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    ReleaseAllContexts,
    Serialize,
    SerializeEvalAutomorphismKeyString,
)

DATA_PATH = "../data/dass42_test_singular.csv"
RESULTS_DIR = "results"
PATH_OUT = os.path.join(RESULTS_DIR, "payload_composition.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# ── Setup: identical to base.py / comm_client.py ────────────────────
test_data_singular = pd.read_csv(DATA_PATH)
x = test_data_singular.iloc[0].astype(float).tolist()
n = len(x)

multDepth    = 1
scaleModSize = 50
batchSize    = next_power_of_two(n)
rotations    = [2**i for i in range(int(math.log2(batchSize // 2)) + 1)]

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

pt = cryptoContext.MakeCKKSPackedPlaintext(x)
ct = cryptoContext.Encrypt(publicKey, pt)

# ── Serialize each component individually ───────────────────────────
cc_raw       = Serialize(cryptoContext, BINARY)
evalauto_raw = SerializeEvalAutomorphismKeyString(BINARY, "")
ct_raw       = Serialize(ct, BINARY)

cc_b64       = base64.b64encode(cc_raw)
evalauto_b64 = base64.b64encode(evalauto_raw)
ct_b64       = base64.b64encode(ct_raw)

components = [
    ("CryptoContext",                    cc_raw,       cc_b64),
    ("RotateKey (EvalAutomorphismKey)",  evalauto_raw, evalauto_b64),
    ("Ciphertext",                       ct_raw,       ct_b64),
]

total_raw = sum(len(raw) for _, raw, _ in components)
total_b64 = sum(len(b64) for _, _, b64 in components)

# ── Report + save ────────────────────────────────────────────────────
print(f"{'='*72}")
print(f"  Payload composition (n={n} features, batchSize={batchSize}, "
      f"multDepth={multDepth}, rotations={rotations})")
print(f"{'='*72}")
print(f"{'Component':<34}{'Raw (MB)':>12}{'Base64 (MB)':>14}{'% of total':>12}")

rows = []
for name, raw, b64 in components:
    raw_mb = len(raw) / 1e6
    b64_mb = len(b64) / 1e6
    pct = len(b64) / total_b64 * 100
    print(f"{name:<34}{raw_mb:>12.4f}{b64_mb:>14.4f}{pct:>11.2f}%")
    rows.append({
        "component": name,
        "bytes_raw": len(raw),
        "bytes_base64": len(b64),
        "pct_of_total_base64": pct,
    })

print(f"{'-'*72}")
print(f"{'TOTAL':<34}{total_raw/1e6:>12.4f}{total_b64/1e6:>14.4f}{100.00:>11.2f}%")
print(f"{'='*72}\n")

import csv
write_header = not os.path.exists(PATH_OUT)
with open(PATH_OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["component", "bytes_raw", "bytes_base64", "pct_of_total_base64"])
    if write_header:
        w.writeheader()
    for r in rows:
        w.writerow(r)
    w.writerow({"component": "TOTAL", "bytes_raw": total_raw, "bytes_base64": total_b64, "pct_of_total_base64": 100.00})

print(f"Saved to {PATH_OUT}")

cryptoContext.ClearEvalAutomorphismKeys()
ReleaseAllContexts()
