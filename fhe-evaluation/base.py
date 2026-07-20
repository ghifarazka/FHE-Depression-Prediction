"""
base.py

Base implementation of the model inference using and not using FHE. 
The base code will then be evaluated across 4 metrics in 
`eval-fhe-implementation/` and `eval-communication-cost/`. Finally, 
the code will be expanded into client and server components in 
`../client-side/` and `../server-side/`. 
"""

from openfhe import *
import numpy as np
import pandas as pd
import math

# ========================= HELPER FUNCTIONS =========================

def next_power_of_two(x):
    """
    Returns the next power of 2 after x.
    If x <= 1, returns 1.
    """
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

# ======================== LOAD DATA & MODEL =========================

# Load preprocessed test data (single sample) for inference
test_data_singular = pd.read_csv("data/dass42_test_singular.csv")
x = test_data_singular.iloc[0].astype(float).tolist()
n = len(x)

# Load model weights and bias
weights = np.loadtxt("model/weights.txt").tolist()
bias = np.loadtxt("model/bias.txt").tolist()

# ======================== INFERENCE (PLAIN) =========================

multiplied = [x[i] * weights[i] for i in range(len(x))]
multiplied_sum = sum(multiplied)
result_plain = multiplied_sum + bias

# ====================== INFERENCE (ENCRYPTED) =======================

# ========= 1. Set up CryptoContext, KeyGen, Model Encoding ==========

# Set up CKKS parameters
multDepth = 1       # 1 multiplication operation
scaleModSize = 50   # theoretically 15-digit precision
batchSize = next_power_of_two(n)
rotations = [2**i for i in range(int(math.log2(batchSize//2)) + 1)] # powers of 2 up to batchSize//2

# Initialize CryptoContext with the specified parameters
params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(multDepth)
params.SetScalingModSize(scaleModSize)
params.SetBatchSize(batchSize)
cryptoContext = GenCryptoContext(params)

# Enable cryptographic features commonly used in ML
cryptoContext.Enable(PKESchemeFeature.PKE)
cryptoContext.Enable(PKESchemeFeature.KEYSWITCH)
cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)
cryptoContext.Enable(PKESchemeFeature.ADVANCEDSHE)

# Generate public and secret keys
keypair = cryptoContext.KeyGen()
publicKey = keypair.publicKey
secretKey = keypair.secretKey

# Generate a key for rotation operations
cryptoContext.EvalRotateKeyGen(secretKey, rotations)

# Convert weights and bias to CKKS plaintexts
weights_pt = cryptoContext.MakeCKKSPackedPlaintext(weights)
bias_pt = cryptoContext.MakeCKKSPackedPlaintext([bias] * n)

# ======================= 2. Data Encryption =========================

# Encrypt the input data
pt = cryptoContext.MakeCKKSPackedPlaintext(x)
ct = cryptoContext.Encrypt(publicKey, pt)

# ========================== 3. Inference ============================

# Perform encrypted inference (inner product + bias addition)
result_ct = cryptoContext.EvalInnerProduct(ct, weights_pt, batchSize)
result_ct = cryptoContext.EvalAdd(result_ct, bias_pt)

# ======================= 4. Result Decryption =======================

# Decrypt the result
result_fhe = cryptoContext.Decrypt(result_ct, secretKey)

# Slot 0 should contain the inner product + bias result
result_fhe = result_fhe.GetRealPackedValue()[0]

# =========================== PRINT RESULTS ==========================

# Print the ring dimension of the CryptoContext (OpenFHE sets 128-bit security by default)
print("CryptoContext ring dimension:", cryptoContext.GetRingDimension())

# Print the results of both plain and encrypted inference
print("Inference result (plain):", result_plain)
print("Inference result (encrypted):", result_fhe)

# Print the relative error between the plain and encrypted results
rel_error = abs(result_plain - result_fhe) / (abs(result_plain) + 1e-10)  # avoid division by zero
print("Relative error:", rel_error)

# ========================= CLEAN UP =================================

# Clean CryptoContext
cryptoContext.ClearEvalAutomorphismKeys()
ReleaseAllContexts()
