"""
base.py

Base implementation of the model inference using and not using FHE. 
This base code will be expanded into the evaluation code eval.py and 
client-server codes on client-side/ and server-side/. 
"""

from openfhe import *
import numpy as np
import pandas as pd

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
print("Inference result (plain):", result_plain)

# ====================== INFERENCE (ENCRYPTED) =======================

# ========= 1. Set up CryptoContext, KeyGen, Model Encoding ==========

# Set up CKKS parameters
multDepth = 2
scaleModSize = 50
batchSize = next_power_of_two(n)
rotations = [1, 2, 4, 8, 16, 32, 64]

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

# Generate keys for multiplication and rotation operations
cryptoContext.EvalMultKeyGen(secretKey)
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
result_ct = cryptoContext.EvalInnerProduct(ct, weights_pt, n)
result_ct = cryptoContext.EvalAdd(result_ct, bias_pt)

# ======================= 4. Result Decryption =======================

# Decrypt the result
result_fhe = cryptoContext.Decrypt(result_ct, secretKey)
result_fhe = result_fhe.GetRealPackedValue()[0]

# Slot 0 should contain the inner product + bias result
print("Inference result (encrypted):", result_fhe)

# ========================= CLEAN UP =================================

# Clean CryptoContext
ClearEvalMultKeys()
cryptoContext.ClearEvalAutomorphismKeys()
ReleaseAllContexts()