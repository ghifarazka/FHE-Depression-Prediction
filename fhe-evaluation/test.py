from openfhe import *

a = [10]
b = [5]

# ========= 1. Set up CryptoContext, KeyGen, Model Encoding ==========

# Set up CKKS parameters
multDepth = 1       # 1 multiplication operation
scaleModSize = 50   # theoretically 15-digit precision
batchSize = 1

# Initialize CryptoContext with the specified parameters
params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(multDepth)
params.SetScalingModSize(scaleModSize)
params.SetBatchSize(batchSize)
cryptoContext = GenCryptoContext(params)

# Enable cryptographic features commonly used in ML
cryptoContext.Enable(PKESchemeFeature.PKE)
cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)

# Generate public and secret keys
keypair = cryptoContext.KeyGen()
publicKey = keypair.publicKey
secretKey = keypair.secretKey

multKey = cryptoContext.EvalMultKeyGen(secretKey)

# Convert to CKKS plaintexts
a_pt = cryptoContext.MakeCKKSPackedPlaintext(a)
b_pt = cryptoContext.MakeCKKSPackedPlaintext(b)

# ======================= 2. Data Encryption =========================

# Encrypt the input data
a_ct = cryptoContext.Encrypt(publicKey, a_pt)
b_ct = cryptoContext.Encrypt(publicKey, b_pt)

print(a_ct)

# ========================== 3. Inference ============================

# Perform encrypted inference (inner product + bias addition)
result_ct = cryptoContext.EvalMult(a_ct, b_ct)

# ======================= 4. Result Decryption =======================

# Decrypt the result
result_ct = cryptoContext.Decrypt(result_ct, secretKey).GetRealPackedValue()[0]

print(result_ct)