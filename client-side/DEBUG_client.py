"""
DEBUG_client.py

Basic client code to test the server-side FHE inference endpoint. Takes a
singular example input from the DASS-42 form (before preprocessing), setup a 
CryptoContext, generate relevant keys, encrypts the input, serializes the 
needed components, send them to the server, and decrypts the result. This 
code is only for debugging purposes and is not part of the final client-side 
implementation.
"""

from openfhe import *
import pandas as pd
import joblib
import base64
import requests
import math

# ========================= HELPER FUNCTIONS =========================

def next_power_of_two(x):
	"""
	Returns  the  next  power of  2  after  x
	"""
	if x <= 1:
		return 1
	return 1 << (x - 1).bit_length()

def serialize_to_base64(obj):
	"""
	Takes  any FHE  object and  turns it into 
	base64.
	"""
	try:
		ser = Serialize(obj, BINARY)
		base64_str = base64.b64encode(ser).decode("utf-8")
		return base64_str
	except Exception as e:
		raise RuntimeError(f"Error: {e}")

def serialize_EvalAutomorphismKey_to_base64():
	"""
	Takes  EvalAutomorphismKey  and  turns it
	into base64.
	"""
	try:
		ser = SerializeEvalAutomorphismKeyString(BINARY, "")
		evalAutomorphismKey_ser = base64.b64encode(ser).decode("utf-8")
		return evalAutomorphismKey_ser
	except Exception as e:
		raise RuntimeError(f"Error: {e}")

def deserialize_Ciphertext_from_base64(ciphertext_ser):
    """
    Takes  base64  and   turns  it  into  FHE 
    Ciphertext object.
    """
    try:
        bin_str = base64.b64decode(ciphertext_ser)
        ct = DeserializeCiphertextString(bin_str, BINARY)
        return ct
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

# ===================== LOAD & PREPROCESS DATA =======================

preprocessor = joblib.load("preprocessor.joblib")

# Example input from the form
X_new = pd.DataFrame([{
    "Q1A": 0, "Q2A": 0, "Q4A": 0, "Q6A": 3, "Q7A": 1, "Q8A": 1,
    "Q9A": 1, "Q11A": 3, "Q12A": 2, "Q14A": 0, "Q15A": 3, "Q18A": 2,
    "Q19A": 0, "Q20A": 0, "Q22A": 1, "Q23A": 0, "Q25A": 2, "Q27A": 2,
    "Q28A": 0, "Q29A": 2, "Q30A": 0, "Q32A": 0, "Q33A": 1, "Q35A": 0,
    "Q36A": 1, "Q39A": 1, "Q40A": 1, "Q41A": 0,
    "TIPI1": 6, "TIPI2": 6, "TIPI3": 2, "TIPI4": 7, "TIPI5": 6,
    "TIPI6": 1, "TIPI7": 5, "TIPI8": 7, "TIPI9": 3, "TIPI10": 3,
    "education": 1.0, "urban": 2.0, "gender": 2, "engnat": 1, "hand": 1,
    "orientation": 1, "voted": 2, "married": 1, "familysize": 2,
    "age_group": 1.0, "race_group": "White", "religion_group": "Christian"
}])

X_new_pre = preprocessor.transform(X_new)
x = X_new_pre.flatten().tolist()
n = len(x)

# ================= 1. Set up CryptoContext & KeyGen =================

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

# ======================= 2. Data Encryption =========================

# Encrypt the input data
pt = cryptoContext.MakeCKKSPackedPlaintext(x)
ct = cryptoContext.Encrypt(publicKey, pt)

# ======================== 3. Serializations =========================

# Serialize CryptoContext, RotateKey, and Ciphertext to base64
cc_ser = serialize_to_base64(cryptoContext)
evalauto_ser = serialize_EvalAutomorphismKey_to_base64()
ct_ser = serialize_to_base64(ct)

# =============== 4. Send to Server & Receive Result =================

# Replace with the server's URL
SERVER_URL = "http://148.230.103.61:8000/fhe-predict"
# SERVER_URL = "http://localhost:8000/fhe-predict"

# Send CryptoContext, RotateKey, and Ciphertext to the server
payload = {
	"cryptoContext": cc_ser,
	"evalAutomorphismKey": evalauto_ser,
	"ciphertext": ct_ser
}
result = requests.post(SERVER_URL, json=payload)
resultEncrypted_ser = result.json()["resultEncrypted"]

with open("payload_sample.txt", "w") as f:
    print(payload, file=f)

# print("Server response:", result.status_code, result.text)

# ==================== 5. Result Decryption ====================

resultEncrypted = deserialize_Ciphertext_from_base64(resultEncrypted_ser)
result = cryptoContext.Decrypt(resultEncrypted, secretKey)

# Slot 0 should contain the inner product + bias result
result = result.GetRealPackedValue()[0]
print("Inner product (take slot 0):", result)
