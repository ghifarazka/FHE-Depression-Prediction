from openfhe import *
import numpy as np
import pandas as pd
import joblib
import base64
import requests

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

def serialize_EvalMultKey_to_base64():
	"""
	Takes  EvalMultKey  and  turns  it   into 
	base64.
	"""
	try:
		ser = SerializeEvalMultKeyString(BINARY, "")
		evalMultKey_ser = base64.b64encode(ser).decode("utf-8")
		return evalMultKey_ser
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

# ==================== 1. Load & Preprocess Data ====================

preprocessor = joblib.load("preprocessor.joblib")

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

# ==================== 2. Set up CryptoContext ====================

multDepth = 2
scaleModSize = 50
batchSize = next_power_of_two(n)

params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(multDepth)
params.SetScalingModSize(scaleModSize)
params.SetBatchSize(batchSize)
cryptoContext = GenCryptoContext(params)

# enable the common features used in ML
cryptoContext.Enable(PKESchemeFeature.PKE)
cryptoContext.Enable(PKESchemeFeature.KEYSWITCH)
cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)

# ==================== 3. Key Generation ====================

keypair = cryptoContext.KeyGen()
publicKey = keypair.publicKey
secretKey = keypair.secretKey

cryptoContext.EvalMultKeyGen(secretKey) 
rotations = [1, 2, 4, 8, 16, 32]
cryptoContext.EvalRotateKeyGen(secretKey, rotations)

# ==================== 4. Data Encryption ====================

pt = cryptoContext.MakeCKKSPackedPlaintext(x)
ct = cryptoContext.Encrypt(publicKey, pt)

# ==================== 5. Serializations ====================

cc_ser = serialize_to_base64(cryptoContext)
pk_ser = serialize_to_base64(publicKey)
evalmult_ser = serialize_EvalMultKey_to_base64()
evalauto_ser = serialize_EvalAutomorphismKey_to_base64()
ct_ser = serialize_to_base64(ct)

# ==================== 6. Send to Server ====================

SERVER_URL = "http://13.210.221.158:8000"

payload = {
	"cryptoContext": cc_ser,
	"publicKey": pk_ser,
	"evalMultKey": evalmult_ser,
	"evalAutomorphismKey": evalauto_ser,
	"ciphertext": ct_ser
}
result = requests.post(SERVER_URL + "/fhe-predict", json=payload)
resultEncrypted_ser = result.json()["resultEncrypted"]

# ==================== 7. Result Decryption ====================

resultEncrypted = deserialize_Ciphertext_from_base64(resultEncrypted_ser)
result = cryptoContext.Decrypt(resultEncrypted, secretKey)
result = result.GetRealPackedValue()
# Because of slot-sum, every slot should be the inner product + bias (e.g. single repeated value)
print("Inner product (take slot 0):", result[0])