"""
server.py

Backend server for encrypted model inference using FHE. Receives encrypted 
input from client, performs inference, and returns encrypted result.
"""

from openfhe import *
from flask import Flask, request, jsonify
import numpy as np
import base64

# ========================= HELPER FUNCTIONS =========================

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

def deserialize_CryptoContext_from_base64(cryptoContext_ser):
    """
    Takes  base64  and   turns  it  into  FHE 
    CryptoContext object.
    """
    try:
        bin_str = base64.b64decode(cryptoContext_ser)
        cc = DeserializeCryptoContextString(bin_str, BINARY)
        return cc
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

def deserialize_PublicKey_from_base64(publicKey_ser):
    """
    Takes  base64  and   turns  it  into  FHE 
    PublicKey object.
    """
    try:
        bin_str = base64.b64decode(publicKey_ser)
        pk = DeserializePublicKeyString(bin_str, BINARY)
        return pk
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

def deserialize_EvalMultKey_from_base64(evalMultKey_ser):
    """
    Takes  base64  and  installs  EvalMultKey 
    into the CryptoContext.
    """
    try:
        bin_str = base64.b64decode(evalMultKey_ser)
        DeserializeEvalMultKeyString(bin_str, BINARY)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

def deserialize_EvalAutomorphismKey_from_base64(evalAutomorphismKey_ser):
    """
    Takes  base64 and  installs  EvalAutomor-
    phismKey into the CryptoContext.
    """
    try:
        bin_str = base64.b64decode(evalAutomorphismKey_ser)
        DeserializeEvalAutomorphismKeyString(bin_str, BINARY)
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

# ============================= MAIN APP =============================

app = Flask(__name__)

@app.route("/fhe-predict", methods=["POST"])
def fhe_predict():
    """
    Accepts an encrypted vector, CryptoContext, and Public Key, then 
    sends back the encrypted result.
    """
    try:
        data = request.get_json()

        # 1. Deserialize CryptoContext, PublicKey, and Ciphertext

        cc_ser = data["cryptoContext"]
        cryptoContext = deserialize_CryptoContext_from_base64(cc_ser)
        assert isinstance(cryptoContext, CryptoContext)

        pk_ser = data["publicKey"]
        publicKey = deserialize_PublicKey_from_base64(pk_ser)
        assert isinstance(publicKey, PublicKey)

        evalmult_ser = data["evalMultKey"]
        deserialize_EvalMultKey_from_base64(evalmult_ser)

        evalauto_ser = data["evalAutomorphismKey"]
        deserialize_EvalAutomorphismKey_from_base64(evalauto_ser)

        ct_ser = data["ciphertext"]
        ct = deserialize_Ciphertext_from_base64(ct_ser)
        assert isinstance(ct, Ciphertext)

        print("---- Client Data Deserialized! ----")

        # 2. Load and encode models

        weights = np.loadtxt("models/weights.txt")
        intercept = np.loadtxt("models/intercept.txt")
        n = len(weights.tolist())

        print("---- Model Loaded! ----")

        weights_pt = cryptoContext.MakeCKKSPackedPlaintext(weights.tolist())
        bias_pt = cryptoContext.MakeCKKSPackedPlaintext([intercept]*n)

        # 3. Evaluate

        print("---- Performing Computation... ----")

        ct_mul = cryptoContext.EvalMult(ct, weights_pt) 

        step1 = cryptoContext.EvalAdd(ct_mul, cryptoContext.EvalRotate(ct_mul, 1))
        step2 = cryptoContext.EvalAdd(step1, cryptoContext.EvalRotate(step1, 2))
        step3 = cryptoContext.EvalAdd(step2, cryptoContext.EvalRotate(step2, 4))
        step4 = cryptoContext.EvalAdd(step3, cryptoContext.EvalRotate(step3, 8))
        step5 = cryptoContext.EvalAdd(step4, cryptoContext.EvalRotate(step4, 16))
        step6 = cryptoContext.EvalAdd(step5, cryptoContext.EvalRotate(step5, 32))

        result_ct = cryptoContext.EvalAdd(step6, bias_pt)

        ClearEvalMultKeys()

        print("---- Computation Completed! ----")

        # 4. Send result

        print("---- Sending Result to Client... ----")

        result_ct_ser = serialize_to_base64(result_ct)

        return jsonify({"resultEncrypted": f"{result_ct_ser}"})

    except Exception as e:
        raise RuntimeError(f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
