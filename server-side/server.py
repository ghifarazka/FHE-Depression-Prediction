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

def next_power_of_two(x):
    """
    Returns the next power of 2 after x.
    If x <= 1, returns 1.
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
    Accepts an encrypted vector, CryptoContext, and RotateKey, then 
    sends back the encrypted result.
    """
    try:
        data = request.get_json()

        # 1. Deserialize CryptoContext, RotateKey, and Ciphertext

        cc_ser = data["cryptoContext"]
        cryptoContext = deserialize_CryptoContext_from_base64(cc_ser)
        assert isinstance(cryptoContext, CryptoContext)

        evalauto_ser = data["evalAutomorphismKey"]
        deserialize_EvalAutomorphismKey_from_base64(evalauto_ser)

        ct_ser = data["ciphertext"]
        ct = deserialize_Ciphertext_from_base64(ct_ser)
        assert isinstance(ct, Ciphertext)

        print("---- Client Data Deserialized! ----")

        # 2. Load and encode models

        weights = np.loadtxt("model/weights.txt")
        bias = np.loadtxt("model/bias.txt")
        n = len(weights.tolist())
        batchSize = next_power_of_two(n)

        print("---- Model Loaded! ----")

        weights_pt = cryptoContext.MakeCKKSPackedPlaintext(weights.tolist())
        bias_pt = cryptoContext.MakeCKKSPackedPlaintext([bias]*n)

        # 3. Inference (Encrypted)

        print("---- Performing Computation... ----")

        result_ct = cryptoContext.EvalInnerProduct(ct, weights_pt, batchSize)
        result_ct = cryptoContext.EvalAdd(result_ct, bias_pt)

        print("---- Computation Completed! ----")

        # 4. Send result

        print("---- Sending Result to Client... ----")

        result_ct_ser = serialize_to_base64(result_ct)

        return jsonify({"resultEncrypted": f"{result_ct_ser}"})

    except Exception as e:
        raise RuntimeError(f"Error: {e}")

    # 5. Cleanup
    finally:
        try:
            if cryptoContext is not None:
                cryptoContext.ClearEvalAutomorphismKeys()
        except Exception:
            pass
        try:
            ReleaseAllContexts()
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
