"""
fhe_client.py

This is the "local backend" piece of the client side. It takes raw answers
straight from the HTML form, preprocesses them exactly like the model was
trained, encrypts them with FHE (same recipe as DEBUG_client.py), calls the
remote FHE server for inference, decrypts the result, and turns the raw
number into a depression score + severity level.

Everything here happens on the user's own machine. The only thing that ever
leaves this machine is ciphertext (and the CryptoContext/rotation keys needed
to operate on it) -- the secret key that can decrypt never leaves.
"""

import os
import math
import base64
import threading

import joblib
import pandas as pd
import requests
from openfhe import *

from dass42_questionnaire import fields_DASS42, fields_TIPI, fields_demographics

# ========================= CONFIG / CONSTANTS =========================

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREPROCESSOR_PATH = os.path.join(_HERE, "preprocessor.joblib")

# All fields the preprocessor expects, in (name, label, options) form. Used
# to know, for every question, what Python type the raw form value should be
# cast to before handing it to the preprocessor (e.g. "2" -> 2 vs "2" -> 2.0
# vs "White" -> "White").
_ALL_FIELDS = fields_DASS42 + fields_TIPI + fields_demographics

# Severity table (matches the depression score -> label table used by this
# app). Upper bound of each bucket is exclusive except for the last one.
_SEVERITY_TABLE = [
    (10, "Normal"),
    (14, "Mild"),
    (21, "Moderate"),
    (28, "Severe"),
]
_SEVERITY_LAST_LABEL = "Extremely Severe"

# The openfhe python bindings keep a fair amount of global/static state
# (crypto context registries etc.). To keep things simple and safe if two
# submissions ever land at the same time, we serialize FHE work through a
# single lock rather than letting two encryptions/decryptions interleave.
_FHE_LOCK = threading.Lock()

_PREPROCESSOR = joblib.load(_PREPROCESSOR_PATH)


# ========================= HELPER FUNCTIONS =========================

def next_power_of_two(x):
    """Returns the next power of 2 after x. If x <= 1, returns 1."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def serialize_to_base64(obj):
    """Takes any FHE object and turns it into base64."""
    try:
        ser = Serialize(obj, BINARY)
        return base64.b64encode(ser).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error serializing FHE object: {e}")


def serialize_EvalAutomorphismKey_to_base64():
    """Takes EvalAutomorphismKey and turns it into base64."""
    try:
        ser = SerializeEvalAutomorphismKeyString(BINARY, "")
        return base64.b64encode(ser).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error serializing EvalAutomorphismKey: {e}")


def deserialize_Ciphertext_from_base64(ciphertext_ser):
    """Takes base64 and turns it into an FHE Ciphertext object."""
    try:
        bin_str = base64.b64decode(ciphertext_ser)
        return DeserializeCiphertextString(bin_str, BINARY)
    except Exception as e:
        raise RuntimeError(f"Error deserializing ciphertext: {e}")


def _cast_value(raw_value, options):
    """
    Casts a raw value coming from the HTML form (always a string) to the
    same Python type used for that field's options in
    dass42_questionnaire.py (int, float, or str), so it matches what the
    preprocessor was fit on.
    """
    sample = options[0][0]
    if isinstance(sample, float):
        return float(raw_value)
    if isinstance(sample, int):
        return int(raw_value)
    return str(raw_value)


def build_input_dataframe(form_data):
    """
    Turns the {question_name: raw_value} dict submitted by the form into a
    single-row DataFrame with the right dtypes, ready for
    preprocessor.transform(...).
    """
    row = {}
    missing = []
    for q_name, _label, options in _ALL_FIELDS:
        if q_name not in form_data or form_data[q_name] in (None, ""):
            missing.append(q_name)
            continue
        row[q_name] = _cast_value(form_data[q_name], options)

    if missing:
        raise ValueError(f"Missing answer(s) for: {', '.join(missing)}")

    return pd.DataFrame([row])


def score_to_level(score):
    """Maps a raw depression score to a severity label."""
    for upper_bound, label in _SEVERITY_TABLE:
        if score < upper_bound:
            return label
    return _SEVERITY_LAST_LABEL


def _notify(progress_callback, message):
    if progress_callback is not None:
        progress_callback(message)


# ============================= MAIN ENTRY =============================

def process_fhe(server_url, form_data, progress_callback=None):
    """
    Runs the full pipeline for one form submission:

      1. Preprocess the raw answers the same way the model was trained on.
      2. Set up a fresh CryptoContext + keypair for this request.
      3. Encrypt the preprocessed vector.
      4. Send the ciphertext (+ CryptoContext + rotation keys) to the
         remote FHE server for inference.
      5. Decrypt the encrypted result with the local secret key (which
         never leaves this machine).
      6. Convert the raw score into a depression score + severity level.

    Returns: {"score": float, "level": str}
    Raises: ValueError for bad/missing form input, RuntimeError for FHE or
            network failures.
    """
    with _FHE_LOCK:
        # ---- 1. Preprocess ----
        _notify(progress_callback, "Preparing your answers...")
        X_new = build_input_dataframe(form_data)
        X_new_pre = _PREPROCESSOR.transform(X_new)
        x = X_new_pre.flatten().tolist()
        n = len(x)

        # ---- 2. CryptoContext & KeyGen ----
        multDepth = 1
        scaleModSize = 50
        batchSize = next_power_of_two(n)
        rotations = [2 ** i for i in range(int(math.log2(batchSize // 2)) + 1)]

        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(multDepth)
        params.SetScalingModSize(scaleModSize)
        params.SetBatchSize(batchSize)
        cryptoContext = GenCryptoContext(params)

        cryptoContext.Enable(PKESchemeFeature.PKE)
        cryptoContext.Enable(PKESchemeFeature.KEYSWITCH)
        cryptoContext.Enable(PKESchemeFeature.LEVELEDSHE)
        cryptoContext.Enable(PKESchemeFeature.ADVANCEDSHE)

        keypair = cryptoContext.KeyGen()
        publicKey = keypair.publicKey
        secretKey = keypair.secretKey
        cryptoContext.EvalRotateKeyGen(secretKey, rotations)

        # ---- 3. Encrypt ----
        _notify(progress_callback, "Encrypting your data...")
        pt = cryptoContext.MakeCKKSPackedPlaintext(x)
        ct = cryptoContext.Encrypt(publicKey, pt)

        cc_ser = serialize_to_base64(cryptoContext)
        evalauto_ser = serialize_EvalAutomorphismKey_to_base64()
        ct_ser = serialize_to_base64(ct)

        # ---- 4. Send to server ----
        _notify(progress_callback, "Sending encrypted data to server...")
        payload = {
            "cryptoContext": cc_ser,
            "evalAutomorphismKey": evalauto_ser,
            "ciphertext": ct_ser,
        }
        endpoint = server_url.rstrip("/") + "/fhe-predict"

        try:
            _notify(progress_callback, "Waiting for the server to compute...")
            resp = requests.post(endpoint, json=payload, timeout=180)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not reach FHE server at {endpoint}: {e}")

        try:
            resultEncrypted_ser = resp.json()["resultEncrypted"]
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Unexpected response from FHE server: {e}")

        # ---- 5. Decrypt ----
        _notify(progress_callback, "Decrypting your result...")
        resultEncrypted = deserialize_Ciphertext_from_base64(resultEncrypted_ser)
        decrypted = cryptoContext.Decrypt(resultEncrypted, secretKey)
        score = decrypted.GetRealPackedValue()[0]

        # ---- 6. Interpret ----
        level = score_to_level(score)

        return {"score": round(float(score), 2), "level": level}
