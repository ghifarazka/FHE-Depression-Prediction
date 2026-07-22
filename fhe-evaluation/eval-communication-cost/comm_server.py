"""
comm_server.py

Server-side endpoint for the communication-cost evaluation. Deliberately
does NOT run any inference -- it only deserializes the incoming
CryptoContext + RotateKey + Ciphertext, then immediately reserializes
that SAME ciphertext and sends it back, to represent (without actually
computing) "the encrypted inference result" being transmitted back to
the client. Real inference cost is measured separately, in
`eval-fhe-implementation/`.

Because this process stays alive across all of comm_client.py's trials
(unlike worker.py, which got a fresh interpreter every sample via
subprocess), OpenFHE's global CryptoContext registry has to be
explicitly cleared after every single request -- otherwise state from
one request could interfere with the next. See the `finally` block
below. Flask is also run single-threaded so two requests can never
touch that global state concurrently.

time_server_receive isolates the blocking socket read of the request
body (network-bound: how long the server waited for the bytes to
arrive) from time_server_json_parse and time_server_deserialize (both
CPU-bound: work done on bytes already in memory). Conflating these, as
an earlier version of this script did, is harmless on loopback but
silently misattributes real network transit time as server compute
once client and server are genuinely remote from each other.

Usage:
    python comm_server.py
"""

import base64
import json
import time

from flask import Flask, jsonify, request
from openfhe import (
    BINARY,
    DeserializeCiphertextString,
    DeserializeCryptoContextString,
    DeserializeEvalAutomorphismKeyString,
    ReleaseAllContexts,
    Serialize,
)

app = Flask(__name__)

# ========================= HELPER FUNCTIONS =========================

def serialize_to_base64(obj):
    ser = Serialize(obj, BINARY)
    return base64.b64encode(ser).decode("utf-8")


def deserialize_CryptoContext_from_base64(cc_ser):
    bin_str = base64.b64decode(cc_ser)
    return DeserializeCryptoContextString(bin_str, BINARY)


def deserialize_EvalAutomorphismKey_from_base64(evalauto_ser):
    bin_str = base64.b64decode(evalauto_ser)
    DeserializeEvalAutomorphismKeyString(bin_str, BINARY)


def deserialize_Ciphertext_from_base64(ct_ser):
    bin_str = base64.b64decode(ct_ser)
    return DeserializeCiphertextString(bin_str, BINARY)


# ============================= MAIN APP =============================

@app.route("/comm-echo", methods=["POST"])
def comm_echo():
    """
    Deserializes CryptoContext + RotateKey + Ciphertext, then
    immediately reserializes the SAME ciphertext and returns it --
    simulating the transmission of an inference result without
    actually computing one.
    """
    cryptoContext = None
    try:
        # request.get_json() conflates two very different things: (1)
        # blocking on the socket until the full request body has
        # physically arrived -- network-bound, and (2) parsing that body
        # as JSON -- CPU-bound. On loopback, (1) is instantaneous so this
        # distinction didn't matter. Over a real network, (1) can be the
        # dominant cost, and lumping it in with "deserialize" would
        # misattribute genuine network transit time as server compute.
        # request.get_data() forces the same blocking read, timed alone.
        t_recv0 = time.perf_counter()
        raw_body = request.get_data()
        t_recv1 = time.perf_counter()
        time_server_receive = t_recv1 - t_recv0

        t_parse0 = time.perf_counter()
        payload = json.loads(raw_body)
        t_parse1 = time.perf_counter()
        time_server_json_parse = t_parse1 - t_parse0

        # ── Deserialize (FHE object reconstruction only) ──────────────
        t0 = time.perf_counter()
        cryptoContext = deserialize_CryptoContext_from_base64(payload["cryptoContext"])
        deserialize_EvalAutomorphismKey_from_base64(payload["evalAutomorphismKey"])
        ct = deserialize_Ciphertext_from_base64(payload["ciphertext"])
        t1 = time.perf_counter()
        time_server_deserialize = t1 - t0

        # ── Reserialize (echo -- no inference happens here) ─────────
        t2 = time.perf_counter()

        result_ct_ser = serialize_to_base64(ct)

        t3 = time.perf_counter()
        time_server_reserialize = t3 - t2

        return jsonify({
            "resultEncrypted": result_ct_ser,
            "time_server_receive": time_server_receive,
            "time_server_json_parse": time_server_json_parse,
            "time_server_deserialize": time_server_deserialize,
            "time_server_reserialize": time_server_reserialize,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Reset OpenFHE's global context registry so state from this
        # request can't interfere with the next one -- this process
        # stays alive across every trial comm_client.py sends.
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
    # threaded=False: avoid two requests touching OpenFHE's global
    # context state concurrently.
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=False)
