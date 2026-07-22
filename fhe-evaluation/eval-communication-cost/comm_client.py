"""
comm_client.py

Client-side driver for the communication-cost evaluation of the FHE
client-server architecture.

This script does NOT measure inference (that's `eval-fhe-implementation/`).
It isolates the cost of splitting client and server apart -- serialization,
network transfer, and deserialization -- using the single sample in
`data/dass42_test_singular.csv`, matching base.py's encrypted-inference
setup exactly (multDepth=1, ADVANCEDSHE enabled, no EvalMultKey -- this
project's base.py doesn't need it, and neither does this script).

Protocol:
  1. Set up CryptoContext + RotateKey + Ciphertext ONCE, before the
     timed loop. KeyGen/Encrypt cost is already measured elsewhere
     (eval-fhe-implementation/); re-timing it here would double count it.
  2. Measure network conditions once (iperf3 bandwidth + ping latency),
     immediately before the timed trials.
  3. Run NUM_TRIALS trials, TRIAL_GAP_SEC apart. Each trial:
       (a) fresh-serialize CryptoContext + RotateKey + Ciphertext -> JSON
       (b) POST to the server, timing the full request/response
       (c) the server deserializes, then immediately reserializes the
           SAME ciphertext (no inference -- that's covered elsewhere)
           and self-reports its own two timings in the response body
       (d) deserialize the returned ciphertext
       (e) sanity check: decrypt the round-tripped ciphertext and diff
           against the original input, to catch silent corruption

Why there's no per-direction ("send" vs "return") network timing:
  Splitting those apart would require the client's and server's clocks
  to agree to sub-millisecond precision, which we don't have (no PTP,
  just best-effort NTP at most). Instead this script reports:
    - time_client_total_request : client wall clock bracketing ONLY the
      requests.post() call -- the literal "communication cost" as
      originally defined (time from send to receive). Does NOT include
      client-side serialize/deserialize, which happen outside this
      bracket.
    - time_server_receive       : how long the server blocked waiting
      for the request body to physically arrive over the socket --
      network-bound, not CPU-bound. Reported separately from
      time_server_json_parse/time_server_deserialize (both CPU-bound)
      because conflating them, as an earlier version of this script
      did, is harmless on loopback but silently misattributes real
      network transit time as server compute once client and server
      are genuinely remote from each other.
    - time_server_cpu           : time_server_json_parse +
      time_server_deserialize + time_server_reserialize -- the CPU-bound
      subset of server work only, excluding the receive-wait.
    - time_process_total        : client_serialize + time_server_cpu +
      client_deserialize. A separate, complementary headline number --
      NOT a quantity to net against total_request, since part of it
      (client serialize/deserialize) was never inside that bracket.
    - time_network_derived      : total_request - time_server_cpu --
      the residual round-trip network time (both directions, PLUS the
      server's receive-wait, PLUS any socket/framework overhead not
      captured by the explicit timers). Only time_server_cpu may be
      subtracted here, since it's the only genuinely CPU-bound
      sub-component measured inside the total_request window.
    - time_full_pipeline        : time_client_serialize +
      time_client_total_request + time_client_deserialize -- the true
      start-to-finish time, equal to time_process_total +
      time_network_derived.

Usage:
    python comm_client.py
"""

import base64
import csv
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from openfhe import (
    BINARY,
    CCParamsCKKSRNS,
    DeserializeCiphertextString,
    GenCryptoContext,
    PKESchemeFeature,
    ReleaseAllContexts,
    Serialize,
    SerializeEvalAutomorphismKeyString,
)

# ── Configuration ──────────────────────────────────────────────────
SERVER_HOST = "148.230.103.61"   # TODO: replace with the real remote host for actual testing
SERVER_PORT = 8001
SERVER_URL  = f"http://{SERVER_HOST}:{SERVER_PORT}/comm-echo"

IPERF_PORT     = 5201       # requires `iperf3 -s -p 5201` already running on SERVER_HOST
IPERF_DURATION = 5          # seconds, per direction

NUM_TRIALS    = 10
TRIAL_GAP_SEC = 30

DATA_PATH = "../data/dass42_test_singular.csv"

RESULTS_DIR  = "results"
PATH_COMM    = os.path.join(RESULTS_DIR, "eval_communication.csv")
PATH_ERRORS  = os.path.join(RESULTS_DIR, "eval_communication_errors.csv")
PATH_NETWORK = os.path.join(RESULTS_DIR, "network_conditions.csv")

REQUEST_TIMEOUT_SEC = 180   # bumped from 60s: real-network trials have been observed
                            # taking 30-57s already for a ~21MB payload, too close to
                            # a 60s cap for comfort

os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================= HELPER FUNCTIONS =========================

def next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def serialize_to_base64(obj):
    ser = Serialize(obj, BINARY)
    return base64.b64encode(ser).decode("utf-8")


def serialize_EvalAutomorphismKey_to_base64():
    ser = SerializeEvalAutomorphismKeyString(BINARY, "")
    return base64.b64encode(ser).decode("utf-8")


def deserialize_Ciphertext_from_base64(ciphertext_ser):
    bin_str = base64.b64decode(ciphertext_ser)
    return DeserializeCiphertextString(bin_str, BINARY)


# ===================== NETWORK CONDITIONS CHECK ======================

def run_iperf3(host, port, reverse, duration):
    """Returns throughput in Mbps, or None if iperf3 isn't installed,
    the iperf3 server isn't running on the target host, or the call
    fails for any other reason."""
    cmd = ["iperf3", "-c", host, "-p", str(port), "-t", str(duration), "-J"]
    if reverse:
        cmd.append("-R")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 15)
        data = json.loads(proc.stdout)
        # Receiver-side goodput -- more reliable than sender-side.
        return data["end"]["sum_received"]["bits_per_second"] / 1e6
    except Exception:
        return None


def run_ping(host, count=5):
    """Returns dict of min/avg/max/mdev RTT in ms, or None on failure."""
    cmd = ["ping", "-c", str(count), host]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=count * 2 + 10)
        for line in proc.stdout.splitlines():
            if "min/avg/max" in line:
                stats = line.split("=")[1].strip().split()[0]
                mn, avg, mx, mdev = (float(v) for v in stats.split("/"))
                return {"min": mn, "avg": avg, "max": mx, "mdev": mdev}
    except Exception:
        pass
    return None


def measure_network_conditions():
    """
    Logged ONCE, immediately before the timed trials, per the protocol
    ("network conditions ... immediately before the communication cost
    measurement is performed"). If iperf3 (or its server) isn't
    available, bandwidth fields are written as blank and everything
    else proceeds normally.
    """
    print("Measuring network conditions (iperf3 + ping)...")
    upload_mbps   = run_iperf3(SERVER_HOST, IPERF_PORT, reverse=False, duration=IPERF_DURATION)
    download_mbps = run_iperf3(SERVER_HOST, IPERF_PORT, reverse=True,  duration=IPERF_DURATION)
    ping_stats    = run_ping(SERVER_HOST) or {}

    row = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "target_host":   SERVER_HOST,
        "iperf_port":    IPERF_PORT,
        "upload_mbps":   upload_mbps,
        "download_mbps": download_mbps,
        "ping_min_ms":   ping_stats.get("min"),
        "ping_avg_ms":   ping_stats.get("avg"),
        "ping_max_ms":   ping_stats.get("max"),
        "ping_mdev_ms":  ping_stats.get("mdev"),
    }

    write_header = not os.path.exists(PATH_NETWORK)
    with open(PATH_NETWORK, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"  upload   : {upload_mbps if upload_mbps is not None else 'n/a'} Mbps")
    print(f"  download : {download_mbps if download_mbps is not None else 'n/a'} Mbps")
    print(f"  ping avg : {ping_stats.get('avg', 'n/a')} ms\n")


# ============================ MAIN SETUP =============================

test_data_singular = pd.read_csv(DATA_PATH)
x = test_data_singular.iloc[0].astype(float).tolist()
n = len(x)

multDepth    = 1        # matches base.py
scaleModSize = 50        # matches base.py
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

# No EvalMultKeyGen -- not needed, matches base.py.
cryptoContext.EvalRotateKeyGen(secretKey, rotations)

pt = cryptoContext.MakeCKKSPackedPlaintext(x)
ct = cryptoContext.Encrypt(publicKey, pt)

# ── CSV writers ────────────────────────────────────────────────────
COMM_HEADER = [
    "trial_id", "timestamp", "bytes_sent", "bytes_received",
    "time_client_serialize", "time_client_total_request",
    "time_server_receive", "time_server_json_parse",
    "time_server_deserialize", "time_server_reserialize",
    "time_server_cpu", "time_server_total",
    "time_client_deserialize",
    "time_process_total", "time_network_derived", "time_full_pipeline",
    "max_abs_diff",
]

write_header_comm = not os.path.exists(PATH_COMM)
f_comm = open(PATH_COMM, "a", newline="")
w_comm = csv.writer(f_comm)
if write_header_comm:
    w_comm.writerow(COMM_HEADER)

write_header_err = not os.path.exists(PATH_ERRORS)
f_err = open(PATH_ERRORS, "a", newline="")
w_err = csv.writer(f_err)
if write_header_err:
    w_err.writerow(["trial_id", "error_note"])


def flush_all():
    f_comm.flush()
    f_err.flush()


# ============================= MAIN LOOP =============================

measure_network_conditions()

print(f"{'='*60}")
print(f"  Communication cost evaluation -- {NUM_TRIALS} trials")
print(f"  target: {SERVER_URL}")
print(f"{'='*60}\n")

successful_rows = []

for trial_id in range(NUM_TRIALS):
    print(f"[{trial_id + 1:>2}/{NUM_TRIALS}] trial {trial_id} ... ", end="", flush=True)

    try:
        # ── (1) Serialize payload ───────────────────────────────────
        t0 = time.perf_counter()
        cc_ser       = serialize_to_base64(cryptoContext)
        evalauto_ser = serialize_EvalAutomorphismKey_to_base64()
        ct_ser       = serialize_to_base64(ct)
        payload_json = json.dumps({
            "cryptoContext": cc_ser,
            "evalAutomorphismKey": evalauto_ser,
            "ciphertext": ct_ser,
        })
        t1 = time.perf_counter()
        time_client_serialize = t1 - t0
        bytes_sent = len(payload_json.encode("utf-8"))

        # ── (2)+(3)+(4)+(5) send / server processes / return ────────
        t2 = time.perf_counter()
        response = requests.post(
            SERVER_URL,
            data=payload_json,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        t3 = time.perf_counter()
        time_client_total_request = t3 - t2
        bytes_received = len(response.content)

        resp_data = response.json()
        if resp_data.get("error"):
            raise RuntimeError(f"server-side error: {resp_data['error']}")

        time_server_receive     = resp_data["time_server_receive"]
        time_server_json_parse  = resp_data["time_server_json_parse"]
        time_server_deserialize = resp_data["time_server_deserialize"]
        time_server_reserialize = resp_data["time_server_reserialize"]
        # Everything the server measured, receive-wait included -- kept
        # for transparency, but NOT what gets subtracted from
        # total_request (see time_server_cpu below).
        time_server_total = (
            time_server_receive + time_server_json_parse
            + time_server_deserialize + time_server_reserialize
        )
        # Only the CPU-bound subset: parsing + FHE object work done on
        # bytes already sitting in memory. This excludes the blocking
        # socket read (time_server_receive), which is network-bound and
        # belongs in time_network_derived instead.
        time_server_cpu = (
            time_server_json_parse + time_server_deserialize + time_server_reserialize
        )

        # ── (6) client deserializes result ───────────────────────────
        t4 = time.perf_counter()
        result_ct = deserialize_Ciphertext_from_base64(resp_data["resultEncrypted"])
        t5 = time.perf_counter()
        time_client_deserialize = t5 - t4

        # ── Derived aggregates ────────────────────────────────────────
        # time_client_total_request only ever bracketed the requests.post()
        # call -- it never included client-side serialize/deserialize, so
        # only server-side work that's genuinely CPU-bound (time_server_cpu)
        # can be subtracted from it to isolate network transit. Using
        # time_server_total here would double count the receive-wait,
        # which is network time, not processing time, and using
        # process_total (as an earlier version of this script did) was a
        # bug: it double-subtracted client-side time that was never part
        # of this window at all, driving the result negative on every
        # trial.
        time_process_total = (
            time_client_serialize + time_server_cpu + time_client_deserialize
        )
        time_network_derived = time_client_total_request - time_server_cpu

        # The true start-to-finish time: from the moment the client began
        # serializing to the moment it finished deserializing the result.
        # Equivalently, time_process_total + time_network_derived.
        time_full_pipeline = (
            time_client_serialize + time_client_total_request + time_client_deserialize
        )

        # ── Sanity check: does the round-tripped ciphertext still
        #    decrypt to the original input? Catches silent corruption
        #    introduced anywhere in the (de)serialize cycle. ──────────
        decrypted = cryptoContext.Decrypt(result_ct, secretKey)
        decrypted_vals = decrypted.GetRealPackedValue()[:n]
        max_abs_diff = max(abs(a - b) for a, b in zip(decrypted_vals, x))

        row = [
            trial_id, datetime.now(timezone.utc).isoformat(),
            bytes_sent, bytes_received,
            time_client_serialize, time_client_total_request,
            time_server_receive, time_server_json_parse,
            time_server_deserialize, time_server_reserialize,
            time_server_cpu, time_server_total,
            time_client_deserialize,
            time_process_total, time_network_derived, time_full_pipeline,
            max_abs_diff,
        ]
        w_comm.writerow(row)
        flush_all()
        successful_rows.append(row)

        print(
            f"OK  |  total={time_client_total_request:.4f}s  "
            f"process={time_process_total:.4f}s  "
            f"network(derived)={time_network_derived:.4f}s  "
            f"full_pipeline={time_full_pipeline:.4f}s  "
            f"srv_receive={time_server_receive:.4f}s  "
            f"sent={bytes_sent}B  recv={bytes_received}B  "
            f"max_abs_diff={max_abs_diff:.2e}"
        )

    except Exception as e:
        print(f"FAILED: {e}")
        w_err.writerow([trial_id, str(e)])
        w_comm.writerow([trial_id] + ["NaN"] * (len(COMM_HEADER) - 1))
        flush_all()

    if trial_id < NUM_TRIALS - 1:
        time.sleep(TRIAL_GAP_SEC)

f_comm.close()
f_err.close()

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Communication cost evaluation complete")
print(f"{'='*60}")
print(f"  Trials attempted : {NUM_TRIALS}")
print(f"  Successful       : {len(successful_rows)}")

if successful_rows:
    totals    = [r[5] for r in successful_rows]
    processes = [r[13] for r in successful_rows]
    networks  = [r[14] for r in successful_rows]
    pipelines = [r[15] for r in successful_rows]
    print(
        f"\n  time_client_total_request : mean={statistics.mean(totals):.4f}s"
        f"  stdev={(statistics.stdev(totals) if len(totals) > 1 else 0):.4f}s"
    )
    print(
        f"  time_process_total        : mean={statistics.mean(processes):.4f}s"
        f"  stdev={(statistics.stdev(processes) if len(processes) > 1 else 0):.4f}s"
    )
    print(
        f"  time_network_derived      : mean={statistics.mean(networks):.4f}s"
        f"  stdev={(statistics.stdev(networks) if len(networks) > 1 else 0):.4f}s"
    )
    print(
        f"  time_full_pipeline        : mean={statistics.mean(pipelines):.4f}s"
        f"  stdev={(statistics.stdev(pipelines) if len(pipelines) > 1 else 0):.4f}s"
    )

print(f"\n  Output files:")
print(f"    {PATH_COMM}")
print(f"    {PATH_ERRORS}")
print(f"    {PATH_NETWORK}")
print(f"{'='*60}\n")

# ── Clean up ────────────────────────────────────────────────────────
cryptoContext.ClearEvalAutomorphismKeys()
ReleaseAllContexts()
