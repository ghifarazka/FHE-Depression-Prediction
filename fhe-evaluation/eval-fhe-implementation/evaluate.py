"""
evaluate.py

Evaluation orchestrator for FHE inference benchmarking.
Runs worker.py as a subprocess for each of the first 50 samples in
data/dass42_test.csv, collects metrics, and writes 4 CSVs:
  - results/eval_accuracy.csv
  - results/eval_time.csv
  - results/eval_memory.csv
  - results/eval_errors.csv

Also prints total wall-clock time and a projected estimate for all
5678 samples.
"""

import subprocess
import sys
import os
import json
import csv
import time

# ── Configuration ──────────────────────────────────────────────────
NUM_SAMPLES   = 5678            # head(50) for this trial run
FULL_DATASET  = 5678            # used only for time projection
EPSILON       = 1e-10           # 1e-10 as specified
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
RESULTS_DIR   = "results"
PYTHON_EXE    = sys.executable  # use the same Python that runs this script

# ── Output paths ───────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
PATH_ACCURACY = os.path.join(RESULTS_DIR, "eval_accuracy.csv")

PATH_TIME     = os.path.join(RESULTS_DIR, "eval_time.csv")
PATH_MEMORY   = os.path.join(RESULTS_DIR, "eval_memory.csv")
PATH_ERRORS   = os.path.join(RESULTS_DIR, "eval_errors.csv")

# ── CSV writers setup ──────────────────────────────────────────────
f_acc = open(PATH_ACCURACY, "w", newline="")
f_tim = open(PATH_TIME,     "w", newline="")
f_mem = open(PATH_MEMORY,   "w", newline="")
f_err = open(PATH_ERRORS,   "w", newline="")

w_acc = csv.writer(f_acc)
w_tim = csv.writer(f_tim)
w_mem = csv.writer(f_mem)
w_err = csv.writer(f_err)

w_acc.writerow(["sample_id", "result_plain", "result_fhe", "abs_error", "rel_error"])
w_tim.writerow(["sample_id", "time_plain", "time_fhe_setup", "time_fhe_encrypt",
                "time_fhe_inference", "time_fhe_decrypt", "time_fhe_total", "time_slowdown"])
w_mem.writerow(["sample_id", "mem_plain", "mem_fhe_setup", "mem_fhe_encrypt",
                "mem_fhe_inference", "mem_fhe_decrypt", "mem_fhe_total", "mem_overhead"])
w_err.writerow(["sample_id", "error_note"])

# ── Helper: flush all writers ──────────────────────────────────────
def flush_all():
    for f in (f_acc, f_tim, f_mem, f_err):
        f.flush()

# ── Main loop ─────────────────────────────────────────────────────
eval_start = time.perf_counter()
completed_times = []   # wall-clock seconds per successful sample (for projection)

print(f"{'='*60}")
print(f"  FHE Evaluation — {NUM_SAMPLES} samples (trial run)")
print(f"{'='*60}\n")

for sample_idx in range(NUM_SAMPLES):
    sample_wall_start = time.perf_counter()
    print(f"[{sample_idx+1:>3}/{NUM_SAMPLES}] Sample {sample_idx} ... ", end="", flush=True)

    # ── Spawn worker subprocess ────────────────────────────────────
    try:
        proc = subprocess.run(
            [PYTHON_EXE, WORKER_SCRIPT, str(sample_idx)],
            capture_output=True,
            text=True,
            timeout=300,   # 5-minute hard cap per sample
        )
    except subprocess.TimeoutExpired:
        err_msg = "TIMEOUT: subprocess exceeded 300 s"
        print(f"TIMEOUT")
        w_err.writerow([sample_idx, err_msg])

        # Write NaN rows to all metric CSVs
        w_acc.writerow([sample_idx] + ["NaN"] * 3)
        w_tim.writerow([sample_idx] + ["NaN"] * 7)
        w_mem.writerow([sample_idx] + ["NaN"] * 7)
        flush_all()
        continue
    except Exception as e:
        err_msg = f"SUBPROCESS LAUNCH ERROR: {e}"
        print(f"ERROR")
        w_err.writerow([sample_idx, err_msg])
        w_acc.writerow([sample_idx] + ["NaN"] * 3)
        w_tim.writerow([sample_idx] + ["NaN"] * 7)
        w_mem.writerow([sample_idx] + ["NaN"] * 7)
        flush_all()
        continue

    # ── Parse worker output ────────────────────────────────────────
    raw_stdout = proc.stdout.strip()
    raw_stderr = proc.stderr.strip()

    # Show any stderr from worker (e.g. OpenFHE info prints)
    if raw_stderr:
        # OpenFHE often prints noise to stderr; only show if it looks like a real error
        if "Error" in raw_stderr or "error" in raw_stderr or "Traceback" in raw_stderr:
            print(f"\n  [stderr] {raw_stderr[:300]}")

    if proc.returncode != 0 or not raw_stdout:
        err_msg = (
            f"returncode={proc.returncode} | "
            f"stderr={raw_stderr[:300] if raw_stderr else '(empty)'} | "
            f"stdout={raw_stdout[:200] if raw_stdout else '(empty)'}"
        )
        print(f"FAILED (rc={proc.returncode})")
        w_err.writerow([sample_idx, err_msg])
        w_acc.writerow([sample_idx] + ["NaN"] * 3)
        w_tim.writerow([sample_idx] + ["NaN"] * 7)
        w_mem.writerow([sample_idx] + ["NaN"] * 7)
        flush_all()
        continue

    try:
        data = json.loads(raw_stdout)
    except json.JSONDecodeError as e:
        err_msg = f"JSON PARSE ERROR: {e} | stdout={raw_stdout[:300]}"
        print(f"JSON ERROR")
        w_err.writerow([sample_idx, err_msg])
        w_acc.writerow([sample_idx] + ["NaN"] * 3)
        w_tim.writerow([sample_idx] + ["NaN"] * 7)
        w_mem.writerow([sample_idx] + ["NaN"] * 7)
        flush_all()
        continue

    # ── Worker reported an internal error ─────────────────────────
    if data.get("error"):
        err_msg = data["error"]
        print(f"WORKER ERROR: {err_msg}")
        w_err.writerow([sample_idx, err_msg])
        w_acc.writerow([sample_idx] + ["NaN"] * 3)
        w_tim.writerow([sample_idx] + ["NaN"] * 7)
        w_mem.writerow([sample_idx] + ["NaN"] * 7)
        flush_all()
        continue

    # ── Extract metrics ────────────────────────────────────────────
    result_plain = data["result_plain"]
    result_fhe   = data["result_fhe"]

    time_plain         = data["time_plain"]
    time_fhe_setup     = data["time_fhe_setup"]
    time_fhe_encrypt   = data["time_fhe_encrypt"]
    time_fhe_inference = data["time_fhe_inference"]
    time_fhe_decrypt   = data["time_fhe_decrypt"]

    mem_plain         = data["mem_plain"]
    mem_fhe_setup     = data["mem_fhe_setup"]
    mem_fhe_encrypt   = data["mem_fhe_encrypt"]
    mem_fhe_inference = data["mem_fhe_inference"]
    mem_fhe_decrypt   = data["mem_fhe_decrypt"]

    # ── Derived metrics ────────────────────────────────────────────
    time_fhe_total = time_fhe_setup + time_fhe_encrypt + time_fhe_inference + time_fhe_decrypt
    # mem_fhe_decrypt is the final cumulative RSS snapshot = peak total FHE memory
    mem_fhe_total  = mem_fhe_decrypt

    abs_error     = abs(result_fhe - result_plain)
    rel_error     = abs(result_fhe - result_plain) / (abs(result_plain) + EPSILON)
    time_slowdown = time_fhe_total / (time_plain + EPSILON)
    mem_overhead  = mem_fhe_total  / (mem_plain  + EPSILON)

    # ── Write rows ─────────────────────────────────────────────────
    w_acc.writerow([sample_idx, result_plain, result_fhe, abs_error, rel_error])

    w_tim.writerow([
        sample_idx,
        time_plain,
        time_fhe_setup,
        time_fhe_encrypt,
        time_fhe_inference,
        time_fhe_decrypt,
        time_fhe_total,
        time_slowdown,
    ])

    w_mem.writerow([
        sample_idx,
        mem_plain,
        mem_fhe_setup,
        mem_fhe_encrypt,
        mem_fhe_inference,
        mem_fhe_decrypt,
        mem_fhe_total,
        mem_overhead,
    ])

    flush_all()

    # ── Per-sample timing ──────────────────────────────────────────
    sample_wall = time.perf_counter() - sample_wall_start
    completed_times.append(sample_wall)

    # Rolling ETA
    avg_so_far = sum(completed_times) / len(completed_times)
    remaining  = NUM_SAMPLES - (sample_idx + 1)
    eta_trial  = avg_so_far * remaining

    print(
        f"OK  |  plain={result_plain:.4f}  fhe={result_fhe:.4f}  "
        f"rel_err={rel_error:.2e}  "
        f"wall={sample_wall:.1f}s  ETA(trial)={eta_trial:.0f}s"
    )

# ── Close file handles ────────────────────────────────────────────
for f in (f_acc, f_tim, f_mem, f_err):
    f.close()

# ── Summary ───────────────────────────────────────────────────────
eval_total = time.perf_counter() - eval_start
successful = len(completed_times)
failed     = NUM_SAMPLES - successful

print(f"\n{'='*60}")
print(f"  Trial run complete")
print(f"{'='*60}")
print(f"  Samples attempted : {NUM_SAMPLES}")
print(f"  Successful        : {successful}")
print(f"  Failed / skipped  : {failed}")
print(f"  Total wall time   : {eval_total:.2f} s  ({eval_total/60:.2f} min)")

if successful > 0:
    avg_per_sample = eval_total / successful
    projected_full = avg_per_sample * FULL_DATASET
    print(f"\n  Avg time per sample       : {avg_per_sample:.2f} s")
    print(f"  Projected time (5678 samp): {projected_full:.0f} s"
          f"  ≈  {projected_full/3600:.1f} hours")

print(f"\n  Output files:")
print(f"    {PATH_ACCURACY}")
print(f"    {PATH_TIME}")
print(f"    {PATH_MEMORY}")
print(f"    {PATH_ERRORS}")
print(f"{'='*60}\n")