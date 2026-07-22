"""
app.py

Local "backend" Flask app. This is what the user actually runs on their own
machine. It serves the DASS-42 form, and when it's submitted:

  1. Kicks off the FHE pipeline (preprocess -> encrypt -> call remote FHE
     server -> decrypt) in a background thread, so the request returns
     immediately with a job id.
  2. The page polls /status/<job_id> and shows live step-by-step progress.
  3. Once done, the page redirects to /result?job_id=... which renders the
     final depression score + severity level.

No plaintext answers or secret keys ever leave this machine -- only
ciphertext (and the CryptoContext/rotation keys needed to operate on it) is
sent to the remote FHE server.
"""

import threading
import uuid

from flask import Flask, render_template, request, jsonify

from dass42_questionnaire import fields_DASS42, fields_TIPI, fields_demographics
from fhe_client import process_fhe

app = Flask(__name__)

# ----------------------------- In-memory job store -----------------------------
# NOTE: this is fine for a single-process research/demo deployment. It is not
# persisted anywhere and does not survive a server restart, and is not meant
# to scale to many concurrent users.
JOBS = {}
JOBS_LOCK = threading.Lock()

_BADGE_CLASSES = {
    "Normal": "bg-success",
    "Mild": "bg-info text-dark",
    "Moderate": "bg-warning text-dark",
    "Severe": "bg-severe text-white",
    "Extremely Severe": "bg-danger",
}


def _update_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _run_job(job_id, server_url, form_data):
    def progress(message):
        _update_job(job_id, status=message)

    try:
        result = process_fhe(server_url, form_data, progress_callback=progress)
        _update_job(job_id, done=True, error=None, result=result, status="Done!")
    except Exception as e:
        _update_job(job_id, done=True, error=str(e), status="Failed")


@app.route("/")
def index():
    return render_template(
        "form.html",
        fields={
            "dass": fields_DASS42,
            "tipi": fields_TIPI,
            "demographics": fields_demographics,
        },
    )


@app.route("/submit", methods=["POST"])
def submit():
    """
    Accepts { "SERVER_URL": <string>, "FORM_DATA": { <q_name>: <val>, ... } },
    starts processing in the background, and immediately returns a job id
    the frontend can poll for progress.
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify(error="Invalid JSON payload: " + str(e)), 400

    if not isinstance(data, dict) or "SERVER_URL" not in data or "FORM_DATA" not in data:
        return jsonify(error="Expected JSON with 'SERVER_URL' and 'FORM_DATA' keys"), 400

    server_url = (data.get("SERVER_URL") or "").strip()
    form_data = data.get("FORM_DATA")

    if not server_url:
        return jsonify(error="Server URL is required"), 400
    if not isinstance(form_data, dict) or not form_data:
        return jsonify(error="Form data is missing or empty"), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "Queued...", "done": False, "error": None, "result": None}

    thread = threading.Thread(target=_run_job, args=(job_id, server_url, form_data), daemon=True)
    thread.start()

    return jsonify(job_id=job_id), 202


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Unknown job id"), 404
    return jsonify(job)


@app.route("/result")
def result():
    job_id = request.args.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None or not job.get("done") or not job.get("result"):
        return render_template(
            "result.html",
            error="No result found for this link. Please fill out the form again.",
        )

    score = job["result"]["score"]
    level = job["result"]["level"]
    return render_template(
        "result.html",
        score=score,
        level=level,
        badge_class=_BADGE_CLASSES.get(level, "bg-secondary"),
    )


if __name__ == "__main__":
    print("Starting Flask app. Open http://127.0.0.1:5001 in your browser.")
    # threaded=True is required: the background job thread and the /status
    # polling requests need to be served concurrently.
    app.run(host="127.0.0.1", port=5001, debug=True, threaded=True)
