"""Production Flask surveillance server with Danger Models, Datasets & Live Stream."""

import json
import mimetypes
import os
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from alert_manager import AlertManager, STATUSES, SEVERITIES
from config import (
    ALERTS_DIR, ALERT_COOLDOWN_SECONDS, DATASETS_DIR, DB_PATH,
    MAX_UPLOAD_BYTES, MODELS_DIR, RESULTS_DIR, UPLOAD_DIR
)
from dataset_manager import DatasetManager
from live_monitor import LiveMonitor
from main import run_analysis
from models.registry import ModelRegistry
from zone_drawer import validate_fractional_polygons

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_SORT_KEYS"] = False
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

lock = threading.RLock()
STATE = {
    "status": "idle",
    "job_id": None,
    "filename": None,
    "progress": 0.0,
    "frame": 0,
    "total_frames": 0,
    "processing_fps": 0.0,
    "error": None,
    "summary": None,
}
UPLOADS = {}

alert_manager = AlertManager(DB_PATH, ALERTS_DIR, ALERT_COOLDOWN_SECONDS)
live_monitor = LiveMonitor(alert_manager)
dataset_manager = DatasetManager()
model_registry = ModelRegistry()


def _json_error(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


def _status_update(data):
    with lock:
        STATE["status"] = data.get("status", STATE["status"])
        STATE["progress"] = data.get("percent", STATE["progress"])
        STATE["frame"] = data.get("frame", STATE["frame"])
        STATE["total_frames"] = data.get("total_frames", STATE["total_frames"])
        STATE["processing_fps"] = data.get("processing_fps", STATE["processing_fps"])
        if data.get("summary") is not None:
            STATE["summary"] = data["summary"]


def _analysis_worker(video_path: Path, config: dict, job_id: str):
    try:
        summary = run_analysis(video_path, config, _status_update, job_id)
        out_summary = Path(config["output_dir"]) / "summary.json"
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with lock:
            STATE.update(status="completed", progress=100.0, summary=summary, error=None)
    except Exception as exc:
        with lock:
            STATE.update(status="error", error=str(exc))


# ---------------- PAGES ----------------
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/video")
def video_page():
    return render_template("video.html")

@app.get("/roi")
def roi_page():
    return render_template("roi.html")

@app.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.get("/results")
def results_page():
    return render_template("results.html")

@app.get("/datasets")
def datasets_page():
    return render_template("datasets.html")

@app.get("/models")
def models_page():
    return render_template("models.html")


# ---------------- API: UPLOAD & ANALYSIS ----------------
@app.post("/api/upload")
def upload_video():
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return _json_error("Choose a video file first.")
    filename = secure_filename(uploaded.filename)
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return _json_error("Supported formats: MP4, AVI, MOV, MKV, WebM.")

    job_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{job_id}_{filename}"
    uploaded.save(dest)

    with lock:
        UPLOADS[job_id] = dest
        STATE.update(status="uploaded", job_id=job_id, filename=filename, progress=0)

    return jsonify({"success": True, "job_id": job_id, "filename": filename, "video_url": f"/api/uploaded-video/{job_id}"}), 201

@app.get("/api/uploaded-video/<job_id>")
def get_uploaded_video(job_id):
    path = UPLOADS.get(job_id)
    if not path or not path.is_file():
        return _json_error("Video not found.", 404)
    return send_file(path, mimetype=mimetypes.guess_type(path.name)[0] or "video/mp4")

@app.post("/api/start-analysis")
def start_analysis():
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id")
    config = payload.get("config", {})
    
    with lock:
        if STATE["status"] == "processing":
            return _json_error("Analysis already running.", 409)
        video_path = UPLOADS.get(job_id)
    
    if not video_path or not video_path.is_file():
        return _json_error("Upload a video first.")

    out_dir = RESULTS_DIR / job_id
    config["output_dir"] = str(out_dir)
    config["polygons"] = validate_fractional_polygons(config.get("polygons", []))

    with lock:
        STATE.update(status="processing", progress=0, frame=0, total_frames=0, error=None)

    threading.Thread(target=_analysis_worker, args=(video_path, config, job_id), daemon=True).start()
    return jsonify({"success": True, "job_id": job_id}), 202

@app.get("/api/status")
def get_status():
    with lock:
        return jsonify(STATE)

@app.get("/api/results/<job_id>/summary")
def get_results_summary(job_id):
    p = RESULTS_DIR / job_id / "summary.json"
    if not p.is_file():
        return _json_error("Results not ready.", 404)
    return jsonify(json.loads(p.read_text(encoding="utf-8")))

# CRITICAL FIX: Serves the processed video, clips, frames, and graphs to the browser
@app.get("/api/results/<job_id>/files/<path:filename>")
def get_result_file(job_id, filename):
    base_dir = (RESULTS_DIR / job_id).resolve()
    target_path = (base_dir / filename).resolve()

    if base_dir not in target_path.parents and target_path != base_dir:
        return _json_error("Forbidden.", 403)
    if not target_path.is_file():
        return _json_error("File not found.", 404)

    mime = mimetypes.guess_type(target_path.name)[0] or "application/octet-stream"
    return send_file(target_path, mimetype=mime, conditional=True)

@app.get("/api/results/<job_id>/download")
def download_results_zip(job_id):
    rdir = RESULTS_DIR / job_id
    if not rdir.is_dir():
        return _json_error("Results missing.", 404)
    zip_p = RESULTS_DIR / f"{job_id}_results.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in rdir.rglob("*"):
            if f.is_file() and not f.name.endswith(".zip"):
                zf.write(f, f.relative_to(rdir).as_posix())
    return send_file(zip_p, as_attachment=True, download_name=f"{job_id}_Surveillance_Results.zip")


# ---------------- API: ALERTS ----------------
@app.get("/api/alerts")
def get_alerts():
    return jsonify({
        "alerts": alert_manager.list_alerts(
            status=request.args.get("status"),
            severity=request.args.get("severity"),
            alert_type=request.args.get("type"),
            source=request.args.get("source"),
            limit=200
        )
    })

@app.patch("/api/alerts/<alert_id>")
def patch_alert(alert_id):
    data = request.get_json(silent=True) or {}
    updated = alert_manager.update_status(alert_id, data.get("status"))
    return jsonify({"alert": updated}) if updated else _json_error("Alert not found.", 404)

@app.get("/api/alerts/<alert_id>/snapshot")
def get_alert_snapshot(alert_id):
    row = alert_manager.get_alert(alert_id)
    if not row or not row.get("snapshot_path"):
        return _json_error("Snapshot not recorded.", 404)

    snap_path = Path(row["snapshot_path"]).resolve()
    
    # 1. Primary check
    if snap_path.is_file():
        return send_file(snap_path, mimetype="image/jpeg")
        
    # 2. Fallback check inside ALERTS_DIR
    fallback = (Path(ALERTS_DIR) / snap_path.name).resolve()
    if fallback.is_file():
        return send_file(fallback, mimetype="image/jpeg")

    # 3. Fallback search inside RESULTS_DIR
    for candidate in Path(RESULTS_DIR).rglob(snap_path.name):
        if candidate.is_file():
            return send_file(candidate, mimetype="image/jpeg")

    return _json_error(f"Snapshot image not found at {snap_path}", 404)

@app.get("/api/alerts/export.csv")
def export_alerts_csv():
    target = ALERTS_DIR / "alerts_export.csv"
    alert_manager.export_csv(target)
    return send_file(target, as_attachment=True, download_name="cctv_alerts.csv")


# ---------------- API: LIVE MONITORING ----------------
@app.post("/api/live/start")
def start_live():
    data = request.get_json(silent=True) or {}
    try:
        res = live_monitor.start(
            data.get("stream_url"),
            source_name=data.get("source_name", "IP Camera"),
            polygons=data.get("polygons", []),
            confidence=float(data.get("confidence", 0.45)),
            cooldown=int(data.get("cooldown", 30))
        )
        return jsonify({"success": True, "live": res}), 202
    except Exception as e:
        return _json_error(str(e), 400)

@app.post("/api/live/stop")
def stop_live():
    return jsonify({"success": True, "live": live_monitor.stop()})

@app.get("/api/live/status")
def live_status():
    return jsonify(live_monitor.public_status())

@app.get("/api/live/feed")
def live_feed():
    return Response(live_monitor.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- API: DATASETS & MODELS ----------------
@app.get("/api/datasets/summary")
def get_datasets_summary():
    return jsonify(dataset_manager.get_summary())

@app.post("/api/datasets/create-sample")
def create_dataset_sample():
    data = request.get_json(silent=True) or {}
    cat = data.get("category", "weapons")
    count = dataset_manager.create_starter_sample_dataset(cat, samples_per_class=12)
    return jsonify({"success": True, "samples_created": count, "category": cat})

@app.get("/api/datasets/download-zip/<category>")
def download_dataset_zip(category):
    try:
        zip_path = dataset_manager.package_to_zip(category)
        return send_file(zip_path, as_attachment=True, download_name=zip_path.name)
    except Exception as e:
        return _json_error(str(e), 400)

@app.get("/api/models/status")
def get_models_status():
    model_registry.refresh()
    return jsonify(model_registry.capabilities)

@app.post("/api/models/train")
def trigger_training():
    data = request.get_json(silent=True) or {}
    cat = data.get("category", "weapons")
    epochs = int(data.get("epochs", 15))
    
    def _worker():
        from train_models import train_specialist
        train_specialist(cat, epochs=epochs)
        model_registry.refresh()

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"success": True, "message": f"Training initiated for {cat} ({epochs} epochs)."}), 202


if __name__ == "__main__":
    app.run(
        host=os.environ.get("CCTV_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True
    )