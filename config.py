"""Central configuration and system paths for CCTV Surveillance."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

def _resolve_path(env_var: str, default_subpath: str) -> Path:
    val = os.environ.get(env_var, default_subpath)
    p = Path(val).expanduser()
    return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()

DATA_DIR = _resolve_path("CCTV_DATA_DIR", "data")
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
ALERTS_DIR = DATA_DIR / "alert-evidence"
DB_PATH = DATA_DIR / "alerts.sqlite3"

MODELS_DIR = BASE_DIR / "models" / "weights"
DATASETS_DIR = BASE_DIR / "datasets"
EVAL_DIR = DATA_DIR / "evaluation_reports"

ACTIVITY_MODEL = _resolve_path("CCTV_ACTIVITY_MODEL", str(DATA_DIR / "models" / "activity_knn.joblib"))
SNAKE_MODEL = _resolve_path("CCTV_SNAKE_MODEL", str(DATA_DIR / "models" / "snake_hog_svm.joblib"))

CONFIDENCE_THRESHOLD = float(os.environ.get("CCTV_CONFIDENCE", "0.40"))
ALERT_COOLDOWN_SECONDS = int(os.environ.get("CCTV_ALERT_COOLDOWN", "30"))
MAX_UPLOAD_BYTES = int(os.environ.get("CCTV_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024))) # 1 GB
LIVE_INFERENCE_INTERVAL = float(os.environ.get("CCTV_LIVE_INTERVAL", "0.25"))

for folder in (UPLOAD_DIR, RESULTS_DIR, ALERTS_DIR, MODELS_DIR, DATASETS_DIR, EVAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)