"""Automated Model Training and Evaluation Pipeline for YOLOv8 Specialist Models."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from config import BASE_DIR, DATASETS_DIR, EVAL_DIR, MODELS_DIR

try:
    from ultralytics import YOLO
    ULTRALYTICS_INSTALLED = True
except ImportError:
    ULTRALYTICS_INSTALLED = False


def train_specialist(category: str, epochs: int = 25, imgsz: int = 640, batch: int = 16):
    if not ULTRALYTICS_INSTALLED:
        raise RuntimeError("Install ultralytics to train models: pip install ultralytics")

    cat_dir = DATASETS_DIR / category
    yaml_path = cat_dir / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}. Prepare the dataset first.")

    device = "0" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print(f"[*] TRAINING MODEL: {category.upper()}")
    print(f"[*] Compute Device: {device} (CUDA: {torch.cuda.is_available()})")
    print(f"[*] Epochs: {epochs} | Image Size: {imgsz}")
    print("=" * 60)

    model = YOLO("yolov8n.pt")
    runs_dir = BASE_DIR / "runs" / "train" / category

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch if torch.cuda.is_available() else 4,
        device=device,
        workers=2,
        project=str(runs_dir.parent),
        name=category,
        save=True,
        plots=True,
        exist_ok=True
    )

    # Held-out evaluation
    print(f"\n[*] Evaluating on held-out test split...")
    val_metrics = model.val(split="test")
    
    eval_report = {
        "category": category,
        "epochs": epochs,
        "device": device,
        "map50_95": float(val_metrics.box.map),
        "map50": float(val_metrics.box.map50),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
    }

    report_file = EVAL_DIR / f"{category}_evaluation.json"
    report_file.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    print(f"[+] Evaluation report saved to: {report_file}")

    # Copy best weights to models/weights/
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    target_pt = MODELS_DIR / f"{category}.pt"
    if best_pt.is_file():
        shutil.copy(best_pt, target_pt)
        print(f"[+] Production checkpoint deployed: {target_pt}")
    else:
        print(f"[!] Warning: best.pt was not found at {best_pt}")

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom CCTV danger models.")
    parser.add_argument("--category", choices=["weapons", "dangerous_animals", "fire_smoke", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    cats = ["weapons", "dangerous_animals", "fire_smoke"] if args.category == "all" else [args.category]
    for c in cats:
        try:
            train_specialist(c, epochs=args.epochs)
        except Exception as e:
            print(f"[!] Training failed for {c}: {e}")