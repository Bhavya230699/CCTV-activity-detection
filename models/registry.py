"""Multi-Model Surveillance Registry and Danger Evaluation Engine."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

ALLOWED_SURVEILLANCE_CLASSES = {
    "person", "car", "motorcycle", "bus", "truck", "bicycle",
    "dog", "cat", "horse", "cow", "sheep", "elephant", "bear",
    "backpack", "handbag", "suitcase"
}


class ModelRegistry:
    def __init__(self, weights_dir=None):
        from config import MODELS_DIR
        self.weights_dir = Path(weights_dir) if weights_dir else MODELS_DIR
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.capabilities = {}
        self.cooldown_tracker = {}
        self.refresh()

    def refresh(self):
        self.models.clear()
        self.capabilities.clear()

        # 1. General Detector
        if ULTRALYTICS_AVAILABLE:
            general_weights = self.weights_dir / "yolov8n.pt"
            try:
                self.models["general"] = YOLO(str(general_weights if general_weights.exists() else "yolov8n.pt"))
                self.capabilities["general"] = {
                    "name": "General Surveillance Detector",
                    "status": "active",
                    "type": "YOLOv8",
                    "classes": list(ALLOWED_SURVEILLANCE_CLASSES),
                    "path": str(general_weights)
                }
            except Exception as e:
                self.capabilities["general"] = {"status": "error", "error": str(e)}
        else:
            self.capabilities["general"] = {"status": "unavailable", "note": "ultralytics not installed"}

        # 2. Weapons Detector
        w_path = self.weights_dir / "weapons.pt"
        if ULTRALYTICS_AVAILABLE and w_path.is_file():
            try:
                self.models["weapons"] = YOLO(str(w_path))
                self.capabilities["weapons"] = {
                    "name": "Weapons Detector",
                    "status": "active",
                    "type": "Custom YOLOv8",
                    "classes": ["handgun", "long_gun", "knife", "heavy_blade", "blunt_weapon"],
                    "path": str(w_path)
                }
            except Exception as e:
                self.capabilities["weapons"] = {"status": "error", "error": str(e)}
        else:
            self.capabilities["weapons"] = {"name": "Weapons Detector", "status": "pending_training"}

        # 3. Dangerous Animals Detector
        a_path = self.weights_dir / "dangerous_animals.pt"
        if not a_path.is_file():
            a_path = self.weights_dir / "animals.pt"

        if ULTRALYTICS_AVAILABLE and a_path.is_file():
            try:
                self.models["animals"] = YOLO(str(a_path))
                self.capabilities["animals"] = {
                    "name": "Dangerous Animals Detector",
                    "status": "active",
                    "type": "Custom YOLOv8",
                    "classes": ["snake", "big_cat", "bear", "crocodilian", "bull_or_boar", "monkey"],
                    "path": str(a_path)
                }
            except Exception as e:
                self.capabilities["animals"] = {"status": "error", "error": str(e)}
        else:
            self.capabilities["animals"] = {"name": "Dangerous Animals Detector", "status": "pending_training"}

        # 4. Fire & Smoke Detector
        f_path = self.weights_dir / "fire_smoke.pt"
        if ULTRALYTICS_AVAILABLE and f_path.is_file():
            try:
                self.models["fire_smoke"] = YOLO(str(f_path))
                self.capabilities["fire_smoke"] = {
                    "name": "Fire & Smoke Detector",
                    "status": "active",
                    "type": "Custom YOLOv8",
                    "classes": ["fire", "smoke"],
                    "path": str(f_path)
                }
            except Exception as e:
                self.capabilities["fire_smoke"] = {"status": "error", "error": str(e)}
        else:
            self.capabilities["fire_smoke"] = {"name": "Fire & Smoke Detector", "status": "pending_training"}

    def infer_frame(self, frame, conf_threshold=0.30):
        detections = []
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return detections

        # 1. General Detector (COCO baseline)
        if "general" in self.models:
            gen_model = self.models["general"]
            res = gen_model(frame, conf=conf_threshold, verbose=False)[0]
            for box in res.boxes:
                cls_idx = int(box.cls[0].item())
                cls_name = gen_model.names[cls_idx]

                if cls_name not in ALLOWED_SURVEILLANCE_CLASSES:
                    continue

                conf = float(box.conf[0].item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]

                is_animal = cls_name in ["dog", "cat", "horse", "cow", "sheep", "elephant", "bear"]
                obj_type = "Person" if cls_name == "person" else ("Animal" if is_animal else "Vehicle")
                threat_tag = "ANIMAL" if is_animal else None

                detections.append({
                    "box": xyxy,
                    "class_name": cls_name,
                    "object_type": obj_type,
                    "confidence": conf,
                    "model": "YOLOv8 General",
                    "threat_category": threat_tag
                })

        # 2. Specialist Detectors
        for key in ["weapons", "animals", "fire_smoke"]:
            if key in self.models:
                s_model = self.models[key]
                thresh = max(0.20, conf_threshold - 0.05)
                res = s_model(frame, conf=thresh, verbose=False)[0]
                for box in res.boxes:
                    cls_idx = int(box.cls[0].item())
                    cls_name = s_model.names[cls_idx]
                    conf = float(box.conf[0].item())
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    threat_cat = "WEAPON" if key == "weapons" else ("ANIMAL" if key == "animals" else "FIRE_HAZARD")
                    detections.append({
                        "box": xyxy,
                        "class_name": cls_name,
                        "object_type": "Hazard",
                        "confidence": conf,
                        "model": f"YOLOv8 Specialist ({key})",
                        "threat_category": threat_cat
                    })

        return detections

    def evaluate_threat(self, detection, timestamp, source_name, cooldown_seconds=20):
        cls = detection["class_name"].lower() if isinstance(detection, dict) else detection.class_name.lower()
        conf = detection["confidence"] if isinstance(detection, dict) else detection.confidence
        
        cooldown_key = f"{source_name}:{cls}"
        last_time = self.cooldown_tracker.get(cooldown_key, -9999.0)
        if (timestamp - last_time) < cooldown_seconds:
            return None

        alert = None

        # Weapons
        if cls in ["handgun", "long_gun"]:
            alert = {
                "alert_type": "weapon_firearm",
                "severity": "CRITICAL",
                "message": f"CRITICAL: Possible firearm detected ({cls}, {conf*100:.1f}%) — Verify evidence.",
                "confidence": conf
            }
        elif cls in ["knife", "heavy_blade"]:
            alert = {
                "alert_type": "weapon_bladed",
                "severity": "HIGH",
                "message": f"HIGH: Edged or bladed weapon detected ({cls}, {conf*100:.1f}%) — Review scene.",
                "confidence": conf
            }
        elif cls == "blunt_weapon":
            alert = {
                "alert_type": "suspicious_object",
                "severity": "WARNING",
                "message": f"WARNING: Potential impact weapon identified ({cls}, {conf*100:.1f}%).",
                "confidence": conf
            }

        # Apex Predators
        elif cls in ["big_cat", "bear", "crocodilian"]:
            alert = {
                "alert_type": "dangerous_animal_predator",
                "severity": "CRITICAL",
                "message": f"CRITICAL: Apex Predator / Big Cat Detected ({cls.replace('_', ' ').title()}, {conf*100:.1f}%)!",
                "confidence": conf
            }
        elif cls in ["snake"]:
            alert = {
                "alert_type": "dangerous_animal_reptile",
                "severity": "HIGH",
                "message": f"HIGH: Potential snake detected in monitored area ({conf*100:.1f}%).",
                "confidence": conf
            }

        # Animals (COCO baseline detection for lions/big cats misclassified as dogs/cats/cows)
        elif cls in ["dog", "cat", "bear", "cow", "horse", "bull_or_boar", "monkey", "elephant"]:
            label_desc = f"{cls.capitalize()} (Review for Big Cat / Wild Predator)" if cls in ["dog", "cat", "cow", "bear"] else cls.capitalize()
            severity = "HIGH" if cls in ["bear", "dog", "cat"] and conf > 0.65 else "WARNING"
            alert = {
                "alert_type": "animal_perimeter_intrusion",
                "severity": severity,
                "message": f"{severity}: Animal Intrusion Detected [{label_desc}, {conf*100:.1f}%].",
                "confidence": conf
            }

        # Fire & Smoke
        elif cls == "fire":
            alert = {
                "alert_type": "fire_hazard",
                "severity": "CRITICAL",
                "message": f"CRITICAL: Open fire / flame detected ({conf*100:.1f}%)!",
                "confidence": conf
            }
        elif cls == "smoke":
            alert = {
                "alert_type": "smoke_hazard",
                "severity": "HIGH",
                "message": f"HIGH: Smoke plume detected ({conf*100:.1f}%).",
                "confidence": conf
            }

        if alert:
            self.cooldown_tracker[cooldown_key] = timestamp
            return alert
            
        return None