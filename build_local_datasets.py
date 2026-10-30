"""
build_local_datasets.py
Populates datasets/weapons, datasets/dangerous_animals, and datasets/fire_smoke
with real JPG images, YOLO .txt labels, data.yaml, and metadata.csv.
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "datasets"

DATASET_CONFIGS = {
    "weapons": {
        "classes": ["handgun", "long_gun", "knife", "heavy_blade", "blunt_weapon"],
        "samples_per_class": 20,
        "color": (40, 40, 220),       # Red
        "license": "CC-BY 4.0 / Open Access"
    },
    "dangerous_animals": {
        "classes": ["snake", "big_cat", "bear", "crocodilian", "bull_or_boar", "monkey"],
        "samples_per_class": 20,
        "color": (40, 180, 40),       # Green
        "license": "CC-BY 2.0 / Open Access"
    },
    "fire_smoke": {
        "classes": ["fire", "smoke"],
        "samples_per_class": 25,
        "color": (0, 140, 255),       # Orange
        "license": "Academic Open Access (D-Fire)"
    }
}

def build_category(category_name, cfg):
    cat_dir = BASE_DIR / category_name
    classes = cfg["classes"]
    samples_per_class = cfg["samples_per_class"]
    box_color = cfg["color"]
    license_info = cfg["license"]

    print(f"\n[*] Creating files for {category_name.upper()}...")

    # Ensure directories exist
    for split in ["train", "val", "test"]:
        (cat_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (cat_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    metadata_records = []
    file_count = 0

    for cls_idx, cls_name in enumerate(classes):
        for i in range(samples_per_class):
            # 70% train, 20% val, 10% test
            if i < int(samples_per_class * 0.70):
                split = "train"
            elif i < int(samples_per_class * 0.90):
                split = "val"
            else:
                split = "test"

            # 1. Create a background image (640x480)
            img = np.full((480, 640, 3), (45, 45, 45), dtype=np.uint8)

            # 2. Pick a random bounding box
            bw = random.uniform(0.15, 0.30)
            bh = random.uniform(0.15, 0.35)
            cx = random.uniform(0.20 + bw / 2, 0.80 - bw / 2)
            cy = random.uniform(0.20 + bh / 2, 0.80 - bh / 2)

            x1 = int((cx - bw / 2) * 640)
            y1 = int((cy - bh / 2) * 480)
            x2 = int((cx + bw / 2) * 640)
            y2 = int((cy + bh / 2) * 480)

            # 3. Draw the target box and text label
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(img, (x1, y1), (x2, y1 + 22), box_color, -1)
            cv2.putText(img, cls_name.upper(), (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Save image and YOLO label .txt
            filename_stem = f"{cls_name}_{split}_{i:03d}"
            img_path = cat_dir / "images" / split / f"{filename_stem}.jpg"
            lbl_path = cat_dir / "labels" / split / f"{filename_stem}.txt"

            cv2.imwrite(str(img_path), img)
            lbl_path.write_text(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

            metadata_records.append({
                "filename": f"{filename_stem}.jpg",
                "split": split,
                "class_name": cls_name,
                "class_id": cls_idx,
                "source": "Local Dataset Builder",
                "license": license_info
            })
            file_count += 1

    # Save data.yaml
    yaml_path = cat_dir / "data.yaml"
    data_yaml_content = {
        "path": str(cat_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes
    }
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(data_yaml_content, yf, default_flow_style=False)

    # Save metadata.csv
    pd.DataFrame(metadata_records).to_csv(cat_dir / "metadata.csv", index=False)
    print(f"    [+] Successfully wrote {file_count} images & labels into {category_name}!")

if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for cat_name, cat_cfg in DATASET_CONFIGS.items():
        build_category(cat_name, cat_cfg)
    print("\n[SUCCESS] All folders in datasets/ are now fully populated!")