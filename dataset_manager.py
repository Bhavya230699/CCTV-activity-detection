"""Dataset Manager: Validate, inspect, and package surveillance datasets."""

import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from config import DATASETS_DIR

CANONICAL_SCHEMAS = {
    "weapons": {
        "classes": ["handgun", "long_gun", "knife", "heavy_blade", "blunt_weapon"],
        "license": "CC-BY 4.0 / Open Access",
        "description": "Tactical threat objects: handguns, long guns, knives, blades, bats."
    },
    "dangerous_animals": {
        "classes": ["snake", "big_cat", "bear", "crocodilian", "bull_or_boar", "monkey"],
        "license": "CC-BY 2.0 / Open Access",
        "description": "Dangerous reptiles, predators, and aggressive animals."
    },
    "fire_smoke": {
        "classes": ["fire", "smoke"],
        "license": "Academic Open Access (D-Fire)",
        "description": "Open fire flames and industrial/wildfire smoke plumes."
    }
}


class DatasetManager:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else DATASETS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_summary(self):
        """Scans all dataset folders and returns class counts, splits, and readiness."""
        summary = {}
        for category, schema in CANONICAL_SCHEMAS.items():
            cat_dir = self.base_dir / category
            data_yaml = cat_dir / "data.yaml"

            train_count = len(list((cat_dir / "images" / "train").glob("*.*"))) if (cat_dir / "images" / "train").is_dir() else 0
            val_count = len(list((cat_dir / "images" / "val").glob("*.*"))) if (cat_dir / "images" / "val").is_dir() else 0
            test_count = len(list((cat_dir / "images" / "test").glob("*.*"))) if (cat_dir / "images" / "test").is_dir() else 0
            total = train_count + val_count + test_count

            summary[category] = {
                "name": category.replace("_", " ").title(),
                "classes": schema["classes"],
                "license": schema["license"],
                "description": schema["description"],
                "total_images": total,
                "splits": {"train": train_count, "val": val_count, "test": test_count},
                "configured": data_yaml.is_file(),
                "ready_for_training": total > 0 and data_yaml.is_file(),
            }
        return summary

    def create_starter_sample_dataset(self, category, samples_per_class=10):
        """Utility called by the web UI to re-generate starter samples if needed."""
        if category not in CANONICAL_SCHEMAS:
            return 0
        from build_local_datasets import DATASET_CONFIGS, build_category
        if category in DATASET_CONFIGS:
            build_category(category, DATASET_CONFIGS[category])
        return samples_per_class * len(CANONICAL_SCHEMAS[category]["classes"])

    def package_to_zip(self, category=None):
        """Packages datasets into a downloadable ZIP archive."""
        if category and category != "all":
            if category not in CANONICAL_SCHEMAS:
                raise ValueError(f"Unknown category: {category}")
            src_dir = self.base_dir / category
            zip_filename = f"{category}_dataset.zip"
            zip_path = self.base_dir / zip_filename
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in src_dir.rglob("*"):
                    if p.is_file() and p.name != zip_filename:
                        zf.write(p, p.relative_to(self.base_dir).as_posix())
            return zip_path
        else:
            zip_filename = "all_cctv_datasets.zip"
            zip_path = self.base_dir / zip_filename
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in self.base_dir.rglob("*"):
                    if p.is_file() and p.name != zip_filename and not p.name.endswith(".zip"):
                        zf.write(p, p.relative_to(self.base_dir).as_posix())
            return zip_path