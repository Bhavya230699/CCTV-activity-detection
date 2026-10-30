"""Unified Detection Pipeline integrating YOLOv8 and the Model Registry."""

import math
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from models.registry import ModelRegistry
from zone_drawer import normalize_polygons, point_in_region


@dataclass
class Detection:
    box: List[float]
    class_name: str
    object_type: str
    confidence: float
    track_id: Optional[int]
    model: str
    threat_category: Optional[str] = None
    snake_probability: Optional[float] = None

    @property
    def score(self) -> float:
        return self.confidence

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _iou(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


class DetectionPipeline:
    def __init__(self, confidence: float = 0.40, max_track_gap: int = 15, **kwargs):
        self.confidence = min(max(float(confidence), 0.05), 0.95)
        self.max_track_gap = max(1, int(max_track_gap))
        self.registry = ModelRegistry()
        self.tracks = {}
        self.next_track_id = 1

    def detect(self, frame, polygons=None, track: bool = True, imgsz=None) -> List[Detection]:
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        raw_items = self.registry.infer_frame(frame, conf_threshold=self.confidence)

        if polygons:
            pixel_polys = normalize_polygons(polygons, w, h)
            filtered = []
            for item in raw_items:
                bx = item["box"]
                cx, bottom_y = (bx[0] + bx[2]) / 2, bx[3]
                if point_in_region(cx, bottom_y, pixel_polys):
                    filtered.append(item)
            raw_items = filtered

        detections = [
            Detection(
                box=item["box"],
                class_name=item["class_name"],
                object_type=item["object_type"],
                confidence=item["confidence"],
                track_id=None,
                model=item["model"],
                threat_category=item.get("threat_category"),
                snake_probability=item.get("snake_probability")
            )
            for item in raw_items
        ]

        if track:
            self._track_objects(detections)

        return detections

    def _track_objects(self, detections: List[Detection]):
        used_tracks = set()
        for item in detections:
            box = item.box
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            diag = math.hypot(box[2] - box[0], box[3] - box[1])

            best_match, best_dist = None, 1e9
            for tid, prev in self.tracks.items():
                if tid in used_tracks or prev["class_name"] != item.class_name:
                    continue
                old_box = prev["box"]
                old_cx, old_cy = (old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2
                dist = math.hypot(cx - old_cx, cy - old_cy)
                overlap = _iou(box, old_box)
                if overlap > 0.15 or dist < max(40, diag * 0.75):
                    if dist < best_dist:
                        best_dist = dist
                        best_match = tid

            if best_match is not None:
                tid = best_match
            else:
                tid = self.next_track_id
                self.next_track_id += 1

            used_tracks.add(tid)
            item.track_id = tid
            self.tracks[tid] = {"box": list(box), "class_name": item.class_name, "missed": 0}

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_track_gap:
                    del self.tracks[tid]