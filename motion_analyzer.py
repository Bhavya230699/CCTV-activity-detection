"""Image-plane motion estimation, trajectory tracking, and velocity calculation."""

from __future__ import annotations
from collections import defaultdict, deque
import math

class MotionAnalyzer:
    def __init__(self, source_fps: float = 30.0, movement_threshold_px: float = 3.5, history_size: int = 6):
        self.source_fps = max(float(source_fps or 30.0), 1.0)
        self.threshold = float(movement_threshold_px)
        self.history = defaultdict(lambda: deque(maxlen=max(2, int(history_size))))

    def update(self, track_id: int | str, center_x: float, center_y: float, frame_number: int) -> dict:
        points = self.history[track_id]
        current = (float(center_x), float(center_y), int(frame_number))
        points.append(current)
        
        if len(points) < 2:
            return {"moving": False, "speed": 0.0, "distance": 0.0}
            
        previous = points[0]
        distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
        elapsed_frames = max(current[2] - previous[2], 1)
        elapsed_seconds = elapsed_frames / self.source_fps
        
        moving = distance >= self.threshold
        speed = (distance / elapsed_seconds) if moving and elapsed_seconds > 0 else 0.0
        
        return {
            "moving": moving,
            "speed": round(speed, 2),
            "distance": round(distance, 2)
        }

    def remove_track(self, track_id: int | str):
        self.history.pop(track_id, None)