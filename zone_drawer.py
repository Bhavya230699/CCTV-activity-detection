"""Polygon-based Region of Interest (ROI) validator and ray-casting point checks."""

from __future__ import annotations
import math

def normalize_polygons(polygons: list, width: int, height: int) -> list:
    if not polygons:
        return []
    if (
        isinstance(polygons, (list, tuple))
        and len(polygons) > 0
        and isinstance(polygons[0], (list, tuple))
        and len(polygons[0]) >= 2
        and isinstance(polygons[0][0], (int, float))
    ):
        polygons = [polygons]

    normalized = []
    for poly in polygons:
        try:
            points = [(float(p[0]), float(p[1])) for p in poly]
        except (TypeError, ValueError, IndexError):
            continue
        if len(points) < 3:
            continue

        is_fractional = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
        if is_fractional:
            points = [(x * max(width - 1, 1), y * max(height - 1, 1)) for x, y in points]

        pixels = [
            (min(max(int(round(x)), 0), max(width - 1, 0)),
             min(max(int(round(y)), 0), max(height - 1, 0)))
            for x, y in points
        ]
        if len(set(pixels)) >= 3:
            normalized.append(pixels)
    return normalized

def point_in_region(x: float, y: float, polygons: list) -> bool:
    """Standard ray-casting algorithm to test whether (x,y) is inside any polygon."""
    if not polygons:
        return True
    for poly in polygons:
        inside = False
        j = len(poly) - 1
        for i, (xi, yi) in enumerate(poly):
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False

def validate_fractional_polygons(polygons: list) -> list:
    if not isinstance(polygons, list):
        raise ValueError("ROI polygons must be provided as a list.")
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError("Each ROI must contain at least 3 points.")
        for p in poly:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                raise ValueError("Points must contain [x, y] coordinates.")
            if not (0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0):
                raise ValueError("Polygon points must be normalized coordinates between 0.0 and 1.0.")
    return polygons