"""Recorded-video surveillance processing engine with danger models and event extraction."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from alert_manager import AlertManager
from anomaly_detector import EventExtractor
from config import ALERTS_DIR, DB_PATH
from detection import DetectionPipeline
from models.registry import ModelRegistry
from motion_analyzer import MotionAnalyzer
from zone_drawer import normalize_polygons, point_in_region, validate_fractional_polygons


def _draw_box(frame, item, threat=None):
    box = item["box"] if isinstance(item, dict) else item.box
    class_name = item["class_name"] if isinstance(item, dict) else item.class_name
    track_id = item["track_id"] if isinstance(item, dict) else item.track_id
    confidence = item["confidence"] if isinstance(item, dict) else item.confidence

    x1, y1, x2, y2 = map(int, box)
    if threat and threat.get("severity") == "CRITICAL":
        color = (0, 0, 240)
    elif threat and threat.get("severity") == "HIGH":
        color = (0, 140, 255)
    elif class_name == "person":
        color = (80, 210, 100)
    else:
        color = (255, 180, 40)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} #{track_id or ''} {confidence:.2f}"
    if threat:
        label = f"[{threat['severity']}] {label}"
    cv2.putText(frame, label, (max(x1, 4), max(y1 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)


def _convert_video_to_h264(video_path: Path):
    """Transcodes OpenCV's mp4v into browser-compatible H.264 so Chrome and Edge play it."""
    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        temp_out = video_path.with_name(f"{video_path.stem}_browser.mp4")
        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(temp_out)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if temp_out.is_file() and temp_out.stat().st_size > 0:
            video_path.unlink(missing_ok=True)
            temp_out.rename(video_path)
    except Exception:
        pass


def _generate_graphs(graphs_dir, class_counts, event_count):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    if class_counts:
        ax.bar(list(class_counts.keys()), list(class_counts.values()), color="#3b82f6")
        ax.tick_params(axis="x", labelrotation=25)
    else:
        ax.text(0.5, 0.5, "No objects detected", ha="center", va="center")
        ax.set_xticks([])
    ax.set_title("Detections by Object Class")
    fig.tight_layout()
    fig.savefig(graphs_dir / "objects_by_class.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Motion Incidents"], [event_count], color="#10b981")
    ax.set_title("Observed Motion Incidents")
    fig.tight_layout()
    fig.savefig(graphs_dir / "movement_events.png", dpi=120)
    plt.close(fig)


def run_analysis(video_path, config=None, progress_callback=None, job_id="job"):
    config = config or {}
    video_path = Path(video_path).resolve()
    out_dir = Path(config.get("output_dir", f"data/results/{job_id}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    out_video = out_dir / "processed_video.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = DetectionPipeline(confidence=float(config.get("confidence_threshold", 0.35)))
    alert_mgr = AlertManager(DB_PATH, ALERTS_DIR)
    extractor = EventExtractor(out_dir, fps, config.get("extraction_mode", "both"))
    motion = MotionAnalyzer(fps)

    area_mode = str(config.get("analysis_area", "full")).lower()
    is_region_mode = (area_mode == "region")
    active_polygons = config.get("polygons", []) if is_region_mode else []
    pixel_polys = normalize_polygons(active_polygons, w, h) if is_region_mode else []

    stride = 1 if config.get("analysis_speed") == "slow" else (4 if config.get("analysis_speed") == "fast" else 2)

    frame_idx, start_time = 0, time.time()
    detections_log = []
    object_records = {}
    completed_events = []
    recorded_alerts = []
    stationary_streaks = Counter()
    prebuffer = deque(maxlen=max(int(fps * 2), 1))
    zone_states = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = (frame_idx - 1) / fps
            should_infer = (frame_idx == 1) or ((frame_idx - 1) % stride == 0)

            active_threats = {}

            if should_infer:
                raw_dets = detector.detect(frame, polygons=active_polygons if is_region_mode else None)
                for det in raw_dets:
                    tid = det.track_id or (1000 + frame_idx)
                    cx = (det.box[0] + det.box[2]) / 2
                    cy = (det.box[1] + det.box[3]) / 2

                    m_stat = motion.update(tid, cx, cy, frame_idx)
                    moving = m_stat["moving"]

                    # Danger detection check
                    threat = detector.registry.evaluate_threat(det.to_dict(), timestamp, video_path.name)
                    if threat:
                        active_threats[tid] = threat
                        snap_p = out_dir / "Alert_Evidence" / f"{threat['alert_type']}_{frame_idx:06d}.jpg"
                        snap_p.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(snap_p), frame)
                        alert_record = alert_mgr.create_alert(
                            source=video_path.name,
                            alert_type=threat["alert_type"],
                            severity=threat["severity"],
                            message=threat["message"],
                            confidence=det.confidence,
                            track_id=tid,
                            snapshot_path=str(snap_p)
                        )
                        if alert_record:
                            recorded_alerts.append(alert_record)

                    # Zone boundary check
                    if is_region_mode and det.class_name == "person" and pixel_polys:
                        bot_y = det.box[3]
                        inside = point_in_region(cx, bot_y, pixel_polys)
                        prev_in = zone_states.get(tid, False)
                        if inside and not prev_in:
                            snap_p = out_dir / "Alert_Evidence" / f"zone_entry_{frame_idx:06d}.jpg"
                            snap_p.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(snap_p), frame)
                            zone_alert = alert_mgr.create_alert(
                                source=video_path.name,
                                alert_type="restricted_region_entry",
                                severity="WARNING",
                                message="Person entered configured restricted region.",
                                confidence=det.confidence,
                                track_id=tid,
                                snapshot_path=str(snap_p)
                            )
                            if zone_alert:
                                recorded_alerts.append(zone_alert)
                        zone_states[tid] = inside

                    # Motion extraction
                    if moving:
                        stationary_streaks[tid] = 0
                        if tid not in extractor.events:
                            extractor.start_event(tid, det.class_name, det.object_type, frame_idx, frame, prebuffer)
                        else:
                            extractor.add_frame(tid, frame_idx, frame, m_stat["speed"])
                    else:
                        stationary_streaks[tid] += 1
                        if tid in extractor.events and stationary_streaks[tid] >= 3:
                            evt = extractor.finish_event(tid)
                            if evt:
                                completed_events.append(evt)

                    obj = object_records.setdefault(tid, {
                        "track_id": tid, "class_name": det.class_name, "object_type": det.object_type,
                        "first_seen_s": round(timestamp, 2), "observations": 0, "max_speed_px_s": 0.0,
                        "confidence": det.confidence
                    })
                    obj["observations"] += 1
                    obj["max_speed_px_s"] = max(obj["max_speed_px_s"], m_stat["speed"])

                    detections_log.append({
                        "frame": frame_idx, "time_s": round(timestamp, 2), "class": det.class_name,
                        "conf": round(det.confidence, 3), "track_id": tid, "speed_px_s": m_stat["speed"],
                        "threat": det.threat_category
                    })

                    _draw_box(frame, det, threat)

            if is_region_mode and pixel_polys:
                for poly in pixel_polys:
                    cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, (255, 180, 0), 2)

            writer.write(frame)
            prebuffer.append(frame.copy())

            if progress_callback and frame_idx % 15 == 0:
                progress_callback({
                    "status": "processing",
                    "frame": frame_idx,
                    "total_frames": total_frames,
                    "percent": round(100 * frame_idx / total_frames, 1),
                    "processing_fps": round(frame_idx / max(time.time() - start_time, 0.001), 1)
                })

    finally:
        completed_events.extend(extractor.finalize())
        cap.release()
        writer.release()

    _convert_video_to_h264(out_video)
    for clip_p in (out_dir / "Videos" / "Event_Videos").glob("*.mp4"):
        _convert_video_to_h264(clip_p)

    datasets_dir = out_dir / "Datasets"
    datasets_dir.mkdir(exist_ok=True)
    pd.DataFrame(detections_log).to_csv(datasets_dir / "detections.csv", index=False)
    pd.DataFrame(list(object_records.values())).to_csv(datasets_dir / "objects.csv", index=False)
    pd.DataFrame(completed_events).to_csv(datasets_dir / "events.csv", index=False)

    graphs_dir = out_dir / "Graphs"
    class_counts = dict(Counter(r["class"] for r in detections_log))
    _generate_graphs(graphs_dir, class_counts, len(completed_events))

    event_clips = [p.relative_to(out_dir).as_posix() for p in (out_dir / "Videos" / "Event_Videos").glob("*.mp4")]
    event_frames = [p.relative_to(out_dir).as_posix() for p in (out_dir / "Photos" / "Event_Frames").rglob("*.jpg")]
    graph_files = [p.relative_to(out_dir).as_posix() for p in graphs_dir.glob("*.png")]

    # Compiles summary with explicit list of triggered alerts for results.html
    summary = {
        "job_id": job_id,
        "video": video_path.name,
        "frames_processed": frame_idx,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_seconds": round(frame_idx / fps, 2),
        "total_detections": len(detections_log),
        "total_events": len(completed_events),
        "has_alerts": len(recorded_alerts) > 0,
        "alerts": recorded_alerts,
        "processed_video": "processed_video.mp4",
        "event_clips": event_clips,
        "event_frames": event_frames[:30],
        "graphs": graph_files,
        "datasets": ["Datasets/detections.csv", "Datasets/objects.csv", "Datasets/events.csv"]
    }
    return summary