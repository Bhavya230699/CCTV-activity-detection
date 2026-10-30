"""Threaded Live Camera Monitor with Multi-Model Danger Detection & Universal Stream Reader."""

from __future__ import annotations

import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

from alert_manager import AlertManager
from config import ALERTS_DIR
from detection import DetectionPipeline
from zone_drawer import normalize_polygons, point_in_region, validate_fractional_polygons


class SmartCameraCapture:
    def __init__(self, url: str):
        self.url = str(url).strip()
        self.is_webcam = self.url.isdigit()
        self.is_http = self.url.lower().startswith("http")
        self.cap = None
        self.http_stream = None
        self.bytes_buf = b""
        self._open()

    def _open(self):
        if self.is_webcam:
            self.cap = cv2.VideoCapture(int(self.url))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif not self.is_http:
            self.cap = cv2.VideoCapture(self.url)
        else:
            mjpeg_url = self.url
            if "/video" in mjpeg_url and not mjpeg_url.endswith(".mjpg"):
                mjpeg_url = mjpeg_url.replace("/video", "/video?dummy=param.mjpg")
            
            try:
                self.cap = cv2.VideoCapture(mjpeg_url)
            except Exception:
                self.cap = None

            if self.cap is None or not self.cap.isOpened():
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self._open_http_fallback()

    def _open_http_fallback(self):
        req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0"})
        self.http_stream = urllib.request.urlopen(req, timeout=6)

    def read(self):
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return True, frame
            if self.is_http:
                self.cap.release()
                self.cap = None
                try:
                    self._open_http_fallback()
                except Exception:
                    return False, None

        if self.http_stream is not None:
            try:
                while True:
                    chunk = self.http_stream.read(4096)
                    if not chunk:
                        return False, None
                    self.bytes_buf += chunk

                    if len(self.bytes_buf) > 1000000:
                        self.bytes_buf = self.bytes_buf[-65536:]

                    a = self.bytes_buf.find(b"\xff\xd8")
                    b = self.bytes_buf.find(b"\xff\xd9")
                    if a != -1 and b != -1 and b > a:
                        jpg_bytes = self.bytes_buf[a : b + 2]
                        self.bytes_buf = self.bytes_buf[b + 2 :]
                        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return True, frame
            except Exception:
                return False, None

        return False, None

    def isOpened(self):
        if self.cap is not None:
            return self.cap.isOpened()
        return self.http_stream is not None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.http_stream is not None:
            try:
                self.http_stream.close()
            except Exception:
                pass
            self.http_stream = None


class LiveMonitor:
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.status = "stopped"
        self.error = None
        self.source_name = "Live Camera"
        self.source_fps = 0.0
        self.processing_fps = 0.0
        self.frame_jpeg = None
        self.detections = []
        self.polygons = []
        self.confidence = 0.40
        self.cooldown = 20

    def start(self, stream_url: str, source_name="Camera", polygons=None, confidence=0.40, cooldown=20):
        if not stream_url:
            raise ValueError("Stream URL is required.")
        valid_polys = validate_fractional_polygons(polygons or [])
        
        with self.lock:
            # FIX: If a previous connection was running or stuck, auto-close it cleanly!
            if self.status in {"connecting", "running"}:
                self.stop_event.set()
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=1.0)
                self.status = "stopped"
                self.frame_jpeg = None

            self.stop_event.clear()
            self.status = "connecting"
            self.error = None
            self.source_name = source_name
            self.polygons = valid_polys
            self.confidence = confidence
            self.cooldown = cooldown
            self.thread = threading.Thread(target=self._run, args=(stream_url.strip(),), daemon=True)
            self.thread.start()
            
        return self.public_status()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.lock:
            self.status = "stopped"
            self.frame_jpeg = None
        return self.public_status()

    def public_status(self):
        with self.lock:
            return {
                "status": self.status,
                "error": self.error,
                "source": self.source_name,
                "source_fps": round(self.source_fps, 1),
                "processing_fps": round(self.processing_fps, 1),
                "current_detections": len(self.detections),
                "detections": list(self.detections)
            }

    def _run(self, url: str):
        reader = None
        try:
            reader = SmartCameraCapture(url)

            if not reader.isOpened():
                raise RuntimeError(
                    f"Could not connect to camera at '{url}'. Verify stream URL and network connection."
                )

            with self.lock:
                self.source_fps = 25.0
                self.status = "running"
                self.error = None

            detector = DetectionPipeline(confidence=self.confidence)
            zone_states = {}
            last_sample = 0.0
            frame_count = 0
            t0 = time.time()

            while not self.stop_event.is_set():
                ok, frame = reader.read()
                if not ok or frame is None:
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError("Camera connection lost or stream ended.")
                
                frame_count += 1
                now = time.time()
                h, w = frame.shape[:2]
                pixel_polys = normalize_polygons(self.polygons, w, h)

                if now - last_sample >= 0.25:
                    last_sample = now
                    dets = detector.detect(frame, polygons=None, track=True)
                    
                    cur_dets = []
                    for d in dets:
                        tid = d.track_id or 1001
                        cur_dets.append({"class": d.class_name, "conf": round(d.confidence, 2), "box": d.box})

                        threat = detector.registry.evaluate_threat(d.to_dict(), now, self.source_name, self.cooldown)
                        if threat:
                            snap_p = (Path(ALERTS_DIR) / f"live_{threat['alert_type']}_{uuid.uuid4().hex[:8]}.jpg").resolve()
                            cv2.imwrite(str(snap_p), frame)
                            self.alert_manager.create_alert(
                                source=self.source_name,
                                alert_type=threat["alert_type"],
                                severity=threat["severity"],
                                message=threat["message"],
                                confidence=d.confidence,
                                track_id=tid,
                                snapshot_path=str(snap_p)
                            )

                        if d.class_name == "person" and pixel_polys:
                            cx = (d.box[0] + d.box[2]) / 2
                            bot_y = d.box[3]
                            inside = point_in_region(cx, bot_y, pixel_polys)
                            prev_in = zone_states.get(tid, False)

                            if inside and not prev_in:
                                snap_p = (Path(ALERTS_DIR) / f"live_zone_entry_{uuid.uuid4().hex[:8]}.jpg").resolve()
                                cv2.imwrite(str(snap_p), frame)
                                self.alert_manager.create_alert(
                                    source=self.source_name,
                                    alert_type="restricted_region_entry",
                                    severity="WARNING",
                                    message=f"WARNING: Person #{tid} entered the monitored restricted zone!",
                                    confidence=d.confidence,
                                    track_id=tid,
                                    snapshot_path=str(snap_p)
                                )
                            zone_states[tid] = inside
                    
                    with self.lock:
                        self.detections = cur_dets
                        self.processing_fps = frame_count / max(now - t0, 0.001)

                with self.lock:
                    active_boxes = list(self.detections)
                for d in active_boxes:
                    b = list(map(int, d["box"]))
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 220, 100), 2)
                    cv2.putText(frame, f"{d['class']} {d['conf']}", (b[0], max(b[1]-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 2)

                if pixel_polys:
                    for poly in pixel_polys:
                        cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, (255, 180, 0), 2)

                ok_enc, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_enc:
                    with self.lock:
                        self.frame_jpeg = enc.tobytes()

        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = "error"
        finally:
            if reader is not None:
                reader.release()
            with self.lock:
                if self.status != "error":
                    self.status = "stopped"

    def stream(self):
        while True:
            with self.lock:
                f = self.frame_jpeg
                st = self.status
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            elif st in {"stopped", "error"}:
                break
            time.sleep(0.03)