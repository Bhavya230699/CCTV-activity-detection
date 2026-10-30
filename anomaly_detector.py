"""Event media extractor: generates pre-buffered video clips and snapshot frames."""

from __future__ import annotations
from pathlib import Path
import cv2

class EventExtractor:
    def __init__(self, output_dir: Path | str, video_fps: float, output_mode: str = "both"):
        self.output_dir = Path(output_dir)
        self.video_fps = max(float(video_fps or 30.0), 1.0)
        self.output_mode = output_mode.lower()
        
        self.frames_dir = self.output_dir / "Photos" / "Event_Frames"
        self.videos_dir = self.output_dir / "Videos" / "Event_Videos"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        
        self.events: dict = {}
        self.counter = 0

    def start_event(self, track_id: int | str, class_name: str, object_type: str, frame_number: int, frame, prebuffer: list) -> dict:
        self.counter += 1
        name = f"event_{self.counter:04d}"
        frame_dir = self.frames_dir / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        video_path = self.videos_dir / f"{name}.mp4"
        
        writer = None
        if self.output_mode in {"videos", "both"}:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height)
            )
            if writer.isOpened():
                for b_frame in prebuffer:
                    writer.write(b_frame)
            else:
                writer.release()
                writer = None

        event = {
            "event_id": self.counter,
            "track_id": track_id,
            "class_name": class_name,
            "object_type": object_type,
            "start_frame": int(frame_number),
            "end_frame": int(frame_number),
            "max_speed": 0.0,
            "frame_folder": str(frame_dir),
            "video_path": str(video_path) if writer else "",
            "writer": writer,
            "last_saved_frame": None
        }
        self.events[track_id] = event
        return event

    def add_frame(self, track_id: int | str, frame_number: int, frame, speed: float):
        event = self.events.get(track_id)
        if not event:
            return
            
        event["end_frame"] = int(frame_number)
        event["max_speed"] = max(event["max_speed"], float(speed))
        
        if self.output_mode in {"frames", "both"} and (
            event["last_saved_frame"] is None or frame_number - event["last_saved_frame"] >= max(1, int(self.video_fps))
        ):
            save_path = Path(event["frame_folder"]) / f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(save_path), frame)
            event["last_saved_frame"] = frame_number

        if event["writer"] is not None:
            event["writer"].write(frame)

    def finish_event(self, track_id: int | str) -> dict | None:
        event = self.events.pop(track_id, None)
        if not event:
            return None
        writer = event.pop("writer", None)
        if writer is not None:
            writer.release()
        if self.output_mode == "frames":
            event["video_path"] = ""
        return event

    def finalize(self) -> list:
        finished = []
        for track_id in list(self.events):
            evt = self.finish_event(track_id)
            if evt:
                finished.append(evt)
        return finished