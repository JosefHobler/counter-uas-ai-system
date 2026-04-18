"""
Ground Station SAHI Tracker
Runs YOLO detection with SAHI slicing on a video source and forwards
active tracks to the ground station server via HTTP telemetry.

Usage:
    python dronebig.py --source "your_video.mp4"
    python dronebig.py --source 0  # webcam
"""

import argparse
import cv2
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import requests
import threading
import queue
import time


class MiniTelemetryClient:
    """Fire-and-forget telemetry sender running on a background thread.

    The queue is capped to prevent backlog accumulation when the server
    is slow or unreachable. Oldest frames are silently dropped on overflow.
    """

    def __init__(self, endpoint_url: str, sender_id: str):
        self.endpoint_url = endpoint_url
        self.sender_id = sender_id
        self.q = queue.Queue(maxsize=10)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                break
            try:
                requests.post(self.endpoint_url, json=payload, timeout=0.5)
            except requests.exceptions.RequestException:
                # Drops are expected in degraded network conditions.
                pass
            finally:
                self.q.task_done()

    def send(self, tracks):
        if not self.running:
            return
        payload = {"sender_id": self.sender_id, "timestamp": time.time(), "tracks": tracks}
        # Drop oldest frame if queue is full to prevent backlog.
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            pass

    def stop(self):
        self.running = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Ground Station SAHI Tracker")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to video file or camera index (e.g. '0' for webcam)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path to YOLO model weights (default: best.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path (default: output.mp4)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000/api/telemetry",
        help="Ground station telemetry endpoint URL",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve source — integer for webcam index, string for file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Load model
    print("Loading AI model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.model,
        confidence_threshold=args.confidence,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Initialize supervision tracker and annotators
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,  # Minimum score to start a new track
        lost_track_buffer=60,             # Retain lost track for 60 frames (~2s at 30fps)
        minimum_matching_threshold=0.8    # Strictness for ID re-association (prevents ID swaps)
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print(f"Processing {width}x{height} @ {fps} FPS -> {args.output}")

    telemetry = MiniTelemetryClient(args.server, "base-station-sahi")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        # --- STEP A: SAHI DETECTION ---
        # Convert from BGR (OpenCV) to RGB (SAHI)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            rgb_frame,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )

        # --- STEP B: CONVERT TO SUPERVISION FORMAT ---
        xyxy = []
        confidences = []
        class_ids = []

        for obj in result.object_prediction_list:
            xyxy.append([obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
            confidences.append(obj.score.value)
            class_ids.append(0)  # Drone is class 0

        if len(xyxy) > 0:
            detections = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )
        else:
            detections = sv.Detections.empty()

        # --- STEP C: BYTETRACK ---
        tracked_detections = tracker.update_with_detections(detections)

        # --- STEP D: ANNOTATE AND TRANSMIT ---
        if len(tracked_detections) > 0:
            labels = [
                f"ID:{tracker_id} Drone {conf:.2f}"
                for tracker_id, conf
                in zip(tracked_detections.tracker_id, tracked_detections.confidence)
            ]

            frame = box_annotator.annotate(scene=frame, detections=tracked_detections)
            frame = label_annotator.annotate(scene=frame, detections=tracked_detections, labels=labels)

            # Build telemetry payload and forward to ground station server
            out_tracks = []
            for i, bbox in enumerate(tracked_detections.xyxy):
                x1, y1, x2, y2 = bbox
                tcx = (x1 + x2) / 2
                tcy = (y1 + y2) / 2
                # Simplified Az/El estimate based on FOV centre offset
                az = ((tcx - width / 2) / width) * 80.0
                el = ((height / 2 - tcy) / height) * 50.0
                conf = float(tracked_detections.confidence[i])
                out_tracks.append({
                    "id": int(tracked_detections.tracker_id[i]),
                    "x": tcx, "y": tcy, "w": x2 - x1, "h": y2 - y1,
                    "az": az, "el": el,
                    "confidence": conf,
                    "threat_score": 50,
                    "threat_state": "STABLE",
                    "sensor": "BASE-SAHI"
                })
            if out_tracks:
                telemetry.send(out_tracks)

        out.write(frame)
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    telemetry.stop()
    print(f"Done! Output saved to '{args.output}'")


if __name__ == "__main__":
    main()