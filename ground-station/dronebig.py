"""
Ground Station SAHI Tracker
Runs YOLO detection with SAHI slicing on a video source and forwards
active tracks to the ground station server via HTTP telemetry.

Usage:
    python dronebig.py --source "your_video.mp4"
    python dronebig.py --source 0  # webcam
"""

import argparse
import os
import sys

import cv2
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import torch

# Make the repo-root `common` package importable when run from ground-station/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.telemetry import TelemetryClient


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


def load_detection_model(model_path="best.pt", confidence=0.5, device=None):
    """Load the heavy SAHI/YOLO model once.

    Shared by the standalone tracker (``main``) and the confirm service so there
    is a single place that knows how to construct the model.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence,
        device=device,
    )


def detect_on_image(detection_model, image_rgb, slice_size=640, overlap_ratio=0.2, sliced=True):
    """Run inference on one RGB image.

    Returns a list of detections, each
    ``{"xyxy": (x1, y1, x2, y2), "confidence": float, "class_id": int, "label": str}``.

    ``sliced=True`` (SAHI tiling) is for full frames where targets are tiny — used
    per-frame by the standalone tracker. ``sliced=False`` runs a single plain
    inference and is used per-crop by the confirm service: a crop is already
    small/zoomed, so slicing it is wasted work.
    """
    if sliced:
        result = get_sliced_prediction(
            image_rgb,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
    else:
        result = get_prediction(image_rgb, detection_model)
    dets = []
    for obj in result.object_prediction_list:
        category = getattr(obj, "category", None)
        category_id = getattr(category, "id", 0) or 0
        category_name = getattr(category, "name", None) or "drone"
        dets.append(
            {
                "xyxy": (obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy),
                "confidence": float(obj.score.value),
                "class_id": int(category_id),
                "label": category_name,
            }
        )
    return dets


def main():
    args = parse_args()

    # Resolve source — integer for webcam index, string for file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Load model
    print("Loading AI model...")
    detection_model = load_detection_model(args.model, args.confidence)

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

    telemetry = TelemetryClient(args.server, "base-station-sahi")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        # --- STEP A: SAHI DETECTION (shared with the confirm service) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detect_on_image(detection_model, rgb_frame)

        # --- STEP B: CONVERT TO SUPERVISION FORMAT ---
        if dets:
            detections = sv.Detections(
                xyxy=np.array([d["xyxy"] for d in dets], dtype=float),
                confidence=np.array([d["confidence"] for d in dets], dtype=float),
                class_id=np.array([d["class_id"] for d in dets], dtype=int),
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
                    "x": float(tcx), "y": float(tcy),
                    "w": float(x2 - x1), "h": float(y2 - y1),
                    "az": float(az), "el": float(el),
                    "confidence": conf,
                    # This tracker has no depth proxy, so it does not estimate a
                    # threat. Emit a neutral, in-range value rather than a bogus
                    # one (was threat_score=50, out of the contract's [0,1]).
                    "threat_score": 0.0,
                    "threat_state": "UNKNOWN",
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