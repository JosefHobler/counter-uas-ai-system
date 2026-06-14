"""Candidate-crop transport for the cue/confirm pipeline.

The edge node finds drone-like candidates with its lightweight detector, crops
the region (with margin) out of the full frame, JPEG-encodes it, and ships it to
the ground confirm service. Sending *crops* instead of full video keeps the
tether traffic small and only spends the heavy model's time when there is
actually something to look at.

``build_candidate_payload`` is the pure (cv2-only) packer and is unit-tested;
``CandidateSender`` adds the same non-blocking, drop-oldest delivery as the
telemetry client.
"""

from __future__ import annotations

import base64
import time
from typing import Optional

import cv2
import numpy as np

from common.geometry import clamp_crop_region
from common.telemetry import AsyncPoster


def encode_jpeg_b64(image, quality: int = 85) -> str:
    """Encode a BGR image to a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg_b64(data: str):
    """Decode a base64 JPEG string back to a BGR image."""
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG decoding failed")
    return img


def build_candidate_payload(
    frame,
    bbox,
    *,
    sender_id: str,
    candidate_id: int,
    frame_id: int,
    track_id: Optional[int] = None,
    confidence: float = 0.0,
    margin: float = 0.4,
    jpeg_quality: int = 85,
    timestamp: Optional[float] = None,
) -> Optional[dict]:
    """Crop ``bbox`` out of ``frame`` and pack a ``CandidateCrop``-shaped dict.

    Returns ``None`` if the crop would be empty (box fully off-frame). The dict
    validates against :class:`common.schemas.CandidateCrop`.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = clamp_crop_region(bbox, w, h, margin)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    return {
        "sender_id": sender_id,
        "candidate_id": int(candidate_id),
        "frame_id": int(frame_id),
        "track_id": None if track_id is None else int(track_id),
        "timestamp": timestamp,
        "bbox": {
            "x": float(bbox[0]),
            "y": float(bbox[1]),
            "w": float(bbox[2]),
            "h": float(bbox[3]),
        },
        "crop_x": float(x0),
        "crop_y": float(y0),
        "crop_width": int(x1 - x0),
        "crop_height": int(y1 - y0),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "image_jpeg_b64": encode_jpeg_b64(crop, jpeg_quality),
    }


class CandidateSender:
    """Non-blocking sender of candidate crops to the ground confirm service."""

    def __init__(
        self,
        endpoint_url: str,
        sender_id: str,
        max_queue_size: int = 10,
        post_timeout: float = 0.5,
        margin: float = 0.4,
        jpeg_quality: int = 85,
    ):
        self.sender_id = sender_id
        self.margin = margin
        self.jpeg_quality = jpeg_quality
        self._poster = AsyncPoster(endpoint_url, max_queue_size, post_timeout)
        self._next_id = 0

    def send_candidate(
        self,
        frame,
        bbox,
        *,
        frame_id: int,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[int]:
        """Crop, encode, and queue one candidate. Returns its id, or None if empty."""
        candidate_id = self._next_id
        payload = build_candidate_payload(
            frame,
            bbox,
            sender_id=self.sender_id,
            candidate_id=candidate_id,
            frame_id=frame_id,
            track_id=track_id,
            confidence=confidence,
            margin=self.margin,
            jpeg_quality=self.jpeg_quality,
            timestamp=timestamp if timestamp is not None else time.time(),
        )
        if payload is None:
            return None
        self._next_id += 1
        self._poster.post(payload)
        return candidate_id

    def stop(self):
        self._poster.stop()
