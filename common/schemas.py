"""Telemetry contract shared by every node.

Both producers (the edge detector and the ground-station SAHI tracker) and the
consumer (the telemetry server) validate against these models, so the wire
format has one source of truth and the two sides cannot silently disagree.

The models intentionally bound the numeric fields: ``confidence`` and
``threat_score`` are fractions in ``[0, 1]`` and ``threat_state`` is a closed
enum. A producer that ships, say, ``threat_score=50`` is rejected at the door
instead of poisoning the dashboard with an out-of-range value.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ThreatState(str, Enum):
    """Coarse approach assessment for a track."""

    UNKNOWN = "UNKNOWN"
    STABLE = "STABLE"
    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"


class Track(BaseModel):
    """A single tracked target, expressed in one sender's own frame.

    ``x``/``y``/``w``/``h`` are image-space pixels and ``az``/``el`` are degrees
    off that sender's boresight. There is no shared world frame yet — see the
    coordinate-frame caveat on the server's track merger.
    """

    # Strict contract: reject unknown keys and coerce enums to their str value
    # on dump so the stored/broadcast payload is plain JSON.
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    id: int = Field(..., description="Sender-local track ID, stable per sender")
    x: float = Field(..., description="Image-space centroid X (px)")
    y: float = Field(..., description="Image-space centroid Y (px)")
    w: float = Field(..., ge=0.0, description="Bounding-box width (px)")
    h: float = Field(..., ge=0.0, description="Bounding-box height (px)")
    az: float = Field(..., description="Azimuth offset from boresight (deg)")
    el: float = Field(..., description="Elevation offset from boresight (deg)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    threat_score: float = Field(0.0, ge=0.0, le=1.0, description="Approach strength")
    threat_state: ThreatState = Field(ThreatState.UNKNOWN)
    sensor: str = Field("", description="Free-form sensor/source tag, e.g. 'RGB'")


class TelemetryPayload(BaseModel):
    """One telemetry frame POSTed by a sender to ``/api/telemetry``."""

    model_config = ConfigDict(extra="forbid")

    sender_id: str = Field(..., min_length=1)
    tracks: List[Track]
    timestamp: Optional[float] = None


# --- Cue / confirm pipeline ------------------------------------------------
# The edge node runs a cheap detector and, when it flags a drone-like target,
# ships a *crop* of that region to the ground confirm service, which runs the
# heavy model and decides whether it is really a drone. These two models are the
# wire contract for that exchange.


class BBox(BaseModel):
    """An axis-aligned box in image pixels, top-left origin."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    w: float = Field(..., ge=0.0)
    h: float = Field(..., ge=0.0)


class CandidateCrop(BaseModel):
    """A drone-like region the edge flagged, cropped and sent for confirmation.

    ``bbox`` is the candidate detection in full-frame coordinates; ``crop_x`` /
    ``crop_y`` are the crop's top-left origin in that same full frame, so the
    confirm service can map any box it finds inside the crop back to full-frame
    coordinates. The image itself rides along as a base64 JPEG.
    """

    model_config = ConfigDict(extra="forbid")

    sender_id: str = Field(..., min_length=1)
    candidate_id: int
    frame_id: int
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    bbox: BBox
    crop_x: float
    crop_y: float
    crop_width: int = Field(..., ge=1)
    crop_height: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Edge detector confidence")
    image_jpeg_b64: str = Field(..., min_length=1)


class Confirmation(BaseModel):
    """The ground confirm service's verdict on one candidate crop.

    ``candidate_id`` / ``track_id`` echo the candidate so the result can be
    associated back to the edge track that raised it. ``bbox`` is the heavy
    model's tight box in full-frame coordinates, present only when
    ``is_drone`` is true.
    """

    model_config = ConfigDict(extra="forbid")

    sender_id: str = Field(..., min_length=1)
    candidate_id: int
    frame_id: int
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    is_drone: bool
    label: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[BBox] = None
