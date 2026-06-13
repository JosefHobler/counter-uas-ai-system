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
