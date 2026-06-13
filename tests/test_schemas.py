"""Tests for the shared telemetry contract (common/schemas.py).

The headline case is the ``threat_score=50`` bug: an out-of-range value that
used to flow straight to the dashboard must now be rejected at validation time.
"""

import pytest
from pydantic import ValidationError

from common.schemas import TelemetryPayload, ThreatState, Track


def _valid_track_kwargs(**overrides):
    base = dict(id=1, x=10.0, y=20.0, w=5.0, h=5.0, az=1.0, el=2.0, confidence=0.9)
    base.update(overrides)
    return base


def test_minimal_track_has_safe_defaults():
    t = Track(**_valid_track_kwargs())
    assert t.threat_score == 0.0
    # use_enum_values=True means the dumped value is the plain string.
    assert t.threat_state == ThreatState.UNKNOWN.value
    assert t.sensor == ""


def test_threat_score_out_of_range_is_rejected():
    # The whole reason the contract exists: 50 is not a fraction in [0, 1].
    with pytest.raises(ValidationError):
        Track(**_valid_track_kwargs(threat_score=50))


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        Track(**_valid_track_kwargs(confidence=1.5))
    with pytest.raises(ValidationError):
        Track(**_valid_track_kwargs(confidence=-0.1))


def test_negative_box_dims_rejected():
    with pytest.raises(ValidationError):
        Track(**_valid_track_kwargs(w=-1.0))


def test_unknown_keys_are_forbidden():
    # extra="forbid" catches typo'd / stale field names instead of dropping them.
    with pytest.raises(ValidationError):
        Track(**_valid_track_kwargs(treat_score=0.5))


def test_invalid_threat_state_rejected():
    with pytest.raises(ValidationError):
        Track(**_valid_track_kwargs(threat_state="HOSTILE"))


def test_payload_requires_non_empty_sender_id():
    with pytest.raises(ValidationError):
        TelemetryPayload(sender_id="", tracks=[])


def test_payload_roundtrip_dumps_plain_json():
    payload = TelemetryPayload(
        sender_id="edge-rpi5-alpha",
        tracks=[Track(**_valid_track_kwargs(threat_state=ThreatState.APPROACHING))],
    )
    dumped = payload.model_dump()
    assert dumped["sender_id"] == "edge-rpi5-alpha"
    # Enum coerced to its string value, ready to broadcast as JSON.
    assert dumped["tracks"][0]["threat_state"] == "APPROACHING"
