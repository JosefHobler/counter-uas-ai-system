"""Tests for the shared telemetry contract (common/schemas.py).

The headline case is the ``threat_score=50`` bug: an out-of-range value that
used to flow straight to the dashboard must now be rejected at validation time.
"""

import pytest
from pydantic import ValidationError

from common.schemas import (
    BBox,
    CandidateCrop,
    Confirmation,
    TelemetryPayload,
    ThreatState,
    Track,
)


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


# -- cue/confirm contract ----------------------------------------------------


def _valid_candidate_kwargs(**overrides):
    base = dict(
        sender_id="edge-rpi5-alpha",
        candidate_id=1,
        frame_id=10,
        bbox=BBox(x=5.0, y=5.0, w=20.0, h=20.0),
        crop_x=0.0,
        crop_y=0.0,
        crop_width=40,
        crop_height=40,
        confidence=0.7,
        image_jpeg_b64="Zm9v",  # non-empty
    )
    base.update(overrides)
    return base


def test_candidate_minimal_is_valid():
    c = CandidateCrop(**_valid_candidate_kwargs())
    assert c.track_id is None
    assert c.candidate_id == 1


def test_candidate_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        CandidateCrop(**_valid_candidate_kwargs(confidence=1.5))


def test_candidate_rejects_zero_crop_size():
    with pytest.raises(ValidationError):
        CandidateCrop(**_valid_candidate_kwargs(crop_width=0))


def test_candidate_rejects_empty_image():
    with pytest.raises(ValidationError):
        CandidateCrop(**_valid_candidate_kwargs(image_jpeg_b64=""))


def test_candidate_forbids_unknown_keys():
    with pytest.raises(ValidationError):
        CandidateCrop(**_valid_candidate_kwargs(frmae_id=10))


def test_bbox_rejects_negative_dims():
    with pytest.raises(ValidationError):
        BBox(x=0.0, y=0.0, w=-1.0, h=10.0)


def test_confirmation_drone_with_box():
    conf = Confirmation(
        sender_id="ground",
        candidate_id=1,
        frame_id=10,
        track_id=7,
        is_drone=True,
        label="drone",
        confidence=0.92,
        bbox=BBox(x=10.0, y=12.0, w=18.0, h=16.0),
    )
    assert conf.is_drone is True
    assert conf.bbox is not None


def test_confirmation_negative_has_no_box():
    conf = Confirmation(
        sender_id="ground",
        candidate_id=2,
        frame_id=11,
        is_drone=False,
        confidence=0.0,
    )
    assert conf.is_drone is False
    assert conf.bbox is None
