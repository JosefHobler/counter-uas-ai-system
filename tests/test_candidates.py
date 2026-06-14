"""Tests for candidate-crop packing/transport (common/candidates.py)."""

import numpy as np

from common.candidates import build_candidate_payload, decode_jpeg_b64, encode_jpeg_b64
from common.schemas import CandidateCrop


def _img(h=100, w=120):
    # A non-uniform image so JPEG round-trips to something meaningful.
    rng = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    return rng


def test_jpeg_roundtrip_preserves_shape():
    img = _img()
    decoded = decode_jpeg_b64(encode_jpeg_b64(img))
    assert decoded.shape == img.shape


def test_build_payload_matches_schema():
    img = _img(100, 120)
    payload = build_candidate_payload(
        img,
        (40, 40, 20, 20),
        sender_id="edge-rpi5-alpha",
        candidate_id=3,
        frame_id=120,
        track_id=7,
        confidence=0.8,
        margin=0.5,  # pad 10 each side -> crop (30,30)-(70,70)
    )
    # The packed dict must satisfy the wire contract.
    model = CandidateCrop(**payload)
    assert model.candidate_id == 3
    assert model.track_id == 7
    assert model.crop_width == 40 and model.crop_height == 40
    assert (model.crop_x, model.crop_y) == (30.0, 30.0)

    # And the embedded crop decodes to the advertised size.
    crop = decode_jpeg_b64(model.image_jpeg_b64)
    assert crop.shape[:2] == (model.crop_height, model.crop_width)


def test_build_payload_clamps_confidence():
    payload = build_candidate_payload(
        _img(),
        (10, 10, 20, 20),
        sender_id="edge",
        candidate_id=0,
        frame_id=0,
        confidence=5.0,
    )
    assert payload["confidence"] == 1.0


def test_build_payload_offframe_returns_none():
    payload = build_candidate_payload(
        _img(100, 100),
        (500, 500, 20, 20),  # fully outside
        sender_id="edge",
        candidate_id=0,
        frame_id=0,
    )
    assert payload is None
