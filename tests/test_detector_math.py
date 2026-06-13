"""Tests for the geometry / scoring helpers in edge-rpi5/drone_detector.py."""

import pytest

from drone_detector import bbox_iou, clamp01, detection_priority


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip.
    # inter = 50, union = 100 + 100 - 50 = 150 -> 1/3.
    assert bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_clamp01_bounds():
    assert clamp01(-2.0) == 0.0
    assert clamp01(0.4) == pytest.approx(0.4)
    assert clamp01(5.0) == 1.0


def test_detection_priority_prefers_confirmed_then_sensor_then_conf():
    confirmed_nir = detection_priority(
        {"sensor": "NIR", "confirmed": True, "confidence": 0.1}
    )
    unconfirmed_rgbnir = detection_priority(
        {"sensor": "RGB+NIR", "confirmed": False, "confidence": 0.99}
    )
    # Confirmation dominates regardless of sensor or confidence.
    assert confirmed_nir > unconfirmed_rgbnir


def test_detection_priority_sensor_ranking():
    rgb = detection_priority({"sensor": "RGB", "confirmed": False, "confidence": 0.5})
    nir = detection_priority({"sensor": "NIR", "confirmed": False, "confidence": 0.5})
    assert rgb > nir
