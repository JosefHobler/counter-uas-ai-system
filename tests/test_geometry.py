"""Tests for the pure geometry helpers (common/geometry.py)."""

import pytest

from common.geometry import (
    associate_by_iou,
    bbox_iou,
    clamp_crop_region,
    map_local_box_to_full,
)


# -- bbox_iou (canonical; also re-exported by drone_detector) ----------------


def test_iou_identical_is_one():
    box = (0, 0, 10, 10)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_half_overlap():
    assert bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


# -- clamp_crop_region -------------------------------------------------------


def test_crop_no_margin_is_the_box():
    assert clamp_crop_region((10, 20, 30, 40), 100, 100, margin=0.0) == (10, 20, 40, 60)


def test_crop_margin_pads_by_longer_side():
    # longer side = 40, margin 0.5 -> pad 20 on each side.
    assert clamp_crop_region((10, 20, 30, 40), 200, 200, margin=0.5) == (0, 0, 60, 80)


def test_crop_clamped_to_frame():
    x0, y0, x1, y1 = clamp_crop_region((90, 90, 30, 30), 100, 100, margin=0.0)
    assert (x0, y0, x1, y1) == (90, 90, 100, 100)


def test_crop_fully_offframe_is_empty():
    x0, y0, x1, y1 = clamp_crop_region((200, 200, 10, 10), 100, 100, margin=0.0)
    assert x1 - x0 == 0 or y1 - y0 == 0  # zero-area => nothing to crop


# -- map_local_box_to_full ---------------------------------------------------


def test_map_local_box_adds_crop_origin():
    assert map_local_box_to_full((5, 5, 20, 10), (100, 50)) == (105, 55, 20, 10)


# -- associate_by_iou --------------------------------------------------------


def test_associate_matches_overlapping_boxes():
    a = [(0, 0, 10, 10), (100, 100, 10, 10)]
    b = [(102, 100, 10, 10), (1, 1, 10, 10)]  # b[0]~a[1], b[1]~a[0]
    matches = associate_by_iou(a, b, iou_threshold=0.3)
    pairs = {(i, j) for i, j, _ in matches}
    assert pairs == {(0, 1), (1, 0)}


def test_associate_respects_threshold():
    a = [(0, 0, 10, 10)]
    b = [(9, 0, 10, 10)]  # tiny overlap, IoU ~ 1/19
    assert associate_by_iou(a, b, iou_threshold=0.3) == []


def test_associate_is_one_to_one_greedy():
    # Two a-boxes both overlap one b-box; only the best match wins it.
    a = [(0, 0, 10, 10), (1, 0, 10, 10)]
    b = [(0, 0, 10, 10)]
    matches = associate_by_iou(a, b, iou_threshold=0.1)
    assert len(matches) == 1
    assert matches[0][0] == 0  # the exact-overlap a-box wins
