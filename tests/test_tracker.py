"""Tests for the edge centroid tracker (edge-rpi5/tracker.py).

Covers ID assignment / matching and the bbox-area threat heuristic.
"""

from tracker import CentroidTracker, TrackedObject


def _det(cx, cy, bbox=None):
    return {"centroid": (cx, cy), "bbox": bbox or (cx - 5, cy - 5, 10, 10)}


def test_first_update_registers_all_detections():
    tr = CentroidTracker()
    objs = tr.update([_det(100, 100), _det(300, 300)])
    assert len(objs) == 2
    assert {o.id for o in objs} == {0, 1}


def test_nearby_detection_keeps_same_id():
    tr = CentroidTracker()
    tr.update([_det(100, 100)])
    objs = tr.update([_det(105, 102)])  # small move, well under MAX_DISTANCE
    assert len(objs) == 1
    assert objs[0].id == 0


def test_far_detection_gets_new_id():
    tr = CentroidTracker()
    tr.update([_det(100, 100)])
    # Jump far away: the old track coasts on prediction, a new ID is born.
    objs = tr.update([_det(100, 100), _det(700, 700)])
    ids = {o.id for o in objs}
    assert 0 in ids
    assert len(ids) == 2


def test_missing_detection_coasts_then_disappears():
    tr = CentroidTracker()
    tr.update([_det(100, 100)])
    # Feed empty detections repeatedly; the object should eventually deregister.
    for _ in range(60):
        tr.update([])
    assert len(tr.objects) == 0


# -- threat assessment -------------------------------------------------------


def _obj_with_boxes(boxes):
    obj = TrackedObject(0, (boxes[0][0], boxes[0][1]))
    obj.bbox_history.clear()
    for b in boxes:
        obj.bbox_history.append(b)
    return obj


def test_threat_unknown_without_enough_history():
    obj = _obj_with_boxes([(0, 0, 10, 10), (0, 0, 10, 10)])
    assert obj.get_threat_assessment()["state"] == "UNKNOWN"


def test_threat_stable_for_constant_box():
    boxes = [(0, 0, 20, 20)] * 8
    assert _obj_with_boxes(boxes).get_threat_assessment()["state"] == "STABLE"


def test_threat_approaching_for_growing_box():
    # Area roughly doubles over the window -> closing in.
    boxes = [(0, 0, 10, 10)] * 4 + [(0, 0, 16, 16)] * 4
    result = _obj_with_boxes(boxes).get_threat_assessment()
    assert result["state"] == "APPROACHING"
    assert result["score"] > 0.0


def test_threat_receding_for_shrinking_box():
    boxes = [(0, 0, 16, 16)] * 4 + [(0, 0, 8, 8)] * 4
    assert _obj_with_boxes(boxes).get_threat_assessment()["state"] == "RECEDING"
