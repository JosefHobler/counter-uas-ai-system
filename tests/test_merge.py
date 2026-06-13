"""Tests for cross-sender track merging (common/merge.py)."""

from common.merge import TrackMerger


def _track(tid, az, el, **kw):
    t = dict(id=tid, az=az, el=el, confidence=0.9, threat_score=0.0,
             threat_state="UNKNOWN", sensor="RGB")
    t.update(kw)
    return t


def _state(tracks, received_at=1000.0):
    return {"received_at": received_at, "tracks": tracks}


def test_two_senders_same_target_merge_to_one_track():
    m = TrackMerger(angular_gate_deg=6.0)
    states = {
        "edge": _state([_track(1, 10.0, 5.0)]),
        "ground": _state([_track(7, 10.5, 5.2)]),  # within 6deg gate
    }
    unified = m.merge(states, now=1000.0)
    assert len(unified) == 1
    assert unified[0]["num_sensors"] == 2
    assert {s["sender_id"] for s in unified[0]["sources"]} == {"edge", "ground"}


def test_targets_outside_gate_stay_separate():
    m = TrackMerger(angular_gate_deg=6.0)
    states = {
        "edge": _state([_track(1, 0.0, 0.0)]),
        "ground": _state([_track(7, 40.0, 40.0)]),  # well outside gate
    }
    unified = m.merge(states, now=1000.0)
    assert len(unified) == 2


def test_same_sender_two_nearby_tracks_never_merge():
    # One sensor reporting two close tracks means two targets, not one.
    m = TrackMerger(angular_gate_deg=6.0)
    states = {"edge": _state([_track(1, 10.0, 5.0), _track(2, 10.3, 5.1)])}
    unified = m.merge(states, now=1000.0)
    assert len(unified) == 2


def test_global_id_is_stable_across_frames():
    m = TrackMerger(angular_gate_deg=6.0)
    s1 = {"edge": _state([_track(1, 10.0, 5.0)], received_at=1000.0)}
    first = m.merge(s1, now=1000.0)
    gid = first[0]["global_id"]

    # Same target, slightly drifted, next frame -> same global_id.
    s2 = {"edge": _state([_track(1, 10.4, 5.2)], received_at=1001.0)}
    second = m.merge(s2, now=1001.0)
    assert second[0]["global_id"] == gid


def test_stale_sender_is_dropped():
    m = TrackMerger(angular_gate_deg=6.0, track_ttl_s=3.0)
    states = {"edge": _state([_track(1, 10.0, 5.0)], received_at=1000.0)}
    # now is well past received_at + ttl -> whole sender ignored.
    unified = m.merge(states, now=1010.0)
    assert unified == []


def test_merged_track_takes_worst_threat_state():
    m = TrackMerger(angular_gate_deg=6.0)
    states = {
        "edge": _state([_track(1, 10.0, 5.0, threat_state="STABLE", threat_score=0.2)]),
        "ground": _state([_track(7, 10.2, 5.1, threat_state="APPROACHING",
                                 threat_score=0.8)]),
    }
    unified = m.merge(states, now=1000.0)
    assert len(unified) == 1
    assert unified[0]["threat_state"] == "APPROACHING"
    assert unified[0]["threat_score"] == 0.8  # max across the cluster
