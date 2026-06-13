"""Cross-sender track merging for the ground-station server.

Each sender (the edge node, the ground-station SAHI tracker) reports tracks in
its *own* frame with sender-local IDs. This module fuses those streams into a
single list of global tracks with stable global IDs, so the dashboard sees one
target per physical object instead of one-per-sensor.

Coordinate-frame caveat
------------------------
Association is done purely in the (azimuth, elevation) angular space that each
sender reports. That is only meaningful if the senders are roughly
co-bore-sighted / share an angular frame: there is **no extrinsic calibration
or world-coordinate projection here**. Pixel coordinates are deliberately
dropped from the merged output because ``x``/``y`` from two different cameras
are not comparable. Treat the merge as a best-effort angular gate, not a
calibrated multi-view fusion.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

# Severity ranking used to pick a single threat_state for a merged track.
_THREAT_SEVERITY = {
    "UNKNOWN": 0,
    "STABLE": 1,
    "RECEDING": 2,
    "APPROACHING": 3,
}

DEFAULT_ANGULAR_GATE_DEG = 6.0
DEFAULT_TRACK_TTL_S = 3.0


class _GlobalTrack:
    """Persistent identity for a merged target, carried across merge cycles."""

    __slots__ = ("global_id", "az", "el", "last_update")

    def __init__(self, global_id: int, az: float, el: float, last_update: float):
        self.global_id = global_id
        self.az = az
        self.el = el
        self.last_update = last_update


class TrackMerger:
    """Associate per-sender tracks into stable global tracks.

    Stateful across calls: it remembers global IDs so a target keeps the same
    ``global_id`` frame to frame (and across a brief miss) as long as it stays
    within the angular gate.
    """

    def __init__(
        self,
        angular_gate_deg: float = DEFAULT_ANGULAR_GATE_DEG,
        track_ttl_s: float = DEFAULT_TRACK_TTL_S,
    ):
        self.angular_gate = angular_gate_deg
        self.ttl = track_ttl_s
        self._next_global_id = 1
        self._tracks: Dict[int, _GlobalTrack] = {}

    # -- public API -----------------------------------------------------------

    def merge(
        self, sender_states: Dict[str, dict], now: Optional[float] = None
    ) -> List[dict]:
        """Return the unified global-track list for the current sender states.

        ``sender_states`` mirrors the server's ``latest_telemetry``: a mapping
        of ``sender_id -> {"received_at"/"timestamp": float, "tracks": [...]}``.
        """
        if now is None:
            now = time.time()

        observations = self._collect_fresh_observations(sender_states, now)
        clusters = self._cluster(observations)
        reps = [self._representative(c) for c in clusters]
        assignment = self._associate(reps)

        new_tracks: Dict[int, _GlobalTrack] = {}
        used_gids = set()
        unified: List[dict] = []

        for ci, cluster in enumerate(clusters):
            az, el = reps[ci]
            gid = assignment.get(ci)
            if gid is None:
                gid = self._next_global_id
                self._next_global_id += 1
            used_gids.add(gid)
            new_tracks[gid] = _GlobalTrack(gid, az, el, now)
            unified.append(self._format(gid, cluster, az, el))

        # Keep recently-seen but currently-unmatched IDs alive so a target that
        # blinks out for a frame keeps its global_id when it returns.
        for gid, track in self._tracks.items():
            if gid in used_gids:
                continue
            if now - track.last_update <= self.ttl:
                new_tracks[gid] = track

        self._tracks = new_tracks
        unified.sort(key=lambda t: t["global_id"])
        return unified

    # -- internals ------------------------------------------------------------

    def _collect_fresh_observations(
        self, sender_states: Dict[str, dict], now: float
    ) -> List[dict]:
        observations: List[dict] = []
        for sender_id, state in sender_states.items():
            stamp = state.get("received_at", state.get("timestamp"))
            if stamp is not None and now - stamp > self.ttl:
                continue  # whole sender has gone stale
            for track in state.get("tracks", []):
                observations.append(
                    {
                        "sender_id": sender_id,
                        "source_id": track.get("id"),
                        "az": float(track.get("az", 0.0)),
                        "el": float(track.get("el", 0.0)),
                        "confidence": float(track.get("confidence", 0.0)),
                        "threat_score": float(track.get("threat_score", 0.0)),
                        "threat_state": track.get("threat_state", "UNKNOWN"),
                        "sensor": track.get("sensor", ""),
                    }
                )
        return observations

    @staticmethod
    def _angular_dist(a: dict, b: dict) -> float:
        return math.hypot(a["az"] - b["az"], a["el"] - b["el"])

    def _cluster(self, observations: List[dict]) -> List[List[dict]]:
        """Group observations of the same physical target across senders.

        Greedy union by ascending angular distance, with a hard rule that a
        cluster never contains two tracks from the *same* sender (one sensor
        reporting two nearby tracks means two targets, not one).
        """
        n = len(observations)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # Candidate cross-sender pairs within the gate, closest first.
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if observations[i]["sender_id"] == observations[j]["sender_id"]:
                    continue
                dist = self._angular_dist(observations[i], observations[j])
                if dist <= self.angular_gate:
                    pairs.append((dist, i, j))
        pairs.sort(key=lambda p: p[0])

        root_senders = {i: {observations[i]["sender_id"]} for i in range(n)}
        for _dist, i, j in pairs:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if root_senders[ri] & root_senders[rj]:
                continue  # merging would put two tracks from one sender together
            parent[rj] = ri
            root_senders[ri] |= root_senders[rj]

        clusters: Dict[int, List[dict]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(observations[i])
        return list(clusters.values())

    @staticmethod
    def _representative(cluster: List[dict]) -> tuple:
        """Confidence-weighted mean (az, el) of a cluster."""
        weight = sum(o["confidence"] for o in cluster) + 1e-6
        az = sum(o["az"] * o["confidence"] for o in cluster) / weight
        el = sum(o["el"] * o["confidence"] for o in cluster) / weight
        # Fall back to a plain mean if every confidence was ~0.
        if weight <= 1e-6 * 2:
            az = sum(o["az"] for o in cluster) / len(cluster)
            el = sum(o["el"] for o in cluster) / len(cluster)
        return az, el

    def _associate(self, reps: List[tuple]) -> Dict[int, int]:
        """Greedily match cluster reps to existing global tracks within gate."""
        candidates = []
        for ci, (az, el) in enumerate(reps):
            for gid, track in self._tracks.items():
                dist = math.hypot(az - track.az, el - track.el)
                if dist <= self.angular_gate:
                    candidates.append((dist, ci, gid))
        candidates.sort(key=lambda c: c[0])

        assignment: Dict[int, int] = {}
        used_clusters = set()
        used_gids = set()
        for _dist, ci, gid in candidates:
            if ci in used_clusters or gid in used_gids:
                continue
            assignment[ci] = gid
            used_clusters.add(ci)
            used_gids.add(gid)
        return assignment

    @staticmethod
    def _format(gid: int, cluster: List[dict], az: float, el: float) -> dict:
        worst = max(cluster, key=lambda o: _THREAT_SEVERITY.get(o["threat_state"], 0))
        senders = sorted({o["sender_id"] for o in cluster})
        sensors = sorted({o["sensor"] for o in cluster if o["sensor"]})
        return {
            "global_id": gid,
            "az": round(az, 3),
            "el": round(el, 3),
            "confidence": max(o["confidence"] for o in cluster),
            "threat_score": max(o["threat_score"] for o in cluster),
            "threat_state": worst["threat_state"],
            "sensors": sensors,
            "num_sensors": len(senders),
            "sources": [
                {"sender_id": o["sender_id"], "id": o["source_id"]} for o in cluster
            ],
        }
