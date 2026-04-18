"""
Centroid tracker with Kalman prediction.
Keeps IDs stable across frames and carries tracks through short occlusions.
"""

import math
from collections import OrderedDict, deque

import cv2
import numpy as np

import config


class TrackedObject:
    """Single tracked object with position history and Kalman state."""

    def __init__(self, object_id, centroid):
        self.id = object_id
        self.positions = deque(maxlen=config.TRACK_HISTORY)
        self.positions.append(centroid)
        self.bbox_history = deque(maxlen=config.TRACK_HISTORY)
        self.disappeared = 0
        self.confirmed_by_ai = False
        self.label = ""
        self.confidence = 0.0
        self.bbox = (0, 0, 0, 0)  # x, y, w, h
        self.is_predicted = False

        # state = [x, y, vx, vy], measurement = [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * config.KALMAN_PROCESS_NOISE
        )
        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * config.KALMAN_MEASUREMENT_NOISE
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array(
            [[centroid[0]], [centroid[1]], [0], [0]],
            dtype=np.float32,
        )

    def predict(self):
        predicted = self.kf.predict()
        return float(predicted[0]), float(predicted[1])

    def correct(self, cx, cy):
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.correct(measurement)

    def update(self, centroid, bbox=None, confirmed=False, label="", confidence=0.0):
        self.correct(centroid[0], centroid[1])
        self.positions.append(centroid)
        self.disappeared = 0
        self.is_predicted = False
        if bbox:
            self.bbox = bbox
            self.bbox_history.append(bbox)
        if confirmed:
            self.confirmed_by_ai = True
            self.label = label
            self.confidence = confidence

    def update_predicted(self, predicted_centroid):
        pred_x, pred_y = predicted_centroid
        self.positions.append((pred_x, pred_y))
        self.disappeared += 1
        self.is_predicted = True

        if self.bbox[2] > 0 and self.bbox[3] > 0:
            old_cx = self.bbox[0] + self.bbox[2] / 2
            old_cy = self.bbox[1] + self.bbox[3] / 2
            dx = pred_x - old_cx
            dy = pred_y - old_cy
            self.bbox = (
                int(round(self.bbox[0] + dx)),
                int(round(self.bbox[1] + dy)),
                self.bbox[2],
                self.bbox[3],
            )

    def get_direction(self):
        """
        Return (angle_deg, speed_px_per_frame, direction_label).
        0 = up, 90 = right, 180 = down, 270 = left.
        """
        if len(self.positions) < 5:
            return 0.0, 0.0, "---"

        n = min(15, len(self.positions))
        recent = list(self.positions)[-n:]

        old_x = sum(p[0] for p in recent[: n // 2]) / (n // 2)
        old_y = sum(p[1] for p in recent[: n // 2]) / (n // 2)
        new_x = sum(p[0] for p in recent[n // 2 :]) / (n - n // 2)
        new_y = sum(p[1] for p in recent[n // 2 :]) / (n - n // 2)

        dx = new_x - old_x
        dy = new_y - old_y

        speed = math.sqrt(dx * dx + dy * dy)
        if speed < 1.0:
            return 0.0, 0.0, "STATIONARY"

        angle = math.degrees(math.atan2(dx, -dy))
        if angle < 0:
            angle += 360

        directions = [
            (337.5, 360, "N"),
            (0, 22.5, "N"),
            (22.5, 67.5, "NE"),
            (67.5, 112.5, "E"),
            (112.5, 157.5, "SE"),
            (157.5, 202.5, "S"),
            (202.5, 247.5, "SW"),
            (247.5, 292.5, "W"),
            (292.5, 337.5, "NW"),
        ]

        direction_text = "?"
        for low, high, text in directions:
            if low <= angle < high:
                direction_text = text
                break

        return angle, speed, direction_text

    def get_trail(self):
        return list(self.positions)

    def get_threat_assessment(self):
        """
        Estimate whether the tracked object is approaching or receding.
        Uses bounding-box area growth as a cheap depth proxy.
        """
        valid_boxes = [
            bbox for bbox in self.bbox_history
            if bbox[2] > 0 and bbox[3] > 0
        ]
        if len(valid_boxes) < 4:
            return {
                "state": "UNKNOWN",
                "score": 0.0,
                "delta_pct": 0.0,
                "area_ratio": 1.0,
            }

        sample_count = min(12, len(valid_boxes))
        recent = valid_boxes[-sample_count:]
        midpoint = sample_count // 2
        if midpoint == 0 or midpoint == sample_count:
            return {
                "state": "UNKNOWN",
                "score": 0.0,
                "delta_pct": 0.0,
                "area_ratio": 1.0,
            }

        old_area = sum(max(1.0, box[2] * box[3]) for box in recent[:midpoint]) / midpoint
        new_area = sum(max(1.0, box[2] * box[3]) for box in recent[midpoint:]) / (sample_count - midpoint)

        if old_area <= 0.0:
            return {
                "state": "UNKNOWN",
                "score": 0.0,
                "delta_pct": 0.0,
                "area_ratio": 1.0,
            }

        area_ratio = new_area / old_area
        delta_pct = ((new_area - old_area) / old_area) * 100.0

        # Use log-ratio so doubling and halving behave symmetrically.
        signed_strength = math.log(area_ratio)
        score = min(1.0, abs(signed_strength) / math.log(1.8))

        if score < 0.12:
            state = "STABLE"
        elif signed_strength > 0:
            state = "APPROACHING"
        else:
            state = "RECEDING"

        return {
            "state": state,
            "score": score,
            "delta_pct": delta_pct,
            "area_ratio": area_ratio,
        }


class CentroidTracker:
    """Distance-based tracker using Kalman prediction for matching."""

    def __init__(self):
        self.next_id = 0
        self.objects = OrderedDict()

    def register(self, centroid, bbox=None):
        obj = TrackedObject(self.next_id, centroid)
        if bbox:
            obj.bbox = bbox
            obj.bbox_history.append(bbox)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj

    def deregister(self, object_id):
        del self.objects[object_id]

    def update(self, detections):
        if len(self.objects) == 0:
            for det in detections:
                obj = self.register(det["centroid"], det.get("bbox"))
                if det.get("confirmed"):
                    obj.confirmed_by_ai = True
                    obj.label = det.get("label", "")
                    obj.confidence = det.get("confidence", 0.0)
            return list(self.objects.values())

        predicted_positions = {
            obj_id: obj.predict()
            for obj_id, obj in self.objects.items()
        }

        if len(detections) == 0:
            to_delete = []
            for obj_id, obj in self.objects.items():
                obj.update_predicted(predicted_positions[obj_id])
                if obj.disappeared > config.MAX_PREDICTED_FRAMES:
                    to_delete.append(obj_id)
            for obj_id in to_delete:
                self.deregister(obj_id)
            return list(self.objects.values())

        object_ids = list(self.objects.keys())
        object_centroids = [predicted_positions[oid] for oid in object_ids]
        det_centroids = [d["centroid"] for d in detections]

        distances = np.zeros((len(object_centroids), len(det_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(det_centroids):
                distances[i, j] = math.sqrt(
                    (obj_centroid[0] - det_centroid[0]) ** 2
                    + (obj_centroid[1] - det_centroid[1]) ** 2
                )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > config.MAX_DISTANCE:
                continue

            obj_id = object_ids[row]
            det = detections[col]
            self.objects[obj_id].update(
                det["centroid"],
                det.get("bbox"),
                det.get("confirmed", False),
                det.get("label", ""),
                det.get("confidence", 0.0),
            )

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.objects[obj_id].update_predicted(predicted_positions[obj_id])
            if self.objects[obj_id].disappeared > config.MAX_PREDICTED_FRAMES:
                self.deregister(obj_id)

        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            det = detections[col]
            obj = self.register(det["centroid"], det.get("bbox"))
            if det.get("confirmed"):
                obj.confirmed_by_ai = True
                obj.label = det.get("label", "")
                obj.confidence = det.get("confidence", 0.0)

        return list(self.objects.values())
