"""
Anti-Drone Detection System v3
YOLO handles primary RGB detection.
KNN sky scanning handles small moving targets.
Optional NIR fusion adds a lightweight second sensor branch.
"""

import argparse
import math
import sys
import time
import threading
import queue

import cv2
import numpy as np

import config
from tracker import CentroidTracker
from telemetry_client import TelemetryClient

_yolo_model = None


def resolve_yolo_input_size(model):
    """Return the model's required square input size when it can be inferred."""
    cached_size = getattr(model, "_resolved_imgsz", None)
    if cached_size is not None:
        return cached_size

    imgsz = config.YOLO_INPUT_SIZE
    model_path = getattr(model, "ckpt_path", None) or getattr(model, "model_name", None)
    if not model_path and isinstance(getattr(model, "model", None), str):
        model_path = model.model

    if isinstance(model_path, str) and model_path.lower().endswith(".onnx"):
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_shape = session.get_inputs()[0].shape
            height = input_shape[2] if len(input_shape) > 2 else None
            width = input_shape[3] if len(input_shape) > 3 else None
            if isinstance(height, int) and isinstance(width, int) and height == width:
                imgsz = height
                print(f"[AI] ONNX input size detected: {imgsz}x{imgsz}")
        except Exception as exc:
            print(f"[AI] Warning: could not inspect ONNX input size ({exc}).")

    setattr(model, "_resolved_imgsz", imgsz)
    return imgsz


def get_yolo_model():
    """Load the YOLO model once."""
    global _yolo_model
    if _yolo_model is None:
        print(f"[AI] Loading {config.YOLO_MODEL}...")
        try:
            from ultralytics import YOLO

            _yolo_model = YOLO(config.YOLO_MODEL, task="detect")
            resolve_yolo_input_size(_yolo_model)
            print("[AI] Model ready.")
        except ImportError:
            print("[AI] ERROR: ultralytics is not installed.")
            print("[AI] Run: pip install ultralytics")
            sys.exit(1)
        except Exception as exc:
            print(f"[AI] ERROR: {exc}")
            sys.exit(1)
    return _yolo_model


def preprocess_frame(frame, mode="day"):
    """Preprocess a frame for the selected sensor mode."""
    if mode == "night":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_GRID_SIZE,
        )
        enhanced = clahe.apply(gray)
        frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    elif mode == "thermal":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        frame = cv2.applyColorMap(normalized, config.THERMAL_COLORMAP)
    return frame


def get_sahi_slices(frame_width, frame_height, slice_w, slice_h, overlap_ratio):
    step_x = max(1, int(slice_w * (1 - overlap_ratio)))
    step_y = max(1, int(slice_h * (1 - overlap_ratio)))

    slices = []
    for y in range(0, frame_height, step_y):
        for x in range(0, frame_width, step_x):
            x1, y1 = x, y
            x2 = min(frame_width, x1 + slice_w)
            y2 = min(frame_height, y1 + slice_h)

            if x2 - x1 < slice_w:
                x1 = max(0, x2 - slice_w)
            if y2 - y1 < slice_h:
                y1 = max(0, y2 - slice_h)
            
            if (x1, y1, x2, y2) not in slices:
                slices.append((x1, y1, x2, y2))
            
            if x2 == frame_width:
                break
        if y2 == frame_height:
            break
            
    return slices


def nms_detections(detections, iou_threshold=0.3):
    if not detections:
        return []
        
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    
    for det in ordered:
        duplicate = False
        for kept_det in kept:
            if bbox_iou(det["bbox"], kept_det["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
            
    return kept


class SmartCropper:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorKNN(history=300, dist2Threshold=400, detectShadows=False)
        self.warmup = 5
        self.frame_count = 0

    def get_motion_crops(self, frame, slice_w, slice_h):
        self.frame_count += 1
        
        # Fast downscale for motion detection to save CPU
        scale = 0.5
        small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if self.frame_count < self.warmup:
            self.bg_sub.apply(gray, learningRate=0.1)
            # Return None to trigger a full SAHI grid sweep during the warmup phase
            return None
            
        fg_mask = self.bg_sub.apply(gray, learningRate=0.005)
        
        # Morph ops to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = frame.shape[:2]
        crops = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = getattr(config, "SAHI_MIN_MOTION_AREA", 10)
            max_area = getattr(config, "SAHI_MAX_MOTION_AREA", 10000)
            if area < min_area or area > max_area: 
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Map back to full res
            x, y, cw, ch = int(x/scale), int(y/scale), int(cw/scale), int(ch/scale)
            
            # Center a slice_w x slice_h box around the moving contour
            cx = x + cw // 2
            cy = y + ch // 2
            
            x1 = max(0, cx - slice_w // 2)
            y1 = max(0, cy - slice_h // 2)
            x2 = min(w, x1 + slice_w)
            y2 = min(h, y1 + slice_h)
            
            if x2 - x1 < slice_w: x1 = max(0, x2 - slice_w)
            if y2 - y1 < slice_h: y1 = max(0, y2 - slice_h)
            
            crop_rect = (x1, y1, x2, y2)
            if crop_rect not in crops:
                crops.append(crop_rect)
            
        max_crops = getattr(config, "SAHI_MAX_CROPS", 4)
        if len(crops) > max_crops:
            # Overload triggered: camera panning or extreme noise
            # Returning None signals detect_with_yolo to perform a standard full-grid scan
            return None
            
        return crops


_smart_cropper = None
_sahi_frame_tick = 0

def detect_with_yolo(frame, model):
    """Run YOLO on the frame, optionally using SAHI slicing."""
    global _smart_cropper, _sahi_frame_tick
    imgsz = resolve_yolo_input_size(model)
    use_sahi = getattr(config, "SAHI_ENABLED", False)
    use_smart_crop = getattr(config, "SAHI_MOTION_SMART_CROP", False)
    sahi_interval = getattr(config, "SAHI_FULL_SCAN_INTERVAL", 30)
    
    if use_sahi:
        _sahi_frame_tick += 1
        slice_w = getattr(config, "SAHI_SLICE_WIDTH", imgsz)
        slice_h = getattr(config, "SAHI_SLICE_HEIGHT", imgsz)
        overlap = getattr(config, "SAHI_OVERLAP_RATIO", 0.2)
        
        h, w = frame.shape[:2]
        
        slices = []
        if use_smart_crop:
            if _smart_cropper is None:
                _smart_cropper = SmartCropper()
                
            motion_crops = _smart_cropper.get_motion_crops(frame, slice_w, slice_h)
            
            if _sahi_frame_tick % sahi_interval == 0 or motion_crops is None:
                slices = get_sahi_slices(w, h, slice_w, slice_h, overlap)
            else:
                slices = motion_crops
        else:
            slices = get_sahi_slices(w, h, slice_w, slice_h, overlap)
            
        if not slices:
            return []
            
        all_detections = []
        
        for (x1, y1, x2, y2) in slices:
            crop = frame[y1:y2, x1:x2]
            results = model(crop, stream=False, verbose=False, conf=config.YOLO_CONFIDENCE, imgsz=imgsz)
            
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if not config.ACCEPT_ALL_CLASSES and cls not in config.DRONE_CLASSES:
                        continue

                    crop_x1, crop_y1, crop_x2, crop_y2 = map(int, box.xyxy[0])
                    global_x1 = crop_x1 + x1
                    global_y1 = crop_y1 + y1
                    global_w = crop_x2 - crop_x1
                    global_h = crop_y2 - crop_y1
                    cx = global_x1 + global_w // 2
                    cy = global_y1 + global_h // 2

                    class_names = model.names if hasattr(model, "names") else {}
                    cls_name = class_names.get(cls, f"cls{cls}")

                    all_detections.append({
                        "centroid": (cx, cy),
                        "bbox": (global_x1, global_y1, global_w, global_h),
                        "confirmed": True,
                        "label": f"TARGET [{cls_name}]",
                        "confidence": conf,
                        "area": global_w * global_h,
                    })
                    
        return nms_detections(all_detections, iou_threshold=0.3)
    
    results = model(
        frame,
        stream=False,
        verbose=False,
        conf=config.YOLO_CONFIDENCE,
        imgsz=imgsz,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if not config.ACCEPT_ALL_CLASSES and cls not in config.DRONE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            class_names = model.names if hasattr(model, "names") else {}
            cls_name = class_names.get(cls, f"cls{cls}")

            detections.append(
                {
                    "centroid": (cx, cy),
                    "bbox": (x1, y1, w, h),
                    "confirmed": True,
                    "label": f"TARGET [{cls_name}]",
                    "confidence": conf,
                    "area": w * h,
                }
            )

    return detections


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def prepare_sensor_detection(det, sensor, default_confidence):
    prepared = dict(det)
    prepared["sensor"] = sensor
    prepared["confidence"] = clamp01(prepared.get("confidence", default_confidence))
    prepared["confirmed"] = bool(prepared.get("confirmed", False))
    return prepared


def is_same_target(det_a, det_b, max_distance):
    ax, ay = det_a["centroid"]
    bx, by = det_b["centroid"]
    return math.hypot(ax - bx, ay - by) <= max_distance


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def detection_priority(det):
    sensor = det.get("sensor", "")
    sensor_score = {
        "RGB+NIR": 4,
        "RGB": 3,
        "NIR": 2,
    }.get(sensor, 1)
    confirmed_score = 1 if det.get("confirmed", False) else 0
    confidence = clamp01(det.get("confidence", 0.0))
    return (confirmed_score, sensor_score, confidence)


def suppress_duplicate_detections(detections):
    """Collapse nearby same-target detections before tracker update."""
    if len(detections) <= 1:
        return detections

    ordered = sorted(detections, key=detection_priority, reverse=True)
    kept = []

    for det in ordered:
        duplicate = False
        for kept_det in kept:
            same_centroid = is_same_target(det, kept_det, config.FUSION_MATCH_DISTANCE * 0.75)
            same_box = bbox_iou(det["bbox"], kept_det["bbox"]) >= 0.35
            if same_centroid or same_box:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept


def primary_target_priority(obj, frame_center):
    """Prefer live, confident, stable tracks for the zoom window."""
    bx, by, bw, bh = obj.bbox
    bbox_ok = 1 if bw > 0 and bh > 0 else 0
    live_track = 0 if getattr(obj, "is_predicted", False) else 1
    ai_confirmed = 1 if getattr(obj, "confirmed_by_ai", False) else 0
    confidence = clamp01(getattr(obj, "confidence", 0.0))
    trail_len = len(obj.get_trail())

    cx, cy = frame_center
    obj_cx = bx + bw / 2
    obj_cy = by + bh / 2
    center_dist = math.hypot(obj_cx - cx, obj_cy - cy) if bbox_ok else 1e9

    return (
        bbox_ok,
        live_track,
        ai_confirmed,
        confidence,
        trail_len,
        -center_dist,
    )


def choose_primary_target(tracked_objects, frame_center):
    """Choose a visible target for the zoom window without startup delay."""
    if not tracked_objects:
        return None

    candidates = [
        obj
        for obj in tracked_objects
        if obj.bbox[2] > 0 and obj.bbox[3] > 0
    ]
    if not candidates:
        return tracked_objects[0]

    return max(candidates, key=lambda obj: primary_target_priority(obj, frame_center))


def adaptive_confirm_threshold(det):
    """Smaller / farther-looking targets get a lower confirmation threshold."""
    _, _, w, h = det["bbox"]
    area = max(1, w * h)
    max_dim = max(w, h)

    if area <= 24 * 24 or max_dim <= 28:
        return 0.28
    if area <= 40 * 40 or max_dim <= 42:
        return 0.40
    if area <= 80 * 80:
        return 0.52
    return config.FUSION_CONFIRM_THRESHOLD


def score_nir_detection(frame, bbox):
    """Score an NIR candidate using local contrast and texture."""
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    pad = max(8, int(max(w, h) * 0.75))
    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(width, x2 + pad)
    oy2 = min(height, y2 + pad)
    outer = gray[oy1:oy2, ox1:ox2]

    roi_mean = float(np.mean(roi))
    roi_std = float(np.std(roi))
    outer_mean = float(np.mean(outer)) if outer.size else roi_mean
    contrast = abs(roi_mean - outer_mean)
    area_norm = min(1.0, (w * h) / 1600.0)

    score = (
        0.45 * clamp01(contrast / 80.0)
        + 0.35 * clamp01(roi_std / 60.0)
        + 0.20 * area_norm
    )
    return clamp01(score)


def fuse_detections(rgb_detections, nir_detections):
    """Soft-OR fusion with a confidence bonus when both sensors agree."""
    fused = []
    used_nir = set()

    for rgb_det in rgb_detections:
        best_idx = None
        best_dist = None
        for idx, nir_det in enumerate(nir_detections):
            if idx in used_nir:
                continue
            dist = math.hypot(
                rgb_det["centroid"][0] - nir_det["centroid"][0],
                rgb_det["centroid"][1] - nir_det["centroid"][1],
            )
            if dist > config.FUSION_MATCH_DISTANCE:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            # Keep RGB baseline intact. Fusion must not weaken a valid RGB hit.
            fused_confidence = clamp01(rgb_det.get("confidence", 0.0))
            det = dict(rgb_det)
            det["fusion_score"] = fused_confidence
            det["sensor"] = "RGB"
            det["confirmed"] = bool(rgb_det.get("confirmed", False)) or (
                fused_confidence >= adaptive_confirm_threshold(det)
            )
            fused.append(det)
            continue

        nir_det = nir_detections[best_idx]
        used_nir.add(best_idx)

        rgb_score = rgb_det.get("confidence", 0.0)
        nir_score = nir_det.get("confidence", 0.0)
        boosted_confidence = clamp01(
            rgb_score * config.FUSION_RGB_WEIGHT
            + nir_score * config.FUSION_NIR_WEIGHT
            + config.FUSION_BONUS
        )
        # Never let fusion reduce the original RGB confidence.
        fused_confidence = max(clamp01(rgb_score), boosted_confidence)

        # Preserve RGB geometry when RGB exists so distant small targets
        # do not get diluted by the helper sensor branch.
        fused_bbox = rgb_det["bbox"]
        fused_centroid = rgb_det["centroid"]

        fused_det = {
            "centroid": fused_centroid,
            "bbox": fused_bbox,
            "confirmed": bool(rgb_det.get("confirmed", False)) or (
                fused_confidence >= adaptive_confirm_threshold(rgb_det)
            ),
            "label": "TARGET [RGB+NIR]",
            "confidence": fused_confidence,
            "fusion_score": fused_confidence,
            "sensor": "RGB+NIR",
            "area": fused_bbox[2] * fused_bbox[3],
        }
        if rgb_det.get("label"):
            fused_det["label"] = rgb_det["label"]
        fused.append(
            fused_det
        )

    for idx, nir_det in enumerate(nir_detections):
        if idx in used_nir:
            continue
        # NIR-only targets remain possible, but with adaptive confirmation.
        fused_confidence = max(
            clamp01(nir_det.get("confidence", 0.0)),
            clamp01(nir_det.get("confidence", 0.0) * config.FUSION_NIR_WEIGHT),
        )
        det = dict(nir_det)
        det["fusion_score"] = fused_confidence
        det["sensor"] = "NIR"
        det["confirmed"] = bool(nir_det.get("confirmed", False)) or (
            fused_confidence >= adaptive_confirm_threshold(det)
        )
        fused.append(det)

    return fused


def put_text_lines(img, lines, x, y, color, scale, thickness, align="left"):
    y_offset = y
    for line in lines:
        if align == "center":
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, scale, thickness)
            cv2.putText(img, line, (x - tw // 2, y_offset), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)
        elif align == "right":
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, scale, thickness)
            cv2.putText(img, line, (x - tw, y_offset), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)
        else:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, scale, thickness)
            cv2.putText(img, line, (x, y_offset), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)
        y_offset += th + max(4, int(8 * scale))


def format_threat_line(obj):
    assessment = obj.get_threat_assessment()
    state = assessment["state"]
    score = assessment["score"]
    delta_pct = assessment["delta_pct"]

    if state == "UNKNOWN":
        return "Threat: ---"
    if state == "STABLE":
        return f"Threat: STABLE ({score:.2f})"
    return f"Threat: {state} ({score:.2f}, {delta_pct:+.0f}%)"


def draw_hud(frame, tracked_objects, fps, frame_center, detection_mode="YOLO", show_threat=False):
    """Render the HUD onto the frame."""
    height, width = frame.shape[:2]
    cx, cy = frame_center
    color = config.COLOR_CROSSHAIR
    scale = 1.0 # font scale
    thick = 1

    # Full screen crosshairs
    cv2.line(frame, (cx, 0), (cx, height), color, 1)
    cv2.line(frame, (0, cy), (width, cy), color, 1)

    # Center reticle
    ret_size = 40
    ret_gap = 20
    # Top-left corner
    cv2.line(frame, (cx - ret_gap - ret_size, cy - ret_gap), (cx - ret_gap, cy - ret_gap), color, 2)
    cv2.line(frame, (cx - ret_gap, cy - ret_gap), (cx - ret_gap, cy - ret_gap - ret_size), color, 2)
    # Top-right corner
    cv2.line(frame, (cx + ret_gap, cy - ret_gap), (cx + ret_gap + ret_size, cy - ret_gap), color, 2)
    cv2.line(frame, (cx + ret_gap, cy - ret_gap), (cx + ret_gap, cy - ret_gap - ret_size), color, 2)
    # Bottom-left corner
    cv2.line(frame, (cx - ret_gap - ret_size, cy + ret_gap), (cx - ret_gap, cy + ret_gap), color, 2)
    cv2.line(frame, (cx - ret_gap, cy + ret_gap), (cx - ret_gap, cy + ret_gap + ret_size), color, 2)
    # Bottom-right corner
    cv2.line(frame, (cx + ret_gap, cy + ret_gap), (cx + ret_gap + ret_size, cy + ret_gap), color, 2)
    cv2.line(frame, (cx + ret_gap, cy + ret_gap), (cx + ret_gap, cy + ret_gap + ret_size), color, 2)

    # Center box inner mark
    cv2.line(frame, (cx - 5, cy - 5), (cx - 5, cy + 5), color, 1)
    cv2.line(frame, (cx + 5, cy - 5), (cx + 5, cy + 5), color, 1)
    cv2.line(frame, (cx - 5, cy - 5), (cx + 5, cy - 5), color, 1)
    cv2.line(frame, (cx - 5, cy + 5), (cx + 5, cy + 5), color, 1)
    # And small plus
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 1)

    # Texts
    top_left_lines = [
      #    "TARGET SELECTOR", "Klase: \"Aa\"", "Henerasyon: 1", "Bloke: \"Ka\"", "Bersyon: 2", "",
       #   "TARGET TRACKING", "Uri/Klase: \"Ga\"", "Henerasyon: 1", "Bloke: \"Aa\"", "Bersyon: 2"
    ]
    put_text_lines(frame, top_left_lines, 20, 30, color, scale, thick + 1, align="left")

    top_center_lines = [
      #    "DESIGNER AND DEVELOPER", ">>>>> Time Lapse Coder, 2025 <<<<<"
        ]
    put_text_lines(frame, top_center_lines, cx, 30, color, scale, thick + 1, align="center")

    top_right_lines = [
      #    "TEST SETTINGS", "Source video: 3-CH", "Seeker mode: 3-CH", "Frames to skip: 0"
    ]
    put_text_lines(frame, top_right_lines, width - 200, 30, color, scale, thick + 1, align="left")
    
    # Bottom Left
    bottom_left_lines = [
    #    "TRACKER COMMANDS",
      #    "Target Box: 'Q'   Lock: 'C'   Cancel Lock: 'V'   White/Black Hot: 'H'   Manual Adjust: 'N'"
    ]
    put_text_lines(frame, bottom_left_lines, 20, height - 40, color, scale, thick, align="left")

    # Bottom Right
    bottom_right_line = ""
     #  "Code for demo by Time Lapse Coder"
    put_text_lines(frame, [bottom_right_line], width - 20, height - 20, (255, 255, 255), scale, thick, align="right")

    # Sensor Mode
    put_text_lines(frame, [f"SENSOR MODE: {config.MODE.upper()}"], cx, cy + 150, color, scale, thick + 1, align="center")

    # Find main target for the zoom window immediately.
    primary_target = choose_primary_target(tracked_objects, frame_center)

    t_x, t_y = "---", "---"
    t_az, t_el = "---", "---"
    t_conf = "---"
    if primary_target:
        bx, by, bw, bh = primary_target.bbox
        if bw > 0 and bh > 0:
            tcx = bx + bw // 2
            tcy = by + bh // 2
            t_x, t_y = str(tcx), str(tcy)
            t_az = f"{((tcx - cx) / width) * config.FOV_X:.1f}"
            t_el = f"{((cy - tcy) / height) * config.FOV_Y:.1f}"
            t_conf = f"{clamp01(getattr(primary_target, 'confidence', 0.0)):.2f}"

    mid_right_data = [
        "TRACKER DATA",
        f"Image XY: {t_x}, {t_y}",
        f"Az / El: {t_az}, {t_el}",
        f"Conf: {t_conf}",
    ]
    if show_threat and primary_target:
        mid_right_data.append(format_threat_line(primary_target))
    put_text_lines(frame, mid_right_data, width - 200, 180, color, scale, thick + 1, align="left")

    # Draw simple brackets for all targets on main screen
    for obj in tracked_objects:
        bx, by, bw, bh = obj.bbox
        if bw > 0 and bh > 0:
            bw_p = max(5, bw // 4)
            bh_p = max(5, bh // 4)
            cv2.line(frame, (bx, by), (bx + bw_p, by), color, 1)
            cv2.line(frame, (bx, by), (bx, by + bh_p), color, 1)
            cv2.line(frame, (bx + bw, by), (bx + bw - bw_p, by), color, 1)
            cv2.line(frame, (bx + bw, by), (bx + bw, by + bh_p), color, 1)
            cv2.line(frame, (bx, by + bh), (bx + bw_p, by + bh), color, 1)
            cv2.line(frame, (bx, by + bh), (bx, by + bh - bh_p), color, 1)
            cv2.line(frame, (bx + bw, by + bh), (bx + bw - bw_p, by + bh), color, 1)
            cv2.line(frame, (bx + bw, by + bh), (bx + bw, by + bh - bh_p), color, 1)

            if show_threat:
                threat_text = format_threat_line(obj).replace("Threat: ", "")
                text_y = max(20, by - 8)
                cv2.putText(
                    frame,
                    threat_text,
                    (bx, text_y),
                    cv2.FONT_HERSHEY_PLAIN,
                    scale,
                    color,
                    thick + 1,
                )

    # Draw PIP Seeker View
    pip_w, pip_h = 300, 300
    pip_x = width - pip_w - 50
    pip_y = height // 2 - pip_h // 2 + 100
    
    # Draw translucent base for PIP
    overlay = frame.copy()
    cv2.rectangle(overlay, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), config.COLOR_INFO_BG, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    # Draw border
    cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (0, 0, 0), 1)
    
    put_text_lines(frame, ["SEEKER VIEW"], pip_x + 10, pip_y + 25, color, scale, thick + 1, align="left")

    if primary_target:
        bx, by, bw, bh = primary_target.bbox
        if bw > 0 and bh > 0:
            # Crop around target
            pad = 50
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(width, bx + bw + pad)
            y2 = min(height, by + bh + pad)
            
            crop = frame[y1:y2, x1:x2].copy()  # Uses frame with HUD drawn so far
            cr_h, cr_w = crop.shape[:2]
            
            if cr_h > 0 and cr_w > 0:
                zoom_factor = min(pip_w / cr_w, pip_h / cr_h)
                new_w = int(cr_w * zoom_factor)
                new_h = int(cr_h * zoom_factor)
                
                resized_crop = cv2.resize(crop, (new_w, new_h))
                cx_pip = pip_x + pip_w // 2
                cy_pip = pip_y + pip_h // 2
                
                tgt_x = cx_pip - new_w // 2
                tgt_y = cy_pip - new_h // 2
                
                # Overlay crop onto PIP
                frame[tgt_y:tgt_y+new_h, tgt_x:tgt_x+new_w] = resized_crop
                
                # Draw dashed target box inside PIP
                bx_pip = tgt_x + int((bx - x1) * zoom_factor)
                by_pip = tgt_y + int((by - y1) * zoom_factor)
                bw_pip = int(bw * zoom_factor)
                bh_pip = int(bh * zoom_factor)
                
                # Draw thick brackets
                bw_p = max(5, bw_pip // 4)
                bh_p = max(5, bh_pip // 4)
                cv2.line(frame, (bx_pip, by_pip), (bx_pip + bw_p, by_pip), color, 3)
                cv2.line(frame, (bx_pip, by_pip), (bx_pip, by_pip + bh_p), color, 3)
                cv2.line(frame, (bx_pip + bw_pip, by_pip), (bx_pip + bw_pip - bw_p, by_pip), color, 3)
                cv2.line(frame, (bx_pip + bw_pip, by_pip), (bx_pip + bw_pip, by_pip + bh_p), color, 3)
                cv2.line(frame, (bx_pip, by_pip + bh_pip), (bx_pip + bw_p, by_pip + bh_pip), color, 3)
                cv2.line(frame, (bx_pip, by_pip + bh_pip), (bx_pip, by_pip + bh_pip - bh_p), color, 3)
                cv2.line(frame, (bx_pip + bw_pip, by_pip + bh_pip), (bx_pip + bw_pip - bw_p, by_pip + bh_pip), color, 3)
                cv2.line(frame, (bx_pip + bw_pip, by_pip + bh_pip), (bx_pip + bw_pip, by_pip + bh_pip - bh_p), color, 3)

    return frame


def parse_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def create_runtime_settings(profile_name, use_nir):
    settings = {
        "process_every_n_frames": config.PROCESS_EVERY_N_FRAMES,
        "yolo_input_size": config.YOLO_INPUT_SIZE,
        "show_minimap": config.SHOW_MINIMAP,
        "nir_every_n_frames": 1,
        "nir_resize_scale": 1.0,
    }

    if profile_name == "balanced":
        settings.update(
            {
                "process_every_n_frames": 7 if use_nir else 6,
                "yolo_input_size": 960,
                "show_minimap": False,
                "nir_every_n_frames": 1 if use_nir else 1,
                "nir_resize_scale": 0.9 if use_nir else 1.0,
            }
        )
    elif profile_name == "pi5":
        settings.update(
            {
                "process_every_n_frames": 10 if use_nir else 8,
                "yolo_input_size": 640,
                "show_minimap": False,
                "nir_every_n_frames": 2 if use_nir else 1,
                "nir_resize_scale": 0.75 if use_nir else 1.0,
            }
        )

    return settings


def apply_runtime_settings(settings):
    config.PROCESS_EVERY_N_FRAMES = settings["process_every_n_frames"]
    config.YOLO_INPUT_SIZE = settings["yolo_input_size"]
    config.SHOW_MINIMAP = settings["show_minimap"]


def init_timing_stats():
    return {
        "read": 0.0,
        "preprocess_rgb": 0.0,
        "preprocess_nir": 0.0,
        "yolo": 0.0,
        "fusion": 0.0,
        "tracker": 0.0,
        "hud": 0.0,
        "display": 0.0,
        "frames": 0,
    }


def add_timing(stats, key, started_at):
    stats[key] += time.perf_counter() - started_at


def print_timing_report(stats, elapsed_seconds):
    frame_count = max(1, stats["frames"])
    items = []
    for key in [
        "read",
        "preprocess_rgb",
        "yolo",
        "fusion",
        "tracker",
        "hud",
        "display",
    ]:
        avg_ms = (stats[key] / frame_count) * 1000.0
        share = (stats[key] / elapsed_seconds * 100.0) if elapsed_seconds > 0 else 0.0
        items.append((avg_ms, f"{key}={avg_ms:.1f}ms ({share:.0f}%)"))

    items.sort(reverse=True)
    print("[PROFILE] " + " | ".join(item for _, item in items[:5]))


class DetectionWorker:
    def __init__(self, model, use_nir):
        self.model = model
        self.use_nir = use_nir
        self.q_in = queue.Queue(maxsize=1)
        self.q_out = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                item = self.q_in.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if item is None:
                break
                
            frame, nir_frame = item
            
            yolo_detections = detect_with_yolo(frame, self.model)
            rgb_candidates = [
                prepare_sensor_detection(det, "RGB", det.get("confidence", 0.0))
                for det in yolo_detections
            ]
            
            nir_candidates = []
            
            fused = fuse_detections(rgb_candidates, nir_candidates) if self.use_nir else rgb_candidates
            fused = suppress_duplicate_detections(fused)
            
            if self.q_out.full():
                try:
                    self.q_out.get_nowait()
                except queue.Empty:
                    pass
            self.q_out.put(fused)

    def submit(self, frame, nir_frame):
        if not self.q_in.full():
            try:
                self.q_in.put_nowait((frame, nir_frame))
            except queue.Full:
                pass

    def get_detections(self):
        if not self.q_out.empty():
            try:
                return self.q_out.get_nowait()
            except queue.Empty:
                pass
        return None

    def stop(self):
        self.running = False
        try:
            self.q_in.put(None, timeout=0.1)
        except queue.Full:
            pass
        self.thread.join()

def main():
    parser = argparse.ArgumentParser(description="Anti-Drone Detection System v3")
    parser.add_argument("--source", type=str, default=None, help="RGB video or camera")
    parser.add_argument(
        "--nir-source", type=str, default=None, help="Optional NIR video or camera"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["day", "night", "thermal"],
        help="RGB preprocessing mode",
    )
    parser.add_argument(
        "--nir-mode",
        type=str,
        default="night",
        choices=["day", "night", "thermal"],
        help="NIR preprocessing mode",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "balanced", "pi5"],
        help="Runtime performance profile",
    )
    parser.add_argument("--show-nir", action="store_true", help="Show NIR debug window")
    parser.add_argument(
        "--threat",
        action="store_true",
        help="Show threat scoring based on bounding-box growth",
    )
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print per-stage timing breakdown every second",
    )
    parser.add_argument(
        "--no-real-time",
        action="store_true",
        help="Process frames synchronously (offline mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output video (e.g. output.mp4), forces offline mode",
    )
    args = parser.parse_args()

    if args.output is not None:
        args.no_real_time = True


    if args.source is not None:
        config.VIDEO_SOURCE = parse_source(args.source)
    if args.mode:
        config.MODE = args.mode

    show_nir = args.show_nir
    show_threat = args.threat
    use_nir = args.nir_source is not None
    runtime_settings = create_runtime_settings(args.profile, use_nir)
    apply_runtime_settings(runtime_settings)

    detection_mode = "YOLO"
    if use_nir:
        detection_mode += " + NIR FUSION"

    print("=" * 55)
    print("  ANTI-DRONE DETECTION SYSTEM v3")
    print(f"  {detection_mode}")
    print("=" * 55)
    print(f"  RGB:      {config.VIDEO_SOURCE}")
    if use_nir:
        print(f"  NIR:      {args.nir_source}")
    print(f"  Mode:     {config.MODE}")
    print(f"  Profile:  {args.profile}")
    print(f"  Threat:   {'ON' if show_threat else 'OFF'}")
    print(f"  FOV:      {config.FOV_X} x {config.FOV_Y}")
    print("=" * 55)

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Could not open RGB source.")
        sys.exit(1)

    nir_cap = None
    if use_nir:
        nir_cap = cv2.VideoCapture(parse_source(args.nir_source))
        if not nir_cap.isOpened():
            print("[ERROR] Could not open NIR source.")
            sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video:    {orig_w}x{orig_h} @ {fps_video:.1f} FPS")
    if total_frames > 0 and fps_video > 0:
        print(f"  Length:   {total_frames} frames ({total_frames / fps_video:.1f}s)")
    if args.output:
        print(f"  Output:   {args.output} (Offline processing forced)")
    print()

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_video if fps_video > 0 else 30.0, (orig_w, orig_h))

    model = get_yolo_model()
    tracker_obj = CentroidTracker()
    
    # Initialize Telemetry Client
    telemetry = TelemetryClient(getattr(config, "GROUND_STATION_URL", "http://localhost:8000/api/telemetry"), getattr(config, "NODE_ID", "edge-rpi5-alpha"))
    
    worker = None
    if not args.no_real_time:
        worker = DetectionWorker(model, use_nir)

    frame_counter = 0
    fps_frames = 0
    fps_start = time.time()
    current_fps = 0.0
    timing_stats = init_timing_stats()

    print("[START] Detection running...")
    print("  Keys: q=quit k=sky m=mask n=mode i=nir-window")
    print("  +/-=YOLO confidence")
    print()

    while cap.isOpened():
        read_started = time.perf_counter()
        ret, frame_orig = cap.read()
        if not ret:
            if args.output:
                print(f"\n[INFO] End of video reached. Processed {frame_counter} frames.")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        nir_orig = None
        if nir_cap is not None:
            nir_ret, nir_orig = nir_cap.read()
            if not nir_ret:
                nir_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                nir_ret, nir_orig = nir_cap.read()
                if not nir_ret:
                    nir_orig = None
        add_timing(timing_stats, "read", read_started)

        frame_counter += 1
        fps_frames += 1
        timing_stats["frames"] += 1

        preprocess_started = time.perf_counter()
        frame = preprocess_frame(frame_orig, config.MODE)
        add_timing(timing_stats, "preprocess_rgb", preprocess_started)

        nir_frame = None
        if nir_orig is not None:
            preprocess_nir_started = time.perf_counter()
            nir_frame = preprocess_frame(nir_orig, args.nir_mode)
            if nir_frame.shape[:2] != frame.shape[:2]:
                nir_frame = cv2.resize(nir_frame, (frame.shape[1], frame.shape[0]))
            if runtime_settings["nir_resize_scale"] < 1.0:
                nir_frame = cv2.resize(
                    nir_frame,
                    (0, 0),
                    fx=runtime_settings["nir_resize_scale"],
                    fy=runtime_settings["nir_resize_scale"],
                )
                nir_frame = cv2.resize(nir_frame, (frame.shape[1], frame.shape[0]))
            add_timing(timing_stats, "preprocess_nir", preprocess_nir_started)

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Detection
        if frame_counter % config.PROCESS_EVERY_N_FRAMES == 0:
            yolo_started = time.perf_counter()
            if args.no_real_time:
                # Sync YOLO execution
                yolo_detections = detect_with_yolo(frame, model)
                rgb_candidates = [
                    prepare_sensor_detection(det, "RGB", det.get("confidence", 0.0))
                    for det in yolo_detections
                ]
                nir_candidates = []
                fused = fuse_detections(rgb_candidates, nir_candidates) if use_nir else rgb_candidates
                detections = suppress_duplicate_detections(fused)
                add_timing(timing_stats, "yolo", yolo_started)
            else:
                worker.submit(frame.copy(), nir_frame.copy() if nir_frame is not None else None)
                add_timing(timing_stats, "yolo", yolo_started)

        if not args.no_real_time:
            fusion_started = time.perf_counter()
            new_detections = worker.get_detections()
            detections = new_detections if new_detections is not None else []
            add_timing(timing_stats, "fusion", fusion_started)
        else:
            if frame_counter % config.PROCESS_EVERY_N_FRAMES != 0:
                detections = []

        tracker_started = time.perf_counter()
        tracked_objects = tracker_obj.update(detections)
        add_timing(timing_stats, "tracker", tracker_started)

        # Transmit Telemetry
        if not args.no_real_time or args.output:
            out_tracks = []
            for obj in tracked_objects:
                bx, by, bw, bh = obj.bbox
                if bw > 0 and bh > 0:
                    tcx = bx + bw / 2
                    tcy = by + bh / 2
                    az = ((tcx - center_x) / width) * config.FOV_X
                    el = ((center_y - tcy) / height) * config.FOV_Y
                    assessment = obj.get_threat_assessment()
                    out_tracks.append({
                        "id": getattr(obj, "id", 0),
                        "x": tcx, "y": tcy,
                        "w": bw, "h": bh,
                        "az": az, "el": el,
                        "confidence": float(getattr(obj, "confidence", 0.0)),
                        "threat_score": assessment["score"],
                        "threat_state": assessment["state"],
                        "sensor": getattr(obj, "sensor", "YOLO")
                    })
            if out_tracks:
                telemetry.send_telemetry(out_tracks)

        display_mode = "YOLO"
        if use_nir:
            display_mode += "+NIR"

        hud_started = time.perf_counter()
        frame = draw_hud(
            frame,
            tracked_objects,
            current_fps,
            (center_x, center_y),
            display_mode,
            show_threat=show_threat,
        )
        add_timing(timing_stats, "hud", hud_started)



        display_frame = frame
        if writer is not None:
            writer.write(display_frame)

        display_started = time.perf_counter()
        if width > 1920:
            scale = 1920 / width
            display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        cv2.imshow("Anti-Drone Radar", display_frame)

        if show_nir and nir_frame is not None:
            nir_display = nir_frame.copy()
            cv2.imshow("NIR Debug", nir_display)

        # Artificial framerate limiter to prevent run-away playback on video files
        if fps_video > 0 and not args.output:
            elapsed_loop = time.perf_counter() - read_started
            delay = (1.0 / fps_video) - elapsed_loop
            if delay > 0:
                time.sleep(delay)

        add_timing(timing_stats, "display", display_started)

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_frames / elapsed
            if args.output:
                print(f"[OFFLINE] Processing frame {frame_counter} / {total_frames} ({current_fps:.1f} FPS) ...")
            if args.show_profile:
                print_timing_report(timing_stats, elapsed)
            timing_stats = init_timing_stats()
            fps_frames = 0
            fps_start = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("i"):
            show_nir = not show_nir
            if not show_nir:
                cv2.destroyWindow("NIR Debug")
        elif key == ord("n"):
            modes = ["day", "night", "thermal"]
            idx = modes.index(config.MODE)
            config.MODE = modes[(idx + 1) % len(modes)]
            print(f"[INFO] Mode: {config.MODE}")
        elif key == ord("+") or key == ord("="):
            config.YOLO_CONFIDENCE = min(0.9, config.YOLO_CONFIDENCE + 0.05)
            print(f"[INFO] YOLO confidence: {config.YOLO_CONFIDENCE:.2f}")
        elif key == ord("-"):
            config.YOLO_CONFIDENCE = max(0.05, config.YOLO_CONFIDENCE - 0.05)
            print(f"[INFO] YOLO confidence: {config.YOLO_CONFIDENCE:.2f}")

    print("\n[STOP] Detection stopped.")
    telemetry.stop()
    if worker is not None:
        worker.stop()
    if writer is not None:
        writer.release()
    cap.release()
    if nir_cap is not None:
        nir_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
