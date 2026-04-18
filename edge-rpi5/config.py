"""
Configuration for the anti-drone detection system.
Optimized defaults for Raspberry Pi 5.
"""

# ============================================================
# Input
# ============================================================
VIDEO_SOURCE = "your_video.mp4"  # override at runtime with --source

MODE = "day"  # "day" | "night" | "thermal"

# ============================================================
# Camera
# ============================================================
FOV_X = 80.0
FOV_Y = 50.0



# ============================================================
# YOLO
# ============================================================
YOLO_MODEL = "./best.onnx"
YOLO_CONFIDENCE = 0.3
YOLO_INPUT_SIZE = 1280
ROI_MARGIN = 40
DRONE_CLASSES = [4, 14, 33]
ACCEPT_ALL_CLASSES = True
YOLO_EVERY_N_FRAMES = 5

# ============================================================
# YOLO SAHI (Image Slicing)
# ============================================================
SAHI_ENABLED = True
SAHI_SLICE_WIDTH = 640
SAHI_SLICE_HEIGHT = 640
SAHI_OVERLAP_RATIO = 0.2
SAHI_MOTION_SMART_CROP = True
SAHI_FULL_SCAN_INTERVAL = 30
SAHI_MAX_CROPS = 4
SAHI_MIN_MOTION_AREA = 10
SAHI_MAX_MOTION_AREA = 10000

# ============================================================
# Tracker
# ============================================================
MAX_DISAPPEARED = 15
MAX_DISTANCE = 150
TRACK_HISTORY = 60
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 5e-1
MAX_PREDICTED_FRAMES = 30

# ============================================================
# Multi-sensor fusion
# ============================================================
FUSION_MATCH_DISTANCE = 90
FUSION_RGB_WEIGHT = 0.65
FUSION_NIR_WEIGHT = 0.35
FUSION_BONUS = 0.25
FUSION_CONFIRM_THRESHOLD = 0.55
NIR_DETECT_THRESHOLD = 0.35

# ============================================================
# Night / IR
# ============================================================
CLAHE_CLIP_LIMIT = 3.0
CLAHE_GRID_SIZE = (8, 8)
THERMAL_COLORMAP = 11  # cv2.COLORMAP_INFERNO
THERMAL_THRESHOLD = 200

# ============================================================
# HUD
# ============================================================
SHOW_MINIMAP = False
HUD_FONT_SCALE = 0.5
HUD_THICKNESS = 2
COLOR_CONFIRMED = (0, 255, 0)
COLOR_MOTION = (0, 255, 0)
COLOR_PREDICTED = (0, 255, 0)
COLOR_CROSSHAIR = (0, 255, 0)
COLOR_TRACK_LINE = (0, 255, 0)
COLOR_INFO_BG = (220, 220, 220)

# ============================================================
# Performance
# ============================================================
PROCESS_EVERY_N_FRAMES = 5
MAX_ROIS_PER_FRAME = 3

# ============================================================
# Networking
# ============================================================
GROUND_STATION_URL = "http://localhost:8000/api/telemetry"
NODE_ID = "edge-rpi5-alpha"

