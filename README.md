# Counter-UAS AI System

A distributed drone detection and tracking pipeline. A lightweight edge node — sized to target a Raspberry Pi 5 — handles low-latency field inference, while a heavier ground station PC performs thorough analysis in parallel. Both nodes stream their tracks to a central telemetry server, which relays them in real time to any connected dashboard.

## Demonstration on publicly accesible drone videos (Edge model designed to target Raspberry Pi 5): 
https://github.com/user-attachments/assets/e6949b60-a529-4d16-bb8c-a7056a93df6b

<img width="1280" height="720" alt="thumbnail_1_0 0s (1)" src="https://github.com/user-attachments/assets/5c2f93f9-73d7-47d0-b696-a6601aa10b6c" />

> [!NOTE]
> The edge node was developed and tested on a PC, and is **designed to target** a Raspberry Pi 5 (compact ONNX model, frame-skipping, motion-guided cropping). It has **not yet been benchmarked on Pi 5 hardware** — the demonstration above was recorded on a PC, and on-device latency/throughput are not claimed.

---

## 🏗️ Architecture

```
┌──────────────────────────────┐         ┌────────────────────────────────────────┐
│  Edge Node  (Raspberry Pi 5) │  HTTP   │  Ground Station  (PC)                  │
│                              │  POST   │                                        │
│  drone_detector.py           │ ──────► │  server.py            dronebig.py      │
│  • ONNX model  (9.8 MB)      │         │  • Aggregates tracks  • PyTorch model   │
│  • Motion-smart SAHI crops   │ ──────► │    from all senders     (53 MB)         │
│  • Kalman + centroid tracker │         │  • Holds global state • Full SAHI scan  │
│                              │         │  • REST + WebSocket   • Heavier, GPU-  │
│  • Sends tracks → server     │         │    for dashboard        capable node    │
└──────────────────────────────┘         └───────────────┬────────────────────────┘
                                                         │  WebSocket  /ws/radar
                                                         ▼
                                              ┌─────────────────────┐
                                              │   UI Dashboard      │
                                              │   (separate repo)   │
                                              └─────────────────────┘
```

**How it works:**

1. **Edge node** — runs a compact ONNX model with frame-skipping and motion-guided SAHI cropping to stay fast without a GPU, sized to target a Raspberry Pi 5 (tested on PC). Detected tracks are POSTed to the ground station server over the local network.
2. **Ground station AI** (`dronebig.py`) — a heavier PyTorch model with full SAHI grid scans runs on a PC for maximum detection sensitivity. It also POSTs its tracks to the same server.
3. **Telemetry server** (`server.py`) — a FastAPI broker that collects the latest tracks from every sender, **fuses them into a single global-track list with stable IDs** (`common/merge.py`), and broadcasts live updates to the dashboard over WebSocket. The server doesn't care whether a track came from the Pi or the PC.
4. **Dashboard** — any WebSocket-capable client connects to `/ws/radar` and receives both the raw per-sender tracks and the merged, unified radar feed.

---

## ✅ Status & Limitations

This is an active work-in-progress portfolio project. To keep the docs honest, here is what works today versus what is still stubbed or unverified.

**Working**
- **Edge detector** — ONNX YOLO inference, motion-guided SAHI cropping, Kalman + centroid tracking, a bounding-box-growth threat estimate, the HUD overlay, and non-blocking telemetry.
- **Ground-station tracker** (`dronebig.py`) — PyTorch YOLO with SAHI slicing and ByteTrack, forwarding tracks to the server.
- **Telemetry server** — validated ingest, per-sender latest-state, **cross-sender track merging into a unified global-track list with stable IDs**, and REST + WebSocket fan-out to dashboards.

**Work in progress / not yet implemented**
- **Calibrated multi-view fusion.** Cross-sender merging works, but it associates purely in each sender's reported azimuth/elevation and assumes the senders share an angular frame. There is no extrinsic calibration or world-coordinate projection, so merged tracks live in angular space only (no fused pixel/world position) — see the caveat in `common/merge.py`.
- **NIR sensor fusion.** The fusion code path exists (`fuse_detections`), but no NIR detector is wired in — the NIR candidate list is always empty, so an NIR source currently has no effect on detection.
- **Ground-station threat scoring.** `dronebig.py` emits a placeholder threat value; only the edge node computes a real bounding-box-growth threat estimate.
- **Not benchmarked on Raspberry Pi 5.** Developed and tested on a PC; on-device latency and throughput are unverified.

---

## 📁 Project Structure

```
├── common/                     # Shared code imported by both nodes
│   ├── schemas.py              # Pydantic telemetry contract (single source of truth)
│   ├── telemetry.py            # Non-blocking HTTP track sender (used by both nodes)
│   └── merge.py                # Cross-sender track merger (global IDs)
│
├── edge-rpi5/                  # Lightweight edge detection node
│   ├── drone_detector.py       # Main pipeline entrypoint (CLI)
│   ├── tracker.py              # Centroid + Kalman tracker
│   ├── config.py               # All tunable parameters
│   └── best.onnx               # Compact ONNX model weights
│
└── ground-station/             # Heavy AI node + telemetry server
    ├── server.py               # FastAPI broker (REST + WebSocket + merge)
    ├── dronebig.py             # High-accuracy SAHI tracker (CLI)
    └── best.pt                 # Full PyTorch model weights
```

---

## 📋 Prerequisites

**Python 3.9 or newer** is required. Install all dependencies:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> The edge node requires `edge-rpi5/best.onnx`.
> The ground station tracker requires `ground-station/best.pt`.
> Both accept any YOLO-compatible weights. Update `YOLO_MODEL` in `config.py` to point to a custom path.

---

## 🛰️ Edge Node

The edge node is designed to target a Raspberry Pi 5 but runs on any machine; it was developed and tested on a PC and has not been benchmarked on Pi hardware. It performs continuous detection and streams track data to the ground station server.

```bash
cd edge-rpi5
```

### Basic daytime detection

```bash
python drone_detector.py --source "your_video.mp4" --mode day
```

### Export annotated video (offline mode)

```bash
python drone_detector.py \
    --source "your_video.mp4" \
    --mode day \
    --threat \
    --no-real-time \
    --output "output.mp4"
```

### Live webcam

```bash
python drone_detector.py --source 0 --mode day
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--source` | *(config value)* | RGB video file path or integer camera index |
| `--mode` | *(config value)* | RGB preprocessing: `day`, `night`, or `thermal` |
| `--profile` | `default` | Performance profile: `default`, `balanced`, or `pi5` |
| `--threat` | off | Show threat assessment overlay (approaching / receding) |
| `--show-profile` | off | Print per-stage timing to console every second |
| `--no-real-time` | off | Synchronous frame processing (offline / export mode) |
| `--output` | *(none)* | Save annotated output video. Implies `--no-real-time`. |

### Interactive keys

| Key | Action |
|---|---|
| `q` | Quit |
| `n` | Cycle RGB mode (day → night → thermal) |
| `+` / `=` | Raise YOLO confidence threshold by 0.05 |
| `-` | Lower YOLO confidence threshold by 0.05 |

---

## 🖥️ Ground Station

### 1. Start the telemetry server

The server must be running before any edge node or tracker tries to send data.

```bash
cd ground-station
python server.py
```

Starts on **`http://0.0.0.0:8000`**. Both the edge node and the ground station tracker point to this server by default.

#### API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/telemetry` | Ingest a track payload from any sender (validated against the shared schema) |
| `GET` | `/api/state` | Pull the full latest *per-sender* state (use on dashboard init) |
| `GET` | `/api/unified` | Pull the latest *merged* global-track list (one entry per physical target) |
| `WebSocket` | `/ws/radar` | Real-time stream — sends `full_state` on connect, then `telemetry_update` per POST. Both messages carry a `unified` field with the merged tracks. |

#### Telemetry payload schema

```json
{
  "sender_id": "edge-rpi5-alpha",
  "timestamp": 1713430000.0,
  "tracks": [
    {
      "id": 1,
      "x": 640.0,  "y": 360.0,
      "w": 48.0,   "h": 32.0,
      "az": 12.5,  "el": -3.1,
      "confidence": 0.87,
      "threat_score": 0.42,
      "threat_state": "APPROACHING"
    }
  ]
}
```

#### Unified track shape

The server fuses every sender's tracks into a global list (`/api/unified`, and the `unified` field on WebSocket messages). Each merged track is reported in the angular frame only:

```json
{
  "global_id": 1,
  "az": 10.47,  "el": 2.18,
  "confidence": 0.90,
  "threat_score": 0.40,
  "threat_state": "APPROACHING",
  "sensors": ["BASE-SAHI", "RGB"],
  "num_sensors": 2,
  "sources": [
    { "sender_id": "edge-rpi5-alpha", "id": 1 },
    { "sender_id": "base-station-sahi", "id": 7 }
  ]
}
```

### 2. Run the ground station tracker

`dronebig.py` runs heavier SAHI-sliced inference on a PC and forwards its tracks to the same server alongside the edge node.

```bash
cd ground-station
python dronebig.py --source "your_video.mp4"
```

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--source` | *(required)* | Video file path or integer camera index |
| `--model` | `best.pt` | Path to YOLO `.pt` model weights |
| `--confidence` | `0.5` | Detection confidence threshold |
| `--output` | `output.mp4` | Path to save the annotated output video |
| `--server` | `http://localhost:8000/api/telemetry` | Telemetry endpoint to POST tracks to |

---

## 🔧 Advanced Configuration

All edge node parameters live in `edge-rpi5/config.py`:

| Section | Key Constants |
|---|---|
| **Model** | `YOLO_MODEL`, `YOLO_CONFIDENCE`, `YOLO_INPUT_SIZE` |
| **SAHI** | `SAHI_ENABLED`, `SAHI_SLICE_WIDTH/HEIGHT`, `SAHI_OVERLAP_RATIO`, `SAHI_MOTION_SMART_CROP` |
| **Tracker** | `MAX_DISTANCE`, `MAX_DISAPPEARED`, `MAX_PREDICTED_FRAMES`, `TRACK_HISTORY` |
| **Kalman** | `KALMAN_PROCESS_NOISE`, `KALMAN_MEASUREMENT_NOISE` |
| **Night/IR** | `CLAHE_CLIP_LIMIT`, `CLAHE_GRID_SIZE`, `THERMAL_COLORMAP` |
| **HUD** | `COLOR_CROSSHAIR`, `HUD_FONT_SCALE`, `HUD_THICKNESS` |
| **Networking** | `GROUND_STATION_URL`, `NODE_ID` |

---

## 🤝 Contributing

Pull requests, optimizations, and bug reports are welcome. When reporting a detection artifact or false positive, please include the video sequence and CLI flags that triggered it.
