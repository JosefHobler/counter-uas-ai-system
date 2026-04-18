# Counter-UAS AI System

A distributed drone detection and tracking pipeline. A lightweight edge node runs on a Raspberry Pi 5 for low-latency field inference, while a heavier ground station PC performs thorough analysis in parallel. Both nodes stream their tracks to a central telemetry server, which broadcasts a unified real-time feed to any connected dashboard.

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
│  • NIR sensor fusion         │         │  • REST + WebSocket   • Heavier, GPU-  │
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

1. **Edge node** — the Pi 5 runs a compact ONNX model with frame-skipping and motion-guided SAHI cropping to stay fast without a GPU. It optionally fuses a second NIR camera feed. Detected tracks are POSTed to the ground station server over the local network.
2. **Ground station AI** (`dronebig.py`) — a heavier PyTorch model with full SAHI grid scans runs on a PC for maximum detection sensitivity. It also POSTs its tracks to the same server.
3. **Telemetry server** (`server.py`) — a FastAPI broker that merges tracks from every sender into a single global state and broadcasts live updates to the dashboard over WebSocket. The server doesn't care whether a track came from the Pi or the PC.
4. **Dashboard** — any WebSocket-capable client connects to `/ws/radar` and receives a unified, real-time radar feed.

---

## 📁 Project Structure

```
├── edge-rpi5/                  # Lightweight edge detection node
│   ├── drone_detector.py       # Main pipeline entrypoint (CLI)
│   ├── tracker.py              # Centroid + Kalman tracker
│   ├── telemetry_client.py     # Non-blocking HTTP track sender
│   ├── config.py               # All tunable parameters
│   └── best.onnx               # Compact ONNX model weights
│
└── ground-station/             # Heavy AI node + telemetry server
    ├── server.py               # FastAPI broker (REST + WebSocket)
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

The edge node runs on a Raspberry Pi 5 (or any machine). It performs continuous detection and streams track data to the ground station server.

```bash
cd edge-rpi5
```

### Basic daytime detection

```bash
python drone_detector.py --source "your_video.mp4" --mode day
```

### RGB + NIR fusion with threat scoring (real-time)

```bash
python drone_detector.py \
    --source "rgb_cam.mp4" \
    --nir-source "nir_cam.mp4" \
    --mode day \
    --nir-mode night \
    --threat
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
| `--nir-source` | *(none)* | NIR video or camera. Enables sensor fusion when provided. |
| `--mode` | *(config value)* | RGB preprocessing: `day`, `night`, or `thermal` |
| `--nir-mode` | `night` | NIR preprocessing: `day`, `night`, or `thermal` |
| `--profile` | `default` | Performance profile: `default`, `balanced`, or `pi5` |
| `--threat` | off | Show threat assessment overlay (approaching / receding) |
| `--show-nir` | off | Open a separate debug window for the NIR feed |
| `--show-profile` | off | Print per-stage timing to console every second |
| `--no-real-time` | off | Synchronous frame processing (offline / export mode) |
| `--output` | *(none)* | Save annotated output video. Implies `--no-real-time`. |

### Interactive keys

| Key | Action |
|---|---|
| `q` | Quit |
| `n` | Cycle RGB mode (day → night → thermal) |
| `i` | Toggle NIR debug window |
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
| `POST` | `/api/telemetry` | Ingest a track payload from any sender |
| `GET` | `/api/state` | Pull the full latest state for all senders (use on dashboard init) |
| `WebSocket` | `/ws/radar` | Real-time stream — sends `full_state` on connect, then `telemetry_update` per POST |

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
      "threat_state": "APPROACHING",
      "sensor": "RGB+NIR"
    }
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
| **Fusion** | `FUSION_RGB_WEIGHT`, `FUSION_NIR_WEIGHT`, `FUSION_BONUS`, `FUSION_CONFIRM_THRESHOLD` |
| **Night/IR** | `CLAHE_CLIP_LIMIT`, `CLAHE_GRID_SIZE`, `THERMAL_COLORMAP` |
| **HUD** | `COLOR_CROSSHAIR`, `HUD_FONT_SCALE`, `HUD_THICKNESS` |
| **Networking** | `GROUND_STATION_URL`, `NODE_ID` |

---

## 🤝 Contributing

Pull requests, optimizations, and bug reports are welcome. When reporting a detection artifact or false positive, please include the video sequence and CLI flags that triggered it.
