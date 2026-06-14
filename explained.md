# Counter-UAS AI System — Full Explanation

A complete walkthrough of what this repository is, how it's structured, what
every file does, how data flows end-to-end, and the honest state of each
feature. Read top-to-bottom for a tour, or jump to a section.

- [1. What this project is](#1-what-this-project-is)
- [2. The big picture (architecture & data flow)](#2-the-big-picture-architecture--data-flow)
- [3. Repository layout](#3-repository-layout)
- [4. The shared core — `common/`](#4-the-shared-core--common)
- [5. The edge node — `edge-rpi5/`](#5-the-edge-node--edge-rpi5)
- [6. The ground station — `ground-station/`](#6-the-ground-station--ground-station)
- [7. The tests — `tests/` and `conftest.py`](#7-the-tests--tests-and-conftestpy)
- [8. The wire format (end-to-end)](#8-the-wire-format-end-to-end)
- [9. How to run everything](#9-how-to-run-everything)
- [10. Status & limitations (the honest version)](#10-status--limitations-the-honest-version)
- [11. Key algorithms, explained simply](#11-key-algorithms-explained-simply)
- [12. Design decisions & why they matter](#12-design-decisions--why-they-matter)
- [13. Glossary](#13-glossary)

---

## 1. What this project is

A **distributed drone-detection and tracking pipeline** ("Counter-UAS" =
Counter–Unmanned Aerial System). It has three moving parts:

1. A **lightweight edge node** meant to run in the field (sized to target a
   Raspberry Pi 5) that does fast, low-latency detection on a camera feed.
2. A **heavier ground-station node** on a PC that runs a bigger, more accurate
   model on the same kind of footage.
3. A **telemetry server** that both nodes report to. It collects everyone's
   tracks, fuses them into one global picture, and streams that picture to any
   connected dashboard in real time.

The dashboard UI itself lives in a **separate repository** — this repo is the
detection + telemetry backend, plus the models.

It is a **portfolio / work-in-progress project**, not a deployed product.
Section 10 is explicit about what genuinely works versus what is stubbed.

---

## 2. The big picture (architecture & data flow)

There are two **senders** (detectors) and one **server**. Both senders POST to
the same server; the server never POSTs to anyone — it broadcasts over a
WebSocket.

```
        SENDERS (run the AI)                          SERVER                 CONSUMER
┌──────────────────────────────┐
│  EDGE NODE (target: Pi 5)    │
│  edge-rpi5/drone_detector.py │ ─┐
│   • compact ONNX YOLO        │  │
│   • motion-guided SAHI crops │  │
│   • Kalman + centroid tracker│  │  HTTP POST
│   • bbox-growth threat score │  │  /api/telemetry
│   • HUD overlay              │  │
└──────────────────────────────┘  │     ┌────────────────────────────────────┐
                                   ├───► │  ground-station/server.py          │
┌──────────────────────────────┐  │     │   • validates every payload        │
│  GROUND TRACKER (PC)         │  │     │   • stores latest state per sender │      ┌──────────────┐
│  ground-station/dronebig.py  │ ─┘     │   • MERGES all senders → global    │ ───► │ Dashboard UI │
│   • heavy PyTorch YOLO + SAHI│        │   • REST + WebSocket fan-out       │  WS  │ (separate    │
│   • ByteTrack for stable IDs │        └────────────────────────────────────┘ /ws/ │  repo)       │
└──────────────────────────────┘                                               radar └──────────────┘
```

`dronebig.py` lives in the `ground-station/` folder only because it's meant to
run on the same PC as the server — but it is a **client of the server**, exactly
like the edge node. It runs the AI and POSTs its tracks to `server.py`. There is
no `dronebig.py → dronebig.py` link.

**The flow, in one paragraph:** Each detection node (the edge detector and the
ground tracker) processes video frames, produces a list of tracks (one per
object it currently sees), and POSTs them to the server's `/api/telemetry`
endpoint. The server validates each payload against a shared schema, remembers
the latest tracks from each sender, re-runs a **merge** step that fuses tracks
from different senders that point at the same physical target, and then
broadcasts both the raw per-sender update and the merged "unified" list to every
connected dashboard over a WebSocket. A dashboard that connects mid-stream first
gets a full snapshot, then live updates.

Note the key asymmetry: **`dronebig.py` is a sender, not the server.**
`server.py` is the only server. Both `drone_detector.py` and `dronebig.py` are
clients that push data to it.

---

## 3. Repository layout

```
drones/
├── common/                     # Shared code imported by BOTH nodes
│   ├── __init__.py             # marks it a package; explains the "why share"
│   ├── schemas.py              # Pydantic telemetry contract (single source of truth)
│   ├── telemetry.py            # Non-blocking HTTP track sender (used by both nodes)
│   └── merge.py                # Cross-sender track merger (assigns global IDs)
│
├── edge-rpi5/                  # The lightweight edge detection node
│   ├── drone_detector.py       # Main pipeline + CLI (the big one, ~1260 lines)
│   ├── tracker.py              # Centroid + Kalman per-frame tracker
│   ├── config.py               # All tunable parameters in one place
│   └── best.onnx               # Compact ONNX model weights (~9.8 MB)
│
├── ground-station/             # The heavy AI node + the telemetry server
│   ├── server.py               # FastAPI broker (REST + WebSocket + merge)
│   ├── dronebig.py             # High-accuracy SAHI + ByteTrack tracker (a sender)
│   └── best.pt                 # Full PyTorch model weights (~53 MB)
│
├── tests/                      # Unit tests (pytest) — pure logic, no weights/camera
│   ├── test_schemas.py         # contract validation incl. the threat_score=50 bug
│   ├── test_merge.py           # cross-sender merging behaviour
│   ├── test_tracker.py         # tracker matching + threat heuristic
│   └── test_detector_math.py   # bbox_iou / clamp01 / detection_priority
│
├── conftest.py                 # Pytest path setup so tests can import the modules
├── requirements.txt            # Pinned dependencies (runtime + pytest)
├── README.md                   # User-facing docs
├── explained.md                # ← this file
└── *.mp4                        # Sample / demo videos
```

The three top-level code folders map exactly to the three architecture boxes:
`common/` is the shared contract, `edge-rpi5/` is the field box, `ground-station/`
is the PC box.

---

## 4. The shared core — `common/`

This is a small "monorepo-style" package that both deployable nodes import. Its
whole reason to exist: the telemetry **contract** and the telemetry **client**
should have one definition, not be copy-pasted into each node where they could
drift apart.

### `common/__init__.py`
Just marks the folder as an importable Python package and documents the intent
("single source of truth instead of copy-paste"). No logic.

### `common/schemas.py` — the data contract
Defines the **wire format** using Pydantic models. This is the single most
important file for understanding what data flows around the system.

- **`ThreatState`** — a closed enum: `UNKNOWN`, `STABLE`, `APPROACHING`,
  `RECEDING`. A track can only ever be one of these four.
- **`Track`** — one tracked target from a single sender:
  - `id` — the sender's own track ID (stable per sender, not globally unique).
  - `x, y` — image-space centroid in pixels.
  - `w, h` — bounding-box size in pixels (must be ≥ 0).
  - `az, el` — azimuth/elevation offset from that camera's boresight, in degrees.
  - `confidence` — detection confidence, a fraction in **[0, 1]**.
  - `threat_score` — approach strength, a fraction in **[0, 1]** (default 0).
  - `threat_state` — one of the enum values above (default `UNKNOWN`).
  - `sensor` — a free-form tag like `"RGB"` or `"BASE-SAHI"`.
  - `model_config = extra="forbid"` → any unknown/typo'd field is **rejected**,
    not silently dropped. `use_enum_values=True` → when dumped to JSON the enum
    becomes its plain string.
- **`TelemetryPayload`** — one frame of telemetry POSTed by a sender:
  `sender_id` (non-empty string), `tracks` (a list of `Track`), and an optional
  `timestamp`.

**Why this matters:** the bounds (`ge=0.0, le=1.0`) mean a malformed value like
`threat_score=50` is rejected *at the door* instead of poisoning the dashboard.
Before this contract existed, the ground tracker shipped exactly that bad value.
Both the producers (the two detectors) and the consumer (the server) validate
against these same models, so the two sides physically cannot disagree about the
format.

### `common/telemetry.py` — the non-blocking sender
A single `TelemetryClient` class that both nodes use to ship tracks to the
server **without ever stalling their frame loop**.

How it works:
- On construction it starts a background daemon thread and a bounded queue
  (default size 10).
- `send(tracks)` wraps the tracks in a payload (`sender_id`, `timestamp`,
  `tracks`) and drops it on the queue. **If the queue is full it discards the
  oldest frame** and enqueues the new one — so a slow or unreachable server can
  never build a backlog or slow down detection.
- The background thread drains the queue and does a `requests.post` with a short
  timeout (0.5 s). Network errors are swallowed on purpose — in degraded network
  conditions you'd rather drop a frame than block.
- `stop()` cleanly shuts the worker down.

This file is the result of "Task 4" in the project plan: the edge node and the
ground tracker each used to carry their own near-identical client
(`TelemetryClient` and `MiniTelemetryClient`). They were merged into this one
shared implementation.

### `common/merge.py` — cross-sender track fusion
This is the most algorithmically interesting file. It takes the per-sender
tracks the server is holding and fuses them into **one global list**, so the
dashboard sees one blip per real object instead of one-per-camera.

The public surface is the **`TrackMerger`** class. It is **stateful** across
calls — it remembers global IDs so a target keeps the same `global_id` from
frame to frame, and even across a brief disappearance.

`merge(sender_states, now)` runs this pipeline:

1. **Collect fresh observations.** Walk every sender's latest tracks. Skip a
   sender entirely if its data is older than the TTL (default 3 s) — staleness
   is judged by the *server's* clock (`received_at`), so it doesn't depend on
   the senders' clocks being in sync.
2. **Cluster** observations that are the same physical target. This is a greedy
   union-find over candidate cross-sender pairs whose angular distance is within
   the gate (default 6°), closest pairs first. There's a hard rule: **a cluster
   can never contain two tracks from the same sender** — if one camera reports
   two nearby tracks, that's two targets, not one.
3. **Representative** position per cluster: a confidence-weighted mean of the
   members' `(az, el)` (falls back to a plain mean if all confidences are ~0).
4. **Associate** each cluster to an existing global track within the gate
   (greedy, closest first), so IDs persist. Unmatched clusters get a fresh
   global ID.
5. **Keep-alive:** global tracks that weren't matched this round but are still
   within TTL are retained, so a target that blinks out for a frame keeps its ID
   when it returns.
6. **Format** each merged track: global ID, representative az/el, the *max*
   confidence and *max* threat_score across the cluster, the *worst* (most
   severe) threat_state, the set of contributing sensors, how many sensors saw
   it, and the list of source `(sender_id, id)` pairs.

**The big caveat (documented in the file):** association happens purely in
angular `(az, el)` space and assumes the senders roughly share an angular frame.
There is **no extrinsic calibration and no world-coordinate projection**. Pixel
coordinates are deliberately dropped from the merged output, because `x`/`y`
from two different cameras aren't comparable. So this is a best-effort *angular
gate*, not a calibrated multi-view fusion. This honesty is deliberate — it's the
difference between "I finished a feature" and "I oversold one."

---

## 5. The edge node — `edge-rpi5/`

The field box: fast, GPU-free, designed to target a Raspberry Pi 5 (but it runs
on any machine and was developed/tested on a PC).

### `edge-rpi5/config.py` — all the knobs
A flat module of constants, grouped by area: input/source, camera field-of-view
(`FOV_X=80°`, `FOV_Y=50°`), YOLO settings (model path, confidence, input size,
which classes count as drones), SAHI slicing settings, tracker settings (Kalman
noise, max match distance, history length), multi-sensor fusion weights, night/IR
preprocessing, HUD colours, performance profile knobs, and networking
(`GROUND_STATION_URL`, `NODE_ID`). Everything tunable lives here so the pipeline
code stays free of magic numbers.

### `edge-rpi5/tracker.py` — per-frame tracking
Turns frame-by-frame detections into **stable tracks with persistent IDs**.

- **`TrackedObject`** — one target being followed. It holds:
  - a position history and a bbox history (bounded deques),
  - a **Kalman filter** (state `[x, y, vx, vy]`, measurement `[x, y]`) that
    predicts where the object will be next frame and smooths noisy detections,
  - bookkeeping: `disappeared` counter, whether it was AI-confirmed, label,
    confidence, current bbox, and whether the current position is a prediction
    (a coast) rather than a real measurement.
  - **`get_direction()`** — estimates heading (N/NE/E/…) and speed from recent
    positions.
  - **`get_threat_assessment()`** — the threat heuristic. It uses **bounding-box
    area growth as a cheap depth proxy**: if the box is getting bigger the object
    is getting closer. It compares the average area of the older half of recent
    boxes to the newer half, takes the **log-ratio** (so doubling and halving are
    symmetric), and maps that to a `score` in [0, 1] and a state:
    `APPROACHING` (growing), `RECEDING` (shrinking), `STABLE` (little change), or
    `UNKNOWN` (not enough history). This is the *only* real threat estimate in
    the whole system.
- **`CentroidTracker`** — the matcher. Each frame:
  1. If it has no objects yet, every detection becomes a new track.
  2. Otherwise it asks each existing track's Kalman filter to **predict** its
     new position, builds a distance matrix between predictions and the new
     detections, and greedily matches closest pairs (rejecting matches beyond
     `MAX_DISTANCE`).
  3. Matched tracks are corrected with the real measurement; unmatched tracks
     **coast** on their prediction and are deleted if they coast too long;
     unmatched detections become new tracks.

  This is what carries a target through a few frames of occlusion without
  dropping or swapping its ID.

### `edge-rpi5/drone_detector.py` — the main pipeline (the big file)
The edge entrypoint and CLI. Roughly in order:

- **Model loading** (`get_yolo_model`, `resolve_yolo_input_size`) — loads the
  ONNX YOLO once, and inspects the ONNX file to discover its required input size.
- **Preprocessing** (`preprocess_frame`) — `day` mode is raw; `night` applies
  CLAHE contrast enhancement; `thermal` applies a colormap.
- **SAHI slicing** (`get_sahi_slices`) — "Slicing-Aided Hyper Inference": instead
  of shrinking a big frame to the model's small input (which loses tiny, distant
  drones), it cuts the frame into overlapping tiles and runs detection on each.
  This is how small targets stay detectable.
- **Motion-guided cropping** (`SmartCropper`) — running YOLO on *every* tile is
  expensive. This uses a KNN background subtractor to find where motion is, and
  only slices around moving regions — with a periodic full-grid sweep and a
  fallback to full-grid when motion is everywhere (camera panning). This is the
  main CPU-saver that makes the Pi target plausible.
- **Detection** (`detect_with_yolo`) — runs YOLO on the chosen tiles (or the
  whole frame), maps each tile-local box back to full-frame coordinates, then
  **NMS** (`nms_detections` + `bbox_iou`) removes duplicate boxes.
- **Sensor fusion** (`fuse_detections`, `score_nir_detection`,
  `prepare_sensor_detection`) — a "soft-OR" scheme to combine an RGB detector
  with an optional NIR (near-infrared) detector, giving a confidence bonus when
  both agree. **Important:** the NIR branch is plumbing only — no NIR detector is
  actually wired in, so the NIR candidate list is always empty and fusion is a
  no-op in practice (see Section 10).
- **Deduplication** (`suppress_duplicate_detections`, `detection_priority`) —
  collapses near-duplicate detections, preferring confirmed > better-sensor >
  higher-confidence.
- **Threading** (`DetectionWorker`) — in real-time mode, YOLO runs on a
  background thread with single-slot in/out queues, so heavy inference doesn't
  block frame display. Offline/export mode runs synchronously instead.
- **HUD rendering** (`draw_hud`, `put_text_lines`, `choose_primary_target`,
  `format_threat_line`) — draws the crosshair, per-target brackets, a
  picture-in-picture "seeker" zoom on the primary target, and tracker/threat
  readouts. This is the cinematic "missile-seeker" overlay you see in the demo.
- **`main()`** — wires it all together: parse CLI args, open the video source(s),
  loop over frames (read → preprocess → detect → track → **send telemetry** →
  draw HUD → display/write), and handle keyboard controls (`q` quit, `n` cycle
  mode, `+/-` confidence). For each tracked object it computes az/el from the
  pixel offset and the camera FOV, attaches the threat assessment, and ships the
  batch via the shared `TelemetryClient`.

CLI flags worth knowing: `--source`, `--mode {day,night,thermal}`,
`--profile {default,balanced,pi5}` (trades accuracy for speed),
`--threat` (show the approach overlay), `--no-real-time` / `--output` (offline
export), `--show-profile` (per-stage timing).

---

## 6. The ground station — `ground-station/`

### `ground-station/server.py` — the telemetry broker
A small **FastAPI** app. It's the hub everything else talks to. It has no model
and does no detection — it only ingests, merges, and broadcasts.

- Global state: `latest_telemetry` (the latest payload per sender), a single
  `TrackMerger`, and `latest_unified` (the most recent merged list).
- **`POST /api/telemetry`** — the ingest endpoint. FastAPI validates the body
  against `TelemetryPayload` automatically (so bad data is rejected with a 422
  before any of our code runs). It stamps `received_at` with the server clock,
  stores the sender's tracks, **re-runs the merge across all senders**, then
  fires off a WebSocket broadcast containing both the raw update and the fresh
  unified list. Returns a small ack with counts.
- **`GET /api/state`** — the full latest per-sender state (used by a dashboard on
  load).
- **`GET /api/unified`** — just the latest merged global-track list.
- **`WebSocket /ws/radar`** — the live feed. On connect a client gets a
  `full_state` snapshot, then a `telemetry_update` message every time any sender
  POSTs. `ConnectionManager` tracks connected clients and handles broadcast +
  disconnect.
- CORS is wide open (`allow_origins=["*"]`) so the separate UI repo can connect.
- Run with `python server.py` → serves on `0.0.0.0:8000`.

### `ground-station/dronebig.py` — the heavy tracker (a sender)
The high-accuracy counterpart to the edge node, meant for a PC. Despite living
in `ground-station/`, it is a **telemetry sender**, just like the edge node — it
pushes to `server.py`.

Pipeline per frame:
1. Load a full **PyTorch** YOLO model via SAHI's `AutoDetectionModel` (uses CUDA
   if available, else CPU).
2. Run **`get_sliced_prediction`** — full SAHI grid slicing (640×640 tiles, 20%
   overlap) for maximum sensitivity to small targets.
3. Convert detections into the `supervision` library's format.
4. Feed them to **ByteTrack** (`sv.ByteTrack`) for stable multi-object IDs across
   frames — a well-regarded, off-the-shelf tracker (contrast with the edge
   node's hand-rolled centroid+Kalman tracker).
5. Annotate the frame, compute az/el from each box's offset and the FOV, and
   POST the tracks via the shared `TelemetryClient` (sender id
   `"base-station-sahi"`).

Note the explicit honesty in the code: this tracker has no depth proxy, so it
emits `threat_score=0.0` / `threat_state="UNKNOWN"` — a neutral, in-range value.
The comment even records that this used to be the buggy `threat_score=50` the
contract now forbids.

---

## 7. The tests — `tests/` and `conftest.py`

A focused **pytest** suite (28 tests) over the pure-logic pieces. None of them
need the model weights, a camera, or a running server, so they run anywhere in
under a second.

- **`conftest.py`** — puts the repo root (for the `common` package) and
  `edge-rpi5/` (for `config`, `tracker`, `drone_detector`) on `sys.path`, so the
  tests can import the modules without installing anything.
- **`test_schemas.py`** — the contract: safe defaults, and rejection of the
  `threat_score=50` bug, out-of-range confidence, negative box dims, unknown
  keys, and bad enum values; plus a round-trip dump check.
- **`test_merge.py`** — merging: two senders on one target collapse to a single
  global track with `num_sensors=2`; targets outside the gate stay separate; two
  tracks from one sender never merge; global IDs stay stable across frames; a
  stale sender is dropped; the merged track takes the worst threat state.
- **`test_tracker.py`** — the tracker: first frame registers all detections; a
  nearby detection keeps the same ID; a far jump spawns a new ID; a vanished
  object eventually deregisters. Plus the threat heuristic returning UNKNOWN /
  STABLE / APPROACHING / RECEDING for the right box histories.
- **`test_detector_math.py`** — `bbox_iou` (identical → 1.0, disjoint → 0.0,
  half-overlap → 1/3), `clamp01` bounds, and `detection_priority` ordering.

Run them with `pytest` from the repo root.

---

## 8. The wire format (end-to-end)

What a sender POSTs to `/api/telemetry` (validated against `TelemetryPayload`):

```json
{
  "sender_id": "edge-rpi5-alpha",
  "timestamp": 1713430000.0,
  "tracks": [
    {
      "id": 1,
      "x": 640.0, "y": 360.0,
      "w": 48.0,  "h": 32.0,
      "az": 12.5, "el": -3.1,
      "confidence": 0.87,
      "threat_score": 0.42,
      "threat_state": "APPROACHING",
      "sensor": "RGB"
    }
  ]
}
```

What the server broadcasts over `/ws/radar` after a POST:

```json
{
  "type": "telemetry_update",
  "sender_id": "edge-rpi5-alpha",
  "tracks": [ ... the raw per-sender tracks ... ],
  "unified": [
    {
      "global_id": 1,
      "az": 12.51, "el": -3.08,
      "confidence": 0.87,
      "threat_score": 0.42,
      "threat_state": "APPROACHING",
      "sensors": ["RGB", "BASE-SAHI"],
      "num_sensors": 2,
      "sources": [
        {"sender_id": "edge-rpi5-alpha", "id": 1},
        {"sender_id": "base-station-sahi", "id": 7}
      ]
    }
  ],
  "timestamp": 1713430000.0
}
```

Notice the merged/unified entries have **no `x`/`y`** — pixels from different
cameras aren't comparable, so the merge lives in angular space only.

---

## 9. How to run everything

Prerequisites: Python 3.9+ and `pip install -r requirements.txt`. The edge node
needs `edge-rpi5/best.onnx`; the ground tracker needs `ground-station/best.pt`
(both are in the repo).

**1. Start the server first** (everything else reports to it):
```bash
cd ground-station
python server.py            # serves on http://0.0.0.0:8000
```

**2. Run the edge detector** (in another terminal):
```bash
cd edge-rpi5
python drone_detector.py --source "../2026-03-22 17-45-14.mp4" --mode day --threat
# offline export instead of a live window:
python drone_detector.py --source "<video>" --no-real-time --output out.mp4
```

**3. Optionally run the heavy ground tracker** (another terminal):
```bash
cd ground-station
python dronebig.py --source "<video>"
```

**4. Connect a dashboard** to `ws://localhost:8000/ws/radar` (separate repo), or
just inspect `GET http://localhost:8000/api/unified`.

**Run the tests** (no weights/camera needed):
```bash
pytest
```

---

## 10. Status & limitations (the honest version)

This is a WIP portfolio project. What's real vs. not:

**Working today**
- Edge detector: ONNX YOLO, motion-guided SAHI cropping, Kalman + centroid
  tracking, the bbox-growth threat estimate, the HUD, and non-blocking telemetry.
- Ground tracker (`dronebig.py`): PyTorch YOLO + SAHI + ByteTrack, forwarding
  tracks to the server.
- Server: validated ingest, per-sender state, **cross-sender merge into a unified
  global-track list with stable IDs**, and REST + WebSocket fan-out.

**Work in progress / not implemented**
- **Calibrated multi-view fusion.** The merge associates purely in each sender's
  reported az/el and assumes a shared angular frame. No extrinsic calibration, no
  world-coordinate projection — merged tracks are angular-only.
- **NIR sensor fusion.** The `fuse_detections` code path exists, but no NIR
  detector is wired in, so the NIR candidate list is always empty and fusion has
  no real effect.
- **Ground-station threat scoring.** `dronebig.py` emits a neutral placeholder
  (`0.0` / `UNKNOWN`); only the edge node computes a real threat estimate.
- **Not benchmarked on a Raspberry Pi 5.** Developed and tested on a PC;
  on-device latency/throughput are unverified.

---

## 11. Key algorithms, explained simply

- **SAHI (Slicing-Aided Hyper Inference).** Small/distant drones vanish when you
  shrink a 4K frame to a 640px model input. SAHI instead cuts the frame into
  overlapping tiles, runs detection on each tile, and merges the results — so
  tiny targets survive. The cost is running the model many times per frame, which
  is why the edge node guides slicing with motion.
- **Motion-guided cropping (KNN background subtraction).** The edge node learns
  the static background and finds where pixels are *changing*. It then only runs
  YOLO on tiles around moving blobs, with a periodic full sweep for safety. Big
  CPU win on mostly-static sky footage.
- **Kalman filtering.** A predict/correct loop that models each target's position
  and velocity. It predicts the next position (used to match detections frame to
  frame and to "coast" through brief occlusions) and smooths noisy measurements.
- **Centroid + greedy matching.** With Kalman predictions in hand, the tracker
  matches new detections to existing tracks by nearest distance, gated by a max
  distance, to keep IDs stable.
- **ByteTrack** (in `dronebig.py`). An off-the-shelf, high-quality multi-object
  tracker that associates detections into tracks and is good at avoiding ID
  swaps. The project deliberately uses a hand-rolled tracker on the edge (for
  control/light weight) and ByteTrack on the heavy node.
- **bbox-growth threat heuristic.** Area growing ⇒ getting closer. Using the
  log-ratio of recent average areas makes "twice as big" and "half as big"
  symmetric, and the result is squashed to a [0, 1] score plus a coarse state.
- **Angular-gate merge (union-find).** Treat each track as a point in (az, el).
  Greedily union the closest cross-sender pairs within a few degrees, never
  unioning two tracks from the same sensor, then keep stable global IDs across
  frames with a short time-to-live.

---

## 12. Design decisions & why they matter

- **One shared contract (`schemas.py`).** Producers and consumer validate the
  same Pydantic models, so the format can't silently drift, and out-of-range
  junk (`threat_score=50`) is rejected at ingest instead of corrupting the UI.
- **One shared telemetry client (`telemetry.py`).** Killed a near-identical
  copy-pasted client on each node. One place to fix bugs, one behaviour.
- **Non-blocking, drop-oldest telemetry.** Detection latency is sacred; the
  network is not allowed to slow the frame loop, so a full queue drops the oldest
  frame rather than blocking.
- **Server-clock staleness.** The merge judges freshness by when the server
  *received* data, not by sender timestamps, so unsynchronized clocks don't break
  staleness logic.
- **Honest, scoped merge.** Rather than claim "calibrated fusion," the merge does
  one well-defined thing (angular association with stable IDs) and the code and
  docs state the coordinate-frame caveat plainly. One finished, honestly-described
  feature beats three half-features.
- **Tests on the pure logic.** The bits most likely to silently break (contract,
  merge, matching, geometry) are covered without needing models or hardware.

---

## 13. Glossary

- **Counter-UAS / C-UAS** — counter unmanned-aerial-system; detecting/tracking
  drones.
- **Edge node** — the lightweight field detector (`drone_detector.py`).
- **Ground station** — the PC side: the server plus the heavy tracker.
- **Sender** — anything that POSTs telemetry (both detectors are senders).
- **Track** — one target being followed over time, with a stable ID.
- **az / el** — azimuth (left-right) and elevation (up-down) angle off the
  camera's centerline, in degrees.
- **Boresight** — the camera's optical center direction; az/el are measured
  from it.
- **FOV** — field of view; used to convert pixel offsets into az/el angles.
- **SAHI** — Slicing-Aided Hyper Inference (tile-based detection of small
  objects).
- **NMS** — non-maximum suppression; removes duplicate overlapping boxes.
- **IoU** — intersection-over-union; how much two boxes overlap (0 to 1).
- **Kalman filter** — predict/correct estimator for position and velocity.
- **ByteTrack** — an off-the-shelf multi-object tracking algorithm.
- **Unified / global track** — a merged target combining one or more senders'
  tracks under one global ID.
- **TTL** — time-to-live; how long a stale track/sender is kept before being
  dropped.
- **HUD** — heads-up display; the on-screen overlay (crosshair, brackets, seeker
  PIP).
- **PIP** — picture-in-picture; the zoomed "seeker view" of the primary target.
```
