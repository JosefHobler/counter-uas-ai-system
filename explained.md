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

A **distributed drone-detection pipeline** ("Counter-UAS" = Counter–Unmanned
Aerial System) built around a **cue / confirm** design. It has these moving
parts:

1. A **lightweight edge node** meant to run in the field on a drone (sized to
   target a Raspberry Pi 5) that does fast, low-latency detection on a camera
   feed and **cues** drone-like candidates.
2. A **ground confirm service** (a Jetson Orin NX–class GPU box) that runs a
   bigger, more accurate model **only on the cropped candidates** the edge sends,
   and decides whether each one is really a drone.
3. A **telemetry server / broker** that collects the edge's telemetry and the
   ground's confirmations and streams them to any connected dashboard in real
   time.

The core idea: the cheap edge stage is tuned for **high recall** (better to
over-flag than miss); the heavy ground stage is the **judge** that rejects false
positives — and only small crops, not full video, cross the link.

There is also a **secondary** capability: a standalone heavy tracker
(`dronebig.py`) can post its own tracks, and the broker can merge tracks from
multiple senders into a unified list. That path is separate from cue/confirm and
is angular-only, best-effort (see §6 and §10).

The dashboard UI itself lives in a **separate repository** — this repo is the
detection + telemetry backend, plus the models.

It is a **work-in-progress project**, bench-verified but not yet deployed to the
target hardware. Section 10 is explicit about what genuinely works versus what
is stubbed.

---

## 2. The big picture (architecture & data flow)

The primary flow is **cue → confirm**: the edge crops a drone-like candidate and
sends it to the confirm service, which runs the heavy model and posts a verdict
to the broker, which fans everything out to dashboards over a WebSocket.

```
   EDGE (Pi 5, on the drone)                  GROUND (Jetson Orin NX class)
┌──────────────────────────────┐
│ edge-rpi5/drone_detector.py  │  candidate   ┌──────────────────────────────┐
│  • compact ONNX YOLO         │    crop      │ ground-station/               │
│  • frame-skip + SAHI crops   │ ───────────► │   confirm_service.py          │
│  • Kalman + centroid tracker │/api/candidate│   • heavy PyTorch YOLO        │
│  • cues drone-like regions   │              │   • verdict: drone or not     │
└───────────────┬──────────────┘              └───────────────┬──────────────┘
                │ telemetry tracks                             │ confirmation
                │ /api/telemetry                               │ /api/confirmation
                ▼                                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ ground-station/server.py  — FastAPI broker                   │
        │  • validates ingest  • holds state  • REST + WebSocket        │
        └────────────────────────────────┬─────────────────────────────┘
                                          │ WebSocket /ws/radar
                                          ▼
                                 ┌─────────────────────┐
                                 │  Dashboard UI       │
                                 │  (separate repo)    │
                                 └─────────────────────┘
```

**The cue/confirm flow, in one paragraph:** The edge detector processes frames,
tracks targets, and (when `--confirm-url` is set) crops each drone-like candidate
and POSTs the crop to the confirm service at `/api/candidate`. The confirm
service decodes the crop, runs the heavy model on it, maps any detected box back
to full-frame coordinates, and returns a `Confirmation` (drone or not) — which it
also forwards to the broker at `/api/confirmation`. The edge separately POSTs its
own tracks as telemetry. The broker validates every payload, stores the latest
state, and broadcasts telemetry updates and confirmations to all connected
dashboards over the WebSocket. A dashboard that connects mid-stream first gets a
full snapshot, then live updates.

**Who is a server vs. a client:** `server.py` is the broker (receives, never
initiates). `confirm_service.py` is *both* — an HTTP server to the edge (receives
crops) and a client to the broker (forwards verdicts). `drone_detector.py` is a
pure client. `dronebig.py` (the standalone tracker, secondary path) is also a
pure client that posts telemetry to the broker.

---

## 3. Repository layout

```
drones/
├── common/                     # Shared code imported by BOTH nodes
│   ├── __init__.py             # marks it a package; explains the "why share"
│   ├── schemas.py              # Pydantic contracts: telemetry + candidate/confirmation
│   ├── telemetry.py            # AsyncPoster (non-blocking POST worker) + TelemetryClient
│   ├── candidates.py           # Candidate-crop packing + non-blocking CandidateSender
│   ├── geometry.py             # Pure box math: IoU, crop region, coord mapping, association
│   └── merge.py                # Cross-sender track merger (angular, global IDs) — secondary path
│
├── edge-rpi5/                  # The lightweight edge node (cues candidates)
│   ├── drone_detector.py       # Main pipeline + CLI (the big one, ~1300 lines)
│   ├── tracker.py              # Centroid + Kalman per-frame tracker
│   ├── config.py               # All tunable parameters in one place
│   └── best.onnx               # Compact ONNX model weights (~9.8 MB)
│
├── ground-station/             # The heavy AI node + the broker
│   ├── server.py               # FastAPI broker (telemetry + confirmations, REST + WebSocket)
│   ├── confirm_service.py      # Heavy-model confirm service (receives crops → verdicts)
│   ├── dronebig.py             # Standalone SAHI + ByteTrack tracker; shares model code
│   └── best.pt                 # Full PyTorch model weights (~53 MB)
│
├── tests/                      # Unit tests (pytest) — pure logic, no weights/camera
│   ├── test_schemas.py         # telemetry + candidate/confirmation contract validation
│   ├── test_geometry.py        # IoU, crop region, coord mapping, IoU association
│   ├── test_candidates.py      # candidate-crop packing + JPEG round-trip
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

`common/` is the shared contract + helpers, `edge-rpi5/` is the field box (the
cue), `ground-station/` is the GPU box (the confirm + broker).

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

It also defines the **cue/confirm** contract:

- **`BBox`** — a simple `(x, y, w, h)` pixel box, reused below.
- **`CandidateCrop`** — what the edge ships to the confirm service: `sender_id`,
  `candidate_id`, `frame_id`, optional `track_id`, the candidate `bbox` (in
  full-frame coords), the crop origin (`crop_x`, `crop_y`) and size, the edge
  `confidence`, and the crop itself as a base64 JPEG (`image_jpeg_b64`). The
  origin is carried so the confirm service can map a box it finds *inside* the
  crop back to full-frame coordinates.
- **`Confirmation`** — the ground verdict: echoes `candidate_id` / `track_id` (so
  it can be tied back to the edge track), plus `is_drone`, `label`, `confidence`,
  and a tight `bbox` (present only when `is_drone`).

**Why this matters:** the bounds (`ge=0.0, le=1.0`) mean a malformed value like
`threat_score=50` is rejected *at the door* instead of poisoning the dashboard.
Before this contract existed, the ground tracker shipped exactly that bad value.
Every producer and consumer validates against these same models, so the sides
physically cannot disagree about the format.

### `common/telemetry.py` — non-blocking POST, shared
Two classes that let any node ship data to the server **without ever stalling its
frame loop**.

- **`AsyncPoster`** — the generic worker. On construction it starts a background
  daemon thread and a bounded queue (default size 10). `post(payload)` drops a
  dict on the queue; **if the queue is full it discards the oldest** so a slow or
  unreachable server never builds a backlog. The thread drains the queue and does
  a `requests.post` with a short timeout (0.5 s), swallowing network errors on
  purpose — in degraded conditions you'd rather drop a frame than block.
  `stop()` shuts it down cleanly.
- **`TelemetryClient`** — a thin wrapper that packs a telemetry payload
  (`sender_id`, `timestamp`, `tracks`) and hands it to an `AsyncPoster`.

This started as "kill the duplicated client" (the edge and ground tracker each
had a near-identical `TelemetryClient` / `MiniTelemetryClient`). The
`AsyncPoster` split came later so the cue/confirm `CandidateSender` and the
confirm service's broker-forwarder reuse the *exact* same drop-oldest behavior
instead of re-implementing it.

### `common/geometry.py` — pure box math
Dependency-free helpers (no cv2/numpy), so the math is trivially unit-tested. Box
convention is `(x, y, w, h)` pixels.

- **`bbox_iou`** — intersection-over-union; the canonical copy (the edge's
  `drone_detector` now imports it instead of keeping its own).
- **`clamp_crop_region`** — given a candidate box, frame size, and a margin
  (fraction of the longer side), returns the crop rectangle, clamped to the frame.
- **`map_local_box_to_full`** — translate a box found *inside* a crop back to
  full-frame coordinates (crop origin + local box).
- **`associate_by_iou`** — greedy one-to-one matching of two box lists by
  descending IoU. This is the **single-camera** association tool: cue/confirm
  lives in one frame, so plain pixel IoU is precise and correct — unlike the
  angular gate the multi-sender `merge.py` needs.

### `common/candidates.py` — candidate-crop transport
The edge side of cue/confirm. Sends *crops*, not video, to keep the link light.

- **`encode_jpeg_b64` / `decode_jpeg_b64`** — JPEG ↔ base64 string (cv2).
- **`build_candidate_payload`** — the pure packer: crops `bbox` (with margin) out
  of a frame, encodes it, and returns a `CandidateCrop`-shaped dict (returns
  `None` if the crop would be empty). Unit-tested.
- **`CandidateSender`** — wraps `build_candidate_payload` with an `AsyncPoster`,
  so crops are shipped with the same non-blocking, drop-oldest delivery.

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
preprocessing, HUD colours, performance profile knobs, networking
(`GROUND_STATION_URL`, `NODE_ID`), and the **cue/confirm** knobs
(`CONFIRM_SERVICE_URL`, `CANDIDATE_EVERY_N_FRAMES` — throttle, and
`CANDIDATE_MARGIN` — context padding around each crop). Everything tunable lives
here so the pipeline code stays free of magic numbers.

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
  **cue candidates** → draw HUD → display/write), and handle keyboard controls
  (`q` quit, `n` cycle mode, `+/-` confidence). For each tracked object it
  computes az/el from the pixel offset and the camera FOV, attaches the threat
  assessment, and ships the batch via the shared `TelemetryClient`.
- **Cue/confirm hook** — when `--confirm-url` is set, `main()` also builds a
  `CandidateSender`. Throttled to every `CANDIDATE_EVERY_N_FRAMES`, it crops each
  AI-confirmed track out of the **raw** frame (no HUD) and ships it to the confirm
  service. (The `bbox_iou` used by NMS/dedup now comes from `common.geometry` —
  one shared copy.)

CLI flags worth knowing: `--source`, `--mode {day,night,thermal}`,
`--profile {default,balanced,pi5}` (trades accuracy for speed),
`--threat` (show the approach overlay), `--no-real-time` / `--output` (offline
export), `--confirm-url` (enable cue/confirm crop streaming), `--show-profile`
(per-stage timing).

> **Real-time caveat:** in real-time mode detection runs on a background thread,
> so the box lags a fast drone and a crop from the current frame can miss it.
> Use `--no-real-time` for clean crops, or raise `CANDIDATE_MARGIN`. (See §10.)

---

## 6. The ground station — `ground-station/`

### `ground-station/server.py` — the telemetry broker
A small **FastAPI** app. It's the hub everything else talks to. It has no model
and does no detection — it only ingests, merges, and broadcasts.

- Global state: `latest_telemetry` (the latest payload per sender), a single
  `TrackMerger`, `latest_unified` (the merged list), and `latest_confirmations`
  (the latest verdict per candidate).
- **`POST /api/telemetry`** — ingest. FastAPI validates the body against
  `TelemetryPayload` automatically (bad data → 422 before our code runs). It
  stamps `received_at` with the server clock, stores the tracks, re-runs the
  merge, and broadcasts a `telemetry_update` (raw + unified).
- **`POST /api/confirmation`** — ingest for the confirm service. Validates against
  `Confirmation`, stores it keyed `sender_id:candidate_id`, and broadcasts a
  `confirmation` message.
- **`GET /api/state`** / **`GET /api/unified`** / **`GET /api/confirmations`** —
  pull the latest per-sender state, merged list, and confirmations respectively
  (used by a dashboard on load).
- **`WebSocket /ws/radar`** — the live feed. On connect a client gets a
  `full_state` snapshot (telemetry + unified + confirmations), then
  `telemetry_update` per telemetry POST and `confirmation` per confirm POST.
  `ConnectionManager` tracks clients and handles broadcast + disconnect.
- CORS is wide open (`allow_origins=["*"]`) so the separate UI repo can connect.
- Run with `python server.py` → serves on `0.0.0.0:8000`.

### `ground-station/confirm_service.py` — the heavy confirm service
The ground side of cue/confirm. A FastAPI app that is **both** a server (to the
edge) and a client (to the broker).

- **`configure(...)`** loads the heavy model once (via `dronebig`'s shared
  `load_detection_model`) and optionally wires up an `AsyncPoster` that forwards
  verdicts to the broker.
- **`POST /api/candidate`** — receives a `CandidateCrop`, decodes the JPEG, runs
  the heavy model on the crop with **plain (non-sliced) inference** (a crop is
  already small/zoomed — SAHI tiling would be wasted), picks the best detection
  above `--drone-threshold`, maps its box back to full-frame coords with
  `map_local_box_to_full`, and returns a `Confirmation`. It prints a one-line
  verdict per candidate and, with `--dump-dir`, saves each annotated crop for
  inspection.
- Run with `python confirm_service.py --model best.pt --port 8001 --broker …`.

### `ground-station/dronebig.py` — standalone heavy tracker (secondary path)
The original high-accuracy tracker. It now plays two roles:

- **Shared model code.** `load_detection_model(...)` and
  `detect_on_image(..., sliced=True|False)` were extracted so the confirm service
  reuses the exact same model/inference (sliced for full frames, plain for crops)
  — no duplication.
- **Standalone tracker (a telemetry sender).** Its `main()` reads a video, runs
  full **SAHI** slicing per frame, feeds detections to **ByteTrack**
  (`sv.ByteTrack`) for stable IDs, and POSTs tracks to the broker as sender
  `"base-station-sahi"`. This is the multi-sender/merge path, separate from
  cue/confirm.

Note the explicit honesty in the code: this tracker has no depth proxy, so it
emits `threat_score=0.0` / `threat_state="UNKNOWN"` — a neutral, in-range value
(it once was the buggy `threat_score=50` the contract now forbids).

---

## 7. The tests — `tests/` and `conftest.py`

A focused **pytest** suite (51 tests) over the pure-logic pieces. None of them
need the model weights, a camera, or a running server, so they run anywhere in
about a second.

- **`conftest.py`** — puts the repo root (for the `common` package) and
  `edge-rpi5/` (for `config`, `tracker`, `drone_detector`) on `sys.path`, so the
  tests can import the modules without installing anything.
- **`test_schemas.py`** — the contracts: safe defaults and rejection of the
  `threat_score=50` bug, out-of-range confidence, negative box dims, unknown
  keys, bad enum values; plus the cue/confirm models (`CandidateCrop`,
  `Confirmation`, `BBox`) and a round-trip dump check.
- **`test_geometry.py`** — `bbox_iou`, `clamp_crop_region` (margin + clamping),
  `map_local_box_to_full`, and `associate_by_iou` (greedy one-to-one matching).
- **`test_candidates.py`** — JPEG round-trip, and `build_candidate_payload`
  producing a dict that validates against `CandidateCrop` with the right crop
  origin/size (off-frame boxes return `None`).
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

**Cue/confirm.** What the edge POSTs to `/api/candidate` (`CandidateCrop`):

```json
{
  "sender_id": "edge-rpi5-alpha",
  "candidate_id": 5, "frame_id": 2475, "track_id": 0,
  "timestamp": 1781432885.5,
  "bbox": {"x": 900.0, "y": 480.0, "w": 60.0, "h": 40.0},
  "crop_x": 876.0, "crop_y": 464.0, "crop_width": 108, "crop_height": 72,
  "confidence": 0.41,
  "image_jpeg_b64": "<base64 JPEG of the crop>"
}
```

What the confirm service returns (and forwards to `/api/confirmation`) — a
`Confirmation`:

```json
{
  "sender_id": "edge-rpi5-alpha",
  "candidate_id": 5, "frame_id": 2475, "track_id": 0,
  "timestamp": 1781432885.6,
  "is_drone": true,
  "label": "drone",
  "confidence": 0.89,
  "bbox": {"x": 902.0, "y": 482.0, "w": 55.0, "h": 38.0}
}
```

The confirmation's `bbox` is the heavy model's tight box, already mapped back to
full-frame coordinates (crop origin + the box it found inside the crop).

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

**4. Connect a dashboard** to `ws://localhost:8000/ws/radar` (separate repo), or
just inspect `GET http://localhost:8000/api/unified`.

### The cue/confirm pipeline (the primary path)

Three processes, in order: broker → confirm service → edge.

```bash
# 1. Broker
cd ground-station && python server.py

# 2. Confirm service (heavy model; receives crops, returns verdicts)
python confirm_service.py --model best.pt --port 8001 \
    --broker http://localhost:8000/api/confirmation

# 3. Edge with crop streaming enabled
cd ../edge-rpi5
python drone_detector.py --source "../<video>" \
    --confirm-url http://localhost:8001/api/candidate
```

Watch the confirm service's `[CONFIRM] cand N ... -> DRONE/no` lines, add
`--dump-dir DIR` to save each crop, and read verdicts at
`GET http://localhost:8000/api/confirmations`. For clean crops while testing, add
`--no-real-time` to the edge (avoids the detection-lag described in §10).

### The secondary path (standalone heavy tracker + merge)

```bash
cd ground-station
python dronebig.py --source "<video>"   # posts its own tracks; broker merges senders
```

**Run the tests** (no weights/camera needed):
```bash
pytest
```

> On Windows the dependencies are installed under `py -3.10`; use
> `py -3.10 …` / `py -3.10 -m pytest` if a plain `python` lacks `cv2`.

---

## 10. Status & limitations (the honest version)

This is a WIP project. What's real vs. not:

**Working today**
- Edge detector: ONNX YOLO, motion-guided SAHI cropping, Kalman + centroid
  tracking, the bbox-growth threat estimate, the HUD, and non-blocking telemetry.
- **Cue/confirm pipeline**: the edge crops candidates and ships them to
  `confirm_service.py`, which runs the heavy model and returns a drone/not-drone
  verdict to the broker. Verified end-to-end on the bench (the ground model
  confirms real drones from edge-cued crops at high confidence).
- Standalone ground tracker (`dronebig.py`): PyTorch YOLO + SAHI + ByteTrack,
  forwarding tracks to the broker; shares its model code with the confirm service.
- Broker: validated telemetry + confirmation ingest, per-sender state,
  cross-sender merge into a unified list, and REST + WebSocket fan-out.

**Work in progress / not implemented**
- **Detection latency vs. fast targets.** In real-time mode the edge runs
  detection on a background thread, so the box lags the drone and a crop from the
  current frame can miss a fast mover. Mitigated by `--no-real-time` or a wider
  `CANDIDATE_MARGIN`; proper motion-prediction of the crop region isn't built.
- **Calibrated multi-view fusion** (secondary path). The merge associates purely
  in each sender's reported az/el and assumes a shared angular frame — no
  extrinsic calibration, no world-coordinate projection. The single-camera
  cue/confirm path doesn't need this.
- **NIR sensor fusion.** The `fuse_detections` code path exists, but no NIR
  detector is wired in, so the NIR candidate list is always empty and fusion has
  no real effect.
- **Ground-station threat scoring.** `dronebig.py` emits a neutral placeholder
  (`0.0` / `UNKNOWN`); only the edge node computes a real threat estimate.
- **Not benchmarked on Pi 5 or Jetson Orin NX.** Developed/tested on a PC;
  on-device latency/throughput are unverified, and the Jetson needs its own
  CUDA-enabled PyTorch build (PyPI `torch` won't give it GPU). Running the heavy
  model on a CPU is very slow — the design assumes the Orin's GPU.

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
- **Cue / confirm (two-tier inference).** A cheap, high-recall detector on the
  edge decides *what's worth a closer look* and ships only small crops; a heavy,
  high-precision model on the GPU box is the *judge* that confirms or rejects.
  This spends bandwidth and heavy compute only when something is flagged, instead
  of streaming full video and running the big model on every frame.
- **Pixel-IoU association.** Because the crop and the heavy model's result live in
  the *same* frame, matching is plain intersection-over-union in pixels — precise,
  and far simpler than the angular gate the multi-camera merge needs.

---

## 12. Design decisions & why they matter

- **One shared contract (`schemas.py`).** Producers and consumer validate the
  same Pydantic models, so the format can't silently drift, and out-of-range
  junk (`threat_score=50`) is rejected at ingest instead of corrupting the UI.
- **One shared POST worker (`telemetry.py`).** Killed a near-identical
  copy-pasted client on each node, then split out `AsyncPoster` so the telemetry
  client, the candidate sender, and the confirm-service forwarder all reuse the
  same non-blocking, drop-oldest behaviour.
- **Cue/confirm over crops, not video.** The edge filters first and sends only
  small crops; the heavy GPU model runs only on those. It keeps the tether light
  and the big model idle until there's something to judge — the right division of
  labour for a Pi-on-a-drone + Orin-on-the-ground deployment.
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
- **Ground station** — the GPU side: the broker plus the heavy confirm service.
- **Cue / confirm** — the two-tier pattern: cheap edge detector *cues* candidates,
  heavy ground model *confirms* them.
- **Candidate** — a drone-like region the edge flagged and cropped, sent for
  confirmation (`CandidateCrop`).
- **Confirmation** — the heavy model's verdict on a candidate (`Confirmation`).
- **Sender** — anything that POSTs to the broker (the edge, and `dronebig.py`).
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
