"""Ground-station confirm service for the cue/confirm pipeline.

Receives candidate crops from edge nodes over HTTP, runs the heavy SAHI/YOLO
model (the same one ``dronebig.py`` uses) on each crop, and returns a
``Confirmation``: is this really a drone, and if so a tight box mapped back to
full-frame coordinates. Confirmations are also forwarded to the broker
(``server.py``) so the dashboard sees them.

Run:
    python confirm_service.py --model best.pt \
        --broker http://localhost:8000/api/confirmation --port 8001
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
from fastapi import FastAPI

# Make the repo-root `common` package importable when run from ground-station/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schemas import BBox, CandidateCrop, Confirmation  # noqa: E402
from common.candidates import decode_jpeg_b64  # noqa: E402
from common.geometry import map_local_box_to_full  # noqa: E402
from common.telemetry import AsyncPoster  # noqa: E402

import dronebig  # noqa: E402  reuse load_detection_model / detect_on_image

# A candidate counts as a confirmed drone if the heavy model finds any detection
# inside the crop at or above this confidence.
DEFAULT_DRONE_THRESHOLD = 0.3

app = FastAPI(title="Ground Station Confirm Service")

_state = {
    "model": None,
    "forwarder": None,  # optional AsyncPoster to the broker
    "drone_threshold": DEFAULT_DRONE_THRESHOLD,
    "dump_dir": None,  # if set, every received crop is saved here for inspection
}


def configure(
    model_path: str = "best.pt",
    confidence: float = 0.3,
    device=None,
    broker_url: str | None = None,
    drone_threshold: float = DEFAULT_DRONE_THRESHOLD,
    dump_dir: str | None = None,
):
    """Load the heavy model and (optionally) wire up forwarding to the broker."""
    _state["model"] = dronebig.load_detection_model(model_path, confidence, device)
    _state["forwarder"] = AsyncPoster(broker_url) if broker_url else None
    _state["drone_threshold"] = drone_threshold
    _state["dump_dir"] = dump_dir
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)


@app.post("/api/candidate")
def confirm_candidate(candidate: CandidateCrop) -> Confirmation:
    """Run the heavy model on one candidate crop and return a verdict."""
    if _state["model"] is None:
        # Lazy default so `uvicorn confirm_service:app` works without configure().
        configure()
    model = _state["model"]

    img_bgr = decode_jpeg_b64(candidate.image_jpeg_b64)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Plain (non-sliced) inference: the crop is already small/zoomed.
    dets = dronebig.detect_on_image(model, img_rgb, sliced=False)

    threshold = _state["drone_threshold"]
    strong = [d for d in dets if d["confidence"] >= threshold]
    best = max(strong, key=lambda d: d["confidence"]) if strong else None

    if best is not None:
        x1, y1, x2, y2 = best["xyxy"]
        local_box = (x1, y1, x2 - x1, y2 - y1)
        fx, fy, fw, fh = map_local_box_to_full(
            local_box, (candidate.crop_x, candidate.crop_y)
        )
        confirmation = Confirmation(
            sender_id=candidate.sender_id,
            candidate_id=candidate.candidate_id,
            frame_id=candidate.frame_id,
            track_id=candidate.track_id,
            timestamp=time.time(),
            is_drone=True,
            label=best.get("label", "drone"),
            confidence=float(best["confidence"]),
            bbox=BBox(x=float(fx), y=float(fy), w=float(fw), h=float(fh)),
        )
    else:
        confirmation = Confirmation(
            sender_id=candidate.sender_id,
            candidate_id=candidate.candidate_id,
            frame_id=candidate.frame_id,
            track_id=candidate.track_id,
            timestamp=time.time(),
            is_drone=False,
            label="",
            confidence=0.0,
            bbox=None,
        )

    # Live verdict so you can watch results without curling the broker.
    verdict = "DRONE" if confirmation.is_drone else "no"
    best_conf = best["confidence"] if best is not None else 0.0
    print(
        f"[CONFIRM] cand {candidate.candidate_id} (frame {candidate.frame_id}): "
        f"{len(dets)} dets, best={best_conf:.2f} -> {verdict}"
    )

    dump_dir = _state["dump_dir"]
    if dump_dir:
        annotated = img_bgr.copy()
        if best is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in best["xyxy"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        fname = f"cand_{candidate.candidate_id:05d}_{verdict}_{best_conf:.2f}.jpg"
        cv2.imwrite(os.path.join(dump_dir, fname), annotated)

    forwarder = _state["forwarder"]
    if forwarder is not None:
        forwarder.post(confirmation.model_dump())

    return confirmation


def main():
    parser = argparse.ArgumentParser(description="Ground Station Confirm Service")
    parser.add_argument("--model", default="best.pt", help="Path to YOLO .pt weights")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument(
        "--drone-threshold",
        type=float,
        default=DEFAULT_DRONE_THRESHOLD,
        help="Min confidence inside a crop to call it a drone",
    )
    parser.add_argument(
        "--broker",
        default="http://localhost:8000/api/confirmation",
        help="Broker URL to forward confirmations to (empty to disable)",
    )
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="If set, save every received crop here (annotated) for inspection",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    configure(
        model_path=args.model,
        confidence=args.confidence,
        broker_url=args.broker or None,
        drone_threshold=args.drone_threshold,
        dump_dir=args.dump_dir,
    )

    import uvicorn

    print(f"[CONFIRM] Starting confirm service on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
