import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Make the repo-root `common` package importable when run as `python server.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schemas import TelemetryPayload  # noqa: E402
from common.merge import TrackMerger  # noqa: E402

app = FastAPI(title="Anti-Drone Ground Station API")

# Allow the separate UI repo to connect via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"[WS ERROR] Broadcast failed: {e}")

manager = ConnectionManager()

# Global state to keep the latest telemetry per sender
latest_telemetry: Dict[str, Any] = {}

# Fuses per-sender tracks into one global list with stable IDs. See
# common/merge.py for the angular-gate association and its coordinate caveat.
merger = TrackMerger()

# Most recent merged global-track list, served to dashboards on connect.
latest_unified: List[dict] = []

@app.post("/api/telemetry")
async def receive_telemetry(payload: TelemetryPayload):
    """
    Edge nodes and ground trackers POST their active targets here.
    """
    recv = time.time()
    ts = payload.timestamp or recv

    # Tracks arrive already validated against the shared contract; store them
    # as plain dicts so the rest of the pipeline stays JSON-serializable.
    tracks = [track.model_dump() for track in payload.tracks]

    # Update per-sender state. received_at is the server clock, used for
    # staleness so we don't depend on senders' clocks being in sync.
    latest_telemetry[payload.sender_id] = {
        "timestamp": ts,
        "received_at": recv,
        "tracks": tracks,
        "sender_id": payload.sender_id
    }

    # Re-fuse all senders into the unified global-track list.
    global latest_unified
    latest_unified = merger.merge(latest_telemetry, now=recv)

    # Tell connected dashboard clients immediately: the raw per-sender update
    # plus the freshly merged unified view.
    broadcast_msg = json.dumps({
        "type": "telemetry_update",
        "sender_id": payload.sender_id,
        "tracks": tracks,
        "unified": latest_unified,
        "timestamp": ts
    })

    # Broadcast asynchronously
    asyncio.create_task(manager.broadcast(broadcast_msg))

    return {"status": "ok", "tracks_received": len(payload.tracks), "unified_tracks": len(latest_unified)}

@app.get("/api/state")
async def get_current_state():
    """
    GET endpoint so the dashboard can pull the latest known state immediately on load.
    """
    return latest_telemetry

@app.get("/api/unified")
async def get_unified_state():
    """
    Latest fused global-track list (one entry per physical target across all
    senders). Angular (az/el) frame only — see common/merge.py.
    """
    return {"tracks": latest_unified, "count": len(latest_unified)}

@app.websocket("/ws/radar")
async def websocket_radar(websocket: WebSocket):
    """
    WebSocket endpoint for the UI Dashboard to receive real-time streams.
    """
    await manager.connect(websocket)
    try:
        # Send current state upon connection
        await websocket.send_text(json.dumps({
            "type": "full_state",
            "data": latest_telemetry,
            "unified": latest_unified
        }))
        
        while True:
            # Keep connection alive, wait for incoming messages if need be
            data = await websocket.receive_text()
            # Dashboard could send commands here (e.g. "LOCK TARGET X")
            print(f"[WS MSG] {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    print("[SERVER] Starting Ground Station API on port 8000...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
