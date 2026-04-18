import asyncio
import json
import time
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Anti-Drone Ground Station API")

# Allow the separate UI repo to connect via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryPayload(BaseModel):
    sender_id: str
    tracks: List[Dict[str, Any]]
    timestamp: float = None

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

@app.post("/api/telemetry")
async def receive_telemetry(payload: TelemetryPayload):
    """
    Edge nodes and ground trackers POST their active targets here.
    """
    ts = payload.timestamp or time.time()
    
    # Update global state
    latest_telemetry[payload.sender_id] = {
        "timestamp": ts,
        "tracks": payload.tracks,
        "sender_id": payload.sender_id
    }
    
    # Tell connected dashboard clients immediately
    # You can merge or aggregate here depending on UI needs
    broadcast_msg = json.dumps({
        "type": "telemetry_update",
        "sender_id": payload.sender_id,
        "tracks": payload.tracks,
        "timestamp": ts
    })
    
    # Broadcast asynchronously
    asyncio.create_task(manager.broadcast(broadcast_msg))
    
    return {"status": "ok", "tracks_received": len(payload.tracks)}

@app.get("/api/state")
async def get_current_state():
    """
    GET endpoint so the dashboard can pull the latest known state immediately on load.
    """
    return latest_telemetry

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
            "data": latest_telemetry
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
