import threading
import queue
import time
import requests
import json

class TelemetryClient:
    def __init__(self, endpoint_url: str, sender_id: str, max_queue_size: int = 10):
        self.endpoint_url = endpoint_url
        self.sender_id = sender_id
        self.q = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # Wait for data to send
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            if payload is None:
                break

            try:
                # Fire and forget
                requests.post(self.endpoint_url, json=payload, timeout=0.5)
            except requests.exceptions.RequestException:
                # We expect drops in bad network conditions. We don't want to lock the thread.
                pass
            finally:
                self.q.task_done()

    def send_telemetry(self, tracks: list):
        if not self.running:
            return
            
        payload = {
            "sender_id": self.sender_id,
            "timestamp": time.time(),
            "tracks": tracks
        }
        
        # If queue is full, drop the oldest telemetry frame to prevent backlog
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
                
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            pass

    def stop(self):
        self.running = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1.0)
