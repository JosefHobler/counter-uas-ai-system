"""Non-blocking telemetry client shared by every detection node.

Both the edge detector and the ground-station SAHI tracker need to ship tracks
to the server without ever stalling their frame loop. This is the single
implementation they both import (previously duplicated as ``TelemetryClient``
on the edge and ``MiniTelemetryClient`` in dronebig.py).

Design: a background thread drains a bounded queue and POSTs each frame
fire-and-forget. When the queue is full the oldest frame is dropped, so a slow
or unreachable server slows nothing down and never builds a backlog.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import List

import requests


class TelemetryClient:
    def __init__(
        self,
        endpoint_url: str,
        sender_id: str,
        max_queue_size: int = 10,
        post_timeout: float = 0.5,
    ):
        self.endpoint_url = endpoint_url
        self.sender_id = sender_id
        self.post_timeout = post_timeout
        self.q: "queue.Queue" = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            if payload is None:
                break

            try:
                requests.post(self.endpoint_url, json=payload, timeout=self.post_timeout)
            except requests.exceptions.RequestException:
                # Drops are expected in degraded network conditions; never let a
                # failed POST block the sender thread.
                pass
            finally:
                self.q.task_done()

    def send(self, tracks: List[dict]):
        """Queue one telemetry frame. Drops the oldest frame if the queue is full."""
        if not self.running:
            return

        payload = {
            "sender_id": self.sender_id,
            "timestamp": time.time(),
            "tracks": tracks,
        }

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
