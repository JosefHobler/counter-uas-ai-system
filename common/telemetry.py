"""Non-blocking HTTP helpers shared by every detection node.

``AsyncPoster`` is a fire-and-forget POST worker: a background thread drains a
bounded queue and POSTs each payload, dropping the *oldest* frame when the queue
is full, so a slow or unreachable server never stalls the frame loop and never
builds a backlog. ``TelemetryClient`` wraps it for the telemetry payload shape.

Both the edge detector and the ground-station tracker import these (previously
duplicated as ``TelemetryClient`` and ``MiniTelemetryClient``). The cue/confirm
candidate sender reuses the same ``AsyncPoster``.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import List

import requests


class AsyncPoster:
    """Background worker that POSTs JSON payloads, dropping the oldest if full."""

    def __init__(
        self,
        endpoint_url: str,
        max_queue_size: int = 10,
        post_timeout: float = 0.5,
    ):
        self.endpoint_url = endpoint_url
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

    def post(self, payload: dict):
        """Queue one payload. Drops the oldest queued payload if the queue is full."""
        if not self.running:
            return

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


class TelemetryClient:
    """Non-blocking sender for one telemetry frame (a list of tracks)."""

    def __init__(
        self,
        endpoint_url: str,
        sender_id: str,
        max_queue_size: int = 10,
        post_timeout: float = 0.5,
    ):
        self.sender_id = sender_id
        self._poster = AsyncPoster(endpoint_url, max_queue_size, post_timeout)

    @property
    def running(self) -> bool:
        return self._poster.running

    def send(self, tracks: List[dict]):
        """Queue one telemetry frame. Drops the oldest frame if the queue is full."""
        self._poster.post(
            {
                "sender_id": self.sender_id,
                "timestamp": time.time(),
                "tracks": tracks,
            }
        )

    def stop(self):
        self._poster.stop()
