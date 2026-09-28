"""StateHub: thread-safe bridge between the inference thread and async API/WebSocket clients.

The pipeline calls `on_tick` from its own thread; WebSocket handlers `subscribe()` an asyncio
queue. Broadcasts are throttled to `stream_hz` (alerts are always delivered).
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any


class StateHub:
    def __init__(self, stream_hz: float = 10.0):
        self.min_interval = 1.0 / stream_hz
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._last_push = 0.0
        self.started = time.time()
        self.last_tick_wall: float | None = None

    # called from the pipeline thread
    def on_tick(self, tick, d: dict[str, Any]) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest = d
            self.last_tick_wall = time.time()
            if tick.alert is None and now - self._last_push < self.min_interval:
                return
            self._last_push = now
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(self._put, q, d)
            except RuntimeError:  # loop closed
                self.unsubscribe(loop, q)

    @staticmethod
    def _put(q: asyncio.Queue, d: dict) -> None:
        if q.full():  # slow client: drop the oldest, keep the stream real-time
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(d)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    def subscribe(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Queue]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        with self._lock:
            self._subs.add((loop, q))
        return loop, q

    def unsubscribe(self, loop, q) -> None:
        with self._lock:
            self._subs.discard((loop, q))

    def health(self) -> dict[str, Any]:
        latest = self.latest() or {}
        stale = self.last_tick_wall is None or time.time() - self.last_tick_wall > 2.0
        return {
            "status": "ok" if not stale else "no_ticks",
            "uptime_s": round(time.time() - self.started, 1),
            "camera_ok": latest.get("camera_ok"),
            "processing_fps": latest.get("processing_fps"),
            "latency_ms": latest.get("latency_ms"),
            "risk_level": latest.get("risk_level"),
            "clients": len(self._subs),
        }
