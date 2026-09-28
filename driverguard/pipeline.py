"""Pipeline: frame source -> perception -> engine -> sinks.

    camera frame (RAM only) -> PerceptionExtractor -> FrameSignals -> DriverGuardEngine -> TickOutput
                                                                                    |-> alert sinks (local)
                                                                                    |-> StateHub (API/WS)
                                                                                    |-> EventStore (SQLite)
                                                                                    |-> JSONL (offline eval)
Frames never leave `process_frame`; everything after the engine is metadata only.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Protocol

from .engine import DriverGuardEngine
from .privacy import assert_metadata_only
from .schemas import FrameSignals, RiskLevel, TickOutput


class TickSink(Protocol):
    def on_tick(self, tick: TickOutput, tick_dict: dict) -> None: ...


class JsonlSink:
    def __init__(self, path: str | Path, every_n: int = 1):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.every_n = every_n
        self._n = 0

    def on_tick(self, tick: TickOutput, d: dict) -> None:
        self._n += 1
        if self._n % self.every_n == 0 or tick.alert is not None:
            self.f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.f.close()


class EventStoreSink:
    """Logs alerts and level transitions; persists trip-memory buckets."""

    def __init__(self, store, engine: DriverGuardEngine):
        self.store = store
        self.engine = engine
        self._prev: RiskLevel | None = None
        store.start_trip(engine.trip_id, engine.driver_id)
        engine.memory.on_bucket_closed = lambda b: store.log_minute(engine.trip_id, b.to_dict())

    def on_tick(self, tick: TickOutput, d: dict) -> None:
        if tick.alert is not None:
            self.store.log_event(d, "alert")
        elif self._prev is not None and tick.risk_level != self._prev:
            self.store.log_event(d, "transition")
        self._prev = tick.risk_level

    def close(self) -> None:
        self.store.end_trip(self.engine.trip_id, self.engine.memory.summary())


class RuntimeStats:
    """Processed FPS (successful ticks / s over a sliding 2 s window) and per-frame latency."""

    def __init__(self):
        self._t: deque[float] = deque()
        self.lat: deque[float] = deque(maxlen=600)

    def tick(self, latency_ms: float) -> float:
        now = time.monotonic()
        self._t.append(now)
        while self._t and self._t[0] < now - 2.0:
            self._t.popleft()
        self.lat.append(latency_ms)
        span = self._t[-1] - self._t[0] if len(self._t) > 1 else 0
        return (len(self._t) - 1) / span if span > 0 else 0.0

    def latency_p95(self) -> float | None:
        if not self.lat:
            return None
        s = sorted(self.lat)
        return s[int(0.95 * (len(s) - 1))]


class DriverGuardPipeline:
    def __init__(self, engine: DriverGuardEngine, perception=None, speed_source=None,
                 sinks: Iterable[TickSink] = (), alert_sinks: Iterable = ()):
        self.engine = engine
        self.perception = perception
        self.speed_source = speed_source
        self.sinks = list(sinks)
        self.alert_sinks = list(alert_sinks)
        self.stats = RuntimeStats()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # guards engine.reconfigure vs step
        self.n_ticks = 0

    def _emit(self, tick: TickOutput, latency_ms: float) -> TickOutput:
        tick.processing_fps = round(self.stats.tick(latency_ms), 2)
        tick.latency_ms = round(latency_ms, 2)
        d = tick.to_dict()
        assert_metadata_only(d)
        for s in self.sinks:
            s.on_tick(tick, d)
        if tick.alert is not None:
            for a in self.alert_sinks:
                a.on_alert(tick.alert)
        self.n_ticks += 1
        return tick

    def process_signals(self, sig: FrameSignals, t_start: float | None = None) -> TickOutput:
        t_start = t_start if t_start is not None else time.perf_counter()
        if self.speed_source is not None and sig.speed_kmh is None:
            sig.speed_kmh = self.speed_source.read_speed(sig.ts_ms)
        with self.lock:
            tick = self.engine.step(sig)
        return self._emit(tick, (time.perf_counter() - t_start) * 1000.0)

    def process_frame(self, image_bgr, ts_ms: int, camera_ok: bool = True) -> TickOutput:
        t0 = time.perf_counter()
        sig = self.perception.process(image_bgr, ts_ms, camera_ok)
        return self.process_signals(sig, t0)

    def run_frames(self, source, on_tick=None, max_frames: int | None = None) -> None:
        for fr in source.frames():
            if self.stop_event.is_set():
                break
            tick = self.process_frame(fr.image_bgr, fr.ts_ms, fr.camera_ok)
            if on_tick:
                on_tick(tick, fr)
            if max_frames and self.n_ticks >= max_frames:
                break

    def run_signals(self, signals: Iterable[FrameSignals], on_tick=None, realtime: bool = False,
                    max_ticks: int | None = None) -> None:
        wall0, ts0 = time.monotonic(), None
        for sig in signals:
            if self.stop_event.is_set():
                break
            if realtime:
                ts0 = sig.ts_ms if ts0 is None else ts0
                lag = (sig.ts_ms - ts0) / 1000.0 - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            tick = self.process_signals(sig)
            if on_tick:
                on_tick(tick, None)
            if max_ticks and self.n_ticks >= max_ticks:
                break

    def close(self) -> None:
        for s in self.sinks:
            if hasattr(s, "close"):
                s.close()
        if self.perception is not None:
            self.perception.close()
