"""Frame sources with wall-clock timestamps.

Every temporal rule in DriverGuard runs on timestamps (ms), never on frame counts, so the
behaviour does not change when FPS drops from 30 to 10.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


@dataclass
class Frame:
    image_bgr: np.ndarray | None  # None when the camera failed this tick
    ts_ms: int
    camera_ok: bool
    index: int


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class CameraSource:
    """Live webcam. Timestamps are monotonic ms since the source was opened."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                 max_failures: int = 30):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.max_failures = max_failures
        self._t0 = time.monotonic()
        self._stopped = False

    def _now_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def frames(self) -> Iterator[Frame]:
        i = 0
        failures = 0
        while not self._stopped:
            ok, img = self.cap.read()
            ts = self._now_ms()
            if not ok or img is None:
                failures += 1
                # Report camera loss explicitly -> engine goes SENSOR_DEGRADED (never "normal").
                yield Frame(None, ts, camera_ok=False, index=i)
                if failures > self.max_failures:
                    time.sleep(0.2)  # back off, keep reporting
                i += 1
                continue
            failures = 0
            yield Frame(img, ts, camera_ok=True, index=i)
            i += 1

    def close(self) -> None:
        self._stopped = True
        self.cap.release()


class VideoFileSource:
    """Offline video replay. Timestamps come from the file (frame_index / fps) so replay is
    deterministic. `target_fps` subsamples (e.g. 15 fps on a 30 fps video) to emulate the
    edge device and to speed up batch extraction."""

    def __init__(self, path: str | Path, target_fps: float | None = None, realtime: bool = False,
                 start_ms: int = 0, max_ms: int | None = None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.target_fps = target_fps
        self.realtime = realtime
        self.start_ms = start_ms
        self.max_ms = max_ms

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.n_frames else 0.0

    def frames(self) -> Iterator[Frame]:
        step_ms = 1000.0 / self.target_fps if self.target_fps else 0.0
        next_ms = float(self.start_ms)
        i = -1
        wall0 = time.monotonic()
        while True:
            ok = self.cap.grab()
            if not ok:
                break
            i += 1
            ts = int(round(i * 1000.0 / self.fps))
            if ts < self.start_ms:
                continue
            if self.max_ms is not None and ts - self.start_ms > self.max_ms:
                break
            if step_ms and ts + 1e-6 < next_ms:
                continue  # subsample
            next_ms = max(next_ms + step_ms, ts) if step_ms else ts
            ok, img = self.cap.retrieve()
            if not ok:
                yield Frame(None, ts, camera_ok=False, index=i)
                continue
            if self.realtime:
                lag = (ts - self.start_ms) / 1000.0 - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            yield Frame(img, ts, camera_ok=True, index=i)

    def close(self) -> None:
        self.cap.release()


def open_source(spec: str, target_fps: float | None = None, realtime: bool = False):
    """'0' / '1' -> webcam index; 'synthetic:<scenario>' -> handled by the caller;
    anything else -> video file path."""
    if spec.isdigit():
        return CameraSource(int(spec))
    return VideoFileSource(spec, target_fps=target_fps, realtime=realtime)
