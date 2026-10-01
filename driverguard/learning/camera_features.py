"""Shared camera-only feature contract and causal sampling for training/runtime."""
from __future__ import annotations

import hashlib
import math
from collections import deque
from pathlib import Path

import numpy as np

from ..config import DriverGuardConfig, resolve_path
from ..schemas import FrameSignals

PHYSICAL_FIELDS = (
    "ear_left", "ear_right", "mar", "blink_bs_left", "blink_bs_right",
    "jaw_open_bs", "yaw", "pitch", "roll",
)
PHYSICAL_SCALES = (0.3, 0.3, 0.6, 1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
FEATURE_NAMES = PHYSICAL_FIELDS + tuple(f"{name}_valid" for name in PHYSICAL_FIELDS) + ("face_valid",)
CAMERA_FPS = 10.0
FEATURE_VERSION = "raw_geometry_masks_v1"
SAMPLING_POLICY = "first_observation_per_100ms_bin; missing_bins_invalid; reset_gap_over_300ms"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def input_contract(cfg: DriverGuardConfig, source_manifest: str | Path) -> dict:
    return {
        "schema_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "physical_scales": list(PHYSICAL_SCALES),
        "fps": CAMERA_FPS,
        "landmarker_cadence_hz": CAMERA_FPS,
        "quality": cfg.quality.model_dump(),
        "low_light": cfg.low_light.model_dump(),
        "face_model_sha256": sha256_file(resolve_path(cfg.models.face_landmarker)),
        "extractor_version": FEATURE_VERSION,
        "source_manifest_sha256": sha256_file(source_manifest),
        "sampling": SAMPLING_POLICY,
    }


def encode_frame(signals: FrameSignals, cfg: DriverGuardConfig) -> tuple[np.ndarray, bool]:
    q = cfg.quality
    face_valid = bool(signals.camera_ok and signals.face_detected
                      and math.isfinite(signals.face_quality) and signals.face_quality >= q.min_quality)
    x = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if not face_valid:
        return x, False
    eyes_valid = ((signals.yaw is None or abs(signals.yaw) <= q.max_eye_valid_yaw_deg)
                  and (signals.pitch is None or abs(signals.pitch) <= q.max_eye_valid_pitch_deg))
    for index, (name, scale) in enumerate(zip(PHYSICAL_FIELDS, PHYSICAL_SCALES, strict=True)):
        value = getattr(signals, name)
        if value is None or not math.isfinite(value):
            continue
        if name in {"ear_left", "ear_right", "blink_bs_left", "blink_bs_right"} and not eyes_valid:
            continue
        if name.startswith("blink_bs") or name == "jaw_open_bs":
            if not 0 <= value <= 1:
                continue
        elif name in {"ear_left", "ear_right", "mar"} and value < 0:
            continue
        x[index] = np.clip(value / scale, -4.0, 4.0)
        x[index + len(PHYSICAL_FIELDS)] = 1.0
    x[-1] = 1.0
    return x, True


class CameraFrameBuffer:
    """A bounded past-only, 10 Hz window; missed bins contain invalid evidence."""

    def __init__(self, cfg: DriverGuardConfig, context: int = 50):
        if context < 1:
            raise ValueError("context must be positive")
        self.cfg = cfg
        self.context = context
        self._items: deque = deque(maxlen=context)
        self._last_bin: int | None = None
        self._last_ts: int | None = None

    def clear(self) -> None:
        self._items.clear()
        self._last_bin = self._last_ts = None

    def update(self, signals: FrameSignals) -> bool:
        ts = signals.ts_ms
        if ts < 0 or (self._last_ts is not None and ts <= self._last_ts):
            self.clear()
            return False
        if self._last_ts is not None and ts - self._last_ts > 300:
            self.clear()
        self._last_ts = ts
        bin_index = ts // 100
        if self._last_bin is not None and bin_index == self._last_bin:
            return False
        if self._last_bin is not None:
            for missing in range(self._last_bin + 1, bin_index):
                self._items.append((np.zeros(len(FEATURE_NAMES), np.float32), missing * 100, False))
        vector, valid = encode_frame(signals, self.cfg)
        self._items.append((vector, bin_index * 100, valid))
        self._last_bin = bin_index
        return True

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._items:
            return (np.empty((0, len(FEATURE_NAMES)), np.float32),
                    np.empty(0, np.int64), np.empty(0, bool))
        vectors, timestamps, valid = zip(*self._items, strict=True)
        return np.stack(vectors), np.asarray(timestamps, np.int64), np.asarray(valid, bool)


class CameraFeatureCollector:
    """Preserve every sampled segment offline, including gaps rejected by the head."""

    def __init__(self, cfg: DriverGuardConfig):
        self._sampler = CameraFrameBuffer(cfg, context=4)
        self._items: list = []
        self._last_input_ts: int | None = None

    def update(self, signals: FrameSignals) -> None:
        if self._last_input_ts is not None and signals.ts_ms <= self._last_input_ts:
            raise ValueError("Offline clip timestamps must be strictly increasing")
        self._last_input_ts = signals.ts_ms
        if not self._sampler.update(signals):
            return
        vectors, timestamps, valid = self._sampler.arrays()
        last_ts = self._items[-1][1] if self._items else -1
        for vector, ts, accepted in zip(vectors, timestamps, valid, strict=True):
            if ts > last_ts:
                self._items.append((vector.copy(), ts, accepted))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._items:
            return (np.empty((0, len(FEATURE_NAMES)), np.float32),
                    np.empty(0, np.int64), np.empty(0, bool))
        vectors, timestamps, valid = zip(*self._items, strict=True)
        return np.stack(vectors), np.asarray(timestamps, np.int64), np.asarray(valid, bool)
