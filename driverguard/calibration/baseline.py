"""Per-driver calibration — the core of AI contribution "Hướng 1" (see docs/00_scope_mapping.md).

Problem: a fixed EAR threshold (~0.20–0.25, tuned on mostly Western datasets) produces
systematic false "eye closed" on drivers whose open-eye EAR is naturally low (monolid /
narrow palpebral fissure — common in the Vietnamese target population).

Baseline method implemented here (the "adaptive statistics" option in the GVHD review):
  * warm-up: open-eye level = high percentile of EAR while the face is valid & near-frontal
    (blinks are short, so the 80th percentile ~ open eye), shrunk toward a population prior
    so the first seconds are not noisy (Bayesian-style shrinkage, prior worth N seconds);
  * after warm-up: slow drift tracking (EWMA toward the rolling percentile) with a FLOOR —
    the baseline may not drift down more than `max_drift_down`, otherwise the system would
    "learn" the driver's growing drowsiness and hide it. This tension is a research question
    to measure (false alerts/hour vs event recall), not a solved problem.

`FixedEyeBaseline` is the control condition for the experiment (same pipeline, fixed threshold).
TODO(phase 2): mixture model / learned normaliser; pose-conditioned EAR correction; persist
per-driver profile (pseudonymous id) for warm start on the next trip.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..config import CalibrationCfg, EyeCfg

_SAMPLE_EVERY_MS = 200  # 5 Hz sampling for statistics, independent of camera fps
_ROLLING_S = 300


class FixedEyeBaseline:
    """Population threshold: eye closed iff EAR < fixed_closed_ear (expressed as a ratio so the
    same hysteresis logic applies)."""

    def __init__(self, cal: CalibrationCfg, eye: EyeCfg):
        self.open_level = cal.fixed_closed_ear / eye.closed_ratio_enter
        self.calibrated = True
        self.progress = 1.0

    def update(self, ts_ms: int, ear: float | None, usable: bool) -> None:
        pass

    def ratio(self, ear: float) -> float:
        return ear / self.open_level


class AdaptiveEyeBaseline:
    def __init__(self, cal: CalibrationCfg):
        self.cfg = cal
        self.open_level = cal.prior_ear_open
        self.calibrated = False
        self.calibrated_level: float | None = None
        self._samples: deque[float] = deque(maxlen=int(_ROLLING_S * 1000 / _SAMPLE_EVERY_MS))
        self._obs_ms = 0.0
        self._last_ts: int | None = None
        self._last_sample_ts = -(10**12)
        self._last_drift_ts: int | None = None

    @property
    def progress(self) -> float:
        return min(1.0, self._obs_ms / (self.cfg.warmup_s * 1000.0))

    def _percentile(self) -> float:
        return float(np.percentile(np.fromiter(self._samples, dtype=float), self.cfg.open_percentile))

    def update(self, ts_ms: int, ear: float | None, usable: bool) -> None:
        dt = 0 if self._last_ts is None else min(max(ts_ms - self._last_ts, 0), 200)
        self._last_ts = ts_ms
        if not usable or ear is None:
            return
        self._obs_ms += dt
        if ts_ms - self._last_sample_ts >= _SAMPLE_EVERY_MS:
            self._samples.append(float(ear))
            self._last_sample_ts = ts_ms

        if not self.calibrated:
            if len(self._samples) >= 5:
                p = self._percentile()
                w_obs = self._obs_ms / 1000.0
                w0 = self.cfg.prior_strength_s
                self.open_level = (w0 * self.cfg.prior_ear_open + w_obs * p) / (w0 + w_obs)
            if self._obs_ms >= self.cfg.warmup_s * 1000.0:
                # enough personal data: drop the population prior, keep the empirical level
                self.calibrated = True
                self.open_level = self._percentile()
                self.calibrated_level = self.open_level
                self._last_drift_ts = ts_ms
            return

        # Drift tracking, evaluated at most once per second.
        if self._last_drift_ts is None or ts_ms - self._last_drift_ts >= 1000:
            dt_d = 1000 if self._last_drift_ts is None else ts_ms - self._last_drift_ts
            self._last_drift_ts = ts_ms
            target = self._percentile()
            a = 1.0 - math.exp(-dt_d / (self.cfg.drift_tau_s * 1000.0))
            self.open_level += a * (target - self.open_level)
            floor = self.calibrated_level * (1.0 - self.cfg.max_drift_down)
            self.open_level = max(self.open_level, floor)

    def ratio(self, ear: float) -> float:
        return ear / self.open_level if self.open_level > 1e-6 else 0.0


class PoseBaseline:
    """Driver's neutral head pose (camera may be mounted off-axis). Decisions use pose relative
    to this neutral."""

    def __init__(self, warmup_s: float):
        self.warmup_ms = warmup_s * 1000.0
        self._yaw: list[float] = []
        self._pitch: list[float] = []
        self._obs_ms = 0.0
        self._last_ts: int | None = None
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.calibrated = False

    def update(self, ts_ms: int, yaw: float | None, pitch: float | None, usable: bool) -> None:
        dt = 0 if self._last_ts is None else min(max(ts_ms - self._last_ts, 0), 200)
        self._last_ts = ts_ms
        if self.calibrated or not usable or yaw is None or pitch is None:
            return
        self._obs_ms += dt
        self._yaw.append(yaw)
        self._pitch.append(pitch)
        if len(self._yaw) >= 10:
            self.neutral_yaw = float(np.median(self._yaw))
            self.neutral_pitch = float(np.median(self._pitch))
        if self._obs_ms >= self.warmup_ms:
            self.calibrated = True
            self._yaw.clear()
            self._pitch.clear()

    def relative(self, yaw: float | None, pitch: float | None) -> tuple[float | None, float | None]:
        if yaw is None or pitch is None:
            return None, None
        return yaw - self.neutral_yaw, pitch - self.neutral_pitch


def make_eye_baseline(cal: CalibrationCfg, eye: EyeCfg):
    return FixedEyeBaseline(cal, eye) if cal.method == "fixed" else AdaptiveEyeBaseline(cal)
