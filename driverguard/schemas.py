"""Data contracts.

FrameSignals  — output of perception for ONE frame. No temporal logic. This is what the
                Kaggle feature-extraction notebook writes to parquet, so every temporal /
                risk experiment can be replayed offline without re-running MediaPipe.
TickOutput    — output of the engine for one tick (the metadata-only contract in the
                research doc, section "Output contract"). This is the ONLY thing that may
                leave the process (API, logs, dashboard).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    SENSOR_DEGRADED = "sensor_degraded"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY = {
    RiskLevel.NORMAL: 0,
    RiskLevel.CAUTION: 1,
    RiskLevel.WARNING: 2,
    RiskLevel.CRITICAL: 3,
    RiskLevel.SENSOR_DEGRADED: -1,
}


class Reason(str, Enum):
    # drowsiness evidence
    MICROSLEEP = "MICROSLEEP"
    PROLONGED_EYE_CLOSURE = "PROLONGED_EYE_CLOSURE"
    HIGH_PERCLOS = "HIGH_PERCLOS"  # NOTE: PERCLOS-proxy estimated from EAR, not IR PERCLOS
    ELEVATED_PERCLOS = "ELEVATED_PERCLOS"
    FREQUENT_YAWN = "FREQUENT_YAWN"
    HEAD_NOD = "HEAD_NOD"
    FATIGUE_TREND_RISING = "FATIGUE_TREND_RISING"
    RISK_SCORE = "RISK_SCORE"
    # distraction evidence
    EYES_OFF_ROAD = "EYES_OFF_ROAD"
    LOOKING_DOWN = "LOOKING_DOWN"
    PHONE_USE = "PHONE_USE"
    # context
    LONG_DRIVE = "LONG_DRIVE"
    CIRCADIAN_LOW = "CIRCADIAN_LOW"
    PARKED = "PARKED"
    CALIBRATING = "CALIBRATING"
    # sensor state
    FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"
    LOW_FACE_QUALITY = "LOW_FACE_QUALITY"
    CAMERA_LOST = "CAMERA_LOST"


# Reasons that describe context/information only; they never raise the level by themselves.
INFO_REASONS = {Reason.CALIBRATING, Reason.PARKED, Reason.CIRCADIAN_LOW}


def _f(x: Any) -> Any:
    """JSON-safe float (NaN -> None), rounded to keep payloads small."""
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 4)
    return x


@dataclass
class FrameSignals:
    ts_ms: int
    camera_ok: bool = True
    face_detected: bool = False
    face_quality: float = 0.0
    # normalized [0,1] face bbox (x0, y0, x1, y1)
    face_x0: float | None = None
    face_y0: float | None = None
    face_x1: float | None = None
    face_y1: float | None = None
    ear_left: float | None = None
    ear_right: float | None = None
    blink_bs_left: float | None = None  # MediaPipe blendshape eyeBlinkLeft
    blink_bs_right: float | None = None
    mar: float | None = None
    jaw_open_bs: float | None = None
    yaw: float | None = None  # degrees, camera frame (not yet relative to neutral pose)
    pitch: float | None = None
    roll: float | None = None
    brightness: float | None = None
    blur: float | None = None
    phone_conf: float = 0.0
    phone_near_face: bool = False
    speed_kmh: float | None = None  # filled from (simulated) CAN, not from vision

    def to_row(self) -> dict[str, Any]:
        return {k: _f(v) for k, v in asdict(self).items()}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "FrameSignals":
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in row.items():
            if k not in names:
                continue
            if isinstance(v, float) and math.isnan(v):
                v = None
            kw[k] = v
        kw["ts_ms"] = int(kw["ts_ms"])
        for b in ("camera_ok", "face_detected", "phone_near_face"):
            if b in kw and kw[b] is not None:
                kw[b] = bool(kw[b])
        if kw.get("phone_conf") is None:
            kw["phone_conf"] = 0.0
        if kw.get("face_quality") is None:
            kw["face_quality"] = 0.0
        return cls(**kw)


@dataclass
class AlertEvent:
    ts_ms: int
    level: RiskLevel
    reasons: list[str]
    message: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "level": self.level.value,
            "reasons": list(self.reasons),
            "message": self.message,
            "score": _f(self.score),
        }


@dataclass
class TickOutput:
    timestamp_ms: int
    trip_id: str
    driver_id: str
    # sensor / quality
    camera_ok: bool
    face_valid: bool
    face_quality: float
    calibrated: bool
    # eyes
    ear_left: float | None = None
    ear_right: float | None = None
    ear_norm: float | None = None  # EAR / personal open-eye baseline
    eye_closed: bool | None = None
    closure_ms: int = 0  # ongoing closure duration
    blink_duration_ms: int | None = None  # last completed closure
    blink_rate_per_min: float | None = None
    perclos_proxy: float | None = None
    # mouth
    mar: float | None = None
    yawn_active: bool = False
    yawns_in_window: int = 0
    # head
    head_yaw_deg: float | None = None  # relative to calibrated neutral pose
    head_pitch_deg: float | None = None
    head_roll_deg: float | None = None
    off_road_ms: int = 0
    # phone
    phone_confidence: float = 0.0
    phone_persistence: float = 0.0
    # context
    speed_kmh: float | None = None
    drive_time_min: float = 0.0
    # trip memory
    fatigue_index: float | None = None
    fatigue_trend: float | None = None  # points / minute
    # decision
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    reason_codes: list[str] = field(default_factory=list)
    alert: AlertEvent | None = None
    # runtime
    processing_fps: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, RiskLevel):
                v = v.value
            elif isinstance(v, AlertEvent):
                v = v.to_dict()
            elif isinstance(v, list):
                v = list(v)
            d[f.name] = _f(v)
        return d
