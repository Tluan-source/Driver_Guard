"""Typed configuration.

The YAML file (configs/default.yaml) and the defaults below must stay in sync; the
defaults let the package work even when installed without the repo's configs/ folder.

`apply_updates` is the single entry point for runtime (HITL) threshold changes: it only
accepts whitelisted keys (TUNABLE_KEYS) and re-validates bounds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeCfg(_M):
    target_fps: float = Field(15, gt=0)
    phone_every_n_frames: int = Field(3, ge=1)
    max_frame_gap_ms: int = Field(300, ge=0)


class ModelsCfg(_M):
    face_landmarker: str = "models/face_landmarker.task"
    phone_detector: str = "models/efficientdet_lite0.tflite"
    phone_backend: str = "mediapipe"


class QualityCfg(_M):
    min_face_width_ratio: float = 0.10
    brightness_min: float = 40
    brightness_max: float = 220
    min_blur_var: float = 20.0
    max_eye_valid_yaw_deg: float = 35
    max_eye_valid_pitch_deg: float = 30
    min_quality: float = Field(0.45, ge=0, le=1)
    degraded_after_ms: int = Field(3000, ge=0)


class CalibrationCfg(_M):
    method: str = "adaptive"
    eye_signal: str = "ear"
    prior_ear_open: float = Field(0.28, gt=0)
    prior_strength_s: float = Field(20, ge=0)
    warmup_s: float = Field(60, gt=0)
    open_percentile: float = Field(80, gt=0, lt=100)
    drift_tau_s: float = Field(600, gt=0)
    max_drift_down: float = Field(0.15, ge=0, le=0.5)
    fixed_closed_ear: float = Field(0.21, gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.method not in {"adaptive", "fixed"}:
            raise ValueError("calibration.method must be 'adaptive' or 'fixed'")
        if self.eye_signal not in {"ear", "blendshape", "fused"}:
            raise ValueError("calibration.eye_signal must be ear|blendshape|fused")
        return self


class EyeCfg(_M):
    closed_ratio_enter: float = Field(0.60, ge=0.3, le=0.9)
    closed_ratio_exit: float = Field(0.72, ge=0.3, le=0.95)
    blendshape_closed_enter: float = Field(0.55, ge=0, le=1)
    blendshape_closed_exit: float = Field(0.40, ge=0, le=1)
    max_invalid_gap_ms: int = Field(300, ge=0)
    perclos_window_s: float = Field(60, gt=0)
    perclos_min_valid_ratio: float = Field(0.5, ge=0, le=1)
    blink_rate_window_s: float = Field(60, gt=0)
    long_closure_ms: int = Field(1000, ge=300, le=5000)
    microsleep_ms: int = Field(2000, ge=500, le=10000)

    @model_validator(mode="after")
    def _check(self):
        if self.closed_ratio_exit <= self.closed_ratio_enter:
            raise ValueError("eye.closed_ratio_exit must be > closed_ratio_enter (hysteresis)")
        if self.microsleep_ms <= self.long_closure_ms:
            raise ValueError("eye.microsleep_ms must be > eye.long_closure_ms")
        return self


class YawnCfg(_M):
    mar_enter: float = 0.55
    mar_exit: float = 0.40
    min_duration_ms: int = Field(1500, ge=300, le=10000)
    window_s: float = Field(300, gt=0)


class HeadCfg(_M):
    off_road_yaw_deg: float = Field(30, gt=0, le=90)
    look_down_pitch_deg: float = Field(-20, lt=0, ge=-90)
    distraction_ms: int = Field(2500, ge=500, le=15000)
    nod_drop_deg: float = Field(15, gt=0)
    nod_window_ms: int = Field(700, gt=0)
    nod_cooldown_ms: int = Field(3000, ge=0)


class PhoneCfg(_M):
    min_confidence: float = Field(0.40, ge=0.05, le=0.99)
    window_ms: int = Field(2000, gt=0)
    persistence_ratio: float = Field(0.6, gt=0, le=1)
    min_duration_ms: int = Field(1500, ge=0)
    face_proximity: float = Field(1.6, gt=0)


class ContextCfg(_M):
    long_drive_min: float = Field(240, gt=0)
    rest_reset_min: float = Field(15, gt=0)
    circadian_low_hours: list[tuple[int, int]] = [(1, 5), (13, 15)]
    parked_speed_kmh: float = 5
    high_risk_sensitivity: float = Field(0.85, gt=0.3, le=1.0)


class RiskCfg(_M):
    weights: dict[str, float] = {"eye": 0.40, "perclos": 0.25, "yawn": 0.10, "head": 0.15, "phone": 0.10}
    perclos_normal: float = 0.08
    perclos_caution: float = Field(0.15, gt=0, lt=1)
    perclos_warning: float = Field(0.25, gt=0, lt=1)
    caution_enter: float = Field(35, ge=0, le=100)
    caution_exit: float = Field(25, ge=0, le=100)
    warning_enter: float = Field(60, ge=0, le=100)
    warning_exit: float = Field(45, ge=0, le=100)
    enter_persist_ms: int = Field(1500, ge=0)
    min_dwell_ms: int = Field(3000, ge=0)
    alert_cooldown_ms: int = Field(20000, ge=0)
    frequent_yawn_count: int = Field(2, ge=1)
    phone_can_escalate_to_critical: bool = False

    @model_validator(mode="after")
    def _check(self):
        if not (self.caution_exit < self.caution_enter <= self.warning_enter):
            raise ValueError("risk thresholds must satisfy caution_exit < caution_enter <= warning_enter")
        if not (self.warning_exit < self.warning_enter):
            raise ValueError("risk.warning_exit must be < warning_enter")
        if not (self.perclos_normal < self.perclos_caution < self.perclos_warning):
            raise ValueError("risk perclos thresholds must be increasing")
        missing = {"eye", "perclos", "yawn", "head", "phone"} - set(self.weights)
        if missing:
            raise ValueError(f"risk.weights missing {missing}")
        return self


class TripCfg(_M):
    bucket_s: float = Field(60, gt=0)
    trend_window_min: float = Field(30, gt=0)
    trend_min_buckets: int = Field(10, ge=3)
    trend_slope_caution: float = Field(0.5, gt=0)


class ApiCfg(_M):
    host: str = "127.0.0.1"
    port: int = 8000
    stream_hz: float = Field(10, gt=0, le=60)
    token_ttl_s: int = 43200
    users_file: str = "configs/users.example.yaml"


class StorageCfg(_M):
    db_path: str = "data/runtime/driverguard.db"
    retention_days: int = Field(30, ge=1)


class PrivacyCfg(_M):
    debug_recording: bool = False


class DriverGuardConfig(_M):
    runtime: RuntimeCfg = RuntimeCfg()
    models: ModelsCfg = ModelsCfg()
    quality: QualityCfg = QualityCfg()
    calibration: CalibrationCfg = CalibrationCfg()
    eye: EyeCfg = EyeCfg()
    yawn: YawnCfg = YawnCfg()
    head: HeadCfg = HeadCfg()
    phone: PhoneCfg = PhoneCfg()
    context: ContextCfg = ContextCfg()
    risk: RiskCfg = RiskCfg()
    trip: TripCfg = TripCfg()
    api: ApiCfg = ApiCfg()
    storage: StorageCfg = StorageCfg()
    privacy: PrivacyCfg = PrivacyCfg()


# Keys a fleet_manager may change at runtime (HITL threshold tuning). Everything else
# requires editing the YAML + restart, so safety-relevant structure cannot drift silently.
TUNABLE_KEYS: frozenset[str] = frozenset(
    {
        "eye.closed_ratio_enter",
        "eye.long_closure_ms",
        "eye.microsleep_ms",
        "yawn.min_duration_ms",
        "head.distraction_ms",
        "phone.min_confidence",
        "risk.perclos_caution",
        "risk.perclos_warning",
        "risk.alert_cooldown_ms",
    }
)


def _deep_update(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _dotted_to_nested(updates: dict[str, Any]) -> dict:
    nested: dict = {}
    for key, value in updates.items():
        cur = nested
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return nested


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> DriverGuardConfig:
    """Load YAML config (defaults to configs/default.yaml if present) + dotted overrides."""
    data: dict = {}
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif path:
        raise FileNotFoundError(p)
    if overrides:
        data = _deep_update(data, _dotted_to_nested(overrides))
    return DriverGuardConfig.model_validate(data)


def apply_updates(
    cfg: DriverGuardConfig, updates: dict[str, Any], allowed: frozenset[str] | None = TUNABLE_KEYS
) -> DriverGuardConfig:
    """Return a NEW validated config with dotted-key updates applied.

    Raises KeyError for non-whitelisted keys and pydantic.ValidationError for bad values.
    """
    if allowed is not None:
        bad = set(updates) - set(allowed)
        if bad:
            raise KeyError(f"Keys not tunable at runtime: {sorted(bad)}")
    merged = _deep_update(cfg.model_dump(), _dotted_to_nested(updates))
    return DriverGuardConfig.model_validate(merged)


def get_dotted(cfg: DriverGuardConfig, key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        cur = getattr(cur, part)
    return cur


def tunable_schema(cfg: DriverGuardConfig) -> list[dict[str, Any]]:
    """Current value + bounds of each HITL-tunable key (for the dashboard config panel)."""
    out = []
    for key in sorted(TUNABLE_KEYS):
        section, name = key.split(".")
        model = getattr(cfg, section)
        info = type(model).model_fields[name]
        lo = hi = None
        for m in info.metadata:
            lo = getattr(m, "ge", getattr(m, "gt", lo))
            hi = getattr(m, "le", getattr(m, "lt", hi))
        out.append({"key": key, "value": getattr(model, name), "min": lo, "max": hi,
                    "type": type(getattr(model, name)).__name__})
    return out


def resolve_path(p: str | Path) -> Path:
    """Resolve a repo-relative path (models/, data/) regardless of the working directory."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p
