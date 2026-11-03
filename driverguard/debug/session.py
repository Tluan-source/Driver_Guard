"""DebugSession: frame/FrameSignals -> TickOutput + per-layer trace (LayerDebugFrame).

LayerDebugFrame (JSON sent to ui/test-console every tick):

    {
      "type": "frame", "seq": 12, "t_ms": 803, "source": "webcam" | "synthetic:drowsy",
      "fps": 14.8, "timing_ms": {"perception", "face", "phone", "calibration", "temporal",
                                 "evidence", "state_machine", "output", "total"},
      "overlay": {...} | null,        # webcam only: face box + eye/mouth key points (normalised)
      "layers": {"perception", "calibration", "temporal", "evidence", "state_machine", "output"}
    }

Privacy: `overlay` holds 20 normalised key points so the console can draw where the model is
looking. It is produced ONLY by this local debug session (loopback server) and is never stored
or forwarded; the production TickOutput (layers.output.tick) still passes assert_metadata_only.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DriverGuardConfig, load_config, resolve_path
from ..engine import DriverGuardEngine
from ..pipeline import RuntimeStats
from ..privacy import PrivacyViolation, assert_metadata_only
from ..schemas import FrameSignals, TickOutput

LAYER_ORDER = ("perception", "calibration", "temporal", "evidence", "state_machine", "output")

# MediaPipe Face Mesh indices used for the overlay (same points the EAR/MAR formulas use).
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)
_LEFT_EYE = (362, 385, 387, 263, 373, 380)
_MOUTH = (78, 81, 13, 311, 308, 402, 14, 178)


def _r(x: Any, nd: int = 4) -> Any:
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)
    return x


def _plain(obj: Any) -> Any:
    """Dataclass / enum / numpy -> JSON-safe python."""
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "value") and not isinstance(obj, (int, float, str)):  # Enum
        return obj.value
    return _r(obj)


class _Timed:
    """Wraps a face landmarker / phone detector: remembers the last result and its latency."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None
        self.ms = 0.0
        self.ran = False

    def detect(self, image_bgr, ts_ms):
        t0 = time.perf_counter()
        self.last = self.inner.detect(image_bgr, ts_ms)
        self.ms = (time.perf_counter() - t0) * 1000.0
        self.ran = True
        return self.last

    def close(self):
        self.inner.close()


class DebugSession:
    def __init__(self, config_path: str | Path | None = None, overrides: dict[str, Any] | None = None):
        self.config_path = config_path
        self.base_overrides = dict(overrides or {})
        self.calibration_method = "adaptive"
        self.fast_warmup = False
        self.perception = None
        self.perception_error: str | None = None
        self._face: _Timed | None = None
        self._phone: _Timed | None = None
        self.speed_source = None
        self.live_model = None
        self.live_model_error: str | None = None
        self.cfg: DriverGuardConfig
        self.engine: DriverGuardEngine
        self.reset()

    # ------------------------------------------------------------------ setup
    def _make_cfg(self) -> DriverGuardConfig:
        ov = dict(self.base_overrides)
        ov["calibration.method"] = self.calibration_method
        if self.fast_warmup:
            ov["calibration.warmup_s"] = 15
            ov["calibration.prior_strength_s"] = 5
        return load_config(self.config_path, ov)

    def reset(self, calibration_method: str | None = None, fast_warmup: bool | None = None) -> None:
        """New engine (fresh calibration + temporal state). Perception models are kept."""
        if calibration_method is not None:
            if calibration_method not in ("adaptive", "fixed"):
                raise ValueError("calibration_method must be adaptive|fixed")
            self.calibration_method = calibration_method
        if fast_warmup is not None:
            self.fast_warmup = bool(fast_warmup)
        self.cfg = self._make_cfg()
        self.engine = DriverGuardEngine(self.cfg, driver_id="TEST")
        self.engine.trace_enabled = True
        self.stats = RuntimeStats()
        self.seq = 0
        self.t0_ms: int | None = None
        if getattr(self, "live_model", None) is not None:
            self.live_model.reset()

    def set_model(self, model_id: str | None) -> str | None:
        """Attach (or detach with None) a learned camera head; it runs beside the rule engine."""
        self.live_model, self.live_model_error = None, None
        if not model_id:
            return None
        try:
            from .model_lab import LiveModel

            self.live_model = LiveModel(model_id, self.cfg)
        except Exception as e:  # noqa: BLE001 — surfaced to the UI as text
            self.live_model_error = f"{type(e).__name__}: {e}"
        return self.live_model_error

    def ensure_perception(self) -> str | None:
        """Load MediaPipe models once. Returns an error message (Vietnamese) or None."""
        if self.perception is not None:
            return None
        try:
            from ..perception.extractor import PerceptionExtractor
            from ..perception.face import MediaPipeFaceLandmarker
            from ..perception.phone import MediaPipePhoneDetector, NullPhoneDetector

            face_path = resolve_path(self.cfg.models.face_landmarker)
            if not Path(face_path).exists():
                raise FileNotFoundError(f"Thiếu {face_path}. Chạy: python scripts/data/download_models.py")
            self._face = _Timed(MediaPipeFaceLandmarker(face_path))
            phone_path = resolve_path(self.cfg.models.phone_detector)
            if self.cfg.models.phone_backend == "mediapipe" and Path(phone_path).exists():
                self._phone = _Timed(MediaPipePhoneDetector(phone_path))
            else:
                self._phone = _Timed(NullPhoneDetector())
            self.perception = PerceptionExtractor(self.cfg, self._face, self._phone)
            from ..vehicle import SimulatedCanBus

            self.speed_source = SimulatedCanBus("highway")
            self.perception_error = None
        except Exception as e:  # noqa: BLE001 — surfaced to the UI as text
            self.perception = None
            self.perception_error = f"{type(e).__name__}: {e}"
        return self.perception_error

    def close(self) -> None:
        if self.perception is not None:
            self.perception.close()
            self.perception = None

    def info(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "type": "info",
            "calibration_method": self.calibration_method,
            "fast_warmup": self.fast_warmup,
            "warmup_s": c.calibration.warmup_s,
            "target_fps": c.runtime.target_fps,
            "perception_loaded": self.perception is not None,
            "perception_error": self.perception_error,
            "model_id": self.live_model.model.id if self.live_model is not None else None,
            "model_error": self.live_model_error,
            "thresholds": {
                "closed_ratio_enter": c.eye.closed_ratio_enter, "closed_ratio_exit": c.eye.closed_ratio_exit,
                "long_closure_ms": c.eye.long_closure_ms, "microsleep_ms": c.eye.microsleep_ms,
                "perclos_caution": c.risk.perclos_caution, "perclos_warning": c.risk.perclos_warning,
                "caution_enter": c.risk.caution_enter, "caution_exit": c.risk.caution_exit,
                "warning_enter": c.risk.warning_enter, "warning_exit": c.risk.warning_exit,
                "enter_persist_ms": c.risk.enter_persist_ms, "min_dwell_ms": c.risk.min_dwell_ms,
                "alert_cooldown_ms": c.risk.alert_cooldown_ms,
                "off_road_yaw_deg": c.head.off_road_yaw_deg, "look_down_pitch_deg": c.head.look_down_pitch_deg,
                "distraction_ms": c.head.distraction_ms, "mar_enter": c.yawn.mar_enter,
                "phone_min_confidence": c.phone.min_confidence,
            },
            "weights": dict(c.risk.weights),
        }

    # ------------------------------------------------------------------ processing
    def process_jpeg(self, data: bytes, ts_ms: int) -> dict[str, Any]:
        import cv2

        t0 = time.perf_counter()
        err = self.ensure_perception()
        if err:
            return {"type": "error", "message": err}
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        camera_ok = img is not None
        self._face.ran = self._phone.ran = False
        t_p0 = time.perf_counter()
        sig = self.perception.process(img, ts_ms, camera_ok)
        t_p1 = time.perf_counter()
        sig.speed_kmh = self.speed_source.read_speed(ts_ms) if self.speed_source else None
        overlay = self._overlay(sig)
        timing = {
            "decode": (t_p0 - t0) * 1000.0,
            "perception": (t_p1 - t_p0) * 1000.0,
            "face": self._face.ms if self._face.ran else 0.0,
            "phone": self._phone.ms,  # last run (detector runs every N frames)
            "phone_ran": 1.0 if self._phone.ran else 0.0,
        }
        return self._step(sig, "webcam", timing, overlay, t0)

    def process_signals(self, sig: FrameSignals, source: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        return self._step(sig, source, {"decode": 0.0, "perception": 0.0, "face": 0.0, "phone": None,
                                         "phone_ran": 0.0}, None, t0)

    def _overlay(self, sig: FrameSignals) -> dict[str, Any] | None:
        out: dict[str, Any] = {"face_box": None, "eye_l": None, "eye_r": None, "mouth": None, "phone_box": None}
        obs = self._face.last if self._face else None
        if sig.face_detected and obs is not None:
            pts = obs.landmarks_norm
            out["face_box"] = [_r(sig.face_x0), _r(sig.face_y0), _r(sig.face_x1), _r(sig.face_y1)]
            out["eye_l"] = [[_r(pts[i][0]), _r(pts[i][1])] for i in _LEFT_EYE]
            out["eye_r"] = [[_r(pts[i][0]), _r(pts[i][1])] for i in _RIGHT_EYE]
            out["mouth"] = [[_r(pts[i][0]), _r(pts[i][1])] for i in _MOUTH]
        ph = self._phone.last if self._phone else None
        if ph is not None and getattr(ph, "bbox_norm", None) is not None and ph.confidence > 0:
            out["phone_box"] = [_r(v) for v in ph.bbox_norm]
            out["phone_conf"] = _r(ph.confidence, 3)
        return out

    def _step(self, sig: FrameSignals, source: str, timing: dict, overlay, t0: float) -> dict[str, Any]:
        if self.t0_ms is None:
            self.t0_ms = sig.ts_ms
        tick = self.engine.step(sig)
        tr = self.engine.last_trace or {}
        model = self.live_model.update(sig) if self.live_model is not None else None
        timing.update(tr.get("timing_ms", {}))
        total = (time.perf_counter() - t0) * 1000.0
        timing["total"] = total
        tick.processing_fps = round(self.stats.tick(total), 2)
        tick.latency_ms = round(total, 2)
        self.seq += 1
        return {
            "type": "frame",
            "seq": self.seq,
            "t_ms": sig.ts_ms - self.t0_ms,
            "source": source,
            "fps": tick.processing_fps,
            "timing_ms": {k: _r(v, 3) for k, v in timing.items()},
            "overlay": overlay,
            "layers": self._layers(sig, tick, tr),
            "model": model,
        }

    def _layers(self, sig: FrameSignals, tick: TickOutput, tr: dict) -> dict[str, Any]:
        cfg = self.cfg
        eb, pb = tr["eye_base"], tr["pose_base"]
        eye, yawn, head, phone, ctx, ev = tr["eye"], tr["yawn"], tr["head"], tr["phone"], tr["ctx"], tr["ev"]
        ear = tr["ear"]
        open_level = float(eb.open_level)

        perception = sig.to_row()
        perception["ear"] = _r(ear)

        calibration = {
            "method": self.calibration_method,
            "eye_signal": cfg.calibration.eye_signal,
            "calibrated": bool(eb.calibrated),
            "progress": _r(getattr(eb, "progress", 1.0), 3),
            "warmup_s": cfg.calibration.warmup_s,
            "open_level": _r(open_level),
            "calibrated_level": _r(getattr(eb, "calibrated_level", None)),
            "closed_threshold_ear": _r(open_level * cfg.eye.closed_ratio_enter),
            "open_threshold_ear": _r(open_level * cfg.eye.closed_ratio_exit),
            "ear": _r(ear),
            "ear_norm": _r(tick.ear_norm),
            "openness": _r(tr["openness"]),
            "face_valid": tr["face_valid"],
            "eyes_valid": tr["eyes_valid"],
            "frontal": tr["frontal"],
            "pose_calibrated": bool(pb.calibrated),
            "neutral_yaw": _r(pb.neutral_yaw, 2),
            "neutral_pitch": _r(pb.neutral_pitch, 2),
            "yaw_rel": _r(tr["yaw_rel"], 2),
            "pitch_rel": _r(tr["pitch_rel"], 2),
        }

        temporal = {
            "eye": _plain(eye),
            "yawn": _plain(yawn),
            "head": _plain(head),
            "phone": _plain(phone),
            "context": _plain(ctx),
        }

        evidence = {
            "components": {k: _r(v) for k, v in ev.components.items()},
            "weights": dict(cfg.risk.weights),
            "score": _r(ev.score, 2),
            "candidates": [{"level": c.level.value, "reason": c.reason.value, "immediate": c.immediate}
                           for c in ev.candidates],
            "info": [r.value for r in ev.info],
            "sensitivity": _r(ctx.sensitivity, 3),
        }

        ts = sig.ts_ms
        persist = {lvl: ts - t for lvl, t in tr["sm_target_since"].items()}
        below = tr["sm_below_since"]
        alert = tr["alert"]
        state_machine = {
            "level": tr["level"].value,
            "reasons": tr["reasons"],
            "degraded": bool(tr["degraded"]),
            "sensor_invalid_now": bool(tr["invalid_now"]),
            "persist_ms": persist,
            "enter_persist_ms": cfg.risk.enter_persist_ms,
            "below_ms": (ts - below) if below is not None else None,
            "min_dwell_ms": cfg.risk.min_dwell_ms,
            "alert": alert.to_dict() if alert is not None else None,
        }

        tick_d = tick.to_dict()
        try:
            assert_metadata_only(tick_d)
            privacy_ok, privacy_err = True, None
        except PrivacyViolation as e:
            privacy_ok, privacy_err = False, str(e)
        output = {
            "tick": tick_d,
            "privacy_ok": privacy_ok,
            "privacy_error": privacy_err,
            "fatigue_index": _r(tick.fatigue_index, 3),
            "fatigue_trend": _r(tick.fatigue_trend, 3),
        }
        return {"perception": perception, "calibration": calibration, "temporal": temporal,
                "evidence": evidence, "state_machine": state_machine, "output": output}
