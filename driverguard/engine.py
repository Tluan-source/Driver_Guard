"""DriverGuardEngine: FrameSignals -> TickOutput (temporal + calibration + risk + trip memory).

Pure, camera-free and deterministic: the same FrameSignals sequence always yields the same
TickOutput sequence. This lets us
  * replay features extracted on Kaggle (parquet) to tune thresholds on VALIDATION subjects,
  * run the golden test vectors that the Android (Kotlin) port must reproduce,
  * unit-test every safety rule without MediaPipe.
"""
from __future__ import annotations

import uuid

from .alerts.explain import explain_vi
from .calibration import PoseBaseline, make_eye_baseline
from .config import DriverGuardConfig
from .risk.context import TripContext
from .risk.evidence import assess
from .risk.state_machine import AlertPolicy, RiskStateMachine
from .schemas import FrameSignals, Reason, RiskLevel, TickOutput
from .temporal.primitives import PersistenceTimer
from .temporal.trackers import EyeStateTracker, HeadTracker, PhoneTracker, YawnTracker
from .trip.memory import TripMemory


def _mean(*xs):
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


class DriverGuardEngine:
    def __init__(self, cfg: DriverGuardConfig, driver_id: str = "anon", trip_id: str | None = None,
                 trip_start_epoch_s: float | None = None, tz_offset_h: float | None = None,
                 explain=explain_vi):
        self.cfg = cfg
        self.driver_id = driver_id
        self.trip_id = trip_id or uuid.uuid4().hex[:12]
        gap = cfg.runtime.max_frame_gap_ms
        self.eye_base = make_eye_baseline(cfg.calibration, cfg.eye)
        self.pose_base = PoseBaseline(cfg.calibration.warmup_s)
        self.eye = EyeStateTracker(cfg.eye, gap)
        self.yawn = YawnTracker(cfg.yawn)
        self.head = HeadTracker(cfg.head)
        self.phone = PhoneTracker(cfg.phone, gap)
        self.context = TripContext(cfg.context, trip_start_epoch_s, tz_offset_h)
        self.memory = TripMemory(cfg.trip, cfg.eye.long_closure_ms)
        self.sm = RiskStateMachine(cfg)
        self.alerts = AlertPolicy(cfg, explain)
        self.degraded_timer = PersistenceTimer(cfg.quality.degraded_after_ms, off_ms=0)
        self.last: TickOutput | None = None

    # ------------------------------------------------------------------ HITL
    def reconfigure(self, cfg: DriverGuardConfig) -> None:
        """Swap thresholds at runtime WITHOUT losing temporal state (trackers re-read cfg)."""
        self.cfg = cfg
        self.eye.cfg, self.yawn.cfg, self.head.cfg, self.phone.cfg = cfg.eye, cfg.yawn, cfg.head, cfg.phone
        self.context.cfg, self.memory.cfg = cfg.context, cfg.trip
        self.memory.long_closure_ms = cfg.eye.long_closure_ms
        self.sm.cfg = self.alerts.cfg = cfg
        self.degraded_timer.on_ms = cfg.quality.degraded_after_ms

    # ------------------------------------------------------------------ helpers
    def _openness(self, s: FrameSignals) -> float | None:
        cal, e = self.cfg.calibration, self.cfg.eye
        ear = _mean(s.ear_left, s.ear_right)
        ratio = self.eye_base.ratio(ear) if ear is not None else None
        bs = _mean(s.blink_bs_left, s.blink_bs_right)
        bs_ratio = None
        if bs is not None and e.blendshape_closed_enter > e.blendshape_closed_exit:
            # map blendshape closedness onto the EAR-ratio scale so one hysteresis serves both
            k = (e.closed_ratio_exit - e.closed_ratio_enter) / (e.blendshape_closed_enter - e.blendshape_closed_exit)
            bs_ratio = e.closed_ratio_enter + (e.blendshape_closed_enter - bs) * k
        if cal.eye_signal == "ear":
            return ratio
        if cal.eye_signal == "blendshape":
            return bs_ratio
        return _mean(ratio, bs_ratio)

    # ------------------------------------------------------------------ main step
    def step(self, s: FrameSignals) -> TickOutput:
        cfg, ts = self.cfg, s.ts_ms
        q = cfg.quality
        face_valid = bool(s.camera_ok and s.face_detected and s.face_quality >= q.min_quality)

        # neutral pose + relative angles
        self.pose_base.update(ts, s.yaw, s.pitch, usable=face_valid)
        yaw_rel, pitch_rel = self.pose_base.relative(s.yaw, s.pitch) if face_valid else (None, None)
        eyes_valid = face_valid and (yaw_rel is None or (abs(yaw_rel) <= q.max_eye_valid_yaw_deg and
                                                          abs(pitch_rel) <= q.max_eye_valid_pitch_deg))
        frontal = eyes_valid and (yaw_rel is None or (abs(yaw_rel) < 15 and abs(pitch_rel) < 15))
        ear = _mean(s.ear_left, s.ear_right)
        self.eye_base.update(ts, ear, usable=frontal)

        eye = self.eye.update(ts, self._openness(s) if eyes_valid else None)
        yawn = self.yawn.update(ts, s.mar if face_valid else None)
        head = self.head.update(ts, yaw_rel, pitch_rel)
        phone = self.phone.update(ts, s.phone_conf, s.phone_near_face, s.camera_ok)
        ctx = self.context.update(ts, s.speed_kmh)

        ev = assess(cfg, eye, yawn, head, phone, ctx, self.memory.trend_rising)

        invalid_now = not face_valid
        degraded = self.degraded_timer.update(invalid_now, ts)
        if not s.camera_ok:
            sreason = Reason.CAMERA_LOST
        elif not s.face_detected:
            sreason = Reason.FACE_NOT_VISIBLE
        else:
            sreason = Reason.LOW_FACE_QUALITY
        level, reasons = self.sm.update(ts, ev, degraded, sreason, invalid_now)
        alert = self.alerts.update(ts, level, reasons, ev.score)

        self.memory.update(ts, face_valid, level == RiskLevel.SENSOR_DEGRADED, eye.perclos_proxy,
                           eye.completed_closure_ms, yawn.new_yawn,
                           alert.level.value if alert else None, max(level.severity, 0))

        info = [r.value for r in ev.info]
        if not self.eye_base.calibrated:
            info.append(Reason.CALIBRATING.value)
        codes = reasons + [r for r in info if r not in reasons]

        tick = TickOutput(
            timestamp_ms=ts, trip_id=self.trip_id, driver_id=self.driver_id,
            camera_ok=s.camera_ok, face_valid=face_valid, face_quality=s.face_quality,
            calibrated=self.eye_base.calibrated,
            ear_left=s.ear_left, ear_right=s.ear_right,
            ear_norm=(self.eye_base.ratio(ear) if (ear is not None and eyes_valid) else None),
            eye_closed=eye.closed, closure_ms=eye.closure_ms, blink_duration_ms=eye.last_blink_ms,
            blink_rate_per_min=eye.blink_rate_per_min, perclos_proxy=eye.perclos_proxy,
            mar=s.mar if face_valid else None, yawn_active=yawn.active, yawns_in_window=yawn.yawns_in_window,
            head_yaw_deg=yaw_rel, head_pitch_deg=pitch_rel,
            head_roll_deg=s.roll if face_valid else None, off_road_ms=head.off_road_ms,
            phone_confidence=s.phone_conf, phone_persistence=phone.persistence,
            speed_kmh=s.speed_kmh, drive_time_min=ctx.drive_time_min,
            fatigue_index=self.memory.fatigue_index, fatigue_trend=self.memory.trend,
            risk_score=ev.score, risk_level=level, reason_codes=codes, alert=alert,
        )
        self.last = tick
        return tick
