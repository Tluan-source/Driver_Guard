"""Temporal trackers: eye state / blink / PERCLOS-proxy, yawn, head attention, phone use.

Naming note (GVHD review 9.1): classic PERCLOS is measured with IR eye trackers as the time
fraction with >= 80% pupil occlusion. We estimate eye closure from landmark EAR relative to a
personal baseline, so the quantity is called PERCLOS-proxy everywhere in code and reports.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..config import EyeCfg, HeadCfg, PhoneCfg, YawnCfg
from ..schemas import Reason
from .primitives import HysteresisSwitch, PersistenceTimer, TimeWindow


# --------------------------------------------------------------------------- eyes
@dataclass
class EyeState:
    closed: bool | None
    closure_ms: int  # ongoing closure duration (0 if open/unknown)
    completed_closure_ms: int | None  # a closure that ended at this tick
    last_blink_ms: int | None
    perclos_proxy: float | None
    perclos_coverage: float
    blink_rate_per_min: float | None
    mean_blink_ms: float | None


class EyeStateTracker:
    """Input: `openness` = EAR / personal open-eye level (ratio; ~1.0 open, <0.6 closed), or
    None when the eyes are not reliably visible."""

    def __init__(self, cfg: EyeCfg, max_gap_ms: float):
        self.cfg = cfg
        self.switch = HysteresisSwitch(cfg.closed_ratio_enter, cfg.closed_ratio_exit, on_when_below=True)
        self.perclos_win: TimeWindow[bool] = TimeWindow(cfg.perclos_window_s * 1000, max_gap_ms)
        self.valid_win: TimeWindow[bool] = TimeWindow(cfg.blink_rate_window_s * 1000, max_gap_ms)
        self.closures: deque[tuple[int, int]] = deque()  # (end_ts, duration_ms)
        self._closed_since: int | None = None
        self._last_valid_ts: int | None = None
        self.last_blink_ms: int | None = None

    def update(self, ts: int, openness: float | None) -> EyeState:
        cfg = self.cfg
        # thresholds can be changed at runtime (HITL) -> refresh every tick
        self.switch.enter, self.switch.exit = cfg.closed_ratio_enter, cfg.closed_ratio_exit
        completed = None
        if openness is not None:
            closed: bool | None = self.switch.update(openness)
            self._last_valid_ts = ts
            if closed and self._closed_since is None:
                self._closed_since = ts
            elif not closed and self._closed_since is not None:
                completed = ts - self._closed_since
                self.closures.append((ts, completed))
                self.last_blink_ms = completed
                self._closed_since = None
            self.perclos_win.add(ts, bool(closed))
            self.valid_win.add(ts, True)
        else:
            gap_ok = (self._last_valid_ts is not None and ts - self._last_valid_ts <= cfg.max_invalid_gap_ms)
            if self._closed_since is not None and gap_ok:
                closed = True  # brief dropout in the middle of a closure: keep the timer running
            else:
                closed = None
                if self._closed_since is not None:  # lost the eyes -> abort, never "assume closed"
                    self._closed_since = None
                    self.switch.on = False
            self.perclos_win.add(ts, None)
            self.valid_win.add(ts, None)

        # windowed statistics
        win_lo = ts - cfg.blink_rate_window_s * 1000
        while self.closures and self.closures[0][0] < min(win_lo, ts - cfg.perclos_window_s * 1000):
            self.closures.popleft()
        perclos, cov = self.perclos_win.time_fraction(lambda v: v, ts)
        if cov < cfg.perclos_min_valid_ratio:
            perclos = None
        blinks = [d for (t, d) in self.closures if t >= win_lo and d < cfg.long_closure_ms]
        valid_min = self.valid_win.valid_time_ms(ts) / 60000.0
        rate = len(blinks) / valid_min if valid_min >= 0.25 else None
        mean_blink = sum(blinks) / len(blinks) if blinks else None
        closure_ms = ts - self._closed_since if self._closed_since is not None else 0
        return EyeState(closed, int(closure_ms), completed, self.last_blink_ms, perclos, cov, rate, mean_blink)


# --------------------------------------------------------------------------- yawn
@dataclass
class YawnState:
    active: bool
    new_yawn: bool
    yawns_in_window: int


class YawnTracker:
    """MAR above threshold for >= min_duration_ms. Talking opens the mouth briefly; a yawn is
    long. Yawn is WEAK evidence (GVHD 9.2) — it only contributes to CAUTION."""

    def __init__(self, cfg: YawnCfg):
        self.cfg = cfg
        self.switch = HysteresisSwitch(cfg.mar_enter, cfg.mar_exit, on_when_below=False)
        self._open_since: int | None = None
        self._counted = False
        self.yawns: deque[int] = deque()

    def update(self, ts: int, mar: float | None) -> YawnState:
        cfg = self.cfg
        new = False
        on = self.switch.update(mar) if mar is not None else False
        if on:
            if self._open_since is None:
                self._open_since = ts
            if not self._counted and ts - self._open_since >= cfg.min_duration_ms:
                self._counted = True
                new = True
                self.yawns.append(ts)
        else:
            self._open_since = None
            self._counted = False
        while self.yawns and self.yawns[0] < ts - cfg.window_s * 1000:
            self.yawns.popleft()
        return YawnState(active=on and self._counted, new_yawn=new, yawns_in_window=len(self.yawns))


# --------------------------------------------------------------------------- head
@dataclass
class HeadState:
    off_road_ms: int
    distracted: bool
    reason: Reason | None
    nod: bool
    nods_in_window: int


class HeadTracker:
    """Eyes-off-road / looking-down from pose RELATIVE to the calibrated neutral pose.
    A quick mirror check (< distraction_ms) never alerts."""

    def __init__(self, cfg: HeadCfg, window_s: float = 300):
        self.cfg = cfg
        self.timer = PersistenceTimer(cfg.distraction_ms, off_ms=500)
        self._hist: deque[tuple[int, float]] = deque()
        self._last_nod = -(10**12)
        self.nods: deque[int] = deque()
        self.window_ms = window_s * 1000
        self._reason: Reason | None = None

    def update(self, ts: int, yaw_rel: float | None, pitch_rel: float | None) -> HeadState:
        cfg = self.cfg
        self.timer.on_ms = cfg.distraction_ms
        nod = False
        if yaw_rel is None or pitch_rel is None:
            off = False
        else:
            looking_down = pitch_rel < cfg.look_down_pitch_deg
            turned = abs(yaw_rel) > cfg.off_road_yaw_deg
            off = looking_down or turned
            if off:
                self._reason = Reason.LOOKING_DOWN if looking_down else Reason.EYES_OFF_ROAD
            # nod: fast pitch drop within nod_window_ms
            self._hist.append((ts, pitch_rel))
            while self._hist and self._hist[0][0] < ts - cfg.nod_window_ms:
                self._hist.popleft()
            peak = max(p for _, p in self._hist)
            if (peak - pitch_rel >= cfg.nod_drop_deg and pitch_rel < -cfg.nod_drop_deg / 2
                    and ts - self._last_nod >= cfg.nod_cooldown_ms):
                nod = True
                self._last_nod = ts
                self.nods.append(ts)
        while self.nods and self.nods[0] < ts - self.window_ms:
            self.nods.popleft()
        active = self.timer.update(off, ts)
        return HeadState(self.timer.held_ms(ts) if off else 0, active,
                         self._reason if active else None, nod, len(self.nods))


# --------------------------------------------------------------------------- phone
@dataclass
class PhoneState:
    persistence: float
    active: bool


class PhoneTracker:
    def __init__(self, cfg: PhoneCfg, max_gap_ms: float):
        self.cfg = cfg
        self.win: TimeWindow[bool] = TimeWindow(cfg.window_ms, max_gap_ms)
        self.timer = PersistenceTimer(cfg.min_duration_ms, off_ms=1000)

    def update(self, ts: int, conf: float, near_face: bool, camera_ok: bool) -> PhoneState:
        cfg = self.cfg
        self.timer.on_ms = cfg.min_duration_ms
        self.win.add(ts, (conf >= cfg.min_confidence and near_face) if camera_ok else None)
        ratio, _ = self.win.time_fraction(lambda v: v, ts)
        ratio = ratio or 0.0
        active = self.timer.update(ratio >= cfg.persistence_ratio, ts)
        return PhoneState(ratio, active)
