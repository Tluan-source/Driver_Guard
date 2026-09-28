"""Deterministic risk state machine + alert policy.

    NORMAL -> CAUTION -> WARNING -> CRITICAL        (+ SENSOR_DEGRADED, separate)

* Upward: immediate candidates raise the level at once; persistent candidates must hold for
  `enter_persist_ms`.
* Downward (hysteresis): the target must stay below the current level for `min_dwell_ms`
  AND the score must be below the exit threshold of the current level.
* SENSOR_DEGRADED: face/camera unusable for >= degraded_after_ms. Before that timeout the
  previous level is HELD (a short dropout never resets to NORMAL). Degraded is never NORMAL:
  "cannot assess" is a different statement from "driver is fine".
"""
from __future__ import annotations

from ..config import DriverGuardConfig
from ..schemas import AlertEvent, Reason, RiskLevel
from .evidence import Evidence

_ESCALATING = (RiskLevel.CAUTION, RiskLevel.WARNING, RiskLevel.CRITICAL)


class RiskStateMachine:
    def __init__(self, cfg: DriverGuardConfig):
        self.cfg = cfg
        self.level = RiskLevel.NORMAL
        self.reasons: list[str] = []
        self._since: dict[RiskLevel, int] = {}  # persistent candidate level -> first ts seen
        self._below_since: int | None = None
        self._degraded = False

    def _exit_threshold(self, level: RiskLevel) -> float:
        r = self.cfg.risk
        return r.caution_exit if level == RiskLevel.CAUTION else r.warning_exit

    def update(self, ts: int, ev: Evidence, sensor_degraded: bool, sensor_reason: Reason | None,
               sensor_invalid_now: bool) -> tuple[RiskLevel, list[str]]:
        r = self.cfg.risk
        if sensor_degraded:
            self._degraded = True
            self.level = RiskLevel.SENSOR_DEGRADED
            self.reasons = [sensor_reason.value] if sensor_reason else []
            self._since.clear()
            self._below_since = None
            return self.level, list(self.reasons)
        if self._degraded:  # recovered -> restart evaluation from scratch
            self._degraded = False
            self.level = RiskLevel.NORMAL
            self.reasons = []
        if sensor_invalid_now:
            # short dropout: hold the current level and reasons, don't accumulate timers
            return self.level, list(self.reasons)

        # --- persistence bookkeeping for non-immediate candidates
        persist_levels = {c.level for c in ev.candidates if not c.immediate}
        for lvl in (RiskLevel.CAUTION, RiskLevel.WARNING, RiskLevel.CRITICAL):
            if any(pl.severity >= lvl.severity for pl in persist_levels):
                self._since.setdefault(lvl, ts)
            else:
                self._since.pop(lvl, None)

        target = RiskLevel.NORMAL
        for c in ev.candidates:
            if c.immediate and c.level.severity > target.severity:
                target = c.level
        for lvl, t0 in self._since.items():
            if ts - t0 >= r.enter_persist_ms and lvl.severity > target.severity:
                target = lvl

        cur = self.level
        if target.severity > cur.severity:
            self.level = target
            self._below_since = None
        elif target.severity < cur.severity:
            if self._below_since is None:
                self._below_since = ts
            if ts - self._below_since >= r.min_dwell_ms and ev.score < self._exit_threshold(cur):
                self.level = target
                self._below_since = None
        else:
            self._below_since = None

        active = [c for c in ev.candidates if c.level.severity >= max(self.level.severity, 1)]
        if active:
            active.sort(key=lambda c: -c.level.severity)
            seen: list[str] = []
            for c in active:
                if c.reason.value not in seen:
                    seen.append(c.reason.value)
            self.reasons = seen
        elif self.level == RiskLevel.NORMAL:
            self.reasons = []
        # else: level held by dwell/hysteresis -> keep the last reasons (sticky explanation)
        return self.level, list(self.reasons)


class AlertPolicy:
    """Turns level changes into alert events (sound / visual / log) with cooldown."""

    def __init__(self, cfg: DriverGuardConfig, explain):
        self.cfg = cfg
        self.explain = explain
        self._prev = RiskLevel.NORMAL
        self._last_alert_ts = -(10**12)
        self._last_reasons: set[str] = set()

    def update(self, ts: int, level: RiskLevel, reasons: list[str], score: float) -> AlertEvent | None:
        prev, self._prev = self._prev, level
        alert = False
        if level == RiskLevel.SENSOR_DEGRADED:
            alert = prev != RiskLevel.SENSOR_DEGRADED
        elif level in _ESCALATING:
            # a genuinely new reason (not the aggregate score) may re-alert after a short gap
            new_reason = bool(set(reasons) - self._last_reasons - {Reason.RISK_SCORE.value})
            since = ts - self._last_alert_ts
            escalated = prev == RiskLevel.SENSOR_DEGRADED or level.severity > prev.severity
            cooled = since >= self.cfg.risk.alert_cooldown_ms
            reminder = level.severity >= RiskLevel.WARNING.severity and (
                cooled or (new_reason and since >= min(self.cfg.risk.alert_cooldown_ms, 5000)))
            if escalated or reminder:
                alert = True
        if not alert:
            if level == RiskLevel.NORMAL:
                self._last_reasons = set()
            return None
        self._last_alert_ts = ts
        self._last_reasons = set(reasons)
        return AlertEvent(ts, level, list(reasons), self.explain(level, reasons), score)
