"""Trip context: continuous driving time, time of day, vehicle speed.

This is the deterministic version of "agent lập luận mức rủi ro theo ngữ cảnh" in the brief:
context never raises an alert on its own (except LONG_DRIVE -> CAUTION), it changes how
sensitive the rules are, and it suppresses alerts that make no sense (phone use / eyes closed
while parked).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..config import ContextCfg


@dataclass
class ContextState:
    speed_kmh: float | None
    parked: bool | None  # None = speed unknown
    drive_time_min: float
    local_hour: int
    long_drive: bool
    circadian_low: bool
    sensitivity: float  # multiply thresholds by this (<1 = more sensitive)


class TripContext:
    def __init__(self, cfg: ContextCfg, trip_start_epoch_s: float | None = None, tz_offset_h: float | None = None):
        """tz_offset_h=None -> the machine's local time zone; pass e.g. 7 for reproducible replays/tests."""
        self.cfg = cfg
        self.start_epoch = trip_start_epoch_s if trip_start_epoch_s is not None else time.time()
        self.tz = timezone(timedelta(hours=tz_offset_h)) if tz_offset_h is not None else None
        self._last_ts: int | None = None
        self._drive_ms = 0.0
        self._stopped_ms = 0.0

    def update(self, ts_ms: int, speed_kmh: float | None) -> ContextState:
        cfg = self.cfg
        dt = 0 if self._last_ts is None else max(0, ts_ms - self._last_ts)
        self._last_ts = ts_ms
        parked = None if speed_kmh is None else speed_kmh <= cfg.parked_speed_kmh
        if parked:
            self._stopped_ms += dt
            if self._stopped_ms >= cfg.rest_reset_min * 60000:
                self._drive_ms = 0.0  # a real rest resets continuous driving time
        else:
            self._drive_ms += dt  # unknown speed -> assume driving (conservative)
            self._stopped_ms = 0.0
        hour = datetime.fromtimestamp(self.start_epoch + ts_ms / 1000.0, tz=self.tz).hour
        circ = any(lo <= hour < hi for lo, hi in cfg.circadian_low_hours)
        drive_min = self._drive_ms / 60000.0
        long_drive = drive_min >= cfg.long_drive_min
        sens = cfg.high_risk_sensitivity if (circ or long_drive) else 1.0
        return ContextState(speed_kmh, parked, drive_min, hour, long_drive, circ, sens)
