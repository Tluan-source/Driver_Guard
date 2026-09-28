"""Trip memory — whole-trip fatigue trend (đề bài: "memory theo dõi xu hướng mệt mỏi cả chuyến").

GVHD review (2, 9.3): an instant beep at a microsleep is late; the larger value is detecting a
DEGRADING TREND minutes earlier and supporting the "keep driving or rest?" decision, and giving
the fleet manager a trip-level risk record (metadata only, never faces).

Per `bucket_s` (default 60 s) we aggregate the tick stream into one row, compute a
fatigue index (0-100, seed formula — to be validated/learned), and a robust slope
(Theil-Sen) over the last `trend_window_min` minutes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from statistics import median

from ..config import TripCfg


def _norm(x: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


@dataclass
class Bucket:
    t_start_ms: int
    total_ms: float = 0.0
    valid_ms: float = 0.0
    degraded_ms: float = 0.0
    perclos_wsum: float = 0.0
    perclos_w: float = 0.0
    blink_durations: list[int] = field(default_factory=list)
    long_closures: int = 0
    yawns: int = 0
    alerts: dict[str, int] = field(default_factory=dict)
    max_severity: int = 0
    fatigue_index: float | None = None

    def finalize(self) -> None:
        if self.total_ms <= 0 or self.valid_ms < 0.3 * self.total_ms:
            self.fatigue_index = None
            return
        perclos = self.perclos_wsum / self.perclos_w if self.perclos_w else 0.0
        mean_blink = sum(self.blink_durations) / len(self.blink_durations) if self.blink_durations else 150.0
        fi = (0.45 * _norm(perclos, 0.05, 0.30)
              + 0.20 * _norm(mean_blink, 150.0, 500.0)
              + 0.20 * min(1.0, self.long_closures / 3.0)
              + 0.15 * min(1.0, self.yawns / 3.0))
        self.fatigue_index = round(100.0 * fi, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["perclos_mean"] = round(self.perclos_wsum / self.perclos_w, 4) if self.perclos_w else None
        d.pop("perclos_wsum")
        d.pop("perclos_w")
        d["blinks"] = len(self.blink_durations)
        d.pop("blink_durations")
        return d


def theil_sen_slope(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(ys[j] - ys[i]) / (xs[j] - xs[i]) for i, j in combinations(range(len(xs)), 2) if xs[j] != xs[i]]
    return median(pairs) if pairs else None


class TripMemory:
    def __init__(self, cfg: TripCfg, long_closure_ms: int):
        self.cfg = cfg
        self.long_closure_ms = long_closure_ms
        self.buckets: list[Bucket] = []
        self._cur: Bucket | None = None
        self._last_ts: int | None = None
        self.fatigue_index: float | None = None
        self.trend: float | None = None
        self.on_bucket_closed = None  # optional callback(Bucket) -> e.g. persist to SQLite

    def update(self, ts: int, face_valid: bool, degraded: bool, perclos: float | None,
               completed_closure_ms: int | None, new_yawn: bool, alert_level: str | None,
               severity: int) -> None:
        bucket_ms = self.cfg.bucket_s * 1000
        if self._cur is None:
            self._cur = Bucket(t_start_ms=ts)
        while ts >= self._cur.t_start_ms + bucket_ms:
            self._close()
            self._cur = Bucket(t_start_ms=self.buckets[-1].t_start_ms + int(bucket_ms))
        dt = 0 if self._last_ts is None else min(max(ts - self._last_ts, 0), 500)
        self._last_ts = ts
        b = self._cur
        b.total_ms += dt
        if face_valid:
            b.valid_ms += dt
        if degraded:
            b.degraded_ms += dt
        if perclos is not None:
            b.perclos_wsum += perclos * dt
            b.perclos_w += dt
        if completed_closure_ms is not None:
            if completed_closure_ms >= self.long_closure_ms:
                b.long_closures += 1
            else:
                b.blink_durations.append(completed_closure_ms)
        if new_yawn:
            b.yawns += 1
        if alert_level:
            b.alerts[alert_level] = b.alerts.get(alert_level, 0) + 1
        b.max_severity = max(b.max_severity, severity)

    def _close(self) -> None:
        b = self._cur
        b.finalize()
        self.buckets.append(b)
        if b.fatigue_index is not None:
            self.fatigue_index = b.fatigue_index
        self._update_trend()
        if self.on_bucket_closed:
            self.on_bucket_closed(b)

    def _update_trend(self) -> None:
        n = int(self.cfg.trend_window_min * 60 / self.cfg.bucket_s)
        recent = [b for b in self.buckets[-n:] if b.fatigue_index is not None]
        if len(recent) < self.cfg.trend_min_buckets:
            self.trend = None
            return
        xs = [b.t_start_ms / 60000.0 for b in recent]
        ys = [b.fatigue_index for b in recent]
        self.trend = theil_sen_slope(xs, ys)

    @property
    def trend_rising(self) -> bool:
        return (self.trend is not None and self.trend >= self.cfg.trend_slope_caution
                and (self.fatigue_index or 0) >= 25)

    def summary(self) -> dict:
        closed = self.buckets
        total = sum(b.total_ms for b in closed) or 1.0
        alerts: dict[str, int] = {}
        for b in closed:
            for k, v in b.alerts.items():
                alerts[k] = alerts.get(k, 0) + v
        hours = total / 3.6e6
        return {
            "minutes": round(total / 60000.0, 2),
            "degraded_ratio": round(sum(b.degraded_ms for b in closed) / total, 4),
            "alerts": alerts,
            "alerts_per_hour": round(sum(alerts.values()) / hours, 2) if hours > 0 else None,
            "long_closures": sum(b.long_closures for b in closed),
            "yawns": sum(b.yawns for b in closed),
            "fatigue_index_last": self.fatigue_index,
            "fatigue_trend_per_min": None if self.trend is None else round(self.trend, 3),
            "fatigue_series": [(b.t_start_ms, b.fatigue_index) for b in closed],
        }
