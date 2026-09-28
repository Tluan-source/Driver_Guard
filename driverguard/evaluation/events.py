"""Event-level evaluation (frame metrics alone are not enough — research doc "Event-level evaluation").

    ground truth:  [==========]
    prediction:       [========]      -> ONE true positive, onset delay = pred.start - gt.start

Also: false alerts per hour — only meaningful when the engine runs CONTINUOUSLY over full
videos (never on balanced clips).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

LEVEL_RANK = {"normal": 0, "caution": 1, "warning": 2, "critical": 3, "sensor_degraded": -1}


@dataclass(frozen=True)
class Interval:
    start_ms: int
    end_ms: int
    label: str = ""

    @property
    def dur(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    onset_delays_ms: list[int] = field(default_factory=list)
    duration_errors_ms: list[int] = field(default_factory=list)
    matched: list[tuple[Interval, Interval]] = field(default_factory=list)
    false_positives: list[Interval] = field(default_factory=list)
    misses: list[Interval] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)

    def onset_percentile(self, q: float) -> float | None:
        return float(np.percentile(self.onset_delays_ms, q)) if self.onset_delays_ms else None


def match_events(gt: Iterable[Interval], pred: Iterable[Interval], tolerance_ms: int = 500) -> MatchResult:
    """Greedy one-to-one matching: a prediction matches a GT event if it overlaps the GT interval
    extended by `tolerance_ms` on both sides. Earliest predictions are matched first."""
    gt = sorted(gt, key=lambda i: i.start_ms)
    pred = sorted(pred, key=lambda i: i.start_ms)
    used = [False] * len(pred)
    res = MatchResult(0, 0, 0)
    for g in gt:
        lo, hi = g.start_ms - tolerance_ms, g.end_ms + tolerance_ms
        hit = None
        for j, p in enumerate(pred):
            if used[j]:
                continue
            if p.start_ms <= hi and p.end_ms >= lo:
                hit = j
                break
            if p.start_ms > hi:
                break
        if hit is None:
            res.fn += 1
            res.misses.append(g)
        else:
            used[hit] = True
            p = pred[hit]
            res.tp += 1
            res.matched.append((g, p))
            res.onset_delays_ms.append(p.start_ms - g.start_ms)
            res.duration_errors_ms.append(p.dur - g.dur)
    for j, p in enumerate(pred):
        if not used[j]:
            res.fp += 1
            res.false_positives.append(p)
    return res


def intervals_from_ticks(ticks: list[dict], pred: Callable[[dict], bool], label: str = "",
                         merge_gap_ms: int = 0) -> list[Interval]:
    """Turn a tick stream into intervals where `pred(tick)` is True."""
    out: list[Interval] = []
    start = last = None
    for t in ticks:
        ts = t["timestamp_ms"]
        if pred(t):
            if start is None:
                start = ts
            last = ts
        elif start is not None:
            out.append(Interval(start, last, label))
            start = None
    if start is not None:
        out.append(Interval(start, last, label))
    if merge_gap_ms and out:
        merged = [out[0]]
        for iv in out[1:]:
            if iv.start_ms - merged[-1].end_ms <= merge_gap_ms:
                merged[-1] = Interval(merged[-1].start_ms, iv.end_ms, label)
            else:
                merged.append(iv)
        out = merged
    return out


def level_at_least(level: str) -> Callable[[dict], bool]:
    r = LEVEL_RANK[level]
    return lambda t: LEVEL_RANK.get(t["risk_level"], 0) >= r


def alerts_from_ticks(ticks: list[dict], min_level: str = "warning") -> list[Interval]:
    r = LEVEL_RANK[min_level]
    return [Interval(t["alert"]["ts_ms"], t["alert"]["ts_ms"], t["alert"]["level"])
            for t in ticks if t.get("alert") and LEVEL_RANK.get(t["alert"]["level"], -1) >= r]


def false_alerts_per_hour(n_false: int, total_ms: float) -> float | None:
    return n_false / (total_ms / 3.6e6) if total_ms > 0 else None
