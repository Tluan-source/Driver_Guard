"""Timestamp-based temporal primitives (never frame-count based)."""
from __future__ import annotations

import math
from collections import deque
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class TimeWindow(Generic[T]):
    """Samples (ts_ms, value) kept for `window_ms`.

    `time_fraction(pred)` is TIME-WEIGHTED: each sample holds until the next one (capped at
    `max_gap_ms`), so the result is the same at 10, 15 or 30 fps. Samples whose value is None
    are "invalid" and are excluded from the denominator — a frame where the eyes are not
    visible is never counted as "eyes closed".
    """

    def __init__(self, window_ms: float, max_gap_ms: float = 300):
        self.window_ms = window_ms
        self.max_gap_ms = max_gap_ms
        self._buf: deque[tuple[int, T | None]] = deque()

    def add(self, ts_ms: int, value: T | None) -> None:
        self._buf.append((ts_ms, value))
        self.prune(ts_ms)

    def prune(self, now_ms: int) -> None:
        lo = now_ms - self.window_ms
        while self._buf and self._buf[0][0] < lo:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def values(self) -> list[T]:
        return [v for _, v in self._buf if v is not None]

    def _durations(self, now_ms: int):
        items = list(self._buf)
        for i, (ts, v) in enumerate(items):
            nxt = items[i + 1][0] if i + 1 < len(items) else now_ms
            yield v, max(0.0, min(nxt - ts, self.max_gap_ms))

    def valid_time_ms(self, now_ms: int) -> float:
        return sum(d for v, d in self._durations(now_ms) if v is not None)

    def time_fraction(self, pred: Callable[[T], bool], now_ms: int) -> tuple[float | None, float]:
        """Returns (fraction of valid time where pred(value) is True, valid coverage of window)."""
        valid = hit = 0.0
        for v, d in self._durations(now_ms):
            if v is None:
                continue
            valid += d
            if pred(v):
                hit += d
        coverage = valid / self.window_ms if self.window_ms else 0.0
        return (hit / valid if valid > 0 else None), coverage


class TimeEWMA:
    """Exponential moving average with a time constant (alpha depends on dt)."""

    def __init__(self, tau_ms: float, init: float | None = None):
        self.tau_ms = tau_ms
        self.value = init
        self._ts: int | None = None

    def update(self, x: float, ts_ms: int) -> float:
        if self.value is None or self._ts is None:
            self.value = x
        else:
            dt = max(0, ts_ms - self._ts)
            a = 1.0 - math.exp(-dt / self.tau_ms) if self.tau_ms > 0 else 1.0
            self.value += a * (x - self.value)
        self._ts = ts_ms
        return self.value


class PersistenceTimer:
    """Debounce: condition must hold continuously for `on_ms` to activate, and be false for
    `off_ms` to deactivate."""

    def __init__(self, on_ms: float, off_ms: float = 0):
        self.on_ms = on_ms
        self.off_ms = off_ms
        self.active = False
        self._true_since: int | None = None
        self._false_since: int | None = None

    def update(self, cond: bool, ts_ms: int) -> bool:
        if cond:
            self._false_since = None
            if self._true_since is None:
                self._true_since = ts_ms
            if not self.active and ts_ms - self._true_since >= self.on_ms:
                self.active = True
        else:
            self._true_since = None
            if self.active:
                if self._false_since is None:
                    self._false_since = ts_ms
                if ts_ms - self._false_since >= self.off_ms:
                    self.active = False
                    self._false_since = None
        return self.active

    def held_ms(self, ts_ms: int) -> int:
        return 0 if self._true_since is None else int(ts_ms - self._true_since)

    def reset(self) -> None:
        self.active = False
        self._true_since = self._false_since = None


class HysteresisSwitch:
    """Binary state with separate enter/exit thresholds (low-is-on or high-is-on)."""

    def __init__(self, enter: float, exit: float, on_when_below: bool):
        self.enter, self.exit, self.below = enter, exit, on_when_below
        self.on = False

    def update(self, x: float) -> bool:
        if self.below:
            if not self.on and x < self.enter:
                self.on = True
            elif self.on and x > self.exit:
                self.on = False
        else:
            if not self.on and x > self.enter:
                self.on = True
            elif self.on and x < self.exit:
                self.on = False
        return self.on
