"""Causal measurements of complete, observed eyelid closing/reopening cycles.

Openness is a nonnegative ratio such as EAR / calibrated open-eye EAR. The
thresholds describe that proxy, not the percentage of pupil occlusion. Measurements
are available only after the eyes cross the open threshold again, so a partial
cycle never contributes a made-up duration to model features.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BlinkKinetics:
    """Band-to-band durations; slope uses openness units per second."""

    completed_at_ms: float
    closing_ms: float
    reopening_ms: float
    reopening_slope_per_s: float
    reopening_closing_ratio: float


class BlinkKineticsTracker:
    """Measure a high -> low -> high cycle using timestamped observations.

    Closing duration runs from a downward crossing of ``open_threshold`` to
    ``closed_threshold``. Reopening duration runs from the final upward crossing
    of ``closed_threshold`` to ``open_threshold``. A closed plateau therefore
    does not masquerade as slow reopening. Crossings use linear interpolation
    only between adjacent, valid observations within ``max_gap_ms``.

    Every invalid sample or timestamp discontinuity aborts the current cycle.
    A cycle must start with an observed open eye; starting inside a closure does
    not provide enough evidence to measure closing/reopening asymmetry.
    """

    def __init__(
        self,
        closed_threshold: float = 0.25,
        open_threshold: float = 0.8,
        max_gap_ms: float = 150.0,
    ):
        if not (math.isfinite(closed_threshold) and math.isfinite(open_threshold)
                and 0 <= closed_threshold < open_threshold):
            raise ValueError("thresholds must be finite with 0 <= closed < open")
        if not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be finite and positive")
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.max_gap_ms = max_gap_ms
        self.reset()

    def reset(self) -> None:
        self._previous: tuple[float, float] | None = None
        self._closing_start: float | None = None
        self._closing_ms: float | None = None
        self._reopening_start: float | None = None

    def _abort_cycle(self) -> None:
        self._closing_start = self._closing_ms = self._reopening_start = None

    @staticmethod
    def _crossing(previous: tuple[float, float], ts_ms: float, value: float, level: float) -> float:
        previous_ts, previous_value = previous
        return previous_ts + (level - previous_value) / (value - previous_value) * (ts_ms - previous_ts)

    def update(self, ts_ms: float, openness: float | None) -> BlinkKinetics | None:
        """Return a newly completed cycle, or ``None``; never return stale evidence."""
        if (not math.isfinite(ts_ms) or ts_ms < 0 or openness is None
                or not math.isfinite(openness) or openness < 0):
            self.reset()
            return None
        previous = self._previous
        self._previous = (float(ts_ms), float(openness))
        if previous is None:
            return None
        dt_ms = ts_ms - previous[0]
        if dt_ms <= 0 or dt_ms > self.max_gap_ms:
            self._abort_cycle()
            return None

        previous_value = previous[1]
        if self._closing_start is None:
            if previous_value >= self.open_threshold and openness < self.open_threshold:
                self._closing_start = self._crossing(previous, ts_ms, openness, self.open_threshold)
            else:
                return None

        if self._closing_ms is None:
            if openness <= self.closed_threshold:
                closed_at = self._crossing(previous, ts_ms, openness, self.closed_threshold)
                self._closing_ms = closed_at - self._closing_start
            elif openness >= self.open_threshold:
                self._abort_cycle()  # a small eyelid fluctuation never reached the closed band
            return None

        if openness <= self.closed_threshold:
            self._reopening_start = None
            return None
        if previous_value <= self.closed_threshold:
            self._reopening_start = self._crossing(previous, ts_ms, openness, self.closed_threshold)
        if openness < self.open_threshold or self._reopening_start is None:
            return None

        reopened_at = self._crossing(previous, ts_ms, openness, self.open_threshold)
        reopening_ms = reopened_at - self._reopening_start
        closing_ms = self._closing_ms
        self._abort_cycle()
        if reopening_ms <= 0 or closing_ms <= 0:
            return None
        return BlinkKinetics(
            completed_at_ms=reopened_at,
            closing_ms=closing_ms,
            reopening_ms=reopening_ms,
            reopening_slope_per_s=(self.open_threshold - self.closed_threshold) * 1000 / reopening_ms,
            reopening_closing_ratio=reopening_ms / closing_ms,
        )


def blink_kinetics_features(
    timestamps_ms: Iterable[float],
    openness: Iterable[float | None],
    *,
    closed_threshold: float = 0.25,
    open_threshold: float = 0.8,
    max_gap_ms: float = 150.0,
) -> dict[str, float | int | None]:
    """Aggregate complete cycles in one window without filling missing values.

    The two iterables must have the same length. Features are measured solely
    from this window; a truncated cycle at either edge is excluded. A zero
    ``blink_count`` means no complete observed cycle, while kinetics remain
    ``None`` rather than becoming an artificial zero-duration blink.
    """
    tracker = BlinkKineticsTracker(closed_threshold, open_threshold, max_gap_ms)
    cycles: list[BlinkKinetics] = []
    sentinel = object()
    timestamps = iter(timestamps_ms)
    values = iter(openness)
    while True:
        timestamp = next(timestamps, sentinel)
        value = next(values, sentinel)
        if timestamp is sentinel and value is sentinel:
            break
        if timestamp is sentinel or value is sentinel:
            raise ValueError("timestamps_ms and openness must have the same length")
        cycle = tracker.update(timestamp, value)
        if cycle is not None:
            cycles.append(cycle)
    count = len(cycles)
    return {
        "blink_count": count,
        "mean_closing_ms": sum(c.closing_ms for c in cycles) / count if count else None,
        "mean_reopening_ms": sum(c.reopening_ms for c in cycles) / count if count else None,
        "mean_reopening_slope_per_s": (
            sum(c.reopening_slope_per_s for c in cycles) / count if count else None
        ),
        "mean_reopening_closing_ratio": (
            sum(c.reopening_closing_ratio for c in cycles) / count if count else None
        ),
    }
