"""Masked, past-only camera statistics; closure events are 10 Hz proxies."""
from __future__ import annotations

import numpy as np

from .camera_features import PHYSICAL_FIELDS

STATISTICS_VERSION = "masked_temporal_statistics_v1"
STATISTICS = ("mean", "std", "p10", "median", "p90", "trend_per_second", "change_per_second", "availability")
STATISTIC_NAMES = tuple(f"{field}_{stat}" for field in PHYSICAL_FIELDS for stat in STATISTICS) + (
    "closed_eye_fraction_proxy", "max_closed_eye_seconds_proxy", "eye_closing_transitions_per_second_proxy",
    "open_mouth_fraction_proxy", "max_open_mouth_seconds_proxy", "face_valid_fraction", "eyes_available_fraction",
)


def _max_duration(condition: np.ndarray, available: np.ndarray, timestamps: np.ndarray) -> float:
    longest, start = 0.0, None
    for index, active in enumerate(condition & available):
        if active:
            if start is None:
                start = index
            longest = max(longest, (timestamps[index] - timestamps[start]) / 1000 + 0.1)
        else:
            start = None
    return float(longest)


def temporal_statistics(x: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != 19 or len(x) != len(ts_ms) or len(x) != len(valid) or len(x) < 2:
        raise ValueError("Temporal statistics require aligned camera feature arrays with at least two samples")
    if not np.isfinite(x).all() or not np.isfinite(ts_ms).all() or np.any(np.diff(ts_ms) <= 0):
        raise ValueError("Temporal statistics require finite features and increasing timestamps")
    times = (ts_ms - ts_ms[0]) / 1000.0
    result = []
    for index in range(9):
        available = valid & (x[:, index + 9] == 1)
        values, observed_times = x[available, index], times[available]
        if len(values):
            centered_times = observed_times - observed_times.mean()
            denominator = float(centered_times @ centered_times)
            trend = float(centered_times @ (values - values.mean()) / denominator) if denominator > 0 else 0.0
            adjacent = available[:-1] & available[1:]
            changes = np.abs(np.diff(x[:, index])[adjacent] / np.diff(times)[adjacent])
            result.extend((float(values.mean()), float(values.std()), *np.quantile(values, (0.1, 0.5, 0.9)),
                           trend, float(changes.mean()) if len(changes) else 0.0, float(available.mean())))
        else:
            result.extend((0.0,) * 8)
    eyes = valid & (x[:, 9] == 1) & (x[:, 10] == 1)
    closed = (x[:, 0] + x[:, 1]) / 2 < 0.2 / 0.3
    adjacent_eyes = eyes[:-1] & eyes[1:]
    transitions = np.count_nonzero((~closed[:-1]) & closed[1:] & adjacent_eyes)
    mouth = valid & (x[:, 11] == 1)
    opened = x[:, 2] > 0.5 / 0.6
    duration = times[-1] + 0.1
    result.extend((float(closed[eyes].mean()) if eyes.any() else 0.0,
                   _max_duration(closed, eyes, ts_ms), transitions / duration,
                   float(opened[mouth].mean()) if mouth.any() else 0.0,
                   _max_duration(opened, mouth, ts_ms), float(valid.mean()), float(eyes.mean())))
    vector = np.asarray(result, dtype=np.float64)
    if vector.shape != (len(STATISTIC_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Invalid temporal summary")
    return vector
