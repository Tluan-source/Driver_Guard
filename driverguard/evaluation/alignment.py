"""Validate explicit paired clock anchors without guessing dataset time units."""
from __future__ import annotations

import numpy as np


def fit_clock(video_seconds, psg_seconds, *, video_duration, psg_duration, tolerance_s=.1):
    video, psg = np.asarray(video_seconds, float), np.asarray(psg_seconds, float)
    if video.ndim != 1 or video.shape != psg.shape or len(video) < 3:
        raise ValueError("At least three paired one-dimensional clock anchors are required")
    if not np.isfinite(video).all() or not np.isfinite(psg).all():
        raise ValueError("Clock anchors must be finite")
    if not np.all(np.diff(video) > 0) or not np.all(np.diff(psg) > 0):
        raise ValueError("Clock anchors must be strictly increasing")
    if not (np.isfinite(video_duration) and np.isfinite(psg_duration)
            and np.isfinite(tolerance_s) and video_duration > 0 and psg_duration > 0 and tolerance_s > 0):
        raise ValueError("Durations and tolerance must be positive and finite")
    if video[0] < 0 or psg[0] < 0 or video[-1] > video_duration or psg[-1] > psg_duration:
        raise ValueError("Clock anchors lie outside recording bounds")
    slope, offset = np.linalg.lstsq(np.column_stack([video, np.ones_like(video)]), psg, rcond=None)[0]
    residual = np.abs(psg - (slope * video + offset))
    return {"scale": float(slope), "offset_s": float(offset),
            "drift_ppm": float((slope - 1) * 1e6), "max_residual_s": float(residual.max()),
            "within_tolerance": bool(residual.max() <= tolerance_s),
            "usable_video_interval_s": [float(video[0]), float(video[-1])],
            "scope": "anchor consistency only; physiological alignment requires verified source semantics"}


def validate_indices(indices, *, index_base, n_psg_samples, n_video_frames):
    values = np.asarray(indices)
    if index_base not in (0, 1):
        raise ValueError("Declare index_base 0 or 1; do not infer it")
    if values.ndim != 1 or len(values) != n_video_frames or not len(values):
        raise ValueError("Index mapping must contain one value per decoded video frame")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("PSG sample indices must be finite integers")
    zero = values.astype(np.int64) - index_base
    if np.any(np.diff(zero) <= 0) or zero[0] < 0 or zero[-1] >= n_psg_samples:
        raise ValueError("PSG sample indices must increase and stay within recording bounds")
    return {"mapped_frames": len(zero), "first_sample_zero_based": int(zero[0]),
            "last_sample_zero_based": int(zero[-1])}
