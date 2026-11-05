"""Conservative, masked video-frame/PSG mapping for the official DROZY release."""
from __future__ import annotations

import numpy as np


def map_frames(signal, sfreq, timestamps, *, max_error_s=.015):
    signal, times = np.asarray(signal, float), np.asarray(timestamps, float)
    if (signal.ndim != 1 or times.ndim != 1 or not len(signal) or not len(times)
            or not np.isfinite(signal).all() or not np.isfinite(times).all()
            or not np.isfinite(sfreq) or sfreq <= 0 or not 0 < max_error_s < 1/60
            or times[0] < 0 or times[-1] > len(signal)/sfreq or np.any(np.diff(times) <= 0)):
        raise ValueError('Require finite in-bounds timestamps, waveform and sampling rate')
    edges = np.flatnonzero(np.diff((signal > 0).astype(np.int8)))+1
    samples = np.full(len(times), -1, dtype=np.int64)
    errors = np.full(len(times), np.nan)
    accepted = np.zeros(len(times), dtype=bool)
    if not len(edges):
        return samples, accepted, errors
    edge_times = edges/sfreq
    k = np.searchsorted(edge_times, times)
    left, right = np.clip(k-1, 0, len(edges)-1), np.clip(k, 0, len(edges)-1)
    nearest = np.where(abs(edge_times[left]-times) < abs(edge_times[right]-times), left, right)
    errors = edge_times[nearest]-times
    within = np.abs(errors) <= max_error_s
    counts = np.bincount(nearest[within], minlength=len(edges))
    accepted = (counts[nearest] == 1) & within
    samples[accepted] = edges[nearest[accepted]]
    return samples, accepted, errors


def map_grid(table, timestamps, samples, frame_accepted, sfreq, *, max_span_s=.15, max_grid_error_s=.1):
    values, times = np.asarray(table, float), np.asarray(timestamps, float)
    if (values.shape != (18000, 3) or not np.isfinite(values).all()
            or np.shape(samples) != times.shape or np.shape(frame_accepted) != times.shape
            or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(sfreq) or sfreq <= 0):
        raise ValueError('Invalid interpolation table or source clock arrays')
    selected_samples = np.asarray(samples)[np.asarray(frame_accepted, bool)]
    if (not np.isfinite(selected_samples).all() or np.any(selected_samples < 0)
            or not np.equal(selected_samples, np.floor(selected_samples)).all()
            or np.any(np.diff(selected_samples) <= 0)
            or not np.isfinite(max_span_s) or max_span_s <= 0
            or not np.isfinite(max_grid_error_s) or max_grid_error_s <= 0):
        raise ValueError('Accepted PSG samples and tolerances must be valid')
    a, b, w = values.T
    if (not np.equal(a, np.floor(a)).all() or not np.equal(b, np.floor(b)).all()
            or np.any((a < 1) | (a > len(times))) or np.any((b < 0) | (b > len(times)))
            or np.any((w < 0) | (w > 1)) or np.any((b == 0) & (w != 1))
            or np.any((b != 0) & (b < a)) or np.any(np.diff(a) < 0)):
        raise ValueError('Invalid one-based frame references or weights')
    left = a.astype(int)-1
    right = np.where(b == 0, a, b).astype(int)-1
    active_right = w < 1
    source_time = w*times[left] + (1-w)*times[right]
    grid_0, grid_1 = np.arange(18000)/30, (np.arange(18000)+1)/30
    within_grid = np.maximum(abs(source_time-grid_0), abs(source_time-grid_1)) <= max_grid_error_s
    accepted = (frame_accepted[left] & (~active_right | frame_accepted[right])
                & ((times[right]-times[left]) <= max_span_s) & within_grid)
    mapped = np.full(len(values), np.nan)
    mapped[accepted] = (w[accepted]*samples[left[accepted]]
                        + (1-w[accepted])*samples[right[accepted]])/sfreq
    return mapped, accepted, {'unmatched_source_rows': int((~(frame_accepted[left]
                                & (~active_right | frame_accepted[right]))).sum()),
                             'large_gap_rows': int(((times[right]-times[left]) > max_span_s).sum()),
                             'grid_out_of_tolerance_rows': int((~within_grid).sum())}
