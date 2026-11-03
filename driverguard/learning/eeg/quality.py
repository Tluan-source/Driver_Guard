"""Fixed, label-independent artifact checks for the robust EEG feature contract."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

ROBUST_PREPROCESSING = {
    "name": "robust_v1",
    "dc_removal": "per-channel window median",
    "reference": "samplewise median of accepted channels",
    "repair": "MNE standard_1020 spherical-spline interpolation of bad electrodes",
    "max_absolute_centered_uv": 500.0,
    "min_peak_to_peak_uv": 0.5,
    "max_relative_robust_scale": 8.0,
    "relative_scale_floor_uv": 1.0,
    "max_bad_channel_fraction": 0.2,
    "nonfinite_policy": "reject entire window",
    "threshold_basis": "Fixed conservative artifact screening; no labels or model scores used",
    "invalid_window_policy": "Preserve timestamp and mask; placeholder features must not be predicted",
    "channel_reason_bits": {"flat": 1, "gross_amplitude": 2, "relative_scale": 4, "nonfinite": 8},
    "window_reason_bits": {"too_many_bad_channels": 1, "nonfinite": 2, "flat_reference": 4},
}


@lru_cache(maxsize=128)
def _interpolation_matrix(channel_names: tuple[str, ...], bad_indices: tuple[int, ...]) -> np.ndarray:
    """Build a public-MNE interpolation operator once per observed bad-channel mask."""
    import mne

    channel_count = len(channel_names)
    identity = np.eye(channel_count, dtype=np.float64)
    raw = mne.io.RawArray(
        identity, mne.create_info(list(channel_names), 1000.0, "eeg"), verbose="ERROR"
    )
    raw.set_montage("standard_1020", match_case=False, verbose="ERROR")
    raw.info["bads"] = [channel_names[index] for index in bad_indices]
    raw.interpolate_bads(reset_bads=False, verbose="ERROR")
    return raw.get_data()


def robust_reference_window(
    volts: np.ndarray, channel_names: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray | bool | int | float]]:
    """Return referenced microvolts and quality flags without looking outside a window.

    Untrustworthy windows return zeros with quality_valid=False. At most 20% of
    channels are reconstructed; this is an approximation, not a recovered EEG
    measurement. Both the original bad-channel mask and timestamps are retained.
    """
    samples = np.asarray(volts, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] != len(channel_names):
        raise ValueError("EEG channels do not match the robust preprocessing contract")
    finite_channels = np.isfinite(samples).all(axis=1)
    safe = samples.copy()
    safe[~finite_channels] = 0.0
    centered_uv = (safe - np.median(safe, axis=1, keepdims=True)) * 1e6
    peak_to_peak = np.ptp(centered_uv, axis=1)
    robust_scale = 1.4826 * np.median(np.abs(centered_uv), axis=1)
    absolute_peak = np.max(np.abs(centered_uv), axis=1)
    median_scale = max(float(np.median(robust_scale)), ROBUST_PREPROCESSING["relative_scale_floor_uv"])
    reasons = np.zeros(samples.shape[0], dtype=np.uint8)
    reasons[peak_to_peak < ROBUST_PREPROCESSING["min_peak_to_peak_uv"]] |= 1
    reasons[absolute_peak > ROBUST_PREPROCESSING["max_absolute_centered_uv"]] |= 2
    reasons[robust_scale > ROBUST_PREPROCESSING["max_relative_robust_scale"] * median_scale] |= 4
    reasons[~finite_channels] |= 8
    bad = reasons != 0
    bad_fraction = float(np.mean(bad))
    window_reason = 0
    if bad_fraction > ROBUST_PREPROCESSING["max_bad_channel_fraction"]:
        window_reason |= 1
    if not finite_channels.all():
        window_reason |= 2
    referenced = np.zeros_like(centered_uv)
    if not window_reason:
        # The suspect electrodes never contribute to the reference.
        referenced = centered_uv - np.median(centered_uv[~bad], axis=0, keepdims=True)
        referenced[bad] = 0.0
        if bad.any():
            referenced = _interpolation_matrix(channel_names, tuple(np.flatnonzero(bad))) @ referenced
        if not np.any(np.ptp(referenced, axis=1) > 0):
            window_reason |= 4
            referenced.fill(0.0)
    quality = {
        "quality_valid": window_reason == 0,
        "quality_bad_channel_mask": bad,
        "quality_channel_reason_code": reasons,
        "quality_bad_channel_fraction": bad_fraction,
        "quality_good_channel_count": int(np.count_nonzero(~bad)),
        "quality_channel_peak_to_peak_uv": peak_to_peak.astype(np.float32),
        "quality_channel_robust_scale_uv": robust_scale.astype(np.float32),
        "quality_max_peak_to_peak_uv": float(peak_to_peak.max()),
        "quality_max_robust_scale_uv": float(robust_scale.max()),
        "quality_reason_code": window_reason,
    }
    return referenced, quality
