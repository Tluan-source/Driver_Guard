"""Causal, recording-local EEG bandpower extraction shared by train and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import welch

BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 14.0),
    ("beta", 14.0, 31.0),
    ("gamma", 31.0, 50.0),
)
# Original protocol uses these 30 effective EEG electrodes; the final four
# CNT channels are unused and flat, and EOG/mastoids are not EEG inputs.
CHANNEL_NAMES = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
)


def bandpower_window(volts: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return channel-major log10 bandpower in microvolt squared for one window."""
    if sample_rate < 100 or not np.isfinite(sample_rate):
        raise ValueError("Sample rate must be finite and at least 100 Hz")
    samples = np.asarray(volts, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < round(sample_rate):
        raise ValueError("EEG window must have at least two channels and one second of samples")
    if not np.isfinite(samples).all():
        raise ValueError("EEG window contains non-finite samples")
    referenced = (samples - samples.mean(axis=0, keepdims=True)) * 1e6
    if not np.any(np.ptp(referenced, axis=1) > 0):
        raise ValueError("All EEG channels are flat after common average reference")
    frequencies, psd = welch(
        referenced,
        fs=sample_rate,
        window="hann",
        nperseg=round(sample_rate),
        noverlap=round(sample_rate) // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    step_hz = frequencies[1] - frequencies[0]
    bandpowers = []
    for _, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        bandpowers.append(np.sum(psd[:, mask], axis=1) * step_hz)
    powers = np.stack(bandpowers, axis=1)
    return np.log10(np.maximum(powers, 1e-12)).reshape(-1).astype(np.float32)


def _read_cnt(path: str | Path):
    import mne

    path = Path(path)
    raw = mne.io.read_raw_cnt(
        path, preload=False, data_format="int32", recompute_n_samples=True, verbose="ERROR"
    )
    try:
        channel_count = len(raw.ch_names)
        # Neuroscan CNT has a 900-byte setup and 75-byte header per channel.
        # Trailing event bytes make this a conservative bound on available samples.
        available = (path.stat().st_size - 900 - 75 * channel_count) // (4 * channel_count)
        if int(raw.n_times) <= 0 or int(raw.n_times) > available:
            raise ValueError("CNT decoded sample count exceeds the physical file size")
        if float(raw.info["sfreq"]) != 1000.0:
            raise ValueError("This CNT feature contract requires the original 1000 Hz sample rate")
        missing = set(CHANNEL_NAMES) - set(raw.ch_names)
        if missing:
            raise ValueError(f"Recording lacks required EEG channels: {sorted(missing)}")
    except Exception:
        raw.close()
        raise
    return raw


def cnt_recording_info(path: str | Path) -> dict:
    """Return the physically validated recording dimensions without loading EEG."""
    import mne

    raw = _read_cnt(path)
    try:
        return {
            "n_samples": int(raw.n_times),
            "sample_rate_hz": float(raw.info["sfreq"]),
            "duration_seconds": int(raw.n_times) / float(raw.info["sfreq"]),
            "recorded_channel_count": len(raw.ch_names),
            "eeg_channel_count": len(CHANNEL_NAMES),
            "mne_version": mne.__version__,
        }
    finally:
        raw.close()


def extract_cnt(path: str | Path, window_seconds: float = 4.0
                ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the published Neuroscan CNT format and return non-overlap windows.

    Return (x [time, 150], window_end_seconds [time], ordered EEG channels).
    An incomplete final window is discarded. No feature uses future samples.
    """
    if not np.isfinite(window_seconds) or window_seconds < 1:
        raise ValueError("window_seconds must be finite and at least one second")
    # Header numsamples is malformed in this release. The paper specifies 5 min
    # at 1000 Hz; int32 gives ~300k samples, while int16 misreads each 32-bit word.
    raw = _read_cnt(path)
    try:
        sample_rate = float(raw.info["sfreq"])
        window_samples = round(window_seconds * sample_rate)
        count = int(raw.n_times) // window_samples
        if count == 0:
            raise ValueError("Recording is shorter than one complete EEG window")
        features = []
        for index in range(count):
            start = index * window_samples
            values = raw.get_data(picks=list(CHANNEL_NAMES), start=start, stop=start + window_samples)
            features.append(bandpower_window(values, sample_rate))
        x = np.stack(features)
        ends = np.arange(1, count + 1, dtype=np.float64) * window_samples / sample_rate
    finally:
        raw.close()
    return x, ends, list(CHANNEL_NAMES)
