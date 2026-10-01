"""Feature transforms shared by fitting and checkpoint inference."""
from __future__ import annotations

import numpy as np


def transform_features(x: np.ndarray, mode: str = "absolute") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if mode == "absolute":
        return x
    if mode != "relative_log_power":
        raise ValueError(f"Unknown EEG feature transform {mode!r}")
    if x.shape[-1] % 5:
        raise ValueError("Relative EEG power requires channel-major groups of five bands")
    bands = x.reshape(*x.shape[:-1], -1, 5)
    maximum = bands.max(axis=-1, keepdims=True)
    log_total = maximum + np.log10(np.sum(np.power(10., bands - maximum), axis=-1, keepdims=True))
    return (bands - log_total).reshape(x.shape).astype(np.float32)
