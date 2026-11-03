"""Binary long-context adapter for CameraRuntime; causal smoothing, current-frame quality gating."""
from __future__ import annotations

from collections import deque

import numpy as np

from driverguard.learning.camera.model import CameraPrediction

from .data import quality_gate
from .inference import DANNPredictor


class DANNRuntimeAdapter:
    def __init__(self, predictor: DANNPredictor, *, stride_ms: int = 2500, smooth_ms: int = 300000):
        if predictor.class_names != ("alert", "drowsy"):
            raise ValueError("Runtime warning policy currently requires binary alert/drowsy")
        if stride_ms < 1 or smooth_ms < 0:
            raise ValueError("Invalid cadence or smoothing duration")
        self.predictor = predictor
        self.context = predictor.context
        self.input_contract = predictor.meta["input_contract"]
        # Binary argmax selects alert on a tie; this fixed cutoff preserves that tie rule.
        self.decision_threshold = float(np.nextafter(0.5, 1.0))
        self.stride_ms, self.smooth_ms = stride_ms, smooth_ms
        self.reset()

    def reset(self):
        self.history = deque()
        self.score_timestamp_ms = None
        self.raw_score = None

    def predict_window(self, x, ts_ms, valid):
        quality = quality_gate(np.asarray(x), np.asarray(ts_ms), np.asarray(valid), self.context)
        if quality.reason is not None:
            self.reset()
            return quality
        now = int(ts_ms[-1])
        if self.score_timestamp_ms is not None and now < self.score_timestamp_ms:
            self.reset()
        if self.score_timestamp_ms is None or now - self.score_timestamp_ms >= self.stride_ms:
            result = self.predictor.predict_window(x, ts_ms, valid)
            if result.probabilities is None:
                self.reset()
                return CameraPrediction(None, result.reason, result.valid_fraction)
            self.raw_score = result.probabilities[1]
            self.score_timestamp_ms = now
            self.history.append((now, self.raw_score))
        while len(self.history) > 1 and self.history[0][0] <= now - self.smooth_ms:
            self.history.popleft()
        score = float(np.mean([value for _, value in self.history])) if self.smooth_ms else self.raw_score
        return CameraPrediction(score, None, quality.valid_fraction)
