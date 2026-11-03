"""Safe checkpoint loading and quality-gated camera-only inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from driverguard.learning.camera.model import validate_input_contract

from .data import LABEL_MODES, quality_gate
from .model import CONTEXTS, CameraDANN, ClassifierExport


@dataclass(frozen=True)
class DANNPrediction:
    probabilities: tuple[float, ...] | None
    label: int | None
    reason: str | None
    valid_fraction: float


class DANNPredictor:
    def __init__(self, checkpoint: str | Path):
        self.meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if self.meta.get("format_version") != 1 or self.meta.get("architecture") not in (
            "camera_gru_dann", "camera_gru_ablation"
        ):
            raise ValueError("Unsupported GRU/DANN inference checkpoint")
        validate_input_contract(self.meta.get("input_contract"))
        self.context = self.meta.get("context_frames")
        pooling = self.meta["model_config"].get("pooling", "frame")
        allowed = {seconds * 10 for seconds in CONTEXTS.get(pooling, ())}
        if self.context not in allowed or self.meta.get("feature_dim") != 19:
            raise ValueError("Invalid GRU/DANN input shape")
        self.class_names = tuple(self.meta.get("class_names", ()))
        if self.class_names != LABEL_MODES.get(self.meta.get("label_mode")):
            raise ValueError("Checkpoint label mapping is invalid")
        model = CameraDANN(**self.meta["model_config"])
        if model.classifier.out_features != len(self.class_names):
            raise ValueError("Classifier size disagrees with label mapping")
        model.load_state_dict(self.meta["state_dict"], strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Checkpoint parameters must be finite")
        if torch.any(model.scale <= 0):
            raise ValueError("Normalization scale must be positive")
        self.model = ClassifierExport(model).eval()

    def predict_window(self, x, ts_ms, valid) -> DANNPrediction:
        x, ts_ms, valid = np.asarray(x), np.asarray(ts_ms), np.asarray(valid)
        quality = quality_gate(x, ts_ms, valid, self.context)
        if quality.reason is not None:
            return DANNPrediction(None, None, quality.reason, quality.valid_fraction)
        with torch.inference_mode():
            probabilities = self.model(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
        if not np.isfinite(probabilities).all():
            return DANNPrediction(None, None, "nonfinite_prediction", quality.valid_fraction)
        return DANNPrediction(tuple(float(v) for v in probabilities), int(probabilities.argmax()),
                              None, quality.valid_fraction)
