"""Causal current-state camera GRU with a portable input and quality contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .camera_features import (
    CAMERA_FPS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    PHYSICAL_SCALES,
    SAMPLING_POLICY,
)
from .camera_features import PHYSICAL_FIELDS as PHYSICAL_FEATURES

CONTEXT = 50
FEATURE_DIM = 19
MIN_VALID_FRACTION = 0.6
MAX_GAP_MS = 300


def validate_input_contract(contract: dict) -> None:
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("Camera checkpoint requires input contract schema_version=1")
    if tuple(contract.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Camera input feature order does not match the 19-feature schema")
    scales = np.asarray(contract.get("physical_scales", ()), dtype=float)
    if scales.shape != (9,) or not np.array_equal(scales, PHYSICAL_SCALES):
        raise ValueError("Camera input contract has incorrect fixed physical scales")
    if contract.get("fps") != CAMERA_FPS:
        raise ValueError("Camera input contract requires 10 Hz sampling")
    if "landmarker_cadence_hz" in contract and contract["landmarker_cadence_hz"] != CAMERA_FPS:
        raise ValueError("Camera input contract landmarker cadence must match the 10 Hz model stream")
    for name in ("quality", "low_light"):
        if not isinstance(contract.get(name), dict) or not contract[name]:
            raise ValueError(f"Camera input contract requires a nonempty {name} configuration")
    for name in ("face_model_sha256", "source_manifest_sha256"):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Camera input contract requires a valid {name}")
    if contract.get("extractor_version") != FEATURE_VERSION:
        raise ValueError("Camera input contract extractor_version does not match this implementation")
    if "sampling" in contract and contract["sampling"] != SAMPLING_POLICY:
        raise ValueError("Camera input contract sampling policy does not match this implementation")


class CameraGRU(nn.Module):
    """Only the current and preceding frames contribute to the last-step logit."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


@dataclass(frozen=True)
class CameraPrediction:
    score: float | None
    reason: str | None
    valid_fraction: float


def validate_feature_arrays(x: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError("Camera features must have shape [frames,19]")
    if ts_ms.ndim != 1 or valid.ndim != 1 or len(ts_ms) != len(x) or len(valid) != len(x):
        raise ValueError("Camera timestamps and validity must contain one value per frame")
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(ts_ms.dtype, np.number):
        raise ValueError("Camera features and timestamps must be numeric")
    if valid.dtype != np.bool_:
        raise ValueError("Camera frame validity must be a Boolean array")
    masks = x[:, 9:]
    if np.isfinite(x).all():
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("Camera feature validity masks must be zero or one")
        if not np.array_equal(x[:, -1].astype(bool), valid):
            raise ValueError("Camera face_valid feature must match frame validity")
        if np.any(x[~valid] != 0):
            raise ValueError("Invalid raw-face frames must contain only zero features and masks")
        physical_count = len(PHYSICAL_FEATURES)
        if np.any(x[:, :physical_count][x[:, physical_count:2 * physical_count] == 0] != 0):
            raise ValueError("Unavailable physical features must be zero")


def window_quality(x: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray) -> CameraPrediction:
    """Shared training/online gate; rejected windows never acquire a normal label."""
    validate_feature_arrays(x, ts_ms, valid)
    fraction = float(np.mean(valid)) if len(valid) else 0.0
    if len(x) != CONTEXT:
        return CameraPrediction(None, "insufficient_history", fraction)
    if not np.isfinite(x).all():
        return CameraPrediction(None, "nonfinite_features", fraction)
    if not np.isfinite(ts_ms).all() or np.any(np.diff(ts_ms) <= 0):
        return CameraPrediction(None, "invalid_timestamps", fraction)
    if np.any(np.diff(ts_ms) > MAX_GAP_MS):
        return CameraPrediction(None, "timestamp_gap", fraction)
    if not valid[-1]:
        return CameraPrediction(None, "current_frame_invalid", fraction)
    if fraction < MIN_VALID_FRACTION:
        return CameraPrediction(None, "insufficient_quality", fraction)
    return CameraPrediction(None, None, fraction)


class CameraPredictor:
    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint)
        self.meta = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(self.meta, dict) or self.meta.get("format_version") != 1:
            raise ValueError("Unsupported camera checkpoint format")
        if self.meta.get("architecture") != "camera_gru":
            raise ValueError("Unsupported camera model architecture")
        if self.meta.get("context") != CONTEXT or self.meta.get("feature_dim") != FEATURE_DIM:
            raise ValueError("Camera checkpoint must use 50-frame, 19-feature windows")
        validate_input_contract(self.meta.get("input_contract"))
        self.input_contract = self.meta["input_contract"]
        self.context = CONTEXT
        self.n_features = FEATURE_DIM
        self.decision_threshold = float(self.meta.get("decision_threshold", float("nan")))
        if not np.isfinite(self.decision_threshold) or not 0 <= self.decision_threshold <= 1:
            raise ValueError("Camera decision threshold must be in [0,1]")
        hidden_size = self.meta.get("hidden_size")
        dropout = self.meta.get("dropout")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or not 1 <= hidden_size <= 256:
            raise ValueError("Invalid camera hidden size")
        if not isinstance(dropout, (int, float)) or not np.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("Invalid camera dropout")
        self.model = CameraGRU(hidden_size=hidden_size, dropout=dropout)
        self.model.load_state_dict(self.meta["state_dict"], strict=True)
        self.model.eval()
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Camera checkpoint contains nonfinite parameters")

    def predict_window(self, x: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray) -> CameraPrediction:
        features, timestamps, accepted = np.asarray(x), np.asarray(ts_ms), np.asarray(valid)
        result = window_quality(features, timestamps, accepted)
        if result.reason is not None:
            return result
        with torch.inference_mode():
            score = float(torch.sigmoid(self.model(torch.as_tensor(features, dtype=torch.float32)[None]))[0])
        if not np.isfinite(score):
            return CameraPrediction(None, "nonfinite_prediction", result.valid_fraction)
        return CameraPrediction(score, None, result.valid_fraction)
