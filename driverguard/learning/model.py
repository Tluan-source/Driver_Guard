"""Small causal EEG TCN and a self-contained, versioned inference checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import causal_windows, read_eeg_features
from .features import transform_features


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(F.gelu(self.conv(F.pad(x, (self.left_pad, 0)))))


class VigilanceTCN(nn.Module):
    def __init__(self, n_features: int = 85, width: int = 48, dropout: float = 0.2):
        super().__init__()
        self.project = nn.Conv1d(n_features, width, 1)
        self.blocks = nn.Sequential(*(CausalBlock(width, d, dropout) for d in (1, 2, 4)))
        self.head = nn.Sequential(nn.Linear(width, 24), nn.GELU(), nn.Dropout(dropout), nn.Linear(24, 1))

    def forward(self, x):
        z = F.gelu(self.project(x.transpose(1, 2)))
        return torch.sigmoid(self.head(self.blocks(z)[:, :, -1])).squeeze(-1)


class VigilancePredictor:
    """Input is the exact EEG feature schema saved at training, not camera EAR."""
    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint)
        self.meta = torch.load(self.path, map_location="cpu", weights_only=True)
        if self.meta.get("format_version") != 1:
            raise ValueError("Unsupported vigilance checkpoint version")
        self.context = int(self.meta["context"])
        self.feature_key = self.meta["feature_key"]
        self.feature_transform = self.meta.get("feature_transform", "absolute")
        if self.feature_transform not in {"absolute", "relative_log_power"}:
            raise ValueError("Unsupported checkpoint feature transform")
        self.mean = np.asarray(self.meta["mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["std"], dtype=np.float32)
        if (self.mean.ndim != 1 or not len(self.mean) or self.std.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError("Checkpoint normalization must be finite with positive, matching std")
        if not 1 <= self.context <= 15:
            raise ValueError("Checkpoint context must be in [1,15]")
        if self.meta["architecture"] not in {"tcn", "ridge", "constant", "rbf_svm", "logistic"}:
            raise ValueError("Unsupported checkpoint architecture")
        if not np.isfinite(self.meta["decision_threshold"]) or not 0 <= self.meta["decision_threshold"] <= 1:
            raise ValueError("Checkpoint decision threshold must be in [0,1]")
        self.n_features = len(self.mean)
        self.model = None
        if self.meta["architecture"] == "tcn":
            self.model = VigilanceTCN(**self.meta["model_config"])
            if self.model.project.in_channels != self.n_features:
                raise ValueError("Checkpoint model and normalization feature counts differ")
            self.model.load_state_dict(self.meta["state_dict"])
            if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
                raise ValueError("Nonfinite model weights in checkpoint")
            self.model.eval()
        elif self.meta["architecture"] in {"ridge", "logistic"}:
            coef = np.asarray(self.meta["coef"])
            if coef.shape != self.mean.shape or not np.isfinite(coef).all() or not np.isfinite(self.meta["intercept"]):
                raise ValueError("Invalid ridge/logistic weights in checkpoint")
        elif self.meta["architecture"] == "rbf_svm":
            support = np.asarray(self.meta["support_vectors"], dtype=np.float32)
            dual = np.asarray(self.meta["dual_coef"], dtype=np.float32)
            if (support.ndim != 2 or support.shape[1] != self.n_features or not len(support)
                    or dual.shape != (len(support),) or not np.isfinite(support).all()
                    or not np.isfinite(dual).all() or not np.isfinite(self.meta["intercept"])
                    or not np.isfinite(self.meta["gamma"]) or self.meta["gamma"] <= 0):
                raise ValueError("Invalid SVM weights in checkpoint")
            self.support_vectors = support
            self.dual_coef = dual
        elif not np.isfinite(self.meta["value"]) or not 0 <= self.meta["value"] <= 1:
            raise ValueError("Invalid constant prediction in checkpoint")

    def predict_windows(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.context, self.n_features) or not np.isfinite(x).all():
            raise ValueError(f"Expected finite [batch, {self.context}, {self.n_features}] EEG features")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x = (transform_features(x, self.feature_transform) - self.mean) / self.std
        if not np.isfinite(x).all():
            raise ValueError("EEG normalization overflow; sensor values are outside the usable range")
        architecture = self.meta["architecture"]
        if architecture == "ridge":
            return np.clip(x[:, -1] @ np.asarray(self.meta["coef"]) + self.meta["intercept"], 0, 1)
        if architecture == "logistic":
            decision = x[:, -1] @ np.asarray(self.meta["coef"]) + self.meta["intercept"]
            return 1 / (1 + np.exp(-np.clip(decision, -60, 60)))
        if architecture == "constant":
            return np.full(len(x), self.meta["value"], dtype=np.float32)
        if architecture == "rbf_svm":
            current = x[:, -1]
            support = self.support_vectors
            distances = np.maximum(np.sum(current * current, axis=1)[:, None]
                                   + np.sum(support * support, axis=1)[None, :]
                                   - 2 * current @ support.T, 0)
            decision = np.exp(-self.meta["gamma"] * distances) @ self.dual_coef + self.meta["intercept"]
            return 1 / (1 + np.exp(-np.clip(decision, -60, 60)))
        if architecture != "tcn":
            raise ValueError(f"Unsupported architecture {architecture!r}")
        if not len(x):
            return np.empty(0, dtype=np.float32)
        with torch.inference_mode():
            prediction = np.concatenate([self.model(torch.from_numpy(x[i:i + batch_size])).numpy()
                                         for i in range(0, len(x), batch_size)])
        if not np.isfinite(prediction).all():
            raise ValueError("Model inference produced nonfinite predictions")
        return prediction

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_windows(causal_windows(features, self.context))

    def predict_file(self, path: str | Path) -> np.ndarray:
        return self.predict(read_eeg_features(path, self.feature_key, self.n_features))

    def decisions(self, score: np.ndarray) -> np.ndarray:
        return np.asarray(score) >= self.meta["decision_threshold"]
