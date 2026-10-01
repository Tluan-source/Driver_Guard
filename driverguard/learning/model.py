"""Small causal EEG TCN and a self-contained, versioned inference checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from torch import nn
from torch.nn import functional as F

from .data import causal_windows
from .features import transform_features


def linear_ensemble_scores(transformed: np.ndarray, members: list[dict]) -> np.ndarray:
    """Evaluate portable member parameters after the shared feature transform."""
    features = np.asarray(transformed, dtype=np.float64)
    predictions = []
    for member in members:
        standardized = (features - np.asarray(member["mean"])) / np.asarray(member["std"])
        decision = standardized @ np.asarray(member["coef"]) + float(member["intercept"])
        calibration = member["calibration"]
        if calibration["method"] == "identity":
            predictions.append(expit(decision))
        elif calibration["method"] == "sigmoid_decision":
            predictions.append(expit(float(calibration["slope"]) * decision
                                     + float(calibration["intercept"])))
        else:
            raw = expit(decision)
            predictions.append(expit(float(calibration["slope"]) * logit(np.clip(raw, 1e-6, 1 - 1e-6))
                                     + float(calibration["intercept"])))
    score = np.mean(predictions, axis=0)
    if not np.isfinite(score).all():
        raise ValueError("Ensemble inference produced nonfinite predictions")
    return score


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
        if self.meta["architecture"] not in {"tcn", "ridge", "constant", "rbf_svm", "logistic",
                                             "calibrated_linear_ensemble"}:
            raise ValueError("Unsupported checkpoint architecture")
        if not np.isfinite(self.meta["decision_threshold"]) or not 0 <= self.meta["decision_threshold"] <= 1:
            raise ValueError("Checkpoint decision threshold must be in [0,1]")
        self.n_features = len(self.mean)
        dataset_meta = self.meta.get("dataset_metadata", {})
        if not isinstance(dataset_meta, dict):
            raise ValueError("Checkpoint dataset metadata must be a dictionary")  # noqa: TRY004
        self.preprocessing = self.meta.get("preprocessing", dataset_meta.get("preprocessing", "legacy_car"))
        if "preprocessing" in dataset_meta and dataset_meta["preprocessing"] != self.preprocessing:
            raise ValueError("Checkpoint preprocessing conflicts with dataset metadata")
        if self.preprocessing not in {"legacy_car", "robust_v1"}:
            raise ValueError("Unsupported checkpoint EEG preprocessing")
        self.channel_names = self.meta.get("channel_names", dataset_meta.get("channel_names"))
        if (dataset_meta.get("channel_names") is not None
                and self.channel_names != dataset_meta["channel_names"]):
            raise ValueError("Checkpoint channel names conflict with dataset metadata")
        if ("preprocessing_config" in self.meta and "preprocessing_config" in dataset_meta
                and self.meta["preprocessing_config"] != dataset_meta["preprocessing_config"]):
            raise ValueError("Checkpoint preprocessing configuration conflicts with dataset metadata")
        if self.channel_names is not None and (
            not isinstance(self.channel_names, (list, tuple))
            or len(self.channel_names) * 5 != self.n_features
            or len(set(self.channel_names)) != len(self.channel_names)
            or not all(isinstance(name, str) and name for name in self.channel_names)
        ):
            raise ValueError("Invalid checkpoint EEG channel names")
        if self.preprocessing == "robust_v1" and (self.context != 1 or not self.channel_names):
            raise ValueError("Robust checkpoint requires context=1 and verified EEG channels")
        self.model = None
        if self.meta["architecture"] == "calibrated_linear_ensemble":
            if self.context != 1 or self.feature_transform != "relative_log_power":
                raise ValueError("Calibrated linear ensemble requires context=1 and relative log power")
            self.members = self.meta.get("members")
            if not isinstance(self.members, list) or not self.members:
                raise ValueError("Calibrated linear ensemble requires nonempty members")
            for member in self.members:
                self._validate_member(member)
        elif self.meta["architecture"] == "tcn":
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

    def _validate_member(self, member: dict) -> None:
        if not isinstance(member, dict):
            raise ValueError("Ensemble members must be parameter dictionaries")  # noqa: TRY004
        for key in ("mean", "std", "coef"):
            value = np.asarray(member.get(key), dtype=float)
            if value.shape != (self.n_features,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid ensemble member {key}")
        if np.any(np.asarray(member["std"]) <= 0) or not np.isfinite(member.get("intercept", np.nan)):
            raise ValueError("Invalid ensemble member normalization/intercept")
        calibration = member.get("calibration", {})
        if calibration.get("method") not in {"identity", "sigmoid_logit", "sigmoid_decision"}:
            raise ValueError("Invalid ensemble member probability calibration")
        if calibration["method"] in {"sigmoid_logit", "sigmoid_decision"} and (
            not np.isfinite(calibration.get("slope", np.nan)) or calibration["slope"] < 0
            or not np.isfinite(calibration.get("intercept", np.nan))
        ):
            raise ValueError("Ensemble sigmoid calibration must be finite and monotonic")
        fit, held = member.get("fit_subjects"), member.get("calibration_subjects")
        if (not isinstance(fit, list) or not isinstance(held, list) or not fit or not held
                or not all(isinstance(subject, str) and subject for subject in fit + held)
                or len(set(fit)) != len(fit) or len(set(held)) != len(held)
                or set(fit).intersection(held)):
            raise ValueError("Ensemble fitting and calibration subjects must be disjoint")

    def _quality(self, quality_valid, size: int) -> np.ndarray:
        if quality_valid is None:
            if self.preprocessing == "robust_v1":
                raise ValueError("Robust EEG inference requires explicit quality_valid")
            return np.ones(size, dtype=bool)
        mask = np.asarray(quality_valid)
        if mask.dtype != np.bool_ or mask.shape != (size,):
            raise ValueError("quality_valid must be a boolean mask matching feature rows")
        if self.context != 1 and not mask.all():
            raise ValueError("Quality abstention requires a context=1 checkpoint")
        return mask

    def predict_windows(self, x: np.ndarray, batch_size: int = 512, *, quality_valid=None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.context, self.n_features):
            raise ValueError(f"Expected finite [batch, {self.context}, {self.n_features}] EEG features")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        accepted = self._quality(quality_valid, len(x))
        if not accepted.all():
            result = np.full(len(x), np.nan)
            if accepted.any():
                result[accepted] = self.predict_windows(x[accepted], batch_size,
                                                       quality_valid=np.ones(int(accepted.sum()), dtype=bool))
            return result
        if not np.isfinite(x).all():
            raise ValueError(f"Expected finite [batch, {self.context}, {self.n_features}] EEG features")
        if self.meta["architecture"] == "calibrated_linear_ensemble":
            return linear_ensemble_scores(transform_features(x[:, -1], self.feature_transform), self.members)
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

    def predict_features(self, features: np.ndarray, quality_valid=None) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(f"Expected finite [time, {self.n_features}] EEG features")
        mask = self._quality(quality_valid, len(features))
        windows = features[:, None, :] if self.context == 1 else causal_windows(features, self.context)
        return self.predict_windows(windows, quality_valid=mask)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_features(features)

    def predict_file(self, path: str | Path) -> np.ndarray:
        from .data import read_feature_contract

        features, quality, _, _ = read_feature_contract(
            path, self.feature_key, self.n_features, expected_preprocessing=self.preprocessing,
            expected_channels=self.channel_names)
        return self.predict_features(features, quality_valid=quality)

    def decisions(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError("Decisions require finite scores in [0,1]; abstentions have no decision")
        return score >= self.meta["decision_threshold"]
