"""Portable linear ensembles calibrated on disjoint groups of training subjects."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cross_validation import SessionPrediction, fit_calibration
from .data import FEATURE_KEY, Session
from .features import transform_features
from .model import linear_ensemble_scores


def _session_quality(session: Session) -> tuple[np.ndarray, np.ndarray]:
    mask = session.quality_valid
    if mask is None:
        if session.preprocessing == "robust_v1":
            raise ValueError("Robust training/inference requires explicit quality_valid")
        mask = np.ones(len(session.x), dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (len(session.x),):
            raise ValueError("Session quality_valid must be a boolean mask aligned with feature rows")
    reason = (np.full(len(mask), "", dtype="U64") if session.quality_reason is None
              else np.asarray(session.quality_reason, dtype=str).copy())
    if reason.shape != mask.shape:
        raise ValueError("Session quality reasons must align with feature rows")
    reason[mask] = ""
    reason[~mask & (reason == "")] = "quality_failure"
    return mask, reason


def _subject_class_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=float)
    subjects = np.unique(groups)
    for subject in subjects:
        selected = groups == subject
        classes = np.unique(labels[selected])
        for label in classes:
            rows = selected & (labels == label)
            weights[rows] = 1 / (len(subjects) * len(classes) * rows.sum())
    return weights * len(labels)


@dataclass
class GroupedLinearEnsemble:
    members: list[dict]
    n_features: int
    preprocessing: str
    channel_names: tuple[str, ...] | None
    config: dict
    seed: int
    training_subjects: list[str]

    def predict_session(self, session: Session) -> SessionPrediction:
        if session.preprocessing != self.preprocessing:
            raise ValueError("Session preprocessing differs from the fitted model")
        if self.channel_names is not None and tuple(session.channel_names or ()) != self.channel_names:
            raise ValueError("Session EEG channel order differs from the fitted model")
        x = np.asarray(session.x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("Session feature schema differs from the fitted model")
        accepted, reason = _session_quality(session)
        scores = np.full(len(x), np.nan, dtype=float)
        if accepted.any():
            if not np.isfinite(x[accepted]).all():
                raise ValueError("Accepted EEG features must be finite")
            transformed = transform_features(x[accepted], "relative_log_power")
            scores[accepted] = linear_ensemble_scores(transformed, self.members)
        return SessionPrediction(score=scores, accepted=accepted.copy(), reason=reason)

    def to_checkpoint(self, metadata: dict, *, feature_key: str = FEATURE_KEY,
                      extra: dict | None = None) -> dict:
        metadata = copy.deepcopy(metadata)
        metadata.setdefault("preprocessing", self.preprocessing)
        metadata.setdefault("channel_names", list(self.channel_names) if self.channel_names is not None else None)
        if metadata["preprocessing"] != self.preprocessing or (
            metadata["channel_names"] is not None
            and tuple(metadata["channel_names"]) != self.channel_names
        ):
            raise ValueError("Checkpoint metadata must match the fitted EEG schema")
        if metadata.get("target_type", "binary") != "binary" or metadata.get("label_threshold", 0.5) != 0.5:
            raise ValueError("Grouped binary ensemble requires target_type=binary and label_threshold=0.5")
        checkpoint = {
            "format_version": 1, "architecture": "calibrated_linear_ensemble", "context": 1,
            "feature_key": feature_key, "feature_transform": "relative_log_power",
            "mean": [0.0] * self.n_features, "std": [1.0] * self.n_features,
            "members": copy.deepcopy(self.members), "decision_threshold": 0.5,
            "label_threshold": float(metadata.get("label_threshold", 0.5)),
            "preprocessing": self.preprocessing,
            "channel_names": list(self.channel_names) if self.channel_names is not None else None,
            "dataset_metadata": copy.deepcopy(metadata), "input_contract": metadata.get("feature_extraction"),
            "target": metadata.get("target", "fatigue"), "target_type": "binary",
            "fit_config": copy.deepcopy(self.config), "seed": self.seed,
            "training_subjects": list(self.training_subjects),
            "calibration_protocol": "disjoint training-subject folds; calibrated member ensemble",
        }
        if "preprocessing_config" in metadata:
            checkpoint["preprocessing_config"] = copy.deepcopy(metadata["preprocessing_config"])
        if extra:
            overlap = set(extra).intersection(checkpoint)
            if overlap:
                raise ValueError(f"Extra checkpoint metadata cannot replace fitted parameters: {sorted(overlap)}")
            checkpoint.update(copy.deepcopy(extra))
        return checkpoint

    def save_checkpoint(self, path: str | Path, metadata: dict, *, feature_key: str = FEATURE_KEY,
                        extra: dict | None = None) -> Path:
        path = Path(path)
        if path.exists():
            raise ValueError(f"Checkpoint exists: {path}. Use a new path to preserve prior models.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_checkpoint(metadata, feature_key=feature_key, extra=extra), path)
        return path


def fit_grouped_model(sessions: list[Session], config: dict, seed: int = 42) -> GroupedLinearEnsemble:
    """Fit each scaler/classifier and its calibrator on disjoint training subjects."""
    if not sessions or len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError("Fitting requires nonempty, unique training sessions")
    classifier = config.get("classifier", "lda")
    if classifier not in {"lda", "logistic"}:
        raise ValueError("Grouped linear fitting supports lda or logistic")
    if config.get("feature_transform", "relative_log_power") != "relative_log_power":
        raise ValueError("Grouped linear ensemble requires relative log power")
    n_folds = config.get("calibration_folds", 3)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("calibration_folds must be an integer >= 2")
    schema = {(session.x.shape[1], session.preprocessing,
               tuple(session.channel_names) if session.channel_names is not None else None)
              for session in sessions if session.x.ndim == 2}
    if len(schema) != 1 or any(session.x.ndim != 2 for session in sessions):
        raise ValueError("Training sessions must share the feature, preprocessing and channel schema")
    n_features, preprocessing, channels = next(iter(schema))
    if not n_features or n_features % 5:
        raise ValueError("Channel-major EEG features must have five bands per channel")
    if preprocessing not in {"legacy_car", "robust_v1"}:
        raise ValueError("Unsupported training preprocessing")
    if preprocessing == "robust_v1" and (not channels or len(channels) * 5 != n_features):
        raise ValueError("Robust training requires verified channel names")
    features, targets, identifiers = [], [], []
    for session in sessions:
        y = np.asarray(session.y)
        if y.shape != (len(session.x),) or not np.isin(y, [0, 1]).all() or not session.subject:
            raise ValueError("Training targets must be aligned binary values with subject identifiers")
        accepted, _ = _session_quality(session)
        if not accepted.any():
            continue
        if not np.isfinite(session.x[accepted]).all():
            raise ValueError("Accepted training EEG must be finite")
        features.append(session.x[accepted])
        targets.append(y[accepted].astype(int))
        identifiers.append(np.repeat(session.subject, accepted.sum()))
    if not features:
        raise ValueError("No quality-accepted training features")
    x = transform_features(np.concatenate(features), "relative_log_power").astype(np.float64)
    y, groups = np.concatenate(targets), np.concatenate(identifiers)
    subjects = sorted(str(subject) for subject in set(groups))
    if set(subjects) != {session.subject for session in sessions}:
        raise ValueError("Each training subject must retain quality-accepted windows")
    if len(subjects) < 2:
        raise ValueError("Disjoint group calibration requires at least two training subjects")
    shuffled = np.random.default_rng(seed).permutation(subjects)
    calibration_groups = np.array_split(shuffled, min(n_folds, len(subjects)))
    members = []
    for calibration_subjects in calibration_groups:
        held = np.isin(groups, calibration_subjects)
        fit = ~held
        if len(np.unique(y[fit])) != 2:
            raise ValueError("Each classifier fitting fold requires both target classes")
        weights = _subject_class_weights(y[fit], groups[fit])
        scaler = StandardScaler().fit(x[fit], sample_weight=weights)
        if classifier == "lda":
            shrinkage = config.get("shrinkage", "auto")
            if shrinkage != "auto":
                raise ValueError("LDA ensemble currently requires shrinkage='auto'")
            estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
            estimator.fit(scaler.transform(x[fit]), y[fit])
        else:
            c = float(config.get("C", 0.1))
            if not np.isfinite(c) or c <= 0:
                raise ValueError("Logistic C must be finite and positive")
            estimator = LogisticRegression(C=c, max_iter=2000, random_state=seed)
            estimator.fit(scaler.transform(x[fit]), y[fit], sample_weight=weights)
        held_decision = estimator.decision_function(scaler.transform(x[held]))
        calibration = fit_calibration(y[held], held_decision, groups[held], label_threshold=0.5,
                                      input_type="decision")
        calibration["training_source"] = "disjoint_training_subject_calibration_fold"
        members.append({
            "mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(),
            "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
            "calibration": calibration, "fit_subjects": sorted(str(subject) for subject in set(groups[fit])),
            "calibration_subjects": sorted(str(subject) for subject in set(groups[held])),
            "n_fit_windows": int(fit.sum()), "n_calibration_windows": int(held.sum()),
        })
    return GroupedLinearEnsemble(members, n_features, preprocessing, channels,
                                 copy.deepcopy(config), seed, subjects)
