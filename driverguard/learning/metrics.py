"""Evaluation for PERCLOS-derived reduced vigilance, not a physiological diagnosis.

Targets and predictions are continuous values in [0, 1]. The reference binary label is
``y_true >= label_threshold``. A decision threshold must be selected on validation data
and then held fixed for test metrics, per-group reports and confidence intervals.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_INTERVAL_METRICS = (
    "mae", "rmse", "pearson", "spearman", "balanced_accuracy", "macro_f1",
    "precision", "recall", "auroc", "auprc",
)


def _inputs(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if truth.size == 0:
        raise ValueError("metrics require at least one observation")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("targets and predictions must be finite")
    if ((truth < 0) | (truth > 1)).any() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("targets and predictions must be in [0, 1]")
    return truth, prediction


def _threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _group_inputs(groups: Sequence[str], size: int) -> np.ndarray:
    identifiers = np.asarray(groups)
    if identifiers.ndim != 1 or identifiers.size != size:
        raise ValueError("groups must contain one identifier per observation")
    if not all(isinstance(group, (str, np.str_)) and group for group in identifiers):
        raise ValueError("groups must be nonempty strings")
    return identifiers.astype(str)


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def vigilance_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    decision_threshold: float = 0.35,
    label_threshold: float = 0.35,
) -> dict[str, Any]:
    """Score continuous vigilance estimates and their thresholded predictions.

    Macro F1 always averages the two binary classes. Balanced accuracy averages recall
    over classes present in the reference labels. AUC/AP require both reference classes;
    correlations require nonconstant inputs. Undefined values are returned as ``None``.
    The confusion matrix uses rows=true, columns=predicted: [[TN, FP], [FN, TP]].
    """
    from scipy.stats import rankdata
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    truth, prediction = _inputs(y_true, y_pred)
    decision_threshold = _threshold(decision_threshold, "decision_threshold")
    label_threshold = _threshold(label_threshold, "label_threshold")
    reference, predicted = truth >= label_threshold, prediction >= decision_threshold
    tn = int(np.sum(~reference & ~predicted))
    fp = int(np.sum(~reference & predicted))
    fn = int(np.sum(reference & ~predicted))
    tp = int(np.sum(reference & predicted))
    class_recalls = []
    if tn + fp:
        class_recalls.append(tn / (tn + fp))
    if tp + fn:
        class_recalls.append(tp / (tp + fn))
    both_classes = bool(reference.any() and (~reference).any())
    difference = truth - prediction
    return {
        "n": int(truth.size),
        "label_threshold": label_threshold,
        "decision_threshold": decision_threshold,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "pearson": _correlation(truth, prediction),
        "spearman": _correlation(rankdata(truth), rankdata(prediction)),
        "accuracy": float(np.mean(reference == predicted)),
        "specificity": tn / (tn + fp) if tn + fp else None,
        "balanced_accuracy": float(np.mean(class_recalls)),
        "macro_f1": float(f1_score(reference, predicted, labels=[False, True], average="macro", zero_division=0)),
        "precision": float(precision_score(reference, predicted, zero_division=0)),
        "recall": float(recall_score(reference, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(reference, prediction)) if both_classes else None,
        "auprc": float(average_precision_score(reference, prediction)) if both_classes else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_threshold(
    y_true: Sequence[float], y_pred: Sequence[float], label_threshold: float = 0.35,
) -> float:
    """Select a threshold on VALIDATION predictions by maximum balanced accuracy.

    Both reference classes must be present. Ties choose the candidate nearest to the
    label threshold, then the smaller candidate. Sorting makes selection O(n log n).
    Never call this function on test predictions.
    """
    truth, prediction = _inputs(y_true, y_pred)
    label_threshold = _threshold(label_threshold, "label_threshold")
    labels = truth >= label_threshold
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        raise ValueError("threshold selection requires both reference classes in validation data")
    order = np.argsort(prediction, kind="stable")
    scores, labels = prediction[order], labels[order]
    candidates = np.unique(np.concatenate((scores, [0.0, 1.0, label_threshold])))
    cuts = np.searchsorted(scores, candidates, side="left")
    positive_prefix = np.concatenate(([0], np.cumsum(labels, dtype=int)))
    negative_prefix = np.concatenate(([0], np.cumsum(~labels, dtype=int)))
    sensitivity = (positives - positive_prefix[cuts]) / positives
    specificity = negative_prefix[cuts] / negatives
    balanced_accuracy = (sensitivity + specificity) / 2
    best = candidates[np.isclose(balanced_accuracy, balanced_accuracy.max(), rtol=0, atol=1e-12)]
    return float(best[np.lexsort((best, np.abs(best - label_threshold)))[0]])


def grouped_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[str],
    decision_threshold: float = 0.35,
    label_threshold: float = 0.35,
) -> dict[str, Any]:
    """Return pooled and per-group scores using one frozen decision threshold."""
    truth, prediction = _inputs(y_true, y_pred)
    identifiers = _group_inputs(groups, truth.size)
    return {
        "overall": vigilance_metrics(truth, prediction, decision_threshold, label_threshold),
        "per_group": {
            group: vigilance_metrics(
                truth[identifiers == group], prediction[identifiers == group], decision_threshold, label_threshold,
            )
            for group in np.unique(identifiers)
        },
    }


def bootstrap_group_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[str],
    decision_threshold: float = 0.35,
    label_threshold: float = 0.35,
    n_bootstrap: int = 300,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile intervals from resampling whole groups, usually held-out subjects.

    Each draw samples the original number of groups with replacement, retaining every
    observation in each sampled group. These are pooled, observation-weighted metrics;
    group sizes are preserved. Undefined draws are excluded per metric and counted by
    ``n_valid``. Thresholds remain frozen throughout all bootstrap draws.
    """
    truth, prediction = _inputs(y_true, y_pred)
    identifiers = _group_inputs(groups, truth.size)
    decision_threshold = _threshold(decision_threshold, "decision_threshold")
    label_threshold = _threshold(label_threshold, "label_threshold")
    unique_groups = np.unique(identifiers)
    if unique_groups.size < 2:
        raise ValueError("group bootstrap requires at least two distinct groups")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    indices = [np.flatnonzero(identifiers == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in _INTERVAL_METRICS}
    for _ in range(n_bootstrap):
        selected = np.concatenate([indices[int(group)] for group in rng.integers(0, len(indices), len(indices))])
        metrics = vigilance_metrics(truth[selected], prediction[selected], decision_threshold, label_threshold)
        for metric in _INTERVAL_METRICS:
            if metrics[metric] is not None:
                samples[metric].append(metrics[metric])
    tail = (1 - confidence) / 2
    intervals = {}
    for metric, values in samples.items():
        lo, hi = np.quantile(values, [tail, 1 - tail]) if values else (None, None)
        intervals[metric] = {
            "low": float(lo) if lo is not None else None,
            "high": float(hi) if hi is not None else None,
            "n_valid": len(values),
        }
    return {
        "n_groups": int(unique_groups.size),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "confidence": float(confidence),
        "decision_threshold": decision_threshold,
        "label_threshold": label_threshold,
        "intervals": intervals,
    }
