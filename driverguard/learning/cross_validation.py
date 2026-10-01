"""Nested subject LOSO with training-only selection, calibration and abstention.

Factories own feature extraction and fitting; this module owns split integrity. Every
outer subject is unseen by both the fitted model and its decision threshold. This is
development cross-validation on one cohort, not independent external validation.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from .data import Session
from .metrics import vigilance_metrics


@dataclass
class SessionPrediction:
    score: np.ndarray
    accepted: np.ndarray | None = None
    reason: np.ndarray | None = None


class FittedModel(Protocol):
    def predict_session(self, session: Session) -> SessionPrediction: ...


ModelFactory = Callable[[list[Session], dict, int], FittedModel]


def _subject_weights(groups: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def apply_calibration(scores: np.ndarray, calibration: Mapping[str, Any]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if calibration["method"] == "identity":
        return scores.copy()
    if calibration["method"] == "sigmoid_decision":
        return expit(float(calibration["slope"]) * scores + float(calibration["intercept"]))
    if calibration["method"] != "sigmoid_logit":
        raise ValueError("Unknown probability calibration method")
    return expit(float(calibration["slope"]) * logit(np.clip(scores, 1e-6, 1 - 1e-6))
                 + float(calibration["intercept"]))


def fit_calibration(target: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                    *, label_threshold: float = 0.5, input_type: str = "probability") -> dict:
    """Fit a monotonic sigmoid to subject-weighted training OOF predictions."""
    target = np.asarray(target, dtype=float)
    labels = (target >= label_threshold).astype(float)
    scores, groups = np.asarray(scores, dtype=float), np.asarray(groups)
    if input_type not in {"probability", "decision"}:
        raise ValueError("Calibration input_type must be probability or decision")
    if (labels.ndim != 1 or scores.shape != labels.shape or groups.shape != labels.shape
            or not len(labels) or not np.isfinite(target).all() or np.any((target < 0) | (target > 1))
            or not np.isfinite(scores).all()
            or (input_type == "probability" and np.any((scores < 0) | (scores > 1)))):
        raise ValueError("Calibration requires matching nonempty finite predictions and groups")
    if len(np.unique(labels)) != 2:
        if input_type == "decision":
            return {"method": "sigmoid_decision", "slope": 1.0, "intercept": 0.0,
                    "reason": "single_class_calibration", "n_subjects": len(np.unique(groups))}
        return {"method": "identity", "reason": "single_class_calibration",
                "n_subjects": len(np.unique(groups))}
    decision = scores if input_type == "decision" else logit(np.clip(scores, 1e-6, 1 - 1e-6))
    design = np.column_stack((decision, np.ones(len(scores))))
    weights = _subject_weights(groups)

    def objective(parameters):
        linear = design @ parameters
        regularization = 1e-3 * ((parameters[0] - 1) ** 2 + parameters[1] ** 2)
        value = float(np.sum(weights * (np.logaddexp(0, linear) - labels * linear)) + regularization)
        gradient = design.T @ (weights * (expit(linear) - labels))
        gradient += 2e-3 * np.asarray([parameters[0] - 1, parameters[1]])
        return value, gradient

    fit = minimize(objective, np.asarray([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=((0.0, 10.0), (-20.0, 20.0)))
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Probability calibration failed: {fit.message}")
    return {"method": "sigmoid_decision" if input_type == "decision" else "sigmoid_logit",
            "slope": float(fit.x[0]), "intercept": float(fit.x[1]),
            "n_subjects": len(np.unique(groups)), "subject_weighted": True,
            "training_source": "inner_subject_out_of_fold_predictions"}


def select_group_threshold(target: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                           *, label_threshold: float = 0.5) -> float:
    """Optimize mean subject balanced accuracy, preserving equal subject weight."""
    labels = np.asarray(target) >= label_threshold
    scores, groups = np.asarray(scores, dtype=float), np.asarray(groups)
    if (labels.ndim != 1 or scores.shape != labels.shape or groups.shape != labels.shape
            or not len(labels) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Threshold selection requires matching finite predictions and groups")
    if not labels.any() or labels.all():
        raise ValueError("Threshold selection requires both reference classes")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    subject_values = []
    for subject in np.unique(groups):
        selected = groups == subject
        order = np.argsort(scores[selected], kind="stable")
        sorted_labels = labels[selected][order]
        cuts = np.searchsorted(scores[selected][order], candidates, side="left")
        positive_prefix = np.concatenate(([0], np.cumsum(sorted_labels)))
        negative_prefix = np.concatenate(([0], np.cumsum(~sorted_labels)))
        recalls = []
        if sorted_labels.sum():
            recalls.append((sorted_labels.sum() - positive_prefix[cuts]) / sorted_labels.sum())
        if (~sorted_labels).sum():
            recalls.append(negative_prefix[cuts] / (~sorted_labels).sum())
        subject_values.append(np.mean(recalls, axis=0))
    values = np.mean(subject_values, axis=0)
    best = candidates[np.isclose(values, values.max(), rtol=0, atol=1e-12)]
    return float(best[np.lexsort((best, np.abs(best - 0.5)))[0]])


def _prediction_rows(model: FittedModel, sessions: list[Session]) -> list[dict]:
    rows = []
    for session in sessions:
        prediction = model.predict_session(session)
        scores = np.asarray(prediction.score, dtype=float)
        accepted = (np.ones(len(scores), dtype=bool) if prediction.accepted is None
                    else np.asarray(prediction.accepted, dtype=bool))
        reasons = (np.full(len(scores), "", dtype=str) if prediction.reason is None
                   else np.asarray(prediction.reason, dtype=str))
        if scores.shape != session.y.shape or accepted.shape != scores.shape or reasons.shape != scores.shape:
            raise ValueError("Model predictions, quality mask and reasons must preserve all session windows")
        if not np.isfinite(scores[accepted]).all() or np.any((scores[accepted] < 0) | (scores[accepted] > 1)):
            raise ValueError("Accepted model predictions must be finite and in [0,1]")
        for index, (score, valid, reason, target) in enumerate(zip(scores, accepted, reasons, session.y)):
            rows.append({"subject": session.subject, "session_id": session.session_id,
                         "window_index": index, "target": float(target), "accepted": bool(valid),
                         "abstention_reason": str(reason) if not valid else "",
                         "raw_score": float(score) if valid else None})
    return rows


def _arrays(rows: Sequence[dict]):
    valid = [row for row in rows if row["accepted"]]
    if not valid:
        raise ValueError("No accepted windows available for selection or evaluation")
    return (np.asarray([row["target"] for row in valid]),
            np.asarray([row.get("prediction", row["raw_score"]) for row in valid]),
            np.asarray([row["subject"] for row in valid]))


def _macro_selection_metrics(rows: list[dict], label_threshold: float) -> dict:
    target, scores, groups = _arrays(rows)
    losses, accuracies, aurocs = [], [], []
    for subject in np.unique(groups):
        selected = groups == subject
        labels = target[selected] >= label_threshold
        metric = vigilance_metrics(target[selected], scores[selected], 0.5, label_threshold)
        losses.append(float(log_loss(labels, scores[selected], labels=[False, True])))
        accuracies.append(metric["balanced_accuracy"])
        if metric["auroc"] is not None:
            aurocs.append(metric["auroc"])
    return {"mean_subject_log_loss": float(np.mean(losses)),
            "mean_subject_balanced_accuracy_at_0_5": float(np.mean(accuracies)),
            "mean_subject_auroc": float(np.mean(aurocs)) if aurocs else None,
            "coverage": len(target) / len(rows), "n_subjects": len(np.unique(groups))}


def _inner_select(sessions: list[Session], configs: list[dict], fit_model: ModelFactory,
                  seed: int, label_threshold: float) -> tuple[dict, list[dict], dict]:
    subjects = sorted({session.subject for session in sessions})
    results, predictions = [], []
    for config_index, config in enumerate(configs):
        rows = []
        for subject in subjects:
            training = [session for session in sessions if session.subject != subject]
            held_out = [session for session in sessions if session.subject == subject]
            model = fit_model(training, copy.deepcopy(config), seed)
            rows.extend(_prediction_rows(model, held_out))
        metric = _macro_selection_metrics(rows, label_threshold)
        if metric["n_subjects"] != len(subjects):
            raise ValueError("Every inner validation subject must retain at least one accepted window")
        results.append({"config_index": config_index, "config": copy.deepcopy(config), **metric})
        predictions.append(rows)
    selected = min(results, key=lambda row: (row["mean_subject_log_loss"],
                                            -row["mean_subject_balanced_accuracy_at_0_5"], row["config_index"]))
    index = selected["config_index"]
    selection = {"criterion": "minimum mean-subject log loss, then maximum balanced accuracy at 0.5",
                 "subjects": subjects, "selected_config_index": index, "candidates": results,
                 "inner_folds": [{"validation_subject": subject,
                                  "training_subjects": [other for other in subjects if other != subject]}
                                 for subject in subjects]}
    return copy.deepcopy(configs[index]), predictions[index], selection


def _calibrate_selected(rows: list[dict], label_threshold: float,
                        method: str = "sigmoid", threshold_mode: str = "fixed_0.5") -> tuple[dict, float, dict]:
    target, scores, groups = _arrays(rows)
    if method == "none":
        threshold = (0.5 if threshold_mode == "fixed_0.5" else
                     select_group_threshold(target, scores, groups, label_threshold=label_threshold))
        return {"method": "identity", "reason": "calibration_disabled"}, threshold, {
            "threshold_source": "predefined_0.5" if threshold_mode == "fixed_0.5"
            else "inner_subject_out_of_fold_predictions",
            "threshold_criterion": threshold_mode, "calibration_folds": []}
    cross_fit = np.empty_like(scores)
    calibration_folds = []
    for subject in np.unique(groups):
        held_out = groups == subject
        calibration = fit_calibration(target[~held_out], scores[~held_out], groups[~held_out],
                                      label_threshold=label_threshold)
        cross_fit[held_out] = apply_calibration(scores[held_out], calibration)
        calibration_folds.append({"validation_subject": subject,
                                  "training_subjects": sorted(set(groups[~held_out].tolist()))})
    threshold = (0.5 if threshold_mode == "fixed_0.5" else
                 select_group_threshold(target, cross_fit, groups, label_threshold=label_threshold))
    final_calibration = fit_calibration(target, scores, groups, label_threshold=label_threshold)
    return final_calibration, threshold, {
        "threshold_source": "predefined_0.5" if threshold_mode == "fixed_0.5"
        else "subject_cross_fitted_calibrated_inner_oof_predictions",
        "threshold_criterion": threshold_mode,
        "calibration_folds": calibration_folds,
        "cross_fitted_log_loss": float(log_loss(target >= label_threshold, cross_fit, labels=[False, True],
                                               sample_weight=_subject_weights(groups))),
    }


def _freeze_rows(rows: list[dict], calibration: dict, threshold: float, family: str, fold: str) -> None:
    for row in rows:
        row.update({"family": family, "outer_fold": fold, "decision_threshold": float(threshold)})
        if row["accepted"]:
            score = float(apply_calibration(np.asarray([row["raw_score"]]), calibration)[0])
            row.update({"prediction": score, "reduced_vigilance_pred": int(score >= threshold)})
        else:
            row.update({"prediction": None, "reduced_vigilance_pred": None})


def _frozen_metrics(rows: Sequence[dict], label_threshold: float) -> dict | None:
    valid = [row for row in rows if row["accepted"]]
    if not valid:
        return None
    target, scores, _ = _arrays(valid)
    labels = target >= label_threshold
    decisions = np.asarray([row["reduced_vigilance_pred"] for row in valid], dtype=bool)
    metric = vigilance_metrics(target, scores, 0.5, label_threshold)
    tn, fp = int(np.sum(~labels & ~decisions)), int(np.sum(~labels & decisions))
    fn, tp = int(np.sum(labels & ~decisions)), int(np.sum(labels & decisions))
    recalls = ([tn / (tn + fp)] if tn + fp else []) + ([tp / (tp + fn)] if tp + fn else [])
    metric.update({"decision_threshold": None, "threshold_policy": "frozen_per_outer_fold",
                   "confusion_matrix": [[tn, fp], [fn, tp]], "accuracy": float(np.mean(labels == decisions)),
                   "specificity": tn / (tn + fp) if tn + fp else None,
                   "balanced_accuracy": float(np.mean(recalls)),
                   "macro_f1": float(f1_score(labels, decisions, labels=[False, True],
                                              average="macro", zero_division=0)),
                   "precision": float(precision_score(labels, decisions, zero_division=0)),
                   "recall": float(recall_score(labels, decisions, zero_division=0)),
                   "log_loss": float(log_loss(labels, scores, labels=[False, True])),
                   "brier_score": float(np.mean((labels.astype(float) - scores) ** 2))})
    return metric


def _coverage(rows: Sequence[dict], label_threshold: float) -> dict:
    accepted = sum(row["accepted"] for row in rows)
    reasons: dict[str, int] = {}
    for row in rows:
        if not row["accepted"]:
            reason = row["abstention_reason"] or "unspecified_quality_failure"
            reasons[reason] = reasons.get(reason, 0) + 1
    return {"n_total": len(rows), "n_accepted": accepted, "n_abstained": len(rows) - accepted,
            "coverage": accepted / len(rows) if rows else 0.0, "abstention_reasons": reasons,
            "by_class": {str(label): {
                "n_total": sum(int(row["target"] >= label_threshold) == label for row in rows),
                "n_accepted": sum(row["accepted"] and int(row["target"] >= label_threshold) == label
                                  for row in rows)} for label in (0, 1)}}


def coverage_adjusted_metrics(rows: Sequence[dict], label_threshold=0.5) -> dict:
    """Count abstentions as incorrect for their true class, retaining all windows."""
    recalls, by_class = [], {}
    for label in (0, 1):
        selected = [row for row in rows if int(row["target"] >= label_threshold) == label]
        correct = sum(row["accepted"] and row["reduced_vigilance_pred"] == label for row in selected)
        recall = correct / len(selected) if selected else None
        by_class[str(label)] = {"n_total": len(selected), "n_correct": correct, "recall": recall}
        if recall is not None:
            recalls.append(recall)
    return {"n_total": len(rows), "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "by_class": by_class, "abstention_policy": "incorrect_for_true_class"}


def _percentile_interval(samples: Sequence[float]) -> dict:
    low, high = np.quantile(samples, [0.025, 0.975]) if len(samples) else (None, None)
    return {"low": float(low) if low is not None else None,
            "high": float(high) if high is not None else None, "n_valid": len(samples)}


def summarize_oof(rows: list[dict], *, label_threshold=0.5, bootstrap=300, seed=42) -> dict:
    """Score accepted rows with fold-frozen decisions and bootstrap entire subjects."""
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    subjects = sorted({row["subject"] for row in rows})
    if len(subjects) < 2:
        raise ValueError("Subject bootstrap requires at least two subjects")
    per_subject = {subject: {
        "metrics": _frozen_metrics([row for row in rows if row["subject"] == subject], label_threshold),
        "coverage": _coverage([row for row in rows if row["subject"] == subject], label_threshold),
        "coverage_adjusted": coverage_adjusted_metrics(
            [row for row in rows if row["subject"] == subject], label_threshold)}
        for subject in subjects}
    metric_names = ("auroc", "auprc", "balanced_accuracy", "macro_f1", "recall", "specificity",
                    "log_loss", "brier_score")
    macro = {}
    for name in metric_names:
        values = [report["metrics"][name] for report in per_subject.values()
                  if report["metrics"] is not None and report["metrics"][name] is not None]
        macro[name] = float(np.mean(values)) if values else None
    adjusted_values = [report["coverage_adjusted"]["balanced_accuracy"] for report in per_subject.values()
                       if report["coverage_adjusted"]["balanced_accuracy"] is not None]
    macro["coverage_adjusted_balanced_accuracy"] = float(np.mean(adjusted_values)) if adjusted_values else None
    by_subject = [[row for row in rows if row["subject"] == subject] for subject in subjects]
    rng = np.random.default_rng(seed)
    values = {name: [] for name in metric_names}
    values["coverage_adjusted_balanced_accuracy"] = []
    macro_values = {name: [] for name in values}
    for _ in range(bootstrap):
        chosen = rng.integers(0, len(subjects), len(subjects))
        drawn = [row for index in chosen for row in by_subject[index]]
        adjusted = coverage_adjusted_metrics(drawn, label_threshold)["balanced_accuracy"]
        if adjusted is not None:
            values["coverage_adjusted_balanced_accuracy"].append(adjusted)
        metric = _frozen_metrics(drawn, label_threshold)
        if metric is not None:
            for name in metric_names:
                if metric[name] is not None:
                    values[name].append(metric[name])
        for name, metric_samples in macro_values.items():
            subject_samples = []
            for index in chosen:
                subject_report = per_subject[subjects[index]]
                if name == "coverage_adjusted_balanced_accuracy":
                    subject_value = subject_report["coverage_adjusted"]["balanced_accuracy"]
                else:
                    subject_value = (subject_report["metrics"][name]
                                     if subject_report["metrics"] is not None else None)
                if subject_value is not None:
                    subject_samples.append(subject_value)
            if subject_samples:
                metric_samples.append(float(np.mean(subject_samples)))
    intervals = {name: _percentile_interval(samples) for name, samples in values.items()}
    macro_intervals = {name: _percentile_interval(samples) for name, samples in macro_values.items()}
    return {"overall": _frozen_metrics(rows, label_threshold), "subject_macro": macro,
            "per_subject": per_subject, "coverage": _coverage(rows, label_threshold),
            "coverage_adjusted": coverage_adjusted_metrics(rows, label_threshold),
            "confidence_intervals": {"method": "whole-subject percentile bootstrap",
                                     "confidence": 0.95, "n_bootstrap": int(bootstrap), "seed": int(seed),
                                     "n_subjects": len(subjects), "intervals": intervals,
                                     "interval_population": "pooled accepted windows; coverage-adjusted uses all",
                                     "pooled_intervals": intervals, "subject_macro_intervals": macro_intervals},
            "metric_population": "accepted windows only; abstentions retained in coverage and predictions"}


def paired_comparison(baseline_rows: list[dict], candidate_rows: list[dict], *, label_threshold=0.5,
                      bootstrap=300, seed=42) -> dict:
    """Compare predefined profiles by resampling the same held-out subjects."""
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")

    def index_rows(rows):
        indexed = {(row["subject"], row["session_id"], row["window_index"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("Comparison requires unique subject/session/window keys in each profile")
        return indexed

    baseline, candidate = index_rows(baseline_rows), index_rows(candidate_rows)
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired profiles must preserve identical subject/session/window coverage")
    if any(baseline[key]["target"] != candidate[key]["target"] for key in baseline):
        raise ValueError("Paired profiles must have identical targets")
    subjects = sorted({key[0] for key in baseline})
    if len(subjects) < 2:
        raise ValueError("Paired bootstrap requires at least two subjects")
    names = ("auroc", "auprc", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
             "coverage_adjusted_balanced_accuracy", "coverage")
    arrays = {name: [[], []] for name in names}
    matched_counts = []
    for subject in subjects:
        subject_baseline = [row for row in baseline_rows if row["subject"] == subject]
        subject_candidate = [row for row in candidate_rows if row["subject"] == subject]
        matched_counts.append(sum(baseline[key]["accepted"] and candidate[key]["accepted"]
                                  for key in baseline if key[0] == subject))
        for profile_index, rows in enumerate((subject_baseline, subject_candidate)):
            metrics = _frozen_metrics(rows, label_threshold)
            for name in names:
                if name == "coverage_adjusted_balanced_accuracy":
                    value = coverage_adjusted_metrics(rows, label_threshold)["balanced_accuracy"]
                elif name == "coverage":
                    value = _coverage(rows, label_threshold)["coverage"]
                else:
                    value = metrics[name] if metrics is not None else None
                arrays[name][profile_index].append(value)
    rng = np.random.default_rng(seed)
    bootstrap_differences = {name: [] for name in names}
    for _ in range(bootstrap):
        selected = rng.integers(0, len(subjects), len(subjects))
        for name, profile_values in arrays.items():
            differences = [profile_values[1][index] - profile_values[0][index] for index in selected
                           if profile_values[0][index] is not None and profile_values[1][index] is not None]
            if differences:
                bootstrap_differences[name].append(float(np.mean(differences)))
    comparisons = {}
    for name, profile_values in arrays.items():
        pairs = [(first, second) for first, second in zip(*profile_values)
                 if first is not None and second is not None]
        comparisons[name] = {
            "baseline_subject_macro": float(np.mean([pair[0] for pair in pairs])) if pairs else None,
            "candidate_subject_macro": float(np.mean([pair[1] for pair in pairs])) if pairs else None,
            "difference": float(np.mean([pair[1] - pair[0] for pair in pairs])) if pairs else None,
            "n_subject_pairs": len(pairs),
            "confidence_interval": _percentile_interval(bootstrap_differences[name]),
        }
    return {"method": "paired whole-subject percentile bootstrap of mean subject differences",
            "n_subjects": len(subjects), "subjects": subjects, "n_bootstrap": int(bootstrap),
            "seed": int(seed), "confidence": 0.95, "difference_direction": "candidate_minus_baseline",
            "metrics": comparisons, "n_windows_accepted_by_both_profiles": sum(matched_counts),
            "coverage": {"baseline": _coverage(baseline_rows, label_threshold),
                         "candidate": _coverage(candidate_rows, label_threshold)},
            "limitations": ["Conditional metrics can compare different accepted subsets",
                            "Coverage-adjusted balanced accuracy counts abstentions as incorrect",
                            "Intervals reflect this cohort and do not establish external generalization"]}


def fit_deployment(sessions: list[Session], *, candidates: Mapping[str, list[dict]], fit_model: ModelFactory,
                   label_threshold=0.5, seed=42, calibration: str = "none",
                   threshold_mode: str = "fixed_0.5",
                   progress: Callable[[str], None] | None = None) -> dict:
    """Select on cohort OOF predictions and refit; no independent test score exists."""
    subjects = sorted({session.subject for session in sessions})
    if len(subjects) < 4:
        raise ValueError("Deployment selection requires at least four distinct subjects")
    if not candidates or any(not configs for configs in candidates.values()):
        raise ValueError("Each model family must define at least one candidate")
    if calibration not in {"sigmoid", "none"}:
        raise ValueError("calibration must be sigmoid or none")
    if threshold_mode not in {"fixed_0.5", "inner_balanced_accuracy"}:
        raise ValueError("threshold_mode must be fixed_0.5 or inner_balanced_accuracy")
    deployment = {}
    for family, configs in candidates.items():
        if progress:
            progress(f"Refitting deployment {family} on all development subjects")
        config, inner_rows, selection = _inner_select(sessions, configs, fit_model, seed, label_threshold)
        fitted_calibration, threshold, calibration_audit = _calibrate_selected(
            inner_rows, label_threshold, calibration, threshold_mode)
        deployment[family] = {"model": fit_model(sessions, config, seed), "config": config,
                              "calibration": fitted_calibration, "decision_threshold": threshold,
                              "inner_selection": selection, "calibration_audit": calibration_audit,
                              "training_subjects": subjects,
                              "evaluation_status": "all_cohort_fit_without_independent_test_score"}
    return deployment


def nested_loso(sessions: list[Session], *, candidates: Mapping[str, list[dict]], fit_model: ModelFactory,
                label_threshold=0.5, bootstrap=300, seed=42,
                calibration: str = "sigmoid", refit_deployment: bool = False,
                threshold_mode: str = "fixed_0.5",
                progress: Callable[[str], None] | None = None) -> dict:
    """Benchmark predefined families and refit one deployment model per family.

    The caller must freeze candidate grids before inspecting the outer-fold reports.
    Returned deployment model objects are deliberately excluded from the JSON report.
    """
    subjects = sorted({session.subject for session in sessions})
    if len(subjects) < 4:
        raise ValueError("Nested subject LOSO requires at least four distinct subjects")
    if not candidates or any(not configs for configs in candidates.values()):
        raise ValueError("Each model family must define at least one candidate")
    if not 0 < label_threshold < 1 or not np.isfinite(label_threshold):
        raise ValueError("label_threshold must be finite and strictly between zero and one")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if calibration not in {"sigmoid", "none"}:
        raise ValueError("calibration must be sigmoid or none")
    if threshold_mode not in {"fixed_0.5", "inner_balanced_accuracy"}:
        raise ValueError("threshold_mode must be fixed_0.5 or inner_balanced_accuracy")
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError("Session IDs must be unique")
    for session in sessions:
        if (not session.subject or not session.session_id or session.y.ndim != 1 or not len(session.y)
                or len(session.x) != len(session.y) or not np.isfinite(session.y).all()
                or np.any((session.y < 0) | (session.y > 1))):
            raise ValueError("Sessions must preserve finite aligned targets and nonempty subject identifiers")
    folds, all_rows, deployment = [], [], {}
    for subject in subjects:
        training = [session for session in sessions if session.subject != subject]
        held_out = [session for session in sessions if session.subject == subject]
        for family, configs in candidates.items():
            if progress:
                progress(f"Outer subject {subject}: selecting {family} using training subjects only")
            config, inner_rows, selection = _inner_select(training, configs, fit_model, seed, label_threshold)
            fitted_calibration, threshold, calibration_audit = _calibrate_selected(
                inner_rows, label_threshold, calibration, threshold_mode)
            model = fit_model(training, config, seed)
            rows = _prediction_rows(model, held_out)
            _freeze_rows(rows, fitted_calibration, threshold, family, subject)
            all_rows.extend(rows)
            folds.append({"test_subject": subject, "family": family,
                          "training_subjects": [other for other in subjects if other != subject],
                          "selected_config": config, "selection": selection, "calibration": fitted_calibration,
                          "decision_threshold": threshold, "calibration_audit": calibration_audit,
                          "test_metrics": _frozen_metrics(rows, label_threshold),
                          "test_coverage": _coverage(rows, label_threshold),
                          "test_coverage_adjusted": coverage_adjusted_metrics(rows, label_threshold)})
    family_reports = {family: summarize_oof([row for row in all_rows if row["family"] == family],
                                            label_threshold=label_threshold, bootstrap=bootstrap, seed=seed)
                      for family in candidates}
    if refit_deployment:
        deployment = fit_deployment(sessions, candidates=candidates, fit_model=fit_model,
                                    label_threshold=label_threshold, seed=seed, calibration=calibration,
                                    threshold_mode=threshold_mode, progress=progress)
    report = {"protocol": "nested leave-one-subject-out development cross-validation",
              "n_outer_folds": len(subjects), "subjects": subjects, "label_threshold": float(label_threshold),
              "probability_calibration": calibration, "refit_deployment": refit_deployment,
              "threshold_mode": threshold_mode,
              "candidate_grids": copy.deepcopy(dict(candidates)), "folds": folds, "families": family_reports,
              "deployment": {family: {key: value for key, value in model.items() if key != "model"}
                             for family, model in deployment.items()},
              "limitations": ["All folds share one development cohort; external validation is still required",
                              "Deployment models use all subjects and have no held-out score of their own",
                              "Whole-subject bootstrap does not cover feature-design or model-selection uncertainty",
                              "Conditional metrics can improve when coverage falls; compare coverage as well"]}
    return {"report": report, "rows": all_rows, "deployment": deployment}
