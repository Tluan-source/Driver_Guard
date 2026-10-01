import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from driverguard.learning.cross_validation import (
    SessionPrediction,
    apply_calibration,
    fit_calibration,
    fit_deployment,
    nested_loso,
    paired_comparison,
    select_group_threshold,
    summarize_oof,
)
from driverguard.learning.data import Session


def _sessions():
    return [Session(f"record_{subject}", str(subject),
                    np.asarray([[0.1, 1], [0.2, 0], [0.8, 1], [0.9, 1]], dtype=np.float32),
                    np.asarray([0, 0, 1, 1], dtype=np.float32), Path("unused"), Path("unused"))
            for subject in range(4)]


class RecordingModel:
    def __init__(self, training, config, calls):
        self.training = {session.subject for session in training}
        self.config = config
        calls.append((tuple(sorted(self.training)), dict(config)))

    def predict_session(self, session):
        assert session.subject not in self.training
        accepted = session.x[:, 1].astype(bool)
        scores = session.x[:, 0].astype(float)
        if self.config["name"] == "wrong":
            scores = 1 - scores
        scores[~accepted] = np.nan
        return SessionPrediction(scores, accepted, np.where(accepted, "", "artifact"))


def test_nested_subject_splits_retain_abstentions_and_never_fit_outer_subject():
    calls = []
    result = nested_loso(
        _sessions(), candidates={"robust": [{"name": "good"}, {"name": "wrong"}]},
        fit_model=lambda training, config, seed: RecordingModel(training, config, calls),
        bootstrap=12, calibration="none",
    )
    assert result["deployment"] == {}
    assert len(calls) == 4 * (3 * 2 + 1)
    assert len(result["rows"]) == 16
    report = result["report"]
    assert report["n_outer_folds"] == 4
    for fold in report["folds"]:
        subject = fold["test_subject"]
        assert subject not in fold["training_subjects"]
        assert fold["selected_config"] == {"name": "good"}
        for inner in fold["selection"]["inner_folds"]:
            assert inner["validation_subject"] not in inner["training_subjects"]
            assert subject not in inner["training_subjects"]
    summary = report["families"]["robust"]
    assert summary["coverage"]["n_total"] == 16
    assert summary["coverage"]["n_abstained"] == 4
    assert summary["coverage"]["coverage"] == 0.75
    assert summary["overall"]["n"] == 12
    assert summary["overall"]["balanced_accuracy"] == 1.0
    assert summary["coverage_adjusted"]["balanced_accuracy"] == pytest.approx(0.75)
    for row in result["rows"]:
        if not row["accepted"]:
            assert row["prediction"] is None
            assert row["reduced_vigilance_pred"] is None
            assert row["abstention_reason"] == "artifact"
    json.dumps(report, allow_nan=False)
    json.dumps(result["rows"], allow_nan=False)


def test_outer_labels_cannot_change_outer_model_selection_calibration_or_threshold():
    original = _sessions()
    changed = _sessions()
    changed[0].y = 1 - changed[0].y
    arguments = {"candidates": {"model": [{"name": "good"}, {"name": "wrong"}]},
                 "fit_model": lambda training, config, seed: RecordingModel(training, config, []),
                 "bootstrap": 4}
    first = nested_loso(original, **arguments)["report"]["folds"][0]
    second = nested_loso(changed, **arguments)["report"]["folds"][0]
    for key in ("selected_config", "selection", "calibration", "decision_threshold", "calibration_audit"):
        assert first[key] == second[key]
    assert first["test_metrics"] != second["test_metrics"]


def test_threshold_uses_equal_subject_weight_instead_of_window_count():
    target = np.asarray([0] * 50 + [1] * 50 + [0, 0, 1, 1])
    score = np.asarray([0.5] * 50 + [0.8] * 50 + [0.1, 0.1, 0.4, 0.4])
    groups = np.asarray(["large"] * 100 + ["small"] * 4)
    assert select_group_threshold(target, score, groups) == pytest.approx(0.4)


def test_monotonic_probability_calibration_is_subject_weighted():
    target = np.asarray([0] * 100 + [1])
    scores = np.full(101, 0.5)
    groups = np.asarray(["large"] * 100 + ["small"])
    calibration = fit_calibration(target, scores, groups)
    assert calibration["slope"] >= 0
    assert apply_calibration(np.asarray([0.5]), calibration)[0] == pytest.approx(0.5, abs=0.01)
    output = apply_calibration(np.asarray([0, 0.2, 0.5, 0.8, 1]), calibration)
    assert np.isfinite(output).all()
    assert np.all(np.diff(output) >= 0)
    json.dumps(calibration, allow_nan=False)


def test_bootstrap_preserves_different_frozen_fold_thresholds():
    rows = [
        {"subject": subject, "target": float(target), "accepted": True, "raw_score": score,
         "prediction": score, "reduced_vigilance_pred": int(score >= threshold),
         "decision_threshold": threshold, "abstention_reason": ""}
        for subject, threshold, scores in [("a", 0.8, [0.7, 0.9]), ("b", 0.2, [0.1, 0.3])]
        for target, score in enumerate(scores)
    ]
    first = summarize_oof(rows, bootstrap=20, seed=8)
    assert first == summarize_oof(rows, bootstrap=20, seed=8)
    assert first["overall"]["confusion_matrix"] == [[2, 0], [0, 2]]
    assert first["confidence_intervals"]["intervals"]["balanced_accuracy"] == {
        "low": 1.0, "high": 1.0, "n_valid": 20}
    assert first["confidence_intervals"]["subject_macro_intervals"]["balanced_accuracy"] == {
        "low": 1.0, "high": 1.0, "n_valid": 20}


def _comparison_rows(scores):
    return [{"subject": subject, "session_id": subject, "window_index": index,
             "target": float(index >= 2), "raw_score": score, "prediction": score,
             "accepted": True, "abstention_reason": "", "reduced_vigilance_pred": int(score >= 0.5)}
            for subject in ("a", "b") for index, score in enumerate(scores)]


def test_paired_bootstrap_keeps_subjects_together_and_penalizes_abstentions():
    baseline = _comparison_rows([0.1, 0.9, 0.1, 0.9])
    candidate = _comparison_rows([0.1, 0.2, 0.8, 0.9])
    report = paired_comparison(baseline, candidate, bootstrap=20, seed=9)
    assert report == paired_comparison(baseline, candidate, bootstrap=20, seed=9)
    assert report["metrics"]["balanced_accuracy"]["difference"] == 0.5
    assert report["metrics"]["balanced_accuracy"]["confidence_interval"] == {
        "low": 0.5, "high": 0.5, "n_valid": 20}
    for row in candidate:
        if row["subject"] == "b":
            row.update(accepted=False, prediction=None, reduced_vigilance_pred=None, abstention_reason="artifact")
    report = paired_comparison(baseline, candidate, bootstrap=20, seed=9)
    assert report["metrics"]["balanced_accuracy"]["n_subject_pairs"] == 1
    assert report["metrics"]["coverage_adjusted_balanced_accuracy"]["difference"] == 0
    assert report["metrics"]["coverage_adjusted_balanced_accuracy"]["n_subject_pairs"] == 2
    assert report["n_windows_accepted_by_both_profiles"] == 4
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="identical subject/session/window"):
        paired_comparison(baseline[:-1], candidate)


def test_invalid_prediction_lengths_and_no_coverage_are_rejected():
    class InvalidModel:
        def predict_session(self, session):
            return SessionPrediction(np.asarray([0.5]))

    with pytest.raises(ValueError, match="preserve all session windows"):
        nested_loso(_sessions(), candidates={"x": [{}]}, fit_model=lambda *_: InvalidModel(), bootstrap=2)
    with pytest.raises(ValueError, match="four distinct"):
        nested_loso(_sessions()[:3], candidates={"x": [{}]}, fit_model=lambda *_: InvalidModel(), bootstrap=2)
    with pytest.raises(ValueError, match="calibration must"):
        nested_loso(_sessions(), candidates={"x": [{}]}, fit_model=lambda *_: InvalidModel(),
                    calibration="test_labels", bootstrap=2)
    sessions = _sessions()
    for session in sessions:
        session.x[:, 1] = 0
    with pytest.raises(ValueError, match="No accepted windows"):
        nested_loso(sessions, candidates={"x": [{"name": "good"}]},
                    fit_model=lambda training, config, seed: RecordingModel(training, config, []), bootstrap=2)


def test_deployment_refit_is_explicit_and_has_no_independent_test_score():
    calls = []
    result = fit_deployment(
        _sessions(), candidates={"model": [{"name": "good"}, {"name": "wrong"}]},
        fit_model=lambda training, config, seed: RecordingModel(training, config, calls),
    )["model"]
    assert len(calls) == 4 * 2 + 1
    assert result["config"] == {"name": "good"}
    assert result["training_subjects"] == ["0", "1", "2", "3"]
    assert result["evaluation_status"] == "all_cohort_fit_without_independent_test_score"
    assert result["decision_threshold"] == 0.5
    assert result["model"].training == {"0", "1", "2", "3"}
