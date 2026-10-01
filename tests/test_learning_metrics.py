import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from driverguard.learning.metrics import (
    bootstrap_group_ci,
    grouped_metrics,
    select_threshold,
    vigilance_metrics,
)


def test_continuous_metrics_and_frozen_binary_threshold():
    metrics = vigilance_metrics([0.1, 0.3, 0.5, 0.9], [0.2, 0.6, 0.4, 0.8], decision_threshold=0.5)
    assert metrics["mae"] == pytest.approx(0.15)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.03))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5 / 6)
    assert metrics["spearman"] == pytest.approx(0.8)
    json.dumps(metrics, allow_nan=False)


def test_threshold_selected_on_validation_is_applied_to_groups():
    threshold = select_threshold([0.1, 0.2, 0.7, 0.8], [0.2, 0.6, 0.65, 0.8])
    assert threshold == pytest.approx(0.65)
    report = grouped_metrics(
        [0.1, 0.9, 0.2, 0.8], [0.1, 0.6, 0.7, 0.9], ["driver_a", "driver_a", "driver_b", "driver_b"],
        decision_threshold=threshold,
    )
    assert report["overall"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert report["per_group"]["driver_a"]["confusion_matrix"] == [[1, 0], [1, 0]]
    assert report["per_group"]["driver_b"]["confusion_matrix"] == [[0, 1], [0, 1]]
    assert all(group["decision_threshold"] == threshold for group in report["per_group"].values())


def test_custom_label_threshold_is_frozen_for_group_metrics_and_bootstrap():
    truth = [0.4, 0.6, 0.4, 0.6]
    prediction = [0.45, 0.55, 0.7, 0.8]
    groups = ["a", "a", "b", "b"]
    report = grouped_metrics(truth, prediction, groups, decision_threshold=0.65, label_threshold=0.5)
    assert report["overall"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert report["per_group"]["a"]["confusion_matrix"] == [[1, 0], [1, 0]]
    assert report["per_group"]["b"]["confusion_matrix"] == [[0, 1], [0, 1]]
    ci = bootstrap_group_ci(
        truth, prediction, groups, decision_threshold=0.65, label_threshold=0.5, n_bootstrap=20,
    )
    assert ci["label_threshold"] == 0.5
    assert ci["decision_threshold"] == 0.65
    assert ci["intervals"]["balanced_accuracy"] == {"low": 0.5, "high": 0.5, "n_valid": 20}


def test_undefined_metrics_are_json_null_and_boundary_is_positive():
    metrics = vigilance_metrics([0.35, 0.35], [0.35, 0.35])
    assert metrics["confusion_matrix"] == [[0, 0], [0, 2]]
    assert metrics["pearson"] is None
    assert metrics["spearman"] is None
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert metrics["macro_f1"] == 0.5
    assert metrics["balanced_accuracy"] == 1.0
    assert vigilance_metrics([0.1], [0.1])["recall"] == 0.0
    json.dumps(metrics, allow_nan=False)


def test_group_bootstrap_is_reproducible_and_resamples_whole_subjects():
    truth = [0.8, 0.8, 0.8, 0.8]
    prediction = [0.8, 0.8, 0.2, 0.2]
    groups = ["accurate", "accurate", "inaccurate", "inaccurate"]
    first = bootstrap_group_ci(truth, prediction, groups, n_bootstrap=100, seed=7)
    second = bootstrap_group_ci(truth, prediction, groups, n_bootstrap=100, seed=7)
    assert first == second
    assert first["n_groups"] == 2
    assert first["intervals"]["mae"]["low"] == 0.0
    assert first["intervals"]["mae"]["high"] == pytest.approx(0.6)
    assert first["intervals"]["mae"]["n_valid"] == 100
    assert first["intervals"]["pearson"] == {"low": None, "high": None, "n_valid": 0}
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("truth,prediction", [
    ([], []), ([0.2], [0.1, 0.3]), ([[0.2]], [[0.2]]), ([np.nan], [0.1]),
    ([0.2], [np.inf]), ([-0.1], [0.2]), ([0.2], [1.1]),
])
def test_invalid_metric_inputs_are_rejected(truth, prediction):
    with pytest.raises(ValueError):
        vigilance_metrics(truth, prediction)


def test_validation_and_bootstrap_require_valid_evaluation_groups():
    with pytest.raises(ValueError, match="both reference classes"):
        select_threshold([0.1, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_group_ci([0.1, 0.8], [0.2, 0.7], ["same", "same"])
    with pytest.raises(ValueError, match="one identifier"):
        grouped_metrics([0.1, 0.8], [0.2, 0.7], ["one"])
    with pytest.raises(ValueError, match="nonempty strings"):
        grouped_metrics([0.1, 0.8], [0.2, 0.7], ["one", ""])
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_group_ci([0.1, 0.8], [0.2, 0.7], ["one", "two"], n_bootstrap=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_group_ci([0.1, 0.8], [0.2, 0.7], ["one", "two"], confidence=1.0)
    with pytest.raises(ValueError, match="decision_threshold"):
        vigilance_metrics([0.1], [0.2], decision_threshold=np.nan)
