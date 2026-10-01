import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from driverguard.learning.cross_validation import apply_calibration, fit_calibration
from driverguard.learning.data import Session
from driverguard.learning.fitting import fit_grouped_model
from driverguard.learning.model import VigilancePredictor


def _sessions(tmp_path, *, robust=False):
    rng = np.random.default_rng(13)
    result = []
    for subject in range(6):
        for label in (0, 1):
            x = rng.normal(scale=0.12, size=(20, 10)).astype(np.float32)
            x[:, 1] += label * 0.5
            x[:, 6] += label * 0.4
            x += subject * 0.1
            valid = np.ones(20, dtype=bool)
            valid[-1] = False
            reason = np.where(valid, "", "sensor_failure")
            path = tmp_path / f"s{subject}_{label}.npz"
            result.append(Session(f"s{subject}_{label}", f"s{subject}", x,
                                  np.full(20, label, dtype=np.float32), path, path,
                                  valid, reason, "robust_v1" if robust else "legacy_car",
                                  ("F3", "F4") if robust else None))
    return result


@pytest.mark.parametrize("config", [
    {"classifier": "lda", "shrinkage": "auto", "calibration_folds": 3},
    {"classifier": "logistic", "C": 0.1, "calibration_folds": 3},
])
def test_grouped_checkpoint_roundtrip_preserves_quality_and_predictions(tmp_path, config):
    sessions = _sessions(tmp_path, robust=True)
    fitted = fit_grouped_model(sessions, config, 42)
    path = fitted.save_checkpoint(tmp_path / "model.pt", {"target": "fatigue"})
    predictor = VigilancePredictor(path)
    assert len(predictor.members) == 3
    for member in predictor.members:
        assert not set(member["fit_subjects"]).intersection(member["calibration_subjects"])
        assert set(member["fit_subjects"] + member["calibration_subjects"]) == {f"s{i}" for i in range(6)}
        assert member["calibration"]["slope"] >= 0
        assert member["calibration"]["method"] == "sigmoid_decision"
    calibration_subjects = [subject for member in predictor.members for subject in member["calibration_subjects"]]
    assert sorted(calibration_subjects) == [f"s{i}" for i in range(6)]
    session = sessions[0]
    expected = fitted.predict_session(session)
    actual = predictor.predict_features(session.x, quality_valid=session.quality_valid)
    np.testing.assert_allclose(actual, expected.score, rtol=0, atol=0, equal_nan=True)
    assert np.isnan(actual[-1])
    assert expected.reason[-1] == "sensor_failure"
    assert actual[:-1].min() >= 0 and actual[:-1].max() <= 1
    with pytest.raises(ValueError, match="explicit quality"):
        predictor.predict(session.x)
    with pytest.raises(ValueError, match="abstentions"):
        predictor.decisions(actual)
    with pytest.raises(ValueError, match="exists"):
        fitted.save_checkpoint(path, {})


def test_calibration_subject_labels_do_not_change_member_classifier(tmp_path):
    sessions = _sessions(tmp_path)
    config = {"classifier": "logistic", "C": 1.0, "calibration_folds": 3}
    original = fit_grouped_model(sessions, config, 42)
    held_subject = original.members[0]["calibration_subjects"][0]
    changed = copy.deepcopy(sessions)
    for session in changed:
        if session.subject == held_subject:
            session.y = 1 - session.y
    refit = fit_grouped_model(changed, config, 42)
    for parameter in ("mean", "std", "coef", "intercept"):
        np.testing.assert_array_equal(original.members[0][parameter], refit.members[0][parameter])
    assert original.members[0]["calibration"] != refit.members[0]["calibration"]


def test_invalid_sensor_rows_never_influence_fitting_or_accepted_predictions(tmp_path):
    sessions = _sessions(tmp_path)
    config = {"classifier": "lda", "calibration_folds": 3}
    original = fit_grouped_model(sessions, config, 42)
    corrupted = copy.deepcopy(sessions)
    for session in corrupted:
        session.x[~session.quality_valid] = np.nan
        session.y[~session.quality_valid] = 1 - session.y[~session.quality_valid]
    refit = fit_grouped_model(corrupted, config, 42)
    assert original.members == refit.members
    np.testing.assert_allclose(original.predict_session(sessions[0]).score,
                               refit.predict_session(corrupted[0]).score, equal_nan=True)


def test_robust_predict_file_enforces_preprocessing_and_channel_contract(tmp_path):
    sessions = _sessions(tmp_path, robust=True)
    fitted = fit_grouped_model(sessions, {"classifier": "logistic"}, 42)
    predictor = VigilancePredictor(fitted.save_checkpoint(tmp_path / "model.pt", {}))
    session = sessions[0]
    np.savez(session.feature_path, x=session.x, preprocessing="robust_v1", channel_names=["F3", "F4"],
             quality_valid=session.quality_valid, quality_reason_code=np.where(session.quality_valid, 0, 1),
             window_end_seconds=np.arange(1, 21) * 4.0)
    np.testing.assert_allclose(predictor.predict_file(session.feature_path),
                               fitted.predict_session(session).score, equal_nan=True)
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, session.x)
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        predictor.predict_file(legacy)
    np.savez(session.feature_path, x=session.x, preprocessing="robust_v1", channel_names=["F4", "F3"],
             quality_valid=session.quality_valid, quality_reason_code=np.where(session.quality_valid, 0, 1),
             window_end_seconds=np.arange(1, 21) * 4.0)
    with pytest.raises(ValueError, match="channel order"):
        predictor.predict_file(session.feature_path)


@pytest.mark.parametrize("mutation", ["negative_slope", "overlap", "bad_std", "metadata_conflict", "bad_channels"])
def test_malformed_portable_ensemble_is_rejected(tmp_path, mutation):
    fitted = fit_grouped_model(_sessions(tmp_path, robust=True), {"classifier": "lda"}, 42)
    checkpoint = fitted.to_checkpoint({})
    if mutation == "negative_slope":
        checkpoint["members"][0]["calibration"]["slope"] = -1.0
    elif mutation == "overlap":
        checkpoint["members"][0]["fit_subjects"] += checkpoint["members"][0]["calibration_subjects"][:1]
    elif mutation == "bad_std":
        checkpoint["members"][0]["std"][0] = 0.0
    elif mutation == "metadata_conflict":
        checkpoint["dataset_metadata"]["preprocessing"] = "legacy_car"
    else:
        checkpoint["channel_names"] = ["F4", "F3"]
    path = tmp_path / "malformed.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        VigilancePredictor(path)


def test_fit_rejects_incompatible_session_contracts(tmp_path):
    sessions = _sessions(tmp_path, robust=True)
    sessions[0].channel_names = ("F4", "F3")
    with pytest.raises(ValueError, match="share the feature"):
        fit_grouped_model(sessions, {"classifier": "lda"})
    sessions = _sessions(tmp_path, robust=True)
    sessions[0].quality_valid = None
    with pytest.raises(ValueError, match="explicit quality"):
        fit_grouped_model(sessions, {"classifier": "lda"})


def test_raw_calibration_preserves_order_of_saturated_classifier_decisions():
    from scipy.special import expit

    decision = np.array([-80.0, -40.0, -20.0, 20.0, 40.0, 80.0])
    target = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    assert expit(40.0) == expit(80.0) == 1.0
    calibration = fit_calibration(target, decision, groups, input_type="decision")
    assert calibration["method"] == "sigmoid_decision"
    assert calibration["slope"] > 0
    probability = apply_calibration(decision, calibration)
    assert np.all(np.diff(probability) > 0)
    assert np.isfinite(probability).all()
    with pytest.raises(ValueError, match="finite predictions"):
        fit_calibration(target, decision, groups)


def test_raw_decision_checkpoint_roundtrip_retains_confident_score_ranks(tmp_path):
    from scipy.special import expit

    from driverguard.learning.features import transform_features

    member = {"mean": [0.0] * 5, "std": [1.0] * 5, "coef": [1.0, 0.0, 0.0, 0.0, 0.0],
              "intercept": 80.0,
              "calibration": {"method": "sigmoid_decision", "slope": 0.02, "intercept": 0.0},
              "fit_subjects": ["a"], "calibration_subjects": ["b"]}
    checkpoint = {"format_version": 1, "architecture": "calibrated_linear_ensemble", "context": 1,
                  "feature_key": "de_movingAve", "feature_transform": "relative_log_power",
                  "mean": [0.0] * 5, "std": [1.0] * 5, "members": [member], "decision_threshold": 0.5}
    path = tmp_path / "raw.pt"
    torch.save(checkpoint, path)
    predictor = VigilancePredictor(path)
    x = np.zeros((4, 5), dtype=np.float32)
    x[:, 0] = [-160.0, -120.0, -40.0, 0.0]
    decision = transform_features(x, "relative_log_power")[:, 0].astype(float) + 80.0
    expected = expit(0.02 * decision)
    predicted = predictor.predict(x)
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=0)
    assert np.all(np.diff(predicted) > 0)
    assert expit(decision[-2]) == expit(decision[-1]) == 1.0
    checkpoint["members"][0]["calibration"]["method"] = "sigmoid_logit"
    legacy_path = tmp_path / "old_calibration.pt"
    torch.save(checkpoint, legacy_path)
    legacy_score = VigilancePredictor(legacy_path).predict(x)
    assert legacy_score[-2] == legacy_score[-1]
