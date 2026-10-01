import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
scipy_io = pytest.importorskip("scipy.io")

from driverguard.learning.data import (
    Session,
    causal_windows,
    dataset_metadata,
    load_sessions,
    pack_sessions,
    read_eeg_features,
    split_sessions,
)
from driverguard.learning.model import CausalBlock, VigilancePredictor, VigilanceTCN


@pytest.mark.parametrize("stem,subject", [("4_20151106", "4"), ("1_20151106_1", "1")])
def test_channel_band_order_and_actual_seed_filenames(tmp_path, stem, subject):
    (tmp_path / "DE").mkdir()
    (tmp_path / "perclos_labels").mkdir()
    raw = np.arange(17 * 4 * 5).reshape(17, 4, 5)
    scipy_io.savemat(tmp_path / "DE" / f"{stem}.mat", {"de_movingAve": raw})
    scipy_io.savemat(tmp_path / "perclos_labels" / f"{stem}.mat", {"perclos": np.linspace(0, 1, 4)})
    sessions = load_sessions(tmp_path)
    assert sessions[0].subject == subject
    assert sessions[0].session_id == stem
    np.testing.assert_equal(sessions[0].x[2].reshape(17, 5), raw[:, 2, :])
    scipy_io.savemat(tmp_path / "perclos_labels" / f"{stem}.mat", {"perclos": [0, 1]})
    with pytest.raises(ValueError, match="misaligned"):
        load_sessions(tmp_path)


def test_causal_windows_never_include_future_features():
    x = np.arange(15).reshape(5, 3).astype(np.float32)
    windows = causal_windows(x, 3)
    np.testing.assert_equal(windows[0], [x[0], x[0], x[0]])
    np.testing.assert_equal(windows[2], x[:3])
    changed = x.copy()
    changed[3:] = 999
    np.testing.assert_equal(causal_windows(changed, 3)[:3], windows[:3])


def test_generic_npz_preserves_session_subject_and_target_contract(tmp_path):
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    np.savez_compressed(tmp_path / "driver01_alert_fatigue.npz", x=x, y=y)
    metadata = {"target": "fatigue", "target_type": "binary", "label_threshold": 0.5}
    (tmp_path / "sessions.json").write_text(json.dumps([
        {"session_id": "driver01_day1", "subject": "driver01", "path": "driver01_alert_fatigue.npz"},
    ]), encoding="utf-8")
    (tmp_path / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    sessions = load_sessions(tmp_path)
    assert len(sessions) == 1
    assert sessions[0].subject == "driver01"
    assert sessions[0].session_id == "driver01_day1"
    np.testing.assert_array_equal(sessions[0].x, x)
    np.testing.assert_array_equal(sessions[0].y, y)
    assert dataset_metadata(tmp_path) == metadata
    np.testing.assert_array_equal(read_eeg_features(tmp_path / "driver01_alert_fatigue.npz", n_features=5), x)
    with pytest.raises(ValueError, match="Expected nonempty"):
        read_eeg_features(tmp_path / "driver01_alert_fatigue.npz", n_features=85)


def test_generic_sessions_reject_different_feature_schemas(tmp_path):
    np.savez(tmp_path / "first.npz", x=np.zeros((2, 3)), y=[0, 1])
    np.savez(tmp_path / "second.npz", x=np.zeros((2, 4)), y=[0, 1])
    (tmp_path / "sessions.json").write_text(json.dumps([
        {"session_id": "a", "subject": "a", "path": "first.npz"},
        {"session_id": "b", "subject": "b", "path": "second.npz"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="feature schema"):
        load_sessions(tmp_path)


def _sessions_for_subjects(tmp_path, count):
    sessions = []
    for subject in range(count):
        for day in (1, 2):
            sessions.append(Session(
                f"s{subject}_day{day}", f"s{subject}", np.zeros((2, 3), dtype=np.float32),
                np.array([0, 1], dtype=np.float32), tmp_path / "feature.npz", tmp_path / "feature.npz",
            ))
    return sessions


def test_subject_split_keeps_repeated_sessions_together_and_is_reproducible(tmp_path):
    sessions = _sessions_for_subjects(tmp_path, 12)
    split = split_sessions(sessions, seed=17)
    assert split == split_sessions(sessions, seed=17)
    assert len(split["test"]) >= 2
    assert split["validation"]
    assignments = {subject: stage for stage, subjects in split.items() for subject in subjects}
    assert len(assignments) == 12
    assert sum(len(subjects) for subjects in split.values()) == 12
    for subject in assignments:
        assert len([session for session in sessions if session.subject == subject]) == 2
        assert sum(subject in subjects for subjects in split.values()) == 1


@pytest.mark.parametrize("count", [7, 8, 10, 11])
def test_split_rejects_only_one_test_subject_before_training(tmp_path, count):
    with pytest.raises(ValueError, match="at least two test subjects"):
        split_sessions(_sessions_for_subjects(tmp_path, count))


def test_packed_context_restarts_at_each_session_even_for_same_subject(tmp_path):
    first = np.arange(9).reshape(3, 3).astype(np.float32) + 200
    second = np.arange(6).reshape(2, 3).astype(np.float32) - 30
    path = tmp_path / "features.npz"
    sessions = [
        Session("morning", "driver1", first, np.array([0, 0, 1]), path, path),
        Session("evening", "driver1", second, np.array([1, 0]), path, path),
        Session("excluded", "driver2", first, np.array([0, 0, 0]), path, path),
    ]
    windows, targets, subjects, recordings, indices = pack_sessions(sessions, ["driver1"], context=3)
    assert windows.shape == (5, 3, 3)
    np.testing.assert_array_equal(windows[2], first)
    np.testing.assert_array_equal(windows[3], np.repeat(second[:1], 3, axis=0))
    np.testing.assert_array_equal(windows[4], [second[0], second[0], second[1]])
    np.testing.assert_array_equal(targets, [0, 0, 1, 1, 0])
    np.testing.assert_array_equal(subjects, ["driver1"] * 5)
    np.testing.assert_array_equal(recordings, ["morning"] * 3 + ["evening"] * 2)
    np.testing.assert_array_equal(indices, [0, 1, 2, 0, 1])


def test_invalid_eeg_rejected(tmp_path):
    path = tmp_path / "invalid.npy"
    np.save(path, np.full((5, 85), np.nan))
    with pytest.raises(ValueError, match="Nonfinite"):
        read_eeg_features(path)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(4)
    model = VigilanceTCN().eval()
    x = np.random.default_rng(4).normal(size=(7, 3, 85)).astype(np.float32)
    path = tmp_path / "test.pt"
    torch.save({"format_version": 1, "architecture": "tcn", "context": 3, "feature_key": "de_movingAve",
                "mean": [0.] * 85, "std": [1.] * 85, "state_dict": model.state_dict(),
                "model_config": {}, "decision_threshold": .4}, path)
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).numpy()
    predictor = VigilancePredictor(path)
    np.testing.assert_allclose(predictor.predict_windows(x, batch_size=2), expected, rtol=1e-5)
    with pytest.raises(ValueError, match="Expected finite"):
        predictor.predict_windows(x[:, :2])


def test_tcn_uses_oldest_sample_of_maximum_context():
    model = VigilanceTCN(n_features=3, width=4, dropout=0).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.05)
        original = torch.zeros(1, 15, 3)
        changed = original.clone()
        changed[:, 0] = 2
        assert not torch.equal(model(original), model(changed))


def test_causal_block_output_before_changes_does_not_depend_on_future():
    torch.manual_seed(11)
    block = CausalBlock(width=3, dilation=4, dropout=0).eval()
    original = torch.randn(2, 3, 15)
    changed = original.clone()
    changed[:, :, 8:] += 100
    with torch.inference_mode():
        torch.testing.assert_close(block(original)[:, :, :8], block(changed)[:, :, :8], rtol=0, atol=0)


def _constant_checkpoint(**overrides):
    checkpoint = {
        "format_version": 1, "architecture": "constant", "context": 3,
        "feature_key": "generic", "mean": [0.0] * 3, "std": [1.0] * 3,
        "value": 0.4, "decision_threshold": 0.5,
    }
    checkpoint.update(overrides)
    return checkpoint


def test_generic_checkpoint_predicts_feature_count_from_normalization(tmp_path):
    path = tmp_path / "constant.pt"
    torch.save(_constant_checkpoint(), path)
    predictor = VigilancePredictor(path)
    x = np.ones((4, 3), dtype=np.float32)
    np.testing.assert_allclose(predictor.predict(x), np.full(4, 0.4), rtol=1e-6)
    with pytest.raises(ValueError, match="Expected finite"):
        predictor.predict(np.ones((4, 85), dtype=np.float32))


@pytest.mark.parametrize("overrides", [
    {"mean": [float("nan"), 0.0, 0.0]},
    {"mean": [[0.0, 0.0, 0.0]]},
    {"mean": [], "std": []},
    {"std": [0.0, 1.0, 1.0]},
    {"std": [-1.0, 1.0, 1.0]},
    {"std": [float("inf"), 1.0, 1.0]},
    {"std": [1.0]},
    {"context": 0},
    {"context": 16},
    {"architecture": "unknown"},
])
def test_invalid_checkpoint_normalization_and_contract_are_rejected(tmp_path, overrides):
    path = tmp_path / "invalid.pt"
    torch.save(_constant_checkpoint(**overrides), path)
    with pytest.raises(ValueError):
        VigilancePredictor(path)


def test_nonfinite_tcn_weights_are_rejected_before_inference(tmp_path):
    model = VigilanceTCN(n_features=3, width=4, dropout=0)
    state = model.state_dict()
    state["project.weight"][0, 0, 0] = float("nan")
    path = tmp_path / "invalid_weights.pt"
    torch.save(_constant_checkpoint(
        architecture="tcn", state_dict=state,
        model_config={"n_features": 3, "width": 4, "dropout": 0},
    ), path)
    with pytest.raises(ValueError, match="Nonfinite model weights"):
        VigilancePredictor(path)


@pytest.mark.parametrize("overrides", [
    {"decision_threshold": float("nan")},
    {"decision_threshold": -0.1},
    {"decision_threshold": 1.1},
    {"value": float("inf")},
    {"value": -0.1},
    {"architecture": "ridge", "coef": [0.1, 0.2], "intercept": 0.1},
    {"architecture": "ridge", "coef": [0.1, float("nan"), 0.2], "intercept": 0.1},
    {"architecture": "ridge", "coef": [0.1, 0.2, 0.3], "intercept": float("inf")},
])
def test_invalid_non_neural_checkpoint_values_are_rejected(tmp_path, overrides):
    path = tmp_path / "invalid_values.pt"
    torch.save(_constant_checkpoint(**overrides), path)
    with pytest.raises(ValueError):
        VigilancePredictor(path)
