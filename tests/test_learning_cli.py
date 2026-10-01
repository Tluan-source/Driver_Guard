import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from driverguard.learning.cli import main
from driverguard.learning.data import dataset_manifest, load_sessions


def test_feature_prediction_has_no_label_input(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, x=np.ones((3, 2), dtype=np.float32), y=np.full(3, np.nan),
             window_end_seconds=[4, 8, 12])
    checkpoint = tmp_path / "model.pt"
    torch.save({"format_version": 1, "architecture": "constant", "context": 1,
                "feature_key": "de_movingAve", "mean": [0., 0.], "std": [1., 1.],
                "value": .6, "decision_threshold": .5, "target": "fatigue",
                "dataset_metadata": {"window_seconds": 4}}, checkpoint)
    output = tmp_path / "predictions.csv"
    assert main(["predict", "--checkpoint", str(checkpoint), "--features", str(path), "--out", str(output)]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [float(row["window_end_seconds"]) for row in rows] == [4, 8, 12]
    assert all(int(row["reduced_vigilance_pred"]) == 1 for row in rows)
    assert "target" not in rows[0]


def test_frozen_evaluation_rejects_modified_test_features(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    records = []
    for subject in ("heldout_a", "heldout_b"):
        file = root / f"{subject}.npz"
        np.savez(file, x=np.ones((2, 2)), y=[0., 1.])
        records.append({"session_id": subject, "subject": subject, "path": file.name})
    (root / "sessions.json").write_text(json.dumps(records), encoding="utf-8")
    fingerprint = dataset_manifest(load_sessions(root))
    checkpoint = tmp_path / "model.pt"
    torch.save({"format_version": 1, "architecture": "constant", "context": 1,
                "feature_key": "de_movingAve", "mean": [0., 0.], "std": [1., 1.],
                "value": .5, "decision_threshold": .5, "label_threshold": .5, "seed": 42,
                "split": {"test": ["heldout_a", "heldout_b"]}, "dataset_manifest": fingerprint}, checkpoint)
    np.savez(root / "heldout_a.npz", x=np.zeros((2, 2)), y=[0., 1.])
    with pytest.raises(ValueError, match="content changed"):
        main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(root), "--out", str(tmp_path / "report")])
    assert not (tmp_path / "report").exists()


def test_quality_rejected_prediction_is_blank_and_never_normal(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, x=np.ones((3, 10), dtype=np.float32), y=np.full(3, np.nan),
             preprocessing=np.asarray("robust_v1"), channel_names=["C3", "C4"],
             quality_valid=np.asarray([True, False, True]), quality_reason_code=[0, 1, 0],
             window_end_seconds=[4, 8, 12])
    checkpoint = tmp_path / "model.pt"
    torch.save({"format_version": 1, "architecture": "constant", "context": 1,
                "feature_key": "de_movingAve", "mean": [0.] * 10, "std": [1.] * 10,
                "value": .2, "decision_threshold": .5, "target": "fatigue",
                "dataset_metadata": {"preprocessing": "robust_v1", "window_seconds": 4,
                                     "channel_names": ["C3", "C4"]}}, checkpoint)
    output = tmp_path / "predictions.csv"
    assert main(["predict", "--checkpoint", str(checkpoint), "--features", str(path),
                 "--out", str(output)]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["window_end_seconds"] for row in rows] == ["4.0", "8.0", "12.0"]
    assert rows[0]["reduced_vigilance_pred"] == "0"
    assert rows[1]["prediction"] == rows[1]["reduced_vigilance_pred"] == ""
    assert rows[1]["sensor_status"] == "SENSOR_DEGRADED"
    assert rows[1]["abstention_reason"] == "too_many_bad_channels"
    assert rows[1]["quality_valid"] == "0"


def test_deployment_fit_cannot_be_reported_as_frozen_test(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"format_version": 1, "architecture": "constant", "context": 1,
                "feature_key": "de_movingAve", "mean": [0.], "std": [1.],
                "value": .5, "decision_threshold": .5}, checkpoint)
    with pytest.raises(ValueError, match="no frozen test split"):
        main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(tmp_path / "unused"),
              "--out", str(tmp_path / "report")])
    assert not (tmp_path / "report").exists()


def test_cnt_channel_contract_accepts_checkpoint_list_and_abstains(tmp_path, monkeypatch):
    from driverguard.learning import eeg
    from driverguard.learning.quality import ROBUST_PREPROCESSING

    channels = list(eeg.CHANNEL_NAMES)
    monkeypatch.setattr(eeg, "extract_cnt_quality", lambda *args: (
        np.ones((2, 150), dtype=np.float32), np.asarray([4., 8.]), channels,
        {"quality_valid": np.asarray([True, False]), "quality_reason_code": [0, 1]}))
    checkpoint = tmp_path / "model.pt"
    torch.save({"format_version": 1, "architecture": "constant", "context": 1,
                "feature_key": "de_movingAve", "mean": [0.] * 150, "std": [1.] * 150,
                "value": .6, "decision_threshold": .5, "target": "fatigue",
                "channel_names": channels,
                "dataset_metadata": {"target": "fatigue", "n_features": 150, "window_seconds": 4,
                                     "preprocessing": "robust_v1", "channel_names": channels,
                                     "preprocessing_config": ROBUST_PREPROCESSING}}, checkpoint)
    output = tmp_path / "predictions.csv"
    assert main(["predict", "--checkpoint", str(checkpoint), "--cnt", "unused.cnt",
                 "--out", str(output)]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]["prediction"]) == pytest.approx(.6)
    assert rows[1]["prediction"] == rows[1]["reduced_vigilance_pred"] == ""
