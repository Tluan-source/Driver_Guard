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
