import csv
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from driverguard.learning.data import load_sessions
from driverguard.learning.improvement import improve
from driverguard.learning.model import VigilancePredictor
from driverguard.learning.quality import ROBUST_PREPROCESSING


def _paired_datasets(tmp_path):
    legacy, robust = tmp_path / "legacy", tmp_path / "robust"
    legacy.mkdir()
    robust.mkdir()
    rng = np.random.default_rng(117)
    manifest = []
    metadata = {"target": "fatigue", "target_type": "binary", "label_threshold": 0.5,
                "channel_names": ["C3", "C4"], "n_features": 10, "window_seconds": 4.0,
                "feature_extraction": "synthetic channel-major five-band test features"}
    for index in range(4):
        subject = f"subject_{index}"
        labels = np.repeat([0.0, 1.0], 12).astype(np.float32)
        features = rng.normal(0, 0.2, (24, 10)).astype(np.float32)
        features[:, [1, 6]] += labels[:, None] * 0.6
        features[:, [2, 7]] -= labels[:, None] * 0.3
        features += rng.normal(0, 0.1, (1, 10)).astype(np.float32)
        ends = np.arange(1, 25, dtype=float) * 4
        accepted = np.ones(24, dtype=bool)
        if index == 0:
            accepted[1] = False
        common = {"y": labels, "window_end_seconds": ends, "channel_names": metadata["channel_names"]}
        np.savez_compressed(legacy / f"{subject}.npz", x=features, preprocessing=np.asarray("legacy_car"),
                            **common)
        corrected = features.copy()
        corrected[~accepted] = 0
        np.savez_compressed(robust / f"{subject}.npz", x=corrected, preprocessing=np.asarray("robust_v1"),
                            quality_valid=accepted, quality_reason_code=(~accepted).astype(np.uint8), **common)
        manifest.append({"session_id": subject, "subject": subject, "path": f"{subject}.npz"})
    for root, preprocessing in ((legacy, "legacy_car"), (robust, "robust_v1")):
        contract = {**metadata, "preprocessing": preprocessing}
        if preprocessing == "robust_v1":
            contract["preprocessing_config"] = ROBUST_PREPROCESSING
        (root / "dataset_metadata.json").write_text(json.dumps(contract), encoding="utf-8")
        (root / "sessions.json").write_text(json.dumps(manifest), encoding="utf-8")
    return legacy, robust


def test_improvement_exports_audited_model_and_preserves_all_held_out_windows(tmp_path):
    legacy, robust = _paired_datasets(tmp_path)
    output = tmp_path / "experiment"
    summary = improve(legacy, robust, output, bootstrap=5, threads=1)
    predictor = VigilancePredictor(output / "model.pt")
    assert predictor.preprocessing == "robust_v1"
    assert predictor.meta["decision_threshold"] == 0.5
    assert predictor.meta["training_subjects"] == [f"subject_{index}" for index in range(4)]
    assert "split" not in predictor.meta
    for session in load_sessions(robust):
        scores = predictor.predict_file(session.feature_path)
        assert scores.shape == (24,)
        assert np.isfinite(scores[session.quality_valid]).all()
        assert np.isnan(scores[~session.quality_valid]).all()
    with (output / "robust_nested_loso_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    rejected = [row for row in rows if row["accepted"] == "False"]
    assert len(rejected) == 1
    assert rejected[0]["window_index"] == "1"
    assert rejected[0]["subject"] == "subject_0"
    assert rejected[0]["prediction"] == rejected[0]["raw_score"] == rejected[0]["reduced_vigilance_pred"] == ""
    assert rejected[0]["abstention_reason"] == "too_many_bad_channels"
    report = json.loads((output / "robust_nested_loso.json").read_text(encoding="utf-8"))
    assert report["n_outer_folds"] == 4
    for fold in report["folds"]:
        assert fold["test_subject"] not in fold["training_subjects"]
        for inner in fold["selection"]["inner_folds"]:
            assert fold["test_subject"] not in inner["training_subjects"]
            assert inner["validation_subject"] not in inner["training_subjects"]
    assert summary["robust"]["coverage"]["n_total"] == 96
    assert summary["robust"]["coverage"]["n_accepted"] == 95
    assert summary["paired_comparison"]["n_subjects"] == 4
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["decision_threshold"] == 0.5
    assert protocol["source_file_hashes"] == predictor.meta["source_file_hashes"]
    assert "improvement.py" in protocol["source_file_hashes"]
    assert all(len(value) == 64 for value in protocol["source_file_hashes"].values())
    assert "completed before outer report inspection" in protocol["deployment_selection"]
    selection = json.loads((output / "deployment_selection.json").read_text(encoding="utf-8"))
    assert selection["evaluation_status"] == "all_cohort_fit_without_independent_test_score"
    assert len(predictor.meta["dataset_manifest"]) == 4
    original_hash = hashlib.sha256((output / "model.pt").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already contains an experiment"):
        improve(legacy, robust, output, bootstrap=5, threads=1)
    assert hashlib.sha256((output / "model.pt").read_bytes()).hexdigest() == original_hash
