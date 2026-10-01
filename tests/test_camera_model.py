import csv
import json

import numpy as np
import pytest
import torch

from driverguard.learning.camera_features import FEATURE_VERSION, sha256_file
from driverguard.learning.camera_model import (
    CONTEXT,
    FEATURE_NAMES,
    PHYSICAL_SCALES,
    CameraGRU,
    CameraPredictor,
    window_quality,
)
from driverguard.learning.camera_train import (
    CameraClip,
    camera_windows,
    load_camera_manifest,
    select_camera_threshold,
    train_camera,
)


def _contract():
    return {"schema_version": 1, "feature_names": list(FEATURE_NAMES),
                "physical_scales": list(PHYSICAL_SCALES), "fps": 10.0,
                "quality": {"enabled": True}, "low_light": {"enabled": True},
                "face_model_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
                "extractor_version": FEATURE_VERSION}


def _features(n=CONTEXT, valid=None):
    if valid is None:
        valid = np.ones(n, dtype=bool)
    x = np.zeros((n, 19), dtype=np.float32)
    x[:, :9] = 0.3
    x[:, 9:18] = 1
    x[:, -1] = valid
    x[~valid] = 0
    return x, np.arange(n, dtype=np.int64) * 100, valid


def _checkpoint(path, **changes):
    torch.manual_seed(19)
    model = CameraGRU()
    values = {"format_version": 1, "architecture": "camera_gru", "context": 50, "feature_dim": 19,
                  "hidden_size": 32, "dropout": 0.1, "decision_threshold": 0.5,
                  "input_contract": _contract(), "state_dict": model.state_dict()}
    values.update(changes)
    torch.save(values, path)
    return model.eval()


def _manifest(tmp_path, test_shift=0.0):
    records = []
    for split, subjects in (("train", ("s1", "s2")), ("validation", ("s3",)), ("test", ("s4",))):
        for subject in subjects:
            for label in (0, 1):
                clip_id = f"{subject}_{label}"
                x, timestamps, valid = _features(60)
                x[:, :9] = 0.15 if label else 0.8
                if split == "test":
                    x[:, :9] += test_shift
                path = tmp_path / f"{clip_id}.npz"
                np.savez(path, x=x, ts_ms=timestamps, valid=valid, label=label,
                         subject=subject, clip_id=clip_id)
                records.append({"features": path.name, "subject": subject, "clip_id": clip_id, "split": split,
                                    "illumination": "night" if label else "day"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": records, "input_contract": _contract()}), encoding="utf-8")
    return manifest


def test_camera_export_reproduces_eval_score(tmp_path):
    path = tmp_path / "model.pt"
    model = _checkpoint(path)
    x, timestamps, valid = _features()
    with torch.inference_mode():
        expected = float(torch.sigmoid(model(torch.from_numpy(x)[None]))[0])
    result = CameraPredictor(path).predict_window(x, timestamps, valid)
    assert result.score == pytest.approx(expected, abs=1e-7)
    assert result.reason is None
    assert result.valid_fraction == 1


@pytest.mark.parametrize("cause,reason", [
    ("history", "insufficient_history"), ("current", "current_frame_invalid"),
    ("quality", "insufficient_quality"), ("gap", "timestamp_gap"),
    ("timestamp", "invalid_timestamps"), ("nonfinite", "nonfinite_features"),
])
def test_camera_abstains_instead_of_predicting_normal(tmp_path, cause, reason):
    path = tmp_path / "model.pt"
    _checkpoint(path)
    x, timestamps, valid = _features()
    if cause == "history":
        x, timestamps, valid = x[:-1], timestamps[:-1], valid[:-1]
    elif cause == "current":
        valid[-1] = False
        x[-1] = 0
    elif cause == "quality":
        valid[:21] = False
        x[:21] = 0
    elif cause == "gap":
        timestamps[25:] += 301
    elif cause == "timestamp":
        timestamps[10] = timestamps[9]
    elif cause == "nonfinite":
        x[0, 0] = np.nan
    result = CameraPredictor(path).predict_window(x, timestamps, valid)
    assert result.score is None
    assert result.reason == reason


def test_exact_quality_boundary_and_unavailable_feature_mask():
    valid = np.ones(50, dtype=bool)
    valid[:20] = False
    x, timestamps, valid = _features(valid=valid)
    assert window_quality(x, timestamps, valid).reason is None
    x[22, 9] = 0
    with pytest.raises(ValueError, match="Unavailable physical"):
        window_quality(x, timestamps, valid)
    x[22, 0] = 0
    assert window_quality(x, timestamps, valid).reason is None


def test_camera_contract_and_shape_rejected(tmp_path):
    path = tmp_path / "model.pt"
    _checkpoint(path, input_contract={})
    with pytest.raises(ValueError, match="contract"):
        CameraPredictor(path)
    _checkpoint(path)
    predictor = CameraPredictor(path)
    x, timestamps, valid = _features()
    with pytest.raises(ValueError, match="shape"):
        predictor.predict_window(x[:, :-1], timestamps, valid)
    with pytest.raises(ValueError, match="face_valid"):
        predictor.predict_window(x, timestamps, ~valid)


@pytest.mark.parametrize("cadence", [None, 10.0, 30.0, "10", float("nan")])
def test_camera_checkpoint_landmarker_cadence_matches_runtime(tmp_path, cadence):
    path = tmp_path / "model.pt"
    contract = _contract()
    if cadence is not None:
        contract["landmarker_cadence_hz"] = cadence
    _checkpoint(path, input_contract=contract)
    if cadence is None or cadence == 10.0:
        assert CameraPredictor(path).context == CONTEXT
    else:
        with pytest.raises(ValueError, match="landmarker cadence"):
            CameraPredictor(path)


def test_invalid_raw_face_cannot_contribute_physical_history():
    x, timestamps, valid = _features()
    valid[5] = False
    x[5, -1] = 0
    with pytest.raises(ValueError, match="Invalid raw-face"):
        window_quality(x, timestamps, valid)
    x[5] = 0
    assert window_quality(x, timestamps, valid).reason is None


@pytest.mark.parametrize("malformed", ["hash", "label", "signature", "version", "sampling"])
def test_camera_manifest_rejects_malformed_cache(tmp_path, malformed):
    manifest = _manifest(tmp_path)
    values = json.loads(manifest.read_text())
    record = values["clips"][0]
    feature_path = tmp_path / record["features"]
    expected_message = {
        "hash": "SHA-256", "label": "label disagrees", "signature": "extraction signature",
        "version": "extractor_version",
        "sampling": "sampling policy",
    }
    if malformed == "hash":
        record["features_sha256"] = "0" * 64
    elif malformed == "label":
        record["label"] = 1
    elif malformed == "signature":
        with np.load(feature_path) as npz:
            saved = {key: npz[key].copy() for key in npz.files}
        np.savez(feature_path, **saved, extraction_signature="different preprocessing")
    elif malformed == "version":
        values["input_contract"]["extractor_version"] = "old_version"
    else:
        values["input_contract"]["sampling"] = "different sampling"
    manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError, match=expected_message[malformed]):
        load_camera_manifest(manifest)


def test_camera_manifest_accepts_matching_signature_hash_and_label(tmp_path):
    manifest = _manifest(tmp_path)
    values = json.loads(manifest.read_text())
    record = values["clips"][0]
    feature_path = tmp_path / record["features"]
    with np.load(feature_path) as npz:
        saved = {key: npz[key].copy() for key in npz.files}
    signature = json.dumps(values["input_contract"], sort_keys=True, separators=(",", ":"))
    np.savez(feature_path, **saved, extraction_signature=signature)
    record.update(features_sha256=sha256_file(feature_path), label=0)
    manifest.write_text(json.dumps(values))
    clips, _ = load_camera_manifest(manifest)
    assert len(clips) == 8


def test_online_threshold_uses_window_weights_and_fixed_grid():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.6, 0.98])
    assert select_camera_threshold(labels, scores, np.ones(4)) == 0.5
    # A high-confidence negative window outweighs the lone low-scoring positive.
    assert select_camera_threshold(labels, scores, np.array([1, 10, 1, 10])) == 0.95
    with pytest.raises(ValueError, match="positive weights"):
        select_camera_threshold(labels, scores, np.array([1, 0, 1, 1]))


def test_subjects_cannot_cross_camera_splits(tmp_path):
    manifest = _manifest(tmp_path)
    values = json.loads(manifest.read_text())
    records = values["clips"]
    records[0]["split"] = "validation"
    manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="disjoint"):
        load_camera_manifest(manifest)


def test_development_loader_does_not_open_test_feature_files(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "s4_0.npz").unlink()
    (tmp_path / "s4_1.npz").unlink()
    clips, _ = load_camera_manifest(manifest, splits=("train", "validation"))
    assert len(clips) == 6
    assert {c.split for c in clips} == {"train", "validation"}
    values = json.loads(manifest.read_text())
    values["clips"][-1]["subject"] = "s1"
    manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="disjoint"):
        load_camera_manifest(manifest, splits=("train", "validation"))


def test_camera_windows_exclude_future_frames(tmp_path):
    path = tmp_path / "model.pt"
    _checkpoint(path)
    predictor = CameraPredictor(path)
    x, timestamps, valid = _features(80)
    clip = CameraClip(x, timestamps, valid, 0, "s1", "c1", "train")
    windows = camera_windows([clip], stride=10)
    assert [window.end for window in windows] == [49, 59, 69, 79]
    before = predictor.predict_window(x[:50], timestamps[:50], valid[:50]).score
    clip.x[50:, :9] = 999
    after = predictor.predict_window(clip.x[:50], timestamps[:50], valid[:50]).score
    assert before == after
    assert all(window.end + 1 - CONTEXT >= 0 for window in windows)


def test_training_exports_scores_and_keeps_test_out_of_selection(tmp_path):
    torch.set_num_threads(1)
    manifest = _manifest(tmp_path)
    first_dir = tmp_path / "first"
    first = train_camera(manifest, first_dir, epochs=2, patience=1, batch_size=8)
    with (first_dir / "predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    clips, _ = load_camera_manifest(manifest)
    by_id = {clip.clip_id: clip for clip in clips}
    predictor = CameraPredictor(first_dir / "model.pt")
    for row in predictions:
        clip = by_id[row["clip_id"]]
        end = int(row["end_ms"]) // 100
        score = predictor.predict_window(clip.x[end - 49:end + 1],
                                         clip.ts_ms[end - 49:end + 1], clip.valid[end - 49:end + 1]).score
        assert score == pytest.approx(float(row["score"]), abs=1e-7)
    _manifest(tmp_path, test_shift=3)
    second = train_camera(manifest, tmp_path / "second", epochs=2, patience=1, batch_size=8)
    assert first["best_epoch"] == second["best_epoch"]
    assert first["decision_threshold"] == second["decision_threshold"]
    assert first["history"] == second["history"]
    assert first["metrics"]["validation"] == second["metrics"]["validation"]
    first_weights = torch.load(first_dir / "model.pt", weights_only=True)["state_dict"]
    second_weights = torch.load(tmp_path / "second" / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(first_weights[key], second_weights[key]) for key in first_weights)
    assert first["metrics"]["test"]["clip"]["coverage"] == 1
    assert first["early_onset_prediction"] is False


def test_rejected_test_clip_preserved_in_csv_and_coverage(tmp_path):
    torch.set_num_threads(1)
    manifest = _manifest(tmp_path)
    test_clip = tmp_path / "s4_1.npz"
    with np.load(test_clip) as values:
        saved = {key: values[key].copy() for key in values.files}
    saved["valid"][:] = False
    saved["x"][:] = 0
    np.savez(test_clip, **saved)
    output = tmp_path / "trained"
    report = train_camera(manifest, output, epochs=1, batch_size=8)
    with (output / "clip_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rejected = next(row for row in rows if row["clip_id"] == "s4_1")
    assert rejected["score"] == ""
    assert rejected["reason"] == "no_accepted_windows"
    assert report["metrics"]["test"]["clip"]["coverage"] == 0.5
    assert report["metrics"]["test"]["window"]["coverage_by_label"]["1"] == 0
