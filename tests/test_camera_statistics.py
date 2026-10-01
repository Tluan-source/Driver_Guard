import json

import numpy as np
import pytest
import torch

from driverguard.learning.camera_baseline import run_baseline
from driverguard.learning.camera_features import FEATURE_NAMES, FEATURE_VERSION, PHYSICAL_SCALES
from driverguard.learning.camera_model import CameraPredictor
from driverguard.learning.camera_statistics import STATISTIC_NAMES, temporal_statistics


def _arrays(n=60):
    x = np.zeros((n, 19), dtype=np.float32)
    x[:, :9] = 1
    x[:, 9:] = 1
    return x, np.arange(n, dtype=np.int64) * 100, np.ones(n, bool)


def _manifest(tmp_path):
    contract = {"schema_version": 1, "feature_names": list(FEATURE_NAMES),
                "physical_scales": list(PHYSICAL_SCALES), "fps": 10.0,
                "quality": {"enabled": True}, "low_light": {"enabled": True},
                "face_model_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
                "extractor_version": FEATURE_VERSION}
    records = []
    for split, subjects in (("train", ("s1", "s2")), ("validation", ("s3",)), ("test", ("s4",))):
        for subject in subjects:
            for label in (0, 1):
                clip_id = f"{subject}_{label}"
                path = tmp_path / f"{clip_id}.npz"
                if split != "test":
                    x, ts, valid = _arrays()
                    x[:, :9] = 0.15 if label else 0.8
                    np.savez(path, x=x, ts_ms=ts, valid=valid, label=label, subject=subject, clip_id=clip_id)
                records.append({"features": path.name, "subject": subject, "clip_id": clip_id,
                                "split": split, "label": label})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"input_contract": contract, "clips": records}))
    return path


def test_missing_feature_values_are_not_averaged_as_zero():
    x, ts, valid = _arrays(50)
    x[:25, 0] = 0
    x[:25, 9] = 0
    result = dict(zip(STATISTIC_NAMES, temporal_statistics(x, ts, valid), strict=True))
    assert result["ear_left_mean"] == 1
    assert result["ear_left_availability"] == 0.5
    assert result["eyes_available_fraction"] == 0.5
    assert np.isfinite(list(result.values())).all()


def test_invalid_frames_break_closure_duration_and_transitions():
    x, ts, valid = _arrays(50)
    x[10:20, :2] = 0.1 / 0.3
    x[15] = 0
    valid[15] = False
    result = dict(zip(STATISTIC_NAMES, temporal_statistics(x, ts, valid), strict=True))
    assert result["max_closed_eye_seconds_proxy"] == pytest.approx(0.5)
    assert result["eye_closing_transitions_per_second_proxy"] == pytest.approx(0.2)
    assert result["closed_eye_fraction_proxy"] == pytest.approx(9 / 49)


def test_temporal_summary_is_past_only_and_trend_uses_time():
    x, ts, valid = _arrays(80)
    x[:50, 0] = ts[:50] / 1000 * 0.1
    before = temporal_statistics(x[:50], ts[:50], valid[:50])
    x[50:, :9] = 999
    after = temporal_statistics(x[:50], ts[:50], valid[:50])
    assert np.array_equal(before, after)
    assert before[STATISTIC_NAMES.index("ear_left_trend_per_second")] == pytest.approx(0.1)
    ts[2] = ts[1]
    with pytest.raises(ValueError, match="increasing timestamps"):
        temporal_statistics(x[:50], ts[:50], valid[:50])


def test_cam01_exports_portable_scores_without_opening_test_files(tmp_path):
    manifest = _manifest(tmp_path)
    report = run_baseline(manifest, tmp_path / "cam01")
    assert report["test_evaluated"] is False
    assert set(report["metrics"]) == {"train", "validation"}
    assert report["export_max_absolute_score_error"] < 1e-10
    assert set(report["subject_splits"]["train"]) == {"s1", "s2"}
    assert not (tmp_path / "s4_0.npz").exists()
    predictor = CameraPredictor(tmp_path / "cam01/model.pt")
    x, ts, valid = _arrays(50)
    assert predictor.predict_window(x, ts, valid).score is not None
    valid[-1] = False
    x[-1] = 0
    result = predictor.predict_window(x, ts, valid)
    assert result.score is None and result.reason == "current_frame_invalid"
    meta = torch.load(tmp_path / "cam01/model.pt", weights_only=True)
    meta["scale"][0] = 0
    torch.save(meta, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="temporal scale"):
        CameraPredictor(tmp_path / "invalid.pt")


def test_cam01_refuses_to_replace_an_existing_experiment(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "model.pt").write_bytes(b"keep existing")
    with pytest.raises(FileExistsError):
        run_baseline(tmp_path / "absent.json", output)
    assert (output / "model.pt").read_bytes() == b"keep existing"
