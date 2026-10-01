import json

import numpy as np
import pytest

from driverguard.learning.data import load_sessions, pack_sessions, read_feature_contract


def _robust_archive(path, **changes):
    values = {
        "x": np.arange(30, dtype=np.float32).reshape(3, 10),
        "y": np.asarray([0, 1, 0], dtype=np.float32),
        "channel_names": np.asarray(["C3", "C4"]),
        "preprocessing": np.asarray("robust_v1"),
        "quality_valid": np.asarray([True, False, True]),
        "quality_reason_code": np.asarray([0, 1, 0], dtype=np.uint8),
        "quality_bad_channel_mask": np.asarray([[False, False], [True, True], [False, False]]),
        "window_end_seconds": np.asarray([4.0, 8.0, 12.0]),
    }
    for key, value in changes.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    np.savez(path, **values)


def _manifest(root, preprocessing="robust_v1"):
    (root / "sessions.json").write_text(json.dumps([
        {"session_id": "record_1", "subject": "subject_1", "path": "record.npz"},
    ]), encoding="utf-8")
    (root / "dataset_metadata.json").write_text(json.dumps({
        "preprocessing": preprocessing, "n_features": 10, "channel_names": ["C3", "C4"],
    }), encoding="utf-8")


def test_feature_contract_uses_quality_and_timestamps_without_reading_labels(tmp_path):
    path = tmp_path / "record.npz"
    _robust_archive(path, y=np.full(3, np.nan))
    x, accepted, reasons, ends = read_feature_contract(
        path, n_features=10, expected_preprocessing="robust_v1", expected_channels=["C3", "C4"],
    )
    assert x.shape == (3, 10)
    np.testing.assert_array_equal(accepted, [True, False, True])
    np.testing.assert_array_equal(reasons, ["", "too_many_bad_channels", ""])
    np.testing.assert_array_equal(ends, [4, 8, 12])


def test_session_loader_preserves_contract_and_blocks_legacy_packing(tmp_path):
    _robust_archive(tmp_path / "record.npz")
    _manifest(tmp_path)
    session, = load_sessions(tmp_path)
    assert session.preprocessing == "robust_v1"
    assert session.channel_names == ("C3", "C4")
    np.testing.assert_array_equal(session.quality_valid, [True, False, True])
    np.testing.assert_array_equal(session.window_end_seconds, [4, 8, 12])
    with pytest.raises(ValueError, match="improve workflow"):
        pack_sessions([session], [session.subject], context=1)


@pytest.mark.parametrize("changes,match", [
    ({"quality_valid": None}, "requires.*quality"),
    ({"quality_reason_code": None}, "requires.*quality"),
    ({"channel_names": None}, "requires.*channel"),
    ({"window_end_seconds": None}, "requires.*timestamp"),
    ({"quality_valid": [1, 0, 1]}, "boolean"),
    ({"quality_valid": [True, False]}, "boolean"),
    ({"quality_reason_code": [0, 0, 0]}, "disagree"),
    ({"quality_reason_code": [0, 8, 0]}, "reason codes"),
    ({"window_end_seconds": [4, 4, 12]}, "timestamps"),
    ({"window_end_seconds": [4, 8, np.inf]}, "timestamps"),
    ({"channel_names": ["C3", "C3"]}, "unique"),
    ({"preprocessing": ["robust_v1"]}, "scalar string"),
])
def test_unsafe_robust_contract_is_rejected(tmp_path, changes, match):
    path = tmp_path / "record.npz"
    _robust_archive(path, **changes)
    with pytest.raises(ValueError, match=match):
        read_feature_contract(path)


def test_checkpoint_rejects_legacy_features_or_reversed_channel_order(tmp_path):
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, x=np.zeros((3, 10)))
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        read_feature_contract(legacy, expected_preprocessing="robust_v1")
    robust = tmp_path / "record.npz"
    _robust_archive(robust, channel_names=["C4", "C3"])
    with pytest.raises(ValueError, match="channel order"):
        read_feature_contract(robust, expected_channels=["C3", "C4"])


def test_dataset_metadata_must_match_archive_preprocessing(tmp_path):
    _robust_archive(tmp_path / "record.npz")
    _manifest(tmp_path, preprocessing="legacy_car")
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        load_sessions(tmp_path)


def test_legacy_npz_is_readable_with_safe_quality_defaults(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, x=np.ones((3, 10)))
    x, accepted, reasons, ends = read_feature_contract(path, expected_preprocessing="legacy_car")
    assert x.shape == (3, 10)
    assert accepted.all()
    assert np.all(reasons == "")
    assert ends is None
