import numpy as np

from driverguard.config import load_config
from driverguard.learning.camera_features import CameraFeatureCollector, CameraFrameBuffer, encode_frame
from driverguard.schemas import FrameSignals


def _signals(ts, **changes):
    values = {"ts_ms": ts, "camera_ok": True, "face_detected": True,
              "face_quality": 0.9, "ear_left": 0.3, "ear_right": 0.3, "mar": 0.12,
              "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    values.update(changes)
    return FrameSignals(**values)


def test_camera_encoder_never_promotes_bad_raw_quality():
    cfg = load_config()
    x, valid = encode_frame(_signals(0, face_quality=0.2), cfg)
    assert not valid and not np.any(x)
    x, valid = encode_frame(_signals(0, ear_left=float("nan"), yaw=50), cfg)
    assert valid and np.isfinite(x).all()
    assert x[0] == x[1] == x[9] == x[10] == 0
    assert x[2] > 0 and x[11] == x[-1] == 1


def test_camera_buffer_missing_bins_are_invalid_without_repeating_face():
    buffer = CameraFrameBuffer(load_config(), context=50)
    assert buffer.update(_signals(0))
    assert not buffer.update(_signals(60))
    assert buffer.update(_signals(250))
    x, timestamps, valid = buffer.arrays()
    assert list(timestamps) == [0, 100, 200]
    assert list(valid) == [True, False, True]
    assert not np.any(x[1])


def test_camera_buffer_long_loss_and_timestamp_restart_clear_history():
    buffer = CameraFrameBuffer(load_config(), context=50)
    buffer.update(_signals(0))
    buffer.update(_signals(100))
    buffer.update(_signals(500))
    assert list(buffer.arrays()[1]) == [500]
    assert not buffer.update(_signals(500))
    assert len(buffer.arrays()[0]) == 0


def test_camera_buffer_window_is_past_only_and_bounded():
    buffer = CameraFrameBuffer(load_config(), context=3)
    for ts in [0, 100, 200]:
        buffer.update(_signals(ts))
    before = tuple(a.copy() for a in buffer.arrays())
    buffer.update(_signals(300, ear_left=0.0))
    assert list(before[1]) == [0, 100, 200]
    assert np.all(before[0][:, 0] == 1)
    assert list(buffer.arrays()[1]) == [100, 200, 300]


def test_offline_collector_preserves_long_clips_and_preceding_gap_history():
    collector = CameraFeatureCollector(load_config())
    for ts in range(0, 250100, 100):
        collector.update(_signals(ts))
    collector.update(_signals(251000))
    x, timestamps, valid = collector.arrays()
    assert len(x) == 2502
    assert timestamps[0] == 0 and timestamps[-1] == 251000
    assert np.all(valid) and np.max(np.diff(timestamps)) == 1000
