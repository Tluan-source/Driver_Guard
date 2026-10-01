import numpy as np

from driverguard.config import load_config
from driverguard.learning.camera_cli import CameraRuntime
from driverguard.learning.camera_model import CameraPrediction
from driverguard.schemas import FrameSignals


class FakePredictor:
    context = 50
    decision_threshold = 0.6

    def predict_window(self, x, timestamps, valid):
        if len(x) < 50:
            return CameraPrediction(None, "insufficient_history", float(valid.mean()) if len(valid) else 0)
        if not valid[-1]:
            return CameraPrediction(None, "current_frame_invalid", float(valid.mean()))
        return CameraPrediction(0.8, None, float(valid.mean()))


def _signals(ts, valid=True):
    return FrameSignals(ts_ms=ts, face_detected=True, face_quality=0.9 if valid else 0,
                        ear_left=0.3, ear_right=0.3, mar=0.12, yaw=0, pitch=0, roll=0)


def test_camera_runtime_model_requires_history_and_persistence_for_alert():
    cfg = load_config()
    runtime = CameraRuntime(FakePredictor(), cfg)
    for ts in range(0, 6401, 100):
        result = runtime.step(_signals(ts))
        if ts < 4900:
            assert result["camera_drowsiness_score"] is None
            assert result["assessment_status"] == "cannot_assess"
        if ts == 4900:
            assert result["camera_drowsiness_score"] == 0.8
            assert result["risk_level"] == "normal"
    assert result["risk_level"] == "warning"
    assert "RISK_SCORE" in result["reason_codes"]
    assert result["alert"]["level"] == "warning"


def test_runtime_never_reuses_score_on_invalid_raw_face_and_resets_long_gaps():
    runtime = CameraRuntime(FakePredictor(), load_config())
    for ts in range(0, 5000, 100):
        runtime.step(_signals(ts))
    result = runtime.step(_signals(4950, valid=False))
    assert result["camera_drowsiness_score"] is None
    assert result["score_timestamp_ms"] is None
    assert not result["face_valid"]
    result = runtime.step(_signals(5400))
    assert result["model_status"] == "insufficient_history"
    assert len(runtime.buffer.arrays()[0]) == 1
    for ts in range(5500, 8901, 100):
        result = runtime.step(_signals(ts, valid=False))
    assert result["risk_level"] == "sensor_degraded"
    assert result["camera_drowsiness_score"] is None
    assert np.all(runtime.buffer.arrays()[0][1:] == 0)


def test_runtime_timestamp_reset_never_keeps_a_stale_score():
    runtime = CameraRuntime(FakePredictor(), load_config())
    for ts in range(0, 5000, 100):
        runtime.step(_signals(ts))
    assert runtime.last_prediction.score == 0.8
    result = runtime.step(_signals(4900))
    assert result["camera_drowsiness_score"] is None
    assert result["score_timestamp_ms"] is None


def test_dual_stream_reset_never_populates_history_from_fast_tracking():
    runtime = CameraRuntime(FakePredictor(), load_config(), independent_model_stream=True)
    runtime.step(_signals(0), _signals(0))
    assert len(runtime.buffer.arrays()[0]) == 1
    runtime.step(_signals(0))
    fast = _signals(33)
    fast.ear_left = fast.ear_right = 0.02
    result = runtime.step(fast)
    assert len(runtime.buffer.arrays()[0]) == 0
    assert result["camera_drowsiness_score"] is None
    runtime.step(_signals(100), _signals(100))
    x, timestamps, _ = runtime.buffer.arrays()
    assert timestamps.tolist() == [100]
    assert x[0, 0] == 1.0


def test_dual_stream_timestamp_reset_discards_an_existing_score():
    runtime = CameraRuntime(FakePredictor(), load_config(), independent_model_stream=True)
    for ts in range(0, 5000, 100):
        runtime.step(_signals(ts), _signals(ts))
    assert runtime.last_prediction.score == 0.8
    result = runtime.step(_signals(4900))
    assert result["camera_drowsiness_score"] is None
    assert result["score_timestamp_ms"] is None
    result = runtime.step(_signals(4933))
    assert result["model_status"] == "insufficient_history"
    assert len(runtime.buffer.arrays()[0]) == 0


def test_small_model_threshold_warning_can_recover_and_does_not_bridge_abstention():
    class VariablePredictor(FakePredictor):
        decision_threshold = 0.05
        score = 0.8

        def predict_window(self, x, timestamps, valid):
            result = super().predict_window(x, timestamps, valid)
            return CameraPrediction(self.score, None, result.valid_fraction) if result.score is not None else result

    predictor = VariablePredictor()
    runtime = CameraRuntime(predictor, load_config())
    for ts in range(0, 6501, 100):
        result = runtime.step(_signals(ts))
    assert result["risk_level"] == "warning"
    predictor.score = 0.0
    for ts in range(6600, 9901, 100):
        result = runtime.step(_signals(ts))
    assert result["risk_level"] == "normal"
    predictor.score = 0.8
    runtime.step(_signals(10000))
    runtime.step(_signals(10100, valid=False))
    for ts in range(10200, 11601, 100):
        result = runtime.step(_signals(ts))
    assert result["risk_level"] == "normal"
    result = runtime.step(_signals(11700))
    assert result["risk_level"] == "warning"
