"""Capture + perception with a FAKE landmarker (the real MediaPipe model is not needed in CI)."""
import os
import tempfile
from itertools import pairwise

import cv2
import numpy as np
import pytest

from driverguard.capture import VideoFileSource
from driverguard.config import load_config
from driverguard.engine import DriverGuardEngine
from driverguard.io import read_features, write_features
from driverguard.perception.extractor import PerceptionExtractor
from driverguard.perception.face import FaceObservation
from driverguard.perception.geometry import LEFT_EYE, MOUTH_CORNERS, MOUTH_VERTICAL, RIGHT_EYE
from driverguard.perception.head_pose import rotation_from_euler
from driverguard.perception.phone import PhoneObservation, phone_near_face
from driverguard.perception.quality import face_crop_stats
from driverguard.schemas import RiskLevel


def _synthetic_face(eye_h: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    pts = np.zeros((478, 3), np.float32)
    pts[:, 0] = rng.uniform(0.35, 0.65, 478)
    pts[:, 1] = rng.uniform(0.25, 0.75, 478)
    for idx, cx in ((RIGHT_EYE, 0.44), (LEFT_EYE, 0.56)):
        p1, p2, p3, p4, p5, p6 = idx
        w = 0.06
        pts[p1, :2], pts[p4, :2] = (cx - w / 2, 0.45), (cx + w / 2, 0.45)
        pts[p2, :2], pts[p3, :2] = (cx - w / 6, 0.45 - eye_h / 2), (cx + w / 6, 0.45 - eye_h / 2)
        pts[p6, :2], pts[p5, :2] = (cx - w / 6, 0.45 + eye_h / 2), (cx + w / 6, 0.45 + eye_h / 2)
    pts[MOUTH_CORNERS[0], :2], pts[MOUTH_CORNERS[1], :2] = (0.46, 0.62), (0.54, 0.62)
    for a, b in MOUTH_VERTICAL:
        pts[a, :2], pts[b, :2] = (0.5, 0.615), (0.5, 0.625)
    return pts


class FakeLandmarker:
    def __init__(self):
        self.closed = False

    def detect(self, img, ts):
        M = np.eye(4)
        M[:3, :3] = rotation_from_euler(10, -5, 0)
        return FaceObservation(_synthetic_face(0.005 if self.closed else 0.024),
                               {"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1, "jawOpen": 0.05}, M)

    def close(self):
        pass


class FakePhone:
    def detect(self, img, ts):
        return PhoneObservation(0.8, (0.45, 0.6, 0.55, 0.8))

    def close(self):
        pass


def _frame():
    rng = np.random.default_rng(1)  # textured mid-grey frame so brightness/blur metrics are realistic
    return np.clip(rng.normal(110, 40, (480, 640, 3)), 0, 255).astype(np.uint8)


def test_extractor_signals():
    cfg = load_config()
    lm = FakeLandmarker()
    ex = PerceptionExtractor(cfg, lm, FakePhone())
    s = ex.process(_frame(), 0)
    assert s.face_detected and s.face_quality > cfg.quality.min_quality
    assert 0.2 < s.ear_left < 0.4  # 0.024*480 / (0.06*640) ~ 0.30 in PIXEL space
    assert abs(s.yaw - 10) < 1e-6 and abs(s.pitch + 5) < 1e-6
    lm.closed = True
    s2 = ex.process(_frame(), 66)
    assert s2.ear_left < 0.1
    for i in range(3):
        s3 = ex.process(_frame(), 132 + 66 * i)
    assert s3.phone_conf == 0.8 and s3.phone_near_face
    assert ex.process(None, 400, camera_ok=False).camera_ok is False


class RecordingLandmarker(FakeLandmarker):
    def detect(self, img, ts):
        self.input = img
        return super().detect(img, ts)


class RecordingPhone(FakePhone):
    def detect(self, img, ts):
        self.input = img
        return super().detect(img, ts)


def test_low_light_detector_input_keeps_raw_quality_and_phone_pixels():
    cfg = load_config(overrides={"runtime.phone_every_n_frames": 1})
    lm, phone = RecordingLandmarker(), RecordingPhone()
    ex = PerceptionExtractor(cfg, lm, phone)
    frame = np.random.default_rng(7).integers(8, 28, (480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    s = ex.process(frame, 0)
    assert lm.input is not frame and lm.input.mean() > frame.mean()
    assert phone.input is frame
    assert np.array_equal(frame, original)
    bbox = (s.face_x0 * 640, s.face_y0 * 480, s.face_x1 * 640, s.face_y1 * 480)
    brightness, blur = face_crop_stats(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bbox)
    assert s.brightness == brightness and s.blur == blur
    enhanced_brightness, _ = face_crop_stats(cv2.cvtColor(lm.input, cv2.COLOR_BGR2GRAY), bbox)
    assert enhanced_brightness > s.brightness
    assert s.face_quality < cfg.quality.min_quality


@pytest.mark.parametrize("enhanceable", [False, True])
def test_dark_frames_cannot_supply_valid_healthy_evidence(enhanceable):
    cfg = load_config()
    ex = PerceptionExtractor(cfg, RecordingLandmarker())
    engine = DriverGuardEngine(cfg, trip_start_epoch_s=0, tz_offset_h=0)
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    if enhanceable:
        frame = np.random.default_rng(7).integers(8, 28, frame.shape, dtype=np.uint8)
    for ts in range(0, cfg.quality.degraded_after_ms + 101, 100):
        tick = engine.step(ex.process(frame, ts))
        assert not tick.face_valid
        assert tick.ear_norm is None and tick.eye_closed is None
    assert tick.risk_level == RiskLevel.SENSOR_DEGRADED
    assert "LOW_FACE_QUALITY" in tick.reason_codes


@pytest.mark.parametrize("disabled", [False, True])
def test_extractor_bright_and_disabled_paths_preserve_detector_input(disabled):
    cfg = load_config(overrides={"low_light.enabled": not disabled})
    lm = RecordingLandmarker()
    ex = PerceptionExtractor(cfg, lm)
    frame = _frame() if not disabled else np.full((96, 128, 3), 35, np.uint8)
    ex.process(frame, 0)
    assert lm.input is frame


def test_phone_far_from_face():
    assert not phone_near_face((0.0, 0.8, 0.1, 0.95), (0.4, 0.2, 0.6, 0.6), 1.6)


def test_video_source_timestamps_and_subsampling():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        p = os.path.join(d, "v.avi")
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
        for i in range(90):
            w.write(np.full((120, 160, 3), i, np.uint8))
        w.release()
        src = VideoFileSource(p, target_fps=15)
        ts = [f.ts_ms for f in src.frames()]
        src.close()
    assert 44 <= len(ts) <= 46
    assert all(b > a for a, b in pairwise(ts))
    assert abs((ts[1] - ts[0]) - 67) <= 1


def test_feature_io_roundtrip_csv():
    from driverguard.sim import generate

    rows = [s for _, s in zip(range(50), generate("normal"))]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        p = write_features(rows, os.path.join(d, "f.csv.gz"))
        back = list(read_features(p))
    assert len(back) == 50
    assert back[10].ts_ms == rows[10].ts_ms and abs(back[10].ear_left - rows[10].ear_left) < 1e-3
    assert back[10].face_detected is True
