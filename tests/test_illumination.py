import cv2
import numpy as np
import pytest

from driverguard.config import LowLightCfg
from driverguard.perception.illumination import LowLightEnhancer


def _dark_pattern(height=96, width=128):
    luma = np.tile(np.linspace(12, 52, width, dtype=np.uint8), (height, 1))
    return cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)


def test_normal_exposure_and_disabled_return_original():
    normal = np.full((96, 128, 3), 110, np.uint8)
    out, applied = LowLightEnhancer(LowLightCfg()).prepare(normal)
    assert out is normal and not applied
    dark = _dark_pattern()
    out, applied = LowLightEnhancer(LowLightCfg(enabled=False)).prepare(dark)
    assert out is dark and not applied


@pytest.mark.parametrize("level", [0, 3, 20])
def test_black_and_dim_flat_frames_return_original(level):
    frame = np.full((96, 128, 3), level, np.uint8)
    out, applied = LowLightEnhancer(LowLightCfg()).prepare(frame)
    assert out is frame and not applied


def test_dark_sensor_noise_and_sparse_hot_pixels_are_not_amplified():
    noise = np.random.default_rng(42).integers(10, 14, (96, 128, 3), dtype=np.uint8)
    noise[0, 0] = 255
    out, applied = LowLightEnhancer(LowLightCfg()).prepare(noise)
    assert out is noise and not applied
    mostly_black = np.zeros((96, 128, 3), np.uint8)
    mostly_black[0:4, 0:4] = 255
    out, applied = LowLightEnhancer(LowLightCfg()).prepare(mostly_black)
    assert out is mostly_black and not applied


def test_dark_pattern_is_enhanced_without_mutating_input():
    frame = _dark_pattern()
    original = frame.copy()
    enhancer = LowLightEnhancer(LowLightCfg())
    out, applied = enhancer.prepare(frame)
    assert applied and out is not frame
    assert out.dtype == frame.dtype and out.shape == frame.shape
    assert np.array_equal(frame, original)
    assert np.mean(out) > np.mean(frame)
    assert np.min(out.astype(np.int16) - frame) >= 0
    assert np.max(out.astype(np.int16) - frame) <= enhancer.MAX_LUMA_LIFT
    second, second_applied = enhancer.prepare(frame)
    assert second_applied and np.array_equal(out, second)


def test_periodic_contrast_is_not_lost_to_sampling_alias():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[::2] = 40
    original = frame.copy()
    out, applied = LowLightEnhancer(LowLightCfg()).prepare(frame)
    assert applied and np.mean(out) > np.mean(frame)
    assert np.array_equal(frame, original)


def test_enhancement_approximately_preserves_chroma():
    ycrcb = cv2.cvtColor(_dark_pattern(), cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 1] = 134
    ycrcb[:, :, 2] = 122
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    out, applied = LowLightEnhancer(LowLightCfg()).prepare(frame)
    assert applied
    original_chroma = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(np.int16)
    output_chroma = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(np.int16)
    assert np.max(np.abs(output_chroma - original_chroma)) <= 1


def test_saturated_color_respects_rgb_headroom_and_preserves_chroma():
    frame = np.zeros((96, 256, 3), np.uint8)
    frame[:, :, 0] = np.arange(256, dtype=np.uint8)
    original = frame.copy()
    enhancer = LowLightEnhancer(LowLightCfg())
    out, applied = enhancer.prepare(frame)
    assert applied and np.array_equal(frame, original)
    lift = out.astype(np.int16) - frame
    assert np.array_equal(lift[:, :, 0], lift[:, :, 1])
    assert np.array_equal(lift[:, :, 0], lift[:, :, 2])
    assert np.all((lift >= 0) & (lift <= enhancer.MAX_LUMA_LIFT))
    assert np.array_equal(out[:, -1], frame[:, -1])
    original_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    output_ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    assert np.max(np.abs(output_ycrcb[:, :, 1:] - original_ycrcb[:, :, 1:])) <= 1
    assert np.max(output_ycrcb[:, :, 0] - original_ycrcb[:, :, 0]) <= enhancer.MAX_LUMA_LIFT


@pytest.mark.parametrize("frame", [
    np.zeros((10, 10), np.uint8),
    np.zeros((10, 10, 1), np.uint8),
    np.zeros((10, 10, 4), np.uint8),
    np.zeros((0, 10, 3), np.uint8),
    np.zeros((10, 0, 3), np.uint8),
])
def test_invalid_shape_is_rejected(frame):
    with pytest.raises(ValueError, match="HxWx3"):
        LowLightEnhancer(LowLightCfg()).prepare(frame)


@pytest.mark.parametrize("frame", [None, [[0, 0, 0]], np.zeros((10, 10, 3), np.float32)])
def test_invalid_type_is_rejected(frame):
    with pytest.raises(TypeError):
        LowLightEnhancer(LowLightCfg()).prepare(frame)


def test_reuses_lut_and_clahe_at_camera_resolution(monkeypatch):
    enhancer = LowLightEnhancer(LowLightCfg())
    lut, clahe = enhancer._gamma_lut, enhancer._clahe

    def unexpected_create(*args, **kwargs):
        raise AssertionError("CLAHE must be constructed once")

    monkeypatch.setattr(cv2, "createCLAHE", unexpected_create)
    for _ in range(3):
        out, applied = enhancer.prepare(_dark_pattern(480, 640))
        assert applied and out.shape == (480, 640, 3)
    assert enhancer._gamma_lut is lut and enhancer._clahe is clahe
