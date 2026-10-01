import numpy as np
import pytest

from driverguard.learning.eeg import CHANNEL_NAMES, bandpower_window, robust_bandpower_window
from driverguard.learning.quality import robust_reference_window


def clean_window():
    time = np.arange(4000) / 1000.0
    rng = np.random.default_rng(37)
    phases = np.arange(30) * 2 * np.pi / 30
    return 15e-6 * np.sin(2 * np.pi * 10 * time + phases[:, None]) + rng.normal(0, 1e-6, (30, 4000))


def test_single_electrode_artifact_does_not_spill_into_healthy_reference():
    clean = clean_window()
    bad_index = CHANNEL_NAMES.index("FT7")
    contaminated = clean.copy()
    contaminated[bad_index] = np.sin(2 * np.pi * 6 * np.arange(4000) / 1000) * 0.01 + 100
    clean[bad_index] = 0.0
    expected, clean_quality = robust_bandpower_window(clean, 1000)
    actual, quality = robust_bandpower_window(contaminated, 1000)
    assert quality["quality_valid"] and clean_quality["quality_valid"]
    assert quality["quality_bad_channel_mask"][bad_index]
    assert quality["quality_good_channel_count"] == 29
    assert quality["quality_channel_reason_code"][bad_index] & 2
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    legacy = bandpower_window(contaminated, 1000).reshape(30, 5)
    assert np.median(legacy[:, 1]) > np.median(actual.reshape(30, 5)[:, 1]) + 3


def test_interpolation_retains_good_electrodes_and_reconstructs_finite_signal():
    clean = clean_window()
    bad_index = CHANNEL_NAMES.index("FT7")
    clean[bad_index] = 0.0
    values, quality = robust_reference_window(clean, CHANNEL_NAMES)
    good = ~quality["quality_bad_channel_mask"]
    centered = (clean - np.median(clean, axis=1, keepdims=True)) * 1e6
    expected = centered[good] - np.median(centered[good], axis=0, keepdims=True)
    np.testing.assert_allclose(values[good], expected, atol=1e-9)
    assert np.isfinite(values).all()
    assert np.ptp(values[bad_index]) > 1


def test_too_many_bad_electrodes_signal_abstention_without_dropping_window():
    samples = clean_window()
    samples[:7] = 0.0
    features, quality = robust_bandpower_window(samples, 1000)
    assert not quality["quality_valid"]
    assert quality["quality_reason_code"] & 1
    assert quality["quality_good_channel_count"] == 23
    assert quality["quality_bad_channel_fraction"] == pytest.approx(7 / 30)
    assert features.shape == (150,)
    np.testing.assert_array_equal(features, np.zeros(150))


def test_nonfinite_samples_signal_abstention_even_with_other_channels_usable():
    samples = clean_window()
    samples[3, 14] = np.nan
    features, quality = robust_bandpower_window(samples, 1000)
    assert not quality["quality_valid"]
    assert quality["quality_reason_code"] & 2
    assert quality["quality_channel_reason_code"][3] & 8
    assert np.isfinite(features).all()


def test_flat_and_common_only_windows_are_untrustworthy():
    _, flat = robust_bandpower_window(np.zeros((30, 4000)), 1000)
    assert not flat["quality_valid"]
    signal = clean_window()[0]
    _, common = robust_bandpower_window(np.repeat(signal[None], 30, axis=0), 1000)
    assert not common["quality_valid"]
    assert common["quality_reason_code"] & 4


def test_dc_offset_does_not_change_robust_features_or_quality():
    samples = clean_window()
    expected, expected_quality = robust_bandpower_window(samples, 1000)
    actual, actual_quality = robust_bandpower_window(samples + np.arange(30)[:, None], 1000)
    np.testing.assert_allclose(expected, actual, atol=1e-5)
    np.testing.assert_array_equal(expected_quality["quality_bad_channel_mask"], actual_quality["quality_bad_channel_mask"])
