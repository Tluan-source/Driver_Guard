
import numpy as np
import pytest

from driverguard.learning.eeg import CHANNEL_NAMES, bandpower_window, extract_cnt
from scripts.prepare_fatigue_eeg import unique_recordings


@pytest.mark.parametrize("frequency,band", [(2, 0), (6, 1), (10, 2), (20, 3), (40, 4)])
def test_sinusoid_selects_expected_band_in_channel_major_order(frequency, band):
    time = np.arange(4000) / 1000
    signal = np.sin(2 * np.pi * frequency * time) * 10e-6
    powers = bandpower_window(np.stack([signal, -signal]), 1000).reshape(2, 5)
    assert np.argmax(powers[0]) == band
    np.testing.assert_allclose(powers[0], powers[1])
    assert powers[0, band] > max(np.delete(powers[0], band)) + 1


def test_common_signal_is_cancelled_without_changing_eeg_bandpower():
    time = np.arange(4000) / 1000
    signal = np.sin(2 * np.pi * 10 * time) * 10e-6
    common = np.sin(2 * np.pi * 2 * time) * 500e-6 + 0.012
    clean = np.stack([signal, -signal])
    np.testing.assert_allclose(
        bandpower_window(clean + common, 1000), bandpower_window(clean, 1000), atol=1e-5
    )


@pytest.mark.parametrize("rate", [0, 99, np.nan, np.inf, -np.inf])
def test_invalid_sample_rate_fails_cleanly(rate):
    with pytest.raises(ValueError, match="Sample rate"):
        bandpower_window(np.ones((2, 1000)), rate)


@pytest.mark.parametrize("samples", [np.ones(1000), np.ones((1, 1000)), np.ones((2, 999))])
def test_invalid_eeg_dimensions_are_rejected(samples):
    with pytest.raises(ValueError, match="EEG window"):
        bandpower_window(samples, 1000)


def test_nonfinite_or_flat_eeg_is_not_treated_as_valid_measurement():
    samples = np.ones((2, 1000))
    samples[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bandpower_window(samples, 1000)
    with pytest.raises(ValueError, match="flat"):
        bandpower_window(np.zeros((30, 4000)), 1000)


class FakeRaw:
    def __init__(self, n_times=9000):
        self.n_times = n_times
        self.ch_names = list(CHANNEL_NAMES) + [f"OTHER_{index}" for index in range(10)]
        self.info = {"sfreq": 1000.0}
        self.reads = []
        self.closed = False

    def get_data(self, picks, start, stop):
        self.reads.append((start, stop))
        time = np.arange(start, stop) / 1000
        return np.arange(1, len(picks) + 1)[:, None] * np.sin(2 * np.pi * 10 * time) * 1e-6

    def close(self):
        self.closed = True


def mock_cnt(monkeypatch, tmp_path, raw):
    import mne

    path = tmp_path / "recording.cnt"
    path.write_bytes(b"\0" * (900 + 75 * 40 + 4 * 40 * 9000))
    call = {}

    def reader(file_path, **kwargs):
        call.update(kwargs)
        return raw

    monkeypatch.setattr(mne.io, "read_raw_cnt", reader)
    return path, call


def test_cnt_reader_recomputes_header_and_reads_only_complete_windows(monkeypatch, tmp_path):
    raw = FakeRaw()
    path, call = mock_cnt(monkeypatch, tmp_path, raw)
    x, ends, channels = extract_cnt(path)
    assert call["data_format"] == "int32" and call["recompute_n_samples"] is True
    assert call["preload"] is False
    assert x.shape == (2, 150)
    np.testing.assert_array_equal(ends, [4, 8])
    assert channels == list(CHANNEL_NAMES)
    assert raw.reads == [(0, 4000), (4000, 8000)]
    assert raw.closed


def test_cnt_reader_rejects_impossible_header_before_loading_samples(monkeypatch, tmp_path):
    raw = FakeRaw(n_times=536870426)
    path, _ = mock_cnt(monkeypatch, tmp_path, raw)
    with pytest.raises(ValueError, match="physical file size"):
        extract_cnt(path)
    assert not raw.reads and raw.closed


def test_identical_duplicate_recordings_are_counted_once(tmp_path):
    shallow = tmp_path / "1" / "Fatigue state.cnt"
    deep = tmp_path / "1" / "1" / "Fatigue state.cnt"
    deep.parent.mkdir(parents=True)
    shallow.write_bytes(b"same recording")
    deep.write_bytes(b"same recording")
    recordings, duplicates = unique_recordings(tmp_path)
    assert recordings == [deep]
    assert duplicates[0]["duplicates"] == [str(shallow)]
    shallow.write_bytes(b"different recording")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        unique_recordings(tmp_path)
