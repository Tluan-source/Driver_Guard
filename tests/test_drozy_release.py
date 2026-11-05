import numpy as np
import pytest

from scripts.data.materialize_week6_review import display_oriented
from scripts.evaluation.audit_drozy_release import interpolation_check, sync_check


def test_review_export_honors_display_matrix_rotation():
    image = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(display_oriented(image, -90), [[4, 1], [5, 2], [6, 3]])
    np.testing.assert_array_equal(display_oriented(image, 0), image)
    with pytest.raises(ValueError):
        display_oriented(image, 45)


def _table():
    table = np.zeros((18000, 3))
    table[:, 0], table[:, 2] = np.arange(1, 18001), 1
    return table


def test_interpolation_uses_one_based_video_frames_and_zero_sentinel():
    table = _table()
    # Interpolate frame 2 exactly between frames 1 and 3.
    table[1] = [1, 3, .5]
    result = interpolation_check(table, np.arange(1, 18001)/30, 18000)
    assert result['both_grid_origins_within_100ms']
    assert result['interpolated_rows'] == 1
    assert result['grid_origin_candidates_max_error_s']['1'] < 1e-9


@pytest.mark.parametrize('bad_row', [[1, 0, .5], [0, 0, 1], [1, 18001, .5],
                                    [1.5, 0, 1], [1, 2, np.nan], [1, 2, 1.1]])
def test_interpolation_rejects_invalid_release_mappings(bad_row):
    table = _table()
    table[0] = bad_row
    with pytest.raises(ValueError):
        interpolation_check(table, np.arange(1, 18001)/30, 18000)


def test_equal_pulse_counts_require_actual_paired_clock_consistency():
    signal = np.ones(1000)
    for start, end in [(100, 200), (400, 600), (800, 1000)]:
        signal[start:end] = -1
    times = np.array([.1, .2, .4, .6, .8])
    assert sync_check(signal, 1000, times)['ordinal_clock_verified']
    bad = sync_check(signal, 1000, [.1, .2, .3, .6, .8])
    assert not bad['ordinal_clock_verified']


def test_missing_pulses_and_flat_sync_cannot_certify_alignment():
    signal = np.ones(1000)
    signal[100:200] = -1
    result = sync_check(signal, 1000, [.1, .2, .3])
    assert not result['ordinal_clock_verified']
    assert result['duplicate_nearest_matches'] == 1
    assert not sync_check(np.ones(100), 1000, [.01, .02, .03])['ordinal_clock_verified']
