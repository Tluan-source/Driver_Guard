import math

import pytest

from driverguard.learning.blink import BlinkKineticsTracker, blink_kinetics_features


def _cycle(reopening_end_ms=500):
    return [(0, 1.0), (100, 0.8), (200, 0.2), (300, 0.2), (reopening_end_ms, 0.8)]


def test_reopening_excludes_closed_plateau_and_uses_timestamps():
    tracker = BlinkKineticsTracker(closed_threshold=0.2, max_gap_ms=1000)
    results = [tracker.update(ts, value) for ts, value in _cycle()]
    assert all(result is None for result in results[:-1])
    cycle = results[-1]
    assert cycle is not None
    assert cycle.completed_at_ms == pytest.approx(500)
    assert cycle.closing_ms == pytest.approx(100)
    assert cycle.reopening_ms == pytest.approx(200)
    assert cycle.reopening_slope_per_s == pytest.approx(3)
    assert cycle.reopening_closing_ratio == pytest.approx(2)
    assert tracker.update(600, 1.0) is None


def test_prolonged_reopening_changes_kinetics_in_expected_direction():
    def features(end):
        samples = _cycle(end)
        return blink_kinetics_features(
            [ts for ts, _ in samples], [value for _, value in samples],
            closed_threshold=0.2, max_gap_ms=1000,
        )
    fast, slow = features(400), features(700)
    assert slow["mean_reopening_ms"] == pytest.approx(4 * fast["mean_reopening_ms"])
    assert slow["mean_reopening_slope_per_s"] == pytest.approx(fast["mean_reopening_slope_per_s"] / 4)
    assert slow["mean_reopening_closing_ratio"] == pytest.approx(4 * fast["mean_reopening_closing_ratio"])


@pytest.mark.parametrize("fps", [10, 15, 30, 60])
def test_threshold_interpolation_is_frame_rate_independent(fps):
    tracker = BlinkKineticsTracker(closed_threshold=0.2)
    cycles = []
    for frame in range(int(2 * fps) + 1):
        ts = frame * 1000 / fps
        if ts <= 1000:
            openness = 1.0
        elif ts < 1200:
            openness = 1 - (ts - 1000) / 200
        elif ts < 1250:
            openness = 0.0
        elif ts < 1750:
            openness = (ts - 1250) / 500
        else:
            openness = 1.0
        cycle = tracker.update(ts, openness)
        if cycle is not None:
            cycles.append(cycle)
    assert len(cycles) == 1
    assert cycles[0].closing_ms == pytest.approx(120)
    assert cycles[0].reopening_ms == pytest.approx(300)
    assert cycles[0].reopening_closing_ratio == pytest.approx(2.5)


@pytest.mark.parametrize("invalid", [None, math.nan, math.inf, -0.1])
def test_invalid_eye_sample_aborts_cycle_without_fabricated_evidence(invalid):
    tracker = BlinkKineticsTracker(closed_threshold=0.2)
    for sample in _cycle()[:3]:
        assert tracker.update(*sample) is None
    assert tracker.update(250, invalid) is None
    assert tracker.update(300, 0.2) is None
    assert tracker.update(400, 0.8) is None
    assert tracker.update(500, 1.0) is None


@pytest.mark.parametrize("timestamp", [200, 190, 600, math.nan, math.inf, -1])
def test_timestamp_discontinuity_aborts_incomplete_cycle(timestamp):
    tracker = BlinkKineticsTracker(closed_threshold=0.2)
    for sample in _cycle()[:3]:
        assert tracker.update(*sample) is None
    assert tracker.update(timestamp, 0.2) is None
    assert tracker.update(700, 0.8) is None
    assert tracker.update(800, 1.0) is None


def test_partial_blink_and_window_edges_have_missing_kinetics():
    for timestamps, openness in [
        ([0, 100, 200], [1.0, 0.4, 1.0]),
        ([0, 100, 200], [0.2, 0.4, 1.0]),
        ([0, 100, 200], [1.0, 0.4, 0.2]),
    ]:
        features = blink_kinetics_features(timestamps, openness, closed_threshold=0.2)
        assert features["blink_count"] == 0
        assert all(value is None for key, value in features.items() if key != "blink_count")


def test_reclosure_restarts_reopening_measurement():
    tracker = BlinkKineticsTracker(closed_threshold=0.2)
    samples = [(0, 1.0), (100, 0.2), (200, 0.5), (300, 0.2), (400, 0.8)]
    cycles = [tracker.update(ts, value) for ts, value in samples]
    assert all(c is None for c in cycles[:-1])
    assert cycles[-1] is not None
    assert cycles[-1].reopening_ms == pytest.approx(100)


def test_batch_aggregate_and_length_validation():
    samples = _cycle(400) + [(500 + ts, value) for ts, value in _cycle(500)]
    features = blink_kinetics_features(
        (ts for ts, _ in samples), (value for _, value in samples),
        closed_threshold=0.2, max_gap_ms=1000,
    )
    assert features["blink_count"] == 2
    assert features["mean_reopening_ms"] == pytest.approx(150)
    with pytest.raises(ValueError, match="same length"):
        blink_kinetics_features([0, 100], [1.0])


@pytest.mark.parametrize("kwargs", [
    {"closed_threshold": 0.8}, {"closed_threshold": -1},
    {"open_threshold": math.nan}, {"max_gap_ms": 0}, {"max_gap_ms": math.inf},
])
def test_invalid_measurement_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BlinkKineticsTracker(**kwargs)
