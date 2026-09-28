from driverguard.config import load_config
from driverguard.temporal.primitives import PersistenceTimer, TimeWindow
from driverguard.temporal.trackers import EyeStateTracker


def _perclos_at_fps(fps: float) -> float:
    """20% closed pattern: 200 ms closed every 1000 ms, for 60 s."""
    w = TimeWindow(60_000, max_gap_ms=300)
    step = 1000.0 / fps
    t = 0.0
    while t < 60_000:
        w.add(int(t), (t % 1000) < 200)
        t += step
    frac, cov = w.time_fraction(lambda v: v, int(t))
    return frac


def test_perclos_is_fps_independent():
    vals = [_perclos_at_fps(f) for f in (10, 15, 30)]
    assert all(abs(v - 0.2) < 0.03 for v in vals), vals


def test_invalid_samples_are_not_closed():
    w = TimeWindow(10_000, max_gap_ms=300)
    for i in range(100):
        w.add(i * 100, None if i % 2 else False)  # half invalid, rest open
    frac, cov = w.time_fraction(lambda v: v, 10_000)
    assert frac == 0.0
    assert 0.4 < cov < 0.6


def test_persistence_timer():
    t = PersistenceTimer(on_ms=1000, off_ms=500)
    assert not t.update(True, 0)
    assert not t.update(True, 900)
    assert t.update(True, 1000)
    assert t.update(False, 1200)  # still active during off debounce
    assert not t.update(False, 1700)


def _run_eye(fps: float, closure_ms: int):
    cfg = load_config()
    tr = EyeStateTracker(cfg.eye, 300)
    step = 1000.0 / fps
    t, completed, max_ongoing = 0.0, [], 0
    while t < 5000:
        closed = 1000 <= t < 1000 + closure_ms
        st = tr.update(int(t), 0.3 if closed else 1.0)
        max_ongoing = max(max_ongoing, st.closure_ms)
        if st.completed_closure_ms is not None:
            completed.append(st.completed_closure_ms)
        t += step
    return completed, max_ongoing


def test_blink_duration_uses_timestamps_not_frames():
    for fps in (10, 15, 30):
        completed, _ = _run_eye(fps, 1500)
        assert len(completed) == 1
        assert abs(completed[0] - 1500) <= 1000 / fps + 1, (fps, completed)


def test_eye_dropout_during_closure_does_not_count_as_closed_forever():
    cfg = load_config()
    tr = EyeStateTracker(cfg.eye, 300)
    tr.update(0, 1.0)
    tr.update(100, 0.3)  # closed
    st = tr.update(200, None)  # short dropout -> keeps counting
    assert st.closed is True and st.closure_ms == 100
    st = tr.update(1000, None)  # long dropout -> abort (unknown, not closed)
    assert st.closed is None and st.closure_ms == 0
