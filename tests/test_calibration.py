"""Hướng 1 sanity check on SYNTHETIC data (not evidence — see sim/scenarios.py)."""
import collections

from driverguard.calibration import AdaptiveEyeBaseline
from driverguard.config import load_config
from driverguard.engine import DriverGuardEngine
from driverguard.sim import generate

EPOCH = 1_789_959_600  # 2026-09-21 10:00 in UTC+7 (daytime, outside circadian-low hours)
TZ = 7


def test_adaptive_baseline_converges_to_personal_level():
    cfg = load_config()
    b = AdaptiveEyeBaseline(cfg.calibration)
    for i in range(int(90 * 15)):
        b.update(int(i * 1000 / 15), 0.21, usable=True)
    assert b.calibrated
    assert abs(b.open_level - 0.21) < 0.01
    assert abs(b.ratio(0.21) - 1.0) < 0.05


def test_baseline_floor_prevents_learning_drowsiness():
    cfg = load_config()
    b = AdaptiveEyeBaseline(cfg.calibration)
    t = 0
    for _ in range(int(70 * 15)):
        b.update(t, 0.30, usable=True)
        t += 66
    level = b.calibrated_level
    for _ in range(int(3600 * 15)):  # one hour of droopy eyes
        b.update(t, 0.20, usable=True)
        t += 66
    assert b.open_level >= level * (1 - cfg.calibration.max_drift_down) - 1e-9


def _levels(method: str):
    cfg = load_config(overrides={"calibration.method": method})
    eng = DriverGuardEngine(cfg, trip_start_epoch_s=EPOCH, tz_offset_h=TZ)
    c = collections.Counter()
    for s in generate("narrow_eyes"):
        c[eng.step(s).risk_level.value] += 1
    return c


def test_narrow_eyes_fixed_threshold_false_alarms_adaptive_does_not():
    fixed, adaptive = _levels("fixed"), _levels("adaptive")
    assert fixed["critical"] + fixed["warning"] > 1000  # imported threshold: constant false alarms
    assert adaptive["critical"] == adaptive["warning"] == 0
