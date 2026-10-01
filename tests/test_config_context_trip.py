from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from driverguard.config import TUNABLE_KEYS, LowLightCfg, apply_updates, load_config, tunable_schema
from driverguard.risk.context import TripContext
from driverguard.trip.memory import TripMemory, theil_sen_slope
from driverguard.vehicle import SimulatedCanBus, decode_speed_frame, encode_speed_frame


def test_yaml_matches_defaults_and_validates():
    cfg = load_config()
    assert cfg.low_light == LowLightCfg()
    assert cfg.eye.microsleep_ms > cfg.eye.long_closure_ms
    assert set(cfg.risk.weights) == {"eye", "perclos", "yawn", "head", "phone"}


@pytest.mark.parametrize("updates", [
    {"low_light.min_usable_brightness": 60},
    {"low_light.brightness_trigger": 8},
    {"low_light.gamma": 0.1},
    {"low_light.clahe_clip_limit": 5},
])
def test_low_light_config_rejects_invalid_enhancement_bounds(updates):
    with pytest.raises(ValidationError):
        load_config(overrides=updates)


def test_hitl_whitelist_and_bounds():
    cfg = load_config()
    new = apply_updates(cfg, {"eye.long_closure_ms": 1200})
    assert new.eye.long_closure_ms == 1200 and cfg.eye.long_closure_ms == 1000  # immutable original
    try:
        apply_updates(cfg, {"risk.weights": {"eye": 1}})
        raise AssertionError("non-tunable key accepted")
    except KeyError:
        pass
    try:
        apply_updates(cfg, {"eye.microsleep_ms": 800})  # < long_closure_ms -> invalid
        raise AssertionError("invalid value accepted")
    except ValidationError:
        pass
    schema = {r["key"]: r for r in tunable_schema(cfg)}
    assert set(schema) == set(TUNABLE_KEYS)
    assert schema["eye.long_closure_ms"]["min"] == 300 and schema["eye.long_closure_ms"]["max"] == 5000


def test_can_roundtrip():
    cid, data = encode_speed_frame(87.456)
    assert abs(decode_speed_frame(cid, data) - 87.46) < 1e-9
    assert decode_speed_frame(0x123, data) is None
    bus = SimulatedCanBus("mixed")
    assert bus.read_speed(10_000) == 0.0 and bus.read_speed(400_000) > 60


def test_context_circadian_and_rest_reset():
    cfg = load_config().context
    start = datetime(2026, 9, 28, 2, 30, tzinfo=timezone(timedelta(hours=7))).timestamp()
    ctx = TripContext(cfg, start, tz_offset_h=7)
    st = ctx.update(0, 60)
    assert st.circadian_low and st.sensitivity < 1
    t = 0
    for _ in range(4 * 60 + 1):  # 4h of driving, one tick per minute
        t += 60_000
        st = ctx.update(t, 60)
    assert st.long_drive
    for _ in range(16):  # 16 min parked -> reset
        t += 60_000
        st = ctx.update(t, 0)
    assert st.drive_time_min == 0 and st.parked


def test_theil_sen_robust_to_outlier():
    xs = list(range(10))
    ys = [2.0 * x for x in xs]
    ys[5] = 100
    assert abs(theil_sen_slope(xs, ys) - 2.0) < 1e-9


def test_trip_memory_trend_rising():
    cfg = load_config()
    mem = TripMemory(cfg.trip, cfg.eye.long_closure_ms)
    ts = 0
    for minute in range(25):
        perclos = 0.04 + 0.01 * minute  # slowly increasing closure proportion
        for _ in range(60 * 5):  # 5 Hz ticks
            mem.update(ts, True, False, perclos, None, False, None, 0)
            ts += 200
    assert mem.trend is not None and mem.trend > 0
    assert mem.trend_rising
    s = mem.summary()
    assert s["minutes"] >= 24 and len(s["fatigue_series"]) == len(mem.buckets)
