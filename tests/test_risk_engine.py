"""Safety rules of the deterministic engine, checked on synthetic scenarios (research doc "Test matrix")."""
from driverguard.config import load_config
from driverguard.engine import DriverGuardEngine
from driverguard.schemas import FrameSignals, RiskLevel
from driverguard.sim import generate

EPOCH = 1_789_959_600  # 2026-09-21 10:00 in UTC+7 (daytime, outside circadian-low hours)
TZ = 7


def run(name, fps=15.0, **ov):
    cfg = load_config(overrides=ov or None)
    eng = DriverGuardEngine(cfg, trip_start_epoch_s=EPOCH, tz_offset_h=TZ)
    return [eng.step(s) for s in generate(name, fps=fps)]


def first(ticks, pred):
    for t in ticks:
        if pred(t):
            return t
    return None


def test_normal_driving_no_warning():
    ticks = run("normal")
    assert not any(t.risk_level.severity >= RiskLevel.CAUTION.severity for t in ticks)
    assert not any(t.alert for t in ticks)


def test_microsleep_goes_critical_quickly():
    ticks = run("drowsy")
    t = first(ticks, lambda t: t.risk_level == RiskLevel.CRITICAL)
    assert t is not None
    assert 230_000 + 2000 <= t.timestamp_ms <= 230_000 + 2000 + 150  # rule-true -> alert within one frame
    assert "MICROSLEEP" in t.reason_codes
    assert t.alert is not None


def test_timing_is_correct_at_low_fps():
    """FPS drop 30 -> 10: timers use timestamps, so the critical alert time barely moves."""
    t30 = first(run("drowsy", fps=30), lambda t: t.risk_level == RiskLevel.CRITICAL)
    t10 = first(run("drowsy", fps=10), lambda t: t.risk_level == RiskLevel.CRITICAL)
    assert abs(t30.timestamp_ms - t10.timestamp_ms) <= 150


def test_mirror_check_and_phone_on_mount_do_not_alert():
    ticks = run("distracted")
    during = [t for t in ticks if 110_000 <= t.timestamp_ms < 111_000 or 140_000 <= t.timestamp_ms < 150_000]
    assert all(t.risk_level == RiskLevel.NORMAL for t in during)


def test_phone_in_hand_and_looking_down_warn():
    ticks = run("distracted")
    look = first(ticks, lambda t: "LOOKING_DOWN" in t.reason_codes)
    phone = first(ticks, lambda t: "PHONE_USE" in t.reason_codes)
    assert look and 100_000 < look.timestamp_ms < 104_000 and look.risk_level == RiskLevel.WARNING
    assert phone and 120_000 < phone.timestamp_ms < 124_000


def test_phone_while_parked_is_not_an_alert():
    ticks = run("distracted")
    parked = [t for t in ticks if 158_000 <= t.timestamp_ms < 166_000]
    assert all("PHONE_USE" not in t.reason_codes for t in parked)
    assert all("PARKED" in t.reason_codes for t in parked)


def test_face_lost_is_degraded_not_normal():
    ticks = run("degraded")
    brief = [t for t in ticks if 80_000 <= t.timestamp_ms < 81_000]
    assert all(t.risk_level != RiskLevel.SENSOR_DEGRADED for t in brief)  # short dropout is held
    t = first(ticks, lambda t: t.risk_level == RiskLevel.SENSOR_DEGRADED)
    assert t and 93_000 <= t.timestamp_ms <= 93_200 and "FACE_NOT_VISIBLE" in t.reason_codes
    cam = first(ticks, lambda t: "CAMERA_LOST" in t.reason_codes)
    assert cam and cam.risk_level == RiskLevel.SENSOR_DEGRADED


def test_hysteresis_no_flapping():
    ticks = run("drowsy")
    changes = sum(1 for a, b in zip(ticks, ticks[1:]) if a.risk_level != b.risk_level)
    assert changes < 20


def test_camera_loss_from_the_start():
    cfg = load_config()
    eng = DriverGuardEngine(cfg, trip_start_epoch_s=EPOCH, tz_offset_h=TZ)
    out = [eng.step(FrameSignals(ts_ms=i * 66, camera_ok=False)) for i in range(100)]
    assert out[-1].risk_level == RiskLevel.SENSOR_DEGRADED
    assert "CAMERA_LOST" in out[-1].reason_codes


def test_runtime_reconfigure_keeps_state():
    from driverguard.config import apply_updates

    cfg = load_config()
    eng = DriverGuardEngine(cfg, trip_start_epoch_s=EPOCH, tz_offset_h=TZ)
    gen = generate("drowsy")
    for _, s in zip(range(1000), gen):
        eng.step(s)
    calibrated = eng.eye_base.calibrated
    eng.reconfigure(apply_updates(cfg, {"eye.microsleep_ms": 1500}))
    assert eng.eye_base.calibrated == calibrated
    ticks = [eng.step(s) for s in gen]
    t = first(ticks, lambda t: t.risk_level == RiskLevel.CRITICAL)
    assert t and t.timestamp_ms <= 230_000 + 1500 + 150
