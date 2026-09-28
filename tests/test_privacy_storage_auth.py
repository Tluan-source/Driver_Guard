import os
import tempfile

import numpy as np

from driverguard.api.auth import AuthService, User, hash_password, verify_password
from driverguard.config import load_config
from driverguard.engine import DriverGuardEngine
from driverguard.privacy import PrivacyViolation, assert_metadata_only
from driverguard.sim import generate
from driverguard.storage import EventStore


def _raises(fn):
    try:
        fn()
    except PrivacyViolation:
        return True
    return False


def test_privacy_guard_rejects_images():
    assert _raises(lambda: assert_metadata_only({"x": b"\xff\xd8jpeg"}))
    assert _raises(lambda: assert_metadata_only({"x": np.zeros((4, 4))}))
    assert _raises(lambda: assert_metadata_only({"face_image": "abc"}))
    assert _raises(lambda: assert_metadata_only({"landmarks": [[0.1, 0.2]]}))
    assert _raises(lambda: assert_metadata_only({"note": "A" * 5000}))  # base64-like blob


def test_every_tick_is_metadata_only():
    eng = DriverGuardEngine(load_config(), trip_start_epoch_s=1_789_959_600, tz_offset_h=7)
    for s in generate("demo"):
        assert_metadata_only(eng.step(s).to_dict())


def test_event_store_roundtrip_and_feedback():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store = EventStore(os.path.join(d, "dg.db"))
        eng = DriverGuardEngine(load_config(), driver_id="D001", trip_start_epoch_s=1_789_959_600, tz_offset_h=7)
        store.start_trip(eng.trip_id, eng.driver_id)
        ids = []
        for s in generate("distracted"):
            t = eng.step(s)
            if t.alert:
                ids.append(store.log_event(t.to_dict(), "alert"))
        assert len(ids) >= 2
        assert store.set_feedback(ids[0], "incorrect")
        ev = store.events(driver_id="D001")
        assert ev[-1]["feedback"] == "incorrect" and ev[0]["reasons"]
        assert store.events(driver_id="OTHER") == []
        store.end_trip(eng.trip_id, eng.memory.summary())
        assert store.trips()[0]["summary"]["minutes"] > 0
        store.close()


def test_password_hash():
    h = hash_password("s3cret", iterations=1000)
    assert verify_password("s3cret", h) and not verify_password("nope", h)


def test_tokens_and_roles():
    auth = AuthService([User("d1", "driver", "D001", password="a"), User("m", "fleet_manager", password="b")],
                       secret=b"k" * 32)
    assert auth.login("d1", "wrong") is None
    tok = auth.login("d1", "a")
    p = auth.verify(tok)
    assert p.role == "driver" and p.driver_id == "D001" and not p.is_manager
    assert auth.verify(tok[:-2] + "xx") is None  # tampered
    assert auth.verify(auth.login("m", "b")).is_manager
