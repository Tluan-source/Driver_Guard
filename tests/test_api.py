"""API tests — skipped automatically if FastAPI is not installed (pip install -e ".[api,dev]")."""
import os
import tempfile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from driverguard.api import AuthService, StateHub, User  # noqa: E402
from driverguard.api.app import AppContext, create_app  # noqa: E402
from driverguard.config import load_config  # noqa: E402
from driverguard.engine import DriverGuardEngine  # noqa: E402
from driverguard.pipeline import DriverGuardPipeline, EventStoreSink  # noqa: E402
from driverguard.sim import generate  # noqa: E402
from driverguard.storage import EventStore  # noqa: E402


def _client(tmp):
    cfg = load_config()
    eng = DriverGuardEngine(cfg, driver_id="D001", trip_start_epoch_s=1_789_959_600, tz_offset_h=7)
    hub = StateHub(1000)
    store = EventStore(os.path.join(tmp, "dg.db"))
    pipe = DriverGuardPipeline(eng, sinks=[hub, EventStoreSink(store, eng)])
    pipe.run_signals(generate("distracted"))
    auth = AuthService([User("d1", "driver", "D001", password="a"), User("d2", "driver", "D002", password="c"),
                        User("m", "fleet_manager", password="b")], secret=b"x" * 32)
    ctx = AppContext(cfg=cfg, hub=hub, auth=auth, store=store, pipeline=pipe)
    return TestClient(create_app(ctx)), ctx


def _tok(c, u, p):
    return {"Authorization": "Bearer " + c.post("/auth/login", json={"username": u, "password": p}).json()["token"]}


def test_health_and_auth():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        c, _ = _client(tmp)
        assert c.get("/health").status_code == 200
        assert c.get("/state").status_code == 401
        assert c.post("/auth/login", json={"username": "d1", "password": "x"}).status_code == 401
        s = c.get("/state", headers=_tok(c, "d1", "a")).json()
        assert s["driver_id"] == "D001" and "risk_level" in s
        assert c.get("/state", headers=_tok(c, "d2", "c")).status_code == 403  # other driver's vehicle


def test_events_scoped_by_role_and_feedback():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        c, _ = _client(tmp)
        ev = c.get("/events", headers=_tok(c, "d1", "a")).json()
        assert ev and all(e["driver_id"] == "D001" for e in ev)
        assert c.get("/events", headers=_tok(c, "d2", "c")).json() == []
        r = c.post(f"/events/{ev[0]['id']}/feedback", json={"feedback": "correct"}, headers=_tok(c, "d1", "a"))
        assert r.status_code == 200


def test_config_hitl_manager_only():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        c, ctx = _client(tmp)
        body = {"updates": {"eye.long_closure_ms": 1200}}
        assert c.post("/config", json=body, headers=_tok(c, "d1", "a")).status_code == 403
        r = c.post("/config", json=body, headers=_tok(c, "m", "b"))
        assert r.status_code == 200
        assert ctx.pipeline.engine.cfg.eye.long_closure_ms == 1200
        assert c.post("/config", json={"updates": {"risk.weights": {}}}, headers=_tok(c, "m", "b")).status_code == 403
        assert c.post("/config", json={"updates": {"eye.microsleep_ms": 100}},
                      headers=_tok(c, "m", "b")).status_code == 422


def test_no_image_endpoints_exist():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        c, _ = _client(tmp)
        paths = {r.path for r in c.app.routes}
        assert not any(k in p for p in paths for k in ("jpg", "image", "frame", "video", "upload"))


def test_websocket_stream():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        c, _ = _client(tmp)
        tok = _tok(c, "m", "b")["Authorization"].split()[1]
        with c.websocket_connect(f"/ws/stream?token={tok}") as ws:
            d = ws.receive_json()
            assert "risk_level" in d
