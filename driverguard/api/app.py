"""FastAPI backend — METADATA ONLY.

There is deliberately no endpoint that returns or accepts images/video
(no /face.jpg, no /upload_video, no /cloud/frame-stream). The local camera preview lives in
the `driverguard run --show` OpenCV window on the device itself.

    GET  /health                      process / camera / FPS / latency (no auth)
    POST /auth/login                  -> token (roles: driver, fleet_manager)
    GET  /me
    GET  /state                       latest TickOutput
    WS   /ws/stream?token=...         realtime TickOutput stream (throttled)
    GET  /events                      alert/transition history (driver: own only)
    POST /events/{id}/feedback        driver says an alert was correct / incorrect
    GET  /trips                       trips (+ summaries)
    GET  /trips/current/summary       live trip-memory summary
    GET  /trips/{trip_id}/minutes     per-minute fatigue buckets
    GET  /config                      tunable thresholds (+ bounds)
    POST /config                      HITL threshold update (fleet_manager only, audited)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from ..config import REPO_ROOT, DriverGuardConfig, apply_updates, get_dotted, tunable_schema
from ..privacy import assert_metadata_only
from .auth import AuthService, Principal
from .hub import StateHub


@dataclass
class AppContext:
    cfg: DriverGuardConfig
    hub: StateHub
    auth: AuthService
    store: Any = None  # EventStore
    pipeline: Any = None  # DriverGuardPipeline (for live reconfigure + trip summary)


class LoginIn(BaseModel):
    username: str
    password: str


class FeedbackIn(BaseModel):
    feedback: str  # correct | incorrect


class ConfigIn(BaseModel):
    updates: dict[str, Any]


def create_app(ctx: AppContext) -> FastAPI:
    app = FastAPI(title="DriverGuard API", version="0.1.0",
                  description="Metadata-only driver monitoring API. No image endpoints by design.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    def principal(authorization: str | None = Header(default=None)) -> Principal:
        token = authorization.split(" ", 1)[1] if authorization and " " in authorization else authorization
        p = ctx.auth.verify(token)
        if p is None:
            raise HTTPException(401, "invalid or missing token")
        return p

    def manager(p: Principal = Depends(principal)) -> Principal:
        if not p.is_manager:
            raise HTTPException(403, "fleet_manager role required")
        return p

    def _safe(payload):
        assert_metadata_only(payload)
        return payload

    @app.get("/health")
    def health():
        return ctx.hub.health()

    @app.post("/auth/login")
    def login(body: LoginIn):
        token = ctx.auth.login(body.username, body.password)
        if token is None:
            raise HTTPException(401, "wrong username or password")
        p = ctx.auth.verify(token)
        return {"token": token, "role": p.role, "username": p.username, "driver_id": p.driver_id}

    @app.get("/me")
    def me(p: Principal = Depends(principal)):
        return {"username": p.username, "role": p.role, "driver_id": p.driver_id}

    @app.get("/state")
    def state(p: Principal = Depends(principal)):
        latest = ctx.hub.latest()
        if latest is None:
            raise HTTPException(503, "no data yet")
        if not p.is_manager and p.driver_id and latest.get("driver_id") != p.driver_id:
            raise HTTPException(403, "not your vehicle")
        return _safe(latest)

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket, token: str = Query(default="")):
        p = ctx.auth.verify(token)
        if p is None:
            await ws.close(code=4401)
            return
        await ws.accept()
        loop, q = ctx.hub.subscribe()
        try:
            latest = ctx.hub.latest()
            if latest:
                await ws.send_json(latest)
            while True:
                d = await q.get()
                if not p.is_manager and p.driver_id and d.get("driver_id") != p.driver_id:
                    continue
                await ws.send_json(d)
        except (WebSocketDisconnect, RuntimeError):  # client went away (detected on the next send)
            pass
        finally:
            ctx.hub.unsubscribe(loop, q)

    def _store():
        if ctx.store is None:
            raise HTTPException(503, "event store disabled")
        return ctx.store

    @app.get("/events")
    def events(trip_id: str | None = None, level: str | None = None, since_id: int = 0,
               limit: int = Query(200, le=1000), p: Principal = Depends(principal)):
        driver = None if p.is_manager else p.driver_id
        return _safe(_store().events(trip_id=trip_id, driver_id=driver, level=level, since_id=since_id,
                                     limit=limit))

    @app.post("/events/{event_id}/feedback")
    def feedback(event_id: int, body: FeedbackIn, p: Principal = Depends(principal)):
        try:
            ok = _store().set_feedback(event_id, body.feedback)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if not ok:
            raise HTTPException(404, "event not found")
        return {"ok": True}

    @app.get("/trips")
    def trips(p: Principal = Depends(principal)):
        return _safe(_store().trips(driver_id=None if p.is_manager else p.driver_id))

    @app.get("/trips/current/summary")
    def current_summary(p: Principal = Depends(principal)):
        if ctx.pipeline is None:
            raise HTTPException(503, "no live pipeline")
        eng = ctx.pipeline.engine
        if not p.is_manager and p.driver_id and eng.driver_id != p.driver_id:
            raise HTTPException(403, "not your trip")
        return _safe({"trip_id": eng.trip_id, "driver_id": eng.driver_id, **eng.memory.summary()})

    @app.get("/trips/{trip_id}/minutes")
    def trip_minutes(trip_id: str, p: Principal = Depends(principal)):
        return _safe(_store().minute_stats(trip_id))

    @app.get("/config")
    def get_config(p: Principal = Depends(principal)):
        return {"tunable": tunable_schema(ctx.cfg), "calibration_method": ctx.cfg.calibration.method}

    @app.post("/config")
    def post_config(body: ConfigIn, p: Principal = Depends(manager)):
        try:
            new_cfg = apply_updates(ctx.cfg, body.updates)
        except KeyError as e:
            raise HTTPException(403, str(e))
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False))
        old = {k: get_dotted(ctx.cfg, k) for k in body.updates}
        ctx.cfg = new_cfg
        if ctx.pipeline is not None:
            with ctx.pipeline.lock:
                ctx.pipeline.engine.reconfigure(new_cfg)
        if ctx.store is not None:
            ctx.store.audit_config(p.username, {"old": old, "new": body.updates})
        return {"ok": True, "tunable": tunable_schema(new_cfg)}

    dist = REPO_ROOT / "dashboard" / "dist"
    if Path(dist).is_dir():  # serve the built dashboard (npm run build) at /
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return app
