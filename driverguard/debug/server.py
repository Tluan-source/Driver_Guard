"""Loopback-only debug server for ui/test-console.

    driverguard debug-ui            -> http://127.0.0.1:8010

    GET  /dbg/health                 liveness
    WS   /dbg/ws                     one DebugSession per connection
         client -> server  text  {"cmd": "start", "source": "webcam"|"synthetic", "scenario": "drowsy", "speed": 4,
                                  "calibration": "adaptive"|"fixed", "fast_warmup": true}
                           text  {"cmd": "reset"} | {"cmd": "pause"} | {"cmd": "resume"} | {"cmd": "stop"}
                           text  {"cmd": "model", "id": "<checkpoint path>" | null}  attach a learned head
                           bytes one JPEG frame from the browser webcam (webcam source)
         server -> client  {"type": "info", ...} | {"type": "frame", ...} | {"type": "error", "message"}
    GET  /dbg/models                 Model Lab: camera checkpoints + validation metrics
    GET  /dbg/clips                  Model Lab: train/validation feature clips (test is never listed)
    POST /dbg/replay                 Model Lab: {"model", "feature_set", "clip", "stride"} -> timeline
    GET  /                           built console (ui/test-console/dist), if present

Frames arrive from the browser on the same machine, are decoded in RAM, processed and dropped;
nothing is written to disk. Built on Starlette (already installed with FastAPI) so the debug tool
adds no dependency to the production API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..config import REPO_ROOT
from . import model_lab
from .session import DebugSession

UI_DIST = REPO_ROOT / "ui" / "test-console" / "dist"
UI_SOURCE = REPO_ROOT / "ui" / "test-console"
SCENARIOS = ("normal", "drowsy", "distracted", "degraded", "narrow_eyes", "demo")


class _Conn:
    def __init__(self, ws: WebSocket, config_path, overrides):
        self.ws = ws
        self.session = DebugSession(config_path, overrides)
        self.source = "idle"
        self.synth_task: asyncio.Task | None = None
        self.paused = False
        self.clock0 = time.monotonic()
        self.busy = False

    def now_ms(self) -> int:
        return int((time.monotonic() - self.clock0) * 1000)

    async def send(self, msg: dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def stop_synth(self) -> None:
        if self.synth_task is not None:
            self.synth_task.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await self.synth_task
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning("Synthetic session stopped after error: %s", exc)
            self.synth_task = None

    async def run_synth(self, scenario: str, speed: float = 1.0) -> None:
        """Replays a synthetic scenario. speed > 1 = fast-forward (same ticks, less wall time)."""
        from ..sim import loop

        fps = self.session.cfg.runtime.target_fps
        period = 1.0 / (fps * speed)
        nxt = time.monotonic()
        for sig in loop(scenario, fps=fps):
            while self.paused:
                await asyncio.sleep(0.1)
                nxt = time.monotonic()
            frame = self.session.process_signals(sig, f"synthetic:{scenario}")
            await self.send(frame)
            nxt += period
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    async def handle_cmd(self, msg: dict[str, Any]) -> None:
        cmd = msg.get("cmd")
        if cmd in ("hello", "info"):
            await self.send(self.session.info())
        elif cmd == "start":
            await self.stop_synth()
            src = msg.get("source", "webcam")
            self.session.reset(msg.get("calibration"), msg.get("fast_warmup"))
            self.paused = False
            if src == "synthetic":
                scenario = msg.get("scenario", "demo")
                if scenario not in SCENARIOS:
                    await self.send({"type": "error", "message": f"Kịch bản không hợp lệ: {scenario}"})
                    return
                self.source = f"synthetic:{scenario}"
                await self.send(self.session.info())
                speed = float(msg.get("speed", 1.0))
                speed = min(max(speed, 0.25), 16.0)
                self.synth_task = asyncio.create_task(self.run_synth(scenario, speed))
            else:
                self.source = "webcam"
                err = await run_in_threadpool(self.session.ensure_perception)
                info = self.session.info()
                await self.send(info)
                if err:
                    await self.send({"type": "error", "message": err})
        elif cmd == "model":
            model_id = msg.get("id")
            err = await run_in_threadpool(self.session.set_model, model_id)
            await self.send(self.session.info())
            if err:
                await self.send({"type": "error", "message": err})
        elif cmd == "reset":
            self.session.reset()
            await self.send(self.session.info())
        elif cmd == "pause":
            self.paused = True
        elif cmd == "resume":
            self.paused = False
        elif cmd == "stop":
            await self.stop_synth()
            self.source = "idle"
        else:
            await self.send({"type": "error", "message": f"Lệnh không hỗ trợ: {cmd}"})

    async def handle_frame(self, data: bytes) -> None:
        if self.source != "webcam":
            return
        if self.paused:
            await self.send({"type": "skip"})
            return
        frame = await run_in_threadpool(self.session.process_jpeg, data, self.now_ms())
        await self.send(frame)


def create_debug_app(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Starlette:
    ui = UI_DIST if (UI_DIST / "index.html").is_file() else UI_SOURCE

    async def health(_request):
        return JSONResponse({"status": "ok", "ui_available": (ui / "index.html").is_file(),
                             "ui_built": (UI_DIST / "index.html").is_file()})

    async def ws_endpoint(ws: WebSocket):
        host = ws.client.host if ws.client else ""
        if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            await ws.close(code=4403)
            return
        await ws.accept()
        conn = _Conn(ws, config_path, overrides)
        try:
            await conn.send(conn.session.info())
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await conn.handle_frame(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        await conn.handle_cmd(json.loads(msg["text"]))
                    except ValueError as e:
                        await conn.send({"type": "error", "message": str(e)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await conn.stop_synth()
            conn.session.close()

    def _loopback(request: Request) -> bool:
        return (request.client.host if request.client else "") in ("127.0.0.1", "::1", "localhost", "testclient")

    async def models(request: Request):
        if not _loopback(request):
            return JSONResponse({"error": "loopback only"}, status_code=403)
        return JSONResponse({"models": await run_in_threadpool(model_lab.list_models)})

    async def clips(request: Request):
        if not _loopback(request):
            return JSONResponse({"error": "loopback only"}, status_code=403)
        return JSONResponse(await run_in_threadpool(model_lab.list_clips))

    async def replay(request: Request):
        if not _loopback(request):
            return JSONResponse({"error": "loopback only"}, status_code=403)
        try:
            body = await request.json()
            result = await run_in_threadpool(model_lab.replay, str(body["model"]), str(body["feature_set"]),
                                             str(body["clip"]), int(body.get("stride", model_lab.DEFAULT_STRIDE)))
        except (KeyError, TypeError, ValueError, FileNotFoundError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(result)

    routes = [Route("/dbg/health", health), WebSocketRoute("/dbg/ws", ws_endpoint),
              Route("/dbg/models", models), Route("/dbg/clips", clips),
              Route("/dbg/replay", replay, methods=["POST"])]
    app = Starlette(routes=routes)
    if (ui / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui, html=True), name="console")
    return app
