"""DriverGuard command line.

    driverguard simulate --scenario drowsy                 # no camera / model needed
    driverguard run --source 0 --show                      # webcam + local preview window
    driverguard run --source video.mp4 --jsonl runs/x.jsonl
    driverguard serve --source synthetic:demo              # API + WebSocket for the dashboard
    driverguard serve --source 0
    driverguard extract --video v.mp4 --out data/features/v.parquet   # perception only
    driverguard replay --features data/features/v.parquet --jsonl runs/v_ticks.jsonl
"""
from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from datetime import datetime

from .config import load_config, resolve_path
from .engine import DriverGuardEngine
from .pipeline import DriverGuardPipeline, EventStoreSink, JsonlSink

LEVEL_COLORS = {  # BGR for the local preview
    "normal": (80, 180, 60), "caution": (0, 200, 255), "warning": (0, 120, 255),
    "critical": (0, 0, 230), "sensor_degraded": (160, 160, 160),
}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. eye.microsleep_ms=1800")
    p.add_argument("--driver-id", default="D001", help="pseudonymous driver id (never a real name)")
    p.add_argument("--can", default="highway", help="simulated CAN speed profile: mixed|highway|urban|parked|none")
    p.add_argument("--trip-start", default=None, help="ISO datetime of trip start, e.g. 2026-09-28T02:30 (circadian tests)")
    p.add_argument("--tz-offset", type=float, default=None, help="UTC offset in hours for the trip clock (default: local)")


def _cfg(args):
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        overrides[k] = v
    return load_config(args.config, overrides)


def _engine(args, cfg) -> DriverGuardEngine:
    start = datetime.fromisoformat(args.trip_start).timestamp() if args.trip_start else None
    return DriverGuardEngine(cfg, driver_id=args.driver_id, trip_start_epoch_s=start, tz_offset_h=args.tz_offset)


def _speed(args):
    from .vehicle import SimulatedCanBus

    return None if args.can == "none" else SimulatedCanBus(args.can)


def _summary(levels: collections.Counter, alerts: list, engine: DriverGuardEngine, n: int, secs: float) -> None:
    print("\n=== Summary ===")
    print(f"ticks: {n}  wall: {secs:.1f}s  ({n / secs if secs else 0:.1f} ticks/s)")
    for lvl in ("normal", "caution", "warning", "critical", "sensor_degraded"):
        if levels[lvl]:
            print(f"  {lvl:16s} {levels[lvl]:7d} ticks")
    print(f"alerts: {len(alerts)}")
    for a in alerts[:30]:
        print(f"  {a.ts_ms / 1000:8.1f}s  {a.level.value:15s} {','.join(a.reasons)}")
    s = engine.memory.summary()
    print(f"trip: {s['minutes']} min, degraded_ratio={s['degraded_ratio']}, fatigue_index={s['fatigue_index_last']}")


# ---------------------------------------------------------------------------- commands
def cmd_simulate(args) -> int:
    from .sim import generate

    cfg = _cfg(args)
    eng = _engine(args, cfg)
    sinks = [JsonlSink(args.jsonl)] if args.jsonl else []
    pipe = DriverGuardPipeline(eng, sinks=sinks)
    levels, alerts = collections.Counter(), []

    def on_tick(t, _):
        levels[t.risk_level.value] += 1
        if t.alert:
            alerts.append(t.alert)

    t0 = time.time()
    pipe.run_signals(generate(args.scenario, fps=args.fps, open_ear=args.open_ear), on_tick=on_tick)
    pipe.close()
    _summary(levels, alerts, eng, pipe.n_ticks, time.time() - t0)
    return 0


def cmd_run(args) -> int:
    import cv2

    from .alerts import ConsoleAlertSink, SoundAlertSink
    from .capture import open_source
    from .perception import PerceptionExtractor

    cfg = _cfg(args)
    eng = _engine(args, cfg)
    sinks = [JsonlSink(args.jsonl)] if args.jsonl else []
    if args.db:
        from .storage import EventStore

        sinks.append(EventStoreSink(EventStore(resolve_path(cfg.storage.db_path)), eng))
    alert_sinks = [ConsoleAlertSink(), SoundAlertSink(enabled=not args.mute)]
    pipe = DriverGuardPipeline(eng, PerceptionExtractor.from_config(cfg), _speed(args), sinks, alert_sinks)
    src = open_source(args.source, target_fps=args.fps, realtime=args.realtime)
    levels, alerts = collections.Counter(), []
    t0 = time.time()

    def on_tick(t, fr):
        levels[t.risk_level.value] += 1
        if t.alert:
            alerts.append(t.alert)
        if not args.show:
            if pipe.n_ticks % 15 == 0:
                sys.stdout.write(f"\r{t.timestamp_ms / 1000:7.1f}s {t.risk_level.value:15s} fps={t.processing_fps:5.1f} "
                                 f"ear_n={t.ear_norm or 0:.2f} perclos={t.perclos_proxy or 0:.2f} "
                                 f"score={t.risk_score:5.1f} {','.join(t.reason_codes[:3]):40s}")
                sys.stdout.flush()
            return
        img = fr.image_bgr
        if img is None:
            return
        # LOCAL preview only: the frame never leaves this process.
        h, w = img.shape[:2]
        color = LEVEL_COLORS.get(t.risk_level.value, (255, 255, 255))
        cv2.rectangle(img, (0, 0), (w, 34), color, -1)
        cv2.putText(img, f"{t.risk_level.value.upper()}  score {t.risk_score:.0f}  {','.join(t.reason_codes[:2])}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        lines = [f"FPS {t.processing_fps:.1f}  lat {t.latency_ms:.0f}ms",
                 f"EAR_n {t.ear_norm or 0:.2f}  closed {t.eye_closed}  {t.closure_ms}ms",
                 f"PERCLOS-proxy {t.perclos_proxy if t.perclos_proxy is not None else '-'}",
                 f"yaw {t.head_yaw_deg or 0:.0f}  pitch {t.head_pitch_deg or 0:.0f}  phone {t.phone_persistence:.2f}",
                 f"calibrated {t.calibrated}  speed {t.speed_kmh or 0:.0f}"]
        for i, s in enumerate(lines):
            cv2.putText(img, s, (8, 58 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("DriverGuard (local preview)", img)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            pipe.stop_event.set()

    try:
        pipe.run_frames(src, on_tick=on_tick, max_frames=args.max_frames)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        pipe.close()
        if args.show:
            cv2.destroyAllWindows()
    _summary(levels, alerts, eng, pipe.n_ticks, time.time() - t0)
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn

        from .api.app import AppContext, create_app
    except ImportError:
        print('FastAPI/uvicorn missing. Install:  pip install -e ".[api]"')
        return 2
    from .api import AuthService, StateHub
    from .storage import EventStore

    cfg = _cfg(args)
    eng = _engine(args, cfg)
    hub = StateHub(cfg.api.stream_hz)
    store = EventStore(resolve_path(cfg.storage.db_path))
    store.purge_older_than(cfg.storage.retention_days)
    users_file = resolve_path("configs/users.yaml")
    if not users_file.exists():
        users_file = resolve_path(cfg.api.users_file)
        print(f"[auth] using demo accounts from {users_file} — create configs/users.yaml for real use")
    auth = AuthService.from_file(users_file, ttl_s=cfg.api.token_ttl_s)

    sinks = [hub, EventStoreSink(store, eng)]
    if args.source.startswith("synthetic:"):
        from .sim import loop

        pipe = DriverGuardPipeline(eng, speed_source=None, sinks=sinks)
        target = lambda: pipe.run_signals(loop(args.source.split(":", 1)[1]), realtime=True)  # noqa: E731
    else:
        from .alerts import SoundAlertSink
        from .capture import open_source
        from .perception import PerceptionExtractor

        pipe = DriverGuardPipeline(eng, PerceptionExtractor.from_config(cfg), _speed(args), sinks,
                                   [SoundAlertSink(enabled=not args.mute)])
        src = open_source(args.source, target_fps=args.fps, realtime=True)
        target = lambda: pipe.run_frames(src)  # noqa: E731

    worker = threading.Thread(target=target, name="driverguard-pipeline", daemon=True)
    worker.start()
    app = create_app(AppContext(cfg=cfg, hub=hub, auth=auth, store=store, pipeline=pipe))
    print(f"DriverGuard API on http://{args.host}:{args.port}  (docs: /docs)  source={args.source}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        pipe.stop_event.set()
        worker.join(timeout=2)
        pipe.close()
    return 0


def cmd_extract(args) -> int:
    from .capture import VideoFileSource
    from .io import write_features
    from .perception import PerceptionExtractor

    cfg = _cfg(args)
    ext = PerceptionExtractor.from_config(cfg)
    src = VideoFileSource(args.video, target_fps=args.fps)
    rows = []
    t0 = time.time()
    for fr in src.frames():
        rows.append(ext.process(fr.image_bgr, fr.ts_ms, fr.camera_ok))
        if len(rows) % 300 == 0:
            print(f"\r{len(rows)} frames  {len(rows) / (time.time() - t0):.1f} fps", end="", flush=True)
    src.close()
    ext.close()
    out = write_features(rows, args.out, meta={"video": str(args.video), "fps": args.fps, "src_fps": src.fps,
                                               "n": len(rows)})
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


def cmd_replay(args) -> int:
    from .io import read_features

    cfg = _cfg(args)
    eng = _engine(args, cfg)
    sinks = [JsonlSink(args.jsonl)] if args.jsonl else []
    pipe = DriverGuardPipeline(eng, speed_source=_speed(args), sinks=sinks)
    levels, alerts = collections.Counter(), []

    def on_tick(t, _):
        levels[t.risk_level.value] += 1
        if t.alert:
            alerts.append(t.alert)

    t0 = time.time()
    pipe.run_signals(read_features(args.features), on_tick=on_tick)
    pipe.close()
    _summary(levels, alerts, eng, pipe.n_ticks, time.time() - t0)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Vietnamese alert text on Windows consoles / redirected output
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="driverguard", description="DriverGuard DMS (metadata-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="run the engine on a synthetic scenario (no camera/model)")
    _common(p)
    p.add_argument("--scenario", default="drowsy", help="normal|drowsy|distracted|degraded|narrow_eyes|demo")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--open-ear", type=float, default=None, help="override driver's open-eye EAR")
    p.add_argument("--jsonl", default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("run", help="webcam or video file, local alerts (+ optional preview)")
    _common(p)
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--fps", type=float, default=None, help="subsample video files to this fps")
    p.add_argument("--realtime", action="store_true", help="pace video files at real time")
    p.add_argument("--show", action="store_true", help="local OpenCV preview window")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--db", action="store_true", help="log events to SQLite")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="pipeline + FastAPI/WebSocket for the dashboard")
    _common(p)
    p.add_argument("--source", default="synthetic:demo", help="synthetic:<scenario> | camera index | video path")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mute", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("extract", help="video -> per-frame FrameSignals (parquet/csv)")
    _common(p)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help=".parquet (needs pyarrow) or .csv/.csv.gz")
    p.add_argument("--fps", type=float, default=15.0)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("replay", help="run the engine on extracted features")
    _common(p)
    p.add_argument("--features", required=True)
    p.add_argument("--jsonl", default=None)
    p.set_defaults(func=cmd_replay)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
