"""Runtime benchmark: processed FPS, per-frame latency p50/p95, and FPS per minute over a long run
(to expose thermal throttling). Run it ON THE TARGET DEVICE — laptop numbers are not acceptance.

    python -m driverguard.evaluation.runtime_bench --source 0 --seconds 1800 --out runs/bench_laptop.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def main() -> None:
    from ..capture import open_source
    from ..config import load_config
    from ..engine import DriverGuardEngine
    from ..perception import PerceptionExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--out", default="runs/runtime_bench.json")
    args = ap.parse_args()

    cfg = load_config()
    ext = PerceptionExtractor.from_config(cfg)
    eng = DriverGuardEngine(cfg)
    src = open_source(args.source, realtime=not args.source.isdigit())
    lat_p, lat_e, per_min = [], [], {}
    t0 = time.monotonic()
    n = 0
    for fr in src.frames():
        a = time.perf_counter()
        sig = ext.process(fr.image_bgr, fr.ts_ms, fr.camera_ok)
        b = time.perf_counter()
        eng.step(sig)
        c = time.perf_counter()
        lat_p.append((b - a) * 1000)
        lat_e.append((c - b) * 1000)
        n += 1
        minute = int((time.monotonic() - t0) // 60)
        per_min[minute] = per_min.get(minute, 0) + 1
        if time.monotonic() - t0 >= args.seconds:
            break
    src.close()
    ext.close()
    el = time.monotonic() - t0
    rep = {
        "host": platform.platform(), "processor": platform.processor(), "seconds": round(el, 1), "frames": n,
        "processed_fps": round(n / el, 2) if el else None,
        "perception_ms_p50": float(np.percentile(lat_p, 50)), "perception_ms_p95": float(np.percentile(lat_p, 95)),
        "engine_ms_p50": float(np.percentile(lat_e, 50)), "engine_ms_p95": float(np.percentile(lat_e, 95)),
        "fps_per_minute": {k: round(v / 60, 2) for k, v in sorted(per_min.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
