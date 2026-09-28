"""Synthetic FrameSignals scenarios.

Purpose: develop and test the temporal/risk/dashboard layers WITHOUT a camera, model files or
datasets, and generate golden test vectors for the Android port.

IMPORTANT: synthetic results are sanity checks of the logic, NOT evidence of real-world
performance. Never report them as experimental results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..schemas import FrameSignals
from ..vehicle.can_sim import SimulatedCanBus


@dataclass
class Ev:
    kind: str  # long_closure | drowsy | yawn | talk | look_down | mirror | phone | phone_mount |
    #            face_lost | camera_lost | parked | nod | low_quality
    t0: float  # seconds
    dur: float  # seconds


SCENARIOS: dict[str, dict] = {
    "normal": dict(duration=180, events=[
        Ev("mirror", 40, 0.8), Ev("talk", 50, 0.4), Ev("talk", 51, 0.3), Ev("talk", 52.5, 0.5),
        Ev("phone_mount", 100, 12), Ev("mirror", 130, 1.0), Ev("face_lost", 150, 0.25)]),
    "drowsy": dict(duration=260, events=[
        Ev("drowsy", 90, 170), Ev("yawn", 100, 4.5), Ev("yawn", 130, 5), Ev("yawn", 170, 4),
        Ev("nod", 185, 0.6), Ev("long_closure", 200, 1.3), Ev("long_closure", 230, 2.6)]),
    "distracted": dict(duration=170, events=[
        Ev("look_down", 100, 4), Ev("mirror", 110, 0.8), Ev("phone", 120, 6), Ev("phone_mount", 140, 10),
        Ev("parked", 155, 15), Ev("phone", 158, 8)]),
    "degraded": dict(duration=140, events=[
        Ev("face_lost", 80, 1.0), Ev("face_lost", 90, 8), Ev("camera_lost", 110, 5), Ev("low_quality", 125, 6)]),
    "narrow_eyes": dict(duration=180, open_ear=0.21, events=[Ev("mirror", 60, 0.8)]),
    "demo": dict(duration=420, can="highway", events=[
        Ev("mirror", 70, 0.8), Ev("yawn", 90, 4.5), Ev("yawn", 110, 5),
        Ev("look_down", 130, 4), Ev("phone", 150, 6), Ev("phone_mount", 165, 8),
        Ev("drowsy", 190, 120), Ev("nod", 230, 0.6), Ev("long_closure", 250, 1.3),
        Ev("long_closure", 280, 2.6), Ev("face_lost", 330, 8), Ev("parked", 360, 30)]),
}


def _active(events: list[Ev], kind: str, t: float) -> Ev | None:
    for e in events:
        if e.kind == kind and e.t0 <= t < e.t0 + e.dur:
            return e
    return None


def generate(name: str, fps: float = 15.0, seed: int = 0, t_offset_ms: int = 0,
             open_ear: float | None = None, can_profile: str | None = None) -> Iterator[FrameSignals]:
    spec = SCENARIOS[name]
    events: list[Ev] = spec["events"]
    duration = spec["duration"]
    ear_open = open_ear if open_ear is not None else spec.get("open_ear", 0.30)
    bus = SimulatedCanBus(can_profile or spec.get("can", "highway"))
    rng = np.random.default_rng(seed)

    # blink schedule (ms): drowsy phase -> longer, more frequent blinks
    blinks: list[tuple[float, float]] = []
    t = rng.uniform(1000, 3000)
    while t < duration * 1000:
        if _active(events, "drowsy", t / 1000):
            dur, gap = rng.uniform(260, 480), rng.uniform(1400, 2800)
        else:
            dur, gap = rng.uniform(110, 200), rng.uniform(2500, 5000)
        blinks.append((t, dur))
        t += dur + gap
    bi = 0

    n = int(duration * fps)
    for i in range(n):
        ts = int(round(i * 1000.0 / fps))
        tsec = ts / 1000.0
        while bi < len(blinks) and blinks[bi][0] + blinks[bi][1] < ts:
            bi += 1
        in_blink = bi < len(blinks) and blinks[bi][0] <= ts < blinks[bi][0] + blinks[bi][1]
        closed = in_blink or _active(events, "long_closure", tsec) is not None
        droop = 0.86 if _active(events, "drowsy", tsec) else 1.0
        level = ear_open * droop
        ear_l = (0.30 * ear_open if closed else level) + rng.normal(0, 0.010)
        ear_r = (0.30 * ear_open if closed else level * 0.97) + rng.normal(0, 0.010)
        blink_bs = float(np.clip(1.0 - (ear_l + ear_r) / (2 * ear_open), 0, 1))

        mar = 0.08 + abs(rng.normal(0, 0.02))
        if _active(events, "talk", tsec):
            mar = 0.62
        if _active(events, "yawn", tsec):
            mar = 0.85

        yaw = 8.0 + rng.normal(0, 1.5)  # camera mounted slightly off-axis
        pitch = -6.0 + rng.normal(0, 1.2)
        if _active(events, "mirror", tsec):
            yaw += 45
        if _active(events, "look_down", tsec):
            pitch -= 34
        if _active(events, "nod", tsec):
            pitch -= 22

        phone_conf, near = 0.02, False
        if _active(events, "phone", tsec):
            phone_conf, near = 0.72, True
        elif _active(events, "phone_mount", tsec):
            phone_conf, near = 0.80, False

        speed = bus.read_speed(t_offset_ms + ts)
        if _active(events, "parked", tsec):
            speed = 0.0

        camera_ok = _active(events, "camera_lost", tsec) is None
        face = camera_ok and _active(events, "face_lost", tsec) is None
        quality = 0.25 if _active(events, "low_quality", tsec) else 0.9

        if not camera_ok:
            yield FrameSignals(ts_ms=t_offset_ms + ts, camera_ok=False, speed_kmh=speed)
            continue
        if not face:
            yield FrameSignals(ts_ms=t_offset_ms + ts, camera_ok=True, face_detected=False,
                               phone_conf=phone_conf, phone_near_face=False, speed_kmh=speed)
            continue
        yield FrameSignals(
            ts_ms=t_offset_ms + ts, camera_ok=True, face_detected=True, face_quality=quality,
            face_x0=0.35, face_y0=0.25, face_x1=0.65, face_y1=0.70,
            ear_left=float(ear_l), ear_right=float(ear_r), blink_bs_left=blink_bs, blink_bs_right=blink_bs,
            mar=float(mar), jaw_open_bs=float(min(1.0, mar)), yaw=float(yaw), pitch=float(pitch),
            roll=float(rng.normal(0, 1.0)), brightness=120.0, blur=80.0,
            phone_conf=phone_conf, phone_near_face=near, speed_kmh=speed,
        )


def loop(name: str, fps: float = 15.0) -> Iterator[FrameSignals]:
    """Endless stream (for dashboard development): repeats the scenario with increasing ts."""
    offset = 0
    k = 0
    while True:
        last = offset
        for s in generate(name, fps=fps, seed=k, t_offset_ms=offset):
            last = s.ts_ms
            yield s
        offset = last + int(1000 / fps)
        k += 1
