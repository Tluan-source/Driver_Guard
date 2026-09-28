"""Simulated vehicle speed over a fake CAN bus (đề bài: "mô phỏng cảm biến tốc độ qua CAN giả lập").

The simulator encodes speed into a CAN-like frame and decodes it again, so the rest of the
system consumes speed exactly as it would from a real OBD-II/CAN adapter later. Swapping in a
real bus = implement `SpeedSource.read_speed(ts_ms)` with python-can.

Frame layout (project-defined, NOT a real OEM DBC):
    CAN ID 0x3E9, byte0..1 = speed * 100 (uint16 big-endian, km/h), bytes 2..7 reserved.
"""
from __future__ import annotations

import csv
import math
from bisect import bisect_right
from pathlib import Path
from typing import Protocol

SPEED_CAN_ID = 0x3E9


def encode_speed_frame(speed_kmh: float) -> tuple[int, bytes]:
    raw = max(0, min(65535, int(round(speed_kmh * 100))))
    return SPEED_CAN_ID, raw.to_bytes(2, "big") + bytes(6)


def decode_speed_frame(can_id: int, data: bytes) -> float | None:
    if can_id != SPEED_CAN_ID or len(data) < 2:
        return None
    return int.from_bytes(data[:2], "big") / 100.0


class SpeedSource(Protocol):
    def read_speed(self, ts_ms: int) -> float | None: ...


class SimulatedCanBus:
    """Deterministic speed profiles (same ts -> same speed) so replays are reproducible.

    profiles: mixed (urban + red light + highway, 10-min cycle), highway, urban, parked, none.
    """

    def __init__(self, profile: str = "mixed"):
        if profile not in {"mixed", "highway", "urban", "parked", "none"}:
            raise ValueError(f"unknown CAN profile {profile}")
        self.profile = profile

    def speed_at(self, ts_ms: int) -> float | None:
        t = ts_ms / 1000.0
        wobble = 3.0 * math.sin(t / 7.0) + 1.5 * math.sin(t / 2.3)
        if self.profile == "none":
            return None
        if self.profile == "parked":
            return 0.0
        if self.profile == "highway":
            return max(0.0, 90.0 + wobble)
        if self.profile == "urban":
            return max(0.0, 45.0 + 2 * wobble)
        c = t % 600.0  # mixed
        if c < 20:
            return 0.0
        if c < 60:
            return (c - 20) / 40.0 * 50.0
        if c < 300:
            return max(0.0, 48.0 + 2 * wobble)
        if c < 330:
            return 0.0  # red light
        if c < 360:
            return (c - 330) / 30.0 * 90.0
        return max(0.0, 90.0 + wobble)

    def read_speed(self, ts_ms: int) -> float | None:
        v = self.speed_at(ts_ms)
        if v is None:
            return None
        return decode_speed_frame(*encode_speed_frame(v))


class CsvSpeedSource:
    """Replay a recorded speed log: CSV with columns ts_ms,speed_kmh (step interpolation)."""

    def __init__(self, path: str | Path):
        self.ts: list[int] = []
        self.v: list[float] = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.ts.append(int(float(row["ts_ms"])))
                self.v.append(float(row["speed_kmh"]))

    def read_speed(self, ts_ms: int) -> float | None:
        i = bisect_right(self.ts, ts_ms) - 1
        return self.v[i] if i >= 0 else None
