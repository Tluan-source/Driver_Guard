"""Interval labels for the self-recorded "golden" set (GVHD 6.3).

Label EVENT INTERVALS (start/end), not individual frames; derive frame labels by script.
Two annotators label the same 15% of videos independently -> report Cohen's kappa.

CSV format (one row per event):
    video_id,start_ms,end_ms,label,annotator
    S03_night_01,12500,14100,eye_closure,A
    S03_night_01,40200,45800,yawn,A

Suggested labels: eye_closure (>= 500 ms), microsleep, yawn, talking, look_down, mirror_check,
phone_hand, phone_mount, occlusion, face_out_of_frame, camera_covered.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .events import Interval


def load_intervals(path: str | Path, annotator: str | None = None) -> dict[str, list[Interval]]:
    out: dict[str, list[Interval]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if annotator and r.get("annotator") != annotator:
                continue
            out[r["video_id"]].append(Interval(int(float(r["start_ms"])), int(float(r["end_ms"])), r["label"]))
    return dict(out)


def frame_labels(intervals: list[Interval], label: str, duration_ms: int, step_ms: int = 100) -> np.ndarray:
    grid = np.zeros(int(np.ceil(duration_ms / step_ms)), dtype=bool)
    for iv in intervals:
        if iv.label != label:
            continue
        grid[int(iv.start_ms // step_ms): int(np.ceil(iv.end_ms / step_ms))] = True
    return grid


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
