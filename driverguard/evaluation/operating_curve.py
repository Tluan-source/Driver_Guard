"""Operating-point curves instead of hard pass/fail gates (GVHD review 8.3).

Report event recall vs false alerts per hour while sweeping one threshold, choose the operating
point on the VALIDATION subjects, then run the test subjects ONCE with that frozen config.

    curve = sweep(val_videos, val_gt, "eye.long_closure_ms", [600, 800, 1000, 1200, 1500], cfg)
    op = pick_operating_point(curve, max_fa_per_hour=1.0)
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..config import DriverGuardConfig, apply_updates
from ..engine import DriverGuardEngine
from ..schemas import FrameSignals
from .events import Interval, intervals_from_ticks, level_at_least, match_events

_FIXED_EPOCH = 1_789_959_600.0  # 2026-09-21 10:00 UTC+7 — deterministic daytime clock for replays
_TZ = 7


def replay(signals: Iterable[FrameSignals], cfg: DriverGuardConfig, trip_start_epoch_s: float = _FIXED_EPOCH,
           tz_offset_h: float = _TZ) -> list[dict[str, Any]]:
    eng = DriverGuardEngine(cfg, trip_start_epoch_s=trip_start_epoch_s, tz_offset_h=tz_offset_h)
    return [eng.step(s).to_dict() for s in signals]


def evaluate(videos: dict[str, list[FrameSignals]], gt: dict[str, list[Interval]], cfg: DriverGuardConfig,
             level: str = "warning", gt_labels: set[str] | None = None, tolerance_ms: int = 500) -> dict[str, Any]:
    tp = fp = fn = 0
    total_ms = 0.0
    delays: list[int] = []
    for vid, sigs in videos.items():
        ticks = replay(sigs, cfg)
        if not ticks:
            continue
        total_ms += ticks[-1]["timestamp_ms"] - ticks[0]["timestamp_ms"]
        pred = intervals_from_ticks(ticks, level_at_least(level), merge_gap_ms=1000)
        g = [iv for iv in gt.get(vid, []) if gt_labels is None or iv.label in gt_labels]
        m = match_events(g, pred, tolerance_ms)
        tp, fp, fn = tp + m.tp, fp + m.fp, fn + m.fn
        delays += m.onset_delays_ms
    hours = total_ms / 3.6e6
    return {
        "tp": tp, "fp": fp, "fn": fn, "hours": round(hours, 3),
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_alerts_per_hour": fp / hours if hours > 0 else None,
        "onset_p50_ms": float(np.percentile(delays, 50)) if delays else None,
        "onset_p95_ms": float(np.percentile(delays, 95)) if delays else None,
    }


def sweep(videos, gt, key: str, values: Iterable[Any], base_cfg: DriverGuardConfig, **kw) -> list[dict[str, Any]]:
    rows = []
    for v in values:
        cfg = apply_updates(base_cfg, {key: v}, allowed=None)
        rows.append({"key": key, "value": v, **evaluate(videos, gt, cfg, **kw)})
    return rows


def pick_operating_point(curve: list[dict[str, Any]], max_fa_per_hour: float = 1.0) -> dict[str, Any] | None:
    ok = [r for r in curve if r["false_alerts_per_hour"] is not None and r["false_alerts_per_hour"] <= max_fa_per_hour
          and r["recall"] is not None]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["recall"], -r["false_alerts_per_hour"]))
