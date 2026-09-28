import os
import tempfile

import numpy as np

from driverguard.config import load_config
from driverguard.evaluation.events import Interval, false_alerts_per_hour, intervals_from_ticks, level_at_least, match_events
from driverguard.evaluation.labels import cohen_kappa, frame_labels, load_intervals
from driverguard.evaluation.operating_curve import evaluate, pick_operating_point, sweep
from driverguard.evaluation.splits import assert_disjoint, driver_split, group_kfold, leave_one_subject_out
from driverguard.evaluation.uta_rldd import parse_path
from driverguard.sim import generate
from pathlib import Path


def test_event_matching():
    gt = [Interval(1000, 2000), Interval(5000, 6000), Interval(9000, 9500)]
    pred = [Interval(1300, 2100), Interval(7000, 7200), Interval(9400, 9600)]
    m = match_events(gt, pred, tolerance_ms=200)
    assert (m.tp, m.fp, m.fn) == (2, 1, 1)
    assert m.onset_delays_ms == [300, 400]
    assert abs(m.recall - 2 / 3) < 1e-9 and abs(m.precision - 2 / 3) < 1e-9
    assert false_alerts_per_hour(2, 30 * 60_000) == 4.0


def test_intervals_from_ticks():
    ticks = [{"timestamp_ms": i * 100, "risk_level": "warning" if 10 <= i < 20 or 22 <= i < 25 else "normal"}
             for i in range(40)]
    iv = intervals_from_ticks(ticks, level_at_least("warning"), merge_gap_ms=500)
    assert iv == [Interval(1000, 2400)]


def test_splits_are_subject_disjoint():
    subs = [f"S{i:02d}" for i in range(14)]
    for tr, te in group_kfold(subs, 5):
        assert_disjoint(tr, te)
        assert set(tr) | set(te) == set(subs)
    assert len(leave_one_subject_out(subs)) == 14
    a, b, c = driver_split(subs)
    assert_disjoint(a, b, c)
    try:
        assert_disjoint(["S01"], ["S01"])
        raise RuntimeError("leak not detected")
    except AssertionError:
        pass


def test_interval_labels_and_kappa():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        p = os.path.join(d, "labels.csv")
        with open(p, "w") as f:
            f.write("video_id,start_ms,end_ms,label,annotator\n"
                    "v1,1000,2000,eye_closure,A\nv1,1100,2100,eye_closure,B\n")
        a = load_intervals(p, "A")["v1"]
        b = load_intervals(p, "B")["v1"]
    fa, fb = frame_labels(a, "eye_closure", 5000), frame_labels(b, "eye_closure", 5000)
    k = cohen_kappa(fa, fb)
    assert 0.7 < k < 1.0
    assert cohen_kappa(fa, fa) == 1.0


def test_uta_path_parsing():
    r = parse_path(Path("uta/Fold3_part2/37/10.mov"))
    assert r["fold"] == 3 and r["subject"] == "37" and r["label"] == 10
    assert parse_path(Path("uta/Fold1_part1/01/readme.txt")) is None


def test_operating_curve_on_synthetic():
    videos = {"drowsy": list(generate("drowsy")), "normal": list(generate("normal"))}
    gt = {"drowsy": [Interval(200_000, 201_300, "eye_closure"), Interval(230_000, 232_600, "eye_closure")]}
    cfg = load_config()
    res = evaluate(videos, gt, cfg, level="warning", gt_labels={"eye_closure"})
    assert res["recall"] == 1.0
    curve = sweep(videos, gt, "eye.long_closure_ms", [800, 1000, 1500], cfg, level="warning",
                  gt_labels={"eye_closure"})
    assert len(curve) == 3
    assert pick_operating_point(curve, max_fa_per_hour=1000) is not None
    assert np.isfinite(curve[0]["false_alerts_per_hour"])
