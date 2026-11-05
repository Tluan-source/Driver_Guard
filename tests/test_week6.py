import csv
import json

import numpy as np
import pandas as pd
import pytest

from driverguard.capture.source import CameraSource
from driverguard.evaluation.alignment import fit_clock, validate_indices
from driverguard.learning.camera.dann.train import accuracy_target_met
from scripts.data.materialize_week6_review import cut_interval
from scripts.evaluation.audit_drozy_alignment import audit
from scripts.evaluation.prepare_week6 import create_review_queue, review_progress
from scripts.evaluation.run_week6_ablation import smoke_manifest


def test_camera_timestamp_ties_preserve_order_but_long_gap_is_preserved(monkeypatch):
    camera = CameraSource.__new__(CameraSource)
    camera._t0, camera._last_timestamp_ms, camera.timestamp_adjustments = 0, -1, 0
    clock = iter([.1001, .1009, .1012, .6001])
    monkeypatch.setattr("driverguard.capture.source.time.monotonic", lambda: next(clock))
    assert [camera._now_ms() for _ in range(4)] == [100, 101, 102, 600]
    assert camera.timestamp_adjustments == 2


@pytest.mark.parametrize("accuracy,expected", [(.80, False), (.85, False), (.85001, True), (None, False)])
def test_primary_target_does_not_substitute_macro_f1(accuracy, expected):
    assert accuracy_target_met({"window": {"accuracy": accuracy, "macro_f1": .99}}) is expected


def test_alignment_recovers_clock_offset_and_drift():
    times = np.array([0., 10., 30., 60.])
    result = fit_clock(times, 1.0002*times + 2, video_duration=60, psg_duration=65)
    assert result["within_tolerance"]
    assert result["offset_s"] == pytest.approx(2)
    assert result["drift_ppm"] == pytest.approx(200)


@pytest.mark.parametrize("video,psg", [([0, 1], [0, 1]), ([0, 0, 1], [0, 1, 2]),
                                     ([0, 1, 2], [0, 2, 1]), ([0, np.nan, 2], [0, 1, 2]),
                                     ([0, 1, 8], [0, 1, 8])])
def test_alignment_rejects_incomplete_or_invalid_anchors(video, psg):
    with pytest.raises(ValueError):
        fit_clock(video, psg, video_duration=5, psg_duration=5)


def test_anchor_residuals_can_fail_even_for_monotonic_clocks():
    result = fit_clock([0, 1, 2, 3], [0, 1, 2.5, 3], video_duration=3, psg_duration=4, tolerance_s=.1)
    assert not result["within_tolerance"]


def test_sample_mapping_requires_explicit_base_and_correct_length():
    assert validate_indices([1, 11, 21], index_base=1, n_psg_samples=21, n_video_frames=3)["last_sample_zero_based"] == 20
    with pytest.raises(ValueError):
        validate_indices([1, 11, 22], index_base=1, n_psg_samples=21, n_video_frames=3)
    with pytest.raises(ValueError):
        validate_indices([1, 11], index_base=1, n_psg_samples=21, n_video_frames=3)
    with pytest.raises(ValueError):
        validate_indices([1., 1.5, 2.], index_base=0, n_psg_samples=21, n_video_frames=3)


def test_missing_drozy_cannot_report_ready(tmp_path):
    result = audit(tmp_path / "missing")
    assert result["status"] == "missing_data" and not result["ready_for_eeg_supervision"]


def _queue(tmp_path):
    clips, rows = [], []
    for i in range(40):
        clips.append({"clip_id": f"C{i}", "subject": str(100+i), "split": "train", "n_frames": 6000,
                      "part_index": None, "source_sha256": "a"*64})
        rows.append({"clip_id": f"C{i}", "label": i % 2, "prediction": 1-i % 2,
                     "accepted": True, "end_ms": 200000})
    create_review_queue({"clips": clips}, pd.DataFrame(rows), tmp_path)
    return clips, rows


def test_review_queue_is_blinded_and_never_marks_missing_videos_complete(tmp_path):
    _queue(tmp_path)
    with (tmp_path / "reviewer_a.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert "source_label" not in reader.fieldnames and "subject" not in reader.fieldnames
        assert len(list(reader)) == 40
    result = review_progress(tmp_path)
    assert not result["ready_for_training_labels"] and result["verified_videos"] == 0


def test_preparation_preserves_existing_human_notes(tmp_path):
    clips, rows = _queue(tmp_path)
    path = tmp_path / "reviewer_a.csv"
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields, reviews = reader.fieldnames, list(reader)
    reviews[0]["notes"] = "Human note retained"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews)
    create_review_queue({"clips": clips}, pd.DataFrame(rows), tmp_path)
    assert "Human note retained" in path.read_text()


def test_review_rejects_incomplete_intervals(tmp_path):
    _queue(tmp_path)
    path = tmp_path / "reviewer_a.csv"
    path.write_text(path.read_text().replace("0,10,", "1,10,", 1))
    with pytest.raises(ValueError, match="interval"):
        review_progress(tmp_path)


def test_smoke_manifest_never_contains_outer_holdout(tmp_path):
    clips = [{"split": split, "subject": str(100+i), "label": label, "features": "unused.npz"}
             for split in ("train", "validation", "test") for i in range(3) for label in (0, 1, 2)]
    source = tmp_path / "fold.json"
    source.write_text(json.dumps({"clips": clips}))
    out = smoke_manifest(source, tmp_path / "smoke.json")
    selected = json.loads(out.read_text())["clips"]
    assert len(selected) == 8 and all(c["split"] != "test" and c["label"] in (0, 2) for c in selected)


def test_review_cut_uses_timestamps_and_preserves_source(tmp_path):
    from fractions import Fraction

    import av
    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        encoder = container.add_stream("mpeg4", rate=10)
        encoder.width = encoder.height = 32
        encoder.pix_fmt = "yuv420p"
        for index in range(40):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index, np.uint8), format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in encoder.encode(frame):
                container.mux(packet)
        for packet in encoder.encode():
            container.mux(packet)
    before = source.read_bytes()
    result = cut_interval(source, tmp_path / "review.mp4", 1, 3)
    assert result["frames"] == 20 and result["max_sample_timing_error_s"] < .001
    assert source.read_bytes() == before
    with pytest.raises(ValueError):
        cut_interval(source, tmp_path / "review.mp4", 1, 3)
