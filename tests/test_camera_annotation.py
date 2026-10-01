import csv
import json

import pytest

from driverguard.learning.camera_features import sha256_file
from scripts.prepare_camera_annotation import FIELDS, prepare, validate


def _sources(tmp_path, split="train"):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source fixture; no decoding required")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"clips": [{"clip_id": "S01_0", "subject": "01", "split": split,
                                         "path": str(video), "sha256": sha256_file(video),
                                         "media": {"duration_seconds": 10.0}}]}))
    queue = tmp_path / "queue.csv"
    queue.write_text(f"clip_id,subject,split\nS01_0,01,{split}\n")
    return raw, queue


def _write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_anonymous_review_stays_unreviewed_and_detects_changed_video(tmp_path):
    raw, queue = _sources(tmp_path)
    output = tmp_path / "review"
    prepare(raw, queue, output)
    annotations = output / "annotations.csv"
    report = validate(raw, output / "private_manifest.json", annotations)
    assert report["status"] == "review_pending"
    assert report["unreviewed_intervals"] == 1
    assert "S01_0" not in annotations.read_text()
    assert "score" not in annotations.read_text()
    (output / "videos/R001.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate(raw, output / "private_manifest.json", annotations)


def test_review_package_refuses_test_clips_and_overwriting(tmp_path):
    raw, queue = _sources(tmp_path, split="test")
    with pytest.raises(ValueError, match="development"):
        prepare(raw, queue, tmp_path / "review")
    raw, queue = _sources(tmp_path)
    output = tmp_path / "review"
    prepare(raw, queue, output)
    with pytest.raises(FileExistsError):
        prepare(raw, queue, output)


@pytest.mark.parametrize("start,end,state,visibility,reviewer,notes,match", [
    (0, 11, "alert", "usable", "a", "eyes open", "outside"),
    (0, 10, "alert", "unusable", "a", "face hidden", "uncertain"),
    (0, 10, "alert", "usable", "", "eyes open", "reviewer"),
    (0, 10, "alert", "usable", "a", "", "evidence"),
    (1, 10, "alert", "usable", "a", "eyes open", "gaps"),
])
def test_invalid_annotation_cannot_pass(tmp_path, start, end, state, visibility, reviewer, notes, match):
    raw, queue = _sources(tmp_path)
    output = tmp_path / "review"
    prepare(raw, queue, output)
    path = output / "annotations.csv"
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    row.update(start_s=start, end_s=end, state=state, visibility=visibility,
               review_status="reviewed", reviewer=reviewer, notes=notes)
    _write_rows(path, [row])
    with pytest.raises(ValueError, match=match):
        validate(raw, output / "private_manifest.json", path)


def test_complete_review_requires_adjudication_and_nonoverlapping_intervals(tmp_path):
    raw, queue = _sources(tmp_path)
    output = tmp_path / "review"
    prepare(raw, queue, output)
    path = output / "annotations.csv"
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    row.update(state="alert", visibility="usable", review_status="reviewed", reviewer="a", notes="eyes open")
    rows = [{**row, "start_s": 0, "end_s": 5}, {**row, "start_s": 5, "end_s": 10}]
    _write_rows(path, rows)
    assert validate(raw, output / "private_manifest.json", path)["status"] == "review_complete_requires_adjudication"
    rows[1]["start_s"] = 4
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="overlaps"):
        validate(raw, output / "private_manifest.json", path)
