import hashlib
import json
import ssl
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.prepare_camera_uta import (
    SystemTrustAdapter,
    build_manifest,
    download_clip,
    enumerate_files,
    parse_source_file,
    select_clips,
    split_subjects,
    validate_media,
    write_json,
)


def test_system_trust_keeps_certificate_and_hostname_verification():
    import requests

    adapter = SystemTrustAdapter()
    request = requests.Request("GET", "https://www.kaggle.com").prepare()
    _, pool = adapter.build_connection_pool_key_attributes(request, True)
    assert pool["ssl_context"] is adapter.context
    assert adapter.context.verify_mode == ssl.CERT_REQUIRED
    assert adapter.context.check_hostname


def source_file(subject="01", state=0, minute=0, segment=0, length=10):
    name = (f"UTA-RLDD Face Cropped Video/len{length}/{subject}/{state}/"
            f"{subject}_{state}_{minute}_{segment:02d}.mp4")
    return {**parse_source_file(name), "source_bytes": 123}


def test_cropped_path_uses_actual_subject_and_state_not_filename_suffix():
    clip = source_file(subject="12", state=10, minute=9, segment=2)
    assert clip["subject"] == "12"
    assert clip["target"] == 1 and clip["source_label"] == 10
    assert clip["source_video_id"] == "12_10"
    assert clip["source_minute"] == 9 and clip["source_segment"] == 2
    assert source_file(state=0)["target"] == 0


@pytest.mark.parametrize("name", [
    "/UTA-RLDD Face Cropped Video/len10/01/0/01_0_0_00.mp4",
    "../UTA-RLDD Face Cropped Video/len10/01/0/01_0_0_00.mp4",
    "UTA-RLDD Face Cropped Video/len10/01/10/02_10_0_00.mp4",
    "UTA-RLDD Face Cropped Video/len10/01/10/01_0_0_00.mp4",
    "UTA-RLDD Face Cropped Video/len10/01/5/01_5_0_00.mp4",
    "UTA-RLDD Face Cropped Video\\len10\\01\\0\\01_0_0_00.mp4",
])
def test_invalid_or_traversing_source_path_is_rejected(name):
    with pytest.raises(ValueError):
        parse_source_file(name)


def test_selection_is_numeric_evenly_spaced_and_excludes_missing_class():
    files = [source_file(state=state, minute=minute) for state in (0, 10) for minute in (0, 1, 2, 9, 10)]
    files += [source_file(subject="42", state=0, minute=minute) for minute in range(5)]
    selected, excluded = select_clips(list(reversed(files)))
    assert [clip["source_minute"] for clip in selected] == [0, 2, 10, 0, 2, 10]
    assert {clip["subject"] for clip in selected} == {"01"}
    assert excluded == [{"subject": "42", "class_counts": {"0": 5, "10": 0},
                         "reason": "missing_class_or_insufficient_clips"}]
    with pytest.raises(ValueError, match="one clip-length"):
        select_clips(files + [source_file(length=30)])
    with pytest.raises(ValueError, match="Duplicate"):
        select_clips(files + [files[0]])


def test_split_is_deterministic_and_all_subject_variants_share_one_partition(tmp_path):
    subjects = [f"{n:02d}" for n in range(1, 61) if n != 42]
    split = split_subjects(subjects)
    assert {name: len(items) for name, items in split.items()} == {"train": 39, "validation": 10, "test": 10}
    assert split == split_subjects(list(reversed(subjects)))
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])
    files = [source_file(subject=s, state=state, minute=m) for s in subjects for state in (0, 10)
             for m in range(3)]
    manifest = build_manifest(files, {"currentVersionNumber": 1, "licenseName": "CC BY-NC-SA 4.0"}, tmp_path)
    assert len(manifest["clips"]) == 354
    assert manifest["label_type"] == "weak_original_video_level"
    assert "No synchronized EEG" in manifest["limitations"]
    for s in subjects:
        clips = [clip for clip in manifest["clips"] if clip["subject"] == s]
        assert len({clip["split"] for clip in clips}) == 1
        assert sum(clip["target"] for clip in clips) == 3
    assert manifest["selection_sha256"] == build_manifest(files, {"currentVersionNumber": 1}, tmp_path)[
        "selection_sha256"]


class FakeResponse:
    def __init__(self, *, obj=None, payload=b"abc", length=None, content_type="video/mp4"):
        self.obj = obj
        self.payload = payload
        self.headers = {"Content-Type": content_type}
        if length is not None:
            self.headers["Content-Length"] = str(length)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def json(self):
        return self.obj

    def iter_content(self, chunk_size):
        yield self.payload[:1]
        yield b""
        yield self.payload[1:]


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def listing(items, token=None):
    return FakeResponse(obj={"datasetFiles": [{"name": item["source_file"], "totalBytes": item["source_bytes"]}
                                             for item in items], "nextPageToken": token})


def test_public_listing_stops_after_length_folder_and_checks_pagination():
    first, second, later = source_file(), source_file(segment=1), source_file(length=20)
    session = FakeSession([listing([first], "NEXT"), listing([second, later], "UNNEEDED")])
    files = enumerate_files(session=session)
    assert [item["source_file"] for item in files] == [first["source_file"], second["source_file"]]
    assert session.calls[1][1]["params"]["pageToken"] == "NEXT"
    session = FakeSession([listing([first], "LOOP"), listing([second], "LOOP")])
    with pytest.raises(ValueError, match="repeated"):
        enumerate_files(session=session)
    session = FakeSession([listing([first], "NEXT"), listing([first], None)])
    with pytest.raises(ValueError, match="duplicate"):
        enumerate_files(session=session)


def test_streamed_clip_checks_length_hash_and_reuses_verified_cache(tmp_path):
    clip = {**source_file(), "source_bytes": 3}
    validator_calls = []

    def validator(path, length):
        validator_calls.append((path, length))
        assert path.read_bytes() == b"abc"
        return {"fps": 10}

    session = FakeSession([FakeResponse(length=3)])
    result = download_clip(clip, tmp_path, session=session, validator=validator)
    assert session.calls[0][1]["stream"] is True
    assert "%2F" in session.calls[0][0] and "%20" in session.calls[0][0]
    assert result["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert Path(result["path"]).read_bytes() == b"abc"
    assert not list(tmp_path.rglob("*.part"))
    empty_session = FakeSession([])
    reused = download_clip({**clip, **result}, tmp_path, session=empty_session, validator=validator)
    assert reused == result and not empty_session.calls
    assert len(validator_calls) == 2
    Path(result["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed"):
        download_clip({**clip, **result}, tmp_path, session=empty_session, validator=validator)


def frozen_corpus(tmp_path, monkeypatch):
    import scripts.prepare_camera_uta as prepare

    files = [source_file(subject=f"{subject:02d}", state=state) for subject in range(1, 7) for state in (0, 10)]
    for clip in files:
        clip["source_bytes"] = 3
    metadata = {"currentVersionNumber": 1, "ref": prepare.DATASET, "title": "Frozen source title"}
    manifest = build_manifest(files, metadata, tmp_path, clips_per_class=1)
    media = {"fps": 10}
    for clip in manifest["clips"]:
        path = Path(clip["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"abc")
        clip.update(download_status="downloaded", sha256=hashlib.sha256(b"abc").hexdigest(), media=media)
    manifest["download_summary"] = {"attempted": len(files), "failed": 0, "downloaded": len(files),
                                    "completed_at_utc": "2026-01-01T00:00:00+00:00", "source_version_unchanged": True}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(tmp_path / "files.json", {"dataset": prepare.DATASET, "dataset_version": 1,
                                         "clip_length_seconds": 10, "files": files})
    monkeypatch.setattr(prepare, "request_json", lambda *_: {**metadata, "title": "Updated source title"})
    monkeypatch.setattr(prepare, "validate_media", lambda *_: dict(media))
    monkeypatch.setattr(prepare, "public_session", lambda: pytest.fail("Completed cache must not download media"))
    return prepare, manifest_path, manifest


def test_completed_rerun_verifies_without_changing_frozen_manifest_hash(tmp_path, monkeypatch):
    prepare, manifest_path, manifest = frozen_corpus(tmp_path, monkeypatch)
    original = manifest_path.read_bytes()
    assert prepare.main(["--download", "--out", str(tmp_path), "--clips-per-class", "1"]) == 0
    assert manifest_path.read_bytes() == original
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_completed_rerun_rejects_hash_corruption_without_changing_snapshot(tmp_path, monkeypatch):
    prepare, manifest_path, manifest = frozen_corpus(tmp_path, monkeypatch)
    original = manifest_path.read_bytes()
    Path(manifest["clips"][0]["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed"):
        prepare.main(["--download", "--out", str(tmp_path), "--clips-per-class", "1"])
    assert manifest_path.read_bytes() == original


@pytest.mark.parametrize("response", [
    FakeResponse(length=4), FakeResponse(payload=b"ab"), FakeResponse(payload=b"abcd"),
    FakeResponse(content_type="application/zip"), FakeResponse(content_type="text/html"),
])
def test_bad_downloads_are_bounded_and_never_committed(tmp_path, response):
    clip = {**source_file(), "source_bytes": 3}
    session = FakeSession([response, response])
    delays = []
    with pytest.raises(ValueError):
        download_clip(clip, tmp_path, session=session, retries=2, sleeper=delays.append,
                      validator=lambda *_: pytest.fail("Invalid bytes must not reach media validation"))
    assert len(session.calls) == 2 and delays == [0.5]
    assert not list(tmp_path.rglob("*.mp4")) and not list(tmp_path.rglob("*.part"))


def test_unreadable_media_is_rejected_and_real_cropped_video_is_decoded(tmp_path):
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="readable"):
        validate_media(invalid, 10)
    path = tmp_path / "valid.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
    assert writer.isOpened()
    for frame in range(10):
        writer.write(np.full((224, 224, 3), frame * 10, dtype=np.uint8))
    writer.release()
    result = validate_media(path, 10)
    assert result["fps"] == 10 and result["n_frames"] == result["decoded_frames"] == 10
    assert result["duration_seconds"] == 1
