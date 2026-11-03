"""Prepare a small, subject-disjoint camera corpus from public cropped UTA clips.

This derivative has weak recording-level labels, not EEG supervision or onset labels.
Only one clip-length variant is selected, so overlapping variants cannot enter splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import cv2
import requests

DATASET = "mathiasviborg/uta-rldd-videos-cropped-by-faces"
API = "https://www.kaggle.com/api/v1/datasets"
SOURCE_URL = f"https://www.kaggle.com/datasets/{DATASET}"
ORIGINAL_URL = "https://sites.google.com/view/utarldd/home"
PAPER_URL = "https://arxiv.org/abs/1904.07312"
FILE_PATTERN = re.compile(
    r"UTA-RLDD Face Cropped Video/len(?P<length>5|10|20|30|60)/"
    r"(?P<subject>\d{2})/(?P<state>0|10)/"
    r"(?P=subject)_(?P=state)_(?P<minute>\d+)_(?P<segment>\d+)\.mp4"
)
_LOCAL = threading.local()


class SystemTrustAdapter(requests.adapters.HTTPAdapter):
    """Use verified OS trust roots, including the Windows certificate store."""

    def __init__(self):
        self.context = ssl.create_default_context()
        super().__init__()

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host, pool = super().build_connection_pool_key_attributes(request, verify, cert)
        if verify is True:
            pool["ssl_context"] = self.context
        return host, pool


def public_session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
        _LOCAL.session.mount("https://", SystemTrustAdapter())
    return _LOCAL.session


def parse_source_file(name: str) -> dict:
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or "\\" in name:
        raise ValueError(f"Unsafe source file: {name!r}")
    match = FILE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported cropped UTA file: {name!r}")
    subject = match["subject"]
    state = int(match["state"])
    minute, segment = int(match["minute"]), int(match["segment"])
    return {
        "clip_id": f"S{subject}_{state}_{minute:04d}_{segment:02d}",
        "subject": subject,
        "source_label": state,
        "target": 0 if state == 0 else 1,
        "source_video_id": f"{subject}_{state}",
        "source_minute": minute,
        "source_segment": segment,
        "clip_length_seconds": int(match["length"]),
        "source_file": name,
    }


def request_json(url: str, *, params=None, session=None, retries=3, sleeper=time.sleep) -> dict:
    client = session or public_session()
    for attempt in range(retries):
        try:
            with client.get(url, params=params, timeout=(10, 30)) as response:
                response.raise_for_status()
                result = response.json()
            if not isinstance(result, dict):
                raise TypeError("Public metadata response must be an object")
            return result
        except (requests.RequestException, ValueError, TypeError):
            if attempt + 1 == retries:
                raise
            sleeper(min(0.5 * 2**attempt, 4))
    raise ValueError("retries must be positive")


def enumerate_files(*, clip_length=10, session=None, max_pages=200, progress=False) -> list[dict]:
    """Read sorted public pages through the requested folder, detecting broken pagination."""
    token = None
    seen_tokens, names = set(), set()
    selected = []
    previous_name = ""
    folder = f"UTA-RLDD Face Cropped Video/len{clip_length}/"
    for page in range(max_pages):
        params = {"pageSize": 200}
        if token:
            params["pageToken"] = token
        obj = request_json(f"{API}/list/{DATASET}", params=params, session=session)
        entries = obj.get("datasetFiles")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Public file listing is empty or malformed")
        passed_folder = False
        for entry in entries:
            name = entry.get("name")
            if not isinstance(name, str) or name in names or name < previous_name:
                raise ValueError("Public file listing has duplicate or unsorted paths")
            previous_name = name
            names.add(name)
            if name.startswith(folder):
                item = parse_source_file(name)
                size = entry.get("totalBytes")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ValueError("Public file size must be a positive integer")
                selected.append({**item, "source_bytes": size})
            elif name > folder and selected:
                passed_folder = True
        if progress and (page % 10 == 0 or passed_folder):
            print(f"Listing page {page + 1}: {len(selected)} len{clip_length} clips", flush=True)
        next_token = obj.get("nextPageToken") or obj.get("nextPageTokenNullable")
        if passed_folder or not next_token:
            return selected
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise ValueError("Public file listing repeated a pagination token")
        seen_tokens.add(next_token)
        token = next_token
    raise ValueError("Public file listing exceeded the page limit")


def select_clips(files: list[dict], clips_per_class=3) -> tuple[list[dict], list[dict]]:
    if clips_per_class < 1:
        raise ValueError("clips_per_class must be positive")
    grouped: dict[tuple[str, int], list[dict]] = {}
    identities = set()
    for item in files:
        key = item["source_file"]
        if key in identities:
            raise ValueError("Duplicate source file in selection")
        identities.add(key)
        grouped.setdefault((item["subject"], item["source_label"]), []).append(item)
    if len({item["clip_length_seconds"] for item in files}) != 1:
        raise ValueError("Select exactly one clip-length variant")
    selected, excluded = [], []
    for subject in sorted({item["subject"] for item in files}):
        classes = [grouped.get((subject, state), []) for state in (0, 10)]
        if any(len(items) < clips_per_class for items in classes):
            excluded.append({"subject": subject, "class_counts": {"0": len(classes[0]), "10": len(classes[1])},
                             "reason": "missing_class_or_insufficient_clips"})
            continue
        for items in classes:
            items = sorted(items, key=lambda item: (item["source_minute"], item["source_segment"]))
            indices = ([len(items) // 2] if clips_per_class == 1 else
                       [round(i * (len(items) - 1) / (clips_per_class - 1)) for i in range(clips_per_class)])
            selected.extend(dict(items[index]) for index in indices)
    if not selected:
        raise ValueError("No subjects have enough clips from both reference classes")
    return selected, excluded


def split_subjects(subjects, seed=42) -> dict[str, list[str]]:
    subjects = sorted(set(subjects))
    if len(subjects) < 6:
        raise ValueError("At least six subjects are required for train/validation/test")
    random.Random(seed).shuffle(subjects)
    train_end = round(0.66 * len(subjects))
    validation_end = train_end + round(0.17 * len(subjects))
    return {"train": sorted(subjects[:train_end]), "validation": sorted(subjects[train_end:validation_end]),
            "test": sorted(subjects[validation_end:])}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_media(path: Path, clip_length: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Downloaded file is not a readable video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (width != 224 or height != 224 or not math.isfinite(fps) or abs(fps - 10) > 0.1
                or frames < 2 or frames > math.ceil((clip_length + 1) * 10)):
            raise ValueError("Video differs from the declared 224x224, 10 FPS cropped contract")
        decoded = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape[:2] != (height, width):
                raise ValueError("Video contains an invalid decoded frame")
            decoded += 1
            if decoded > frames:
                raise ValueError("Video frame count differs from its header")
        if decoded != frames:
            raise ValueError("Downloaded video is truncated or contains undecodable frames")
        return {"width": width, "height": height, "fps": fps, "n_frames": frames,
                "decoded_frames": decoded, "duration_seconds": frames / fps}
    finally:
        capture.release()


def download_clip(clip: dict, destination: Path, *, session=None, retries=3,
                  validator=validate_media, sleeper=time.sleep) -> dict:
    parsed = parse_source_file(clip["source_file"])
    relative = Path("videos") / parsed["subject"] / str(parsed["source_label"]) / Path(clip["source_file"]).name
    destination = destination.resolve()
    target = (destination / relative).resolve()
    if not target.is_relative_to(destination):
        raise ValueError(f"Download target {str(target)!r} escapes the selected destination {str(destination)!r}")
    expected_size = int(clip["source_bytes"])
    if expected_size < 1:
        raise ValueError("Expected source size must be positive")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        digest = sha256_file(target)
        if clip.get("sha256") and digest != clip["sha256"]:
            raise ValueError(f"Previously downloaded clip changed: {relative}")
        media = validator(target, parsed["clip_length_seconds"])
        return {"path": str(target), "download_status": "downloaded", "sha256": digest, "media": media}
    temporary = target.with_suffix(target.suffix + ".part")
    client = session or public_session()
    url = f"{API}/download/{DATASET}/{quote(clip['source_file'], safe='')}"
    for attempt in range(retries):
        try:
            digest, size = hashlib.sha256(), 0
            with client.get(url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if content_type not in {"video/mp4", "application/octet-stream"}:
                    raise ValueError(f"Expected one MP4, received {content_type!r}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected_size:
                    raise ValueError("HTTP Content-Length differs from public file metadata")
                with temporary.open("wb") as stream:
                    for block in response.iter_content(chunk_size=128 * 1024):
                        if not block:
                            continue
                        size += len(block)
                        if size > expected_size:
                            raise ValueError("Download exceeds the selected clip size")
                        digest.update(block)
                        stream.write(block)
            if size != expected_size:
                raise ValueError("Downloaded bytes differ from public file metadata")
            media = validator(temporary, parsed["clip_length_seconds"])
            checksum = digest.hexdigest()
            if clip.get("sha256") and checksum != clip["sha256"]:
                raise ValueError("Downloaded clip differs from its recorded SHA256")
            temporary.replace(target)
            return {"path": str(target), "download_status": "downloaded", "sha256": checksum, "media": media}
        except (requests.RequestException, OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            sleeper(min(0.5 * 2**attempt, 4))
    raise ValueError("retries must be positive")


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_cached_clip(clip: dict, destination: Path) -> None:
    """Verify a frozen clip without transferring or changing any corpus files."""
    parsed = parse_source_file(clip["source_file"])
    destination = destination.resolve()
    relative = Path("videos") / parsed["subject"] / str(parsed["source_label"]) / Path(clip["source_file"]).name
    target = (destination / relative).resolve()
    if not target.is_relative_to(destination) or Path(clip["path"]).resolve() != target:
        raise ValueError(f"Frozen clip path differs from its selected destination: {relative}")
    if not target.is_file() or target.stat().st_size != clip["source_bytes"]:
        raise ValueError(f"Frozen clip is missing or has a different byte count: {relative}")
    if sha256_file(target) != clip["sha256"]:
        raise ValueError(f"Previously downloaded clip changed: {relative}")
    if validate_media(target, parsed["clip_length_seconds"]) != clip["media"]:
        raise ValueError(f"Frozen clip media properties changed: {relative}")


def completed_manifest(manifest: dict) -> bool:
    summary = manifest.get("download_summary", {})
    clips = manifest["clips"]
    return bool(clips) and all(
        clip.get("download_status") == "downloaded"
        and isinstance(clip.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", clip["sha256"]) is not None
        and isinstance(clip.get("media"), dict)
        for clip in clips
    ) and (summary.get("failed") == 0 and summary.get("downloaded") == len(clips)
           and summary.get("source_version_unchanged") is True)


def build_manifest(files, metadata, destination: Path, *, clips_per_class=3, seed=42) -> dict:
    selected, excluded = select_clips(files, clips_per_class)
    split = split_subjects([clip["subject"] for clip in selected], seed)
    partition = {subject: name for name, subjects in split.items() for subject in subjects}
    for clip in selected:
        relative = Path("videos") / clip["subject"] / str(clip["source_label"]) / Path(clip["source_file"]).name
        clip.update(split=partition[clip["subject"]], path=str((destination / relative).resolve()),
                    download_status="pending")
    selection = {"clip_length_seconds": selected[0]["clip_length_seconds"], "clips_per_class": clips_per_class,
                 "method": "Evenly spaced numeric clip positions: first, middle, last for three clips",
                 "seed": seed, "subject_count": len(partition), "clip_count": len(selected),
                 "selected_bytes": sum(clip["source_bytes"] for clip in selected), "excluded_subjects": excluded}
    fingerprint = hashlib.sha256(json.dumps({"files": [(c["source_file"], c["source_bytes"]) for c in selected],
                                             "split": split, "version": metadata["currentVersionNumber"]},
                                            sort_keys=True).encode()).hexdigest()
    return {
        "format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET, "dataset_version": metadata["currentVersionNumber"],
        "selection_sha256": fingerprint, "target": "camera_observable_drowsiness",
        "label_mapping": {"0": 0, "10": 1}, "label_type": "weak_original_video_level",
        "source": {"url": SOURCE_URL, "title": metadata.get("title"),
                   "declared_mirror_license": metadata.get("licenseName"), "description": metadata.get("description"),
                   "authoritative_dataset_url": ORIGINAL_URL, "original_paper_url": PAPER_URL,
                   "rights_note": "Mirror license is the uploader declaration. Original terms are not established "
                                 "by this snapshot; consult the authoritative dataset source. Do not redistribute "
                                 "raw video or derived features without checking applicable terms.",
                   "source_video_identity_note": "Grouping ID derived from subject/state paths; not an original "
                                                 "recording checksum. Crop timing follows filename positions.",
                   "version_note": "Public individual-file routes serve current content; version metadata is "
                                   "checked before and after download, with expected bytes and local SHA256."},
        "limitations": ["No synchronized EEG", "No drowsiness-onset timestamps", "No event-level ground truth",
                        "Clip labels inherited from whole recordings; not independently verified at each frame",
                        "Smoothed face crops at 224x224 and 10 FPS differ from full-frame live camera input",
                        "10 FPS can undersample rapid eyelid closing/reopening phases",
                        "Synthetic darkness cannot establish real night or NIR performance",
                        "Short balanced clips cannot establish real-world false alerts per hour",
                        "Subject 42 lacks the drowsy class in the source derivative"],
        "selection": selection, "split": split, "clips": selected,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-only", action="store_true", help="save selection and provenance, without media")
    mode.add_argument("--download", action="store_true", help="stream selected clips, checking bytes and video")
    parser.add_argument("--out", type=Path, default=Path("data/raw/camera_uta_pilot"))
    parser.add_argument("--clip-length", type=int, choices=(10, 30), default=10)
    parser.add_argument("--clips-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-downloads", type=int, default=None, help="limit media transfers for a first smoke run")
    parser.add_argument("--refresh-listing", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.clips_per_class < 1 or (args.max_downloads is not None and args.max_downloads < 1):
        parser.error("workers, clips-per-class and max-downloads must be positive")
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metadata = request_json(f"{API}/view/{DATASET}")
    version = metadata.get("currentVersionNumber")
    if not isinstance(version, int) or version < 1 or metadata.get("ref") != DATASET:
        raise ValueError("Dataset metadata does not identify the expected source/version")
    cache_path = destination / "files.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
    if (cache and not args.refresh_listing and cache.get("dataset_version") == version
            and cache.get("clip_length_seconds") == args.clip_length):
        files = cache["files"]
    else:
        files = enumerate_files(clip_length=args.clip_length, progress=True)
        write_json(cache_path, {"dataset": DATASET, "dataset_version": version,
                               "clip_length_seconds": args.clip_length, "files": files})
    manifest = build_manifest(files, metadata, destination, clips_per_class=args.clips_per_class, seed=args.seed)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("selection_sha256") != manifest["selection_sha256"]:
            raise ValueError("Output has a different frozen selection; choose another --out directory")
        previous_clips = {clip["source_file"]: clip for clip in previous["clips"]}
        if len(previous_clips) != len(manifest["clips"]):
            raise ValueError("Frozen manifest has duplicate or missing selected clips")
        for clip in manifest["clips"]:
            old = previous_clips.get(clip["source_file"])
            if old is None or any(old.get(key) != value for key, value in clip.items() if key != "download_status"):
                raise ValueError("Frozen clip selection or destination differs from its recorded snapshot")
        manifest = previous
        if completed_manifest(manifest):
            if args.download:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    pending = [pool.submit(verify_cached_clip, clip, destination) for clip in manifest["clips"]]
                    for future in as_completed(pending):
                        future.result()
                final_metadata = request_json(f"{API}/view/{DATASET}")
                if final_metadata.get("currentVersionNumber") != version or final_metadata.get("ref") != DATASET:
                    raise ValueError("Source version changed during cache verification; do not use this corpus")
            print(json.dumps({"manifest": str(manifest_path), "selection": manifest["selection"],
                              "download_summary": manifest["download_summary"],
                              "cache_verified": len(manifest["clips"]) if args.download else 0}, indent=2), flush=True)
            return 0
    write_json(destination / "kaggle_metadata.json", metadata)
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "selection": manifest["selection"],
                      "split_counts": {name: len(subjects) for name, subjects in manifest["split"].items()}},
                     indent=2), flush=True)
    if args.list_only:
        return 0
    clips = manifest["clips"][:args.max_downloads] if args.max_downloads else manifest["clips"]
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download_clip, clip, destination): clip for clip in clips}
        for completed, future in enumerate(as_completed(pending), 1):
            clip = pending[future]
            try:
                clip.update(future.result())
                clip.pop("download_error", None)
            except (requests.RequestException, OSError, ValueError, cv2.error) as error:
                failures += 1
                clip.update(download_status="failed", download_error=f"{type(error).__name__}: {error}")
            write_json(manifest_path, manifest)
            print(f"{completed}/{len(clips)} {clip['clip_id']} {clip['split']}: {clip['download_status']}", flush=True)
    final_metadata = request_json(f"{API}/view/{DATASET}")
    manifest["download_summary"] = {"attempted": len(clips), "failed": failures,
                                    "downloaded": sum(c["download_status"] == "downloaded" for c in manifest["clips"]),
                                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "source_version_unchanged": final_metadata.get("currentVersionNumber") == version}
    write_json(manifest_path, manifest)
    if not manifest["download_summary"]["source_version_unchanged"]:
        raise ValueError("Source version changed during download; do not use this corpus")
    print(json.dumps(manifest["download_summary"], indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
