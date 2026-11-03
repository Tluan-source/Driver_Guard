"""Hash and probe UTA-RLDD full videos, producing an extraction-ready manifest version.

The metadata intake (prepare_uta_full.py) never opens bulk media. This step reads every
file once for SHA-256, opens it with the same OpenCV backend used for extraction, decodes
the first frame after container rotation, and writes a new manifest version. Full decode
and per-frame timestamp drift are verified during feature extraction, which already
decodes every frame.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
# Above 10 Hz every 100 ms feature bin can receive a frame; several phones recorded at 12-19 fps.
MIN_FPS, MAX_FPS = 11.0, 61.0
# Below this the 10 Hz stream is built from sparse frames; reported as a possible acquisition confound.
LOW_FPS = 20.0
MIN_DURATION_S = 60.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"opened": False}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, image = cap.read()
        return {"opened": True, "fps": round(fps, 4), "frame_count": frames,
                "duration_s": round(frames / fps, 2) if fps > 0 else None,
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace"),
                "rotation_meta_deg": int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0),
                "rotation_applied": bool(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)),
                "first_frame_decoded": bool(ok and image is not None),
                "decoded_width": int(image.shape[1]) if ok and image is not None else None,
                "decoded_height": int(image.shape[0]) if ok and image is not None else None}
    finally:
        cap.release()


def media_problems(media: dict) -> list[str]:
    if not media.get("opened"):
        return ["cannot_open"]
    problems = []
    if not MIN_FPS <= media["fps"] <= MAX_FPS:
        problems.append("fps_out_of_range")
    if media["frame_count"] <= 0 or (media["duration_s"] or 0) < MIN_DURATION_S:
        problems.append("too_short_or_unknown_length")
    if not media["first_frame_decoded"]:
        problems.append("first_frame_decode_failed")
    if media["rotation_meta_deg"] % 90 or (media["rotation_meta_deg"] and not media["rotation_applied"]):
        problems.append("rotation_not_applied")
    return problems


def validate(source_dir: Path, out_dir: Path, workers: int = 4, project_root: Path = ROOT) -> tuple[dict, dict]:
    manifest_bytes = (source_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("dataset") != "uta_rldd_full":
        raise ValueError("Expected a uta_rldd_full metadata manifest")
    if out_dir.exists():
        raise ValueError(f"Refusing to overwrite an existing manifest version: {out_dir}")
    clips = manifest["clips"]

    def check(clip: dict) -> tuple[str, dict]:
        path = (project_root / clip["path"]).resolve()
        if not path.is_relative_to(project_root.resolve()) or path.stat().st_size != clip["bytes"]:
            raise ValueError(f"Missing, moved or resized video: {clip['path']}")
        return sha256_file(path), probe_video(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, clips))
    failures = {}
    hashes = Counter(digest for digest, _ in results)
    for clip, (digest, media) in zip(clips, results, strict=True):
        problems = media_problems(media) + (["duplicate_content"] if hashes[digest] > 1 else [])
        if problems:
            failures[clip["clip_id"]] = problems
    if failures:
        raise ValueError(f"Media validation failed: {json.dumps(failures)}")

    version = out_dir.name
    generated = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    result = copy.deepcopy(manifest)
    result.update(version=version, generated_at=generated, status="media_validated",
                  extraction_ready=True,
                  derived_from={"version": manifest["version"],
                                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
    for clip, (digest, media) in zip(result["clips"], results, strict=True):
        clip.update(sha256=digest, media=media, extraction_ready=True,
                    validation={"file_size": "matches_intake", "content_hash": "sha256",
                                "decode": "first_frame_ok; every frame decoded during extraction",
                                "pts": "frame index/fps; drift against decoder PTS checked during extraction",
                                "rotation": "container rotation applied by OpenCV before landmarks"})
    result["limitations"] = [
        "Part continuity is unverified; reset sequences at each file boundary",
        "Timestamps use frame index/fps; extraction rejects clips whose decoder PTS drifts over 100 ms",
        "Some recordings are 12-19 fps, and for some subjects frame rate differs between labels",
        "Pilot and full UTA share subjects and do not create an independent test cohort",
        "No synchronized EEG or verified onset labels",
    ]
    media = [m for _, m in results]
    audit = {"schema_version": 1, "generated_at": generated, "version": version,
             "status": result["status"], "extraction_ready": True, "files": len(clips),
             "bytes": sum(c["bytes"] for c in clips),
             "duration_hours": round(sum(m["duration_s"] for m in media) / 3600, 2),
             "duration_hours_by_split": {
                 split: round(sum(m["duration_s"] for c, m in zip(clips, media, strict=True)
                                  if c["split"] == split) / 3600, 2)
                 for split in ("train", "validation", "test")},
             "fps": dict(Counter(str(round(m["fps"], 2)) for m in media).most_common()),
             "low_fps_clips": {c["clip_id"]: m["fps"] for c, m in zip(clips, media, strict=True)
                               if m["fps"] < LOW_FPS},
             "subjects_with_fps_varying_by_label": sorted(
                 subject for subject in {c["subject"] for c in clips}
                 if len({m["fps"] < LOW_FPS for c, m in zip(clips, media, strict=True)
                         if c["subject"] == subject}) > 1),
             "codecs": dict(Counter(m["codec"] for m in media)),
             "rotation_meta_deg": dict(Counter(str(m["rotation_meta_deg"]) for m in media)),
             "decoded_resolutions": dict(Counter(f"{m['decoded_width']}x{m['decoded_height']}" for m in media)),
             "checks": {"content_hash": "sha256_all", "unique_content": True, "open": "all",
                        "first_frame_decode": "all", "fps_range": [MIN_FPS, MAX_FPS],
                        "min_duration_s": MIN_DURATION_S, "full_decode": "during_extraction",
                        "presentation_timestamps": "during_extraction"}}
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "split.json").write_bytes((source_dir / "split.json").read_bytes())
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/manifests/uta_rldd_full/v001")
    parser.add_argument("--out", type=Path, default=ROOT / "data/manifests/uta_rldd_full/v002")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    _, audit = validate(args.source, args.out, args.workers)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
