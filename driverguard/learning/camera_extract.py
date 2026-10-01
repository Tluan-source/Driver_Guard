"""Extract the shared camera feature contract from a versioned video manifest."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..capture import VideoFileSource
from ..config import DriverGuardConfig, load_config, resolve_path
from ..perception.extractor import PerceptionExtractor
from ..perception.face import MediaPipeFaceLandmarker
from .camera_features import CAMERA_FPS, CameraFeatureCollector, input_contract, sha256_file


def _extract_clip(job: dict) -> dict:
    cfg = DriverGuardConfig.model_validate(job["config"])
    clip = job["clip"]
    feature_path = Path(job["out"])
    exposure = job["exposure"]
    feature_id = clip["clip_id"] + (f"#exposure={exposure:g}" if exposure != 1 else "")
    record = {"subject": clip["subject"], "clip_id": feature_id, "source_clip_id": clip["clip_id"],
              "split": clip["split"], "label": clip["target"], "features": str(feature_path),
              "synthetic_exposure_factor": exposure, "source_sha256": clip["sha256"]}
    record["illumination"] = "source_rgb_unknown" if exposure == 1 else f"synthetic_exposure_{exposure:g}"
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as data:
            if str(data["extraction_signature"].item()) != job["signature"]:
                raise ValueError(f"Existing feature cache uses different preprocessing: {feature_path}")
            record.update(n_frames=len(data["valid"]), valid_fraction=float(data["valid"].mean()))
            for name in ("mean_raw_face_brightness", "mean_raw_face_quality"):
                if name in data:
                    value = float(data[name])
                    record[name] = value if np.isfinite(value) else None
        record["features_sha256"] = sha256_file(feature_path)
        return record
    if sha256_file(clip["path"]) != clip["sha256"]:
        raise ValueError(f"Source video hash mismatch: {clip['path']}")
    extractor = PerceptionExtractor(cfg, MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker)))
    source = VideoFileSource(clip["path"], target_fps=CAMERA_FPS)
    buffer = CameraFeatureCollector(cfg)
    brightness, quality = [], []
    try:
        for frame in source.frames():
            image = frame.image_bgr
            if image is not None and exposure != 1:
                image = np.rint(image.astype(np.float32) * exposure).astype(np.uint8)
            signals = extractor.process(image, frame.ts_ms, frame.camera_ok)
            buffer.update(signals)
            quality.append(signals.face_quality)
            if signals.brightness is not None:
                brightness.append(signals.brightness)
    finally:
        source.close()
        extractor.close()
    x, timestamps, valid = buffer.arrays()
    if len(x) == 0:
        raise ValueError(f"No frames decoded: {clip['path']}")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, x=x, ts_ms=timestamps, valid=valid,
                        label=np.asarray(clip["target"], np.int64), subject=np.asarray(clip["subject"]),
                        clip_id=np.asarray(feature_id), extraction_signature=np.asarray(job["signature"]),
                        mean_raw_face_brightness=np.asarray(np.mean(brightness) if brightness else np.nan),
                        mean_raw_face_quality=np.asarray(np.mean(quality)))
    record.update(n_frames=len(valid), valid_fraction=float(valid.mean()),
                  mean_raw_face_brightness=float(np.mean(brightness)) if brightness else None,
                  mean_raw_face_quality=float(np.mean(quality)),
                  features_sha256=sha256_file(feature_path))
    return record


def extract_camera_manifest(manifest_path: str | Path, out_dir: str | Path, workers: int = 2,
                            train_exposures: tuple[float, ...] = (1.0, 0.6)) -> Path:
    manifest_path, out_dir = Path(manifest_path).resolve(), Path(out_dir).resolve()
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_config(overrides={"models.phone_backend": "none"})
    contract = input_contract(cfg, manifest_path)
    signature = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    clips = source["clips"]
    if not clips or any(not c.get("sha256") or not Path(c["path"]).is_file() for c in clips):
        raise ValueError("Download and validate every selected source clip before extraction")
    if workers < 1 or 1.0 not in train_exposures or any(not 0 < e <= 1 for e in train_exposures):
        raise ValueError("workers must be positive; exposures must include 1 and be in (0,1]")
    jobs = []
    for clip in clips:
        exposures = train_exposures if clip["split"] == "train" else (1.0,)
        for exposure in sorted(set(exposures), reverse=True):
            filename = f"{clip['subject']}_{clip['source_label']}_{Path(clip['path']).stem}_exp{exposure:g}.npz"
            jobs.append({"clip": clip, "exposure": exposure, "out": str(out_dir / filename),
                         "config": cfg.model_dump(), "signature": signature})
    records = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_clip, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 10 == 0 or index == len(jobs):
                print(f"camera features {index}/{len(jobs)}  wall={time.monotonic()-started:.1f}s", flush=True)
    manifest = {"input_contract": contract, "source_manifest": str(manifest_path),
                "source": source.get("source", {}),
                "limitations": ["Weak video-level labels, no EEG or onset labels.",
                                "Cropped RGB 224x224/10 FPS, no validated real night operation.",
                                "Exposure augmentation is synthetic and train-only."],
                "clips": sorted(records, key=lambda record: record["clip_id"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / "manifest.json"
    result.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Camera feature manifest: {result}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/features/camera_uta_pilot"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-exposures", type=float, nargs="+", default=[1.0, 0.6])
    args = parser.parse_args()
    extract_camera_manifest(args.manifest, args.out, args.workers, tuple(args.train_exposures))


if __name__ == "__main__":
    main()
