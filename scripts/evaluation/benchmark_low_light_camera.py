"""Functional camera smoke test using synthetic exposure, without camera capture.

This tests face perception and sensor-quality gating on a repeated still image.
It does not measure real night accuracy, drowsiness accuracy, or capture-to-alert
latency. Each condition uses an independent MediaPipe VIDEO tracker and engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from driverguard.config import LowLightCfg, load_config, resolve_path
from driverguard.engine import DriverGuardEngine
from driverguard.perception.extractor import PerceptionExtractor
from driverguard.perception.face import MediaPipeFaceLandmarker
from driverguard.perception.illumination import LowLightEnhancer
from driverguard.schemas import RiskLevel

EXPOSURES = (1.0, 0.4, 0.2, 0.08, 0.0)
WARMUP_FRAMES = 5
SIMULATED_FRAME_INTERVAL_MS = 100


class _TimedEnhancer(LowLightEnhancer):
    def __init__(self, cfg: LowLightCfg):
        super().__init__(cfg)
        self.last_ms = 0.0
        self.last_applied = False

    def prepare(self, image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        start = time.perf_counter()
        output, applied = super().prepare(image_bgr)
        self.last_ms = (time.perf_counter() - start) * 1000.0
        self.last_applied = applied
        return output, applied


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
    }


def _optional_mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _load_image(path: Path | None) -> tuple[Path, np.ndarray]:
    if path is None:
        from matplotlib.cbook import get_sample_data

        path = Path(get_sample_data("grace_hopper.jpg", asfileobj=False))
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode local image: {path}")
    h, w = image.shape[:2]
    scale = min(1.0, 640 / w, 480 / h)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return path, image


def _run_condition(image: np.ndarray, exposure: float, enabled: bool, iterations: int) -> dict:
    cfg = load_config(overrides={"models.phone_backend": "none", "low_light.enabled": enabled})
    frame = np.rint(image.astype(np.float32) * exposure).astype(np.uint8)
    landmarker = MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker))
    extractor = PerceptionExtractor(cfg, landmarker)
    timed_enhancer = _TimedEnhancer(cfg.low_light)
    extractor._low_light = timed_enhancer
    engine = DriverGuardEngine(cfg, trip_id="synthetic-exposure-smoke", trip_start_epoch_s=0, tz_offset_h=0)
    perception_ms, enhancement_ms = [], []
    detections, valid, applied, quality, brightness, blur = [], [], [], [], [], []
    last_tick = None
    try:
        for index in range(WARMUP_FRAMES + iterations):
            ts_ms = index * SIMULATED_FRAME_INTERVAL_MS
            start = time.perf_counter()
            signals = extractor.process(frame, ts_ms)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            last_tick = engine.step(signals)
            if index < WARMUP_FRAMES:
                continue
            perception_ms.append(elapsed_ms)
            enhancement_ms.append(timed_enhancer.last_ms)
            applied.append(timed_enhancer.last_applied)
            detections.append(signals.face_detected)
            valid.append(last_tick.face_valid)
            quality.append(signals.face_quality)
            if signals.brightness is not None:
                brightness.append(signals.brightness)
            if signals.blur is not None:
                blur.append(signals.blur)
    finally:
        extractor.close()
    assert last_tick is not None
    return {
        "exposure_factor": exposure,
        "enhancement_enabled": enabled,
        "measured_frames": iterations,
        "warmup_frames_discarded": WARMUP_FRAMES,
        "face_detection_fraction": float(np.mean(detections)),
        "raw_valid_fraction": float(np.mean(valid)),
        "enhancement_applied_fraction": float(np.mean(applied)),
        "raw_frame_mean_luma": round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0].mean()), 4),
        "raw_face_brightness_mean_when_detected": _optional_mean(brightness),
        "raw_face_blur_mean_when_detected": _optional_mean(blur),
        "raw_face_quality_mean_including_misses": _optional_mean(quality),
        "perception_including_enhancement": _latency(perception_ms),
        "enhancement_within_perception": _latency(enhancement_ms),
        "simulated_duration_ms": last_tick.timestamp_ms,
        "degraded_timeout_elapsed": last_tick.timestamp_ms >= cfg.quality.degraded_after_ms,
        "final_face_valid": last_tick.face_valid,
        "final_risk_level": last_tick.risk_level.value,
        "final_reason_codes": last_tick.reason_codes,
        "sensor_degraded": last_tick.risk_level == RiskLevel.SENSOR_DEGRADED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Local face image; defaults to matplotlib's Grace Hopper sample")
    parser.add_argument("--out", type=Path, default=Path("outputs/night_camera_smoke.json"))
    parser.add_argument("--iterations", type=int, default=30, help="Measured frames per condition, after five warmups")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    source_path, image = _load_image(args.image)
    cfg = load_config(overrides={"models.phone_backend": "none"})
    model_path = resolve_path(cfg.models.face_landmarker).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Local face-landmarker model not found: {model_path}")
    report = {
        "status": "synthetic_exposure_smoke_test",
        "limitations": [
            "Repeated still image with numerical exposure scaling; no real night video or sensor noise simulation.",
            "Face detection and raw quality only; no drowsiness or early-warning accuracy measurement.",
            "Stage processing time excludes capture, frame queues, engine/alert work, and model initialization.",
            "Synthetic timestamps test sensor timeout only; they do not represent measured capture-to-alert latency.",
            "Global brightness gating may skip a dark face when surrounding dashboard or streetlights are bright.",
        ],
        "source_image": {"path": str(source_path), "sha256": _sha256(source_path)},
        "face_landmarker": {"path": str(model_path), "sha256": _sha256(model_path)},
        "image_resolution": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": version("numpy"), "opencv-python": version("opencv-python"),
            "mediapipe": version("mediapipe"),
        },
        "config": {"low_light": cfg.low_light.model_dump(), "quality": cfg.quality.model_dump(),
                   "phone_backend": "none"},
        "synthetic_frame_interval_ms": SIMULATED_FRAME_INTERVAL_MS,
        "conditions": [],
    }
    for exposure in EXPOSURES:
        for enabled in (False, True):
            condition = _run_condition(image, exposure, enabled, args.iterations)
            report["conditions"].append(condition)
            print(f"exposure={exposure:g} enhanced={enabled}: "
                  f"detected={condition['face_detection_fraction']:.2f} "
                  f"raw_valid={condition['raw_valid_fraction']:.2f} "
                  f"perception_p95_ms={condition['perception_including_enhancement']['p95_ms']:.2f} "
                  f"state={condition['final_risk_level']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Synthetic smoke report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
