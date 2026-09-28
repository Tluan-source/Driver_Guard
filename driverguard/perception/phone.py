"""Phone detection.

Default backend: MediaPipe Object Detector (EfficientDet-Lite0, COCO, class "cell phone").
Chosen over Ultralytics YOLO for the MVP because it is Apache-2.0 licensed and runs natively on
Android with the same model file. A YOLO-nano ONNX backend can be added behind the same
interface later — remember Ultralytics is AGPL-3.0 (see docs/02_data_plan.md, license gate).

Detection alone never means "phone use": the temporal PhoneTracker requires the phone to be
near the face (not on the dashboard mount) and to persist for >= phone.min_duration_ms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass
class PhoneObservation:
    confidence: float
    bbox_norm: tuple[float, float, float, float] | None  # x0,y0,x1,y1 in [0,1]


class PhoneDetectorLike(Protocol):
    def detect(self, image_bgr: np.ndarray, ts_ms: int) -> PhoneObservation: ...
    def close(self) -> None: ...


class NullPhoneDetector:
    def detect(self, image_bgr: np.ndarray, ts_ms: int) -> PhoneObservation:
        return PhoneObservation(0.0, None)

    def close(self) -> None:
        pass


class MediaPipePhoneDetector:
    def __init__(self, model_path: str | Path, score_threshold: float = 0.2):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found. Run: python scripts/download_models.py")
        self._mp = mp
        opts = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            max_results=3,
            score_threshold=score_threshold,
            category_allowlist=["cell phone"],
        )
        self._det = vision.ObjectDetector.create_from_options(opts)
        self._last_ts = -1

    def detect(self, image_bgr: np.ndarray, ts_ms: int) -> PhoneObservation:
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._det.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb),
                                         int(ts_ms))
        best, box = 0.0, None
        for d in res.detections:
            s = max((c.score for c in d.categories), default=0.0)
            if s > best:
                b = d.bounding_box
                best = float(s)
                box = (b.origin_x / w, b.origin_y / h, (b.origin_x + b.width) / w, (b.origin_y + b.height) / h)
        return PhoneObservation(best, box)

    def close(self) -> None:
        self._det.close()


def phone_near_face(phone_bbox, face_bbox, proximity: float) -> bool:
    """True if the phone box centre lies inside the face box expanded `proximity` times.

    Cheap heuristic that rejects a phone sitting on the dashboard mount (far from the face).
    TODO(phase 1): add hand landmarks (MediaPipe Hand Landmarker) for phone-in-hand evidence.
    """
    if phone_bbox is None or face_bbox is None or None in face_bbox:
        return False
    fx0, fy0, fx1, fy1 = face_bbox
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    hw, hh = (fx1 - fx0) / 2 * proximity, (fy1 - fy0) / 2 * proximity
    px, py = (phone_bbox[0] + phone_bbox[2]) / 2, (phone_bbox[1] + phone_bbox[3]) / 2
    return (cx - hw) <= px <= (cx + hw) and (cy - hh) <= py <= (cy + hh * 1.5)  # allow below chin
