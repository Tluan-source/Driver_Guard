"""MediaPipe Face Landmarker wrapper (Tasks API, VIDEO running mode).

VIDEO mode is synchronous and deterministic: the same video always gives the same
landmarks, which is what offline evaluation needs. For live camera we also use VIDEO mode
and simply read the newest frame each loop (frames that arrive while we are busy are
dropped by the camera buffer) — temporal logic is timestamp-based so drops are safe.

Model file: models/face_landmarker.task  (download: `python scripts/download_models.py`)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass
class FaceObservation:
    landmarks_norm: np.ndarray  # (478, 3) normalized x,y in [0,1], z relative
    blendshapes: dict[str, float] = field(default_factory=dict)
    transform: np.ndarray | None = None  # (4,4) facial transformation matrix


class FaceLandmarkerLike(Protocol):
    def detect(self, image_bgr: np.ndarray, ts_ms: int) -> FaceObservation | None: ...
    def close(self) -> None: ...


class MediaPipeFaceLandmarker:
    def __init__(self, model_path: str | Path, num_faces: int = 1,
                 min_detection_conf: float = 0.5, min_presence_conf: float = 0.5,
                 min_tracking_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run: python scripts/download_models.py"
            )
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, image_bgr: np.ndarray, ts_ms: int) -> FaceObservation | None:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(mp_image, int(ts_ms))
        if not res.face_landmarks:
            return None
        # Single-driver ROI assumption: pick the largest face (closest to camera).
        best = 0
        if len(res.face_landmarks) > 1:
            areas = []
            for lms in res.face_landmarks:
                xs = [p.x for p in lms]
                ys = [p.y for p in lms]
                areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
            best = int(np.argmax(areas))
        lms = res.face_landmarks[best]
        pts = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
        bs = {}
        if res.face_blendshapes:
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[best]}
        tf = None
        if res.facial_transformation_matrixes:
            tf = np.asarray(res.facial_transformation_matrixes[best], dtype=np.float64)
        return FaceObservation(landmarks_norm=pts, blendshapes=bs, transform=tf)

    def close(self) -> None:
        self._lm.close()
