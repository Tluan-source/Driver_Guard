"""Perception stage: one BGR frame -> FrameSignals.

No temporal state here except the phone-detector cadence (run every N frames and re-use the
last result for a short time). Keeping this stage stateless is what allows the Kaggle
notebook to extract features once and every later experiment to replay them.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import DriverGuardConfig, resolve_path
from ..schemas import FrameSignals
from .face import FaceLandmarkerLike, MediaPipeFaceLandmarker
from .geometry import LEFT_EYE, RIGHT_EYE, bbox_from_landmarks, eye_aspect_ratio, landmarks_to_pixels, mouth_aspect_ratio
from .head_pose import pose_from_landmarks_pnp, pose_from_transform
from .phone import MediaPipePhoneDetector, NullPhoneDetector, PhoneDetectorLike, PhoneObservation, phone_near_face
from .quality import face_crop_stats, quality_score


class PerceptionExtractor:
    def __init__(self, cfg: DriverGuardConfig, landmarker: FaceLandmarkerLike,
                 phone_detector: PhoneDetectorLike | None = None):
        self.cfg = cfg
        self.landmarker = landmarker
        self.phone = phone_detector or NullPhoneDetector()
        self._n = 0
        self._last_phone = PhoneObservation(0.0, None)
        self._last_phone_ts = -10**9

    @classmethod
    def from_config(cls, cfg: DriverGuardConfig) -> "PerceptionExtractor":
        lm = MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker))
        if cfg.models.phone_backend == "mediapipe":
            ph: PhoneDetectorLike = MediaPipePhoneDetector(resolve_path(cfg.models.phone_detector))
        else:
            ph = NullPhoneDetector()
        return cls(cfg, lm, ph)

    def process(self, image_bgr: np.ndarray | None, ts_ms: int, camera_ok: bool = True) -> FrameSignals:
        self._n += 1
        if image_bgr is None or not camera_ok:
            return FrameSignals(ts_ms=ts_ms, camera_ok=False)

        h, w = image_bgr.shape[:2]
        sig = FrameSignals(ts_ms=ts_ms, camera_ok=True)
        obs = self.landmarker.detect(image_bgr, ts_ms)

        face_bbox_norm = None
        if obs is not None:
            pts = landmarks_to_pixels(obs.landmarks_norm, w, h)
            x0, y0, x1, y1 = bbox_from_landmarks(pts)
            face_bbox_norm = (x0 / w, y0 / h, x1 / w, y1 / h)
            sig.face_detected = True
            sig.face_x0, sig.face_y0, sig.face_x1, sig.face_y1 = face_bbox_norm
            sig.ear_left = eye_aspect_ratio(pts, LEFT_EYE)
            sig.ear_right = eye_aspect_ratio(pts, RIGHT_EYE)
            sig.mar = mouth_aspect_ratio(pts)
            bs = obs.blendshapes
            sig.blink_bs_left = bs.get("eyeBlinkLeft")
            sig.blink_bs_right = bs.get("eyeBlinkRight")
            sig.jaw_open_bs = bs.get("jawOpen")
            pose = pose_from_transform(obs.transform) if obs.transform is not None else \
                pose_from_landmarks_pnp(pts, w, h)
            if pose is not None:
                sig.yaw, sig.pitch, sig.roll = pose
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            sig.brightness, sig.blur = face_crop_stats(gray, (x0, y0, x1, y1))
            sig.face_quality = quality_score((x1 - x0) / w, sig.brightness, sig.blur,
                                             sig.yaw, sig.pitch, self.cfg.quality)

        # Phone detector cadence: every N frames; hold the last result for up to 2N frames' time.
        n_every = self.cfg.runtime.phone_every_n_frames
        if self._n % n_every == 0:
            self._last_phone = self.phone.detect(image_bgr, ts_ms)
            self._last_phone_ts = ts_ms
        hold_ms = 2 * n_every * 1000.0 / max(self.cfg.runtime.target_fps, 1.0)
        if ts_ms - self._last_phone_ts <= hold_ms:
            sig.phone_conf = self._last_phone.confidence
            sig.phone_near_face = phone_near_face(self._last_phone.bbox_norm, face_bbox_norm,
                                                  self.cfg.phone.face_proximity)
        return sig

    def close(self) -> None:
        self.landmarker.close()
        self.phone.close()
