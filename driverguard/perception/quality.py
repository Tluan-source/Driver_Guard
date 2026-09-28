"""Face quality Q in [0,1] — a project-defined signal-reliability score, NOT an identity score.

Q = size_score * brightness_score * blur_score * pose_score
When Q < quality.min_quality the correct decision is "cannot assess" (SENSOR_DEGRADED after a
timeout), never "drowsy" and never "normal".
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityCfg


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def face_crop_stats(gray: np.ndarray, bbox_px: tuple[float, float, float, float]) -> tuple[float, float]:
    """Mean brightness and Laplacian variance (sharpness) of the face crop, scale-normalized."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = bbox_px
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0, 0.0
    crop = gray[y0:y1, x0:x1]
    crop = cv2.resize(crop, (96, max(8, int(96 * crop.shape[0] / crop.shape[1]))))
    return float(crop.mean()), float(cv2.Laplacian(crop, cv2.CV_64F).var())


def quality_score(face_width_ratio: float, brightness: float, blur: float,
                  yaw: float | None, pitch: float | None, cfg: QualityCfg) -> float:
    size_s = _ramp(face_width_ratio, cfg.min_face_width_ratio * 0.5, cfg.min_face_width_ratio)
    if brightness < cfg.brightness_min:
        bright_s = _ramp(brightness, cfg.brightness_min * 0.5, cfg.brightness_min)
    elif brightness > cfg.brightness_max:
        bright_s = 1.0 - _ramp(brightness, cfg.brightness_max, 255.0)
    else:
        bright_s = 1.0
    blur_s = _ramp(blur, cfg.min_blur_var * 0.5, cfg.min_blur_var)
    pose_s = 1.0
    if yaw is not None and pitch is not None:
        pose_s = (1.0 - _ramp(abs(yaw), 45.0, 75.0)) * (1.0 - _ramp(abs(pitch), 35.0, 60.0))
    return float(size_s * bright_s * blur_s * pose_s)
