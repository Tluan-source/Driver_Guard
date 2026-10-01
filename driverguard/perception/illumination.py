"""Bounded luma enhancement for underexposed RGB camera frames.

This can expose details already present in the image; it cannot restore missing
sensor information. Perception quality must still be measured on the original.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import LowLightCfg


class LowLightEnhancer:
    # Reject dim flat frames/noise using original 5th-to-95th percentile contrast.
    MIN_DYNAMIC_RANGE = 8.0
    MAX_LUMA_LIFT = 64

    def __init__(self, cfg: LowLightCfg):
        self.cfg = cfg
        self._gamma_lut = np.rint(
            255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** cfg.gamma
        ).astype(np.uint8)
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(8, 8))

    def prepare(self, image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return a detector input and whether enhancement was applied.

        Inputs must be nonempty uint8 BGR images. Bypassed frames retain object
        identity. All gates use raw luma: mean must be at least the configured
        minimum and below the trigger, with robust contrast of at least 8 levels.
        The output retains chroma, and luma can increase by at most 64 levels.
        There is no history, so preprocessing cannot use future frames.
        """
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy array")
        if image_bgr.dtype != np.uint8:
            raise TypeError("image_bgr must have dtype uint8")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) == 0:
            raise ValueError("image_bgr must be a nonempty HxWx3 BGR image")
        if not self.cfg.enabled:
            return image_bgr, False

        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        luma = image_ycrcb[:, :, 0]
        brightness = cv2.mean(luma)[0]
        if brightness < self.cfg.min_usable_brightness or brightness >= self.cfg.brightness_trigger:
            return image_bgr, False

        # A full histogram avoids periodic patterns aliasing onto a sampling stride.
        histogram = cv2.calcHist([luma], [0], None, [256], [0, 256]).reshape(-1)
        cumulative = np.cumsum(histogram, dtype=np.float64)
        low, high = np.searchsorted(cumulative, cumulative[-1] * np.array([0.05, 0.95]))
        if high - low < self.MIN_DYNAMIC_RANGE:
            return image_bgr, False

        gamma_luma = cv2.LUT(luma, self._gamma_lut)
        local_luma = self._clahe.apply(gamma_luma)
        enhanced = cv2.addWeighted(gamma_luma, 0.5, local_luma, 0.5, 0.0)
        max_channel = cv2.max(image_bgr[:, :, 0], cv2.max(image_bgr[:, :, 1], image_bgr[:, :, 2]))
        headroom = np.minimum(255 - max_channel, self.MAX_LUMA_LIFT)
        lift = np.minimum(cv2.subtract(enhanced, luma), headroom)
        # Equal channel offsets preserve chroma; headroom prevents uint8 overflow.
        return image_bgr + lift[:, :, None], True
