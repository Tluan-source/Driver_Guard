"""Landmark geometry: EAR, MAR (MediaPipe Face Mesh 478-point topology).

EAR (Soukupová & Čech, 2016):  (|p2-p6| + |p3-p5|) / (2 |p1-p4|)
All distances are computed in PIXEL space (x*W, y*H) so the ratio is not distorted by a
non-square frame.
"""
from __future__ import annotations

import numpy as np

# (p1, p2, p3, p4, p5, p6) — p1/p4 = eye corners, (p2,p6) and (p3,p5) = upper/lower lid pairs.
# "RIGHT" / "LEFT" refer to the subject's own eyes (image-left eye = subject's right eye).
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Inner-lip vertical pairs and mouth corners.
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)

# Points used for the solvePnP head-pose fallback.
PNP_IDX = {"nose_tip": 1, "chin": 152, "left_eye_outer": 263, "right_eye_outer": 33,
           "mouth_left": 291, "mouth_right": 61}


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(pts: np.ndarray, idx: tuple[int, ...]) -> float | None:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horiz = _d(p1, p4)
    if horiz < 1e-6:
        return None
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * horiz)


def mouth_aspect_ratio(pts: np.ndarray) -> float | None:
    horiz = _d(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if horiz < 1e-6:
        return None
    vert = sum(_d(pts[a], pts[b]) for a, b in MOUTH_VERTICAL) / len(MOUTH_VERTICAL)
    return vert / horiz


def landmarks_to_pixels(norm_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    """(N,3) normalized landmarks -> pixel coords (z scaled by width like MediaPipe)."""
    out = norm_xyz.astype(np.float32).copy()
    out[:, 0] *= width
    out[:, 1] *= height
    out[:, 2] *= width
    return out


def bbox_from_landmarks(pts_px: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = pts_px[:, 0].min(), pts_px[:, 1].min()
    x1, y1 = pts_px[:, 0].max(), pts_px[:, 1].max()
    return float(x0), float(y0), float(x1), float(y1)
