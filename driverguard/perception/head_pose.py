"""Head pose (yaw / pitch / roll in degrees).

Primary: MediaPipe facial transformation matrix (4x4). Fallback: OpenCV solvePnP.

Sign convention is camera-dependent; DriverGuard never uses absolute angles for decisions —
the engine subtracts the driver's calibrated neutral pose (see calibration.PoseBaseline).
TODO(week 3): verify on webcam that turning head to the driver's left gives yaw > 0 and
looking down gives pitch < 0; flip signs in `pose_from_transform` if not.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import PNP_IDX


def euler_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX decomposition -> (yaw, pitch, roll) degrees. yaw=Y axis, pitch=X axis, roll=Z axis."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def rotation_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of euler_from_rotation (used in tests / synthetic data)."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_from_transform(matrix: np.ndarray) -> tuple[float, float, float]:
    M = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return euler_from_rotation(M[:3, :3])


# Generic 3D face model (mm) for the PnP fallback — rough, only for relative angles.
_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -63.6, -12.5),  # chin
        (43.3, 32.7, -26.0),  # left eye outer corner
        (-43.3, 32.7, -26.0),  # right eye outer corner
        (28.9, -28.9, -24.1),  # mouth left
        (-28.9, -28.9, -24.1),  # mouth right
    ],
    dtype=np.float64,
)


def pose_from_landmarks_pnp(pts_px: np.ndarray, width: int, height: int) -> tuple[float, float, float] | None:
    img_pts = np.array(
        [pts_px[PNP_IDX[k], :2] for k in
         ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right")],
        dtype=np.float64,
    )
    f = float(width)
    cam = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img_pts, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return euler_from_rotation(R)
