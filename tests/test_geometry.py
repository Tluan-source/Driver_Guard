import numpy as np

from driverguard.perception.geometry import LEFT_EYE, MOUTH_CORNERS, MOUTH_VERTICAL, RIGHT_EYE, eye_aspect_ratio, mouth_aspect_ratio
from driverguard.perception.head_pose import euler_from_rotation, rotation_from_euler


def _eye(pts, idx, cx, cy, w, h):
    p1, p2, p3, p4, p5, p6 = idx
    pts[p1] = (cx - w / 2, cy, 0)
    pts[p4] = (cx + w / 2, cy, 0)
    pts[p2] = (cx - w / 6, cy - h / 2, 0)
    pts[p3] = (cx + w / 6, cy - h / 2, 0)
    pts[p6] = (cx - w / 6, cy + h / 2, 0)
    pts[p5] = (cx + w / 6, cy + h / 2, 0)


def test_ear_open_vs_closed():
    pts = np.zeros((478, 3), np.float32)
    _eye(pts, RIGHT_EYE, 100, 100, 30, 9)  # open: EAR = 9/30 = 0.30
    _eye(pts, LEFT_EYE, 160, 100, 30, 2)  # nearly closed
    assert abs(eye_aspect_ratio(pts, RIGHT_EYE) - 0.30) < 1e-3
    assert eye_aspect_ratio(pts, LEFT_EYE) < 0.1


def test_ear_degenerate_returns_none():
    pts = np.zeros((478, 3), np.float32)
    assert eye_aspect_ratio(pts, RIGHT_EYE) is None


def test_mar():
    pts = np.zeros((478, 3), np.float32)
    pts[MOUTH_CORNERS[0]] = (100, 200, 0)
    pts[MOUTH_CORNERS[1]] = (150, 200, 0)
    for a, b in MOUTH_VERTICAL:
        pts[a] = (125, 190, 0)
        pts[b] = (125, 210, 0)
    assert abs(mouth_aspect_ratio(pts) - 20 / 50) < 1e-6


def test_euler_roundtrip():
    for yaw, pitch, roll in [(0, 0, 0), (25, -10, 5), (-40, 15, -8), (10, -30, 0)]:
        R = rotation_from_euler(yaw, pitch, roll)
        y, p, r = euler_from_rotation(R)
        assert abs(y - yaw) < 1e-6 and abs(p - pitch) < 1e-6 and abs(r - roll) < 1e-6
