"""Download the on-device models into ./models (run on your own machine — needs internet).

    python scripts/download_models.py

Models
  face_landmarker.task      MediaPipe Face Landmarker (478 landmarks + 52 blendshapes + transform)
  efficientdet_lite0.tflite MediaPipe Object Detector, COCO (class "cell phone") — Apache-2.0

If a URL changes, see https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
and https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "efficientdet_lite0.tflite":
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
}

ROOT = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    ROOT.mkdir(exist_ok=True)
    ok = True
    for name, url in MODELS.items():
        dst = ROOT / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {name} already present ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[get ] {name} <- {url}")
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            sha = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
            print(f"       ok {dst.stat().st_size / 1e6:.1f} MB sha256:{sha}…")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"       FAILED: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
