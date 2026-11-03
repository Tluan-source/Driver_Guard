"""Controlled local capture failure/reset/model switch checks; frames stay in RAM."""
import json
from pathlib import Path

import torch

from driverguard.capture.source import CameraSource
from driverguard.config import resolve_path
from driverguard.learning.camera.cli import CameraRuntime, checkpoint_config, load_runtime_predictor
from driverguard.perception.extractor import PerceptionExtractor
from driverguard.perception.face import MediaPipeFaceLandmarker
from driverguard.schemas import FrameSignals


def main():
    torch.set_num_threads(4)
    source = CameraSource()
    ext = None
    results = []
    try:
        frame = next(source.frames())
        assert frame.camera_ok
        for mode in ("gru", "dann"):
            path = Path(f"models/from_kaggle/week5_gru_dann/fold1_{mode}_seed42/inference/model.pt")
            predictor = load_runtime_predictor(path)
            cfg = checkpoint_config(predictor)
            if ext is None:
                ext = PerceptionExtractor(cfg, MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker)))
            signals = ext.process(frame.image_bgr, frame.ts_ms + len(results)*100, True)
            runtime = CameraRuntime(predictor, cfg)
            result = runtime.step(signals)
            assert result["camera_drowsiness_score"] is None
            assert len(runtime.buffer.arrays()[0]) == 1
            reset_runtime = CameraRuntime(load_runtime_predictor(path), cfg)
            assert len(reset_runtime.buffer.arrays()[0]) == 0
            results.append({"mode": mode, "new_model_requires_warmup": True, "new_runtime_history_empty": True})
        source.close()
        ok, image = source.cap.read()
        assert not ok and image is None
        failed = runtime.step(FrameSignals(ts_ms=signals.ts_ms+100, camera_ok=ok))
        assert failed["camera_drowsiness_score"] is None and not failed["face_valid"]
        assert failed["assessment_status"] == "cannot_assess"
        report = {"status": "passed", "camera_read_failure_after_release": True,
                  "score_suppressed_on_camera_loss": True, "model_switch_checks": results,
                  "scope": "real webcam frame and software-released camera; physical USB unplug not tested",
                  "raw_images_saved": False}
        Path("outputs/week5_webcam_benchmark/controls.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        source.close()
        if ext is not None:
            ext.close()


if __name__ == "__main__":
    main()
