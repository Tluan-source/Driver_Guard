"""Measure live camera capture + perception + learned runtime; save metadata only."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from driverguard.capture.source import CameraSource
from driverguard.config import resolve_path
from driverguard.learning.camera.cli import CameraRuntime, checkpoint_config, load_runtime_predictor
from driverguard.perception.extractor import PerceptionExtractor
from driverguard.perception.face import MediaPipeFaceLandmarker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=150)
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path("outputs/week5_webcam_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("models/from_kaggle/week5_gru_dann/fold1_gru_seed42/inference/model.pt")
    predictor = load_runtime_predictor(checkpoint)
    cfg = checkpoint_config(predictor)
    runtime = CameraRuntime(predictor, cfg, independent_model_stream=True)
    fast = PerceptionExtractor(cfg, MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker)))
    head = PerceptionExtractor(cfg, MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker)))
    source = CameraSource()
    rows, process_ms = [], []
    start = time.perf_counter()
    last_bin = None
    try:
        for frame in source.frames():
            after_capture = time.perf_counter()
            sig = fast.process(frame.image_bgr, frame.ts_ms, frame.camera_ok)
            hs = None
            if last_bin != frame.ts_ms // 100:
                last_bin = frame.ts_ms // 100
                hs = head.process(frame.image_bgr, frame.ts_ms, frame.camera_ok)
            result = runtime.step(sig, hs)
            end = time.perf_counter()
            process_ms.append((end-after_capture)*1000)
            rows.append({**result, "elapsed_s": end-start, "processing_ms": process_ms[-1]})
            if len(rows) % 300 == 0:
                print(f"frames={len(rows)} seconds={end-start:.1f} face_valid={result['face_valid']} status={result['model_status']}", flush=True)
            if end-start >= args.seconds:
                break
    finally:
        source.close()
        fast.close()
        head.close()
    elapsed = time.perf_counter()-start
    times = np.array([r["timestamp_ms"] for r in rows])
    gaps = np.diff(times)
    report = {"host": platform.platform(), "checkpoint": str(checkpoint), "seconds": elapsed, "frames": len(rows),
              "processed_fps": len(rows)/elapsed, "processing_ms_p50": float(np.percentile(process_ms, 50)),
              "processing_ms_p95": float(np.percentile(process_ms, 95)),
              "face_valid_fraction": sum(r["face_valid"] for r in rows)/len(rows),
              "frames_with_model_score": sum(r["camera_drowsiness_score"] is not None for r in rows),
              "first_score_s": next((r["elapsed_s"] for r in rows if r["camera_drowsiness_score"] is not None), None),
              "timestamp_gaps_over_300ms": int((gaps > 300).sum()),
              "latency_scope": "processing after OpenCV read; sensor exposure and internal driver queue age unknown",
              "raw_images_saved": False, "accuracy_evaluated": False,
              "limitations": ["camera frame-drop count unavailable without sensor sequence numbers", "no labeled events: false alerts/hour not established", "physical camera loss and model switching require separate controlled checks"]}
    (out / "summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (out / "metadata.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
