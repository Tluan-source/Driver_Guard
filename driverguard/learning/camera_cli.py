"""Camera-only checkpoint inference with the existing fast safety rules.

Run: python -m driverguard.learning.camera_cli run --checkpoint MODEL --source 0 --show
Predict: python -m driverguard.learning.camera_cli predict --checkpoint MODEL --features CLIP.npz --out scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..alerts import ConsoleAlertSink, SoundAlertSink
from ..capture import open_source
from ..config import DriverGuardConfig, resolve_path
from ..engine import DriverGuardEngine
from ..perception.extractor import PerceptionExtractor
from ..perception.face import MediaPipeFaceLandmarker
from ..privacy.guard import assert_metadata_only
from ..risk.evidence import Candidate, Evidence
from ..risk.state_machine import AlertPolicy, RiskStateMachine
from ..schemas import FrameSignals, Reason, RiskLevel
from .camera_features import CameraFrameBuffer, sha256_file
from .camera_model import CONTEXT, CameraPrediction, CameraPredictor


def checkpoint_config(predictor: CameraPredictor) -> DriverGuardConfig:
    cfg = DriverGuardConfig.model_validate({
        "quality": predictor.input_contract["quality"],
        "low_light": predictor.input_contract["low_light"],
        "models": {"phone_backend": "none"},
        "runtime": {"target_fps": 10},
    })
    if sha256_file(resolve_path(cfg.models.face_landmarker)) != predictor.input_contract["face_model_sha256"]:
        raise ValueError("Face Landmarker differs from the model's training asset")
    return cfg


class CameraRuntime:
    def __init__(self, predictor: CameraPredictor, cfg: DriverGuardConfig, *, independent_model_stream: bool = False):
        self.predictor, self.cfg = predictor, cfg
        self.independent_model_stream = independent_model_stream
        self._last_input_ts: int | None = None
        self.buffer = CameraFrameBuffer(cfg, predictor.context)
        self.engine = DriverGuardEngine(cfg)
        threshold = predictor.decision_threshold * 100
        policy_cfg = cfg.model_copy(deep=True)
        policy_cfg.risk = policy_cfg.risk.model_copy(update={
            "warning_enter": max(threshold, 0.000001), "caution_enter": max(threshold, 0.000001),
            "warning_exit": max(0.0000005, threshold * 0.8), "caution_exit": max(0.0000005, threshold * 0.8),
        })
        self.policy_cfg = policy_cfg
        self.learned_sm = RiskStateMachine(policy_cfg)
        self.alerts = AlertPolicy(cfg, self.engine.alerts.explain)
        self.last_prediction = CameraPrediction(None, "insufficient_history", 0.0)
        self.score_timestamp_ms: int | None = None

    def step(self, signals: FrameSignals, model_signals: FrameSignals | None = None) -> dict:
        tick = self.engine.step(signals)
        timestamp_invalid = signals.ts_ms < 0 or (
            self._last_input_ts is not None and signals.ts_ms <= self._last_input_ts
        )
        if timestamp_invalid or (
            self._last_input_ts is not None and signals.ts_ms - self._last_input_ts > 300
        ):
            self.buffer.clear()
        self._last_input_ts = signals.ts_ms if signals.ts_ms >= 0 else None
        # Fast tracking must never supply learned features, including after a reset.
        head_signals = model_signals if self.independent_model_stream else (model_signals or signals)
        if not timestamp_invalid and head_signals is not None and self.buffer.update(head_signals):
            self.last_prediction = self.predictor.predict_window(*self.buffer.arrays())
            self.score_timestamp_ms = signals.ts_ms if self.last_prediction.score is not None else None
        if len(self.buffer.arrays()[0]) == 0:
            self.last_prediction = CameraPrediction(None, "insufficient_history", 0.0)
            self.score_timestamp_ms = None
        if not tick.face_valid:
            self.last_prediction = CameraPrediction(None, "current_frame_invalid", self.last_prediction.valid_fraction)
            self.score_timestamp_ms = None
        prediction = self.last_prediction
        if prediction.score is None:
            self.learned_sm = RiskStateMachine(self.policy_cfg)
        candidates = []
        if prediction.score is not None and prediction.score >= self.predictor.decision_threshold:
            candidates.append(Candidate(RiskLevel.WARNING, Reason.RISK_SCORE, False))
        learned_level, learned_reasons = self.learned_sm.update(
            signals.ts_ms, Evidence({}, 100 * (prediction.score or 0.0), candidates),
            tick.risk_level == RiskLevel.SENSOR_DEGRADED, Reason.LOW_FACE_QUALITY,
            prediction.score is None,
        )
        level, reasons = tick.risk_level, list(tick.reason_codes)
        if tick.face_valid and prediction.score is not None and learned_level.severity > level.severity:
            level = learned_level
            reasons = list(dict.fromkeys(learned_reasons + reasons))
        score = max(tick.risk_score, 100 * (prediction.score or 0.0))
        alert = self.alerts.update(signals.ts_ms, level, reasons, score)
        result = {
            "timestamp_ms": signals.ts_ms, "camera_drowsiness_score": prediction.score,
            "score_timestamp_ms": self.score_timestamp_ms,
            "decision_threshold": self.predictor.decision_threshold,
            "model_status": prediction.reason or "assessed", "window_valid_fraction": prediction.valid_fraction,
            "face_valid": tick.face_valid, "face_quality": signals.face_quality,
            "assessment_status": "assessed" if prediction.score is not None else "cannot_assess",
            "raw_face_brightness": signals.brightness, "raw_face_blur": signals.blur,
            "ear_left": signals.ear_left, "ear_right": signals.ear_right, "mar": signals.mar,
            "rule_risk_level": tick.risk_level.value, "risk_level": level.value,
            "reason_codes": reasons, "alert": alert.to_dict() if alert else None,
            "early_warning_trained": False,
        }
        assert_metadata_only(result)
        return result


class _PreviewLandmarker:
    def __init__(self, model_path: Path):
        self.base = MediaPipeFaceLandmarker(model_path)
        self.observation = None

    def detect(self, image, ts):
        self.observation = self.base.detect(image, ts)
        return self.observation

    def close(self):
        self.base.close()


def _preview(frame, observation, result) -> None:
    image = frame.copy()
    h, w = image.shape[:2]
    if observation is not None:
        for index in (1, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14):
            point = observation.landmarks_norm[index]
            cv2.circle(image, (int(point[0] * w), int(point[1] * h)), 2, (80, 240, 120), -1)
    score = result["camera_drowsiness_score"]
    state = result["risk_level"].upper()
    if score is None and state == "NORMAL":
        state = "CANNOT ASSESS"
    lines = [f"{state}  model={score:.3f}" if score is not None else state,
             f"{result['model_status']}  valid={result['window_valid_fraction']:.0%}",
             f"raw quality={result['face_quality']:.2f}  {result.get('processing_ms', 0):.1f} ms"]
    for index, text in enumerate(lines):
        y = 24 + index * 24
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.imshow("DriverGuard camera checkpoint (local)", image)


def run_camera(args) -> None:
    predictor = CameraPredictor(args.checkpoint)
    cfg = checkpoint_config(predictor)
    runtime = CameraRuntime(predictor, cfg, independent_model_stream=True)
    landmarker = _PreviewLandmarker(resolve_path(cfg.models.face_landmarker))
    extractor = PerceptionExtractor(cfg, landmarker)
    # Independent tracking at the training cadence; fast safety rules keep native cadence.
    head_extractor = PerceptionExtractor(cfg, MediaPipeFaceLandmarker(resolve_path(cfg.models.face_landmarker)))
    source = open_source(args.source, target_fps=30, realtime=args.realtime)
    sound = SoundAlertSink(enabled=not args.mute)
    console = ConsoleAlertSink()
    output = None
    durations = []
    n_frames, n_scores = 0, 0
    head_bin = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        output = args.jsonl.open("w", encoding="utf-8")
    try:
        for frame in source.frames():
            started = time.perf_counter()
            signals = extractor.process(frame.image_bgr, frame.ts_ms, frame.camera_ok)
            head_signals = None
            if head_bin != frame.ts_ms // 100:
                head_bin = frame.ts_ms // 100
                head_signals = head_extractor.process(frame.image_bgr, frame.ts_ms, frame.camera_ok)
            result = runtime.step(signals, head_signals)
            result["processing_ms"] = (time.perf_counter() - started) * 1000
            durations.append(result["processing_ms"])
            n_frames += 1
            n_scores += result["camera_drowsiness_score"] is not None
            if output:
                output.write(json.dumps(result, allow_nan=False) + "\n")
            if result["alert"]:
                from ..schemas import AlertEvent

                value = result["alert"]
                event = AlertEvent(value["ts_ms"], RiskLevel(value["level"]), value["reasons"],
                                   value["message"], value["score"])
                console.on_alert(event)
                sound.on_alert(event)
            if n_frames % 30 == 0:
                print(json.dumps(result, ensure_ascii=True), flush=True)
            if args.show and frame.image_bgr is not None:
                _preview(frame.image_bgr, landmarker.observation, result)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.max_frames is not None and n_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        extractor.close()
        head_extractor.close()
        if output:
            output.close()
        if args.show:
            cv2.destroyAllWindows()
    print(json.dumps({"frames": n_frames, "frames_with_score": n_scores,
                      "processing_p50_ms": float(np.median(durations)) if durations else None,
                      "processing_p95_ms": float(np.percentile(durations, 95)) if durations else None,
                      "latency_scope": "perception+model+rules; excludes capture/queues/alert sinks/preview",
                      "early_warning_trained": False}), flush=True)


def predict_features(args) -> None:
    predictor = CameraPredictor(args.checkpoint)
    with np.load(args.features, allow_pickle=False) as data:
        x, timestamps, valid = data["x"], data["ts_ms"], data["valid"]
        subject, clip_id = str(data["subject"].item()), str(data["clip_id"].item())
        if "extraction_signature" in data:
            expected = json.dumps(predictor.input_contract, sort_keys=True, separators=(",", ":"))
            if str(data["extraction_signature"].item()) != expected:
                raise ValueError("Feature preprocessing differs from the camera checkpoint")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject", "clip_id", "end_ms", "score", "valid_fraction", "reason"])
        writer.writeheader()
        for end in range(CONTEXT, len(x) + 1, args.stride):
            prediction = predictor.predict_window(x[end-CONTEXT:end], timestamps[end-CONTEXT:end], valid[end-CONTEXT:end])
            writer.writerow({"subject": subject, "clip_id": clip_id, "end_ms": int(timestamps[end-1]),
                             "score": prediction.score, "valid_fraction": prediction.valid_fraction,
                             "reason": prediction.reason})
    print(f"Camera-only predictions: {args.out.resolve()}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--source", default="0")
    run.add_argument("--show", action="store_true")
    run.add_argument("--realtime", action="store_true")
    run.add_argument("--mute", action="store_true")
    run.add_argument("--max-frames", type=int)
    run.add_argument("--jsonl", type=Path)
    run.set_defaults(func=run_camera)
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--features", required=True, type=Path)
    predict.add_argument("--out", required=True, type=Path)
    predict.add_argument("--stride", type=int, default=10)
    predict.set_defaults(func=predict_features)
    args = parser.parse_args()
    if getattr(args, "stride", 1) < 1 or (getattr(args, "max_frames", None) is not None and args.max_frames < 1):
        parser.error("stride/max-frames must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
