"""Synthetic low-light benchmark: darken real UTA validation videos, re-extract, re-score models.

Each validation alert/drowsy recording is decoded once; every sampled frame is darkened per
condition (pixel x exposure, optionally plus shot/read sensor noise) and fed to its own perception
extractor with the training preprocessing (quality gate on raw pixels, gamma/CLAHE only for
landmarks). Features go to data/features/uta_rldd_full_v002_lowlight/, metrics to --out.

This is NOT a night test: no NIR, no headlights/dashboard glare, no real sensor pipeline.
Validation was used during development, so absolute accuracies are optimistic; test is never opened.

    python -m scripts.evaluation.low_light_benchmark --workers 6 --out outputs/audits/<date>/low_light_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from driverguard.config import REPO_ROOT, DriverGuardConfig, resolve_path

FEATURE_SET = "uta_rldd_full_v002"
OUT_FEATURES = REPO_ROOT / "data" / "features" / "uta_rldd_full_v002_lowlight"
VIDEO_MANIFEST = REPO_ROOT / "data" / "manifests" / "uta_rldd_full" / "v002" / "manifest.json"
# (name, exposure, sensor noise)
CONDITIONS = (
    ("exp0.5", 0.5, False),
    ("exp0.3", 0.3, False),
    ("exp0.2", 0.2, False),
    ("exp0.12", 0.12, False),
    ("exp0.3_noise", 0.3, True),
    ("exp0.12_noise", 0.12, True),
)
MODELS = (
    "runs/gru_dann/uta_full_v002_20261009_window/dann_alert_drowsy_120s_seed42/inference/model.pt",
    "runs/gru_dann/uta_full_v002_20261009_window/gru_alert_drowsy_120s_seed42/inference/model.pt",
    "models/camera_cam02_multicue/full_seed42/model.pt",
    "models/camera_drowsiness_v1/model.pt",
)


def darken(image: np.ndarray, exposure: float, noise: bool, rng: np.random.Generator) -> np.ndarray:
    value = image.astype(np.float32) * exposure
    if noise:  # shot noise (variance ~ signal) + read noise (sigma 2 levels), 8-bit units
        value = value + rng.normal(0.0, 1.0, value.shape).astype(np.float32) * np.sqrt(value + 4.0)
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _extract(job: dict) -> list[dict]:
    import cv2

    from driverguard.capture import VideoFileSource
    from driverguard.learning.camera.features import CAMERA_FPS, CameraFeatureCollector
    from driverguard.perception.extractor import PerceptionExtractor
    from driverguard.perception.face import MediaPipeFaceLandmarker

    cv2.setNumThreads(1)
    cfg = DriverGuardConfig.model_validate(job["config"])
    clip = job["clip"]
    conditions = job["conditions"]
    todo = [c for c in conditions if not Path(job["out"][c[0]]).exists()]
    stats = {name: {"frames": 0, "face": 0, "brightness": []} for name, _, _ in todo}
    if todo:
        face_model = resolve_path(cfg.models.face_landmarker)
        extractors = {n: PerceptionExtractor(cfg, MediaPipeFaceLandmarker(face_model)) for n, _, _ in todo}
        collectors = {n: CameraFeatureCollector(cfg) for n, _, _ in todo}
        rng = np.random.default_rng(int(clip["sha256"][:8], 16))
        source = VideoFileSource(REPO_ROOT / clip["path"], target_fps=CAMERA_FPS,
                                 timestamps=job["timestamps"], max_ms=job["max_ms"])
        try:
            for frame in source.frames():
                original = frame.image_bgr
                if original is not None and job["max_edge"]:
                    height, width = original.shape[:2]
                    scale = min(1.0, job["max_edge"] / max(height, width))
                    if scale < 1:
                        original = cv2.resize(original, (round(width * scale), round(height * scale)),
                                              interpolation=cv2.INTER_AREA)
                for name, exposure, noise in todo:
                    image = None if original is None else darken(original, exposure, noise, rng)
                    sig = extractors[name].process(image, frame.ts_ms, frame.camera_ok)
                    collectors[name].update(sig)
                    s = stats[name]
                    s["frames"] += 1
                    s["face"] += bool(sig.face_detected)
                    if sig.brightness is not None:
                        s["brightness"].append(sig.brightness)
                if stats[todo[0][0]]["frames"] % 500 == 0:
                    print(f"{clip['clip_id']}: {frame.ts_ms/1000:.0f}s extracted", flush=True)
        finally:
            source.close()
            for e in extractors.values():
                e.close()
        for name, exposure, noise in todo:
            x, ts, valid = collectors[name].arrays()
            s = stats[name]
            temporary = Path(job["out"][name]).with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, x=x, ts_ms=ts, valid=valid,
                                label=np.asarray(clip["target"], np.int64), subject=np.asarray(clip["subject"]),
                                clip_id=np.asarray(f"{clip['clip_id']}#{name}"),
                                extraction_signature=np.asarray(job["signature"]),
                                exposure=np.asarray(exposure), sensor_noise=np.asarray(noise),
                                face_detected_fraction=np.asarray(s["face"] / max(s["frames"], 1)),
                                mean_raw_face_brightness=np.asarray(np.mean(s["brightness"]) if s["brightness"] else np.nan))
            temporary.replace(job["out"][name])
    records = []
    for name, exposure, noise in conditions:
        with np.load(job["out"][name], allow_pickle=False) as d:
            if str(d["extraction_signature"].item()) != job["signature"]:
                raise ValueError("Cached feature extraction signature differs")
            records.append({"clip_id": clip["clip_id"], "subject": clip["subject"], "target": clip["target"],
                            "condition": name, "exposure": exposure, "sensor_noise": noise,
                            "features": str(job["out"][name]), "valid_fraction": float(d["valid"].mean()),
                            "face_detected_fraction": float(d["face_detected_fraction"]),
                            "mean_raw_face_brightness": float(d["mean_raw_face_brightness"])})
    return records


def score(model_id: str, path: str, target_name: str, duration_seconds: int | None = None) -> dict:
    from driverguard.debug import model_lab

    model = model_lab.load_model(model_id)
    with np.load(path, allow_pickle=False) as d:
        x, ts, valid = d["x"], d["ts_ms"], d["valid"]
    if duration_seconds is not None:
        selected = ts <= duration_seconds * 1000
        x, ts, valid = x[selected], ts[selected], valid[selected]
    ctx = model.context
    ends = list(range(ctx, len(x) + 1, model_lab.DEFAULT_STRIDE))
    res = model.predict_batch([x[e - ctx:e] for e in ends], [ts[e - ctx:e] for e in ends],
                              [valid[e - ctx:e] for e in ends])
    task = model.class_names.index(target_name)
    preds = []
    for p, _, _ in res:
        if p is not None:
            preds.append(int(p[1] >= model.threshold) if model.threshold is not None else int(np.argmax(p)))
    return {"n": len(ends), "accepted": len(preds), "correct": sum(p == task for p in preds), "task": task}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--duration-seconds", type=int)
    ap.add_argument("--limit-recordings", type=int)
    ap.add_argument("--max-edge", type=int, default=0,
                    help="Resize received frames before every condition, including a newly extracted baseline")
    args = ap.parse_args()
    if args.duration_seconds is not None and args.duration_seconds < 120:
        ap.error("Duration must provide at least 120 seconds of real context")
    import torch
    torch.set_num_threads(2)
    started = time.time()
    feats = json.loads((REPO_ROOT / "data/features" / FEATURE_SET / "manifest.json").read_text(encoding="utf-8"))
    videos = {c["clip_id"]: c for c in json.loads(VIDEO_MANIFEST.read_text(encoding="utf-8"))["clips"]}
    contract = feats["input_contract"]
    cfg = {"quality": contract["quality"], "low_light": contract["low_light"], "models": {"phone_backend": "none"}}
    signature = json.dumps({**contract, "synthetic_low_light": "pixel*exposure (+shot/read noise)",
                            "duration_seconds": args.duration_seconds, "max_edge": args.max_edge},
                           sort_keys=True, separators=(",", ":"))
    classes = feats["class_names"]
    base = [r for r in feats["clips"] if r["split"] == "validation" and classes[r["label"]] in ("alert", "drowsy")]
    if args.limit_recordings is not None:
        base = base[:args.limit_recordings]
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    conditions = (("original", 1.0, False), *CONDITIONS) if args.max_edge else CONDITIONS
    jobs = []
    for r in base:
        clip = videos[r["clip_id"]]
        jobs.append({"clip": clip, "config": cfg, "signature": signature,
                     "conditions": conditions, "max_edge": args.max_edge,
                     "timestamps": r.get("timestamp_source", "index"),
                     "max_ms": args.duration_seconds * 1000 if args.duration_seconds else None,
                     "out": {n: str(args.feature_dir / f"{r['clip_id']}_{n}.npz") for n, _, _ in conditions}})
    records = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_extract, j) for j in jobs]
        for done, f in enumerate(as_completed(futures), 1):
            records.extend(f.result())
            print(f"extracted {done}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    for r in ([] if args.max_edge else base):  # resized runs re-extract their own original-light baseline
        rec = {"clip_id": r["clip_id"], "subject": r["subject"], "target": videos[r["clip_id"]]["target"],
               "condition": "original", "exposure": 1.0, "sensor_noise": False, "features": r["features"],
               "valid_fraction": r["valid_fraction"], "face_detected_fraction": None,
               "mean_raw_face_brightness": r.get("mean_raw_face_brightness")}
        if not Path(rec["features"]).is_file():
            rec["features"] = str(REPO_ROOT / "data/features" / FEATURE_SET / Path(r["features"]).name)
        if args.duration_seconds is not None:
            with np.load(rec["features"], allow_pickle=False) as data:
                selected = data["ts_ms"] <= args.duration_seconds * 1000
                rec["valid_fraction"] = float(data["valid"][selected].mean())
            rec["mean_raw_face_brightness"] = None  # original per-frame brightness not stored
        records.append(rec)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(
        {"source": FEATURE_SET, "split": "validation", "conditions": conditions, "clips": records,
         "max_edge": args.max_edge,
         "duration_seconds": args.duration_seconds, "limit_recordings": args.limit_recordings,
         "limitations": [__doc__.split("\n\n")[1]]}, ensure_ascii=False, indent=1), encoding="utf-8")

    order = ["original"] + [c[0] for c in CONDITIONS]
    table = []
    for cond in order:
        rows = [r for r in records if r["condition"] == cond]
        brightness = [r["mean_raw_face_brightness"] for r in rows
                      if r["mean_raw_face_brightness"] is not None and np.isfinite(r["mean_raw_face_brightness"])]
        entry = {"condition": cond, "exposure": rows[0]["exposure"], "sensor_noise": rows[0]["sensor_noise"],
                 "mean_raw_face_brightness": float(np.mean(brightness)) if brightness else None,
                 "face_detected_fraction": None if rows[0]["face_detected_fraction"] is None
                 else float(np.mean([r["face_detected_fraction"] for r in rows])),
                 "valid_frame_fraction": float(np.mean([r["valid_fraction"] for r in rows])), "models": {}}
        for model_id in MODELS:
            n = acc = cor = 0
            drowsy_n = drowsy_hit = 0
            for r in rows:
                s = score(model_id, r["features"], classes[r["target"]], args.duration_seconds)
                n, acc, cor = n + s["n"], acc + s["accepted"], cor + s["correct"]
                if s["task"] == 1:  # drowsy recording: a rejected window is a missed alert
                    drowsy_n += s["n"]
                    drowsy_hit += s["correct"]
            entry["models"][model_id] = {
                "coverage": acc / n if n else 0.0,
                "accuracy_on_accepted": cor / acc if acc else None,
                "drowsy_detected_of_all_windows": drowsy_hit / drowsy_n if drowsy_n else None,
            }
        table.append(entry)
        print(cond, json.dumps(entry["models"][MODELS[0]]), flush=True)
    payload = {"date": time.strftime("%Y-%m-%d"), "recordings": len(base),
               "max_edge": args.max_edge,
               "duration_seconds": args.duration_seconds,
               "scope": "prefix/subset validation benchmark" if args.duration_seconds or args.limit_recordings else "full validation recordings",
               "drivers": len({r["subject"] for r in base}), "quality_gate": contract["quality"],
               "low_light_preprocessing": contract["low_light"], "results": table,
               "caveat": "Làm tối nhân tạo trên video RGB ban ngày; không thay cho test ban đêm thật/NIR.",
               "elapsed_s": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
