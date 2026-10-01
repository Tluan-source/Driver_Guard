"""Commands for training, inference and frozen-split EEG evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import FEATURE_KEY, dataset_manifest, load_sessions, pack_sessions, read_feature_contract
from .metrics import bootstrap_group_ci, grouped_metrics
from .model import VigilancePredictor
from .train import train, write_json, write_predictions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="driverguard-model")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="train EEG vigilance candidates and freeze validation winner")
    p.add_argument("--data", default="data/features/fatigue_eeg")
    p.add_argument("--out", default="models/eeg_vigilance_v2")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--feature-key", choices=["de_movingAve"], default=FEATURE_KEY)
    p.add_argument("--feature-transform", choices=["absolute", "relative_log_power"], default=None)
    p = sub.add_parser("improve", help="quality-aware ensemble, nested subject LOSO and deployment checkpoint")
    p.add_argument("--baseline-data", default="data/features/fatigue_eeg")
    p.add_argument("--data", default="data/features/fatigue_eeg_robust_v1")
    p.add_argument("--out", default="models/eeg_vigilance_v3_1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=1000)
    p = sub.add_parser("predict", help="EEG feature NPZ/NPY/MAT or original CNT -> metadata CSV")
    p.add_argument("--checkpoint", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", help="features with exactly the checkpoint's channel/band order")
    source.add_argument("--cnt", help="original driver-fatigue Neuroscan CNT recording")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate", help="evaluate checkpoint on its frozen test subjects")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/features/fatigue_eeg")
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=300)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "train":
        train(args.data, args.out, epochs=args.epochs, context=args.context, seed=args.seed,
              feature_key=args.feature_key, bootstrap=args.bootstrap, threads=args.threads,
              feature_transform=args.feature_transform)
        return 0
    if args.command == "improve":
        from .improvement import improve

        improve(args.baseline_data, args.data, args.out, seed=args.seed,
                bootstrap=args.bootstrap, threads=args.threads)
        return 0
    predictor = VigilancePredictor(args.checkpoint)
    output = Path(args.out)
    if args.command == "predict":
        metadata = predictor.meta.get("dataset_metadata", {})
        preprocessing = predictor.preprocessing
        if args.cnt:
            from .eeg import extract_cnt, extract_cnt_quality

            if metadata.get("target") != "fatigue" or metadata.get("n_features") != 150:
                raise ValueError("CNT inference requires the original driver-fatigue spectral feature contract")
            if preprocessing == "robust_v1":
                from .quality import ROBUST_PREPROCESSING

                if metadata.get("preprocessing_config") != ROBUST_PREPROCESSING:
                    raise ValueError("Checkpoint quality configuration differs from the CNT extractor")
                x, ends, channels, quality = extract_cnt_quality(args.cnt, metadata["window_seconds"])
                valid = quality["quality_valid"]
                bits = ROBUST_PREPROCESSING["window_reason_bits"]
                reasons = [",".join(name for name, bit in bits.items() if int(code) & bit)
                           for code in quality["quality_reason_code"]]
            elif preprocessing == "legacy_car":
                x, ends, channels = extract_cnt(args.cnt, metadata["window_seconds"])
                valid, reasons = np.ones(len(x), dtype=bool), [""] * len(x)
            else:
                raise ValueError("Unsupported CNT preprocessing contract")
            if tuple(channels) != tuple(predictor.channel_names or ()):
                raise ValueError("CNT channel order differs from the checkpoint")
            pred = predictor.predict_features(x, valid)
        else:
            x, valid, reasons, ends = read_feature_contract(
                args.features, predictor.feature_key, predictor.n_features,
                expected_preprocessing=preprocessing,
                expected_channels=predictor.channel_names)
            pred = predictor.predict_features(x, valid)
            if ends is None:
                seconds = metadata.get("window_seconds")
                ends = np.arange(1, len(pred) + 1) * seconds if seconds else [None] * len(pred)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["window_index", "window_end_seconds", "prediction", "reduced_vigilance_pred",
                             "quality_valid", "sensor_status", "abstention_reason"])
            for index, (end, score, accepted, reason) in enumerate(zip(ends, pred, valid, reasons)):
                writer.writerow([index, end, float(score) if accepted else None,
                                 int(score >= predictor.meta["decision_threshold"]) if accepted else None,
                                 int(accepted), "EEG_VALID" if accepted else "SENSOR_DEGRADED", reason])
        print(json.dumps({"output": str(output), "n_windows": len(pred),
                          "n_accepted": int(np.sum(valid)), "coverage": float(np.mean(valid)),
                          "mean_prediction": float(np.mean(pred[valid])) if np.any(valid) else None,
                          "target": predictor.meta["target"],
                          "decision_threshold": predictor.meta["decision_threshold"]}))
        return 0
    if "split" not in predictor.meta or not predictor.meta["split"].get("test"):
        raise ValueError("This deployment model uses the full development cohort and has no frozen test split. "
                         "Use robust_nested_loso_predictions.csv for development metrics; "
                         "use predict on new independent EEG recordings for external testing.")
    sessions = load_sessions(args.data, predictor.feature_key)
    subjects = predictor.meta["split"]["test"]
    if set(subjects) != {s.subject for s in sessions if s.subject in subjects}:
        raise ValueError("Missing held-out test subjects; evaluation would not match frozen protocol")
    expected = {s["session_id"]: s for s in predictor.meta["dataset_manifest"] if s["subject"] in subjects}
    actual = {s["session_id"]: s for s in dataset_manifest(sessions) if s["subject"] in subjects}
    if expected != actual:
        raise ValueError("Test sessions or content changed since checkpoint training")
    x, y, groups, recordings, indices = pack_sessions(sessions, subjects, predictor.context)
    pred = predictor.predict_windows(x)
    threshold = predictor.meta["decision_threshold"]
    label_threshold = predictor.meta["label_threshold"]
    result = grouped_metrics(y, pred, groups, decision_threshold=threshold, label_threshold=label_threshold)
    result["confidence_intervals"] = bootstrap_group_ci(y, pred, groups, decision_threshold=threshold,
                                                         n_bootstrap=args.bootstrap, seed=predictor.meta["seed"],
                                                         label_threshold=label_threshold)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    write_predictions(output / "predictions.csv", y, pred, groups, recordings, indices, threshold)
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
