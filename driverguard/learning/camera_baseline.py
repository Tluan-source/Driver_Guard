"""CAM-01: subject-grouped development-only temporal logistic regression."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .camera_features import sha256_file
from .camera_model import CONTEXT, FEATURE_DIM, CameraPredictor
from .camera_statistics import STATISTIC_NAMES, STATISTICS_VERSION, temporal_statistics
from .camera_train import (
    _clip_rows,
    _prediction_rows,
    _row_metrics,
    _write_csv,
    camera_window_weights,
    camera_windows,
    load_camera_manifest,
    select_camera_threshold,
)
from .metrics import bootstrap_group_ci

CONFIGURATIONS = (0.01, 0.1, 1.0, 10.0)


def _estimator(regularization: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=regularization, max_iter=1500, random_state=seed)),
    ])


def _fit(model: Pipeline, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, scale__sample_weight=weights, classifier__sample_weight=weights)


def run_baseline(manifest_path: str | Path, out: str | Path, seed: int = 42, stride: int = 10) -> dict:
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty experiment directory")
    started = time.perf_counter()
    clips, manifest = load_camera_manifest(manifest_path, splits=("train", "validation"))
    windows = camera_windows(clips, stride)
    accepted = [window for window in windows if window.reason is None]
    x = np.asarray([
        temporal_statistics(clips[w.clip].x[w.end + 1 - CONTEXT:w.end + 1],
                            clips[w.clip].ts_ms[w.end + 1 - CONTEXT:w.end + 1],
                            clips[w.clip].valid[w.end + 1 - CONTEXT:w.end + 1]) for w in accepted
    ])
    y = np.asarray([clips[w.clip].label for w in accepted])
    groups = np.asarray([clips[w.clip].subject for w in accepted])
    partitions = {split: np.asarray([i for i, w in enumerate(accepted) if clips[w.clip].split == split])
                  for split in ("train", "validation")}
    train, validation = partitions["train"], partitions["validation"]
    if not len(train) or not len(validation) or any(set(y[index]) != {0, 1} for index in partitions.values()):
        raise ValueError("Both development splits need accepted windows from both classes")
    train_weights = camera_window_weights(clips, [accepted[i] for i in train])
    validation_weights = camera_window_weights(clips, [accepted[i] for i in validation])
    folds = min(5, len(set(groups[train])))
    if folds < 2:
        raise ValueError("CAM-01 needs at least two training subjects for grouped cross-validation")
    configurations, models = [], []
    with threadpool_limits(limits=4):
        for regularization in CONFIGURATIONS:
            fit_started = time.perf_counter()
            oof = np.empty(len(train))
            fold_subjects = []
            for fit_index, heldout_index in GroupKFold(folds).split(x[train], y[train], groups[train]):
                fit, heldout = train[fit_index], train[heldout_index]
                fold_weights = camera_window_weights(clips, [accepted[i] for i in fit])
                model = _estimator(regularization, seed)
                _fit(model, x[fit], y[fit], fold_weights)
                oof[heldout_index] = model.predict_proba(x[heldout])[:, 1]
                fold_subjects.append(sorted(set(groups[heldout])))
            model = _estimator(regularization, seed)
            _fit(model, x[train], y[train], train_weights)
            validation_scores = model.predict_proba(x[validation])[:, 1]
            record = {"id": f"CAM-01-logistic-C{regularization:g}", "C": regularization,
                      "train_grouped_oof_log_loss": float(log_loss(y[train], oof, sample_weight=train_weights)),
                      "validation_log_loss": float(log_loss(y[validation], validation_scores,
                                                             sample_weight=validation_weights)),
                      "fold_heldout_subjects": fold_subjects,
                      "fit_seconds": time.perf_counter() - fit_started}
            configurations.append(record)
            models.append(model)
            print(json.dumps(record), flush=True)
        # The independent validation set only selects the final decision threshold.
        best = min(range(len(models)), key=lambda i: configurations[i]["train_grouped_oof_log_loss"])
        model = models[best]
        scores = model.predict_proba(x)[:, 1]
        threshold = select_camera_threshold(y[validation], scores[validation], validation_weights)
    rows = _prediction_rows(clips, windows, scores)
    clip_rows = _clip_rows(clips, rows)
    metrics = {split: {"window": _row_metrics([row for row in rows if row["split"] == split], threshold),
                       "clip": _row_metrics([row for row in clip_rows if row["split"] == split], threshold)}
               for split in ("train", "validation")}
    scale, classifier = model["scale"], model["classifier"]
    checkpoint = {"format_version": 1, "architecture": "camera_temporal_logistic", "context": CONTEXT,
                  "feature_dim": FEATURE_DIM, "input_contract": manifest["input_contract"],
                  "decision_threshold": threshold, "statistic_names": list(STATISTIC_NAMES),
                  "statistics_version": STATISTICS_VERSION,
                  "center": torch.as_tensor(scale.mean_, dtype=torch.float64),
                  "scale": torch.as_tensor(scale.scale_, dtype=torch.float64),
                  "coefficient": torch.as_tensor(classifier.coef_[0], dtype=torch.float64),
                  "intercept": float(classifier.intercept_[0]), "task": "camera_current_state_weak_video_labels"}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    predictor = CameraPredictor(checkpoint_path)
    exported = np.asarray([
        predictor.predict_window(clips[w.clip].x[w.end + 1 - CONTEXT:w.end + 1],
                                 clips[w.clip].ts_ms[w.end + 1 - CONTEXT:w.end + 1],
                                 clips[w.clip].valid[w.end + 1 - CONTEXT:w.end + 1]).score for w in accepted
    ])
    parity_error = float(np.max(np.abs(exported - scores)))
    if parity_error > 1e-10:
        raise ValueError("Exported logistic scores differ from scikit-learn")
    _write_csv(output / "predictions.csv", rows)
    _write_csv(output / "clip_predictions.csv", clip_rows)
    validation_rows = [row for row in rows if row["split"] == "validation" and row["accepted"]]
    if len({row["subject"] for row in validation_rows}) >= 2:
        ci = bootstrap_group_ci([row["label"] for row in validation_rows],
                                [row["score"] for row in validation_rows],
                                [row["subject"] for row in validation_rows], threshold,
                                label_threshold=0.5, n_bootstrap=300, seed=seed)
        ci["scope"] = "selection_validation; conditional on chosen model/threshold; not final-test uncertainty"
    else:
        ci = {"status": "unavailable", "reason": "fewer_than_two_validation_subjects"}
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    report = {"run_id": "CAM-01", "task": checkpoint["task"], "test_evaluated": False,
              "seed": seed, "stride": stride, "context": CONTEXT, "configurations": configurations,
              "selection": "minimum training grouped-OOF subject/class/clip/window weighted log loss",
              "best_configuration": configurations[best]["id"], "decision_threshold": threshold,
              "threshold_selection": "validation subject/class/clip/window weighted balanced accuracy",
              "metrics": metrics, "validation_subject_bootstrap_ci": ci,
              "wall_seconds": time.perf_counter() - started, "export_max_absolute_score_error": parity_error,
              "git_sha_at_run": git_sha, "working_tree_changes_at_run": bool(dirty.strip()),
              "implementation_sha256": {
                  name: sha256_file(Path(__file__).parent / name) for name in (
                      "camera_baseline.py", "camera_statistics.py", "camera_model.py", "camera_train.py")},
              "feature_manifest_sha256": sha256_file(manifest_path),
              "source_manifest_sha256": manifest["input_contract"]["source_manifest_sha256"],
              "checkpoint_sha256": sha256_file(checkpoint_path),
              "input_contract": manifest["input_contract"],
              "subject_splits": {split: sorted({clip.subject for clip in clips if clip.split == split})
                                 for split in ("train", "validation")},
              "limitations": ["Development validation, not an independent final test.",
                              "Weak recording labels; no EEG onset or real-night cohort.",
                              "Closure/open-mouth events are 10 Hz proxies, not validated blink kinetics.",
                              "One deterministic logistic seed; neural candidates need separate seed checks."]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "configurations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "C", "train_grouped_oof_log_loss",
                                                   "validation_log_loss", "fit_seconds"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in configurations)
    print(json.dumps({"best_configuration": report["best_configuration"], "threshold": threshold,
                      "validation_window": metrics["validation"]["window"]["overall_accepted_only"],
                      "test_evaluated": False}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    run_baseline(args.manifest, args.out, args.seed, args.stride)


if __name__ == "__main__":
    main()
