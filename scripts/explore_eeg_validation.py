"""Reproduce the 119-candidate EEG development grid using train/validation only.

The original experiment used the v1 split and wrote runs/eeg_validation_experiment.json.
This script preserves its float64 feature representations and fixed grids. Production
checkpoint transforms use float32, so very small numerical differences are expected.
No test features, targets, predictions or metrics are read by this script.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from driverguard.learning.metrics import grouped_metrics, select_threshold


def read_split(path: Path) -> dict[str, list[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for stage in ("train", "validation", "test"):
        subjects = split.get(stage)
        if (not isinstance(subjects, list) or not subjects
                or not all(isinstance(subject, str) and subject for subject in subjects)
                or len(set(subjects)) != len(subjects)):
            raise ValueError(f"Split {stage!r} must contain unique, nonempty subject identifiers")
        if seen.intersection(subjects):
            raise ValueError("Train, validation and test subject identifiers must be disjoint")
        seen.update(subjects)
    return split


def read_allowed(data: Path, split: dict[str, list[str]], stage: str):
    if stage not in {"train", "validation"}:
        raise ValueError("This development script only permits train and validation data")
    data = data.resolve()
    rows = json.loads((data / "sessions.json").read_text(encoding="utf-8"))
    subjects = set(split[stage])
    held_out_paths = {(data / row["path"]).resolve() for row in rows if row["subject"] in split["test"]}
    selected = [row for row in rows if row["subject"] in subjects]
    if {row["subject"] for row in selected} != subjects:
        raise ValueError(f"Missing {stage} subjects in the session manifest")
    features, labels, groups, paths = [], [], [], []
    opened = set()
    for row in selected:
        if row["subject"] in split["test"]:
            raise ValueError("Held-out subject reached the development loader")
        path = (data / row["path"]).resolve()
        if path in held_out_paths:
            raise ValueError("An allowed session cannot reference a held-out subject file")
        if not path.is_relative_to(data) or path in opened:
            raise ValueError("Session paths must be unique and stay inside the data directory")
        opened.add(path)
        with np.load(path, allow_pickle=False) as archive:
            x = np.asarray(archive["x"], dtype=np.float64)
            y = np.asarray(archive["y"]).reshape(-1)
        if (x.ndim != 2 or x.shape[1] != 150 or not len(x)
                or len(y) != len(x) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()):
            raise ValueError(f"Expected finite [time,150] EEG with aligned binary labels: {path}")
        features.append(x)
        labels.append(y.astype(int))
        groups.extend([row["subject"]] * len(y))
        paths.append(str(path))
    return np.concatenate(features), np.concatenate(labels), np.asarray(groups), paths


def representations(x: np.ndarray) -> dict[str, np.ndarray]:
    bands = x.reshape(len(x), 30, 5)
    # Float64 matches the original exploration; both transforms remove amplitude offsets.
    relative = bands - logsumexp(bands * np.log(10), axis=2, keepdims=True) / np.log(10)
    centered = bands - bands.mean(axis=2, keepdims=True)
    ratios = np.stack((bands[:, :, 1] - bands[:, :, 3], bands[:, :, 2] - bands[:, :, 3],
                       bands[:, :, 1] - bands[:, :, 2], bands[:, :, 0] - bands[:, :, 3]), axis=2)
    regions = [range(8), range(8, 18), range(18, 27), range(27, 30)]
    return {
        "absolute_logpower": x,
        "relative_logpower": relative.reshape(len(x), -1),
        "centered_logpower": centered.reshape(len(x), -1),
        "log_band_ratios": ratios.reshape(len(x), -1),
        "mean_logpower": bands.mean(axis=1),
        "mean_relative_logpower": relative.mean(axis=1),
        "regional_relative_logpower": np.concatenate([relative[:, list(region)].mean(axis=1)
                                                       for region in regions], axis=1),
    }


def models():
    candidates = []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        candidates.append((f"logistic_C{c}", LogisticRegression(C=c, max_iter=2000, random_state=42)))
    for c in (0.1, 1.0, 10.0):
        for gamma in ("scale", 0.01):
            candidates.append((f"svc_C{c}_gamma{gamma}", SVC(C=c, gamma=gamma, probability=False,
                                                           random_state=42)))
    for depth in (3, 6, None):
        for leaf in (5, 20):
            candidates.append((f"forest_depth{depth}_leaf{leaf}", RandomForestClassifier(
                n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                max_features="sqrt", n_jobs=4, random_state=42,
            )))
    return candidates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/features/fatigue_eeg"))
    parser.add_argument("--split", type=Path, default=Path("models/eeg_vigilance_v2/split.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/eeg_validation_experiment.json"),
                        help="validation report path; existing reports are preserved")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}. Choose a new --out to preserve prior results.")
    split = read_split(args.split)
    start = time.perf_counter()
    train_x, train_y, _, train_paths = read_allowed(args.data, split, "train")
    val_x, val_y, val_groups, val_paths = read_allowed(args.data, split, "validation")
    if set(train_paths).intersection(val_paths):
        raise ValueError("Train and validation cannot reference the same source file")
    if len(np.unique(train_y)) != 2 or len(np.unique(val_y)) != 2:
        raise ValueError("Training and validation must both contain the two reference classes")
    train_representations, val_representations = representations(train_x), representations(val_x)
    results = []
    for representation, x in train_representations.items():
        for model_name, model in models():
            pipeline = make_pipeline(StandardScaler(), model)
            pipeline.fit(x, train_y)
            if hasattr(pipeline, "predict_proba"):
                prediction = pipeline.predict_proba(val_representations[representation])[:, 1]
            else:
                prediction = expit(pipeline.decision_function(val_representations[representation]))
            threshold = select_threshold(val_y, prediction, label_threshold=0.5)
            metrics = grouped_metrics(val_y, prediction, val_groups, threshold, label_threshold=0.5)
            group_auc = [group["auroc"] for group in metrics["per_group"].values()
                         if group["auroc"] is not None]
            row = {"representation": representation, "model": model_name, "metrics": metrics,
                   "mean_subject_auroc": float(np.mean(group_auc)) if group_auc else None}
            results.append(row)
        best = max([row for row in results if row["representation"] == representation],
                   key=lambda row: row["metrics"]["overall"]["auroc"])
        print(json.dumps({"representation": representation, "best": best["model"],
                          "auroc": best["metrics"]["overall"]["auroc"],
                          "mean_subject_auroc": best["mean_subject_auroc"],
                          "per_subject_auroc": {g: m["auroc"] for g, m in best["metrics"]["per_group"].items()},
                          "elapsed_seconds": round(time.perf_counter() - start, 1)}), flush=True)
    results.sort(key=lambda row: (-row["metrics"]["overall"]["auroc"],
                                 -(row["mean_subject_auroc"] if row["mean_subject_auroc"] is not None else -1),
                                 row["metrics"]["overall"]["rmse"]))
    report = {
        "protocol": "Training/validation only; no held-out NPZ files opened; normalization fit on train",
        "split_file": str(args.split.resolve()), "seed": 42,
        "original_experiment": {"report": "runs/eeg_validation_experiment.json",
                                "split": "models/eeg_vigilance_v1/split.json", "candidate_count": 119},
        "train_subjects": split["train"], "validation_subjects": split["validation"],
        "opened_paths": train_paths + val_paths,
        "n_train_windows": len(train_y), "n_validation_windows": len(val_y),
        "n_candidates": len(results), "wall_seconds": time.perf_counter() - start,
        "limitations": ["Small validation subject count; 119-candidate exploration can overfit validation",
                        "SVC sigmoid decision scores are uncalibrated",
                        "No test evaluation or label inversion", "Recording-level fatigue labels",
                        "Float64 representations preserve the original exploration; checkpoints use float32"],
        "ranked_validation_results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "top5": [
        {"representation": row["representation"], "model": row["model"],
         "auroc": row["metrics"]["overall"]["auroc"],
         "balanced_accuracy": row["metrics"]["overall"]["balanced_accuracy"],
         "mean_subject_auroc": row["mean_subject_auroc"]} for row in results[:5]
    ]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
