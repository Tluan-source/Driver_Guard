"""Reproduce held-out camera metrics and figures from frozen prediction CSVs.

No training or threshold selection occurs here. Confidence intervals resample
whole held-out subjects, preserving correlation between clips and windows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from driverguard.learning.metrics import bootstrap_group_ci, grouped_metrics

BOOTSTRAP_DRAWS = 300
BOOTSTRAP_SEED = 42
LABEL_THRESHOLD = 0.5
SCALAR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "precision", "recall", "specificity",
                  "auroc", "auprc", "mae", "rmse")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path, kind: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"subject", "clip_id", "split", "label", "score", "accepted", "reason", "valid_fraction"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain camera {kind} prediction columns: {sorted(required)}")
        rows = []
        seen = set()
        for record in reader:
            row = dict(record)
            row["label"] = int(record["label"])
            if row["label"] not in (0, 1):
                raise ValueError("Camera reference labels must be binary 0/1")
            if record["accepted"].lower() not in {"true", "false", "1", "0"}:
                raise ValueError("Camera accepted flags must be true/false or 1/0")
            row["accepted"] = record["accepted"].lower() in {"true", "1"}
            row["score"] = float(record["score"]) if record["score"] else None
            row["valid_fraction"] = float(record["valid_fraction"])
            if not np.isfinite(row["valid_fraction"]) or not 0 <= row["valid_fraction"] <= 1:
                raise ValueError("Camera valid fractions must be finite and in [0,1]")
            if row["accepted"] != (row["score"] is not None):
                raise ValueError("Rejected camera rows cannot have a score; accepted rows must have a score")
            if row["score"] is not None and (not np.isfinite(row["score"]) or not 0 <= row["score"] <= 1):
                raise ValueError("Camera accepted scores must be finite and in [0,1]")
            if not row["subject"] or not row["clip_id"] or row["split"] not in {"train", "validation", "test"}:
                raise ValueError("Camera rows require subject, clip_id and a recognized split")
            if row["split"] != "train" and "#exposure=" in row["clip_id"]:
                raise ValueError("Synthetic exposure augmentation must remain in train")
            key = row["clip_id"] if kind == "clip" else (row["clip_id"], record.get("end_ms"))
            if key in seen:
                raise ValueError(f"Duplicate camera {kind} prediction: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty camera prediction file: {path}")
    return rows


def _interval(values: list[float]) -> dict:
    low, high = np.quantile(values, [0.025, 0.975]) if values else (None, None)
    return {"low": float(low) if low is not None else None,
            "high": float(high) if high is not None else None, "n_valid": len(values)}


def _coverage_and_macro_ci(rows: list[dict], subjects: list[str], per_subject: dict) -> dict:
    counts = np.asarray([
        [sum(row["subject"] == subject for row in rows),
         sum(row["subject"] == subject and row["accepted"] for row in rows),
         *[sum(row["subject"] == subject and row["label"] == label for row in rows) for label in (0, 1)],
         *[sum(row["subject"] == subject and row["label"] == label and row["accepted"] for row in rows)
           for label in (0, 1)]] for subject in subjects
    ], dtype=float)
    samples = {"overall": [], "label_0": [], "label_1": []}
    macro_samples = {name: [] for name in SCALAR_METRICS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        draw = rng.integers(0, len(subjects), len(subjects))
        totals = counts[draw].sum(axis=0)
        if totals[0]:
            samples["overall"].append(float(totals[1] / totals[0]))
        for label in (0, 1):
            if totals[2 + label]:
                samples[f"label_{label}"].append(float(totals[4 + label] / totals[2 + label]))
        for name in SCALAR_METRICS:
            values = [per_subject[subjects[index]][name] for index in draw
                      if per_subject[subjects[index]] is not None and per_subject[subjects[index]][name] is not None]
            if values:
                macro_samples[name].append(float(np.mean(values)))
    return {
        "n_groups": len(subjects), "n_bootstrap": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED, "confidence": 0.95,
        "coverage_intervals": {name: _interval(values) for name, values in samples.items()},
        "subject_macro_intervals_accepted_only": {name: _interval(values) for name, values in macro_samples.items()},
    }


def _summarize(rows: list[dict], threshold: float, subjects: list[str]) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    grouped = grouped_metrics([row["label"] for row in accepted], [row["score"] for row in accepted],
                              [row["subject"] for row in accepted], threshold, LABEL_THRESHOLD) if accepted else None
    per_subject = {subject: grouped["per_group"].get(subject) if grouped else None for subject in subjects}
    macro, defined_subjects = {}, {}
    for name in SCALAR_METRICS:
        values = [value[name] for value in per_subject.values() if value is not None and value[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
        defined_subjects[name] = len(values)
    accepted_groups = len({row["subject"] for row in accepted})
    pooled_ci = bootstrap_group_ci(
        [row["label"] for row in accepted], [row["score"] for row in accepted],
        [row["subject"] for row in accepted], decision_threshold=threshold,
        label_threshold=LABEL_THRESHOLD, n_bootstrap=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
    ) if accepted_groups >= 2 else {"status": "unavailable_fewer_than_two_accepted_subjects"}
    pooled_ci["scope"] = "pooled_accepted_rows; subjects_without_any_accepted_rows_are_not_represented"
    return {
        "n_total": len(rows), "n_accepted": len(accepted),
        "n_subjects_total": len(subjects), "n_subjects_with_accepted_rows": accepted_groups,
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "class_counts_all": {str(label): sum(row["label"] == label for row in rows) for label in (0, 1)},
        "class_counts_accepted": {str(label): sum(row["label"] == label for row in accepted) for label in (0, 1)},
        "coverage_by_label": {
            str(label): (sum(row["label"] == label for row in accepted) / sum(row["label"] == label for row in rows)
                         if any(row["label"] == label for row in rows) else None) for label in (0, 1)},
        "coverage_by_subject": {
            subject: (sum(row["subject"] == subject for row in accepted) / sum(row["subject"] == subject for row in rows)
                      if any(row["subject"] == subject for row in rows) else None) for subject in subjects},
        "rejection_reasons": dict(Counter(row["reason"] or "unspecified" for row in rows if not row["accepted"])),
        "overall_accepted_only": grouped["overall"] if grouped else None,
        "per_subject_accepted_only": per_subject,
        "subject_macro_accepted_only": macro,
        "n_subjects_with_defined_metric": defined_subjects,
        "pooled_metrics_subject_bootstrap_ci": pooled_ci,
        "all_subjects_coverage_and_macro_bootstrap_ci": _coverage_and_macro_ci(rows, subjects, per_subject),
    }


def _plot(rows_by_kind: dict[str, list[dict]], summaries: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row_index, kind in enumerate(("clip", "window")):
        rows, summary = rows_by_kind[kind], summaries[kind]
        accepted = [row for row in rows if row["accepted"]]
        labels = np.asarray([row["label"] for row in accepted])
        roc_ax, confusion_ax, coverage_ax = axes[row_index]
        roc_ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
        if len(np.unique(labels)) == 2:
            false_positive, true_positive, _ = roc_curve(labels, [row["score"] for row in accepted])
            auc = summary["overall_accepted_only"]["auroc"]
            roc_ax.plot(false_positive, true_positive, color="#16806a", label=f"Pooled AUROC = {auc:.3f}")
            roc_ax.legend(loc="lower right", fontsize=9)
        else:
            roc_ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
        roc_ax.set(title=f"Held-out {kind}: accepted-only ROC", xlabel="False positive rate",
                   ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        matrix = summary["overall_accepted_only"]["confusion_matrix"] if accepted else [[0, 0], [0, 0]]
        confusion_ax.imshow(matrix, cmap="Blues", vmin=0)
        for y in (0, 1):
            for x in (0, 1):
                confusion_ax.text(x, y, str(matrix[y][x]), ha="center", va="center",
                                  color="white" if matrix[y][x] > np.max(matrix) / 2 else "black")
        confusion_ax.set(title=f"Held-out {kind}: accepted-only counts", xlabel="Predicted", ylabel="Reference",
                         xticks=[0, 1], xticklabels=["Alert", "Drowsy"],
                         yticks=[0, 1], yticklabels=["Alert", "Drowsy"])
        coverage = [summary["coverage"], summary["coverage_by_label"]["0"], summary["coverage_by_label"]["1"]]
        coverage_ax.bar(["All", "Alert", "Drowsy"], [value or 0 for value in coverage],
                        color=["#465c76", "#16806a", "#b0434f"])
        for x, value in enumerate(coverage):
            coverage_ax.text(x, (value or 0) + 0.03, f"{value:.1%}" if value is not None else "N/A", ha="center")
        coverage_ax.set(title=f"Held-out {kind}: coverage of all rows", ylabel="Accepted fraction", ylim=(0, 1.18))
    fig.suptitle("Camera current-state prototype | weak video labels | unseen subjects", fontsize=12)
    fig.savefig(out / "heldout_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/camera_drowsiness_v1"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/camera_v1"))
    args = parser.parse_args()
    paths = {name: args.model_dir / name for name in ("model.pt", "metrics.json", "clip_predictions.csv", "predictions.csv")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(f"Complete camera artifacts are required: {', '.join(missing)}")
    metrics = json.loads(paths["metrics.json"].read_text(encoding="utf-8"))
    threshold = float(metrics["decision_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frozen camera decision threshold must be finite and in [0,1]")
    splits = metrics["subject_splits"]
    if any(set(splits[first]) & set(splits[second]) for first, second in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Camera train/validation/test subjects overlap")
    if len(splits["test"]) < 2 or len(splits["test"]) != len(set(splits["test"])):
        raise ValueError("Camera report requires at least two distinct held-out subjects")
    all_rows = {"clip": _load_rows(paths["clip_predictions.csv"], "clip"),
                "window": _load_rows(paths["predictions.csv"], "window")}
    for kind, rows in all_rows.items():
        if any(row["subject"] not in splits[row["split"]] for row in rows):
            raise ValueError(f"Camera {kind} CSV subject membership disagrees with frozen splits")
        for split in ("train", "validation", "test"):
            expected = metrics["metrics"][split][kind]
            subset = [row for row in rows if row["split"] == split]
            if len(subset) != expected["n_total"] or sum(row["accepted"] for row in subset) != expected["n_accepted"]:
                raise ValueError(f"Camera {kind} CSV counts disagree with frozen {split} metrics")
    clip_lookup = {row["clip_id"]: row for row in all_rows["clip"]}
    for row in all_rows["window"]:
        clip = clip_lookup.get(row["clip_id"])
        if clip is None or any(row[key] != clip[key] for key in ("subject", "split", "label")):
            raise ValueError("Camera window CSV does not match its clip metadata")
    rows_by_kind = {kind: [row for row in rows if row["split"] == "test"] for kind, rows in all_rows.items()}
    summaries = {kind: _summarize(rows, threshold, splits["test"]) for kind, rows in rows_by_kind.items()}
    original_clips = {row["clip_id"].split("#exposure=", 1)[0] for row in all_rows["clip"]}
    report = {
        "status": "camera_current_state_weak_video_label_prototype",
        "early_onset_prediction": False, "real_night_accuracy_established": False,
        "decision_threshold": threshold, "label_threshold": LABEL_THRESHOLD,
        "threshold_selection": metrics.get("threshold_selection"),
        "clip_aggregation": metrics.get("clip_score"),
        "dataset": {"source": "mathiasviborg/uta-rldd-videos-cropped-by-faces",
                    "n_subjects": len(set().union(*(set(value) for value in splits.values()))),
                    "n_original_clips": len(original_clips), "n_feature_clips_including_train_augmentation": len(all_rows["clip"]),
                    "subject_counts_by_split": {name: len(value) for name, value in splits.items()},
                    "source_label_mapping": {"0": "alert=0", "10": "drowsy=1"},
                    "train_augmentation": "RGB exposure 0.6; validation/test use source exposure only"},
        "architecture": {"type": "causal_geometry_mask_GRU", "hidden_size": 32,
                         "n_features": len(metrics["input_contract"]["feature_names"]),
                         "windowing": metrics["windowing"]},
        "provenance": {
            "checkpoint_sha256": _sha256(paths["model.pt"]),
            "source_manifest_sha256": metrics["input_contract"]["source_manifest_sha256"],
            "feature_manifest_sha256": metrics["manifest_sha256"],
            "artifact_sha256": {name: _sha256(path) for name, path in paths.items() if name != "model.pt"},
        },
        "test_subjects": splits["test"], "test": summaries,
        "limitations": [
            "Weak video-level labels repeated over windows; no frame-level event or EEG onset ground truth.",
            "Accepted-only metrics must be interpreted with all-row and class-specific coverage.",
            "Clip scores average accepted windows offline; online scores use past-only windows.",
            "Scores are uncalibrated model outputs, not calibrated probabilities.",
            "RGB exposure augmentation does not establish real night, NIR, or external-cohort accuracy.",
            "Short balanced clips do not establish false alarms per driving hour or early-warning lead time.",
            "Latency and FPS require a separate hardware benchmark.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _plot(rows_by_kind, summaries, args.out)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "threshold": threshold, "dataset": report["dataset"],
                      "test_clip": summaries["clip"]["overall_accepted_only"],
                      "test_clip_coverage": summaries["clip"]["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
