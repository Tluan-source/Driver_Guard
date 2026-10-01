"""Train and evaluate weak video-labelled camera GRUs with a fixed subject split."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .camera_features import sha256_file
from .camera_model import (
    CONTEXT,
    FEATURE_DIM,
    MAX_GAP_MS,
    MIN_VALID_FRACTION,
    CameraGRU,
    validate_feature_arrays,
    validate_input_contract,
    window_quality,
)
from .metrics import vigilance_metrics

THRESHOLD_GRID = np.arange(1, 20, dtype=float) / 20


@dataclass
class CameraClip:
    x: np.ndarray
    ts_ms: np.ndarray
    valid: np.ndarray
    label: int
    subject: str
    clip_id: str
    split: str
    illumination: str = "unknown"


@dataclass(frozen=True)
class CameraWindow:
    clip: int
    end: int
    valid_fraction: float
    reason: str | None


def load_camera_manifest(manifest_path: str | Path) -> tuple[list[CameraClip], dict]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_input_contract(manifest.get("input_contract"))
    records = manifest.get("clips")
    if not isinstance(records, list) or not records:
        raise ValueError("Camera manifest requires nonempty clips")
    clips = []
    seen = set()
    subjects = {name: set() for name in ("train", "validation", "test")}
    for record in records:
        split = record.get("split")
        if split not in subjects:
            raise ValueError("Each camera clip must have train/validation/test split")
        feature_path = Path(record["features"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        if "features_sha256" in record and sha256_file(feature_path) != record["features_sha256"]:
            raise ValueError("Camera feature cache SHA-256 disagrees with manifest")
        with np.load(feature_path, allow_pickle=False) as values:
            x = np.asarray(values["x"], dtype=np.float32)
            ts_ms = np.asarray(values["ts_ms"])
            valid = np.asarray(values["valid"])
            label_value = values["label"].item()
            subject = values["subject"].item()
            clip_id = values["clip_id"].item()
            if "extraction_signature" in values:
                expected = json.dumps(manifest["input_contract"], sort_keys=True, separators=(",", ":"))
                if values["extraction_signature"].item() != expected:
                    raise ValueError("Camera feature cache extraction signature does not match input contract")
        validate_feature_arrays(x, ts_ms, valid)
        if label_value not in (0, 1):
            raise ValueError("Camera clip labels must be binary 0/1")
        if record.get("label", label_value) != label_value:
            raise ValueError("Camera manifest label disagrees with the NPZ")
        if not isinstance(subject, str) or not subject or not isinstance(clip_id, str) or not clip_id:
            raise ValueError("Camera subject and clip_id must be nonempty strings")
        if record.get("subject", subject) != subject or record.get("clip_id", clip_id) != clip_id:
            raise ValueError("Camera manifest identifiers disagree with the NPZ")
        if clip_id in seen:
            raise ValueError("Camera clip_id must be unique")
        seen.add(clip_id)
        subjects[split].add(subject)
        clips.append(CameraClip(x, ts_ms, valid, int(label_value), subject, clip_id, split,
                                str(record.get("illumination", "unknown"))))
    if any(not value for value in subjects.values()):
        raise ValueError("Camera train, validation and test each require at least one subject")
    if any(subjects[first] & subjects[second] for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test"),
    )):
        raise ValueError("Camera subjects must be disjoint across train/validation/test")
    return clips, manifest


def camera_windows(clips: list[CameraClip], stride: int = 10) -> list[CameraWindow]:
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Camera window stride must be a positive integer")
    windows = []
    for index, clip in enumerate(clips):
        for end in range(CONTEXT - 1, len(clip.x), stride):
            start = end + 1 - CONTEXT
            result = window_quality(clip.x[start:end + 1], clip.ts_ms[start:end + 1], clip.valid[start:end + 1])
            windows.append(CameraWindow(index, end, result.valid_fraction, result.reason))
    return windows


class _WindowDataset(Dataset):
    def __init__(self, clips: list[CameraClip], windows: list[CameraWindow]):
        self.clips, self.windows = clips, windows
        counts = Counter(window.clip for window in windows)
        by_subject: dict[str, dict[int, set[int]]] = {}
        for window in windows:
            clip = clips[window.clip]
            by_subject.setdefault(clip.subject, {}).setdefault(clip.label, set()).add(window.clip)
        self.weights = []
        for window in windows:
            clip = clips[window.clip]
            classes = by_subject[clip.subject]
            weight = 1 / (len(by_subject) * len(classes) * len(classes[clip.label]) * counts[window.clip])
            self.weights.append(weight * len(windows))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        clip = self.clips[window.clip]
        return (torch.from_numpy(clip.x[window.end + 1 - CONTEXT:window.end + 1]),
                float(clip.label), self.weights[index])


def _scores(model: CameraGRU, clips: list[CameraClip], windows: list[CameraWindow], batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    with torch.inference_mode():
        for x, _, _ in DataLoader(_WindowDataset(clips, windows), batch_size=batch_size, shuffle=False):
            scores.extend(torch.sigmoid(model(x)).tolist())
    return np.asarray(scores, dtype=float)


def _prediction_rows(clips: list[CameraClip], windows: list[CameraWindow], scores: np.ndarray) -> list[dict]:
    rows = []
    position = 0
    for window in windows:
        clip = clips[window.clip]
        score = None
        if window.reason is None:
            score = float(scores[position])
            position += 1
        rows.append({"subject": clip.subject, "clip_id": clip.clip_id, "split": clip.split, "label": clip.label,
                         "score": score, "valid_fraction": window.valid_fraction, "accepted": score is not None,
                         "reason": window.reason, "end_ms": int(clip.ts_ms[window.end]), "illumination": clip.illumination})
    if position != len(scores):
        raise ValueError("Camera score count does not match accepted windows")
    return rows


def _clip_rows(clips: list[CameraClip], rows: list[dict]) -> list[dict]:
    result = []
    for clip in clips:
        windows = [row for row in rows if row["clip_id"] == clip.clip_id]
        accepted = [row for row in windows if row["accepted"]]
        score = float(np.mean([row["score"] for row in accepted])) if accepted else None
        result.append({"subject": clip.subject, "clip_id": clip.clip_id, "split": clip.split, "label": clip.label,
                           "score": score, "valid_fraction": float(np.mean(clip.valid)) if len(clip.valid) else 0.0,
                           "accepted": score is not None, "reason": None if accepted else "no_accepted_windows",
                           "window_coverage": len(accepted) / len(windows) if windows else 0.0,
                           "n_windows": len(windows), "illumination": clip.illumination})
    return result


def _row_metrics(rows: list[dict], threshold: float) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    metric = lambda group: vigilance_metrics(
        [row["label"] for row in group], [row["score"] for row in group], threshold, label_threshold=0.5,
    ) if group else None
    per_subject = {subject: metric([row for row in accepted if row["subject"] == subject])
                   for subject in sorted({row["subject"] for row in rows})}
    scalar_names = ("accuracy", "balanced_accuracy", "macro_f1", "precision", "recall", "specificity",
                    "auroc", "auprc", "rmse", "mae")
    macro = {}
    for name in scalar_names:
        values = [value[name] for value in per_subject.values() if value is not None and value[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
    return {
        "n_total": len(rows), "n_accepted": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "overall_accepted_only": metric(accepted), "per_subject_accepted_only": per_subject,
        "subject_macro_accepted_only": macro,
        "coverage_by_subject": {
            subject: sum(row["accepted"] for row in rows if row["subject"] == subject)
            / sum(row["subject"] == subject for row in rows) for subject in per_subject
        },
        "coverage_by_label": {
            str(label): sum(row["accepted"] for row in rows if row["label"] == label)
            / sum(row["label"] == label for row in rows)
            for label in sorted({row["label"] for row in rows})
        },
        "per_illumination": {
            name: {"metrics_accepted_only": metric([row for row in accepted if row["illumination"] == name]),
                   "coverage": sum(row["accepted"] for row in rows if row["illumination"] == name)
                   / sum(row["illumination"] == name for row in rows)}
            for name in sorted({row["illumination"] for row in rows})
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ["subject", "clip_id", "split", "label", "score", "valid_fraction"])
        writer.writeheader()
        writer.writerows(rows)


def select_camera_threshold(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Select an online-window threshold on a fixed grid using validation alone."""
    labels = np.asarray(labels)
    scores, weights = np.asarray(scores, dtype=float), np.asarray(weights, dtype=float)
    if (labels.ndim != 1 or scores.shape != labels.shape or weights.shape != labels.shape
            or set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1)) or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("Camera threshold selection requires binary labels, finite scores and positive weights")
    grid = THRESHOLD_GRID
    balanced_accuracy = []
    for threshold in grid:
        correct = (scores >= threshold) == labels
        balanced_accuracy.append(np.mean([
            np.average(correct[labels == label], weights=weights[labels == label]) for label in (0, 1)
        ]))
    best = grid[np.isclose(balanced_accuracy, np.max(balanced_accuracy), rtol=0, atol=1e-12)]
    return float(best[np.lexsort((best, np.abs(best - 0.5)))[0]])


def train_camera(
    manifest_path: str | Path, out_dir: str | Path, epochs: int = 30, patience: int = 7,
    seed: int = 42, batch_size: int = 64, stride: int = 10,
) -> dict:
    if min(epochs, patience, batch_size) < 1:
        raise ValueError("Camera epochs, patience and batch size must be positive")
    output = Path(out_dir)
    if (output / "model.pt").exists():
        raise FileExistsError("Camera model already exists; use a new output directory")
    clips, manifest = load_camera_manifest(manifest_path)
    windows = camera_windows(clips, stride)
    accepted = [window for window in windows if window.reason is None]
    split_windows = {split: [window for window in accepted if clips[window.clip].split == split]
                     for split in ("train", "validation", "test")}
    for split in ("train", "validation"):
        if {clips[window.clip].label for window in split_windows[split]} != {0, 1}:
            raise ValueError(f"Camera {split} needs accepted windows from both classes")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CameraGRU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_data = _WindowDataset(clips, split_windows["train"])
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    validation_data = _WindowDataset(clips, split_windows["validation"])
    validation_labels = np.asarray([clips[window.clip].label for window in split_windows["validation"]])
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, label, weight in loader:
            optimizer.zero_grad()
            losses = F.binary_cross_entropy_with_logits(model(x), label.float(), reduction="none")
            loss = (losses * weight.float()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            count += len(x)
        score = _scores(model, clips, split_windows["validation"], batch_size)
        clipped = np.clip(score, 1e-7, 1 - 1e-7)
        losses = -(validation_labels * np.log(clipped) + (1 - validation_labels) * np.log1p(-clipped))
        validation_loss = float(np.average(losses, weights=validation_data.weights))
        history.append({"epoch": epoch, "train_loss": total_loss / count, "validation_loss": validation_loss})
        print(f"camera epoch {epoch}/{epochs} train_loss={total_loss / count:.6f} "
              f"validation_loss={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    # Runtime applies a window score, so validation threshold selection uses windows too.
    validation_score = _scores(model, clips, split_windows["validation"], batch_size)
    threshold = select_camera_threshold(validation_labels, validation_score, np.asarray(validation_data.weights))
    rows = _prediction_rows(clips, windows, _scores(model, clips, accepted, batch_size))
    clip_rows = _clip_rows(clips, rows)
    timestamp_deltas = np.concatenate([np.diff(clip.ts_ms) for clip in clips if len(clip.ts_ms) > 1])
    report = {
        "task": "camera_current_state_weak_video_labels", "label_source": "binary_video_level",
        "early_onset_prediction": False, "decision_threshold": threshold,
        "threshold_kind": "online_window_current_state",
        "threshold_selection": "validation_subject_class_clip_window_weighted_balanced_accuracy",
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "epoch_selection": "validation_subject_class_clip_balanced_log_loss",
        "loss_weighting": "equal_subject_then_observed_class_then_clip_then_window",
        "best_epoch": best_epoch, "history": history,
        "windowing": {"context_frames": CONTEXT, "nominal_fps": 10.0, "stride_frames": stride,
                      "min_valid_fraction": MIN_VALID_FRACTION, "max_gap_ms": MAX_GAP_MS,
                      "current_frame_must_be_valid": True,
                      "observed_timestamp_delta_ms": {
                          "median": float(np.median(timestamp_deltas)) if len(timestamp_deltas) else None,
                          "p95": float(np.percentile(timestamp_deltas, 95)) if len(timestamp_deltas) else None,
                          "max": float(np.max(timestamp_deltas)) if len(timestamp_deltas) else None}},
        "clip_score": "mean_of_accepted_window_scores; unavailable_if_none",
        "metrics": {split: {
            "window": _row_metrics([row for row in rows if row["split"] == split], threshold),
            "clip": _row_metrics([row for row in clip_rows if row["split"] == split], threshold),
        } for split in ("train", "validation", "test")},
        "subject_splits": {split: sorted({clip.subject for clip in clips if clip.split == split})
                           for split in ("train", "validation", "test")},
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "input_contract": manifest["input_contract"], "seed": seed,
    }
    checkpoint = {"format_version": 1, "architecture": "camera_gru", "context": CONTEXT,
                      "feature_dim": FEATURE_DIM, "hidden_size": 32, "dropout": 0.1,
                      "decision_threshold": threshold, "input_contract": manifest["input_contract"],
                      "threshold_kind": report["threshold_kind"],
                      "task": report["task"], "state_dict": best_state}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "model.pt")
    _write_csv(output / "predictions.csv", rows)
    _write_csv(output / "clip_predictions.csv", clip_rows)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = train_camera(args.manifest, args.out, args.epochs, args.patience,
                          args.seed, args.batch_size, args.stride)
    print(json.dumps({"best_epoch": report["best_epoch"], "threshold": report["decision_threshold"],
                      "test_clip": report["metrics"]["test"]["clip"]}, indent=2))


if __name__ == "__main__":
    main()
