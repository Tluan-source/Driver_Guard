"""Aggregate GRU/DANN validation runs across seeds, including causal smoothing.

    python -m scripts.evaluation.summarize_gru_dann runs/gru_dann/uta_full_v002_20261009 \
        --out outputs/audits/20261009/gru_dann_uta_full_summary.json

Smoothed scores average a window's class probabilities with the accepted windows of the same
recording that ended within the previous `--smooth-seconds`; no future window is used. The
recording unit averages all accepted windows of a recording, so it describes a whole 10-minute
video, not an online decision.

Two secondary analyses, binary label modes only:
- seed ensemble: mean probabilities of all seeds of one mode on identical windows;
- driver enrollment: the first `--enroll-seconds` of each driver's alert recording are treated
  as a calibration drive assumed alert (as at the start of a trip) and removed from scoring;
  the score is logit(p_drowsy) minus that driver's enrollment mean. The decision offset is
  chosen on the other validation drivers (leave-one-driver-out), never on the scored driver.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from driverguard.learning.camera.dann.metrics import classification_metrics


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def smoothed_predictions(rows: list[dict], n_classes: int, seconds: float) -> tuple[list[int], list[int]]:
    """Causal moving average of probabilities inside each recording."""
    by_clip = defaultdict(list)
    for row in rows:
        if row["accepted"] == "True":
            by_clip[row["clip_id"]].append(row)
    labels, predictions = [], []
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda row: int(row["end_ms"]))
        ends = np.array([int(row["end_ms"]) for row in clip_rows])
        scores = np.array([[float(row[f"prob_{i}"]) for i in range(n_classes)] for row in clip_rows])
        cumulative = np.vstack((np.zeros(n_classes), np.cumsum(scores, 0)))
        starts = np.searchsorted(ends, ends - seconds * 1000, side="right")
        for index, start in enumerate(starts):
            average = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
            labels.append(int(clip_rows[index]["label"]))
            predictions.append(int(average.argmax()))
    return labels, predictions


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _binary_rows(rows: list[dict]) -> list[dict]:
    return [{"subject": r["subject"], "clip_id": r["clip_id"], "label": int(r["label"]),
             "end_ms": int(r["end_ms"]), "p": float(r["prob_1"])} for r in rows if r["accepted"] == "True"]


def ensemble_rows(runs: list[list[dict]]) -> list[dict]:
    """Average drowsy probability over runs that scored exactly the same windows."""
    keyed = [{(r["clip_id"], r["end_ms"]): r for r in rows} for rows in runs]
    if any(set(k) != set(keyed[0]) for k in keyed):
        raise ValueError("Ensembled runs must share accepted windows")
    return [{**row, "p": float(np.mean([k[key]["p"] for k in keyed]))} for key, row in keyed[0].items()]


def binary_metrics(rows: list[dict], scores: np.ndarray, offset: float | np.ndarray) -> dict:
    labels = np.array([r["label"] for r in rows])
    predictions = (scores > offset).astype(int)
    window = classification_metrics(labels, predictions, 2)
    recordings = defaultdict(list)
    for index, row in enumerate(rows):
        recordings[row["clip_id"]].append(index)
    rec_labels = [labels[i[0]] for i in recordings.values()]
    rec_predictions = [int(np.mean(scores[i]) > np.mean(np.broadcast_to(offset, scores.shape)[i]))
                       for i in recordings.values()]
    recording = classification_metrics(rec_labels, rec_predictions, 2)
    return {"window_accuracy": window["accuracy"], "window_macro_f1": window["macro_f1"],
            "window_balanced_accuracy": window["balanced_accuracy"],
            "recording_accuracy": recording["accuracy"], "recording_macro_f1": recording["macro_f1"],
            "n_windows": len(rows), "n_recordings": len(rec_labels)}


def enrollment_metrics(rows: list[dict], enroll_seconds: float) -> dict:
    """Driver-relative score after an assumed-alert enrollment; offset by leave-one-driver-out."""
    limit = enroll_seconds * 1000
    enrollment = [r for r in rows if r["label"] == 0 and r["end_ms"] <= limit]
    scored = [r for r in rows if not (r["label"] == 0 and r["end_ms"] <= limit)]
    base = {s: np.mean(_logit(np.array([r["p"] for r in enrollment if r["subject"] == s])))
            for s in {r["subject"] for r in enrollment}}
    scored = [r for r in scored if r["subject"] in base]
    scores = _logit(np.array([r["p"] for r in scored])) - np.array([base[r["subject"]] for r in scored])
    labels = np.array([r["label"] for r in scored])
    subjects = np.array([r["subject"] for r in scored])
    grid = np.linspace(-2, 6, 161)
    offsets = np.zeros(len(scored))
    for subject in set(subjects):
        other = subjects != subject
        hits = scores[other][None, :] > grid[:, None]
        truth = labels[other]
        balanced = (hits[:, truth == 1].mean(1) + (~hits[:, truth == 0]).mean(1)) / 2
        offsets[~other] = grid[int(np.argmax(balanced))]
    result = binary_metrics(scored, scores, offsets)
    result.update(enroll_seconds=enroll_seconds, n_enrollment_windows=len(enrollment),
                  offset_mean=float(offsets.mean()), offset_selection="leave_one_driver_out_balanced_accuracy")
    return result


def secondary_analyses(paths: list[Path], enroll_seconds: float) -> dict:
    groups = defaultdict(list)
    for path in paths:
        protocol = json.loads((path / "metrics" / "protocol.json").read_text(encoding="utf-8"))
        if len(protocol["class_names"]) != 2:
            continue
        config = protocol["config"]
        groups[(config["label_mode"], config["context_seconds"], config["mode"])].append(
            _binary_rows(_rows(path / "predictions" / "validation_windows.csv")))
    result = {}
    for (label_mode, context, mode), members in sorted(groups.items()):
        key = f"{label_mode}/{context}s/{mode}"
        ensemble = ensemble_rows(members)
        result[key] = {
            "seeds": len(members),
            "ensemble": binary_metrics(ensemble, np.array([r["p"] for r in ensemble]), 0.5),
            "enrollment_per_seed": [enrollment_metrics(rows, enroll_seconds) for rows in members],
            "enrollment_ensemble": enrollment_metrics(ensemble, enroll_seconds),
        }
    return result


def run_metrics(run: Path, smooth_seconds: float) -> dict:
    summary = json.loads((run / "metrics" / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "metrics" / "protocol.json").read_text(encoding="utf-8"))
    n_classes = len(summary["class_names"])
    validation = summary["metrics"]["validation"]
    labels, predictions = smoothed_predictions(_rows(run / "predictions" / "validation_windows.csv"),
                                               n_classes, smooth_seconds)
    smoothed = classification_metrics(labels, predictions, n_classes)
    return {
        "run": run.name, "mode": protocol["config"]["mode"], "seed": protocol["config"]["seed"],
        "label_mode": summary["label_mode"], "context_seconds": protocol["config"]["context_seconds"],
        "pooling": protocol["config"].get("pooling", "frame"), "best_epoch": summary["best_epoch"],
        "window": {key: validation["window"][key] for key in
                   ("accuracy", "macro_f1", "balanced_accuracy", "auroc_macro", "coverage", "confusion_matrix")},
        "window_subject_macro_f1": validation["window"]["subject_macro"]["macro_f1"],
        "window_ci": validation["window"]["confidence_interval"]["intervals"],
        "smoothed": {key: smoothed[key] for key in ("accuracy", "macro_f1", "confusion_matrix")},
        "recording": {key: validation["clip"][key] for key in
                      ("accuracy", "macro_f1", "auroc_macro", "confusion_matrix")},
        "target_validation_met": summary["target"]["validation_met"],
    }


def aggregate(runs: list[dict]) -> dict:
    groups = defaultdict(list)
    for run in runs:
        groups[(run["label_mode"], run["context_seconds"], run["mode"])].append(run)
    result = []
    for (label_mode, context, mode), members in sorted(groups.items()):
        entry = {"label_mode": label_mode, "context_seconds": context, "mode": mode,
                 "seeds": sorted(run["seed"] for run in members)}
        for unit, metrics in (("window", ("accuracy", "macro_f1", "auroc_macro")),
                              ("smoothed", ("accuracy", "macro_f1")),
                              ("recording", ("accuracy", "macro_f1"))):
            for metric in metrics:
                values = [run[unit][metric] for run in members if run[unit][metric] is not None]
                entry[f"{unit}_{metric}"] = {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else 0.0} \
                    if values else None
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smooth-seconds", type=float, default=300.0)
    parser.add_argument("--enroll-seconds", type=float, default=300.0,
                        help="window end time limit inside the alert recording used for enrollment")
    arguments = parser.parse_args()
    paths = [path.parents[1] for path in sorted(arguments.root.glob("*/metrics/summary.json"))]
    runs = [run_metrics(path, arguments.smooth_seconds) for path in paths]
    if not runs:
        raise SystemExit(f"No completed runs below {arguments.root}")
    output = {"source": str(arguments.root).replace("\\", "/"), "smooth_seconds": arguments.smooth_seconds,
              "split": "validation", "runs": runs, "aggregate": aggregate(runs),
              "secondary": secondary_analyses(paths, arguments.enroll_seconds)}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for entry in output["aggregate"]:
        print(entry["label_mode"], entry["context_seconds"], entry["mode"],
              *(f"{key}={value['mean']:.4f}+/-{value['sd']:.4f}" for key, value in entry.items()
                if isinstance(value, dict)))
    for key, value in output["secondary"].items():
        per_seed = value["enrollment_per_seed"]
        print(key, "ensemble", {k: round(v, 4) for k, v in value["ensemble"].items() if isinstance(v, float)})
        print(key, "enrollment ensemble",
              {k: round(v, 4) for k, v in value["enrollment_ensemble"].items() if isinstance(v, float)})
        print(key, "enrollment per-seed window acc",
              [round(r["window_accuracy"], 4) for r in per_seed])


if __name__ == "__main__":
    main()
