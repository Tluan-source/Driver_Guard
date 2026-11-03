"""Multiclass metrics with rejected-window coverage and subject-level aggregation."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import Corpus


def classification_metrics(labels, predictions, n_classes: int) -> dict:
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diagonal()
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(n_classes), where=support + predicted > 0)
    tn = matrix.sum() - support - predicted + tp
    specificity = np.divide(tn, matrix.sum() - support, out=np.zeros(n_classes),
                            where=matrix.sum() - support > 0)
    return {
        "accuracy": float(tp.sum() / len(labels)) if len(labels) else None,
        "macro_f1": float(f1.mean()) if len(labels) else None,
        "balanced_accuracy": float(recall[support > 0].mean()) if len(labels) else None,
        "confusion_matrix": matrix.tolist(),
        "per_class": {str(i): {"precision": float(precision[i]), "recall": float(recall[i]),
                                "f1": float(f1[i]), "specificity": float(specificity[i]),
                                "support": int(support[i])} for i in range(n_classes)},
    }


def row_metrics(rows: list[dict], n_classes: int) -> dict:
    accepted = [r for r in rows if r["accepted"]]
    result = classification_metrics([r["label"] for r in accepted],
                                    [r["prediction"] for r in accepted], n_classes)
    result.update(n_total=len(rows), n_accepted=len(accepted),
                  coverage=len(accepted) / len(rows) if rows else 0.0)
    result["metric_population"] = "accepted_predictions_only; rejected windows included in coverage"
    result["auroc_macro"] = None
    result["auprc_macro"] = None
    if accepted:
        from sklearn.metrics import average_precision_score, roc_auc_score
        labels = np.asarray([r["label"] for r in accepted])
        auc, ap = [], []
        for label in range(n_classes):
            truth = labels == label
            scores = np.asarray([r[f"prob_{label}"] for r in accepted])
            auroc = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None
            auprc = float(average_precision_score(truth, scores)) if truth.any() else None
            result["per_class"][str(label)].update(auroc=auroc, auprc=auprc)
            auc.append(auroc)
            ap.append(auprc)
        result["auroc_macro"] = float(np.mean(auc)) if all(value is not None for value in auc) else None
        result["auprc_macro"] = float(np.mean(ap)) if all(value is not None for value in ap) else None
    per_subject = {}
    for subject in sorted({r["subject"] for r in rows}):
        group = [r for r in accepted if r["subject"] == subject]
        per_subject[subject] = classification_metrics([r["label"] for r in group],
                                                       [r["prediction"] for r in group], n_classes)
    result["per_subject"] = per_subject
    result["subject_macro"] = {}
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = [value[metric] for value in per_subject.values() if value[metric] is not None]
        result["subject_macro"][metric] = float(np.mean(values)) if values else None
    result["per_illumination"] = {}
    for name in sorted({r["illumination"] for r in rows}):
        group = [r for r in accepted if r["illumination"] == name]
        result["per_illumination"][name] = classification_metrics(
            [r["label"] for r in group], [r["prediction"] for r in group], n_classes)
    result["coverage_by_subject"] = {
        subject: sum(r["accepted"] for r in rows if r["subject"] == subject)
        / sum(r["subject"] == subject for r in rows) for subject in per_subject
    }
    result["coverage_by_label"] = {
        str(label): sum(r["accepted"] for r in rows if r["label"] == label)
        / sum(r["label"] == label for r in rows) for label in sorted({r["label"] for r in rows})
    }
    return result


def subject_bootstrap(rows: list[dict], n_classes: int, seed: int, repetitions: int = 500) -> dict:
    """Cluster resampling preserves correlated windows belonging to one person."""
    subjects = sorted({r["subject"] for r in rows})
    groups = {subject: [r for r in rows if r["subject"] == subject and r["accepted"]] for subject in subjects}
    generator = np.random.default_rng(seed)
    values = {key: [] for key in ("accuracy", "macro_f1")}
    if len(subjects) >= 2:
        for _ in range(repetitions):
            sample = [row for subject in generator.choice(subjects, len(subjects), replace=True)
                      for row in groups[subject]]
            metrics = classification_metrics([r["label"] for r in sample],
                                             [r["prediction"] for r in sample], n_classes)
            for key, samples in values.items():
                if metrics[key] is not None:
                    samples.append(metrics[key])
    return {"method": "percentile_subject_cluster_bootstrap", "confidence": 0.95,
            "repetitions": repetitions, "n_subjects": len(subjects), "seed": seed,
            "intervals": {key: np.quantile(value, [0.025, 0.975]).tolist() if value else None
                          for key, value in values.items()}}


def prediction_rows(corpus: Corpus, windows, probabilities: np.ndarray, n_classes: int):
    rows, position = [], 0
    for window in windows:
        clip = corpus.clips[window.clip]
        accepted = window.reason is None
        probability = probabilities[position] if accepted else None
        row = {"subject": clip.subject, "clip_id": clip.clip_id, "split": clip.split,
               "label": clip.label, "prediction": int(probability.argmax()) if accepted else None,
               "accepted": accepted, "reason": window.reason, "valid_fraction": window.valid_fraction,
               "end_ms": int(clip.ts_ms[window.end]), "illumination": clip.illumination}
        row.update({f"prob_{i}": float(probability[i]) if accepted else None for i in range(n_classes)})
        rows.append(row)
        position += accepted
    if position != len(probabilities):
        raise ValueError("Probability count differs from accepted windows")
    return rows


def clip_rows(corpus: Corpus, windows: list[dict], split: str, n_classes: int):
    rows = []
    for clip in corpus.clips:
        if clip.split != split:
            continue
        selected = [r for r in windows if r["clip_id"] == clip.clip_id]
        accepted = [r for r in selected if r["accepted"]]
        probability = np.mean([[r[f"prob_{i}"] for i in range(n_classes)] for r in accepted], axis=0) \
            if accepted else None
        row = {"subject": clip.subject, "clip_id": clip.clip_id, "split": split,
               "label": clip.label, "prediction": int(probability.argmax()) if accepted else None,
               "accepted": bool(accepted), "reason": None if accepted else "no_accepted_windows",
               "valid_fraction": float(clip.valid.mean()) if len(clip.valid) else 0.0,
               "n_windows": len(selected), "window_coverage": len(accepted) / len(selected) if selected else 0.0,
               "illumination": clip.illumination}
        row.update({f"prob_{i}": float(probability[i]) if accepted else None for i in range(n_classes)})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["subject", "clip_id"])
        writer.writeheader()
        writer.writerows(rows)
