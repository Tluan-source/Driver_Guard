"""Independently reconcile restored week-5 hashes, folds, CSVs and ensembles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = (row["subject"], row["clip_id"], int(row["end_ms"]))
        if key in result:
            raise ValueError("Duplicate participant/recording/timestamp in predictions")
        accepted = row["accepted"] == "True"
        probabilities = [float(row[f"prob_{index}"]) for index in range(2)] if accepted else None
        if probabilities is not None and (not np.isfinite(probabilities).all()
                or not np.isclose(sum(probabilities), 1, atol=1e-6)):
            raise ValueError("Invalid class probabilities")
        prediction = int(row["prediction"]) if accepted else None
        if accepted and int(np.argmax(probabilities)) != prediction:
            raise ValueError("Stored decision differs from argmax")
        result[key] = {"label": int(row["label"]), "accepted": accepted,
                       "prediction": prediction, "probabilities": probabilities}
    return result


def metrics(rows):
    accepted = [row for row in rows if row["accepted"]]
    y = [r["label"] for r in accepted]
    prediction = [r["prediction"] for r in accepted]
    return {"window_accuracy": float(accuracy_score(y, prediction)),
            "window_macro_f1": float(f1_score(y, prediction, average="macro")),
            "coverage": len(accepted) / len(rows), "accepted_windows": len(accepted), "scheduled_windows": len(rows)}


def audit(root, out):
    root = Path(root)
    recovery = json.loads((root / "recovery_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in recovery["files_sha256"].items():
        if sha(root / name) != checksum:
            raise ValueError(f"Recovered file changed: {name}")
    summary = json.loads((root / "cv_summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "cv_comparison.json").read_text(encoding="utf-8"))
    folds = json.loads((root / "manifests/folds.json").read_text(encoding="utf-8"))
    original = json.loads(Path("docs/results/camera_v1/summary.json").read_text(encoding="utf-8"))["test_subjects"]
    outer, audits, ensembles = [], [], {mode: [] for mode in ("gru", "dann")}
    expected_runs = {f"fold{f}_{m}_seed{s}" for f in range(1, 6) for m in ensembles for s in (42, 43, 44)}
    if {r["run"] for r in summary["runs"]} != expected_runs:
        raise ValueError("Run inventory differs from the declared 30 configurations")
    for fold in folds["folds"]:
        groups = [set(fold[key]) for key in ("train", "validation", "test")]
        if (list(map(len, groups)) != [32, 8, 10] or any(set(original) & group for group in groups)
                or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))):
            raise ValueError("Invalid subject partition or original test leakage")
        outer.extend(fold["test"])
        for mode, ensemble_rows in ensembles.items():
            members = []
            for seed in (42, 43, 44):
                name = f"fold{fold['fold']}_{mode}_seed{seed}"
                run = root / "experiments" / name
                protocol = json.loads((run / "metrics/protocol.json").read_text(encoding="utf-8"))
                saved = json.loads((run / "metrics/summary.json").read_text(encoding="utf-8"))
                checkpoint = run / "inference/model.pt"
                if saved["checkpoint_sha256"] != sha(checkpoint):
                    raise ValueError(f"Checkpoint differs from historical summary: {name}")
                if any(set(protocol["subject_splits"][key]) != set(fold[key]) for key in ("train", "validation", "test")):
                    raise ValueError("Training protocol disagrees with the frozen fold")
                meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
                architecture = "camera_gru_dann" if mode == "dann" else "camera_gru_ablation"
                if (meta["context_frames"] != 1200 or meta["architecture"] != architecture
                        or any(not torch.isfinite(t).all() for t in meta["state_dict"].values())):
                    raise ValueError("Invalid 120-second checkpoint")
                predictions = read_rows(run / "predictions/test_windows.csv")
                if {key[0] for key in predictions} != set(fold["test"]):
                    raise ValueError("Prediction subjects differ from outer development fold")
                actual = metrics(list(predictions.values()))
                historical = next(r for r in summary["runs"] if r["run"] == name)["outer_metrics"]["window"]
                if not np.isclose(actual["window_accuracy"], historical["accuracy"], atol=1e-12, rtol=0):
                    raise ValueError("CSV accuracy does not reconcile with historical report")
                audits.append({"run": name, "checkpoint_hash_matches": True, "csv_accuracy_matches": True,
                               "outer_development_metrics": actual})
                members.append(predictions)
            if any(set(member) != set(members[0]) for member in members):
                raise ValueError("Seeds did not predict the same scheduled timestamps")
            for key in sorted(members[0]):
                peers = [member[key] for member in members]
                if any((p["label"], p["accepted"]) != (peers[0]["label"], peers[0]["accepted"]) for p in peers):
                    raise ValueError("Seed label/coverage mismatch")
                accepted = peers[0]["accepted"]
                probability = np.mean([p["probabilities"] for p in peers], axis=0) if accepted else None
                ensemble_rows.append({"label": peers[0]["label"], "accepted": accepted,
                                      "prediction": int(np.argmax(probability)) if accepted else None})
    if len(set(outer)) != 50 or any(count != 1 for count in Counter(outer).values()):
        raise ValueError("Outer folds must hold out all fifty development participants exactly once")
    recomputed = {mode: metrics(rows) for mode, rows in ensembles.items()}
    for mode in ensembles:
        for key, value in recomputed[mode].items():
            if not np.isclose(value, comparison["models"][mode][key], atol=1e-12, rtol=0):
                raise ValueError(f"Ensemble reconciliation failed: {mode}/{key}")
    report = {"status": "passed", "n_runs": len(audits), "n_outer_subjects": len(set(outer)),
              "recovered_files_hash_verified": len(recovery["files_sha256"]), "runs": audits,
              "recomputed_ensembles": recomputed, "original_test_included": False,
              "scope": "historical development-CV artifacts verified; independent acceptance not established",
              "inference_replayed": False, "source_feature_replay_required": True}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "runs"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path("models/recovered_week5"), type=Path)
    parser.add_argument("--out", default=Path("outputs/recovery_week5/artifact_audit.json"), type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.out), indent=2))
