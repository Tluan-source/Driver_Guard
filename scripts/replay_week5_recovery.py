"""Replay every recovered checkpoint on unchanged development feature samples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from driverguard.learning.camera.dann.data import quality_gate
from driverguard.learning.camera.dann.inference import DANNPredictor
from driverguard.learning.camera.features import sha256_file


def replay_all(root, features, out, samples_per_run=32, tolerance=1e-5):
    torch.set_num_threads(4)
    source = json.loads((Path(features) / "manifest.json").read_text(encoding="utf-8"))
    records = {r["clip_id"]: r for r in source["clips"]}
    if len({r["subject"] for r in records.values()}) != 50:
        raise ValueError("Expected the fifty-participant development bundle")
    for record in records.values():
        if sha256_file(Path(features) / Path(record["features"]).name) != record["features_sha256"]:
            raise ValueError("Recovered feature bytes changed")
    results = []
    for run in sorted((Path(root) / "experiments").iterdir()):
        if not run.is_dir():
            continue
        predictor = DANNPredictor(run / "inference/model.pt")
        with (run / "predictions/validation_windows.csv").open(encoding="utf-8", newline="") as stream:
            saved = [r for r in csv.DictReader(stream) if r["accepted"] == "True"]
        selected = [saved[i] for i in np.linspace(0, len(saved) - 1, min(samples_per_run, len(saved)), dtype=int)]
        xs, expected, files = [], [], {}
        for row in selected:
            record = records[row["clip_id"]]
            if record["split"] not in {"train", "validation"}:
                raise ValueError("Original test data is forbidden in development replay")
            path = Path(features) / Path(record["features"]).name
            if path not in files:
                with np.load(path, allow_pickle=False) as data:
                    files[path] = (data["x"], data["ts_ms"], data["valid"])
            x, timestamps, valid = files[path]
            end = int(np.searchsorted(timestamps, int(row["end_ms"]))) + 1
            if end <= 0 or timestamps[end - 1] != int(row["end_ms"]):
                raise ValueError("Saved timestamp is absent from restored features")
            start = end - predictor.context
            q = quality_gate(x[start:end], timestamps[start:end], valid[start:end], predictor.context)
            if q.reason is not None:
                raise ValueError("Historical accepted endpoint fails its quality gate")
            xs.append(x[start:end])
            expected.append([float(row["prob_0"]), float(row["prob_1"])])
        with torch.inference_mode():
            actual = predictor.model(torch.as_tensor(np.stack(xs), dtype=torch.float32)).numpy()
        differences = np.abs(actual - np.asarray(expected))
        maximum = float(differences.max())
        if not np.isfinite(actual).all() or maximum > tolerance:
            raise ValueError(f"CPU replay differs from Kaggle probabilities for {run.name}: {maximum}")
        results.append({"run": run.name, "sampled_validation_windows": len(xs),
                        "max_probability_difference": maximum, "checkpoint_sha256": sha256_file(run / "inference/model.pt")})
        print(f"replayed {run.name}: {len(xs)} windows, max difference {maximum:.3g}", flush=True)
    if len(results) != 30:
        raise ValueError("All thirty checkpoints must be replayed")
    report = {"status": "passed", "runs": results, "n_checkpoints": len(results),
              "feature_hashes_verified": len(records), "sampled_windows": sum(r["sampled_validation_windows"] for r in results),
              "max_probability_difference": max(r["max_probability_difference"] for r in results),
              "tolerance": tolerance, "original_test_opened": False,
              "scope": "CPU consistency samples on inner validation; no new research quality or full replay claim"}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "runs"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/recovered_week5"))
    parser.add_argument("--features", type=Path, default=Path("data/features/week5_development"))
    parser.add_argument("--out", type=Path, default=Path("outputs/recovery_week5/cpu_replay.json"))
    args = parser.parse_args()
    print(json.dumps(replay_all(args.root, args.features, args.out), indent=2))
