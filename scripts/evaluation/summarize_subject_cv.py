"""Aggregate outer-fold camera predictions without looking at original UTA test data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.evaluation.compare_models import acc_of, ensemble, macro_f1, paired_delta, subject_bootstrap


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    accepted = []
    for row in rows:
        if row["accepted"] != "True":
            continue
        probs = tuple(float(row[f"prob_{i}"]) for i in range(2))
        accepted.append((row["subject"], int(row["label"]), int(row["prediction"]), probs[1],
                         probs, int(row["end_ms"]), row["clip_id"]))
    return accepted, len(rows)


def summarize(root: Path, protocol_path: Path, out: Path | None = None):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    folds = json.loads((root / "manifests/folds.json").read_text(encoding="utf-8"))["folds"]
    models, totals = {}, {}
    for mode in protocol["modes"]:
        seeds = []
        for seed in protocol["seeds"]:
            rows, total = [], 0
            seen = set()
            for fold in folds:
                run = root / "experiments" / f"fold{fold['fold']}_{mode}_seed{seed}"
                summary = json.loads((run / "metrics/summary.json").read_text(encoding="utf-8"))
                if summary["status"] != "completed" or summary["metrics"].get("test") is None:
                    raise ValueError("Cannot aggregate an incomplete outer fold")
                records, count = read_rows(run / "predictions/test_windows.csv")
                if any(r[0] not in fold["test"] for r in records):
                    raise ValueError("Prediction contains a subject outside the outer fold")
                keys = {(r[6], r[5]) for r in records}
                if seen & keys:
                    raise ValueError("Duplicate out-of-fold endpoints")
                seen |= keys
                rows.extend(records)
                total += count
            seeds.append({"rows": rows, "smooth": [], "clips": [], "n_windows": total,
                          "class_names": ["alert", "drowsy"], "threshold": None})
        models[mode] = ensemble(seeds)
        totals[mode] = seeds[0]["n_windows"]
    values = {}
    for mode, model in models.items():
        rows = model["rows"]
        values[mode] = {"window_accuracy": acc_of(rows),
                        "window_accuracy_ci95": subject_bootstrap(rows, acc_of, np.random.default_rng(20261019)),
                        "window_macro_f1": (macro_f1(np.array([r[1] for r in rows]),
                                                     np.array([r[2] for r in rows]), 2) if rows else None),
                        "coverage": len(rows)/totals[mode] if totals[mode] else 0,
                        "accepted_windows": len(rows), "scheduled_windows": totals[mode]}
    delta = paired_delta(models["gru"], models["dann"], np.random.default_rng(20261019))
    candidate = "dann" if delta["ci95"] is not None and delta["ci95"][0] > 0 else "gru"
    selected = values[candidate]["window_accuracy"]
    report = {"scope": "outer-fold development evaluation; original UTA test excluded",
              "ensemble_rule": "equal probability mean of all three predeclared seeds",
              "models": values, "dann_minus_gru": delta, "candidate_mode": candidate,
              "primary_development_target_met": selected is not None and selected > protocol["primary"]["strictly_greater_than"],
              "independent_target_confirmed": False,
              "limitation": "Architecture previously developed on this population; new cohort needed for acceptance."}
    destination = out if out is not None else root / "cv_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, help="Write a fresh report without changing recovered artifacts")
    a = p.parse_args()
    print(json.dumps(summarize(a.root, a.protocol, a.out), indent=2))
