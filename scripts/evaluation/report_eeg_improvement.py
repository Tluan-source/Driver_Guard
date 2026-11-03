"""Export compact versionable metrics and a research figure from a completed run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="models/eeg_vigilance_v3")
    parser.add_argument("--out", default="docs/results")
    args = parser.parse_args(argv)
    run, output = Path(args.run), Path(args.out)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    report = {"experiment": run.name, "protocol": protocol["evaluation"],
              "selected_config": summary["selected_config"],
              "decision_threshold": summary["decision_threshold"],
              "candidate_grids": protocol["candidate_grids"], "seed": protocol["seed"],
              "bootstrap": protocol["bootstrap"], "versions": protocol["versions"],
              "source_file_hashes": protocol["source_file_hashes"],
              "checkpoint_sha256": hashlib.sha256((run / "model.pt").read_bytes()).hexdigest(),
              "legacy": summary["legacy"], "robust": summary["robust"],
              "paired_comparison": summary["paired_comparison"], "limitations": summary["limitations"]}
    if "legacy_selected_config" in summary:
        report["legacy_selected_config"] = summary["legacy_selected_config"]
        report["legacy_checkpoint_sha256"] = hashlib.sha256((run / "legacy_model.pt").read_bytes()).hexdigest()
    (output / "eeg_v3_development.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    subjects = sorted(summary["robust"]["per_subject"])
    indices = np.arange(len(subjects))
    figure = Figure(figsize=(12, 10), layout="constrained")
    axes = figure.subplots(3, 1, sharex=True)
    profiles = (("legacy", "Legacy CAR + calibrated ensemble", "#c45555"),
                ("robust", "Robust EEG + calibrated ensemble", "#008779"))
    for offset, (name, label, color) in zip((-0.18, 0.18), profiles):
        reports = summary[name]["per_subject"]
        auc = [reports[s]["metrics"]["auroc"] for s in subjects]
        balanced = [reports[s]["coverage_adjusted"]["balanced_accuracy"] for s in subjects]
        coverage = [reports[s]["coverage"]["coverage"] for s in subjects]
        for axis, values in zip(axes, (auc, balanced, coverage)):
            axis.bar(indices + offset, values, width=0.34, label=label, color=color)
    titles = ("AUROC on accepted windows", "Balanced accuracy: abstentions count as incorrect",
              "Coverage: fraction of windows receiving a prediction")
    for axis, title, label in zip(axes, titles, ("AUROC", "Balanced accuracy", "Coverage")):
        axis.set_ylim(0, 1.05)
        axis.set_title(title, fontsize=11, loc="left")
        axis.set_ylabel(label, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    axes[0].axhline(0.5, color="#666666", linestyle=":", linewidth=1)
    axes[1].axhline(0.5, color="#666666", linestyle=":", linewidth=1)
    axes[-1].set_xticks(indices, [subject.removeprefix("subject_") for subject in subjects])
    axes[-1].set_xlabel("Held-out subject")
    figure.suptitle("Nested subject LOSO on the 12-person development cohort", fontsize=14)
    figure.savefig(output / "eeg_v3_subject_comparison.png", dpi=140)
    print(json.dumps({"report": str(output / "eeg_v3_development.json"),
                      "figure": str(output / "eeg_v3_subject_comparison.png")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
