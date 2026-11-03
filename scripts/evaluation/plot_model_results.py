"""Render frozen EEG validation/test results without changing the model or threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("models/eeg_vigilance_v2"))
    args = parser.parse_args()
    report = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(args.run / "predictions_test.csv")
    result = report["test"][report["selected"]]["overall"]
    labels = rows["target"].to_numpy() >= result["label_threshold"]
    prediction = rows["prediction"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fpr, tpr, _ = roc_curve(labels, prediction)
    axes[0, 0].plot(fpr, tpr, color="#087f8c", label=f"AUROC = {result['auroc']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#8c8c8c", linewidth=1)
    axes[0, 0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out EEG ROC",
                   xlim=(0, 1), ylim=(0, 1))
    axes[0, 0].legend(loc="lower right")
    precision, recall, _ = precision_recall_curve(labels, prediction)
    axes[0, 1].plot(recall, precision, color="#c44747", label=f"AUPRC = {result['auprc']:.3f}")
    axes[0, 1].axhline(labels.mean(), linestyle="--", color="#8c8c8c", linewidth=1)
    axes[0, 1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall", xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].legend(loc="lower left")
    confusion = np.asarray(result["confusion_matrix"])
    axes[1, 0].imshow(confusion, cmap="Greens")
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=20,
                            color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    axes[1, 0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Fatigue"],
                   yticklabels=["Normal", "Fatigue"], xlabel="Predicted", ylabel="True",
                   title=f"Frozen threshold = {result['decision_threshold']:.6g}")
    names = list(report["validation"])
    validation_auc = [report["validation"][name]["auroc"] for name in names]
    test_auc = [report["test"][name]["overall"]["auroc"] for name in names]
    positions = np.arange(len(names))
    axes[1, 1].barh(positions - .18, validation_auc, height=.35, color="#087f8c", label="Validation")
    axes[1, 1].barh(positions + .18, test_auc, height=.35, color="#c44747", label="Test")
    axes[1, 1].set(yticks=positions, yticklabels=names, xlabel="AUROC", xlim=(0, 1),
                   title="Candidates selected using validation")
    axes[1, 1].tick_params(axis="y", labelsize=8)
    axes[1, 1].legend(loc="lower right")
    fig.suptitle(f"DriverGuard EEG fatigue: {report['selected']}\n"
                 "12 subjects; 8 train / 2 validation / 2 test; recording-level state labels", fontsize=13)
    target = args.run / "evaluation.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
