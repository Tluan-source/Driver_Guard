"""Analyze only the recovered outer-development predictions, without opening original test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def summarize(frame):
    accepted = frame[frame.accepted]
    matrix = confusion_matrix(accepted.label, accepted.prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"scheduled": len(frame), "accepted": len(accepted), "coverage": len(accepted)/len(frame),
                "accuracy": float((accepted.label == accepted.prediction).mean()) if len(accepted) else None,
                "macro_f1": float(f1_score(accepted.label, accepted.prediction, labels=[0, 1], average="macro", zero_division=0)) if len(accepted) else None,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "drowsy_recall": float(tp/(tp+fn)) if tp+fn else None,
                "specificity": float(tn/(tn+fp)) if tn+fp else None}


def main():
    root = Path("models/recovered_week5")
    out = Path("outputs/week5_error_analysis")
    out.mkdir(parents=True, exist_ok=True)
    folds = json.loads((root / "manifests/folds.json").read_text(encoding="utf-8"))
    frames = []
    for fold in folds["folds"]:
        for mode in ("gru", "dann"):
            for seed in (42, 43, 44):
                run = f"fold{fold['fold']}_{mode}_seed{seed}"
                data = pd.read_csv(root / "experiments" / run / "predictions/test_windows.csv", dtype={"subject": str})
                assert set(data.subject) == set(fold["test"])
                assert set(data.split) == {"test"}
                data["run"], data["fold"], data["mode"], data["seed"] = run, fold["fold"], mode, seed
                data["quality_band"] = pd.cut(data.valid_fraction, [-.01, .6, .8, .95, 1.01], labels=["<=0.60", "0.60-0.80", "0.80-0.95", ">0.95"])
                frames.append(data)
    all_rows = pd.concat(frames, ignore_index=True)
    assert not all_rows.duplicated(["mode", "fold", "seed", "clip_id", "end_ms"]).any()
    assert not (set(all_rows.subject) & {"02", "07", "08", "09", "15", "16", "18", "41", "49", "57"})
    for dimension in ("run", "subject", "fold", "seed", "label", "quality_band"):
        records = []
        for key, frame in all_rows.groupby(["mode", dimension], observed=True):
            records.append({"mode": key[0], dimension: key[1], **summarize(frame)})
        pd.DataFrame(records).to_csv(out / f"by_{dimension}.csv", index=False)
    all_rows[~all_rows.accepted].groupby(["mode", "reason"], dropna=False).size().rename("count").to_csv(out / "rejection_reasons.csv")
    keys = ["mode", "fold", "subject", "clip_id", "end_ms"]
    ensemble = all_rows.groupby(keys, sort=True, as_index=False).agg(
        label=("label", "first"), accepted=("accepted", "all"), members=("seed", "nunique"),
        prob_0=("prob_0", "mean"), prob_1=("prob_1", "mean"))
    assert (ensemble.members == 3).all()
    assert (all_rows.groupby(keys).label.nunique() == 1).all()
    ensemble["prediction"] = (ensemble.prob_1 > ensemble.prob_0).astype(int)
    historical = json.loads((root / "cv_comparison.json").read_text(encoding="utf-8"))
    scores = {}
    for mode, frame in ensemble.groupby("mode"):
        scores[mode] = summarize(frame)
        assert np.isclose(scores[mode]["accuracy"], historical["models"][mode]["window_accuracy"])
    ensemble[ensemble.accepted & (ensemble.label != ensemble.prediction)].to_csv(out / "ensemble_errors.csv", index=False)
    per_subject = [{"mode": mode, "subject": subject, **summarize(frame)} for (mode, subject), frame in ensemble.groupby(["mode", "subject"])]
    pd.DataFrame(per_subject).sort_values("accuracy").to_csv(out / "ensemble_by_subject.csv", index=False)
    report = {"scope": "outer development CV; individual-run tables count each seed separately", "original_test_opened": False,
              "runs": 30, "subjects": 50, "individual_scheduled_rows": len(all_rows), "ensembles": scores,
              "limitations": ["illumination is source_rgb_unknown; no real day/night attribution", "no temporal human onset labels", "no false-alerts/hour estimate from window classification"]}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
