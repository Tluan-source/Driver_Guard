"""Prepare week-6 evidence and blinded review forms from available development artifacts."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from driverguard.execution import file_sha256
from driverguard.learning.camera.features import PHYSICAL_FIELDS, PHYSICAL_SCALES
from scripts.data.prepare_camera_annotation import FIELDS, OPTIONS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def create_review_queue(manifest, predictions, out, seed=20261103):
    records = {row["clip_id"]: row for row in manifest["clips"]}
    if {r["subject"] for r in records.values()} & {"02", "07", "08", "09", "15", "16", "18", "41", "49", "57"}:
        raise ValueError("Original UTA test people cannot enter the review queue")
    accepted = predictions[predictions.accepted].copy()
    accepted["error"] = accepted.label != accepted.prediction
    ranking = accepted.groupby(["clip_id", "label"], as_index=False).agg(error_fraction=("error", "mean"))
    ranking = ranking.sort_values(["error_fraction", "clip_id"], ascending=[False, True])
    selected = []
    for label in (0, 1):
        group = ranking[ranking.label == label]
        group = group[group.clip_id.map(lambda name: records[name].get("part_index") is None)]
        if len(group) < 20:
            raise ValueError("Need twenty unique whole development recordings per binary class")
        selected.extend(group.head(20).to_dict("records"))
    random.Random(seed).shuffle(selected)
    mapping = []
    for index, row in enumerate(selected, 1):
        source = records[row["clip_id"]]
        if source["split"] not in ("train", "validation"):
            raise ValueError("Review preparation cannot include original test")
        candidates = accepted[(accepted.clip_id == row["clip_id"]) & accepted.error]
        if candidates.empty:
            candidates = accepted[accepted.clip_id == row["clip_id"]]
        endpoint = int(candidates.end_ms.min()) / 1000
        duration = source["n_frames"] / 10
        start = max(0., min(endpoint - 5, duration - 10))
        mapping.append({"review_id": f"R{index:03d}", "clip_id": row["clip_id"],
                        "subject": source["subject"], "original_split": source["split"],
                        "source_sha256": source["source_sha256"], "start_s": start, "end_s": start + 10,
                        "context_start_s": max(0., endpoint - 120), "context_end_s": endpoint,
                        "video": f"videos/R{index:03d}.mp4", "video_sha256": None,
                        "selection_error_fraction": row["error_fraction"], "source_label": int(row["label"])})
    out.mkdir(parents=True, exist_ok=True)
    package = {"version": "week6_A2_candidate_queue_v1", "selection_seed": seed,
               "scope": "40 new error-enriched development intervals; not a recovery of the original A2 forty",
               "materialized_videos": 0, "status": "awaiting_raw_videos_and_human_review", "clips": mapping}
    destination = out / "private_manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        stable = lambda p: [{k: c[k] for k in ("review_id", "clip_id", "source_sha256", "start_s", "end_s",
                                             "selection_error_fraction")} for c in p["clips"]]
        if stable(previous) != stable(package):
            raise FileExistsError("Existing review selection differs; preserve work and use a new folder")
        package = previous
    else:
        write_json(destination, package)
    for name in ("reviewer_a", "reviewer_b", "adjudication"):
        path = out / f"{name}.csv"
        if path.exists():
            continue  # Never overwrite human review work.
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for item in mapping:
                writer.writerow({"review_id": item["review_id"], "start_s": 0, "end_s": 10,
                                 "state": "uncertain", "eye_closure": "uncertain", "yawn": "uncertain",
                                 "head_nod": "uncertain", "visibility": "uncertain", "illumination": "uncertain",
                                 "glasses": "uncertain", "review_status": "unreviewed", "reviewer": "", "notes": ""})
    return package


def review_progress(out):
    package = json.loads((out / "private_manifest.json").read_text(encoding="utf-8"))
    counts = {}
    reviewer_ids = {}
    review_ids = {c["review_id"] for c in package["clips"]}
    materialized = 0
    for clip in package["clips"]:
        path = (out / clip["video"]).resolve()
        if not path.is_relative_to((out / "videos").resolve()):
            raise ValueError("Review video path escapes package")
        raw = Path(clip.get("source_path") or "__missing_raw_video__")
        if (path.is_file() and clip.get("video_sha256") and file_sha256(path) == clip["video_sha256"]
                and raw.is_file() and file_sha256(raw) == clip["source_sha256"]):
            materialized += 1
    for name in ("reviewer_a", "reviewer_b", "adjudication"):
        with (out / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if {r["review_id"] for r in rows} != review_ids:
            raise ValueError("Review forms must cover exactly the selected forty IDs")
        complete = 0
        reviewer_ids[name] = set()
        for review_id in review_ids:
            intervals = sorted([r for r in rows if r["review_id"] == review_id], key=lambda r: float(r["start_s"]))
            end = 0.
            all_reviewed = True
            for row in intervals:
                start, stop = float(row["start_s"]), float(row["end_s"])
                if not np.isfinite([start, stop]).all() or abs(start-end) > 1e-6 or stop <= start or stop > 10:
                    raise ValueError("Review intervals must cover [0,10) without gaps or overlaps")
                if any(row[key] not in choices for key, choices in OPTIONS.items()):
                    raise ValueError("Unknown review category")
                if row["visibility"] != "usable" and row["state"] != "uncertain":
                    raise ValueError("Unusable visibility must have uncertain state")
                reviewed = row["review_status"] == "reviewed"
                if reviewed and (not row["reviewer"].strip() or not row["notes"].strip()):
                    raise ValueError("Reviewed intervals require reviewer identity and observed evidence")
                if reviewed:
                    reviewer_ids[name].add(row["reviewer"].strip().casefold())
                all_reviewed &= reviewed
                end = stop
            if abs(end-10) > 1e-6:
                raise ValueError("Review intervals do not cover the whole clip")
            complete += all_reviewed
        counts[name] = complete
    if reviewer_ids["reviewer_a"] & reviewer_ids["reviewer_b"]:
        raise ValueError("Reviewer A and B must be independent people")
    ready = materialized == 40 and all(value == 40 for value in counts.values())
    return {"selected_intervals": len(package["clips"]), "verified_videos": materialized,
            "reviewed_clip_counts": counts, "ready_for_training_labels": ready,
            "status": ("reviewed_adjudicated" if ready else "awaiting_human_review"
                       if materialized == 40 else "awaiting_raw_videos_and_human_review"),
            "scope": "checks source-crop hash and form completeness; human judgment is not automatically validated"}


def feature_diagnostics(manifest, feature_root, out):
    rows = []
    for record in manifest["clips"]:
        if record["split"] not in ("train", "validation"):
            raise ValueError("Only development feature files may be opened")
        path = feature_root / record["features"]
        if file_sha256(path) != record["features_sha256"]:
            raise ValueError("Feature hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            x = data["x"]
            for i, (name, scale) in enumerate(zip(PHYSICAL_FIELDS, PHYSICAL_SCALES, strict=True)):
                values = x[(x[:, i+9] == 1) & (x[:, -1] == 1), i] * scale
                rows.append({"clip_id": record["clip_id"], "subject": record["subject"],
                             "label_name": manifest["class_names"][record["label"]], "feature": name,
                             "observed_frames": len(values), "available_fraction": len(values)/len(x),
                             "mean": float(values.mean()) if len(values) else None,
                             "std": float(values.std()) if len(values) else None,
                             "p10": float(np.percentile(values, 10)) if len(values) else None,
                             "p90": float(np.percentile(values, 90)) if len(values) else None})
    pd.DataFrame(rows).to_csv(out / "feature_diagnostics.csv", index=False)
    return {"records": len(manifest["clips"]), "feature_statistics_rows": len(rows),
            "scope": "masked descriptive geometry; no new onset labels or causal attribution"}


def main():
    out = Path("outputs/week6")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/features/week5_development/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = []
    for fold in range(1, 6):
        for seed in (42, 43, 44):
            path = Path(f"models/recovered_week5/experiments/fold{fold}_gru_seed{seed}/predictions/test_windows.csv")
            rows = pd.read_csv(path, dtype={"subject": str})
            all_rows.append(rows)
    all_rows = pd.concat(all_rows, ignore_index=True)
    key = ["subject", "clip_id", "end_ms"]
    if not (all_rows.groupby(key).label.nunique() == 1).all():
        raise ValueError("Ensemble members disagree on labels")
    if not (all_rows.groupby(key).size() == 3).all():
        raise ValueError("Three predeclared seed predictions required per endpoint")
    ensemble = all_rows.groupby(key, as_index=False).agg(label=("label", "first"), accepted=("accepted", "all"),
                                                       prob_0=("prob_0", "mean"), prob_1=("prob_1", "mean"))
    ensemble["prediction"] = (ensemble.prob_1 > ensemble.prob_0).astype(int)
    create_review_queue(manifest, ensemble, out / "a2_review")
    diagnostics = feature_diagnostics(manifest, manifest_path.parent, out)
    progress = review_progress(out / "a2_review")
    write_json(out / "a2_progress.json", progress)
    write_json(out / "preparation.json", {"source_sheet_rows": [34, 35, 36, 37, 38],
                                         "feature_diagnostics": diagnostics, "a2": progress,
                                         "feature_manifest_sha256": file_sha256(manifest_path)})
    print(json.dumps({"feature_diagnostics": diagnostics, "a2": progress}, indent=2))


if __name__ == "__main__":
    main()
