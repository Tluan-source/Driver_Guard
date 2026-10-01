"""A1 camera audit: integrity metadata globally, feature/error review on development only."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from driverguard.learning.camera_features import PHYSICAL_FIELDS, sha256_file
from driverguard.learning.camera_model import CameraPredictor
from driverguard.learning.camera_train import _row_metrics, camera_windows, load_camera_manifest


def resources() -> dict:
    result = {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
              "python": platform.python_version(), "torch": torch.__version__,
              "torch_cuda_available": torch.cuda.is_available(),
              "cuda_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "disk_free_gib": shutil.disk_usage(Path.cwd()).free / 2**30}
    if os.name == "nt":
        query = ("[pscustomobject]@{cpu=(Get-CimInstance Win32_Processor).Name;"
                 "ram_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
                 "gpus=@(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM)}"
                 " | ConvertTo-Json -Depth 4")
        try:
            response = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                                      capture_output=True, text=True, check=True, timeout=20)
            result["hardware"] = json.loads(response.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            result["hardware_query_error"] = str(error)
    return result


def read_development_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["split"] not in {"train", "validation"}:
                continue
            rows.append({**row, "label": int(row["label"]), "score": float(row["score"]) if row["score"] else None,
                         "accepted": row["accepted"].lower() == "true",
                         "valid_fraction": float(row["valid_fraction"])})
    return rows


def select_review(records: list[dict], scores: dict[str, float | None], per_group: int = 10) -> list[dict]:
    result = []
    for split in ("train", "validation"):
        for label in (0, 1):
            candidates = sorted([r for r in records if r["split"] == split and r["label"] == label],
                                key=lambda r: (-(abs(scores[r["clip_id"]] - label)
                                                 if scores[r["clip_id"]] is not None else 2), r["clip_id"]))
            selected, subjects = [], set()
            for record in candidates:
                if record["subject"] not in subjects and len(selected) < per_group:
                    selected.append(record)
                    subjects.add(record["subject"])
            for record in candidates:
                if len(selected) >= per_group:
                    break
                if record not in selected:
                    selected.append(record)
            result.extend(selected)
    return result


def sampled_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result = []
        for position in (0, max(0, count // 2), max(0, count - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            success, frame = cap.read()
            if not success or frame is None:
                raise ValueError(f"Cannot decode development review frame: {path}")
            result.append(frame)
        return result
    finally:
        cap.release()


def frame_fingerprint(frame: np.ndarray) -> int | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.std() < 5:
        return None
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")


def write_review(selected: list[dict], raw: dict[str, dict], scores: dict, threshold: float, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in selected:
        score = scores[record["clip_id"]]
        original = raw[record["source_clip_id"]]
        rows.append({"subject": record["subject"], "clip_id": record["clip_id"], "split": record["split"],
                     "source_file": original["source_file"], "weak_video_label": record["label"],
                     "v1_mean_window_score": score,
                     "v1_clip_prediction": int(score >= threshold) if score is not None else None,
                     "valid_fraction": record["valid_fraction"], "review_status": "unreviewed",
                     "reviewed_state": "", "uncertain_intervals_seconds": "", "reviewer": "", "notes": ""})
    with (output / "review_queue.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for page in range(0, len(selected), 10):
        batch = selected[page:page + 10]
        sheet = np.full((len(batch) * 240, 912, 3), 245, np.uint8)
        for index, record in enumerate(batch):
            score = scores[record["clip_id"]]
            label = record["label"]
            incorrect = score is not None and int(score >= threshold) != label
            color = (20, 30, 190) if incorrect else (100, 70, 20)
            texts = [record["clip_id"], record["split"], f"weak label={label}",
                     f"v1 score={score:.3f}" if score is not None else "v1 unavailable",
                     f"raw valid={record['valid_fraction']:.0%}", "NOT HUMAN REVIEWED"]
            for line, text in enumerate(texts):
                cv2.putText(sheet, text, (8, index * 240 + 25 + line * 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
            for column, frame in enumerate(sampled_frames(raw[record["source_clip_id"]]["path"])):
                resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                sheet[index * 240:index * 240 + 224, 232 + column * 224:232 + (column + 1) * 224] = resized
        if not cv2.imwrite(str(output / f"contact_sheet_{page // 10 + 1:02d}.png"), sheet):
            raise OSError("Cannot write development contact sheet")


def audit(raw_path: Path, feature_path: Path, predictions_path: Path, checkpoint: Path,
          output: Path, review_output: Path) -> dict:
    for target in (output, review_output):
        if target.exists() and any(target.iterdir()):
            raise FileExistsError("Use new empty audit and review directories")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    if sha256_file(raw_path) != features["input_contract"]["source_manifest_sha256"]:
        raise ValueError("Feature contract references a different raw manifest")
    by_raw_id = {record["clip_id"]: record for record in raw["clips"]}
    if len(by_raw_id) != len(raw["clips"]):
        raise ValueError("Duplicate raw clip identifiers")
    predictor = CameraPredictor(checkpoint)
    if predictor.input_contract != features["input_contract"]:
        raise ValueError("Baseline and features have different input contracts")
    sha_groups, source_groups = defaultdict(list), defaultdict(list)
    for index, record in enumerate(raw["clips"]):
        path = Path(record["path"])
        if path.stat().st_size != record["source_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Raw video integrity failed: {record['clip_id']}")
        cap = cv2.VideoCapture(str(path))
        try:
            properties = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          "fps": cap.get(cv2.CAP_PROP_FPS), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        finally:
            cap.release()
        if any(not np.isclose(properties[key], record["media"][key]) for key in properties):
            raise ValueError(f"Raw video properties changed: {record['clip_id']}")
        sha_groups[record["sha256"]].append(record)
        source_groups[record["source_video_id"]].append(record)
        if (index + 1) % 60 == 0:
            print(f"A1 raw integrity {index + 1}/{len(raw['clips'])}", flush=True)
    duplicate_hashes = [[r["clip_id"] for r in group] for group in sha_groups.values() if len(group) > 1]
    if any(len({r["split"] for r in group}) > 1 for group in sha_groups.values()):
        raise ValueError("Exact raw video duplicate crosses splits")
    if any(len({r["split"] for r in group}) > 1 for group in source_groups.values()):
        raise ValueError("Source recording identity crosses splits")
    for record in features["clips"]:
        original = by_raw_id[record["source_clip_id"]]
        if (record["split"] != original["split"] or record["subject"] != original["subject"]
                or record["label"] != original["target"] or record["source_sha256"] != original["sha256"]):
            raise ValueError("Feature/source provenance mismatch")
        if record["synthetic_exposure_factor"] != 1 and record["split"] != "train":
            raise ValueError("Synthetic exposure escaped training split")
        if sha256_file(record["features"]) != record["features_sha256"]:
            raise ValueError("Feature file checksum changed")
    clips, _ = load_camera_manifest(feature_path, splits=("train", "validation"))
    development = [r for r in features["clips"] if r["split"] in {"train", "validation"}]
    if any(len(c.x) != r["n_frames"] or not np.isclose(c.valid.mean(), r["valid_fraction"])
           for c, r in zip(clips, development, strict=True)):
        raise ValueError("Feature metadata differs from arrays")
    windows = camera_windows(clips)
    rows = read_development_predictions(predictions_path)
    expected = {(clips[w.clip].clip_id, int(clips[w.clip].ts_ms[w.end])) for w in windows}
    observed = [(r["clip_id"], int(r["end_ms"])) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Baseline development predictions do not align with feature windows")
    original_records = [r for r in development if r["synthetic_exposure_factor"] == 1]
    original_ids = {r["clip_id"] for r in original_records}
    original_rows = [r for r in rows if r["clip_id"] in original_ids]
    mean_scores = {}
    for record in original_records:
        values = [r["score"] for r in original_rows if r["clip_id"] == record["clip_id"] and r["accepted"]]
        mean_scores[record["clip_id"]] = float(np.mean(values)) if values else None
    selected = select_review(original_records, mean_scores)
    fingerprints = []
    for record in original_records:
        fingerprint = frame_fingerprint(sampled_frames(by_raw_id[record["source_clip_id"]]["path"])[1])
        if fingerprint is not None:
            fingerprints.append((record, fingerprint))
    similar_pairs = []
    for index, (first, fingerprint) in enumerate(fingerprints):
        for second, other in fingerprints[index + 1:]:
            if first["split"] != second["split"]:
                distance = (fingerprint ^ other).bit_count()
                if distance <= 2:
                    similar_pairs.append({"clip_a": first["clip_id"], "clip_b": second["clip_id"],
                                          "dhash_hamming": distance})
    summaries = {}
    for split in ("train", "validation"):
        records = [c for c in clips if c.split == split]
        accepted = [w for w in windows if clips[w.clip].split == split and w.reason is None]
        scheduled = [w for w in windows if clips[w.clip].split == split]
        summaries[split] = {"subjects": len({c.subject for c in records}), "feature_clips": len(records),
                            "frames": sum(len(c.x) for c in records),
                            "raw_valid_fraction": float(np.mean(np.concatenate([c.valid for c in records]))),
                            "window_coverage": len(accepted) / len(scheduled),
                            "rejections": dict(Counter(w.reason for w in scheduled if w.reason)),
                            "feature_availability": {field: float(np.mean(np.concatenate([
                                c.x[:, i + 9] for c in records]))) for i, field in enumerate(PHYSICAL_FIELDS)}}
    distributions = {}
    for split in ("train", "validation"):
        distributions[split] = {}
        for label in (0, 1):
            ids = {r["clip_id"] for r in original_records if r["split"] == split and r["label"] == label}
            chosen = [c for c in clips if c.clip_id in ids]
            distributions[split][str(label)] = {
                "original_clips": len(chosen),
                "mean_raw_face_brightness": float(np.mean([
                    r["mean_raw_face_brightness"] for r in original_records if r["clip_id"] in ids])),
                "mean_physical_features": {field: float(np.mean(np.concatenate([
                    c.x[c.x[:, i + 9] == 1, i] for c in chosen]))) for i, field in enumerate(PHYSICAL_FIELDS)}
            }
    write_review(selected, by_raw_id, mean_scores, predictor.decision_threshold, review_output)
    report = {"task": "A1", "status": "technical_audit_complete_human_label_review_pending",
              "blockers_found": [], "test_predictions_evaluated": False,
              "npz_array_inspection_splits": ["train", "validation"],
              "integrity": {"raw_videos_verified": len(raw["clips"]),
                            "feature_checksums_verified": len(features["clips"]),
                            "subject_partition_disjoint": True, "source_recording_partition_disjoint": True,
                            "synthetic_exposure_train_only": True,
                            "exact_duplicate_video_hash_groups": duplicate_hashes,
                            "near_duplicate_development_middle_frame_candidates": similar_pairs,
                            "near_duplicate_method": "64bit dHash hamming<=2, train/validation only; heuristic"},
              "resources": resources(), "development_quality": summaries,
              "source_exposure_distributions": distributions,
              "v1_development_source_exposure_window_metrics": {
                  split: _row_metrics([r for r in original_rows if r["split"] == split], predictor.decision_threshold)
                  for split in ("train", "validation")},
              "review_clip_ids": [r["clip_id"] for r in selected],
              "review_queue_size": len(selected), "review_queue": str(review_output / "review_queue.csv"),
              "source_manifest_sha256": sha256_file(raw_path), "feature_manifest_sha256": sha256_file(feature_path),
              "baseline_checkpoint_sha256": sha256_file(checkpoint),
              "findings": ["Recording-level labels remain weak and unverified per frame/window.",
                           "All pilot clips are 10 seconds at 10 FPS; insufficient for fast blink kinetics or long contexts.",
                           "Face crops differ from full-frame webcam input; external cohort still needed.",
                           "No real-night or paired EEG onset data in this corpus.",
                           "Review queue uses baseline errors as priorities, not as new ground-truth labels."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"task": "A1", "review_clips": len(selected), "technical_blockers": 0,
                      "validation_v1": report["v1_development_source_exposure_window_metrics"]
                      ["validation"]["overall_accepted_only"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review-out", required=True, type=Path)
    args = parser.parse_args()
    audit(args.raw, args.features, args.baseline_predictions, args.baseline_checkpoint, args.out, args.review_out)


if __name__ == "__main__":
    main()
