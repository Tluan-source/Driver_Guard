"""Audit local data and checkpoints without training or changing frozen splits.

Writes a machine-readable dashboard snapshot and detailed evidence in a dated
audit directory. Raw-video sizes/ZIP directory checks are not full decode/CRC
verification. Checkpoints are loaded with torch's restricted weights-only loader.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def inventory(root: Path, directory: str) -> dict:
    path = root / directory
    files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    return {"path": directory, "file_count": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "extensions": dict(sorted(Counter(p.suffix.lower() for p in files).items()))}


def pilot_audit(root: Path) -> tuple[dict, dict]:
    path = root / "data/features/camera_uta_pilot/manifest.json"
    result = {"id": "camera_uta_pilot", **inventory(root, "data/features/camera_uta_pilot")}
    if not path.exists():
        return {**result, "status": "missing", "blockers": ["Missing feature manifest"]}, {}
    manifest = read_json(path)
    rows = manifest["clips"]
    groups: dict[str, set] = defaultdict(set)
    sources: dict[str, set] = defaultdict(set)
    hashes: dict[str, set] = defaultdict(set)
    summaries = {}
    errors = []
    checks = []
    for row in rows:
        split = row["split"]
        groups[split].add(row["subject"])
        sources[row.get("source_clip_id", row["clip_id"])].add(split)
        feature_path = Path(row["features"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        check = {"clip_id": row["clip_id"], "split": split,
                 "path": relative(feature_path, root), "status": "passed"}
        try:
            actual_hash = sha256(feature_path)
            hashes[actual_hash].add(split)
            if actual_hash != row.get("features_sha256"):
                raise ValueError("Feature SHA256 differs from manifest")
            with np.load(feature_path, allow_pickle=False) as archive:
                x, ts, valid = archive["x"], archive["ts_ms"], archive["valid"]
                if (x.shape != (len(ts), 19) or valid.shape != ts.shape or valid.dtype.kind != "b"
                        or not np.isfinite(x).all() or not np.isfinite(ts).all()
                        or not (np.diff(ts) > 0).all()):
                    raise ValueError("Invalid feature dimensions, timestamps or finite values")
                if int(archive["label"]) != row["label"] or str(archive["subject"]) != row["subject"]:
                    raise ValueError("NPZ label/subject differs from manifest")
                if len(x) != row["n_frames"]:
                    raise ValueError("NPZ frame count differs from manifest")
                check.update(frames=len(x), duration_seconds=float((ts[-1] - ts[0]) / 1000),
                             valid_fraction=float(valid.mean()), sha256=actual_hash)
        except Exception as exc:  # noqa: BLE001 - record each corrupt asset without abandoning the audit
            check.update(status="failed", error=str(exc))
            errors.append(check)
        if split != "train" and row.get("synthetic_exposure_factor", 1) != 1:
            errors.append({"clip_id": row["clip_id"], "error": "Augmentation outside training"})
        checks.append(check)
    overlaps = {}
    for left, left_subjects in groups.items():
        selected = [row for row in rows if row["split"] == left]
        summaries[left] = {"records": len(selected), "subjects": len(left_subjects),
                           "subject_ids": sorted(left_subjects),
                           "original_clips": len({row.get("source_clip_id", row["clip_id"]) for row in selected}),
                           "labels": dict(Counter(str(row["label"]) for row in selected))}
        for right, right_subjects in groups.items():
            if left < right:
                overlaps[f"{left}:{right}"] = sorted(left_subjects & right_subjects)
    source_leaks = {k: sorted(v) for k, v in sources.items() if len(v) > 1}
    content_leaks = {k: sorted(v) for k, v in hashes.items() if len(v) > 1}
    leak_free = not any(overlaps.values()) and not source_leaks and not content_leaks
    result.update(status="ready_for_binary_development" if not errors and leak_free else "failed_audit",
                  records=len(rows), subjects=len(set().union(*groups.values())), classes=[0, 1],
                  label_mapping={"0": "alert (source 0)", "1": "drowsy (source 10)"},
                  splits=summaries, manifest_sha256=sha256(path),
                  checks={"npz_checked": len(checks), "npz_failed": len(errors),
                          "sha256_verified": len(checks) - len([c for c in checks if c["status"] == "failed"]),
                          "subject_disjoint": not any(overlaps.values()),
                          "source_clip_disjoint": not source_leaks, "content_disjoint": not content_leaks,
                          "feature_dim": 19, "fps": manifest["input_contract"]["fps"]},
                  blockers=["Binary pilot only; class 5/low vigilance is absent",
                            "10-second clips cannot supply 20-second context",
                            "Weak video labels; 40-clip human annotation review is pending",
                            "Existing test cohort was already evaluated for camera v1"])
    return result, {"checks": checks, "errors": errors, "subject_overlap": overlaps,
                    "source_leakage": source_leaks, "content_leakage": content_leaks}


def uta_audit(root: Path, pilot: dict) -> tuple[dict, dict]:
    folder = root / "data/raw/uta_rldd_full"
    videos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    subjects = sorted({p.parent.name for p in videos})
    records = [{"path": relative(p, root), "subject": p.parent.name,
                "source_label": p.stem.split("_")[0], "part": p.stem.split("_")[1:],
                "bytes": p.stat().st_size} for p in videos]
    archive_checks = []
    for archive in sorted(folder.glob("*.zip")):
        item = {"path": relative(archive, root), "crc_checked": False}
        try:
            with zipfile.ZipFile(archive) as zf:
                members = [i for i in zf.infolist() if Path(i.filename).suffix.lower() in VIDEO_SUFFIXES]
                mismatches = []
                for info in members:
                    candidates = [folder / info.filename, folder / archive.stem / info.filename]
                    match = next((p for p in candidates if p.is_file()), None)
                    if match is None or match.stat().st_size != info.file_size:
                        mismatches.append(info.filename)
                item.update(status="passed" if not mismatches else "failed", video_members=len(members),
                            missing_or_size_mismatches=mismatches)
        except (OSError, zipfile.BadZipFile) as exc:
            item.update(status="failed", error=str(exc))
        archive_checks.append(item)
    pilot_subjects = set()
    for split in pilot.get("splits", {}).values():
        pilot_subjects.update(split["subject_ids"])
    result = {"id": "uta_rldd_full", **inventory(root, "data/raw/uta_rldd_full"),
              "status": "raw_present_features_missing", "subjects": len(subjects), "records": len(videos),
              "classes": [0, 5, 10], "label_mapping": {"0": "alert", "5": "low vigilance", "10": "drowsy"},
              "video_bytes": sum(p.stat().st_size for p in videos),
              "checks": {"archive_directory_checked": len(archive_checks),
                         "archive_size_match": all(c["status"] == "passed" for c in archive_checks),
                         "crc_checked": False, "full_decode_checked": False,
                         "pilot_subject_overlap": len(set(subjects) & pilot_subjects),
                         "split_recordings": [r["path"] for r in records if r["part"]]},
              "blockers": ["Full-video content hashes, decode/PTS and verified extraction are not complete",
                           "No full three-class feature manifest at audit time",
                           "59 pilot subjects overlap; full UTA is not a fresh independent test cohort"]}
    intake_path = root / "data/manifests/uta_rldd_full/v001/manifest.json"
    if intake_path.exists():
        intake = read_json(intake_path)
        result.update(status="metadata_ready_media_validation_pending", manifest=relative(intake_path, root),
                      target_classes=intake["class_names"],
                      split_subject_counts={name: len(ids) for name, ids in intake["subject_splits"].items()})
        result["checks"].update(metadata_manifest_present=True, extraction_ready=intake["extraction_ready"],
                                split_policy="Inherit 59 pilot subjects; subject42 added to training")
    return result, {"videos": records, "archive_checks": archive_checks,
                    "scope": "ZIP central directory names/sizes only; no bulk hashing or full decode"}


def edf_audit(root: Path) -> tuple[dict, dict]:
    folder = root / "data/raw/dd_database"
    rows = []
    for path in sorted(folder.glob("*.edf")):
        row = {"path": relative(path, root), "annotation_file": "annotations" in path.stem}
        try:
            with path.open("rb") as stream:
                header = stream.read(256)
                count = int(header[252:256])
                channel_header = stream.read(256 * count)
            duration = float(header[244:252])
            samples_start = 216 * count
            labels = [channel_header[i * 16:(i + 1) * 16].decode("ascii").strip() for i in range(count)]
            rates = [int(channel_header[samples_start + i * 8:samples_start + (i + 1) * 8]) / duration
                     for i in range(count)]
            row.update(status="passed", channels=labels, sample_rates_hz=rates,
                       record_duration_seconds=duration, records=int(header[236:244]))
        except Exception as exc:  # noqa: BLE001 - a corrupt EDF must not suppress other audit results
            row.update(status="failed", error=str(exc))
        rows.append(row)
    signals = [r for r in rows if not r["annotation_file"]]
    result = {"id": "dd_database", **inventory(root, "data/raw/dd_database"),
              "status": "headers_audited_labels_pending", "subjects": len({Path(r["path"]).stem.split("_")[0] for r in signals}),
              "records": len(signals), "classes": [],
              "checks": {"edf_headers": len(rows), "failed": sum(r["status"] == "failed" for r in rows),
                         "signal_files": len(signals), "annotation_files": len(rows) - len(signals)},
              "blockers": ["Waveform quality/event annotation semantics still need validation",
                           "No paired face video; cannot be treated as synchronized camera supervision"]}
    return result, {"edf_headers": rows, "scope": "EDF headers only; waveform quality not verified"}


def other_datasets(root: Path) -> list[dict]:
    from driverguard.learning.eeg.data import load_sessions

    rows = []
    for name in ("fatigue_eeg", "fatigue_eeg_robust_v1"):
        path = root / f"data/features/{name}/sessions.json"
        sessions = read_json(path) if path.exists() else []
        checks = {"paired_camera": False}
        try:
            loaded = load_sessions(path.parent)
            checks.update(feature_contract_status="passed", npz_checked=len(loaded),
                          windows=sum(len(session.x) for session in loaded),
                          accepted_windows=sum(int(session.quality_valid.sum()) for session in loaded))
        except Exception as exc:  # noqa: BLE001 - retain the failing dataset as audit evidence
            checks.update(feature_contract_status="failed", error=str(exc))
        rows.append({"id": name, **inventory(root, f"data/features/{name}"),
                     "status": "independent_eeg_baseline", "records": len(sessions),
                     "subjects": len({r["subject"] for r in sessions}), "classes": [0, 1],
                     "checks": checks,
                     "blockers": ["Independent EEG; no synchronized camera/EEG sessions"]})
    archive = root / "data/raw/drozy/DROZY.zip"
    valid_zip = archive.exists() and zipfile.is_zipfile(archive)
    rows.append({"id": "drozy", **inventory(root, "data/raw/drozy"),
                 "status": "archive_directory_readable" if valid_zip else "incomplete_download",
                 "records": 0, "subjects": 0, "classes": [],
                 "checks": {"zip_directory_readable": valid_zip, "crc_checked": False,
                            "zip_bytes": archive.stat().st_size if archive.exists() else 0},
                 "blockers": ["No verified extracted synchronized video/PSG/timestamps/KSS/PVT data"]})
    for name in ("nthu_ddd", "dmd", "ul_dd", "seed_vig_features"):
        item = inventory(root, f"data/raw/{name}")
        rows.append({"id": name, **item, "status": "not_audited_for_training", "classes": [],
                     "blockers": ["No validated training manifest in this audit"]})
    return rows


def compact_camera_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    report = read_json(path)
    metrics = []
    for split, levels in report.get("metrics", {}).items():
        if not isinstance(levels, dict):
            continue
        for level, values in levels.items():
            if not isinstance(values, dict):
                continue
            score = values.get("overall_accepted_only", {})
            if score:
                metrics.append({"split": split, "unit": level, "coverage": values.get("coverage"),
                                **{k: score.get(k) for k in ("n", "accuracy", "macro_f1", "auroc")}})
    return metrics


def checkpoint_audit(root: Path) -> list[dict]:
    import torch

    from driverguard.learning.camera.model import CONTEXT, CameraPredictor
    from driverguard.learning.camera.train import camera_windows, load_camera_manifest
    from driverguard.learning.eeg.model import VigilancePredictor

    clips, _ = load_camera_manifest(root / "data/features/camera_uta_pilot/manifest.json", splits=("train",))
    window = next(w for w in camera_windows(clips) if w.reason is None)
    clip = clips[window.clip]
    section = slice(window.end + 1 - CONTEXT, window.end + 1)
    rows = []
    for path in sorted((root / "models").rglob("*.pt")):
        row = {"id": str(path.relative_to(root / "models")).replace("\\", "/"),
               "path": relative(path, root), "bytes": path.stat().st_size, "sha256": sha256(path),
               "load_status": "not_checked", "inference_status": "not_checked", "metrics": []}
        try:
            metadata = torch.load(path, map_location="cpu", weights_only=True)
            family = metadata.get("architecture", metadata.get("model_type", "unknown"))
            row.update(load_status="passed", family=family, context=metadata.get("context"),
                       input_dim=metadata.get("feature_dim", len(metadata.get("mean", []))),
                       classes=metadata.get("class_names", [0, 1]),
                       optimizer_resume="optimizer_state_dict" in metadata or "optimizer_state" in metadata,
                       label_source=metadata.get("label_source"),
                       metrics=compact_camera_metrics(path.parent / "metrics.json") if "camera_" in path.as_posix() else [])
            if "camera_" in path.as_posix():
                predictor = CameraPredictor(path)
                score = predictor.predict_window(clip.x[section], clip.ts_ms[section], clip.valid[section]).score
                if score is None or not np.isfinite(score):
                    raise ValueError("No finite prediction on quality-accepted training sample")
                row.update(inference_status="passed", inference_sample={"split": "train", "score": float(score)})
            elif "eeg_" in path.as_posix():
                predictor = VigilancePredictor(path)
                robust = predictor.preprocessing == "robust_v1"
                candidates = root / "data/features" / ("fatigue_eeg_robust_v1" if robust else "fatigue_eeg")
                sample = candidates / "subject_03_normal.npz"
                scores = predictor.predict_file(sample)
                finite = np.isfinite(scores)
                if not finite.any():
                    raise ValueError("No finite EEG predictions on existing feature sample")
                row.update(inference_status="passed", inference_sample={"path": relative(sample, root),
                           "predictions": len(scores), "finite": int(finite.sum())})
        except Exception as exc:  # noqa: BLE001 - checkpoint loader failures are an audit outcome
            row.update(error=str(exc))
            row["load_status" if row["load_status"] != "passed" else "inference_status"] = "failed"
        rows.append(row)
    return rows


def run(root: Path, out: Path) -> dict:
    from driverguard.project import doctor
    out.mkdir(parents=True, exist_ok=True)
    pilot, pilot_detail = pilot_audit(root)
    uta, uta_detail = uta_audit(root, pilot)
    edf, edf_detail = edf_audit(root)
    checkpoints = checkpoint_audit(root)
    environment = {"doctor": doctor()}
    tests_path = out / "tests.json"
    if tests_path.exists():
        environment["tests"] = read_json(tests_path)
    annotation_path = root / "outputs/camera_annotation_a2/annotations.csv"
    if annotation_path.exists():
        with annotation_path.open(encoding="utf-8-sig", newline="") as stream:
            annotations = list(csv.DictReader(stream))
        environment["annotation_review"] = {
            "path": relative(annotation_path, root), "records": len(annotations),
            "states": dict(Counter(row["review_status"] for row in annotations))}
    blockers = [
        {"id": "quality_target", "severity": "critical", "summary": "No verified GRU+DANN result above 85%",
         "action": "Train on Kaggle, select on validation, evaluate a locked candidate with coverage and subject grouping"},
        {"id": "full_features", "severity": "high", "summary": "Full three-class UTA feature pipeline is incomplete",
         "action": "Validate full-video hashes/decode/PTS using the metadata manifest, then extract versioned features"},
        {"id": "evaluation_cohort", "severity": "high", "summary": "Pilot and full UTA overlap; old test already inspected",
         "action": "Declare development scores honestly and reserve a new independent cohort for final acceptance"},
        {"id": "annotation", "severity": "high", "summary": "40 A2 clips still need human label adjudication",
         "action": "Complete annotation review without inferring physiological onset from video labels"},
        {"id": "drozy", "severity": "medium", "summary": "DROZY download is incomplete",
         "action": "Finish archive download and verify integrity before paired EEG experiments"},
    ]
    report = {"schema_version": 1,
              "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
              "timezone": "Asia/Bangkok", "status": "blocked",
              "target": {"metric": "accuracy_or_macro_f1", "threshold": 0.85, "comparison": ">",
                         "achieved": False, "reason": "No verified GRU+DANN acceptance evaluation is present"},
              "datasets": [pilot, uta, edf, *other_datasets(root)], "checkpoints": checkpoints,
              "evidence": [
                  {"id": "camera_v1_test", "status": "below_target", "summary": "Window accuracy 58.06%, macro-F1 57.05%; clip accuracy 55%",
                   "path": "docs/results/camera_v1/summary.json"},
                  {"id": "cam02_validation", "status": "development_only", "summary": "Full seed42 window accuracy 63.89%, macro-F1 63.60%; test not evaluated",
                   "path": "models/camera_cam02_multicue/full_seed42/metrics.json"},
                  {"id": "roadmap", "status": "in_progress", "summary": "Full UTA/three-class/DANN/EEG supervision/acceptance gates remain open",
                   "path": "docs/ROADMAP.md"}],
              "blockers": blockers, "environment": environment,
              "scope": ["All pilot NPZ files checked against SHA256, dimensions, labels and split identities",
                        "All models/**/*.pt restricted-load and existing-data inference checked",
                        "Full raw-video decode, raw-video content hashes and bulk ZIP CRC were not run",
                        "No model trained; no new held-out performance score generated"]}
    for name, payload in (("project_audit.json", report), ("camera_pilot_checks.json", pilot_detail),
                          ("uta_raw_inventory.json", uta_detail), ("dd_edf_headers.json", edf_detail)):
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/audits/20261005")
    args = parser.parse_args()
    result = run(ROOT, args.out)
    print(json.dumps({"report": relative(args.out / "project_audit.json", ROOT),
                      "datasets": len(result["datasets"]), "checkpoints": len(result["checkpoints"]),
                      "target_achieved": result["target"]["achieved"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
