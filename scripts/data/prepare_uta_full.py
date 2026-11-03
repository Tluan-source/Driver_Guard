"""Inventory official UTA-RLDD full videos without decoding or hashing bulk media.

This creates a portable metadata manifest, not an extraction-ready manifest.
Existing pilot subject splits are inherited so related crops and original videos
never cross partitions. Subject 42, absent from the binary pilot, is train-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
LABELS = {0: 0, 5: 1, 10: 2}
CLASS_NAMES = ["alert", "low_vigilance", "drowsy"]
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}
PATTERN = re.compile(
    r"Fold(?P<fold>[1-5])_part(?P<archive_part>[12])/"
    r"(?P<subject>\d{2})/(?P<label>0|5|10)(?:_(?P<part>[12]))?\.(?:mov|mp4|m4v)",
    re.IGNORECASE,
)


def parse_video_path(path: str) -> dict:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path:
        raise ValueError(f"Unsafe relative video path: {path!r}")
    match = PATTERN.fullmatch(path)
    if match is None or not 1 <= int(match["subject"]) <= 60:
        raise ValueError(f"Unsupported UTA full video path: {path!r}")
    subject, label = match["subject"], int(match["label"])
    if (int(subject) - 1) // 12 + 1 != int(match["fold"]):
        raise ValueError(f"Subject does not belong to declared official fold: {path!r}")
    recording_id = f"UTA_S{subject}_{label}"
    part = int(match["part"]) if match["part"] else None
    return {"subject": subject, "source_label": label, "target": LABELS[label],
            "class_name": CLASS_NAMES[LABELS[label]], "official_fold": int(match["fold"]),
            "recording_id": recording_id, "source_video_id": f"{subject}_{label}",
            "part_index": part, "clip_id": recording_id + (f"_part{part}" if part else "")}


def inherit_splits(pilot: dict, subjects: set[str]) -> dict[str, list[str]]:
    assignments = {}
    for row in pilot.get("clips", []):
        subject, split = row["subject"], row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown pilot split: {split}")
        if subject in assignments and assignments[subject] != split:
            raise ValueError(f"Pilot subject leakage: {subject}")
        assignments[subject] = split
    if not assignments:
        raise ValueError("Pilot manifest must include subject split assignments")
    missing_existing = set(assignments) - subjects
    if missing_existing:
        raise ValueError(f"Full corpus is missing pilot subjects: {sorted(missing_existing)}")
    unseen = subjects - set(assignments)
    if unseen - {"42"}:
        raise ValueError(f"Unapproved subject additions require an explicit protocol: {sorted(unseen)}")
    if "42" in unseen:
        assignments["42"] = "train"
    return {split: sorted(subject for subject in subjects if assignments[subject] == split)
            for split in ("train", "validation", "test")}


def build_manifest(raw_root: Path, pilot_path: Path, *, project_root: Path = ROOT,
                   require_complete: bool = True) -> tuple[dict, dict, dict]:
    raw_root, pilot_path, project_root = raw_root.resolve(), pilot_path.resolve(), project_root.resolve()
    if not raw_root.is_dir():
        raise ValueError(f"Raw directory missing: {raw_root}")
    if not raw_root.is_relative_to(project_root) or not pilot_path.is_relative_to(project_root):
        raise ValueError("Raw data and pilot manifest must be inside the project root")
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes.decode("utf-8-sig"))
    rows = []
    groups = defaultdict(list)
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if not path.resolve().is_relative_to(raw_root):
            raise ValueError("Media symlink resolves outside the declared raw dataset")
        row = parse_video_path(path.relative_to(raw_root).as_posix())
        if path.stat().st_size == 0:
            raise ValueError(f"Empty video: {path}")
        row.update(path=path.relative_to(project_root).as_posix(), bytes=path.stat().st_size,
                   sha256=None, validation={"file_size": "present_nonempty", "content_hash": "pending",
                                            "decode": "pending", "pts": "pending", "rotation": "pending"},
                   extraction_ready=False, temporal_reset=True,
                   temporal_continuity="unverified; reset at each file boundary")
        rows.append(row)
        groups[row["recording_id"]].append(row)
    if not rows:
        raise ValueError("No UTA videos found")
    subjects = {row["subject"] for row in rows}
    if require_complete and subjects != {f"{i:02}" for i in range(1, 61)}:
        raise ValueError("Complete official corpus must contain subjects 01 through 60")
    recordings = []
    for recording_id, parts in sorted(groups.items()):
        indices = [row["part_index"] for row in parts]
        if len(indices) != len(set(indices)) or (None in indices and len(indices) > 1):
            raise ValueError(f"Duplicate or mixed whole/part recording: {recording_id}")
        if indices != [None] and sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError(f"Recording parts must be contiguous from 1: {recording_id}")
        if indices != [None] and len(indices) < 2:
            raise ValueError(f"Incomplete split recording: {recording_id}")
        recordings.append({"recording_id": recording_id, "subject": parts[0]["subject"],
                           "source_label": parts[0]["source_label"], "target": parts[0]["target"],
                           "clip_ids": [p["clip_id"] for p in sorted(parts, key=lambda p: p["part_index"] or 0)],
                           "parts": len(parts), "continuous_timeline_verified": False})
    for subject in subjects:
        if {row["source_label"] for row in rows if row["subject"] == subject} != {0, 5, 10}:
            raise ValueError(f"Subject {subject} is missing one of labels 0/5/10")
    if require_complete and (len(rows) != 182 or len(recordings) != 180):
        raise ValueError("Expected 182 files representing 180 subject/state recordings")
    splits = inherit_splits(pilot, subjects)
    lookup = {subject: name for name, ids in splits.items() for subject in ids}
    for row in rows:
        row["split"] = lookup[row["subject"]]
        row["domain_id"] = f"subject_{row['subject']}"
    for recording in recordings:
        recording["split"] = lookup[recording["subject"]]
    generated = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    split_doc = {"schema_version": 1, "subject_splits": splits,
                 "source": pilot_path.relative_to(project_root).as_posix(),
                 "source_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
                 "policy": "Inherit every pilot subject; subject 42 is added to train only",
                 "new_subject_assignments": {"42": "train"} if "42" not in {r["subject"] for r in pilot["clips"]} else {},
                 "test_status": "Previously inspected in camera v1; not a fresh final acceptance cohort",
                 "adaptation": "Training subjects only; validation and test excluded from domain adaptation"}
    manifest = {"schema_version": 1, "dataset": "uta_rldd_full", "version": "v001",
                "generated_at": generated, "status": "metadata_validated_media_validation_pending",
                "extraction_ready": False, "path_base": "repository_root",
                "raw_root": raw_root.relative_to(project_root).as_posix(),
                "label_mapping": {str(label): target for label, target in LABELS.items()},
                "class_names": CLASS_NAMES, "label_type": "weak_video_level_self_report",
                "source": {"url": "https://sites.google.com/view/utarldd",
                           "cloud_rights": "Not inferred from presence of downloaded files"},
                "subject_splits": splits, "split_source_sha256": split_doc["source_sha256"],
                "recordings": recordings, "clips": rows,
                "limitations": ["No raw-video content hash or full decode/PTS validation in metadata intake",
                                "Part continuity is unverified; reset sequences at each file boundary",
                                "Existing VideoFileSource uses index/FPS, not decoder PTS; VFR adapter is required",
                                "Pilot and full UTA share subjects and do not create an independent test cohort",
                                "No synchronized EEG or verified onset labels"]}
    audit = {"schema_version": 1, "generated_at": generated, "status": manifest["status"],
             "extraction_ready": False, "subjects": len(subjects), "files": len(rows),
             "recordings": len(recordings), "bytes": sum(row["bytes"] for row in rows),
             "label_counts_files": dict(Counter(str(row["source_label"]) for row in rows)),
             "split_subject_counts": {name: len(ids) for name, ids in splits.items()},
             "split_file_counts": dict(Counter(row["split"] for row in rows)),
             "split_recordings": [r["recording_id"] for r in recordings if r["parts"] > 1],
             "checks": {"paths_relative": True, "all_labels_present_per_subject": True,
                        "subjects_disjoint": True, "pilot_splits_preserved": True,
                        "parts_grouped": True, "content_hash": "pending", "full_decode": "pending",
                        "presentation_timestamps": "pending", "rotation": "pending"}}
    return manifest, split_doc, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/uta_rldd_full")
    parser.add_argument("--pilot", type=Path, default=ROOT / "data/features/camera_uta_pilot/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/manifests/uta_rldd_full/v001")
    args = parser.parse_args()
    documents = build_manifest(args.raw, args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, document in zip(("manifest", "split", "audit"), documents):
        (args.out / f"{name}.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(documents[2], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
