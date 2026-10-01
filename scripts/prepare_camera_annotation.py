"""Prepare anonymous A2 development videos, then validate human interval reviews."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from driverguard.learning.camera_features import sha256_file

FIELDS = ("review_id", "start_s", "end_s", "state", "eye_closure", "yawn", "head_nod",
          "visibility", "illumination", "glasses", "review_status", "reviewer", "notes")
OPTIONS = {
    "state": {"alert", "drowsy_visible", "uncertain"},
    "eye_closure": {"none", "blink", "sustained", "uncertain"},
    "yawn": {"absent", "present", "uncertain"},
    "head_nod": {"absent", "present", "uncertain"},
    "visibility": {"usable", "occluded", "unusable", "uncertain"},
    "illumination": {"adequate", "dim", "uncertain"},
    "glasses": {"absent", "present", "uncertain"},
    "review_status": {"unreviewed", "reviewed"},
}


def _raw_records(raw: dict) -> dict:
    records = raw["clips"]
    by_id = {record["clip_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate raw clip identifiers")
    subjects = defaultdict(set)
    for record in records:
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown raw split")
        subjects[record["subject"]].add(record["split"])
    if any(len(splits) != 1 for splits in subjects.values()):
        raise ValueError("Raw subject crosses splits")
    return by_id


def prepare(raw_path: Path, queue_path: Path, output: Path, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty annotation directory")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    by_id = _raw_records(raw)
    with queue_path.open(newline="", encoding="utf-8") as stream:
        queue = list(csv.DictReader(stream))
    if not queue or len({row["clip_id"] for row in queue}) != len(queue):
        raise ValueError("Review queue must contain unique development clip identifiers")
    selected = []
    for row in queue:
        record = by_id[row["clip_id"]]
        if record["split"] not in {"train", "validation"} or row["split"] != record["split"]:
            raise ValueError("Annotation preparation accepts development clips only")
        if row["subject"] != record["subject"] or sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("Annotation source provenance mismatch")
        selected.append(record)
    random.Random(seed).shuffle(selected)
    (output / "videos").mkdir(parents=True)
    mapping, rows = [], []
    for index, record in enumerate(selected, start=1):
        review_id = f"R{index:03d}"
        duration = float(record["media"]["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Annotation source duration must be positive")
        video = output / "videos" / f"{review_id}.mp4"
        shutil.copyfile(record["path"], video)
        if sha256_file(video) != record["sha256"]:
            raise ValueError("Anonymous video copy checksum mismatch")
        mapping.append({"review_id": review_id, "clip_id": record["clip_id"],
                        "subject": record["subject"], "split": record["split"],
                        "duration_seconds": duration, "video": f"videos/{review_id}.mp4",
                        "sha256": record["sha256"]})
        rows.append({**{key: "uncertain" for key in OPTIONS}, "review_id": review_id,
                     "start_s": 0, "end_s": duration, "review_status": "unreviewed",
                     "reviewer": "", "notes": ""})
    package = {"format_version": 1, "task": "A2_camera_visible_sign_review", "seed": seed,
               "source_manifest_sha256": sha256_file(raw_path), "queue_sha256": sha256_file(queue_path),
               "selection": "v1_error_enriched_development; not representative prevalence or final test",
               "clips": mapping}
    (output / "private_manifest.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "prepared_unreviewed", "clips": len(mapping), "directory": str(output.resolve())}


def validate(raw_path: Path, package_path: Path, annotations_path: Path) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = _raw_records(raw)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("format_version") != 1 or package["source_manifest_sha256"] != sha256_file(raw_path):
        raise ValueError("Annotation package references a different raw manifest or format")
    clips = {record["review_id"]: record for record in package["clips"]}
    if not clips or len(clips) != len(package["clips"]):
        raise ValueError("Annotation package requires unique review identifiers")
    for clip in clips.values():
        source = records[clip["clip_id"]]
        if source["split"] not in {"train", "validation"} or any(
            clip[key] != source[key] for key in ("subject", "split", "sha256")
        ) or clip["duration_seconds"] != source["media"]["duration_seconds"]:
            raise ValueError("Annotation package source provenance mismatch")
        video = (package_path.parent / clip["video"]).resolve()
        if not video.is_relative_to((package_path.parent / "videos").resolve()):
            raise ValueError("Anonymous video must stay in the package videos directory")
        if sha256_file(video) != clip["sha256"]:
            raise ValueError("Anonymous video checksum changed")
    intervals = defaultdict(list)
    counts, unreviewed = Counter(), 0
    with annotations_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Annotation CSV header differs from the A2 schema")
        for row in reader:
            if row["review_id"] not in clips or any(row[key] not in values for key, values in OPTIONS.items()):
                raise ValueError("Annotation has an unknown review identifier or category")
            start, end = float(row["start_s"]), float(row["end_s"])
            duration = clips[row["review_id"]]["duration_seconds"]
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
                raise ValueError("Annotation interval is outside the source duration")
            if row["review_status"] == "reviewed":
                if not row["reviewer"].strip() or not row["notes"].strip():
                    raise ValueError("Reviewed intervals require a reviewer and observable evidence notes")
                if row["visibility"] != "usable" and row["state"] != "uncertain":
                    raise ValueError("Unavailable visual evidence must keep state uncertain")
                counts[row["state"]] += 1
            else:
                unreviewed += 1
            intervals[row["review_id"]].append((start, end))
    for review_id, clip in clips.items():
        cursor = 0.0
        for start, end in sorted(intervals[review_id]):
            if abs(start - cursor) > 1e-6:
                raise ValueError("Annotation intervals contain overlaps or gaps")
            cursor = end
        if abs(cursor - clip["duration_seconds"]) > 1e-6:
            raise ValueError("Annotation intervals must cover every selected clip")
    return {"schema_valid": True, "clips": len(clips), "intervals": sum(map(len, intervals.values())),
            "unreviewed_intervals": unreviewed, "reviewed_states": dict(counts),
            "status": "review_pending" if unreviewed else "review_complete_requires_adjudication",
            "annotations_sha256": sha256_file(annotations_path), "test_included": False,
            "ground_truth_scope": "visible signs only; not EEG onset; not automatically imported for training"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--raw", required=True, type=Path)
    create.add_argument("--queue", required=True, type=Path)
    create.add_argument("--out", required=True, type=Path)
    create.add_argument("--seed", type=int, default=42)
    check = sub.add_parser("validate")
    check.add_argument("--raw", required=True, type=Path)
    check.add_argument("--package", required=True, type=Path)
    check.add_argument("--annotations", required=True, type=Path)
    args = parser.parse_args()
    result = (prepare(args.raw, args.queue, args.out, args.seed) if args.command == "prepare"
              else validate(args.raw, args.package, args.annotations))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
