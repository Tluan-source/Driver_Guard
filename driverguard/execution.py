"""Build portable source and development-feature archives using the standard library."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(root: Path) -> dict:
    exported = root / "build_info.json"
    if exported.is_file():
        return json.loads(exported.read_text(encoding="utf-8"))
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"git_sha": sha, "working_tree_dirty": dirty, "provenance_source": "git"}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "working_tree_dirty": None, "provenance_source": "unavailable"}


def bundle_source(root: Path, output: Path) -> dict:
    root = root.resolve()
    files = sorted((root / "driverguard").rglob("*.py"))
    files += sorted((root / "scripts").rglob("*.py"))
    files += [root / p for p in ("pyproject.toml", "README.md", "configs/default.yaml")]
    files += sorted((root / "kaggle").glob("*.py"))
    files += sorted((root / "kaggle").glob("*.ipynb"))
    if (root / "configs/project.json").is_file():
        files.append(root / "configs/project.json")
    # Training configurations are public experiment settings, kept separate
    # from private user/auth configuration at configs/users.yaml.
    for suffix in ("*.json", "*.yaml", "*.yml"):
        files += sorted((root / "configs" / "training").rglob(suffix))
    if not files or any(not path.is_file() for path in files):
        raise ValueError("Source tree is incomplete")
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("Source files must stay inside the source root")
    info = {**source_provenance(root), "format_version": 1,
            "files": {p.relative_to(root).as_posix(): file_sha256(p) for p in files}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("build_info.json", json.dumps(info, indent=2) + "\n")
    return {"archive": str(output), "files": len(files) + 1, "sha256": file_sha256(output)}


def _feature_path(manifest: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest.parent / candidate
    root = manifest.parent.resolve()
    path = candidate.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Features must stay inside the manifest directory")
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise ValueError(f"Missing or unsupported feature file: {path}")
    return path


def bundle_camera_features(manifest: Path, output: Path) -> dict:
    manifest = manifest.resolve()
    original = json.loads(manifest.read_text(encoding="utf-8"))
    contract, records = original.get("input_contract"), original.get("clips")
    if not isinstance(contract, dict) or not isinstance(records, list) or not records:
        raise ValueError("Expected a camera NPZ manifest with input_contract and clips")
    subjects = {split: set() for split in ("train", "validation", "test")}
    identifiers = set()
    selected = []
    for record in records:
        split, subject, identifier = record.get("split"), record.get("subject"), record.get("clip_id")
        if split not in subjects or not isinstance(subject, str) or not subject:
            raise ValueError("Every record requires a recognized split and subject")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Every record requires a unique clip_id")
        identifiers.add(identifier)
        subjects[split].add(subject)
        if split != "test":
            selected.append(record)
    names = list(subjects)
    if any(subjects[a] & subjects[b] for i, a in enumerate(names) for b in names[i + 1:]):
        raise ValueError("Subject leakage across splits")
    if not subjects["train"] or not subjects["validation"]:
        raise ValueError("Train and validation must both contain subjects")
    portable, entries, seen = [], [], set()
    for record in selected:
        path = _feature_path(manifest, record["features"])
        relative = path.relative_to(manifest.parent).as_posix()
        if relative in seen:
            raise ValueError("Duplicate NPZ paths")
        seen.add(relative)
        digest = file_sha256(path)
        if record.get("features_sha256", digest) != digest:
            raise ValueError(f"Feature checksum mismatch: {path}")
        row = copy.deepcopy(record)
        row["features"] = relative
        row["features_sha256"] = digest
        portable.append(row)
        entries.append((path, relative))
    converted = {"input_contract": copy.deepcopy(contract), "clips": portable,
                 "limitations": original.get("limitations", []),
                 "export": {"purpose": "development_train_validation_only", "test_included": False,
                            "original_feature_manifest_sha256": file_sha256(manifest),
                            "record_counts": dict(Counter(r["split"] for r in portable)),
                            "source_subject_counts": {k: len(v) for k, v in subjects.items()}}}
    # Multiclass training must retain the explicit semantics of 0/1/2.
    for key in ("class_names", "label_mode", "label_mapping", "dataset_version", "split_version"):
        if key in original:
            converted[key] = copy.deepcopy(original[key])
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in entries:
            archive.write(path, relative)
        archive.writestr("manifest.json", json.dumps(converted, indent=2) + "\n")
    return {"archive": str(output), "sha256": file_sha256(output), **converted["export"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("bundle-source", help="Export code/config; no raw data, models or secrets")
    source.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    source.add_argument("--out", required=True, type=Path)
    feature = commands.add_parser("bundle-camera", help="Export NPZ train/validation; never open test NPZs")
    feature.add_argument("--manifest", required=True, type=Path)
    feature.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = (bundle_source(args.root, args.out) if args.command == "bundle-source"
              else bundle_camera_features(args.manifest, args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
