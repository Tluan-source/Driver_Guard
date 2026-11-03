"""Safely recover immutable Kaggle week-5 outputs and record every file hash."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath


def recover(archive_path: Path, destination: Path):
    destination = destination.resolve()
    checksums = {}
    with zipfile.ZipFile(archive_path) as archive:
        seen = set()
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            target = (destination / info.filename).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in info.filename
                    or ":" in info.filename or not target.is_relative_to(destination)
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.filename.casefold() in seen):
                raise ValueError(f"Unsafe or duplicate archive entry: {info.filename}")
            seen.add(info.filename.casefold())
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC failed: {corrupt}")
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = (destination / info.filename).resolve()
            content = archive.read(info)
            checksum = hashlib.sha256(content).hexdigest()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Refusing to overwrite a different recovered file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)
            checksums[info.filename] = checksum
    runs = sorted({match[1] for name in checksums
                   if (match := re.fullmatch(r"experiments/(fold[1-5]_(?:gru|dann)_seed4[234])/inference/model.pt", name))})
    expected = {f"fold{fold}_{mode}_seed{seed}" for fold in range(1, 6)
                for mode in ("gru", "dann") for seed in (42, 43, 44)}
    if set(runs) != expected:
        raise ValueError("Archive does not contain all thirty declared week-5 models")
    report = {"archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "crc_valid": True,
              "runs": runs, "n_runs": len(runs), "files_sha256": checksums,
              "scope": "restored historical outer-fold development outputs; not newly trained models",
              "original_test_opened": False, "source_code_present": any(name.endswith(".py") for name in checksums)}
    (destination / "recovery_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "files_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", default=Path("models/recovered_week5"), type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.archive, args.out), indent=2))
