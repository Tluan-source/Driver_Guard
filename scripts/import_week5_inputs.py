"""Extract the private Kaggle source/features bundle without executing its code."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def extract(archive_path, destination):
    destination = Path(destination).resolve()
    checksums = {}
    with zipfile.ZipFile(archive_path) as archive:
        seen = set()
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            target = (destination / entry.filename).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename or ":" in entry.filename
                    or not target.is_relative_to(destination) or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.filename.casefold() in seen):
                raise ValueError(f"Unsafe or duplicate archive member: {entry.filename}")
            seen.add(entry.filename.casefold())
        if archive.testzip() is not None:
            raise ValueError("Kaggle input archive has a CRC error")
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            payload = archive.read(entry)
            checksum = hashlib.sha256(payload).hexdigest()
            target = (destination / entry.filename).resolve()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Existing recovery file differs: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(payload)
            checksums[entry.filename] = checksum
    report = {"archive_sha256": hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
              "crc_valid": True, "file_count": len(checksums), "files_sha256": checksums,
              "scope": "original private Kaggle source and development features; preserved unchanged"}
    (destination / "input_recovery_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "files_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/recovery_week5"), type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.out), indent=2))
