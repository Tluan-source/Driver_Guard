"""Download the official DROZY research release without replacing existing files; verify ZIP CRC."""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import requests

from driverguard.execution import file_sha256

URL = "https://orbi.uliege.be/bitstream/2268/191620/4/DROZY.zip"
LICENSE = "https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf"


def download(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    if destination.exists():
        return
    for attempt in range(3):
        size = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={size}-"} if size else {}
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if size and response.status_code != 206:
                    raise RuntimeError("Server cannot resume this download; retained partial, no overwrite")
                if size and not response.headers.get("Content-Range", "").startswith(f"bytes {size}-"):
                    raise RuntimeError("Unexpected byte range")
                reported = size
                with partial.open("ab" if size else "xb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        size += len(chunk)
                        if size - reported >= 64 * 1024 * 1024:
                            print(f"downloaded {size / 1024**2:.0f} MiB", flush=True)
                            reported = size
            if not zipfile.is_zipfile(partial):
                raise RuntimeError("Download ended without a complete ZIP directory")
            partial.replace(destination)
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)


def audit_archive(archive: Path, extract_to: Path | None):
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        for item in entries:
            p = Path(item.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in item.filename:
                raise ValueError("Unsafe ZIP member")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not allowed")
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        if extract_to is not None:
            if extract_to.exists():
                raise FileExistsError("Extraction destination already exists; refusing to overwrite")
            extract_to.mkdir(parents=True)
            z.extractall(extract_to)
    return {"archive": str(archive), "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive), "zip_crc": "passed", "entries": len(entries),
            "uncompressed_bytes": sum(e.file_size for e in entries),
            "extracted_to": str(extract_to) if extract_to else None,
            "source_url": URL, "license_url": LICENSE,
            "use": "local research/evaluation; no third-party redistribution",
            "alignment_verified": False, "ready_for_eeg_supervision": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--extract-to", type=Path)
    p.add_argument("--audit", type=Path, required=True)
    args = p.parse_args()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    try:
        download(args.destination)
        report = {"status": "archive_verified", **audit_archive(args.destination, args.extract_to)}
    except Exception as exc:
        report = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}", "source_url": URL}
        args.audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    args.audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
