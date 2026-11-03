"""Download the original CC BY 4.0 driver-fatigue EEG recordings from Figshare."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/5202739/versions/1"


def download(destination: Path, item: dict) -> dict:
    name = Path(item["name"])
    if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
        raise ValueError(f"Unsafe archive name: {name}")
    target = destination / name
    if not target.exists() or target.stat().st_size != item["size"]:
        temporary = target.with_suffix(".zip.part")
        urllib.request.urlretrieve(item["download_url"], temporary)
        temporary.replace(target)
    content = target.read_bytes()
    if len(content) != item["size"] or hashlib.md5(content).hexdigest() != item["computed_md5"]:
        raise ValueError(f"Integrity verification failed: {target}")
    subject_dir = destination / name.stem
    subject_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(target) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {relative}")
            entries.append({"path": entry.filename, "bytes": entry.file_size})
        archive.extractall(subject_dir)
    return {
        "archive": item["name"],
        "bytes": item["size"],
        "sha256": hashlib.sha256(content).hexdigest(),
        "md5": item["computed_md5"],
        "url": item["download_url"],
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/driver_fatigue_eeg"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with urllib.request.urlopen(ARTICLE, timeout=90) as response:
        article = json.load(response)
    if article["license"]["name"] != "CC BY 4.0":
        raise ValueError("Dataset license changed; inspect before downloading.")
    args.out.mkdir(parents=True, exist_ok=True)
    assets = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, args.out, item) for item in article["files"]]
        for future in as_completed(futures):
            asset = future.result()
            assets.append(asset)
            print(f"{len(assets)}/{len(article['files'])} {asset['archive']}: "
                  f"{[entry['path'] for entry in asset['files']]}", flush=True)
    manifest = {
        "title": article["title"],
        "doi": article["doi"],
        "dataset_url": "https://figshare.com/articles/dataset/5202739",
        "kaggle_mirror": "https://www.kaggle.com/datasets/jcxuitsme/eeg-driver-fatigue-detection",
        "download_date_local": "2026-10-01",
        "license": article["license"],
        "authors": [author["full_name"] for author in article["authors"]],
        "description": article["description"],
        "subject_count": len(assets),
        "split_rule": "Group by subject archive number; both states from one subject share a split.",
        "assets": sorted(assets, key=lambda asset: int(Path(asset["archive"]).stem)),
    }
    (args.out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out / "figshare_metadata.json").write_text(json.dumps(article, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {len(assets)} subjects in {args.out}")


if __name__ == "__main__":
    main()
