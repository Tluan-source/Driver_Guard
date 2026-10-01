"""Download paired SEED-VIG EEG features and PERCLOS labels from Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "mojahidmahin/seed-vig"
API = "https://www.kaggle.com/api/v1/datasets"
OFFICIAL = "https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html"


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def download(destination: Path, item: dict) -> dict:
    relative = Path(item["name"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/download/{DATASET}/{urllib.parse.quote(item['name'], safe='')}"
    url += "?datasetVersionNumber=1"
    if not target.exists() or target.stat().st_size != item["totalBytes"]:
        temporary = target.with_suffix(".mat.part")
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size != item["totalBytes"]:
            raise ValueError(f"Size mismatch: {target}")
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"path": item["name"], "bytes": item["totalBytes"], "sha256": digest, "url": url}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/seed_vig_features"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--license-agreement-approved", action="store_true",
                        help="Confirm an approved official SEED license agreement already exists")
    args = parser.parse_args()
    if not args.license_agreement_approved:
        parser.error("Official SEED access requires an approved institutional license agreement: "
                     "https://bcmi.sjtu.edu.cn/home/seed/downloads.html. A Kaggle mirror license "
                     "does not replace that agreement.")
    metadata = read_json(f"{API}/view/{DATASET}")
    if metadata["currentVersionNumber"] != 1:
        raise ValueError("Dataset version changed; inspect the new metadata before downloading.")
    entries: list[dict] = []
    token = ""
    while True:
        url = f"{API}/list/{DATASET}"
        if token:
            url += "?pageToken=" + urllib.parse.quote(token)
        page = read_json(url)
        entries.extend(page["datasetFiles"])
        token = page.get("nextPageToken", "")
        if not token:
            break
    selected = [item for item in entries if item["name"].startswith(("DE/", "perclos_labels/"))]
    feature_sessions = {Path(item["name"]).stem for item in selected if item["name"].startswith("DE/")}
    label_sessions = {
        Path(item["name"]).stem for item in selected if item["name"].startswith("perclos_labels/")
    }
    if not feature_sessions or feature_sessions != label_sessions:
        raise ValueError("EEG feature sessions and label sessions are not paired.")
    assets = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, args.out, item) for item in selected]
        for future in as_completed(futures):
            asset = future.result()
            assets.append(asset)
            print(f"{len(assets)}/{len(selected)} {asset['path']}", flush=True)
    manifest = {
        "dataset": DATASET,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET}",
        "dataset_version": 1,
        "download_date_local": "2026-10-01",
        "official_source": OFFICIAL,
        "citation": "Wei-Long Zheng and Bao-Liang Lu, A multimodal approach to estimating vigilance "
        "using EEG and forehead EOG. Journal of Neural Engineering 14(2):026017, 2017.",
        "mirror_license": metadata["licenseName"],
        "license_warning": "Kaggle uploader labels the mirror MIT. Original SEED terms govern the "
        "underlying recordings: use only after the official agreement is signed and approved, "
        "for academic research only. Do not redistribute.",
        "label": "Continuous PERCLOS from synchronized eye tracking, 0 to 1; vigilance proxy, "
        "not a clinical sleepiness diagnosis.",
        "features": "17 EEG channels x 885 time samples x 5 frequency bands; psd_movingAve, "
        "psd_LDS, de_movingAve, de_LDS. Use EEG features as input and paired PERCLOS as target.",
        "session_count": len(feature_sessions),
        "subject_count": len({session.split('_')[0] for session in feature_sessions}),
        "split_rule": "Group by numeric subject prefix; all sessions from one subject share a split.",
        "assets": sorted(assets, key=lambda asset: asset["path"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out / "kaggle_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Complete: {len(feature_sessions)} paired sessions in {args.out}")


if __name__ == "__main__":
    main()
