"""Prepare original driver-fatigue EEG as subject-grouped spectral sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from driverguard.learning.eeg import BANDS, CHANNEL_NAMES, cnt_recording_info, extract_cnt

PAPER = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188756"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unique_recordings(root: Path) -> tuple[list[Path], list[dict]]:
    grouped = {}
    duplicates = []
    for source in sorted(root.rglob("*.cnt")):
        if not source.parent.name.isdigit():
            raise ValueError(f"Cannot identify numeric subject for {source}")
        key = (int(source.parent.name), source.stem.lower())
        grouped.setdefault(key, []).append(source)
    canonical = []
    for key, sources in grouped.items():
        selected = max(sources, key=lambda source: len(source.relative_to(root).parts))
        hashes = {str(source): sha256(source) for source in sources}
        if len(set(hashes.values())) != 1:
            raise ValueError(f"Conflicting duplicate recordings for subject/state {key}")
        canonical.append(selected)
        if len(sources) > 1:
            duplicates.append({"selected": str(selected), "duplicates": [str(s) for s in sources if s != selected],
                               "sha256": hashes[str(selected)]})
    return sorted(canonical), duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/driver_fatigue_eeg"))
    parser.add_argument("--out", type=Path, default=Path("data/features/fatigue_eeg"))
    parser.add_argument("--window-seconds", type=float, default=4.0)
    args = parser.parse_args()
    sources, duplicates = unique_recordings(args.raw)
    if not sources:
        raise ValueError(f"No CNT recordings under {args.raw}; run download_driver_fatigue_eeg.py first")
    args.out.mkdir(parents=True, exist_ok=True)
    sessions = []
    for source in sources:
        subject = source.parent.name
        if not subject.isdigit():
            raise ValueError(f"Cannot identify numeric subject for {source}")
        if source.stem.lower() == "normal state":
            label, state = 0, "normal"
        elif source.stem.lower() == "fatigue state":
            label, state = 1, "fatigue"
        else:
            raise ValueError(f"Unsupported experimental state in {source}")
        session_id = f"subject_{int(subject):02d}_{state}"
        target = args.out / f"{session_id}.npz"
        recording = cnt_recording_info(source)
        x, ends, channels = extract_cnt(source, args.window_seconds)
        np.savez_compressed(
            target,
            x=x,
            y=np.full(x.shape[0], label, dtype=np.float32),
            window_end_seconds=ends,
            channel_names=np.asarray(channels),
        )
        sessions.append({
            "session_id": session_id,
            "subject": f"subject_{int(subject):02d}",
            "path": target.name,
            "windows": int(x.shape[0]),
            "state": state,
            "source": str(source),
            "source_cnt_sha256": sha256(source),
            **recording,
        })
        print(f"{session_id}: {recording['n_samples']} samples, {x.shape[0]} windows, "
              f"{x.shape[1]} features", flush=True)
    subject_ids = {session["subject"] for session in sessions}
    for subject in subject_ids:
        states = {session["state"] for session in sessions if session["subject"] == subject}
        if states != {"normal", "fatigue"}:
            raise ValueError(f"Subject {subject} lacks one experimental state")
    metadata = {
        "dataset": "Driver fatigue EEG, Min et al. 2017",
        "target": "fatigue",
        "target_type": "binary",
        "label_threshold": 0.5,
        "label_mapping": {"Normal state.cnt": 0, "Fatigue state.cnt": 1},
        "label_level": "recording-level experimental state; all windows inherit the state",
        "protocol": "12 healthy men aged 19-24 in a static simulator. Normal recording is the last "
        "5 minutes after 20 minutes driving. Fatigue recording is the last 5 minutes after 40-100 "
        "minutes, validated by Chalder/Li subjective fatigue scales, video/EOG signs, and performance.",
        "paper_url": PAPER,
        "paper_doi": "10.1371/journal.pone.0188756",
        "data_doi": "10.6084/m9.figshare.5202739.v1",
        "license": "CC BY 4.0, verified in original Figshare version 1 metadata",
        "citation": "Min J, Wang P, Hu J (2017). Driver fatigue detection through multiple entropy "
        "fusion analysis in an EEG-based system. PLoS ONE 12(12):e0188756.",
        "kaggle_mirror": "https://www.kaggle.com/datasets/jcxuitsme/eeg-driver-fatigue-detection",
        "channel_names": list(CHANNEL_NAMES),
        "bands": [{"name": name, "low_hz": low, "high_hz_exclusive": high} for name, low, high in BANDS],
        "feature_names": [f"{channel}_{name}_log10_power_uv2" for channel in CHANNEL_NAMES for name, _, _ in BANDS],
        "n_features": len(CHANNEL_NAMES) * len(BANDS),
        "window_seconds": args.window_seconds,
        "sample_rate": 1000.0,
        "sample_rate_hz": 1000.0,
        "feature_extraction": "Non-overlap recording-local windows; common average EEG reference; "
        "Welch PSD using 1-second Hann segments, 50% overlap, constant detrend; integrate [low, high) "
        "bands then log10 power in microvolt squared. Channel-major, band-minor order. No future filtering.",
        "cnt_decoding": "Explicit int32 required because original CNT numsamples header is malformed. "
        "int32 yields ~300000 samples / 1000Hz (~5min) matching the paper; int16 yields ~600000 and "
        "misreads high/low words. Always recompute sample count from event-table/data byte sizes "
        "and verify the physical file-size bound before reading each window. "
        "EOG HEOL/HEOR/VEOU/VEOL, mastoids A1/A2 and unused flat "
        "FT9/FT10/PO1/PO2 excluded, retaining the 30 effective EEG channels reported in the paper.",
        "reader_version": sessions[0]["mne_version"],
        "deduplicated_recordings": duplicates,
        "limitations": [
            "No synchronized camera files, eye-state labels, event onset annotations, or real-road validation",
            "Recording labels are experimental fatigue states, not clinical diagnoses",
            "Upstream CNT preprocessing by the data authors is not fully described in the archive",
            "Dataset includes only 12 young male subjects in one simulator",
        ],
        "subject_count": len(subject_ids),
        "session_count": len(sessions),
        "window_count": sum(session["windows"] for session in sessions),
        "split_rule": "Subject-disjoint; keep both normal and fatigue recordings of a person together",
    }
    (args.out / "sessions.json").write_text(json.dumps(sessions, indent=2) + "\n", encoding="utf-8")
    (args.out / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {len(sessions)} sessions, {len(subject_ids)} subjects in {args.out}")


if __name__ == "__main__":
    main()
