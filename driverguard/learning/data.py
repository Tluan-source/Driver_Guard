"""EEG feature contracts and subject-disjoint causal sequences."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..evaluation.splits import assert_disjoint, driver_split

FEATURE_KEY = "de_movingAve"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass
class Session:
    session_id: str
    subject: str
    x: np.ndarray
    y: np.ndarray
    feature_path: Path
    label_path: Path
    quality_valid: np.ndarray | None = None
    quality_reason: np.ndarray | None = None
    preprocessing: str = "legacy_car"
    channel_names: tuple[str, ...] | None = None
    window_end_seconds: np.ndarray | None = None


def read_eeg_features(path: str | Path, feature_key: str = FEATURE_KEY,
                      n_features: int | None = None) -> np.ndarray:
    """Return [time, features], flattened channel first, then frequency band."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            x = archive["x"]
    elif path.suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    else:
        mat = loadmat(path)
        if feature_key not in mat:
            raise ValueError(f"Missing {feature_key!r} in {path}")
        x = np.asarray(mat[feature_key])
        if x.ndim != 3 or x.shape[0] != 17 or x.shape[2] != 5:
            raise ValueError(f"Expected [17, time, 5] in {path}, got {x.shape}")
        x = x.transpose(1, 0, 2).reshape(x.shape[1], 85)
    if x.ndim != 2 or not len(x) or not x.shape[1] or (n_features and x.shape[1] != n_features):
        raise ValueError(f"Expected nonempty [time, {n_features or 'features'}] in {path}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite EEG features in {path}; invalid sensors must not be inferred")
    return np.asarray(x, dtype=np.float32)


def _archive_preprocessing(archive, path: Path) -> str:
    if "preprocessing" not in archive:
        return "legacy_car"
    value = np.asarray(archive["preprocessing"])
    if value.shape != () or value.dtype.kind not in "SU":
        raise ValueError(f"Preprocessing must be a scalar string in {path}")
    preprocessing = value.item()
    if isinstance(preprocessing, bytes):
        preprocessing = preprocessing.decode("utf-8")
    if preprocessing not in {"legacy_car", "robust_v1"}:
        raise ValueError(f"Unsupported EEG preprocessing {preprocessing!r} in {path}")
    return preprocessing


def _archive_channels(archive, path: Path) -> tuple[str, ...] | None:
    if "channel_names" not in archive:
        return None
    values = np.asarray(archive["channel_names"])
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "SU":
        raise ValueError(f"Invalid EEG channel names in {path}")
    channels = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)
    if any(not channel for channel in channels) or len(set(channels)) != len(channels):
        raise ValueError(f"Nonempty unique EEG channel names required in {path}")
    return channels


def read_feature_contract(
    path: str | Path, feature_key: str = FEATURE_KEY, n_features: int | None = None, *,
    expected_preprocessing: str | None = None, expected_channels: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read features and acceptance flags without loading any labels.

    Return (x, quality_valid, quality_reason, window_end_seconds). Legacy files
    default to all accepted; robust_v1 files must carry their quality contract.
    Invalid robust rows remain in place and are never usable model observations.
    """
    path = Path(path)
    x = read_eeg_features(path, feature_key, n_features)
    preprocessing, channels = "legacy_car", None
    accepted = np.ones(len(x), dtype=bool)
    reasons = np.full(len(x), "", dtype="U1")
    ends = None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            preprocessing = _archive_preprocessing(archive, path)
            channels = _archive_channels(archive, path)
            robust = preprocessing == "robust_v1"
            if robust and (channels is None or any(key not in archive for key in (
                "quality_valid", "quality_reason_code", "window_end_seconds",
            ))):
                raise ValueError(f"robust_v1 requires channel, quality, and timestamp metadata in {path}")
            if "quality_valid" in archive:
                accepted = np.asarray(archive["quality_valid"])
                if accepted.dtype.kind != "b" or accepted.shape != (len(x),):
                    raise ValueError(f"quality_valid must be a boolean window mask in {path}")
                accepted = accepted.copy()
            if "quality_reason_code" in archive:
                codes = np.asarray(archive["quality_reason_code"])
                if codes.shape != (len(x),) or codes.dtype.kind not in "iu" or np.any((codes < 0) | (codes > 7)):
                    raise ValueError(f"Invalid quality reason codes in {path}")
                if "quality_valid" not in archive or not np.array_equal(accepted, codes == 0):
                    raise ValueError(f"Quality acceptance and reason codes disagree in {path}")
                names = ((1, "too_many_bad_channels"), (2, "nonfinite"), (4, "flat_reference"))
                reasons = np.asarray([",".join(name for bit, name in names if int(code) & bit) for code in codes])
            elif not accepted.all():
                reasons = np.where(accepted, "", "invalid_sensor_quality")
            if "quality_bad_channel_mask" in archive:
                bad = np.asarray(archive["quality_bad_channel_mask"])
                if channels is None or bad.dtype.kind != "b" or bad.shape != (len(x), len(channels)):
                    raise ValueError(f"Invalid bad-channel quality mask in {path}")
            if "window_end_seconds" in archive:
                ends = np.asarray(archive["window_end_seconds"], dtype=np.float64)
                if (ends.shape != (len(x),) or not np.isfinite(ends).all() or np.any(ends < 0)
                        or np.any(np.diff(ends) <= 0)):
                    raise ValueError(f"Window timestamps must be finite and strictly increasing in {path}")
    if expected_preprocessing is not None and preprocessing != expected_preprocessing:
        raise ValueError(f"EEG preprocessing mismatch in {path}: expected {expected_preprocessing}, got {preprocessing}")
    if expected_channels is not None and channels != tuple(expected_channels):
        raise ValueError(f"EEG channel order mismatch in {path}")
    return x, accepted, reasons, ends


def load_sessions(root: str | Path, feature_key: str = FEATURE_KEY) -> list[Session]:
    root = Path(root)
    sessions = []
    if (root / "sessions.json").exists():
        metadata_path = root / "dataset_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        expected_channels = metadata.get("channel_names")
        expected_preprocessing = metadata.get("preprocessing")
        manifest = json.loads((root / "sessions.json").read_text(encoding="utf-8"))
        seen_paths = set()
        for item in manifest:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Session path must stay inside feature root")
            if path in seen_paths:
                raise ValueError("Duplicate feature path assigned to multiple sessions or subjects")
            seen_paths.add(path)
            x, accepted, reasons, ends = read_feature_contract(
                path, n_features=metadata.get("n_features"), expected_preprocessing=expected_preprocessing,
                expected_channels=expected_channels,
            )
            with np.load(path, allow_pickle=False) as archive:
                y = np.asarray(archive["y"], dtype=np.float32).reshape(-1)
                preprocessing = _archive_preprocessing(archive, path)
                channels = _archive_channels(archive, path)
            if len(x) != len(y) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
                raise ValueError(f"Invalid targets in {path}")
            if not item["session_id"] or not item["subject"]:
                raise ValueError("Nonempty session and subject identifiers required")
            sessions.append(Session(str(item["session_id"]), str(item["subject"]), x, y, path, path,
                                    accepted, reasons, preprocessing, channels, ends))
        if not sessions or len({s.session_id for s in sessions}) != len(sessions):
            raise ValueError("Nonempty unique session IDs required")
        if len({s.x.shape[1] for s in sessions}) != 1:
            raise ValueError("All sessions must share the feature schema")
        if len({s.preprocessing for s in sessions}) != 1 or len({s.channel_names for s in sessions}) != 1:
            raise ValueError("All sessions must share the preprocessing and EEG channel order")
        return sessions
    for path in sorted((root / "DE").glob("*.mat")):
        match = re.fullmatch(r"(\d+)_(\d{8})(?:_[A-Za-z0-9_]+)?", path.stem)
        if not match:
            raise ValueError(f"Cannot recover verified subject ID from {path.name}")
        label_path = root / "perclos_labels" / path.name
        x = read_eeg_features(path, feature_key)
        y = np.asarray(loadmat(label_path)["perclos"]).reshape(-1)
        if len(y) != len(x) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1 + 1e-6)):
            raise ValueError(f"Invalid/misaligned PERCLOS in {label_path}")
        sessions.append(Session(path.stem, str(int(match[1])), x, np.clip(y, 0, 1).astype(np.float32),
                                path, label_path))
    if not sessions:
        raise ValueError(f"No SEED-VIG DE/*.mat and perclos_labels/*.mat in {root}")
    return sessions


def split_sessions(sessions: list[Session], seed: int = 42) -> dict[str, list[str]]:
    if len({s.subject for s in sessions}) < 7:
        raise ValueError("Need at least 7 subjects for nonempty train/validation/test")
    train, val, test = driver_split([s.subject for s in sessions], seed=seed)
    if not val or len(test) < 2:
        raise ValueError("Split needs validation subjects and at least two test subjects for group bootstrap")
    assert_disjoint(train, val, test)
    return {"train": train, "validation": val, "test": test}


def dataset_metadata(root: str | Path) -> dict:
    path = Path(root) / "dataset_metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"target": "perclos", "target_type": "regression", "label_threshold": 0.35,
            "feature_extraction": "SEED-VIG precomputed DE movingAve, 17 channels x 5 bands",
            "limitations": ["PERCLOS is an eye-closure-derived vigilance proxy",
                            "Upstream moving-average causal properties are unverified"]}


def causal_windows(x: np.ndarray, context: int) -> np.ndarray:
    """Left-pad with the first available sample; never include future samples."""
    if context < 1 or not len(x):
        raise ValueError("Positive context and nonempty features required")
    padded = np.concatenate([np.repeat(x[:1], context - 1, axis=0), x])
    return np.stack([padded[i:i + context] for i in range(len(x))]).astype(np.float32)


def pack_sessions(sessions: list[Session], subjects: list[str], context: int):
    selected = [s for s in sessions if s.subject in subjects]
    if not selected:
        raise ValueError("Split contains no sessions")
    if any(s.quality_valid is not None and not np.all(s.quality_valid) for s in selected):
        raise ValueError("Rejected EEG windows cannot be packed by the legacy training path; use the improve workflow")
    return (
        np.concatenate([causal_windows(s.x, context) for s in selected]),
        np.concatenate([s.y for s in selected]),
        np.concatenate([np.repeat(s.subject, len(s.y)) for s in selected]),
        np.concatenate([np.repeat(s.session_id, len(s.y)) for s in selected]),
        np.concatenate([np.arange(len(s.y)) for s in selected]),
    )


def dataset_manifest(sessions: list[Session]) -> list[dict]:
    return [{"session_id": s.session_id, "subject": s.subject, "n_windows": len(s.y),
             "feature_sha256": hashlib.sha256(s.feature_path.read_bytes()).hexdigest(),
             "label_sha256": hashlib.sha256(s.label_path.read_bytes()).hexdigest(),
             "label_min": float(s.y.min()), "label_max": float(s.y.max())} for s in sessions]
