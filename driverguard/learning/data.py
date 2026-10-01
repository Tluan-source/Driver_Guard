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


def load_sessions(root: str | Path, feature_key: str = FEATURE_KEY) -> list[Session]:
    root = Path(root)
    sessions = []
    if (root / "sessions.json").exists():
        metadata_path = root / "dataset_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        expected_channels = metadata.get("channel_names")
        manifest = json.loads((root / "sessions.json").read_text(encoding="utf-8"))
        seen_paths = set()
        for item in manifest:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Session path must stay inside feature root")
            if path in seen_paths:
                raise ValueError("Duplicate feature path assigned to multiple sessions or subjects")
            seen_paths.add(path)
            x = read_eeg_features(path, n_features=metadata.get("n_features"))
            with np.load(path, allow_pickle=False) as archive:
                y = np.asarray(archive["y"], dtype=np.float32).reshape(-1)
                if expected_channels is not None and (
                    "channel_names" not in archive or archive["channel_names"].tolist() != expected_channels
                ):
                    raise ValueError(f"EEG channel order mismatch in {path}")
            if len(x) != len(y) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
                raise ValueError(f"Invalid targets in {path}")
            if not item["session_id"] or not item["subject"]:
                raise ValueError("Nonempty session and subject identifiers required")
            sessions.append(Session(str(item["session_id"]), str(item["subject"]), x, y, path, path))
        if not sessions or len({s.session_id for s in sessions}) != len(sessions):
            raise ValueError("Nonempty unique session IDs required")
        if len({s.x.shape[1] for s in sessions}) != 1:
            raise ValueError("All sessions must share the feature schema")
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
