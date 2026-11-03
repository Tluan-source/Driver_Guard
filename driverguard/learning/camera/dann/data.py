"""Verified manifests, causal windows and training-only preprocessing."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from driverguard.learning.camera.features import sha256_file
from driverguard.learning.camera.model import (
    MAX_GAP_MS,
    MIN_VALID_FRACTION,
    CameraPrediction,
    validate_feature_arrays,
    validate_input_contract,
)
from driverguard.learning.camera.train import CameraClip, CameraWindow

LABEL_MODES = {
    "binary_pilot": ("alert", "drowsy"),
    "three_class": ("alert", "low_vigilance", "drowsy"),
    # Full UTA recordings without the ambiguous low-vigilance class: source 0 -> 0, source 2 -> 1.
    "alert_drowsy": ("alert", "drowsy"),
}
# Label modes derived from a manifest declaring another class set: (source classes, source -> target).
DERIVED_LABEL_MODES = {"alert_drowsy": ("three_class", {0: 0, 2: 1})}
SPLITS = ("train", "validation", "test")


@dataclass
class Corpus:
    clips: list[CameraClip]
    records: list[dict]
    manifest: dict
    manifest_sha256: str
    feature_hashes: dict[str, str]


def load_corpus(path: str | Path, label_mode: str, splits=("train", "validation")) -> Corpus:
    """Check every split's metadata, but open feature files only for requested splits."""
    if label_mode not in LABEL_MODES:
        raise ValueError("Unknown label mode")
    if not splits or not set(splits) <= set(SPLITS) or len(set(splits)) != len(splits):
        raise ValueError("Invalid selected splits")
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_input_contract(manifest.get("input_contract"))
    records = manifest.get("clips")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest requires nonempty clips")
    declared = manifest.get("class_names")
    source_mode, remap = DERIVED_LABEL_MODES.get(label_mode, (label_mode, None))
    if declared is not None and tuple(declared) != LABEL_MODES[source_mode]:
        raise ValueError("Manifest class_names disagree with requested label mode")
    if source_mode == "three_class" and declared is None:
        raise ValueError("three_class requires explicit class_names: alert, low_vigilance, drowsy")
    partitions = {split: set() for split in SPLITS}
    identifiers, sources = set(), {}
    for record in records:
        split, subject, clip_id = (record.get(key) for key in ("split", "subject", "clip_id"))
        if split not in partitions or not isinstance(subject, str) or not subject:
            raise ValueError("Each clip needs a recognized split and nonempty subject")
        if not isinstance(clip_id, str) or not clip_id or clip_id in identifiers:
            raise ValueError("Clip identifiers must be nonempty and unique")
        identifiers.add(clip_id)
        partitions[split].add(subject)
        factor = record.get("synthetic_exposure_factor", 1.0)
        if split != "train" and factor != 1.0:
            raise ValueError("Synthetic exposure augmentation is allowed only in train")
        for key in ("source_clip_id", "source_sha256", "features_sha256"):
            value = record.get(key)
            if value:
                previous = sources.setdefault((key, value), (split, subject))
                if previous != (split, subject):
                    raise ValueError("Duplicate source/features crosses subject or split boundary")
    if any(partitions[a] & partitions[b] for a, b in combinations(SPLITS, 2)):
        raise ValueError("Subjects must be disjoint across train/validation/test")
    if any(not partitions[split] for split in splits):
        raise ValueError("Each selected split requires subjects")
    clips, selected, hashes = [], [], {}
    for record in records:
        if record["split"] not in splits or (remap is not None and record.get("label") not in remap):
            continue
        feature_path = Path(record["features"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        digest = sha256_file(feature_path)
        if record.get("features_sha256", digest) != digest:
            raise ValueError("Feature checksum disagrees with manifest")
        with np.load(feature_path, allow_pickle=False) as values:
            x = np.asarray(values["x"], dtype=np.float32)
            ts = np.asarray(values["ts_ms"])
            valid = np.asarray(values["valid"])
            label, subject, clip_id = (values[key].item() for key in ("label", "subject", "clip_id"))
            if "extraction_signature" in values:
                expected = json.dumps(manifest["input_contract"], sort_keys=True, separators=(",", ":"))
                if values["extraction_signature"].item() != expected:
                    raise ValueError("Feature extraction signature disagrees with input contract")
        validate_feature_arrays(x, ts, valid)
        if (isinstance(label, bool) or not isinstance(label, (int, np.integer))
                or label not in range(len(LABEL_MODES[source_mode]))):
            raise ValueError("Feature label is outside the explicit label mapping")
        if record.get("label", label) != label:
            raise ValueError("Feature label disagrees with manifest")
        if remap is not None:
            if label not in remap:
                raise ValueError("Feature label disagrees with manifest")
            label = remap[label]
        if record["subject"] != subject or record["clip_id"] != clip_id:
            raise ValueError("Feature identifiers disagree with manifest")
        hashes[clip_id] = digest
        selected.append(record)
        clips.append(CameraClip(x, ts, valid, int(label), subject, clip_id, record["split"],
                                str(record.get("illumination", "unknown"))))
    return Corpus(clips, selected, manifest, sha256_file(path), hashes)


def quality_gate(x: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray, context: int) -> CameraPrediction:
    validate_feature_arrays(x, ts_ms, valid)
    fraction = float(np.mean(valid)) if len(valid) else 0.0
    reason = None
    if len(x) != context:
        reason = "insufficient_history"
    elif not np.isfinite(x).all():
        reason = "nonfinite_features"
    elif not np.isfinite(ts_ms).all() or np.any(np.diff(ts_ms) <= 0):
        reason = "invalid_timestamps"
    elif np.any(np.diff(ts_ms) > MAX_GAP_MS):
        reason = "timestamp_gap"
    elif not valid[-1]:
        reason = "current_frame_invalid"
    elif fraction < MIN_VALID_FRACTION:
        reason = "insufficient_quality"
    return CameraPrediction(None, reason, fraction)


def partition_windows(corpus: Corpus, stride: int, context: int = 50) -> dict[str, list[CameraWindow]]:
    if stride < 1 or context < 1:
        raise ValueError("Stride and context must be positive")
    windows = []
    for index, clip in enumerate(corpus.clips):
        for end in range(context - 1, len(clip.x), stride):
            start = end + 1 - context
            quality = quality_gate(clip.x[start:end + 1], clip.ts_ms[start:end + 1],
                                   clip.valid[start:end + 1], context)
            windows.append(CameraWindow(index, end, quality.valid_fraction, quality.reason))
    return {split: [w for w in windows if corpus.clips[w.clip].split == split] for split in SPLITS}


def normalization_from_train(corpus: Corpus, windows: list[CameraWindow], context: int = 50
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Use each accepted training frame once; unavailable fields stay zero at inference."""
    masks = {}
    for window in windows:
        clip = corpus.clips[window.clip]
        if clip.split != "train":
            raise ValueError("Normalization may only use training windows")
        if window.reason is not None:
            continue
        selected = masks.setdefault(window.clip, np.zeros(len(clip.x), dtype=bool))
        selected[window.end + 1 - context:window.end + 1] = True
    count, total, square = (np.zeros(9, dtype=np.float64) for _ in range(3))
    for index, mask in masks.items():
        x = corpus.clips[index].x[mask].astype(np.float64)
        available = x[:, 9:18] * x[:, -1:]
        count += available.sum(0)
        total += (x[:, :9] * available).sum(0)
        square += (x[:, :9] ** 2 * available).sum(0)
    center = total / np.maximum(count, 1)
    variance = np.maximum(square / np.maximum(count, 1) - center ** 2, 0)
    scale = np.maximum(np.sqrt(variance), 1e-4)
    scale[count == 0] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, corpus: Corpus, windows: list[CameraWindow], domain_mapping: dict[str, int],
                 context: int = 50):
        self.corpus = corpus
        self.windows = [w for w in windows if w.reason is None]
        self.domains = domain_mapping
        self.context = context
        counts = Counter(w.clip for w in self.windows)
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for w in self.windows:
            c, r = corpus.clips[w.clip], corpus.records[w.clip]
            groups[c.subject][c.label][r.get("source_clip_id", c.clip_id)].add(w.clip)
        weights = []
        for w in self.windows:
            c, r = corpus.clips[w.clip], corpus.records[w.clip]
            classes = groups[c.subject]
            sources = classes[c.label]
            variants = sources[r.get("source_clip_id", c.clip_id)]
            weights.append(len(self.windows) / (len(groups) * len(classes) * len(sources)
                                                 * len(variants) * counts[w.clip]))
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        clip = self.corpus.clips[window.clip]
        x = clip.x[window.end + 1 - self.context:window.end + 1]
        return torch.from_numpy(x), clip.label, self.domains.get(clip.subject, -1), self.weights[index]
