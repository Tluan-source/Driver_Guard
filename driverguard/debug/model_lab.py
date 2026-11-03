"""Model Lab for ui/test-console: run trained camera checkpoints on stored features or live signals.

    list_models()                 GRU/GRU+DANN inference checkpoints + CAM-01/CAM-02/v1 camera heads
    list_clips()                  feature clips of train/validation (test split is never exposed)
    replay(model_id, clip_id)     causal windows over one clip -> probability timeline + clip summary
    LiveModel(model_id)           same checkpoint fed by live FrameSignals (webcam/synthetic)

Only reads checkpoints/metrics/features already in the repository; nothing is trained, tuned or
written. Validation numbers were used during development and are shown as such, never as test.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import REPO_ROOT, DriverGuardConfig, resolve_path
from ..schemas import FrameSignals

FEATURE_ROOT = REPO_ROOT / "data" / "features"
MODEL_GLOBS = (
    "runs/gru_dann/*/*/inference/model.pt",
    "models/from_kaggle/*/*/inference/model.pt",
    "models/camera_*/model.pt",
    "models/camera_*/*/model.pt",
)
EXPOSED_SPLITS = ("train", "validation")
SMOOTH_S = 300.0
DEFAULT_STRIDE = 25  # 2.5 s, same stride as the validation protocol
_BATCH = 64


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _resolve_repo(rel: str) -> Path:
    path = (REPO_ROOT / rel).resolve()
    if REPO_ROOT not in path.parents:
        raise ValueError("Đường dẫn nằm ngoài repository")
    return path


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------------------- models
def _validation_metrics(checkpoint: Path) -> dict[str, float | None]:
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "inference" else checkpoint.parent
    summary = _read_json(run_dir / "metrics" / "summary.json") or _read_json(run_dir / "metrics.json")
    window = summary.get("metrics", {}).get("validation", {}).get("window", {})
    window = window.get("overall_accepted_only", window)
    return {"accuracy": _num(window.get("accuracy")), "macro_f1": _num(window.get("macro_f1")),
            "auroc": _num(window.get("auroc_macro", window.get("auroc"))),
            "coverage": _num(summary.get("metrics", {}).get("validation", {}).get("window", {}).get("coverage"))}


def _is_smoke(checkpoint: Path) -> bool:
    run_dir = checkpoint.parent.parent
    protocol = _read_json(run_dir / "metrics" / "protocol.json")
    return "smoke" in _rel(checkpoint).lower() or protocol.get("purpose") == "smoke"


@dataclass
class LoadedModel:
    id: str
    family: str  # "gru_dann" | "camera"
    predictor: Any
    context: int
    class_names: tuple[str, ...]
    threshold: float | None
    input_contract: dict

    def predict_batch(self, xs: list[np.ndarray], ts: list[np.ndarray], vs: list[np.ndarray]):
        """-> list of (probabilities | None, reason | None, valid_fraction)."""
        if self.family == "camera":
            out = []
            for x, t, v in zip(xs, ts, vs, strict=True):
                r = self.predictor.predict_window(x, t, v)
                probs = None if r.score is None else (1.0 - r.score, r.score)
                out.append((probs, r.reason, r.valid_fraction))
            return out
        from ..learning.camera.dann.data import quality_gate

        out: list = [None] * len(xs)
        accepted = []
        for i, (x, t, v) in enumerate(zip(xs, ts, vs, strict=True)):
            q = quality_gate(x, t, v, self.context)
            if q.reason is None:
                accepted.append(i)
            out[i] = (None, q.reason, q.valid_fraction)
        for start in range(0, len(accepted), _BATCH):
            idx = accepted[start:start + _BATCH]
            batch = torch.as_tensor(np.stack([xs[i] for i in idx]), dtype=torch.float32)
            with torch.inference_mode():
                probs = self.predictor.model(batch).numpy()
            for i, p in zip(idx, probs, strict=True):
                ok = bool(np.isfinite(p).all())
                out[i] = (tuple(float(v) for v in p) if ok else None,
                          None if ok else "nonfinite_prediction", out[i][2])
        return out


def _describe(path: Path) -> dict[str, Any] | None:
    try:
        meta = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 — unreadable files are just not listed
        return None
    if not isinstance(meta, dict):
        return None
    arch = meta.get("architecture")
    rel = _rel(path)
    if arch in ("camera_gru_dann", "camera_gru_ablation"):
        cfg = meta.get("model_config", {})
        ctx = int(meta.get("context_frames") or 0)
        family, classes = "gru_dann", list(meta.get("class_names", []))
        mode = "GRU+DANN" if arch == "camera_gru_dann" else "GRU"
        label_mode, pooling, threshold = meta.get("label_mode"), cfg.get("pooling", "frame"), None
    elif arch in ("camera_gru", "camera_temporal_logistic", "camera_multicue_gru"):
        ctx, family, classes = int(meta.get("context") or 0), "camera", ["alert", "drowsy"]
        mode = {"camera_gru": "Camera GRU v1", "camera_temporal_logistic": "CAM-01 logistic",
                "camera_multicue_gru": f"CAM-02 {meta.get('cue_set', '')}".strip()}[arch]
        label_mode, pooling, threshold = "binary_pilot", "frame", _num(meta.get("decision_threshold"))
    else:
        return None
    parts = rel.split("/")
    group = parts[2] if parts[0] == "runs" else parts[2] if parts[1] == "from_kaggle" else parts[1]
    run_dir = path.parent.parent if path.parent.name == "inference" else path.parent
    fold = _read_json(run_dir / "week5_provenance.json").get("outer_fold")
    return {
        "id": rel, "family": family, "architecture": arch, "mode": mode, "group": group,
        "name": parts[-3] if parts[-2] == "inference" else parts[-2],
        "label_mode": label_mode, "class_names": classes, "pooling": pooling,
        "context_s": ctx / 10.0, "threshold": threshold,
        "smoke": _is_smoke(path), "validation": _validation_metrics(path),
        "cross_validation": fold is not None, "outer_fold": fold,
        "evaluation_scope": f"CV development fold {fold}" if fold is not None else "development",
        "has_saved_predictions": (path.parent.parent / "predictions" / "validation_windows.csv").is_file(),
    }


@lru_cache(maxsize=1)
def _model_index(_stamp: int) -> tuple[dict, ...]:
    seen, out = set(), []
    for pattern in MODEL_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            info = _describe(path)
            if info is not None:
                out.append(info)
    return tuple(out)


def list_models() -> list[dict[str, Any]]:
    stamp = int(time.time() // 60)  # rescan at most once a minute
    return [dict(m) for m in _model_index(stamp)]


@lru_cache(maxsize=4)
def load_model(model_id: str) -> LoadedModel:
    path = _resolve_repo(model_id)
    info = _describe(path)
    if info is None:
        raise ValueError(f"Không phải checkpoint camera hỗ trợ: {model_id}")
    if info["family"] == "gru_dann":
        from ..learning.camera.dann.inference import DANNPredictor

        p = DANNPredictor(path)
        return LoadedModel(model_id, "gru_dann", p, int(p.context), tuple(p.class_names), None,
                           p.meta["input_contract"])
    from ..learning.camera.model import CameraPredictor

    p = CameraPredictor(path)
    return LoadedModel(model_id, "camera", p, int(p.context), ("alert", "drowsy"),
                       float(p.decision_threshold), p.input_contract)


def contract_check(model: LoadedModel, extraction_signature: str | None) -> dict[str, Any]:
    """Compare preprocessing; the source manifest hash differs by design across datasets."""
    if not extraction_signature:
        return {"ok": None, "mismatch": [], "note": "clip không ghi chữ ký trích xuất"}
    try:
        clip = json.loads(extraction_signature)
    except ValueError:
        return {"ok": False, "mismatch": ["extraction_signature"], "note": "không đọc được"}
    skip = {"source_manifest_sha256"}
    keys = (set(clip) | set(model.input_contract)) - skip
    mismatch = sorted(k for k in keys if clip.get(k) != model.input_contract.get(k))
    return {"ok": not mismatch, "mismatch": mismatch,
            "note": "khớp tiền xử lý (bỏ qua hash manifest nguồn)" if not mismatch else "lệch tiền xử lý"}


# ---------------------------------------------------------------------------------------- clips
def _feature_sets() -> list[Path]:
    return sorted(FEATURE_ROOT.glob("*/manifest.json"))


@lru_cache(maxsize=8)
def _manifest(path_str: str, _mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> dict:
    return _manifest(str(path), path.stat().st_mtime)


def _clip_path(manifest_path: Path, record: dict) -> Path:
    raw = Path(str(record.get("features", "")))
    for candidate in (raw if raw.is_absolute() else manifest_path.parent / raw, manifest_path.parent / raw.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy feature của {record.get('clip_id')}")


def list_clips() -> dict[str, Any]:
    sets = []
    for mpath in _feature_sets():
        try:
            m = _load_manifest(mpath)
        except (OSError, ValueError):
            continue
        classes = list(m.get("class_names") or ["alert", "drowsy"])
        hidden = {}
        clips = []
        for r in m.get("clips", []):
            split = r.get("split")
            if split not in EXPOSED_SPLITS:
                hidden[split] = hidden.get(split, 0) + 1
                continue
            label = r.get("label")
            clips.append({
                "clip_id": r.get("clip_id"), "subject": r.get("subject"), "split": split,
                "label": label, "label_name": classes[label] if isinstance(label, int) and label < len(classes) else None,
                "duration_s": round((r.get("n_frames") or 0) / 10.0, 1),
                "valid_fraction": _num(r.get("valid_fraction")), "illumination": r.get("illumination"),
            })
        sets.append({"id": mpath.parent.name, "class_names": classes, "dataset_version": m.get("dataset_version"),
                     "clips": clips, "hidden_splits": hidden})
    return {"feature_sets": sets}


def _task_label(model: LoadedModel, manifest_classes: list[str], label: Any) -> int | None:
    """Map a manifest label to the model's class index; None when the class is outside the task."""
    if not isinstance(label, int) or label >= len(manifest_classes):
        return None
    name = manifest_classes[label]
    return model.class_names.index(name) if name in model.class_names else None


def _saved_predictions(model_id: str, clip_id: str) -> dict[int, list[float]]:
    csv_path = _resolve_repo(model_id).parent.parent / "predictions" / "validation_windows.csv"
    if not csv_path.is_file():
        return {}
    out = {}
    with csv_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("clip_id") == clip_id and row.get("prob_0"):
                keys = sorted(k for k in row if k.startswith("prob_"))
                out[int(row["end_ms"])] = [float(row[k]) for k in keys]
    return out


def _causal_smooth(t_s: list[float], probs: list[tuple | None], window_s: float = SMOOTH_S,
                   precision: int | None = 4):
    """Mean of accepted window probabilities that ended within the last `window_s` seconds."""
    out, buf = [], deque()
    for t, p in zip(t_s, probs, strict=True):
        if p is not None:
            buf.append((t, np.asarray(p)))
        while buf and buf[0][0] <= t - window_s:
            buf.popleft()
        out.append([(float(v) if precision is None else round(float(v), precision))
                    for v in np.mean([b[1] for b in buf], axis=0)] if buf else None)
    return out


def replay(model_id: str, feature_set: str, clip_id: str, stride: int = DEFAULT_STRIDE,
           precision: int | None = 4) -> dict[str, Any]:
    started = time.perf_counter()
    stride = int(min(max(stride, 5), 600))
    model = load_model(model_id)
    mpath = FEATURE_ROOT / feature_set / "manifest.json"
    if mpath.parent.parent != FEATURE_ROOT or not mpath.is_file():
        raise ValueError(f"Không có feature set {feature_set}")
    manifest = _load_manifest(mpath)
    record = next((r for r in manifest.get("clips", []) if r.get("clip_id") == clip_id), None)
    if record is None:
        raise ValueError(f"Không có clip {clip_id}")
    if record.get("split") not in EXPOSED_SPLITS:
        raise ValueError("Split test bị khóa trong console: chỉ mở sau khi ứng viên đã đóng băng.")
    classes = list(manifest.get("class_names") or ["alert", "drowsy"])
    with np.load(_clip_path(mpath, record), allow_pickle=False) as data:
        x, ts, valid = data["x"], data["ts_ms"], data["valid"]
        signature = str(data["extraction_signature"].item()) if "extraction_signature" in data else None
    ctx = model.context
    ends = list(range(ctx, len(x) + 1, stride))
    results = model.predict_batch([x[e - ctx:e] for e in ends], [ts[e - ctx:e] for e in ends],
                                  [valid[e - ctx:e] for e in ends])
    t_s = [float(ts[e - 1]) / 1000.0 for e in ends]
    probs = [r[0] for r in results]
    task = _task_label(model, classes, record.get("label"))
    thr = model.threshold

    def decide(p):
        if p is None:
            return None
        return int(p[1] >= thr) if thr is not None else int(np.argmax(p))

    smooth = _causal_smooth(t_s, probs, precision=precision)
    preds = [decide(p) for p in probs]
    spreds = [decide(p) for p in smooth]
    acc = [p for p in preds if p is not None]
    sacc = [p for p in spreds if p is not None]
    mean_p = np.mean([p for p in probs if p is not None], axis=0) if acc else None

    saved = _saved_predictions(model_id, clip_id)
    diffs = [max(abs(a - b) for a, b in zip(probs[i], saved[int(ts[e - 1])], strict=False))
             for i, e in enumerate(ends) if probs[i] is not None and int(ts[e - 1]) in saved]
    reasons: dict[str, int] = {}
    for r in results:
        if r[1]:
            reasons[r[1]] = reasons.get(r[1], 0) + 1
    return {
        "model_id": model_id, "feature_set": feature_set, "class_names": list(model.class_names),
        "threshold": thr, "context_s": ctx / 10.0, "stride_s": stride / 10.0,
        "clip": {"clip_id": clip_id, "subject": record.get("subject"), "split": record.get("split"),
                 "label_name": classes[record["label"]] if isinstance(record.get("label"), int) else None,
                 "task_label": task, "duration_s": round(len(x) / 10.0, 1),
                 "valid_fraction": round(float(np.mean(valid)), 4) if len(valid) else 0.0},
        "windows": [{"t": t if precision is None else round(t, 1),
                     "p": None if p is None else [v if precision is None else round(v, precision) for v in p],
                     "pred": pr, "reason": r[1], "vf": round(r[2], 3)}
                    for t, p, pr, r in zip(t_s, probs, preds, results, strict=True)],
        "smoothed": smooth,
        "summary": {
            "n_windows": len(ends), "n_accepted": len(acc),
            "coverage": round(len(acc) / len(ends), 4) if ends else 0.0,
            "reasons": reasons,
            "mean_probabilities": None if mean_p is None else [round(float(v), 4) for v in mean_p],
            "clip_prediction": decide(mean_p) if mean_p is not None else None,
            "window_accuracy": round(float(np.mean([p == task for p in acc])), 4) if task is not None and acc else None,
            "smoothed_accuracy": round(float(np.mean([p == task for p in sacc])), 4) if task is not None and sacc else None,
            "first_prediction_s": next((t for t, p in zip(t_s, probs, strict=True) if p is not None), None),
        },
        "consistency": {"saved_windows": len(saved), "compared": len(diffs),
                        "max_abs_diff": round(max(diffs), 6) if diffs else None},
        "contract": contract_check(model, signature),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "note": "Split validation đã dùng khi phát triển kiến trúc; số liệu chỉ để kiểm thử, không phải test độc lập.",
    }


# ---------------------------------------------------------------------------------------- live
class LiveModel:
    """Feeds live FrameSignals into the checkpoint at its 10 Hz training cadence (past-only)."""

    def __init__(self, model_id: str, session_cfg: DriverGuardConfig):
        from ..learning.camera.features import CameraFrameBuffer, sha256_file

        self.model = load_model(model_id)
        c = self.model.input_contract
        self.cfg = DriverGuardConfig.model_validate({"quality": c["quality"], "low_light": c["low_light"]})
        self.buffer = CameraFrameBuffer(self.cfg, self.model.context)
        self.every = DEFAULT_STRIDE if self.model.family == "gru_dann" else 10
        self.warnings = []
        if session_cfg.low_light.model_dump() != c["low_light"]:
            self.warnings.append("Cấu hình low-light của console khác lúc train model.")
        try:
            if sha256_file(resolve_path(session_cfg.models.face_landmarker)) != c["face_model_sha256"]:
                self.warnings.append("Face Landmarker khác tài sản lúc train model.")
        except OSError:
            self.warnings.append("Không đọc được Face Landmarker để so hash.")
        self.reset()

    def reset(self) -> None:
        self.buffer.clear()
        self.new_bins = 0
        self.last: dict[str, Any] = {"p": None, "reason": "insufficient_history", "vf": 0.0, "t_ms": None}
        self.history: deque = deque()
        self._last_input_ts: int | None = None

    def update(self, sig: FrameSignals) -> dict[str, Any]:
        from ..learning.camera.dann.data import quality_gate
        from ..learning.camera.features import encode_frame

        t0 = time.perf_counter()
        ran = False
        previous = self._last_input_ts
        discontinuity = previous is not None and (sig.ts_ms <= previous or sig.ts_ms - previous > 300)
        if discontinuity:
            self.reset()
        self._last_input_ts = sig.ts_ms
        backwards = previous is not None and sig.ts_ms <= previous
        added = False if backwards else self.buffer.update(sig)
        x, ts, v = self.buffer.arrays()
        _, current_valid = encode_frame(sig, self.cfg)
        reason = "invalid_timestamps" if backwards else (
            "current_frame_invalid" if not current_valid else quality_gate(x, ts, v, self.model.context).reason)
        if added:
            self.new_bins += 1
        if reason is not None:
            self.last = {"p": None, "reason": reason,
                         "vf": float(np.mean(v)) if len(v) else 0.0, "t_ms": sig.ts_ms}
            self.history.clear()
        elif added:
            if self.new_bins >= self.every:
                self.new_bins = 0
                p, reason, vf = self.model.predict_batch([x], [ts], [v])[0]
                self.last = {"p": p, "reason": reason,
                             "vf": round(vf, 3), "t_ms": sig.ts_ms}
                ran = True
                if p is not None:
                    self.history.append((sig.ts_ms, np.asarray(p)))
        while self.history and self.history[0][0] <= sig.ts_ms - SMOOTH_S * 1000:
            self.history.popleft()
        smooth = np.mean([h[1] for h in self.history], axis=0) if self.history else None
        filled = len(self.buffer.arrays()[0])
        p = self.last["p"]
        thr = self.model.threshold
        pred = None if p is None else (int(p[1] >= thr) if thr is not None else int(np.argmax(p)))
        return {
            "model_id": self.model.id, "family": self.model.family, "class_names": list(self.model.class_names),
            "threshold": thr, "context_s": self.model.context / 10.0,
            "history_s": round(filled / 10.0, 1),
            "probabilities": None if p is None else [round(q, 6) for q in p], "prediction": pred,
            "label": None if pred is None else self.model.class_names[pred],
            "reason": self.last["reason"], "valid_fraction": self.last["vf"],
            "smoothed": None if smooth is None else [round(float(v), 4) for v in smooth],
            "n_smoothed": len(self.history), "ran": ran, "ms": round((time.perf_counter() - t0) * 1000, 2),
            "warnings": self.warnings,
        }
