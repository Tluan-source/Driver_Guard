"""Read-only project inventory. Never deserialize weights or expose arbitrary files.

Only known metadata fields are returned. All paths are repository-relative and
the caller must enforce the fleet_manager role.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 8_000_000:
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        # A trainer can be between writes. The next refresh will retry.
        return {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _relative(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _metrics(value: Any) -> dict[str, float | None]:
    value = value if isinstance(value, dict) else {}
    coverage = value.get("coverage")
    for name in ("window", "overall_accepted_only"):
        if isinstance(value.get(name), dict):
            value = value[name]
            coverage = value.get("coverage", coverage)
    result = {name: _number(value.get(name)) for name in ("accuracy", "macro_f1", "auroc", "coverage")}
    result["coverage"] = _number(coverage)
    result["auroc"] = _number(value.get("auroc", value.get("auroc_macro")))
    return result


def _history(value: Any) -> list[dict[str, float | None]]:
    if isinstance(value, dict):
        value = value.get("history", [])
    if not isinstance(value, list):
        return []
    keys = ("epoch", "train_loss", "validation_loss", "val_loss", "task_loss", "domain_loss", "train_domain_loss",
            "validation_accuracy", "validation_macro_f1", "val_accuracy", "val_macro_f1")
    return [{key: _number(row.get(key)) for key in keys if key in row}
            for row in value[-500:] if isinstance(row, dict)]


def _read_history(path: Path) -> list[dict[str, float | None]]:
    try:
        return _history(json.loads(path.read_text(encoding="utf-8"))) if path.stat().st_size < 8_000_000 else []
    except (OSError, ValueError):
        return []


def _run(path: Path, root: Path) -> dict[str, Any] | None:
    relative = _relative(path, root)
    if relative is None:
        return None
    data = _json(path)
    if not data:
        return None
    protocol = _json(path.parent / "protocol.json")
    config = protocol.get("config") if isinstance(protocol.get("config"), dict) else {}
    context_seconds = _number(config.get("context_seconds", data.get("context_seconds")))
    label_mode = config.get("label_mode", data.get("label_mode"))
    matches_task = (data.get("architecture") in ("camera_gru_dann", "camera_gru_ablation")
                    and context_seconds == 120 and label_mode == "alert_drowsy")
    family = "GRU+DANN" if data.get("architecture") == "camera_gru_dann" else "GRU" if data.get("architecture") == "camera_gru_ablation" else "CAM-02" if "cam02" in relative else "GRU"
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    validation = _metrics(metrics.get("validation", data.get("validation", data.get("validation_metrics"))))
    test = _metrics(metrics.get("test", data.get("test", data.get("test_metrics"))))
    history = _history(data.get("history")) or _read_history(path.parent / "history.json")
    run_dir = path.parent.parent if path.parent.name == "metrics" else path.parent
    smoke = "smoke" in run_dir.name.lower() or protocol.get("purpose") == "smoke"
    provenance = _json(run_dir / "cv_provenance.json") or _json(run_dir / "week5_provenance.json")
    fold = provenance.get("outer_fold")
    fold_name = re.match(r"fold([1-5])_(?:gru|dann)_seed\d+$", run_dir.name)
    cross_validation = isinstance(fold, int) or fold_name is not None
    if fold is None and fold_name:
        fold = int(fold_name.group(1))
    checkpoint = run_dir / "model.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "inference/model.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "checkpoints/best.pt"
    # A high development score is a candidate only. Final success requires an
    # explicit independent test protocol, rather than a reused pilot test split.
    independent = (not cross_validation and protocol.get("independent_test") is True
                   and protocol.get("frozen_before_test") is True)
    seed = data.get("seed", config.get("seed"))
    return {"id": run_dir.relative_to(root).as_posix(), "name": run_dir.name,
            "family": family, "path": relative, "seed": seed if isinstance(seed, int) else None,
            "best_epoch": _number(data.get("best_epoch")), "epochs": len(history), "history": history,
            "validation": validation, "test": test, "independent_test": independent,
            "cross_validation": cross_validation, "outer_fold": fold,
            "context_seconds": context_seconds, "label_mode": label_mode,
            "evaluation_scope": "outer development fold" if cross_validation else "development",
            "eligible_for_selection": matches_task and not cross_validation and not smoke and checkpoint.is_file(),
            "checkpoint": _relative(checkpoint, root) if checkpoint.is_file() else None,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "status": "smoke" if smoke else "evaluated" if checkpoint.is_file() else "metrics_only"}


def build_project_overview(root: Path) -> dict[str, Any]:
    """Aggregate declared roadmap, metadata, and artifact presence from disk."""
    root = root.resolve()
    project = _json(root / "configs/project.json")
    audits = sorted((root / "outputs/audits").glob("*/project_audit.json"))
    audit = _json(audits[-1]) if audits else {}
    catalog = _json(root / "configs/datasets_gru_dann_eeg.json")
    dataset_catalog = {item.get("id"): item for item in catalog.get("datasets", []) if isinstance(item, dict)}
    dataset_audits = {item.get("path"): item for item in audit.get("datasets", []) if isinstance(item, dict)}
    datasets = []
    for category in ("raw", "features", "manifests"):
        base = root / "data" / category
        if not base.is_dir():
            continue
        for folder in sorted(base.iterdir()):
            relative = _relative(folder, root)
            if not folder.is_dir() or relative is None:
                continue
            count, size, extensions = 0, 0, Counter()
            for file in folder.rglob("*"):
                if file.is_file() and not file.is_symlink():
                    try:
                        size += file.stat().st_size
                        count += 1
                        extensions[file.suffix.lower()] += 1
                    except OSError:
                        continue
            manifest = _json(folder / "manifest.json")
            records = manifest.get("records", manifest.get("clips", manifest.get("items", [])))
            if not isinstance(records, list):
                records = []
            splits = Counter(str(row.get("split", "unknown")) for row in records if isinstance(row, dict))
            info = dataset_catalog.get(folder.name, {})
            proof = dataset_audits.get(relative, {})
            split_unit = "records"
            if not splits and isinstance(proof.get("split_subject_counts"), dict):
                splits = proof["split_subject_counts"]
                split_unit = "subjects"
            usable = any(ext in extensions for ext in (".npz", ".mp4", ".mov", ".m4v", ".edf", ".cnt", ".mat", ".jpg", ".jpeg", ".png"))
            usable = usable or (category == "manifests" and ".json" in extensions)
            datasets.append({"id": f"{category}/{folder.name}", "name": folder.name, "category": category,
                             "path": relative, "file_count": count, "bytes": size,
                             "extensions": [{"suffix": suffix, "count": n} for suffix, n in extensions.items()],
                             "records": len(records) or proof.get("records", 0), "splits": dict(splits), "split_unit": split_unit,
                             "subjects": proof.get("subjects"), "audit_status": proof.get("status"),
                             "status": "present" if usable else "awaiting_data",
                             "modalities": info.get("modalities", []), "cloud_note": info.get("kaggle", ""),
                             "manifest": _relative(folder / "manifest.json", root) if manifest else None})

    runs = []
    for base in (root / "models", root / "runs/gru_dann"):
        for path in sorted(set(base.rglob("metrics.json")) | set(base.glob("**/metrics/summary.json"))):
            relative = path.relative_to(base).as_posix().lower()
            if base.name == "models" and "camera" not in relative and "gru_dann" not in relative:
                continue
            run = _run(path, root)
            if run:
                runs.append(run)
    runs.sort(key=lambda item: item["modified_at"], reverse=True)
    checkpoints = []
    audit_checkpoints = {item.get("path"): item for item in audit.get("checkpoints", []) if isinstance(item, dict)}
    for path in sorted(set((root / "models").rglob("*.pt")) | set((root / "runs/gru_dann").rglob("*.pt"))):
        relative = _relative(path, root)
        if relative is None:
            continue
        proof = audit_checkpoints.get(relative, {})
        checkpoints.append({"name": path.name, "path": relative, "bytes": path.stat().st_size,
                            "family": "camera" if "camera" in relative or "gru_dann" in relative else "EEG",
                            "sha256": proof.get("sha256"), "load_status": proof.get("load_status", "not_audited"),
                            "inference_status": proof.get("inference_status", "not_audited")})
    stages = [{key: item.get(key, "") for key in ("id", "title", "state", "next")}
              for item in project.get("stages", []) if isinstance(item, dict)]
    blockers = [{key: item.get(key, "") for key in ("id", "severity", "summary", "action")}
                for item in audit.get("blockers", []) if isinstance(item, dict)]
    candidates = [run for run in runs if run["family"] in ("GRU", "GRU+DANN")
                  and run["eligible_for_selection"]]
    best = max(candidates, key=lambda item: item["validation"]["macro_f1"] or 0, default=None)
    passed = [run for run in candidates if run["independent_test"] and run["checkpoint"]
              and (run["test"]["accuracy"] or 0) > .85]
    cv = _json(root / "models/recovered_week5/cv_comparison.json")
    recovery_audit = _json(root / "outputs/recovery_week5/artifact_audit.json")
    cv_overview = {key: cv.get(key) for key in (
        "scope", "ensemble_rule", "models", "dann_minus_gru", "candidate_mode", "limitation")}
    cv_overview.update({"run_count": sum(run["cross_validation"] for run in runs),
                        "artifact_audit": recovery_audit.get("status", "not_available"),
                        "independent_acceptance": False})
    kaggle_files = sorted((root / "outputs/kaggle").glob("*/status.json"))
    kaggle_data = _json(kaggle_files[-1]) if kaggle_files else {}
    kaggle = {key: kaggle_data.get(key) for key in ("status", "url", "accelerator_requested", "accelerator_verified", "message")}
    kaggle["status"] = kaggle["status"] or "not_verified"
    kaggle["accelerator_requested"] = kaggle["accelerator_requested"] or "GPU T4 ×2"
    kaggle["source"] = _relative(kaggle_files[-1], root) if kaggle_files else None
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "timezone": "Asia/Bangkok", "project": "DriverGuard", "current_stage": project.get("current_stage"),
            "target": {"threshold": .85, "operator": ">", "metric": "window_accuracy", "achieved": bool(passed),
                       "context_seconds": 120,
                       "candidate": best["id"] if best else None,
                       "validation": best["validation"] if best else _metrics(None),
                       "note": "Validation chọn ứng viên; chỉ xác nhận mục tiêu bằng test độc lập đã đóng băng."},
            "stages": stages, "datasets": datasets, "runs": runs, "checkpoints": checkpoints,
            "development_cv": cv_overview,
            "blockers": blockers, "kaggle": kaggle,
            "audit": {"status": audit.get("status", "not_available"), "generated_at": audit.get("generated_at"),
                      "path": _relative(audits[-1], root) if audits else None},
            "folders": [{"path": path, "purpose": purpose} for path, purpose in (
                ("data/raw/", "Dữ liệu gốc"), ("data/features/", "Đặc trưng đã trích xuất"),
                ("data/manifests/", "Nhãn, split và provenance"), ("configs/training/", "Cấu hình huấn luyện"),
                ("models/", "Checkpoint và metrics"), ("runs/gru_dann/", "Lịch sử thí nghiệm"),
                ("outputs/audits/", "Báo cáo kiểm định"), ("outputs/kaggle/", "Trạng thái Kaggle"),
                ("kaggle/", "Notebook huấn luyện"), ("ui/dashboard/", "Dashboard quản lý"))]}
