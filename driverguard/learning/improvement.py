"""Reproducible, subject-isolated EEG improvement and deployment export."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

from .cross_validation import fit_deployment, nested_loso, paired_comparison
from .data import dataset_manifest, dataset_metadata, load_sessions
from .fitting import fit_grouped_model
from .model import VigilancePredictor
from .train import write_json

FAMILY = "calibrated_linear"
CANDIDATES = {FAMILY: [
    {"classifier": "lda", "shrinkage": "auto", "calibration_folds": 3},
    {"classifier": "logistic", "C": 0.1, "calibration_folds": 3},
    {"classifier": "logistic", "C": 1.0, "calibration_folds": 3},
]}


def write_oof_predictions(path: Path, rows: list[dict]) -> None:
    """Retain rejected windows and original indices, with empty prediction cells."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cached_factory():
    cache = {}

    def fit(sessions, config, seed):
        key = (tuple(sorted(s.session_id for s in sessions)), json.dumps(config, sort_keys=True), seed)
        if key not in cache:
            cache[key] = fit_grouped_model(sessions, config, seed)
        return cache[key]

    return fit


def _verify_pairing(baseline, robust) -> None:
    original = {s.session_id: s for s in baseline}
    revised = {s.session_id: s for s in robust}
    if original.keys() != revised.keys():
        raise ValueError("Baseline and robust datasets must preserve the same sessions")
    for name, session in original.items():
        other = revised[name]
        if (session.subject != other.subject or session.x.shape != other.x.shape
                or not np.array_equal(session.y, other.y)):
            raise ValueError("Paired comparison requires identical subjects, windows and targets")
        if (session.window_end_seconds is not None and other.window_end_seconds is not None
                and not np.array_equal(session.window_end_seconds, other.window_end_seconds)):
            raise ValueError("Paired comparison requires identical window timestamps")


def improve(baseline_root, robust_root, out, *, bootstrap=1000, seed=42, threads=4):
    """Freeze a small search, export the deployment fit, then estimate generalization."""
    if bootstrap < 1 or threads < 1:
        raise ValueError("bootstrap and threads must be positive")
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output already contains an experiment: {output}; choose a new directory")
    baseline = load_sessions(baseline_root)
    robust = load_sessions(robust_root)
    _verify_pairing(baseline, robust)
    metadata = dataset_metadata(robust_root)
    if metadata.get("preprocessing") != "robust_v1" or metadata.get("target_type") != "binary":
        raise ValueError("Improvement requires the quality-aware robust_v1 binary fatigue dataset")
    if any(s.preprocessing != "legacy_car" for s in baseline):
        raise ValueError("The reference dataset must use legacy_car preprocessing")
    torch.set_num_threads(threads)
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    protocol = {
        "candidate_grids": copy.deepcopy(CANDIDATES), "seed": seed, "bootstrap": bootstrap,
        "feature_transform": "relative_log_power", "decision_threshold": 0.5,
        "selection": "inner mean-subject log loss, balanced accuracy tie-break",
        "calibration": "three subject-disjoint fit/calibration members; sigmoid of raw decision margins",
        "calibration_revision": "raw margins retain ranks lost by probability-to-clipped-logit conversion",
        "preprocessing": "fixed label-independent robust_v1",
        "deployment_selection": "full-cohort inner LOSO, completed before outer report inspection",
        "evaluation": "nested LOSO on one development cohort; no external test claim",
        "baseline_data": str(Path(baseline_root).resolve()),
        "robust_data": str(Path(robust_root).resolve()),
        "source_revision": revision, "source_file_hashes": source_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "dataset_manifest.json", dataset_manifest(robust))
    factories = {"legacy": _cached_factory(), "robust": _cached_factory()}
    label_threshold = float(metadata["label_threshold"])
    with threadpool_limits(limits=1):
        deployments = {}
        for variant, sessions, local_metadata, filename in (
            ("legacy", baseline, dataset_metadata(baseline_root), "legacy_model.pt"),
            ("robust", robust, metadata, "model.pt"),
        ):
            print(f"Selecting {variant} deployment from inner subject-held-out predictions", flush=True)
            local_selection = fit_deployment(
                sessions, candidates=CANDIDATES, fit_model=factories[variant], seed=seed,
                label_threshold=label_threshold, calibration="none", threshold_mode="fixed_0.5")[FAMILY]
            deployments[variant] = local_selection
            extra = {
                "dataset_manifest": dataset_manifest(sessions), "source_revision": revision,
                "source_file_hashes": source_hashes,
                "deployment_selection": {k: v for k, v in local_selection.items() if k != "model"},
                "evaluation_status": "development_cohort_fit_requires_external_validation",
            }
            local_selection["model"].save_checkpoint(output / filename, local_metadata, extra=extra)
            predictor = VigilancePredictor(output / filename)
            for session in sessions:
                expected = local_selection["model"].predict_session(session).score
                actual = predictor.predict_file(session.feature_path)
                if not np.allclose(expected, actual, rtol=1e-5, atol=1e-7, equal_nan=True):
                    raise RuntimeError("Exported checkpoint does not reproduce fitted predictions")
            selection_filename = "deployment_selection.json" if variant == "robust" else "legacy_deployment_selection.json"
            write_json(output / selection_filename, extra["deployment_selection"])
        selected = deployments["robust"]
        results = {}
        for variant, sessions in (("legacy", baseline), ("robust", robust)):
            print(f"Benchmarking {variant} with identical nested subject LOSO rules", flush=True)
            result = nested_loso(
                sessions, candidates=CANDIDATES, fit_model=factories[variant], seed=seed,
                label_threshold=label_threshold, bootstrap=bootstrap, calibration="none",
                threshold_mode="fixed_0.5", progress=lambda message: print(message, flush=True))
            results[variant] = result
            write_json(output / f"{variant}_nested_loso.json", result["report"])
            write_oof_predictions(output / f"{variant}_nested_loso_predictions.csv", result["rows"])
        comparison = paired_comparison(
            results["legacy"]["rows"], results["robust"]["rows"],
            label_threshold=label_threshold, bootstrap=bootstrap, seed=seed)
    summary = {
        "selected_config": selected["config"], "decision_threshold": 0.5,
        "checkpoint": str((output / "model.pt").resolve()),
        "legacy_checkpoint": str((output / "legacy_model.pt").resolve()),
        "legacy_selected_config": deployments["legacy"]["config"],
        "legacy": results["legacy"]["report"]["families"][FAMILY],
        "robust": results["robust"]["report"]["families"][FAMILY],
        "paired_comparison": comparison, "elapsed_seconds": time.perf_counter() - start,
        "limitations": [
            "The original 12 subjects are a development cohort, including previously inspected test subjects",
            "Outer folds isolate each subject but do not undo prior cohort-specific research decisions",
            "Deployment uses all development subjects; its in-sample predictions are not test metrics",
            "Recording-level labels cannot evaluate microsleep onset or false alerts per hour",
        ],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"checkpoint": summary["checkpoint"], "selected_config": selected["config"],
                      "legacy_subject_macro": summary["legacy"]["subject_macro"],
                      "robust_subject_macro": summary["robust"]["subject_macro"],
                      "coverage": summary["robust"]["coverage"]}, indent=2), flush=True)
    return summary
