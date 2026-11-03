"""CAM-02 development experiment: fixed cue ablations/seeds, never open test NPZs."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from driverguard.execution import source_provenance
from driverguard.learning.camera.features import sha256_file
from driverguard.learning.camera.model import CONTEXT, FEATURE_DIM, CameraPredictor
from driverguard.learning.camera.multicue import CUE_SETS, MULTICUE_VERSION, CameraMultiCueGRU
from driverguard.learning.camera.train import (
    _clip_rows,
    _prediction_rows,
    _row_metrics,
    _scores,
    _WindowDataset,
    _write_csv,
    camera_window_weights,
    camera_windows,
    load_camera_manifest,
    select_camera_threshold,
)
from driverguard.learning.metrics import bootstrap_group_ci


def run_experiment(
    manifest_path: str | Path, out: str | Path, *, seeds: tuple[int, ...] = (42, 43, 44),
    cue_sets: tuple[str, ...] = tuple(CUE_SETS), epochs: int = 25, patience: int = 5,
    batch_size: int = 128, stride: int = 10, device: str = "cpu",
) -> dict:
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda or auto")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; enable a GPU accelerator or use cpu")
    torch_device = torch.device(selected_device)
    if selected_device == "cuda":
        # Keep export-to-CPU comparisons on float32 precision rather than TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty experiment directory")
    if (not seeds or len(set(seeds)) != len(seeds) or not cue_sets
            or len(set(cue_sets)) != len(cue_sets) or not set(cue_sets) <= set(CUE_SETS)
            or min(epochs, patience, batch_size) < 1):
        raise ValueError("Require unique seeds/cue sets and positive training settings")
    started = time.perf_counter()
    clips, manifest = load_camera_manifest(manifest_path, splits=("train", "validation"))
    windows = camera_windows(clips, stride)
    accepted = [w for w in windows if w.reason is None]
    partitions = {s: [w for w in accepted if clips[w.clip].split == s] for s in ("train", "validation")}
    for split, items in partitions.items():
        if {clips[w.clip].label for w in items} != {0, 1}:
            raise ValueError(f"Camera {split} requires accepted windows from both classes")
    train_data = _WindowDataset(clips, partitions["train"])
    validation = partitions["validation"]
    labels = np.asarray([clips[w.clip].label for w in validation])
    weights = camera_window_weights(clips, validation)
    provenance = source_provenance(Path(__file__).resolve().parents[3])
    protocol = {
        "run_id": "CAM-02", "test_evaluated": False, "early_onset_prediction": False,
        "label_source": "binary_video_level", "seeds": list(seeds), "cue_sets": list(cue_sets),
        "epochs": epochs, "patience": patience, "batch_size": batch_size, "stride": stride,
        "context": CONTEXT, "hidden_size_per_branch": 16, "dropout": 0.3,
        "learning_rate": 0.001, "weight_decay": 0.01,
        "epoch_selection": "validation subject/class/clip/window weighted log loss",
        "threshold_selection": "validation weighted balanced accuracy, fixed 0.05..0.95 grid",
        "loss_weighting": "equal subject then observed class then clip then window",
        "model_selection": "No automatic promotion; report all cue sets and seeds; full seed42 is reference",
        "coverage_denominator": "scheduled stride windows after 50-frame warmup; includes rejected windows",
        "feature_manifest_sha256": sha256_file(manifest_path),
        "input_contract": manifest["input_contract"],
        "subject_splits": {s: sorted({c.subject for c in clips if c.split == s}) for s in partitions},
        "implementation_sha256": {name: sha256_file(Path(__file__).parent / name) for name in (
            "multicue.py", "multicue_train.py", "model.py", "train.py")},
        "git_sha": provenance.get("git_sha"),
        "working_tree_dirty": provenance.get("working_tree_dirty"),
        "source_build": provenance,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": selected_device, "threads": torch.get_num_threads(),
                    "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if selected_device == "cuda" else None},
        "limitations": [
            "Exploratory development while A2 human annotation remains pending; not a final model.",
            "Validation is used for selection and is not an independent generalization estimate.",
            "No real-night, body pose, gaze tracking, vehicle telemetry or synchronized EEG inputs.",
            "Head orientation is not eye gaze; per-sample differences at 10 Hz are not blink kinetics.",
            "Ablation is observational; cue sets have different parameter counts and masks can be confounders.",
            "Scores are uncalibrated; quality gate is unchanged; warmup excluded from reported coverage.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    # Save protocol before training, including the complete ablation/seed budget.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    runs = []
    for cue_set in cue_sets:
        for seed in seeds:
            run_started = time.perf_counter()
            torch.manual_seed(seed)
            model = CameraMultiCueGRU(cue_set).to(torch_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            history, best_state = [], None
            best_loss, best_epoch, stale = float("inf"), 0, 0
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                for x, label, weight in loader:
                    x, label, weight = x.to(torch_device), label.to(torch_device), weight.to(torch_device)
                    optimizer.zero_grad()
                    loss = (F.binary_cross_entropy_with_logits(model(x), label.float(), reduction="none")
                            * weight.float()).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(x)
                scores = np.clip(_scores(model, clips, validation, batch_size), 1e-7, 1 - 1e-7)
                val_loss = float(np.average(-(labels * np.log(scores) + (1 - labels) * np.log1p(-scores)),
                                            weights=weights))
                history.append({"epoch": epoch, "train_loss": total_loss / len(train_data),
                                "validation_loss": val_loss})
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                print(f"CAM-02 {cue_set} seed={seed} epoch={epoch} val_loss={val_loss:.5f}", flush=True)
                if stale >= patience:
                    break
            model.load_state_dict(best_state)
            scores = _scores(model, clips, validation, batch_size)
            threshold = select_camera_threshold(labels, scores, weights)
            all_scores = _scores(model, clips, accepted, batch_size)
            rows = _prediction_rows(clips, windows, all_scores)
            clip_rows = _clip_rows(clips, rows)
            metrics = {s: {
                "window": _row_metrics([r for r in rows if r["split"] == s], threshold),
                "clip": _row_metrics([r for r in clip_rows if r["split"] == s], threshold),
            } for s in partitions}
            run_dir = output / f"{cue_set}_seed{seed}"
            run_dir.mkdir()
            checkpoint = {
                "format_version": 1, "architecture": "camera_multicue_gru", "context": CONTEXT,
                "feature_dim": FEATURE_DIM, "hidden_size": 16, "dropout": 0.3,
                "cue_set": cue_set, "multicue_version": MULTICUE_VERSION,
                "input_contract": manifest["input_contract"], "decision_threshold": threshold,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "task": "camera_current_state_weak_video_labels",
            }
            torch.save(checkpoint, run_dir / "model.pt")
            predictor = CameraPredictor(run_dir / "model.pt")
            exported = _scores(predictor.model, clips, accepted, batch_size)
            parity_error = float(np.max(np.abs(exported - all_scores)))
            runtime_error = 0.0
            for i in np.linspace(0, len(accepted) - 1, min(12, len(accepted)), dtype=int):
                w = accepted[i]
                c, start, end = clips[w.clip], w.end + 1 - CONTEXT, w.end + 1
                score = predictor.predict_window(c.x[start:end], c.ts_ms[start:end], c.valid[start:end]).score
                runtime_error = max(runtime_error, abs(score - all_scores[i]))
            if max(parity_error, runtime_error) > 1e-6:
                raise ValueError("Export/runtime predictions differ from training inference")
            _write_csv(run_dir / "predictions.csv", rows)
            _write_csv(run_dir / "clip_predictions.csv", clip_rows)
            run = {
                "cue_set": cue_set, "seed": seed, "best_epoch": best_epoch,
                "validation_log_loss": best_loss, "decision_threshold": threshold,
                "parameter_count": sum(p.numel() for p in model.parameters()), "metrics": metrics,
                "history": history, "wall_seconds": time.perf_counter() - run_started,
                "export_max_absolute_score_error": parity_error, "runtime_sample_max_error": runtime_error,
                "checkpoint_sha256": sha256_file(run_dir / "model.pt"),
                "checkpoint": str(run_dir / "model.pt"),
            }
            if len({clips[w.clip].subject for w in validation}) >= 2:
                run["validation_subject_bootstrap_ci"] = bootstrap_group_ci(
                    labels, scores, [clips[w.clip].subject for w in validation], threshold,
                    label_threshold=0.5, n_bootstrap=300, seed=42,
                )
                run["validation_subject_bootstrap_ci"]["scope"] = "conditional selection-validation CI, not final test"
            (run_dir / "metrics.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            runs.append(run)
    aggregate = {}
    for cue_set in cue_sets:
        selected = [r for r in runs if r["cue_set"] == cue_set]
        aggregate[cue_set] = {}
        for name in ("accuracy", "macro_f1", "balanced_accuracy", "recall", "specificity", "auroc"):
            values = [r["metrics"]["validation"]["window"]["overall_accepted_only"][name] for r in selected]
            aggregate[cue_set][name] = {"mean": float(np.mean(values)), "std_across_seeds": float(np.std(values)),
                                        "min": float(min(values)), "max": float(max(values))}
    report = {**protocol, "runs": runs, "validation_seed_summary": aggregate,
              "wall_seconds": time.perf_counter() - started}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cue-sets", nargs="+", choices=tuple(CUE_SETS), default=list(CUE_SETS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run_experiment(args.manifest, args.out, seeds=tuple(args.seeds), cue_sets=tuple(args.cue_sets),
                            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                            device=args.device)
    print(json.dumps(result["validation_seed_summary"], indent=2))


if __name__ == "__main__":
    main()
