"""Train GRU/DANN with training-subject domains and validation-only selection.

Example: python -m driverguard.learning.camera.dann.train --manifest features/manifest.json
    --out-dir runs/gru_dann/seed42 --device cuda --require-gpus 2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from driverguard.execution import source_provenance
from driverguard.learning.camera.features import sha256_file
from driverguard.learning.camera.model import MAX_GAP_MS, MIN_VALID_FRACTION

from .data import LABEL_MODES, Corpus, WindowDataset, load_corpus, normalization_from_train, partition_windows
from .metrics import clip_rows, prediction_rows, row_metrics, subject_bootstrap, write_csv
from .model import CONTEXTS, POOLINGS, CameraDANN, reversal_schedule


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "dann"
    label_mode: str = "binary_pilot"
    epochs: int = 60
    patience: int = 12
    batch_size: int = 256
    stride: int = 10
    context_seconds: int = 5
    pooling: str = "frame"
    hidden_size: int = 96
    layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    domain_weight: float = 0.2
    max_reversal: float = 1.0
    input_noise: float = 0.0
    # The adversary must learn subject identity faster than the encoder hides it; without
    # its own rate a small, decayed domain head stays at chance and reversal does nothing.
    domain_lr_multiplier: float = 10.0
    seed: int = 42
    device: str = "auto"
    require_gpus: int = 0
    num_workers: int = 0

    def validate(self):
        if self.mode not in ("gru", "dann") or self.label_mode not in LABEL_MODES:
            raise ValueError("Unknown architecture mode or label mode")
        if min(self.epochs, self.patience, self.batch_size, self.stride, self.hidden_size, self.layers) < 1:
            raise ValueError("Epochs, patience, batch, stride and model dimensions must be positive")
        if self.pooling not in POOLINGS:
            raise ValueError("Pooling must be frame, second or window")
        if self.context_seconds not in CONTEXTS[self.pooling]:
            raise ValueError(f"{self.pooling} pooling supports context seconds {CONTEXTS[self.pooling]}; "
                             "no padded history")
        numeric = (self.dropout, self.learning_rate, self.weight_decay, self.domain_weight, self.max_reversal,
                   self.input_noise, self.domain_lr_multiplier)
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Training configuration must be finite")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.domain_lr_multiplier <= 0:
            raise ValueError("Invalid dropout or learning rate")
        if min(self.weight_decay, self.domain_weight, self.max_reversal, self.input_noise,
               self.num_workers, self.require_gpus) < 0:
            raise ValueError("Regularization, workers and GPU requirements cannot be negative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def resolve_device(config: TrainConfig):
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_cuda = config.device == "cuda" or (config.device == "auto" and available > 0)
    if use_cuda and not available:
        raise RuntimeError("CUDA is required but unavailable; enable Kaggle GPU T4 x2 before training")
    count = available if use_cuda else 0
    if count < config.require_gpus:
        raise RuntimeError(f"Requested {config.require_gpus} CUDA GPUs; found {count}. Enable GPU T4 x2.")
    if count > 1 and config.batch_size < count:
        raise ValueError("Batch size must be at least the number of GPUs")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device, {"device": str(device), "gpu_count": count,
                    "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)],
                    "data_parallel": count > 1, "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda}


def seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state(generator: torch.Generator):
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"kind": np_state[0], "keys": np_state[1].tolist(),
            "position": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values["kind"], np.asarray(values["keys"], dtype=np.uint32), values["position"],
                         values["has_gauss"], values["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same GPU count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def augment(x: torch.Tensor, scale: torch.Tensor, strength: float) -> torch.Tensor:
    """Training-only jitter of observed physical fields, in units of the train-set standard deviation."""
    observed = x[:, :, 9:18] * x[:, :, -1:]
    x = x.clone()
    x[:, :, :9] += torch.randn_like(observed) * strength * scale * observed
    return x


def probabilities(model, dataset, config, device):
    model.eval()
    scores = []
    # no_grad, not inference_mode: DataParallel replicas of the GRU call
    # flatten_parameters in place, which inference tensors forbid.
    with torch.no_grad():
        for x, _, _, _ in DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0):
            logits, _ = model(x.to(device, non_blocking=True), 0.0)
            scores.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty((0, len(LABEL_MODES[config.label_mode])))


def evaluate(model, corpus: Corpus, windows, split, mapping, config, device):
    dataset = WindowDataset(corpus, windows, mapping, config.context_seconds * 10)
    scores = probabilities(model, dataset, config, device)
    rows = prediction_rows(corpus, windows, scores, len(LABEL_MODES[config.label_mode]))
    clips = clip_rows(corpus, rows, split, len(LABEL_MODES[config.label_mode]))
    return {"window": row_metrics(rows, len(LABEL_MODES[config.label_mode])),
            "clip": row_metrics(clips, len(LABEL_MODES[config.label_mode]))}, rows, clips


def train_experiment(manifest_path: str | Path, out_dir: str | Path, config: TrainConfig,
                     resume: str | Path | None = None, evaluate_test: bool = False,
                     stop_after_epoch: int | None = None) -> dict:
    """Train to budget/early stop. stop_after_epoch permits deterministic interruption tests."""
    config.validate()
    device, hardware = resolve_device(config)
    output = Path(out_dir)
    if resume is None and any((output / folder).exists() for folder in ("checkpoints", "metrics", "inference")):
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    seed_everything(config.seed)
    corpus = load_corpus(manifest_path, config.label_mode)
    context = config.context_seconds * 10
    windows = partition_windows(corpus, config.stride, context)
    mapping = {subject: index for index, subject in enumerate(sorted(
        {clip.subject for clip in corpus.clips if clip.split == "train"}))}
    if config.mode == "dann" and len(mapping) < 2:
        raise ValueError("DANN requires at least two training subjects/domains")
    datasets = {split: WindowDataset(corpus, windows[split], mapping, context) for split in ("train", "validation")}
    for split, dataset in datasets.items():
        labels = {corpus.clips[w.clip].label for w in dataset.windows}
        if labels != set(range(len(LABEL_MODES[config.label_mode]))):
            raise ValueError(f"{split} needs accepted unpadded windows from every requested class")
    center, scale = normalization_from_train(corpus, windows["train"], context)
    model_config = {"n_classes": len(LABEL_MODES[config.label_mode]), "n_domains": len(mapping),
                    "hidden_size": config.hidden_size, "layers": config.layers, "dropout": config.dropout,
                    "pooling": config.pooling}
    base = CameraDANN(**model_config).to(device)
    base.center.copy_(torch.from_numpy(center).to(device))
    base.scale.copy_(torch.from_numpy(scale).to(device))
    model = nn.DataParallel(base) if hardware["data_parallel"] else base
    domain_parameters = list(base.domain_head.parameters())
    domain_ids = {id(parameter) for parameter in domain_parameters}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in base.parameters() if id(p) not in domain_ids]},
        {"params": domain_parameters, "lr": config.learning_rate * config.domain_lr_multiplier,
         "weight_decay": 0.0},
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                         patience=max(1, config.patience // 3))
    generator = torch.Generator().manual_seed(config.seed)
    # A one-sample final batch cannot be batch-normalized in training mode.
    loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=config.num_workers, pin_memory=device.type == "cuda",
                        drop_last=len(datasets["train"]) % config.batch_size == 1)
    history, best_score, best_epoch, best_state, stale, start_epoch = [], -1.0, 0, None, 0, 1
    architecture = "camera_gru_dann" if config.mode == "dann" else "camera_gru_ablation"
    protocol = {
        "schema_version": 1, "architecture": architecture, "config": asdict(config),
        "manifest_sha256": corpus.manifest_sha256, "feature_hashes_train_validation": corpus.feature_hashes,
        "input_contract": corpus.manifest["input_contract"], "label_mode": config.label_mode,
        "class_names": list(LABEL_MODES[config.label_mode]), "label_source": "weak_video_level",
        "subject_splits": {split: sorted({r["subject"] for r in corpus.manifest["clips"] if r["split"] == split})
                           for split in ("train", "validation", "test")},
        "domain_mapping": mapping, "domain_kind": "training_subject_only",
        "adaptation": "none; validation/test features never enter optimizer or normalization",
        "normalization": "unique frames in accepted training windows; observed physical fields only",
        "selection_metric": "validation.window.subject_macro.macro_f1", "decision_rule": "argmax; no test threshold tuning",
        "loss_weighting": "equal_subject_class_source_clip_exposure_variant_window",
        "augmentation": f"gaussian jitter {config.input_noise:g} train SD on observed fields; train only"
        if config.input_noise else "none",
        "windowing": {"context_frames": context, "pooling": config.pooling, "stride_frames": config.stride, "fps": 10,
                      "min_valid_fraction": MIN_VALID_FRACTION, "max_gap_ms": MAX_GAP_MS,
                      "current_frame_must_be_valid": True, "padding": False,
                      "coverage_denominator": "scheduled windows after full history; warmup excluded"},
        "determinism": "fixed seeds; deterministic PyTorch kernels; resume requires identical hardware/config",
        "hardware": hardware,
        "source_provenance": source_provenance(Path(__file__).resolve().parents[4]),
        "source_hashes": {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "environment": {"python": platform.python_version(), "numpy": str(np.__version__),
                        "pytorch": str(torch.__version__)},
    }
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("checkpoint_version") != 1 or saved.get("config") != asdict(config):
            raise ValueError("Resume requires the identical training configuration and checkpoint version")
        if saved.get("manifest_sha256") != corpus.manifest_sha256 or saved.get("feature_hashes") != corpus.feature_hashes:
            raise ValueError("Resume dataset fingerprint differs from the original run")
        if saved.get("hardware") != hardware:
            raise ValueError("Exact resume requires identical hardware and PyTorch versions")
        if saved.get("source_hashes") != protocol["source_hashes"] or saved.get("environment") != protocol["environment"]:
            raise ValueError("Exact resume requires identical training source and Python/NumPy versions")
        base.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        history, best_score, best_epoch = saved["history"], saved["best_score"], saved["best_epoch"]
        best_state, stale, start_epoch = saved["best_state"], saved["stale"], saved["epoch"] + 1
        restore_rng(saved["rng_state"], generator)
    for folder in ("checkpoints", "metrics", "predictions", "inference"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    def save_best():
        # Selected weights and resumable optimizer state have distinct contracts.
        save_checkpoint(output / "checkpoints" / "best.pt", {
            "selection_checkpoint_version": 1, "architecture": architecture,
            "model_config": model_config, "model_state": best_state,
            "epoch": best_epoch, "validation_subject_macro_f1": best_score,
            "resume_supported": False, "resume_from": "last.pt",
        })
    if resume is not None:
        # A resumed Kaggle session may write to a fresh /kaggle/working directory.
        # Preserve all checkpoint artifacts even when the run has already finished.
        save_checkpoint(output / "checkpoints" / "last.pt", saved)
        save_best()
        write_json(output / "metrics" / "history.json", history)
    write_json(output / "metrics" / "protocol.json", protocol)
    print(json.dumps({"event": "training_start", "hardware": hardware, "mode": config.mode,
                      "label_mode": config.label_mode, "train_windows": len(datasets["train"]),
                      "validation_windows": len(datasets["validation"]), "domain_count": len(mapping)}), flush=True)
    for epoch in range(start_epoch, config.epochs + 1):
        if stale >= config.patience:
            break
        start = time.monotonic()
        model.train()
        totals = np.zeros(3, dtype=float)
        n_seen = 0
        for batch, (x, label, domain, weight) in enumerate(loader):
            progress = ((epoch - 1) * len(loader) + batch) / max(config.epochs * len(loader) - 1, 1)
            reversal = reversal_schedule(progress, config.max_reversal) if config.mode == "dann" else 0.0
            x, label, domain, weight = (value.to(device, non_blocking=True) for value in (x, label, domain, weight))
            if config.input_noise:
                x = augment(x, base.scale, config.input_noise)
            optimizer.zero_grad(set_to_none=True)
            class_logits, domain_logits = model(x, reversal)
            class_loss = (F.cross_entropy(class_logits, label, reduction="none") * weight).mean()
            domain_loss = (F.cross_entropy(domain_logits, domain, reduction="none") * weight).mean() \
                if config.mode == "dann" else class_loss.new_zeros(())
            loss = class_loss + config.domain_weight * domain_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; checkpoint was not advanced")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            totals += np.array([loss.item(), class_loss.item(), domain_loss.item()]) * len(x)
            n_seen += len(x)
        validation, _, _ = evaluate(model, corpus, windows["validation"], "validation", mapping, config, device)
        score = validation["window"]["subject_macro"]["macro_f1"]
        entry = {"epoch": epoch, "train_loss": float(totals[0] / n_seen),
                 "train_class_loss": float(totals[1] / n_seen), "train_domain_loss": float(totals[2] / n_seen),
                 "validation_accuracy": validation["window"]["accuracy"],
                 "validation_macro_f1": validation["window"]["macro_f1"],
                 "validation_subject_macro_f1": score, "reversal_strength": reversal,
                 "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - start}
        history.append(entry)
        improved = score > best_score + 1e-12
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
        else:
            stale += 1
        scheduler.step(score)
        checkpoint = {
            "checkpoint_version": 1, "architecture": architecture, "config": asdict(config),
            "model_config": model_config, "hardware": hardware, "manifest_sha256": corpus.manifest_sha256,
            "feature_hashes": corpus.feature_hashes, "epoch": epoch,
            "source_hashes": protocol["source_hashes"], "environment": protocol["environment"],
            "model_state": base.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "rng_state": rng_state(generator),
            "history": history, "best_score": best_score, "best_epoch": best_epoch,
            "best_state": best_state, "stale": stale, "domain_mapping": mapping,
        }
        save_checkpoint(output / "checkpoints" / "last.pt", checkpoint)
        if improved:
            save_best()
        write_json(output / "metrics" / "history.json", history)
        print(json.dumps(entry), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    base.load_state_dict(best_state)
    validation, rows, clips = evaluate(model, corpus, windows["validation"], "validation", mapping, config, device)
    validation["window"]["confidence_interval"] = subject_bootstrap(rows, model_config["n_classes"], config.seed)
    validation["clip"]["confidence_interval"] = subject_bootstrap(clips, model_config["n_classes"], config.seed)
    write_csv(output / "predictions" / "validation_windows.csv", rows)
    write_csv(output / "predictions" / "validation_clips.csv", clips)
    test = None
    interrupted = history[-1]["epoch"] < config.epochs and stale < config.patience
    if evaluate_test:
        if interrupted:
            raise ValueError("Cannot evaluate test before training budget or early stopping is complete")
        # The held-out NPZs are first opened here, after checkpoint selection is frozen.
        test_corpus = load_corpus(manifest_path, config.label_mode, splits=("test",))
        test_windows = partition_windows(test_corpus, config.stride, context)["test"]
        test, rows, clips = evaluate(model, test_corpus, test_windows, "test", mapping, config, device)
        test["window"]["confidence_interval"] = subject_bootstrap(rows, model_config["n_classes"], config.seed)
        test["clip"]["confidence_interval"] = subject_bootstrap(clips, model_config["n_classes"], config.seed)
        write_csv(output / "predictions" / "test_windows.csv", rows)
        write_csv(output / "predictions" / "test_clips.csv", clips)
        write_json(output / "metrics" / "test_feature_hashes.json", test_corpus.feature_hashes)
    export = {"format_version": 1, "architecture": architecture, "model_config": model_config,
              "state_dict": best_state, "input_contract": corpus.manifest["input_contract"],
              "label_mode": config.label_mode, "class_names": list(LABEL_MODES[config.label_mode]),
              "context_frames": context, "feature_dim": 19, "decision_rule": "argmax",
              "quality": protocol["windowing"], "best_epoch": best_epoch,
              "manifest_sha256": corpus.manifest_sha256}
    save_checkpoint(output / "inference" / "model.pt", export)
    def target_met(metrics):
        return any(value is not None and value > 0.85 for value in
                   (metrics["window"]["accuracy"], metrics["window"]["macro_f1"]))
    summary = {
        "schema_version": 1, "status": "interrupted" if interrupted else "completed",
        "architecture": architecture, "label_mode": config.label_mode,
        "class_names": list(LABEL_MODES[config.label_mode]), "best_epoch": best_epoch,
        "selection_metric": protocol["selection_metric"], "best_validation_macro_f1": best_score,
        "metrics": {"validation": validation, "test": test}, "hardware": hardware,
        "target": {"threshold": 0.85, "operator": "strictly_greater", "metric_unit": "causal_window",
                   "validation_met": target_met(validation), "test_met": target_met(test) if test else None,
                   "independently_confirmed": False,
                   "confirmation_note": "Requires a previously unseen independent cohort and coverage/runtime checks"},
        "checkpoint_sha256": sha256_file(output / "inference" / "model.pt"),
        "manifest_sha256": corpus.manifest_sha256,
        "limitations": ["Weak video-level labels; no onset or EEG supervision",
                        "Subject-domain adversarial generalization; no held-out domain adaptation",
                        "Current test cohort has prior development exposure; not a fresh confirmation cohort",
                        {"binary_pilot": "Binary cropped pilot; does not represent full three-class UTA",
                         "alert_drowsy": "Full UTA alert vs drowsy; low-vigilance recordings are excluded",
                         }.get(config.label_mode,
                               "Three-class weak labels; provenance and cohort must be checked independently")],
    }
    write_json(output / "metrics" / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-test", action="store_true", help="Open held-out features only after selection")
    for key, value in asdict(TrainConfig()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    arguments = vars(parser.parse_args())
    manifest, output, resume, test = (arguments.pop(key) for key in ("manifest", "out_dir", "resume", "evaluate_test"))
    result = train_experiment(manifest, output, TrainConfig(**arguments), resume=resume, evaluate_test=test)
    print(json.dumps({"status": result["status"], "target": result["target"],
                      "best_validation_macro_f1": result["best_validation_macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
