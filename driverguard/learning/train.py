"""Validation-selected EEG training; test subjects are evaluated after selection."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    FEATURE_KEY,
    dataset_manifest,
    dataset_metadata,
    load_sessions,
    pack_sessions,
    split_sessions,
)
from .features import transform_features
from .metrics import bootstrap_group_ci, grouped_metrics, select_threshold, vigilance_metrics
from .model import VigilancePredictor, VigilanceTCN


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_predictions(path: Path, y, pred, subjects, sessions, indices, threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "session_id", "window_index", "target", "prediction",
                         "reduced_vigilance_pred"])
        for group, session, index, target, output in zip(subjects, sessions, indices, y, pred):
            writer.writerow([group, session, int(index), float(target), float(output), int(output >= threshold)])


def train_neural(x_train, y_train, x_val, y_val, *, epochs, seed, width=48, batch_size=256,
                 target_type="regression"):
    torch.manual_seed(seed)
    model = VigilanceTCN(n_features=x_train.shape[-1], width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_rmse, best_epoch, stale, history = None, float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            prediction = model(x)
            loss = (torch.nn.functional.binary_cross_entropy(prediction, y) if target_type == "binary"
                    else torch.nn.functional.mse_loss(prediction, y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(y)
        model.eval()
        with torch.inference_mode():
            pred = np.concatenate([model(torch.from_numpy(x_val[i:i + 512])).numpy()
                                   for i in range(0, len(x_val), 512)])
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "validation_rmse": rmse})
        if rmse < best_rmse - 1e-5:
            best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), rmse, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d} validation RMSE={rmse:.4f}", flush=True)
        if stale >= 12:
            break
    return best, best_epoch, history


def train(root: str | Path, out: str | Path, *, epochs=60, context=8, seed=42,
          feature_key=FEATURE_KEY, bootstrap=300, threads=4, feature_transform=None):
    if epochs < 1 or not 1 <= context <= 15 or threads < 1 or bootstrap < 1:
        raise ValueError("Positive epochs/threads/bootstrap and context in [1,15] required (TCN receptive field)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    if (out / "model.pt").exists():
        raise ValueError(f"Training output already exists: {out}. Use a new directory to preserve the test run.")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sessions = load_sessions(root, feature_key)
    metadata = dataset_metadata(root)
    if metadata["target_type"] not in {"binary", "regression"}:
        raise ValueError("Dataset target_type must be binary or regression")
    feature_transform = feature_transform or ("relative_log_power" if metadata["target"] == "fatigue" else "absolute")
    limitations = metadata.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    label_threshold = float(metadata["label_threshold"])
    if not np.isfinite(label_threshold) or not 0 <= label_threshold <= 1:
        raise ValueError("Dataset label_threshold must be finite and in [0,1]")
    if metadata["target_type"] == "binary" and any(not np.isin(s.y, [0, 1]).all() for s in sessions):
        raise ValueError("Binary targets must be exactly 0 or 1")
    split = split_sessions(sessions, seed)
    for partition in ("train", "validation"):
        y_partition = np.concatenate([s.y for s in sessions if s.subject in split[partition]])
        if len(np.unique(y_partition >= label_threshold)) != 2:
            raise ValueError(f"{partition} must contain both vigilance reference classes")
    raw_train = np.concatenate([s.x for s in sessions if s.subject in split["train"]])
    transformed_train = transform_features(raw_train, feature_transform)
    mean, std = transformed_train.mean(axis=0), np.maximum(transformed_train.std(axis=0), 1e-5)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    common = {"format_version": 1, "feature_key": feature_key, "mean": mean.tolist(), "std": std.tolist(),
              "feature_transform": feature_transform,
              "split": split, "label_threshold": label_threshold, "seed": seed,
              "input_contract": metadata["feature_extraction"], "dataset_metadata": metadata,
              "target": metadata["target"], "target_type": metadata["target_type"],
              "dataset_manifest": dataset_manifest(sessions), "source_revision": revision,
              "source_file_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(out / "split.json", split)
    write_json(out / "dataset_manifest.json", dataset_manifest(sessions))
    print(f"Loaded {len(sessions)} sessions, {len({s.subject for s in sessions})} subjects; split={split}",
          flush=True)
    candidates = []
    train_one = pack_sessions(sessions, split["train"], 1)
    val_one = pack_sessions(sessions, split["validation"], 1)
    tr = (transform_features(train_one[0][:, -1], feature_transform) - mean) / std
    va = (transform_features(val_one[0][:, -1], feature_transform) - mean) / std
    constant = dict(common, architecture="constant", context=1, value=float(train_one[1].mean()))
    candidates.append(("mean_baseline", constant, []))
    # A fixed alpha grid is selected using validation only.
    ridge_options = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        ridge = Ridge(alpha=alpha).fit(tr, train_one[1])
        pred = np.clip(ridge.predict(va), 0, 1)
        ridge_options.append((float(np.mean((pred - val_one[1]) ** 2)), alpha, ridge))
    _, alpha, ridge = min(ridge_options, key=lambda item: item[0])
    candidates.append(("ridge", dict(common, architecture="ridge", context=1, alpha=alpha,
                                      coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_)), []))
    if metadata["target_type"] == "binary":
        labels = train_one[1] >= label_threshold
        for mode in ("absolute", "relative_log_power"):
            features = transform_features(raw_train, mode)
            local_mean, local_std = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-5)
            train_features = (transform_features(train_one[0][:, -1], mode) - local_mean) / local_std
            val_features = (transform_features(val_one[0][:, -1], mode) - local_mean) / local_std
            base = dict(common, feature_transform=mode, mean=local_mean.tolist(), std=local_std.tolist(), context=1)
            logistic_options, svm_options = [], []
            for c in (0.1, 1., 10.):
                model = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(train_features, labels)
                prediction = model.predict_proba(val_features)[:, 1]
                scores = vigilance_metrics(val_one[1], prediction, label_threshold=label_threshold)
                logistic_options.append((-scores["auroc"], scores["rmse"], c, model))
                for gamma in ("scale", 0.01):
                    svm = SVC(C=c, gamma=gamma, random_state=seed).fit(train_features, labels)
                    prediction = 1 / (1 + np.exp(-np.clip(svm.decision_function(val_features), -60, 60)))
                    scores = vigilance_metrics(val_one[1], prediction, label_threshold=label_threshold)
                    svm_options.append((-scores["auroc"], scores["rmse"], c, gamma, svm))
            _, _, c, model = min(logistic_options, key=lambda item: item[:2])
            candidates.append((f"logistic_{mode}", dict(base, architecture="logistic", c=c,
                                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0])), []))
            _, _, c, gamma, svm = min(svm_options, key=lambda item: item[:2])
            candidates.append((f"svm_{mode}", dict(base, architecture="rbf_svm", c=c, gamma_spec=gamma,
                                gamma=float(svm._gamma), support_vectors=svm.support_vectors_.tolist(),
                                dual_coef=svm.dual_coef_[0].tolist(), intercept=float(svm.intercept_[0])), []))
    for name, length in (("eeg_single_window", 1), ("eeg_temporal", context)):
        print(f"Training {name}, context={length}", flush=True)
        tr_pack = pack_sessions(sessions, split["train"], length)
        va_pack = pack_sessions(sessions, split["validation"], length)
        state, best_epoch, history = train_neural((transform_features(tr_pack[0], feature_transform) - mean) / std, tr_pack[1],
                                                (transform_features(va_pack[0], feature_transform) - mean) / std, va_pack[1],
                                                epochs=epochs, seed=seed, target_type=metadata["target_type"])
        candidates.append((name, dict(common, architecture="tcn", context=length, state_dict=state,
                                       model_config={"n_features": len(mean), "width": 48, "dropout": 0.2},
                                       best_epoch=best_epoch), history))
    validation_results = {}
    for name, checkpoint, history in candidates:
        path = out / f"{name}.pt"
        checkpoint["decision_threshold"] = label_threshold
        torch.save(checkpoint, path)
        predictor = VigilancePredictor(path)
        x, y, groups, recordings, indices = pack_sessions(sessions, split["validation"], checkpoint["context"])
        pred = predictor.predict_windows(x)
        checkpoint["decision_threshold"] = select_threshold(y, pred, label_threshold=label_threshold)
        torch.save(checkpoint, path)
        validation_results[name] = vigilance_metrics(y, pred, decision_threshold=checkpoint["decision_threshold"],
                                                     label_threshold=label_threshold)
        write_predictions(out / f"{name}_validation.csv", y, pred, groups, recordings, indices,
                          checkpoint["decision_threshold"])
        if history:
            write_json(out / f"{name}_history.json", history)
    criterion = "maximum validation AUROC, then minimum RMSE" if metadata["target_type"] == "binary" else "minimum validation RMSE"
    selected = (min(validation_results, key=lambda name: (-validation_results[name]["auroc"],
                                                         validation_results[name]["rmse"]))
                if metadata["target_type"] == "binary"
                else min(validation_results, key=lambda name: validation_results[name]["rmse"]))
    write_json(out / "selection.json", {"criterion": criterion, "selected": selected,
                                       "validation": validation_results})
    chosen = next(checkpoint for name, checkpoint, _ in candidates if name == selected)
    chosen["selected_candidate"] = selected
    torch.save(chosen, out / "model.pt")
    print(f"Frozen selection: {selected}; evaluating test subjects", flush=True)
    test_results = {}
    for name, checkpoint, _ in candidates:
        predictor = VigilancePredictor(out / f"{name}.pt")
        x, y, groups, recordings, indices = pack_sessions(sessions, split["test"], checkpoint["context"])
        pred = predictor.predict_windows(x)
        test_results[name] = grouped_metrics(y, pred, groups, decision_threshold=checkpoint["decision_threshold"],
                                            label_threshold=label_threshold)
        write_predictions(out / f"{name}_test.csv", y, pred, groups, recordings, indices,
                          checkpoint["decision_threshold"])
        if name == selected:
            write_predictions(out / "predictions_test.csv", y, pred, groups, recordings, indices,
                              checkpoint["decision_threshold"])
            ci = bootstrap_group_ci(y, pred, groups, n_bootstrap=bootstrap, seed=seed,
                                    decision_threshold=checkpoint["decision_threshold"], label_threshold=label_threshold)
    result = {"selected": selected, "validation": validation_results, "test": test_results,
              "selected_test_ci": ci, "split": split, "n_sessions": len(sessions),
              "n_windows": sum(len(s.y) for s in sessions), "wall_seconds": time.perf_counter() - start,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "sklearn": sklearn.__version__},
              "dataset_metadata": metadata,
              "limitations": limitations + [
                              "No synchronized RGB input; EEG-camera fusion is not evaluated",
                              "One subject split, not a published cross-validation score",
                              "Score is not a clinically validated diagnosis or calibrated probability",
                              "No raw EEG hardware streaming adapter yet"]}
    write_json(out / "metrics.json", result)
    print(json.dumps({"checkpoint": str(out / "model.pt"), "selected": selected,
                      "test": test_results[selected]["overall"]}, indent=2), flush=True)
    return result
