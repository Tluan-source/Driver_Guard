"""Compare every trained model; re-score all camera checkpoints on one common benchmark.

Camera benchmark: UTA full v002 *validation* (10 drivers), alert (0) vs drowsy (10) recordings,
windows every 2.5 s, each checkpoint's own context and decision rule. Validation was used during
development of the 120 s models, so numbers are optimistic; test features are never opened.
EEG models are listed from their stored metrics (different datasets/protocols, not re-scored).

    python -m scripts.evaluation.compare_models --out outputs/audits/<date>/model_comparison.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from driverguard.config import REPO_ROOT
from driverguard.debug import model_lab
from driverguard.execution import file_sha256

FEATURES = "uta_rldd_full_v002"
N_BOOT = 2000
SEED = 20261010


def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    ranks[order] = np.arange(1, len(s) + 1)
    for value in np.unique(s):  # average ties
        tie = s == value
        ranks[tie] = ranks[tie].mean()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def macro_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = np.sum((p == c) & (y == c))
        fp, fn = np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def score_model(model_id: str, clips: list[dict]) -> dict:
    """Window rows (subject, label, pred, p_drowsy) + smoothed/clip predictions for one model."""
    model = model_lab.load_model(model_id)
    rows, smooth_rows, clip_rows, total = [], [], [], 0
    contract_mismatches = []
    for c in clips:
        if c["split"] != "validation":
            raise ValueError("Comparison only accepts validation clips")
        if c["label_name"] not in model.class_names:
            continue
        r = model_lab.replay(model_id, FEATURES, c["clip_id"], precision=None)
        task = r["clip"]["task_label"]
        if task is None:
            continue
        total += r["summary"]["n_windows"]
        if r["contract"]["ok"] is not True:
            contract_mismatches.append({"clip_id": c["clip_id"], **r["contract"]})
        last = len(r["class_names"]) - 1
        for w, s in zip(r["windows"], r["smoothed"], strict=True):
            if w["p"] is not None:
                rows.append((c["subject"], task, w["pred"], w["p"][last], tuple(w["p"]), w["t"], c["clip_id"]))
            if s is not None:
                thr = r["threshold"]
                pred = int(s[1] >= thr) if thr is not None else int(np.argmax(s))
                smooth_rows.append((c["subject"], task, pred))
        if r["summary"]["clip_prediction"] is not None:
            clip_rows.append((c["subject"], task, r["summary"]["clip_prediction"]))
    return {"rows": rows, "smooth": smooth_rows, "clips": clip_rows, "n_windows": total,
            "n_classes": len(model.class_names), "threshold": model.threshold,
            "class_names": list(model.class_names), "contract_mismatches": contract_mismatches}


def subject_bootstrap(rows: list[tuple], stat, rng: np.random.Generator) -> tuple[float, float]:
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row[0]].append(row)
    subjects = sorted(by_subject)
    if not subjects:
        return None
    if stat is acc_of:
        counts = np.array([len(by_subject[s]) for s in subjects])
        correct = np.array([sum(r[1] == r[2] for r in by_subject[s]) for s in subjects])
        picks = rng.integers(0, len(subjects), (N_BOOT, len(subjects)))
        values = correct[picks].sum(axis=1) / counts[picks].sum(axis=1)
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]
    values = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(subjects), len(subjects), replace=True)
        values.append(stat([row for i in pick for row in by_subject[subjects[i]]]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def acc_of(rows: list[tuple]) -> float:
    return float(np.mean([r[1] == r[2] for r in rows])) if rows else None


def summarize(scored: dict, rng: np.random.Generator) -> dict:
    rows = scored["rows"]
    y = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    out = {
        "window_accuracy": acc_of(rows),
        "window_accuracy_ci95": subject_bootstrap(rows, acc_of, rng),
        "window_macro_f1": macro_f1(y, p, scored["n_classes"]) if rows else None,
        "smoothed_5min_accuracy": acc_of(scored["smooth"]),
        "recording_accuracy": acc_of(scored["clips"]),
        "n_recordings": len(scored["clips"]),
        "n_accepted_windows": len(rows),
        "coverage": len(rows) / scored["n_windows"] if scored["n_windows"] else 0.0,
        "n_eligible_windows": scored["n_windows"],
        "class_names": scored["class_names"],
        "contract_mismatches": scored["contract_mismatches"],
    }
    if scored["n_classes"] == 2:
        out["auroc"] = auroc(y, np.array([r[3] for r in rows]))
    return out


def paired_delta(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """Accuracy(b) - accuracy(a) on common accepted endpoints, paired by driver."""
    if a["class_names"] != b["class_names"]:
        raise ValueError("Cannot compare different classification tasks")
    ka, kb = ({(r[6], r[5]): r for r in item["rows"]} for item in (a, b))
    common = sorted(set(ka) & set(kb))
    if not common:
        return {"delta": None, "ci95": None, "n_common_windows": 0, "verdict": "no common windows"}
    sa, sb = defaultdict(list), defaultdict(list)
    for key in common:
        ra, rb = ka[key], kb[key]
        if tuple(ra[0:2]) != tuple(rb[0:2]):
            raise ValueError("Paired endpoint subject/label mismatch")
        sa[ra[0]].append(ra[1] == ra[2])
        sb[rb[0]].append(rb[1] == rb[2])
    subjects = sorted(set(sa) & set(sb))
    counts = np.array([len(sa[s]) for s in subjects])
    ca, cb = (np.array([sum(item[s]) for s in subjects]) for item in (sa, sb))
    pick = rng.integers(0, len(subjects), (N_BOOT, len(subjects)))
    deltas = (cb[pick].sum(axis=1) - ca[pick].sum(axis=1)) / counts[pick].sum(axis=1)
    before, after = float(ca.sum()/counts.sum()), float(cb.sum()/counts.sum())
    point = after - before
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    verdict = "tốt lên rõ" if lo > 0 else "kém đi rõ" if hi < 0 else "chưa chắc (CI chứa 0)"
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "verdict": verdict,
            "n_common_windows": len(common), "n_drivers": len(subjects),
            "paired_before_acc": before, "paired_after_acc": after,
            "comparison_scope": "same clip/end timestamp accepted by both; conditional on shared coverage"}


def ensemble(scored: list[dict]) -> dict:
    """Average probabilities of seeds over windows present in every seed (same clip and end time)."""
    keyed = [{(r[6], r[5]): r for r in s["rows"]} for s in scored]
    common = set.intersection(*(set(k) for k in keyed))
    rows = []
    for key in sorted(common):
        probs = np.mean([np.asarray(k[key][4]) for k in keyed], axis=0)
        base = keyed[0][key]
        rows.append((base[0], base[1], int(np.argmax(probs)), float(probs[-1]), tuple(probs), base[5], base[6]))
    return {**scored[0], "rows": rows, "smooth": [], "clips": []}


def eeg_models() -> list[dict]:
    out = []
    for version in ("v1", "v2"):
        m = json.loads((REPO_ROOT / f"models/eeg_vigilance_{version}/metrics.json").read_text(encoding="utf-8"))
        sel = m["selected"]
        val, test = m["validation"][sel], (m.get("test") or {}).get(sel, {})
        out.append({"model": f"EEG {version}", "selected": sel, "protocol": "split 8/2/2 người, cửa sổ",
                    "validation": {k: val.get(k) for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc")},
                    "test": {k: test.get(k) for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc")}})
    for version in ("v3", "v3_1"):
        m = json.loads((REPO_ROOT / f"models/eeg_vigilance_{version}/summary.json").read_text(encoding="utf-8"))
        for variant in ("legacy", "robust"):
            o, sm = m[variant]["overall"], m[variant]["subject_macro"]
            out.append({"model": f"EEG {version.replace('_', '.')} {variant}", "selected": m["selected_config"],
                        "protocol": "nested LOSO 12 người",
                        "nested_loso": {"accuracy": o["accuracy"], "balanced_accuracy": o["balanced_accuracy"],
                                        "macro_f1": o["macro_f1"], "auroc": o["auroc"],
                                        "subject_macro_auroc": sm["auroc"], "n": o["n"]}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    started = time.time()
    model_lab.torch.set_num_threads(args.threads)
    rng = np.random.default_rng(SEED)
    fset = next(s for s in model_lab.list_clips()["feature_sets"] if s["id"] == FEATURES)
    clips = [c for c in fset["clips"] if c["split"] == "validation"]
    models = [m for m in model_lab.list_models() if not m["smoke"]]
    cache_dir = args.out.parent / "comparison_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_lab.FEATURE_ROOT / FEATURES / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["clips"]:
        if record["split"] == "validation":
            path = model_lab._clip_path(manifest_path, record)
            if file_sha256(path) != record["features_sha256"]:
                raise ValueError(f"Validation feature hash mismatch: {record['clip_id']}")
    fingerprint = [file_sha256(Path(__file__)), file_sha256(Path(model_lab.__file__)),
                   file_sha256(manifest_path)]
    scored, results = {}, []
    for index, m in enumerate(models):
        cache_key = hashlib.sha256(json.dumps([*fingerprint, m["id"],
            file_sha256(REPO_ROOT / m["id"])]).encode()).hexdigest()
        cache = cache_dir / f"{cache_key}.json"
        if cache.is_file():
            scored[m["id"]] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            scored[m["id"]] = score_model(m["id"], clips)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(scored[m["id"]], allow_nan=False), encoding="utf-8")
            tmp.replace(cache)
        results.append({**{k: m[k] for k in ("id", "mode", "group", "name", "label_mode", "context_s")},
                        "stored_validation": m["validation"],
                        "common_benchmark": summarize(scored[m["id"]], np.random.default_rng(SEED + index))})
        print(f"{m['id']}: {results[-1]['common_benchmark']['window_accuracy']:.4f}", flush=True)
    groups = defaultdict(list)
    for m in models:
        groups[(m["group"], m["mode"], m["label_mode"])].append(m["id"])
    ensembles = []
    for (group, mode, label_mode), ids in sorted(groups.items()):
        if len(ids) == 3 and label_mode in ("alert_drowsy", "binary_pilot") and scored[ids[0]]["threshold"] is None:
            ens = ensemble([scored[i] for i in ids])
            scored[f"ensemble:{group}:{mode}"] = ens
            ensembles.append({"group": group, "mode": mode, "members": ids,
                              "window_accuracy": acc_of(ens["rows"]),
                              "window_accuracy_ci95": subject_bootstrap(ens["rows"], acc_of, rng)})

    def ref(suffix: str) -> str:
        return next(m["id"] for m in models if m["id"].endswith(suffix))

    steps = [
        ("CAM-01 thống kê temporal + logistic", "camera_drowsiness_v1/model.pt", "camera_cam01_temporal_logistic/model.pt"),
        ("CAM-02 đa nhánh mắt+miệng+đầu", "camera_drowsiness_v1/model.pt", "camera_cam02_multicue/full_seed42/model.pt"),
        ("GRU pilot 5 s (Kaggle)", "camera_cam02_multicue/full_seed42/model.pt", "gru_dann_20261006/gru_5s_seed42/inference/model.pt"),
        ("Thêm DANN, pilot 5 s", "gru_dann_20261006/gru_5s_seed42/inference/model.pt", "gru_dann_20261006/dann_5s_seed42/inference/model.pt"),
        ("UTA full, context 120 s, pooling theo giây (v1)", "gru_dann_20261006/gru_5s_seed42/inference/model.pt", "uta_full_v002_20261009/gru_alert_drowsy_120s_seed42/inference/model.pt"),
        ("Sửa domain head DANN (v2)", "uta_full_v002_20261009/dann_alert_drowsy_120s_seed42/inference/model.pt", "uta_full_v002_20261009_v2/dann_alert_drowsy_120s_seed42/inference/model.pt"),
        ("Pooling window + đặc trưng tương đối (cấu hình cuối)", "uta_full_v002_20261009/gru_alert_drowsy_120s_seed42/inference/model.pt", "uta_full_v002_20261009_window/gru_alert_drowsy_120s_seed42/inference/model.pt"),
        ("Thêm DANN, cấu hình cuối", "uta_full_v002_20261009_window/gru_alert_drowsy_120s_seed42/inference/model.pt", "uta_full_v002_20261009_window/dann_alert_drowsy_120s_seed42/inference/model.pt"),
        ("Tổng: v1 01/10 → GRU+DANN window 09/10", "camera_drowsiness_v1/model.pt", "uta_full_v002_20261009_window/dann_alert_drowsy_120s_seed42/inference/model.pt"),
    ]
    step_results = []
    for title, a, b in steps:
        ia, ib = ref(a), ref(b)
        step_results.append({"step": title, "before": ia, "after": ib,
                             "before_acc": acc_of(scored[ia]["rows"]), "after_acc": acc_of(scored[ib]["rows"]),
                             **paired_delta(scored[ia], scored[ib], rng)})
    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "benchmark": {"features": FEATURES, "split": "validation", "drivers": len({c["subject"] for c in clips}),
                      "tasks": {"binary": "alert vs drowsy; low-vigilance excluded before counting coverage",
                                "three_class": "alert/low_vigilance/drowsy; reported separately"},
                      "paired_comparisons": "same accepted clip/end timestamps, paired driver bootstrap",
                      "checkpoint_selection": "all supported non-smoke checkpoints; no new training",
                      "stride_s": model_lab.DEFAULT_STRIDE / 10, "bootstrap": f"{N_BOOT} lượt theo người",
                      "caveat": "Validation đã dùng khi phát triển model 120 s nên lạc quan; test không được mở."},
        "camera_models": results, "ensembles": ensembles, "improvement_steps": step_results,
        "eeg_models": eeg_models(), "elapsed_s": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
