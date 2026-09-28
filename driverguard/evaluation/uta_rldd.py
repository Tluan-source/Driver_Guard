"""UTA-RLDD helpers.

UTA-RLDD labels are VIDEO-LEVEL (0 = alert, 5 = low vigilant, 10 = drowsy; ~10 min each,
60 participants, official 5 folds x 12 participants). There are NO per-frame eye/blink/yawn
labels, so this dataset can only support:
  * video-level classification from aggregated temporal features (this module), and
  * checking that PERCLOS-proxy / blink statistics trend with the label.
It CANNOT measure eye-state F1, blink F1, onset delay or false alerts/hour (GVHD 6.1).

Usage (after features were extracted on Kaggle, see kaggle/README.md):
    python -m driverguard.evaluation.uta_rldd manifest --root D:/data/uta_rldd --out data/uta_manifest.csv
    python -m driverguard.evaluation.uta_rldd evaluate --manifest data/uta_manifest.csv --features data/features/uta
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mov", ".mp4", ".avi", ".m4v", ".mkv", ".MOV", ".MP4"}
LABELS = {0: "alert", 5: "low_vigilant", 10: "drowsy"}


def parse_path(p: Path) -> dict | None:
    m = re.search(r"(?:^|[^0-9])(0|5|10)$", p.stem)
    if not m:
        return None
    fold = None
    for part in p.parts:
        fm = re.search(r"fold[ _-]?(\d)", part, re.IGNORECASE)
        if fm:
            fold = int(fm.group(1))
    return {"path": str(p), "subject": p.parent.name, "fold": fold, "label": int(m.group(1)),
            "video_id": f"{p.parent.name}_{m.group(1)}"}


def build_manifest(root: str | Path) -> list[dict]:
    rows = [r for p in sorted(Path(root).rglob("*")) if p.suffix in VIDEO_EXT and (r := parse_path(p))]
    return rows


def video_features(ticks: list[dict]) -> dict[str, float]:
    """Aggregate a tick stream (engine replay of one video) into video-level features."""
    def arr(key):
        return np.array([t[key] for t in ticks if t.get(key) is not None], dtype=float)

    per, br, bd = arr("perclos_proxy"), arr("blink_rate_per_min"), arr("blink_duration_ms")
    valid = np.mean([bool(t["face_valid"]) for t in ticks]) if ticks else 0.0
    minutes = (ticks[-1]["timestamp_ms"] - ticks[0]["timestamp_ms"]) / 60000.0 if len(ticks) > 1 else 0.0
    closures = sum(1 for i in range(1, len(ticks)) if ticks[i]["closure_ms"] == 0 and ticks[i - 1]["closure_ms"] >= 1000)
    return {
        "perclos_mean": float(per.mean()) if per.size else np.nan,
        "perclos_p90": float(np.percentile(per, 90)) if per.size else np.nan,
        "blink_rate_mean": float(br.mean()) if br.size else np.nan,
        "blink_dur_mean": float(bd.mean()) if bd.size else np.nan,
        "long_closures_per_min": closures / minutes if minutes else np.nan,
        "valid_ratio": float(valid),
    }


def evaluate_leave_one_fold_out(table: list[dict], feature_keys: list[str]) -> dict:
    """Official protocol: each fold is the test set once, the other 4 train. Returns mean ± std."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    folds = sorted({r["fold"] for r in table if r["fold"] is not None})
    res = []
    if len(folds) < 2:
        raise ValueError(f"need features from >= 2 folds for leave-one-fold-out, got folds={folds}")
    for f in folds:
        tr = [r for r in table if r["fold"] != f and r["fold"] is not None]
        te = [r for r in table if r["fold"] == f]
        if len({r["label"] for r in tr}) < 2:
            continue  # cannot train a classifier on a single class
        Xtr = np.array([[r[k] for k in feature_keys] for r in tr], float)
        Xte = np.array([[r[k] for k in feature_keys] for r in te], float)
        ytr, yte = [r["label"] for r in tr], [r["label"] for r in te]
        clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        res.append({"fold": f, "acc": accuracy_score(yte, pred), "macro_f1": f1_score(yte, pred, average="macro")})
    accs, f1s = [r["acc"] for r in res], [r["macro_f1"] for r in res]
    return {"per_fold": res, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))}


def _main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--manifest", required=True, help="manifest.csv (from `manifest`) or manifest.json (from Kaggle)")
    e.add_argument("--features", required=True, help="folder with <video_id>.parquet from Kaggle")
    args = ap.parse_args()

    if args.cmd == "manifest":
        rows = build_manifest(args.root)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "subject", "fold", "label", "path"])
            w.writeheader()
            w.writerows(rows)
        subj = {r["subject"] for r in rows}
        print(f"{len(rows)} videos, {len(subj)} subjects, folds={sorted({r['fold'] for r in rows})} -> {args.out}")
        return

    from ..config import load_config
    from ..io import read_features
    from .operating_curve import replay

    cfg = load_config()
    table = []
    with open(args.manifest, encoding="utf-8") as f:
        rows = json.load(f) if args.manifest.endswith(".json") else list(csv.DictReader(f))
        for r in rows:
            fp = next((Path(args.features) / f"{r['video_id']}{ext}" for ext in (".parquet", ".csv.gz", ".csv")
                       if (Path(args.features) / f"{r['video_id']}{ext}").exists()), None)
            if fp is None:
                continue
            feats = video_features(replay(read_features(fp), cfg))
            table.append({**r, "fold": int(r["fold"]) if r["fold"] not in (None, "") else None,
                          "label": int(r["label"]), **feats})
    keys = ["perclos_mean", "perclos_p90", "blink_rate_mean", "blink_dur_mean", "long_closures_per_min"]
    print(json.dumps(evaluate_leave_one_fold_out(table, keys), indent=2))


if __name__ == "__main__":
    _main()
