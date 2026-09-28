# %% [markdown]
# # DriverGuard — trích xuất FrameSignals từ video (chạy trên Kaggle)
#
# Notebook này chạy **một lần** phần perception nặng (MediaPipe Face Landmarker) trên toàn bộ video
# của dataset, ghi ra mỗi video một file `.parquet` (EAR, MAR, blendshape, head pose, chất lượng mặt…).
# Sau đó mọi thí nghiệm temporal / risk / hiệu chỉnh (Hướng 1) / GRU (Hướng 2) chạy lại trên laptop
# bằng `driverguard replay` trong vài giây — không cần chạy lại MediaPipe.
#
# **Thiết lập notebook trên Kaggle**
# 1. Settings → Internet: **ON** (để `pip install mediapipe` và tải model). Accelerator: None (CPU là đủ).
# 2. Add Input:
#    * dataset video, ví dụ `rishab260/uta-reallife-drowsiness-dataset` (bản mirror UTA-RLDD — xem cảnh báo license
#      trong `docs/02_data_plan.md`), hoặc dataset riêng tư bạn tự upload (YawDD, DMD);
#    * dataset mã nguồn `driverguard-src` (zip thư mục `driverguard/` của repo, xem `kaggle/README.md`).
# 3. Sửa ô CONFIG bên dưới (DATA_ROOT, FOLDS…), rồi **Save Version → Save & Run All (Commit)** để chạy nền.
# 4. Khi xong: tab Output → tải `features_<tag>.zip`, giải nén vào `data/features/<dataset>/` trên máy.
#
# Giới hạn Kaggle: ~12 giờ/phiên, 4 CPU. UTA-RLDD (~30 giờ video) ở 15 fps nên chia **1–2 fold mỗi phiên**.
# Notebook bỏ qua video đã có output → có thể chạy tiếp ở phiên sau (attach output cũ làm input).

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe>=0.10.14", "pyarrow"], check=False)

# %%
# ============================ CONFIG ============================
DATASET = "uta_rldd"  # uta_rldd | yawdd | generic
DATA_ROOT = "/kaggle/input/uta-reallife-drowsiness-dataset"
SRC_CANDIDATES = ["/kaggle/input/driverguard-src", "/kaggle/input/driverguard-src/DriverGuard", "/kaggle/working/DriverGuard"]
OUT_DIR = "/kaggle/working/features"
PREVIOUS_OUTPUT = None  # e.g. "/kaggle/input/dg-features-fold1" to skip videos already done
FOLDS = [1]  # UTA-RLDD: which official folds to process this session (None = all)
TARGET_FPS = 15.0  # same rate as the edge target
RESIZE_WIDTH = 640  # downscale before MediaPipe (EAR/MAR are ratios -> scale invariant)
N_WORKERS = None  # None = all CPUs
TIME_BUDGET_H = 11.0  # stop starting new videos after this (Kaggle kills at ~12h)
FEATURE_EXT = ".parquet"  # ".parquet" (needs pyarrow, default on Kaggle) or ".csv.gz"
TAG = f"{DATASET}_fold{'-'.join(map(str, FOLDS))}" if FOLDS else DATASET
# ================================================================

# %%
import glob
import json
import multiprocessing
import os
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for c in SRC_CANDIDATES:
    if Path(c, "driverguard", "__init__.py").exists():
        sys.path.insert(0, c)
        break
else:
    raise SystemExit("Không tìm thấy mã nguồn driverguard — attach dataset 'driverguard-src' (xem kaggle/README.md)")

import driverguard  # noqa: E402

print("driverguard", driverguard.__version__)

MODEL = Path("/kaggle/working/models/face_landmarker.task")
if not MODEL.exists():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MODEL)
print("model", MODEL, MODEL.stat().st_size)

# %%
VIDEO_EXT = (".mov", ".mp4", ".avi", ".m4v", ".mkv", ".MOV", ".MP4", ".AVI")


def build_manifest(root: str) -> list[dict]:
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix not in VIDEO_EXT:
            continue
        if DATASET == "uta_rldd":
            from driverguard.evaluation.uta_rldd import parse_path

            r = parse_path(p)
            if r is None:
                continue
        else:
            r = {"path": str(p), "subject": p.parent.name, "fold": None, "label": None,
                 "video_id": re.sub(r"[^A-Za-z0-9_-]+", "_", str(p.relative_to(root).with_suffix("")))}
        rows.append(r)
    return rows


manifest = build_manifest(DATA_ROOT)
print(len(manifest), "videos;", len({r["subject"] for r in manifest}), "subjects;",
      "folds:", sorted({r["fold"] for r in manifest if r["fold"] is not None}))
todo = [r for r in manifest if not FOLDS or r["fold"] in FOLDS]
done_dirs = [OUT_DIR] + ([PREVIOUS_OUTPUT] if PREVIOUS_OUTPUT else [])
todo = [r for r in todo if not any(Path(d, f"{r['video_id']}{FEATURE_EXT}").exists() for d in done_dirs)]
print(len(todo), "videos to process this session")
os.makedirs(OUT_DIR, exist_ok=True)
with open(Path(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)


# %%
def process_video(row: dict) -> dict:
    import cv2

    from driverguard.capture import VideoFileSource
    from driverguard.config import load_config
    from driverguard.io import write_features
    from driverguard.perception import MediaPipeFaceLandmarker, NullPhoneDetector, PerceptionExtractor

    t0 = time.time()
    cfg = load_config(overrides={"models.phone_backend": "none", "runtime.target_fps": TARGET_FPS})
    ex = PerceptionExtractor(cfg, MediaPipeFaceLandmarker(MODEL), NullPhoneDetector())
    src = VideoFileSource(row["path"], target_fps=TARGET_FPS)
    rows = []
    for fr in src.frames():
        img = fr.image_bgr
        if img is not None and RESIZE_WIDTH and img.shape[1] > RESIZE_WIDTH:
            h = int(img.shape[0] * RESIZE_WIDTH / img.shape[1])
            img = cv2.resize(img, (RESIZE_WIDTH, h), interpolation=cv2.INTER_AREA)
        rows.append(ex.process(img, fr.ts_ms, fr.camera_ok))
    src.close()
    ex.close()
    out = Path(OUT_DIR, f"{row['video_id']}{FEATURE_EXT}")
    meta = {**row, "src_fps": src.fps, "target_fps": TARGET_FPS, "n": len(rows),
            "face_rate": sum(r.face_detected for r in rows) / max(1, len(rows)), "secs": round(time.time() - t0, 1)}
    write_features(rows, out, meta=meta)
    return meta


# %%
start = time.time()
results, failed = [], []
workers = N_WORKERS or os.cpu_count()
with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
    futures = {}
    queue = list(todo)
    while queue or futures:
        while queue and len(futures) < workers and (time.time() - start) / 3600 < TIME_BUDGET_H:
            r = queue.pop(0)
            futures[pool.submit(process_video, r)] = r
        if not futures:
            break
        for fut in as_completed(list(futures)):
            r = futures.pop(fut)
            try:
                m = fut.result()
                results.append(m)
                print(f"[{len(results)}/{len(todo)}] {m['video_id']}: {m['n']} frames, face {m['face_rate']:.0%}, "
                      f"{m['secs']}s  (elapsed {(time.time() - start) / 60:.0f} min)", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append({"video_id": r["video_id"], "error": repr(e)})
                print("FAILED", r["video_id"], e, flush=True)
            break  # refill the pool

print(f"done {len(results)}, failed {len(failed)}, remaining {len(todo) - len(results) - len(failed)}")
with open(Path(OUT_DIR, f"run_{TAG}.json"), "w") as f:
    json.dump({"results": results, "failed": failed}, f, indent=1)

# %%
zip_path = f"/kaggle/working/features_{TAG}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
    for p in glob.glob(f"{OUT_DIR}/*"):
        z.write(p, arcname=os.path.basename(p))
print("->", zip_path, round(os.path.getsize(zip_path) / 1e6, 1), "MB")
