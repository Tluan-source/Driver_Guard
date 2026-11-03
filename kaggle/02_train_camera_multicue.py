# %% [markdown]
# # DriverGuard: camera baseline on Kaggle GPU
#
# Required inputs:
# - `driverguard-source`: the source archive exported from local code.
# - `driverguard-camera-dev`: manifest.json + train/validation NPZ files from
#   `bundle-camera`. No test files, videos, EEG or Face Landmarker are needed.
#
# This trains the existing binary CAM-02 baseline, not GRU+DANN or a
# three-class UTA model. Feature data must be permitted for cloud processing.
# Use a Private notebook/dataset. GPU training is optional for this small
# baseline; this notebook also verifies the local-to-Kaggle deployment path.

# %%
from pathlib import Path

SOURCE_INPUT = Path("/kaggle/input/driverguard-source")
FEATURE_INPUT = Path("/kaggle/input/driverguard-camera-dev")
WORKING = Path("/kaggle/working/driverguard-run")
DEVICE = "cuda"  # Requires GPU in notebook Settings; no silent CPU fallback.
INSTALL_DEPENDENCIES = True  # First run needs Internet; False when already installed.
SEEDS = [42, 43, 44]
CUE_SETS = ["full", "eyes_mouth", "head", "quality_only"]
EPOCHS = 25
BATCH_SIZE = 128

# %%
import hashlib
import json
import subprocess
import sys
import zipfile

if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError("The current project supports Python 3.10-3.12; check Kaggle runtime compatibility.")
if not Path("/kaggle/input").is_dir():
    raise RuntimeError("Run this notebook on Kaggle. For local smoke, use the documented CLI.")


def input_directory(mount, marker, output):
    if (mount / marker).is_file():
        return mount
    archives = list(mount.glob("*.zip")) if mount.is_dir() else []
    if len(archives) != 1:
        raise RuntimeError(f"Expected {marker} or one exported ZIP in {mount}; update the input path.")
    if output.exists():
        raise RuntimeError(f"Refusing to mix versions in {output}; choose a new WORKING path.")
    with zipfile.ZipFile(archives[0]) as archive:
        for entry in archive.infolist():
            target = (output / entry.filename).resolve()
            if not target.is_relative_to(output.resolve()) or "\\" in entry.filename:
                raise ValueError("Unsafe archive member")
        archive.extractall(output)
    if not (output / marker).is_file():
        raise RuntimeError(f"Missing {marker} in exported input")
    return output


source = input_directory(SOURCE_INPUT, "build_info.json", WORKING / "source")
features = input_directory(FEATURE_INPUT, "manifest.json", WORKING / "features")
build = json.loads((source / "build_info.json").read_text())
for name, expected in build["files"].items():
    path = (source / name).resolve()
    if not path.is_relative_to(source.resolve()):
        raise ValueError("Source build path escapes source root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Source checksum mismatch: {name}")

manifest = json.loads((features / "manifest.json").read_text())
if manifest.get("export", {}).get("test_included") is not False:
    raise RuntimeError("Use bundle-camera to create an explicit train/validation development export.")
if any(r["split"] not in {"train", "validation"} for r in manifest["clips"]):
    raise RuntimeError("Development notebook must not include test records.")
for record in manifest["clips"]:
    path = (features / record["features"]).resolve()
    if not path.is_relative_to(features.resolve()) or not path.is_file():
        raise RuntimeError("Feature path must be relative and present inside the feature input.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["features_sha256"]:
        raise RuntimeError(f"Feature checksum mismatch: {record['clip_id']}")

print("Source revision:", build.get("git_sha"), "dirty:", build.get("working_tree_dirty"))
print("Data:", manifest["export"])

# %%
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, "-m", "pip", "install", "pydantic>=2.6", "PyYAML>=6.0",
                    "scipy>=1.11", "scikit-learn>=1.4"], check=True)
sys.path.insert(0, str(source))

import numpy as np
import scipy
import sklearn
import torch

if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("Enable GPU in Kaggle Settings before training.")
WORKING.mkdir(parents=True, exist_ok=True)
runtime = {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
           "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
           "device": DEVICE, "cuda_available": torch.cuda.is_available(),
           "source_archive_files": build["files"],
           "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}
(WORKING / "runtime.json").write_text(json.dumps(runtime, indent=2))
print(runtime["device"], runtime["gpu"])

# %%
from driverguard.learning.camera.multicue_train import run_experiment

torch.set_num_threads(min(4, torch.get_num_threads()))
report = run_experiment(features / "manifest.json", WORKING / "experiment",
                        seeds=tuple(SEEDS), cue_sets=tuple(CUE_SETS), epochs=EPOCHS,
                        batch_size=BATCH_SIZE, device=DEVICE)
print(json.dumps(report["validation_seed_summary"], indent=2))

# %% [markdown]
# Save Version / Save & Run All. Outputs are under
# `/kaggle/working/driverguard-run`: checkpoints, predictions, metrics,
# protocol and runtime report. Download these to a new local directory.
# CAM-02 does not resume an interrupted training run; keep every completed
# run's output. New GRU+DANN/EEG trainers will need optimizer/RNG resume.
