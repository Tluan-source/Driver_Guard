# %% [markdown]
# # DriverGuard — GRU + DANN / T4 ×2
#
# Required inputs (private Kaggle datasets):
# - `driverguard-source-v2`: source ZIP from `driverguard.execution bundle-source`.
# - `driverguard-uta-full-v002`: development NPZ ZIP from `bundle-camera`, built
#   from full UTA-RLDD videos (manifest v002). Schema: manifest.json with
#   class_names alert/low_vigilance/drowsy, 19 camera features at 10 Hz,
#   subject-disjoint train/validation, labels 0/1/2. No test, raw video, or EEG.
#
# Default experiment: full UTA alert vs drowsy, 120-second context with
# window pooling, GRU vs DANN, 3 seeds. Local CPU runs of this protocol exist
# (runs/gru_dann/uta_full_v002_20261009_window); Kaggle repeats it on T4 x2.
# For the earlier binary pilot, attach `driverguard-camera-dev`, set FEATURE_NAME
# to it, LABEL_MODE='binary_pilot', POOLING='frame', CONTEXT_SECONDS=[5] and the
# pilot hyperparameters; pilot clips cannot provide 20 seconds.
#
# In Kaggle select Settings → Accelerator → GPU T4 ×2 and Internet ON.
# The notebook verifies both GPU names and refuses CPU/single-GPU fallback.
# Source/data hashes, runtime, optimizer/RNG, metrics, predictions and
# inference weights are saved in separate directories. A validation score
# >85% is a candidate result, not independent final-test acceptance.

# %%
from pathlib import Path

# Kaggle dataset names (or ZIP stems) to use. Only an exact name matches, so an
# older `driverguard-source` input left attached is ignored, not mixed in.
SOURCE_NAME = 'driverguard-source-v2'
FEATURE_NAME = 'driverguard-uta-full-v002'
# Set these to exact mount directories/ZIPs to bypass name discovery.
SOURCE_INPUT = None
FEATURE_INPUT = None
# Use a new directory per source version so a failed run is not resumed.
WORKING = Path('/kaggle/working/driverguard-gru-dann-uta-full-v002')
# alert_drowsy drops low-vigilance recordings (0 -> alert, 10 -> drowsy); three_class keeps all.
LABEL_MODE = 'alert_drowsy'
# Window pooling: per-second summaries (window-relative eye-closure cues included),
# their window statistics and a small GRU over 10 s blocks; 20/30/60/120 s history.
# Frame pooling is the 5/10/20 s pilot architecture.
POOLING = 'window'
CONTEXT_SECONDS = [120]
MODES = ['dann', 'gru']
SEEDS = [42, 43, 44]
EPOCHS = 10
BATCH_SIZE = 128
PATIENCE = 4
# Same values as configs/training/camera_gru_dann_uta_full_v002_window.json
# (docs/reports/24_gru_dann_uta_full_20261009.md).
HYPERPARAMETERS = {'stride': 25, 'hidden_size': 8, 'layers': 1, 'dropout': 0.5,
                   'learning_rate': 1e-3, 'weight_decay': 0.01, 'input_noise': 0.0,
                   'domain_weight': 1.0, 'domain_lr_multiplier': 10.0}
INSTALL_DEPENDENCIES = True
# To resume, add a previous output as a Kaggle input and point here at its
# experiment directory (containing dann_5s_seed42/checkpoints/last.pt, etc.).
RESUME_ROOT = None

# %%
import hashlib
import json
import os
import subprocess
import sys
import zipfile

if not Path('/kaggle/input').is_dir():
    raise RuntimeError('Use Kaggle for this notebook; use the trainer CLI for local CPU smoke.')
# MediaPipe limits the full package to <3.13; this trainer only needs
# NumPy/Torch/Pydantic/PyYAML/SciPy/scikit-learn, verified on 3.13.
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise RuntimeError(f'GRU+DANN training supports Python 3.10–3.13, found {sys.version}.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locate_input(explicit, marker, name):
    if explicit is not None:
        mount = Path(explicit)
        if mount.is_file() and mount.suffix == '.zip':
            return mount
        if (mount / marker).is_file():
            return mount
        matches = list(mount.glob('*.zip'))
    else:
        # Kaggle may keep the ZIP or extract it (optionally into a folder
        # named after the ZIP); accept any of these under the named dataset.
        root = Path('/kaggle/input')
        extracted = [p.parent for p in root.rglob(marker)]
        archives = [p for p in root.rglob('*.zip') if not any(p.is_relative_to(d) for d in extracted)]
        matches = [p for p in extracted + archives if name in p.relative_to(root).parts or p.stem == name]
        matches = [p for p in matches if not any(p != q and p.is_relative_to(q) for q in matches)]
    if len(matches) != 1:
        attached = sorted({str(p.parent) for p in Path('/kaggle/input').rglob(marker)})
        raise RuntimeError(f'Expected exactly one {name!r} input with {marker}; found {matches}. '
                           f'Inputs containing {marker}: {attached}. Fix the name or set the input path.')
    return matches[0]


def unpack(item, marker, destination):
    if item.is_dir():
        return item
    # Input extraction is idempotent only for an identical archive.
    stamp = destination / '.archive-sha256'
    digest = sha256(item)
    if destination.exists():
        if not stamp.is_file() or stamp.read_text().strip() != digest:
            raise RuntimeError(f'Existing extraction does not match input: {destination}')
        return destination
    destination.mkdir(parents=True)
    with zipfile.ZipFile(item) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or '\\' in entry.filename:
                raise ValueError('Unsafe archive path')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlinks are not supported in inputs')
        archive.extractall(destination)
    if not (destination / marker).is_file():
        raise RuntimeError(f'Missing {marker}')
    stamp.write_text(digest)
    return destination


source = unpack(locate_input(SOURCE_INPUT, 'build_info.json', SOURCE_NAME),
                'build_info.json', WORKING / 'inputs' / 'source')
features = unpack(locate_input(FEATURE_INPUT, 'manifest.json', FEATURE_NAME),
                  'manifest.json', WORKING / 'inputs' / 'features')
build = json.loads((source / 'build_info.json').read_text())
for name, expected in build['files'].items():
    path = (source / name).resolve()
    if not path.is_relative_to(source.resolve()) or sha256(path) != expected:
        raise RuntimeError(f'Source checksum/path failed: {name}')
manifest_path = features / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
if manifest.get('export', {}).get('test_included') is not False:
    raise RuntimeError('Expected an explicit development-only bundle-camera export.')
for row in manifest['clips']:
    path = (features / row['features']).resolve()
    if row['split'] not in {'train', 'validation'}:
        raise RuntimeError('Do not load final-test records into this development experiment.')
    if not path.is_relative_to(features.resolve()) or sha256(path) != row['features_sha256']:
        raise RuntimeError(f'Feature checksum/path failed: {row["clip_id"]}')
print('Source input:', source, '| verified files:', len(build['files']))
print('Feature input:', features)
print('Verified data export:', manifest['export'])

# %%
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pydantic>=2.6', 'PyYAML>=6.0',
                    'scipy>=1.11', 'scikit-learn>=1.4'], check=True)
# Torch/NumPy are supplied by Kaggle; do not replace CUDA Torch.
sys.path.insert(0, str(source))
os.environ['PYTHONPATH'] = str(source) + os.pathsep + os.environ.get('PYTHONPATH', '')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(gpu_names) != 2 or not all('T4' in name.upper() for name in gpu_names):
    raise RuntimeError(f'Requires exactly GPU T4 x2. Found {gpu_names}; change Kaggle Accelerator settings.')
runtime_dir = WORKING / 'reports'
runtime_dir.mkdir(parents=True, exist_ok=True)
runtime = {'python': sys.version, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
           'gpu_names': gpu_names, 'accelerator_verified': True,
           'source_build_sha256': sha256(source / 'build_info.json'),
           'feature_manifest_sha256': sha256(manifest_path), 'label_mode': LABEL_MODE,
           'test_included': False, 'target': {'metric': 'accuracy OR macro_f1', 'strictly_greater_than': 0.85}}
(runtime_dir / 'runtime.json').write_text(json.dumps(runtime, indent=2))
print(json.dumps(runtime, indent=2))

# %%
# Each model/seed/context has its own checkpoint, metrics, predictions and
# inference directories. All configurations are declared before training.
summaries = []
for context in CONTEXT_SECONDS:
    for seed in SEEDS:
        for mode in MODES:
            run_name = f'{mode}_{context}s_seed{seed}'
            out_dir = WORKING / 'experiments' / run_name
            args = [sys.executable, '-m', 'driverguard.learning.camera.dann.train',
                    '--manifest', str(manifest_path), '--out-dir', str(out_dir),
                    '--mode', mode, '--label-mode', LABEL_MODE, '--pooling', POOLING,
                    '--context-seconds', str(context), '--epochs', str(EPOCHS),
                    '--batch-size', str(BATCH_SIZE), '--patience', str(PATIENCE),
                    '--seed', str(seed), '--device', 'cuda', '--require-gpus', '2']
            for key, value in HYPERPARAMETERS.items():
                args += ['--' + key.replace('_', '-'), str(value)]
            local_last = out_dir / 'checkpoints' / 'last.pt'
            previous_last = Path(RESUME_ROOT) / run_name / 'checkpoints' / 'last.pt' if RESUME_ROOT else None
            resume = local_last if local_last.is_file() else previous_last
            if resume is not None and resume.is_file():
                args += ['--resume', str(resume)]
            print('Running', run_name, flush=True)
            subprocess.run(args, cwd=source, check=True)
            summary = json.loads((out_dir / 'metrics' / 'summary.json').read_text())
            summaries.append({'run': run_name, 'mode': mode, 'seed': seed, 'context_seconds': context,
                              'summary': summary})
            (runtime_dir / 'experiments.json').write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summary, indent=2), flush=True)

# %% [markdown]
# Save Version → Save & Run All to retain outputs. Download the `experiments`
# directory into `models/from_kaggle/gru_dann_uta_full_v002/` and `reports` into
# `outputs/kaggle/<date>/reports/`. The management dashboard discovers runs.
# Report all seeds; do not promote the best seed as a replicated result.
# Final acceptance requires a locked independent cohort. This notebook
# does not evaluate or adapt to the previously used test subjects.
