# %% [markdown]
# # DriverGuard — Tuần 5: GRU và DANN, 5 outer folds theo tài xế
#
# Gắn hai gói private: driverguard-source-week5-20261019.zip và
# driverguard-features-week5-20261019.zip. Chọn GPU T4 x2, Internet ON.
# 30 run: 5 folds x 2 model x 3 seed. Mỗi fold: 32 người train, 8 chọn epoch,
# 10 outer holdout. Chỉ dùng 50 người development. Không có test UTA gốc.
# Chỉ số chính: accuracy cửa sổ 120 s >85%. Kết quả CV vẫn là development,
# do kiến trúc từng được phát triển trên tập người này. Không gọi là nghiệm thu độc lập.

# %%
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

WORK = Path('/kaggle/working/driverguard-week5-20261019')
SOURCE_ZIP = 'driverguard-source-week5-20261019.zip'
FEATURE_ZIP = 'driverguard-features-week5-20261019.zip'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def unpack(filename, directory):
    matches = list(Path('/kaggle/input').rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one exact input {filename}, found {len(matches)}')
    source = matches[0]
    digest = sha256(source)
    stamp = directory / '.archive_sha256'
    if directory.exists():
        if not stamp.is_file() or stamp.read_text().strip() != digest:
            raise RuntimeError('Existing extracted input has different provenance')
        return directory
    directory.mkdir(parents=True)
    with zipfile.ZipFile(source) as z:
        for item in z.infolist():
            target = (directory / item.filename).resolve()
            if not target.is_relative_to(directory.resolve()) or '\\' in item.filename:
                raise ValueError('Unsafe archive member')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlinks are unsupported')
        z.extractall(directory)
    stamp.write_text(digest)
    return directory


source = unpack(SOURCE_ZIP, WORK / 'source')
features = unpack(FEATURE_ZIP, WORK / 'features')
build = json.loads((source / 'build_info.json').read_text())
for filename, digest in build['files'].items():
    path = (source / filename).resolve()
    if not path.is_relative_to(source.resolve()) or sha256(path) != digest:
        raise ValueError(f'Source integrity failure: {filename}')
manifest = features / 'manifest.json'
data = json.loads(manifest.read_text())
if data.get('export', {}).get('test_included') is not False:
    raise ValueError('Only the development-only feature export is allowed')
for row in data['clips']:
    path = (features / row['features']).resolve()
    if row['split'] not in ('train', 'validation') or not path.is_relative_to(features.resolve()):
        raise ValueError('Unexpected split or feature path')
    if sha256(path) != row['features_sha256']:
        raise ValueError('Feature checksum mismatch')
print('Verified development records:', len(data['clips']))

# %%
subprocess.run([sys.executable, '-m', 'pip', 'install', 'pydantic>=2.6', 'PyYAML>=6.0',
                'scipy>=1.11', 'scikit-learn>=1.4'], check=True)
sys.path.insert(0, str(source))
os.environ['PYTHONPATH'] = str(source)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 2 or not all('T4' in name.upper() for name in names):
    raise RuntimeError(f'GPU T4 x2 required, found {names}')
protocol = source / 'configs/training/evaluation_protocol_v1.json'
config = source / 'configs/training/camera_gru_dann_uta_full_v002_window.json'
out = WORK / 'cv'
subprocess.run([sys.executable, '-m', 'scripts.evaluation.run_subject_cv', '--manifest', str(manifest),
                '--protocol', str(protocol), '--config', str(config), '--out', str(out)],
               cwd=source, check=True)

# %%
print((out / 'cv_comparison.json').read_text())
archive = shutil.make_archive('/kaggle/working/driverguard-week5-results-20261019', 'zip', out)
print('Download:', archive)
# Store downloaded outputs in models/from_kaggle/week5_cv_20261019/.
