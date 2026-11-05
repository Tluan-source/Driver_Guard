# %% [markdown]
# # DriverGuard tuần 6: chuẩn bị và chạy camera ablation
# Gắn source bundle `driverguard-source-week6-20261105.zip` và dataset feature
# development tuần 5. Chọn GPU T4 x2. Hai ablation x5 fold x3 seed =30 run mới.
# Baseline GRU 15 run tuần 5 được giữ làm đối chứng. Kết quả là development CV.

# %%
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Notebook này chạy trong Kaggle; local dùng --smoke hoặc dry-run')

WORK = Path('/kaggle/working/driverguard-week6-20261105')
SOURCE_NAME = 'driverguard-source-week6-20261105.zip'
WORK.mkdir(exist_ok=True)
archives = list(Path('/kaggle/input').rglob(SOURCE_NAME))
if len(archives) == 1:
    extracted = WORK / 'source_input'
    archive_hash = hashlib.sha256(archives[0].read_bytes()).hexdigest()
    if extracted.exists():
        if (extracted / '.archive_sha256').read_text().strip() != archive_hash:
            raise RuntimeError('Source input đã thay đổi; dùng working directory mới')
    else:
        extracted.mkdir()
        with zipfile.ZipFile(archives[0]) as archive:
            for info in archive.infolist():
                target = (extracted / info.filename).resolve()
                if not target.is_relative_to(extracted.resolve()) or '\\' in info.filename:
                    raise ValueError('Unsafe archive path')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are forbidden')
            if archive.testzip() is not None:
                raise ValueError('Source ZIP CRC failure')
            archive.extractall(extracted)
        (extracted / '.archive_sha256').write_text(archive_hash)
    source = extracted
else:
    sources = [p.parent for p in Path('/kaggle/input').rglob('build_info.json')
               if (p.parent / 'configs/training/week6_ablation_v1.json').is_file()]
    if len(sources) != 1:
        raise RuntimeError('Gắn đúng một source bundle tuần 6')
    source = sources[0]

build = json.loads((source / 'build_info.json').read_text())
for name, expected in build['files'].items():
    path = (source / name).resolve()
    if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Source hash mismatch: {name}')

repo = WORK / 'repo'
if not repo.exists():
    shutil.copytree(source, repo)
os.chdir(repo)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[train]'])

# %%
feature_sets = [p for p in Path('/kaggle/input').rglob('manifest.json')
                if p.parent.name == 'driverguard-features-week5-20261019']
if len(feature_sets) != 1:
    raise RuntimeError('Gắn dataset driverguard-week5-dev-20261020 chứa feature development gốc')
manifest = feature_sets[0]
data = json.loads(manifest.read_text())
if len({r['subject'] for r in data['clips']}) != 50 or any(r['split'] == 'test' for r in data['clips']):
    raise ValueError('Chỉ chấp nhận feature 50 người development; khóa test gốc')
for row in data['clips']:
    path = (manifest.parent / Path(row['features']).name).resolve()
    if not path.is_relative_to(manifest.parent.resolve()):
        raise ValueError('Unsafe feature path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['features_sha256']:
        raise ValueError('Feature hash mismatch')
    row['features'] = str(path)
local_manifest = WORK / 'development_manifest.json'
local_manifest.write_text(json.dumps(data, indent=2))
print('Verified feature records:', len(data['clips']))

# %%
command = [sys.executable, '-m', 'scripts.evaluation.run_week6_ablation',
           '--manifest', str(local_manifest), '--settings', 'configs/training/week6_ablation_v1.json',
           '--out', str(WORK / 'results'), '--execute']
subprocess.check_call(command)

# %% [markdown]
# Giữ nguyên plan/provenance/checkpoint/CSV của cả 30 run. So sánh theo người và
# cùng clip/timestamp với baseline tuần 5. Không gọi kết quả CV là test độc lập.
