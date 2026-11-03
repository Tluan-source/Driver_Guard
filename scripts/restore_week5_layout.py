"""Keep legacy entry points working with the recovered camera/eeg packages."""
from __future__ import annotations

import shutil
from pathlib import Path


def alias(path, module, backup):
    if path.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_text(
        '"""Compatibility entry point for the recovered week-5 module layout."""\n'
        'import importlib\nimport sys\n\n'
        f'_module = importlib.import_module("{module}")\n'
        'if __name__ == "__main__":\n    _module.main()\n'
        'else:\n    sys.modules[__name__] = _module\n', encoding="utf-8")


def restore(root):
    root = Path(root).resolve()
    backup = root / "outputs/legacy_before_namespace_restore"
    mappings = {**{f"camera_{name}": f"camera.{name}" for name in (
        "baseline", "cli", "extract", "features", "model", "statistics", "train")},
        **{name: f"eeg.{name}" for name in (
            "cli", "cross_validation", "data", "features", "fitting", "improvement", "model", "quality", "train")},
        "blink": "camera.blink"}
    for old, new in mappings.items():
        alias(root / f"driverguard/learning/{old}.py", f"driverguard.learning.{new}", backup / f"learning/{old}.py")
    for name in ("prepare_camera_annotation", "prepare_camera_uta", "prepare_fatigue_eeg"):
        alias(root / f"scripts/{name}.py", f"scripts.data.{name}", backup / f"scripts/{name}.py")
    for name in ("test_learning_eeg.py", "test_learning_quality.py", "test_learning_cli.py"):
        path = root / "tests" / name
        text = path.read_text(encoding="utf-8")
        text = text.replace("from driverguard.learning.eeg import CHANNEL_NAMES",
                            "from driverguard.learning.eeg.preprocessing import CHANNEL_NAMES")
        text = text.replace("from driverguard.learning import eeg",
                            "from driverguard.learning.eeg import preprocessing as eeg")
        path.write_text(text, encoding="utf-8")
    print(f"Restored {len(mappings) + 3} compatibility entry points")


if __name__ == "__main__":
    restore(Path(__file__).resolve().parents[1])
