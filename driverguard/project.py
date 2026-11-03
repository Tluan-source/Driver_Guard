"""Small project entry point: stage status, environment checks and explicit tasks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_config(root: Path = ROOT) -> dict:
    return json.loads((root / "configs/project.json").read_text(encoding="utf-8"))


def task_command(name: str, extra: list[str], root: Path = ROOT) -> list[str]:
    task = project_config(root)["tasks"][name]
    return [sys.executable, *task["args"], *extra]


def doctor() -> dict:
    packages = ("numpy", "yaml", "pydantic", "cv2", "mediapipe", "torch", "scipy", "sklearn", "pytest")
    availability = {name: importlib.util.find_spec(name) is not None for name in packages}
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "supported_python": (3, 10) <= sys.version_info[:2] < (3, 13),
            "packages_found": availability,
            "face_landmarker": (ROOT / "models/face_landmarker.task").is_file(),
            "camera_pilot_manifest": (ROOT / "data/features/camera_uta_pilot/manifest.json").is_file(),
            "week5_development_manifest": (ROOT / "data/features/week5_development/manifest.json").is_file(),
            "week5_inference_checkpoints": len(list((ROOT / "models/from_kaggle/week5_gru_dann").glob("*/inference/model.pt"))),
            "note": "Discovery only; run week5-audit, week5-replay and week5-runtime to verify recovered artifacts."}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Show stages and next work; statuses are maintained explicitly")
    sub.add_parser("doctor", help="Check interpreter/package discovery without loading models")
    sub.add_parser("tasks", help="List supported commands")
    run = sub.add_parser("run", help="Run one task, forwarding options after --")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("task", choices=tuple(project_config()["tasks"]))
    run.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command in {None, "status"}:
        config = project_config()
        print(f"DriverGuard | stage {config['current_stage']} | updated {config['updated']}")
        for stage in config["stages"]:
            print(f"{stage['id']} [{stage['state']}] {stage['title']}\n   {stage['next']}")
        print("Details: docs/ROADMAP.md | Commands: python -m driverguard.project tasks")
        return 0
    if args.command == "doctor":
        result = doctor()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["supported_python"] and all(result["packages_found"].values()) else 1
    if args.command == "tasks":
        for name, task in project_config()["tasks"].items():
            print(f"{name:16} {task['description']}")
        return 0
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
    command = task_command(args.task, extra)
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
