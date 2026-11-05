"""Prepare the locked week-6 camera ablations; full training runs only on Kaggle T4 x2."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch

from driverguard.execution import file_sha256
from driverguard.learning.camera.dann.train import TrainConfig, train_experiment
from scripts.data.prepare_subject_cv import prepare


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def make_plan(manifest, settings, out):
    config = json.loads(settings.read_text(encoding="utf-8"))
    protocol = Path(config["fold_protocol"])
    base_path = Path(config["base_config"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data = json.loads(manifest.read_text(encoding="utf-8"))
    people = {r["subject"] for r in data["clips"]}
    if len(people) != 50 or any(r["split"] not in ("train", "validation") for r in data["clips"]):
        raise ValueError("Require the fifty-person development-only feature bundle")
    if people & {"02", "07", "08", "09", "15", "16", "18", "41", "49", "57"}:
        raise ValueError("Original UTA test participants are forbidden")
    folds = prepare(manifest, out / "manifests", protocol)
    fields = TrainConfig.__dataclass_fields__
    jobs = []
    for variant in config["variants"]:
        for fold in folds["folds"]:
            for seed in config["seeds"]:
                values = {k: v for k, v in base.items() if k in fields}
                values.update(variant["overrides"], seed=seed, device="cuda", require_gpus=2)
                cfg = TrainConfig(**values)
                cfg.validate()
                if cfg.context_seconds != 120 or cfg.label_mode != "alert_drowsy":
                    raise ValueError("Ablation must retain the frozen task/context")
                jobs.append({"id": f"fold{fold['fold']}_{variant['id']}_seed{seed}",
                             "variant": variant["id"], "fold": fold["fold"], "config": asdict(cfg),
                             "manifest": str(out / "manifests" / f"fold_{fold['fold']}.json"),
                             "baseline_reference": config["baseline"].format(fold=fold["fold"], seed=seed)})
    plan = {"version": config["version"], "settings_sha256": file_sha256(settings),
            "base_config_sha256": file_sha256(base_path), "protocol_sha256": file_sha256(protocol),
            "feature_manifest_sha256": file_sha256(manifest), "jobs": jobs,
            "original_test_included": False, "independent_acceptance": False}
    target = out / "plan.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != plan:
        raise FileExistsError("Plan differs from existing frozen output; choose a new directory")
    write(target, plan)
    return plan


def smoke_manifest(fold_path, destination):
    original = json.loads(fold_path.read_text(encoding="utf-8"))
    chosen = {}
    for split in ("train", "validation"):
        people = sorted({r["subject"] for r in original["clips"] if r["split"] == split})
        chosen[split] = set(people[:2])
    manifest = copy.deepcopy(original)
    manifest["clips"] = [r for r in original["clips"]
                         if r["split"] in chosen and r["subject"] in chosen[r["split"]] and r["label"] in (0, 2)]
    manifest["purpose"] = "short_cpu_smoke; original NPZ bytes; four inner-development subjects; no outer holdout"
    write(destination, manifest)
    return destination


def execute(plan, out, *, smoke=False):
    torch.set_num_threads(4)
    if not smoke:
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if not Path('/kaggle/working').is_dir() or len(names) != 2 or not all('T4' in n.upper() for n in names):
            raise RuntimeError("Full ablation training requires Kaggle GPU T4 x2; local execution is dry-run or short smoke")
    results, seen = [], set()
    for job in plan["jobs"]:
        if smoke and job["variant"] in seen:
            continue
        seen.add(job["variant"])
        destination = out / (f"smoke_{job['variant']}" if smoke else job["id"])
        manifest = Path(job["manifest"])
        config = job["config"].copy()
        if smoke:
            manifest = smoke_manifest(manifest, out / f"smoke_manifest_{job['variant']}.json")
            config.update(device="cpu", require_gpus=0, epochs=2, patience=2, stride=1200, batch_size=32)
        stamp = {"outer_fold": job["fold"], "original_test_included": False,
                 "purpose": "smoke" if smoke else "outer_development_cv",
                 "config": config, "manifest_sha256": file_sha256(manifest),
                 "settings_sha256": plan["settings_sha256"]}
        destination.mkdir(parents=True, exist_ok=True)
        stamp_path = destination / "cv_provenance.json"
        if stamp_path.exists() and json.loads(stamp_path.read_text(encoding="utf-8")) != stamp:
            raise FileExistsError("Existing run provenance differs")
        write(stamp_path, stamp)
        summary = destination / "metrics/summary.json"
        if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("status") == "completed":
            value = json.loads(summary.read_text(encoding="utf-8"))
        else:
            resume = destination / "checkpoints/last.pt"
            value = train_experiment(manifest, destination, TrainConfig(**config),
                                     resume=resume if resume.exists() else None, evaluate_test=not smoke)
        # Smoke status is explicit even though the trainer completes its small epoch budget.
        results.append({"id": job["id"], "variant": job["variant"], "purpose": stamp["purpose"],
                        "fold": job["fold"],
                        "status": value["status"], "checkpoint_sha256": value["checkpoint_sha256"],
                        "validation": value["metrics"]["validation"],
                        "outer_development_metrics": value["metrics"]["test"], "independent_acceptance": False})
        write(out / ("smoke_summary.json" if smoke else "ablation_progress.json"), results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/features/week5_development/manifest.json"))
    parser.add_argument("--settings", type=Path, default=Path("configs/training/week6_ablation_v1.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/week6/ablation"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args.manifest.resolve(), args.settings.resolve(), args.out.resolve())
    print(json.dumps({"prepared_jobs": len(plan["jobs"]), "original_test_included": False}), flush=True)
    if args.execute or args.smoke:
        execute(plan, args.out.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
