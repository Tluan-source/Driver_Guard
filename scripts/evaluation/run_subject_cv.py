"""Run the predeclared week-5 outer CV on Kaggle GPUs; inner holdout selects epochs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driverguard.execution import file_sha256
from driverguard.learning.camera.dann.train import TrainConfig, train_experiment
from scripts.data.prepare_subject_cv import prepare


def run(manifest: Path, protocol_path: Path, config_path: Path, out: Path):
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Full CV training is restricted to Kaggle; use prepare_subject_cv for local audits")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or not all("T4" in n.upper() for n in names):
        raise RuntimeError(f"Requires GPU T4 x2; found {names}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["context_seconds"] != protocol["primary"]["context_seconds"]
            or config["label_mode"] != protocol["task"]):
        raise ValueError("Training config disagrees with locked evaluation protocol")
    out.mkdir(parents=True, exist_ok=True)
    plan = prepare(manifest, out / "manifests", protocol_path)
    fields = TrainConfig.__dataclass_fields__
    base = {k: v for k, v in config.items() if k in fields}
    completed = []
    for fold in plan["folds"]:
        for mode in protocol["modes"]:
            for seed in protocol["seeds"]:
                name = f"fold{fold['fold']}_{mode}_seed{seed}"
                destination = out / "experiments" / name
                checkpoint = destination / "checkpoints/last.pt"
                summary_path = destination / "metrics/summary.json"
                stamp = destination / "week5_provenance.json"
                expected = {"protocol_sha256": file_sha256(protocol_path),
                            "config_sha256": file_sha256(config_path),
                            "fold_manifest_sha256": file_sha256(out / "manifests" / f"fold_{fold['fold']}.json"),
                            "outer_fold": fold['fold'], "original_test_included": False}
                if destination.exists():
                    if not stamp.is_file() or json.loads(stamp.read_text()) != expected:
                        raise ValueError(f"Existing run has different provenance: {name}")
                else:
                    destination.mkdir(parents=True)
                    stamp.write_text(json.dumps(expected, indent=2))
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text())
                else:
                    summary = None
                if summary is None or summary.get("status") != "completed" or summary["metrics"].get("test") is None:
                    settings = TrainConfig(**{**base, "mode": mode, "seed": seed,
                                               "device": "cuda", "require_gpus": 2})
                    summary = train_experiment(out / "manifests" / f"fold_{fold['fold']}.json",
                                               destination, settings,
                                               resume=checkpoint if checkpoint.exists() else None,
                                               evaluate_test=True)
                completed.append({"run": name, "fold": fold["fold"], "mode": mode, "seed": seed,
                                  "outer_metrics": summary["metrics"]["test"],
                                  "original_test_included": False})
                (out / "progress.json").write_text(json.dumps(completed, indent=2, allow_nan=False))
                print(f"Completed {len(completed)}/30: {name}", flush=True)
    report = {"status": "completed", "protocol_sha256": file_sha256(protocol_path),
              "hardware": names, "runs": completed, "independent_acceptance": False,
              "caveat": "Architecture was developed on this population. Outer-fold results remain development evidence."}
    from scripts.evaluation.summarize_subject_cv import summarize
    summarize(out, protocol_path)
    (out / "cv_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "protocol", "config", "out"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.protocol, a.config, a.out)


if __name__ == "__main__":
    main()
