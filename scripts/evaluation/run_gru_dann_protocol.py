"""Run a locked GRU vs GRU+DANN protocol locally, several runs in parallel on CPU.

    python -m scripts.evaluation.run_gru_dann_protocol \
        --config configs/training/camera_gru_dann_uta_full_v002.json \
        --manifest data/features/uta_rldd_full_v002/manifest.json \
        --out runs/gru_dann/uta_full_v002_20261009 --parallel 3 --threads 4

Every (mode, seed) gets its own run directory; finished runs are skipped. Test features are
never opened. `--label-mode` overrides the config's label mode for a secondary protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROTOCOL_KEYS = ("modes", "seeds", "selection_metric")


def commands(config: dict, manifest: Path, out: Path, label_mode: str | None) -> list[tuple[Path, list[str]]]:
    settings = {key: value for key, value in config.items() if not key.startswith("_") and key not in PROTOCOL_KEYS}
    if label_mode:
        settings["label_mode"] = label_mode
    jobs = []
    for seed in config["seeds"]:
        for mode in config["modes"]:
            run = out / f"{mode}_{settings['label_mode']}_{settings['context_seconds']}s_seed{seed}"
            args = [sys.executable, "-m", "driverguard.learning.camera.dann.train", "--manifest", str(manifest),
                    "--out-dir", str(run), "--mode", mode, "--seed", str(seed), "--device", "cpu"]
            for key, value in settings.items():
                args += ["--" + key.replace("_", "-"), str(value)]
            jobs.append((run, args))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-mode")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    jobs = [job for job in commands(config, arguments.manifest, arguments.out, arguments.label_mode)
            if not (job[0] / "metrics" / "summary.json").is_file()]
    arguments.out.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": str(arguments.threads),
                   "MKL_NUM_THREADS": str(arguments.threads)}

    def launch(job):
        run, args = job
        if run.exists():
            last = run / "checkpoints" / "last.pt"
            if not last.is_file():
                raise RuntimeError(f"{run} exists without last.pt; inspect it before rerunning")
            args = [*args, "--resume", str(last)]
        log = arguments.out / f"{run.name}.log"
        with log.open("a", encoding="utf-8") as stream:
            code = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                  check=False).returncode
        print(f"{run.name}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(max_workers=arguments.parallel) as pool:
        codes = list(pool.map(launch, jobs))
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
