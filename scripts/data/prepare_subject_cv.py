"""Create subject-disjoint outer folds from development metadata; never load original test NPZs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from driverguard.execution import file_sha256


def subject_folds(records: list[dict], folds: int = 5, inner_subjects: int = 8, seed: int = 20261019):
    partitions = {name: {r["subject"] for r in records if r["split"] == name}
                  for name in ("train", "validation", "test")}
    if any(partitions[a] & partitions[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Original subject splits overlap")
    subjects = sorted(partitions["train"] | partitions["validation"])
    if folds < 2 or len(subjects) < folds or inner_subjects < 1:
        raise ValueError("Invalid fold population")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects)
    output = []
    for index, outer in enumerate(np.array_split(shuffled, folds)):
        remaining = [str(s) for s in shuffled if s not in set(outer)]
        if len(remaining) <= inner_subjects:
            raise ValueError("No inner training subjects remain")
        inside = np.random.default_rng(seed + index + 1).permutation(remaining)
        output.append({"fold": index + 1, "train": sorted(map(str, inside[inner_subjects:])),
                       "validation": sorted(map(str, inside[:inner_subjects])),
                       "test": sorted(map(str, outer))})
    return output


def prepare(manifest_path: Path, out: Path, protocol_path: Path) -> dict:
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    settings = protocol["cross_validation"]
    folds = subject_folds(original["clips"], settings["outer_folds"],
                          settings["inner_validation_subjects"], settings["split_seed"])
    provenance = {"source_manifest_sha256": file_sha256(manifest_path),
                  "protocol_sha256": file_sha256(protocol_path), "original_test_included": False,
                  "test_field_semantics": "outer development holdout; never original UTA test",
                  "folds": folds}
    if out.exists():
        stamp = out / "folds.json"
        if not stamp.is_file() or json.loads(stamp.read_text(encoding="utf-8")) != provenance:
            raise FileExistsError("CV output exists with different provenance")
        return provenance
    out.mkdir(parents=True)
    for fold in folds:
        assignments = {subject: split for split in ("train", "validation", "test") for subject in fold[split]}
        manifest = copy.deepcopy(original)
        selected = []
        for r in original["clips"]:
            if r["split"] == "test":
                continue
            row = copy.deepcopy(r)
            row["original_split"], row["split"] = r["split"], assignments[r["subject"]]
            path = Path(row["features"])
            row["features"] = str((manifest_path.parent / path).resolve()) if not path.is_absolute() else str(path)
            selected.append(row)
        manifest["clips"] = selected
        manifest["cv"] = {**provenance, "fold": fold["fold"]}
        (out / f"fold_{fold['fold']}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.manifest, args.out, args.protocol), indent=2))


if __name__ == "__main__":
    main()
