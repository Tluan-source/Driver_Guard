"""Inventory DROZY alignment inputs and validate an explicitly interpreted mapping."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from driverguard.evaluation.alignment import fit_clock, validate_indices
from driverguard.execution import file_sha256


def audit(root: Path, mapping: Path | None = None):
    root = root.resolve()
    if not root.is_dir() or not any(root.rglob('*.*')):
        return {"status": "missing_data", "root": str(root), "sessions_verified": 0,
                "ready_for_eeg_supervision": False, "missing": ["DROZY video, PSG, Cam-Sync and interpIndices"]}
    if mapping is None:
        return {"status": "mapping_required", "files": len(list(root.rglob('*.*'))),
                "ready_for_eeg_supervision": False, "note": "Declare MAT keys, units and paired anchors; filename presence does not establish alignment"}
    manifest = json.loads(mapping.read_text(encoding="utf-8"))
    sessions = manifest.get("sessions", [])
    if not sessions or len({s["session_id"] for s in sessions}) != len(sessions):
        raise ValueError("Require nonempty, unique session mappings")
    rows = []
    for session in sessions:
        files = {}
        for name in ("video", "psg", "cam_sync", "interp_indices"):
            path = (root / session[name]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or unsafe {name} file")
            files[name] = {"path": str(path), "sha256": file_sha256(path)}
        cam = loadmat(files["cam_sync"]["path"])
        indices = loadmat(files["interp_indices"]["path"])
        if session["cam_sync_key"] not in cam or session["interp_indices_key"] not in indices:
            raise ValueError("Declared MAT variables are missing")
        # Anchor values are explicit seconds interpreted from the declared source variable.
        anchors = np.asarray(cam[session["cam_sync_key"]], float)
        if anchors.ndim != 2 or anchors.shape[1] != 2:
            raise ValueError("Export declared Cam-Sync anchors as N x 2 with explicit video/PSG columns")
        if session.get("anchor_units") != "seconds" or session.get("anchor_columns") != ["video", "psg"]:
            raise ValueError("Explicit seconds and video/PSG column semantics are required")
        clock = fit_clock(anchors[:, 0], anchors[:, 1], video_duration=session["video_duration_s"],
                          psg_duration=session["psg_duration_s"], tolerance_s=manifest["tolerance_s"])
        index_check = validate_indices(np.asarray(indices[session["interp_indices_key"]]).reshape(-1),
                                      index_base=session["index_base"], n_psg_samples=session["n_psg_samples"],
                                      n_video_frames=session["decoded_video_frames"])
        rows.append({"session_id": session["session_id"], "subject": session["subject"],
                     "clock": clock, "indices": index_check, "files": files})
    return {"status": "structural_checks_complete", "sessions": rows,
            "ready_for_eeg_supervision": False,
            "note": "Recording durations/counts and MAT semantics require independent validation; this check never certifies physiological labels"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/drozy"))
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/week6/drozy_alignment_audit.json"))
    args = parser.parse_args()
    report = audit(args.root, args.mapping)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
