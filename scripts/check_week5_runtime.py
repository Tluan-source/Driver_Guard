"""Exercise recovered weights through the debug API and the 120-second live path."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from starlette.testclient import TestClient

from driverguard.api.project_status import build_project_overview
from driverguard.config import REPO_ROOT, load_config
from driverguard.debug.model_lab import LiveModel
from driverguard.debug.server import create_debug_app
from driverguard.schemas import FrameSignals


def check():
    torch.set_num_threads(4)
    model_id = "models/from_kaggle/week5_gru_dann/fold1_gru_seed42/inference/model.pt"
    predictions = REPO_ROOT / "models/recovered_week5/experiments/fold1_gru_seed42/predictions/validation_windows.csv"
    with predictions.open(encoding="utf-8", newline="") as stream:
        clip = next(csv.DictReader(stream))["clip_id"]
    with TestClient(create_debug_app()) as client:
        assert client.get("/").status_code == 200
        catalog = client.get("/dbg/models").json()["models"]
        assert len(catalog) == 30
        assert all(model["cross_validation"] for model in catalog)
        assert client.get("/dbg/clips").status_code == 200
        response = client.post("/dbg/replay", json={"model": model_id,
                               "feature_set": "week5_development", "clip": clip, "stride": 25})
        assert response.status_code == 200, response.text
        replay = response.json()
        assert replay["contract"]["ok"] is True, replay["contract"]
        assert replay["consistency"]["compared"] > 0
        assert replay["consistency"]["max_abs_diff"] <= 1e-5
    live = LiveModel(model_id, load_config())
    for index in range(1200):
        result = live.update(FrameSignals(ts_ms=index * 100, face_detected=True, face_quality=.95,
                                         ear_left=.3, ear_right=.3, mar=.1, yaw=0., pitch=0., roll=0.))
        if index < 1199:
            assert result["prediction"] is None
    assert result["prediction"] is not None
    first = result
    invalid = live.update(FrameSignals(ts_ms=119950))
    assert invalid["prediction"] is invalid["smoothed"] is None
    live.reset()
    assert live.update(FrameSignals(ts_ms=120000))["prediction"] is None
    overview = build_project_overview(REPO_ROOT)
    assert overview["target"]["candidate"] is None and not overview["target"]["achieved"]
    assert overview["development_cv"]["run_count"] == 30
    report = {"status": "passed", "ui_available": True, "catalog_models": len(catalog),
              "full_clip_replay": {key: replay[key] for key in ("model_id", "clip", "summary", "contract", "consistency")},
              "live": {"first_prediction_history_s": first["history_s"], "warmup_frames": 1200,
                       "invalid_frame_suppresses_score": True, "reset_clears_history": True,
                       "scope": "synthetic signals through real checkpoint; not webcam accuracy"},
              "cv_excluded_from_final_selection": True, "original_test_opened": False}
    out = Path("outputs/recovery_week5/runtime_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    check()
