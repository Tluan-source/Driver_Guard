"""Regression checks for the recovered subject-CV and long-context runtime."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

from driverguard.api import AuthService, StateHub, User
from driverguard.api.app import AppContext, create_app
from driverguard.api.project_status import build_project_overview
from driverguard.config import load_config
from driverguard.debug import model_lab
from driverguard.debug.server import create_debug_app
from driverguard.learning.camera.dann.data import quality_gate
from driverguard.learning.camera.dann.model import gradient_reverse
from driverguard.learning.camera.features import CameraFrameBuffer
from driverguard.schemas import FrameSignals


def _write_run(root, name, accuracy=.99, f1=.99, cv=False, independent=True):
    run = root / "models/from_kaggle/week5_gru_dann" / name
    (run / "metrics").mkdir(parents=True)
    (run / "inference").mkdir()
    (run / "inference/model.pt").write_bytes(b"not deserialized by inventory")
    metrics = {"accuracy": accuracy, "macro_f1": f1, "coverage": .98}
    (run / "metrics/summary.json").write_text(json.dumps({
        "architecture": "camera_gru_ablation", "metrics": {"validation": metrics, "test": metrics}}))
    (run / "metrics/protocol.json").write_text(json.dumps({
        "independent_test": independent, "frozen_before_test": independent,
        "config": {"context_seconds": 120, "label_mode": "alert_drowsy"}}))
    if cv:
        (run / "week5_provenance.json").write_text(json.dumps({"outer_fold": 1}))
    return run


def test_cv_folds_never_win_final_selection_or_confirm_target(tmp_path):
    _write_run(tmp_path, "fold1_gru_seed42", cv=True)
    final = _write_run(tmp_path, "development_candidate", accuracy=.75, f1=.74)
    overview = build_project_overview(tmp_path)
    assert overview["target"]["candidate"] == final.relative_to(tmp_path).as_posix()
    assert not overview["target"]["achieved"]
    cv = next(run for run in overview["runs"] if run["cross_validation"])
    assert not cv["eligible_for_selection"] and not cv["independent_test"]
    assert overview["development_cv"]["run_count"] == 1


@pytest.mark.parametrize("accuracy,f1,independent,passed", [
    (.80, .99, True, False), (.85, .90, True, False),
    (.86, .82, False, False), (.86, .82, True, True)])
def test_acceptance_requires_strict_accuracy_and_independent_protocol(tmp_path, accuracy, f1, independent, passed):
    _write_run(tmp_path, "candidate", accuracy, f1, independent=independent)
    target = build_project_overview(tmp_path)["target"]
    assert target["metric"] == "window_accuracy"
    assert target["achieved"] is passed


def test_project_inventory_is_manager_only(tmp_path):
    auth = AuthService([User("d", "driver", "D001", password="d"),
                        User("m", "fleet_manager", password="m")], secret=b"x" * 32)
    ctx = AppContext(cfg=load_config(), hub=StateHub(), auth=auth, project_root=tmp_path)
    with TestClient(create_app(ctx)) as client:
        assert client.get("/project/overview").status_code == 401
        for user, expected in (("d", 403), ("m", 200)):
            token = client.post("/auth/login", json={"username": user, "password": user}).json()["token"]
            assert client.get("/project/overview", headers={"Authorization": f"Bearer {token}"}).status_code == expected


def test_gradient_reversal_changes_only_backward_direction():
    x = torch.tensor([2., -3.], requires_grad=True)
    reversed_x = gradient_reverse(x, .25)
    assert torch.equal(reversed_x, x)
    (reversed_x ** 2).sum().backward()
    assert torch.allclose(x.grad, -.25 * 2 * x.detach())


def _live(probabilities=(.499999, .500001)):
    cfg = load_config()
    live = model_lab.LiveModel.__new__(model_lab.LiveModel)
    live.cfg = cfg
    live.model = SimpleNamespace(context=1200, id="fake", family="gru_dann",
                                 threshold=None, class_names=("alert", "drowsy"),
                                 predict_batch=lambda *args: [(probabilities, None, 1.)])
    live.buffer = CameraFrameBuffer(cfg, 1200)
    live.every, live.warnings = 25, []
    live.reset()
    return live


def _signal(ts, **changes):
    values = {"ts_ms": ts, "face_detected": True, "face_quality": .95,
              "ear_left": .3, "ear_right": .3, "mar": .1, "yaw": 0., "pitch": 0., "roll": 0.}
    values.update(changes)
    return FrameSignals(**values)


def _warm(live):
    for index in range(1200):
        output = live.update(_signal(index * 100))
        if index < 1199:
            assert output["probabilities"] is None
    assert output["prediction"] == 1
    return output


def test_live_waits_120_seconds_and_preserves_argmax_before_rounding():
    live = _live()
    assert _warm(live)["history_s"] == 120
    live.reset()
    output = live.update(_signal(120000))
    assert output["history_s"] == .1 and output["prediction"] is None
    assert output["smoothed"] is None


def test_invalid_frame_between_sample_bins_immediately_suppresses_old_score():
    live = _live()
    _warm(live)
    output = live.update(_signal(119950, face_detected=False))
    assert output["reason"] == "current_frame_invalid"
    assert output["probabilities"] is output["smoothed"] is None
    assert output["prediction"] is None


@pytest.mark.parametrize("timestamp,reason", [(120500, "insufficient_history"), (119800, "invalid_timestamps")])
def test_gap_or_timestamp_restart_clears_probability_history(timestamp, reason):
    live = _live()
    _warm(live)
    output = live.update(_signal(timestamp))
    assert output["reason"] == reason
    assert output["history_s"] < 1
    assert output["prediction"] is output["smoothed"] is None


def test_quality_gate_rejects_invalid_current_frame_even_with_high_coverage():
    valid = np.ones(1200, dtype=bool)
    valid[-1] = False
    x = np.zeros((1200, 19), dtype=np.float32)
    x[:, -1] = valid
    result = quality_gate(x, np.arange(1200) * 100, valid, 1200)
    assert result.reason == "current_frame_invalid" and result.valid_fraction > .99


def test_console_hides_and_refuses_original_test_clips(tmp_path, monkeypatch):
    features = tmp_path / "features"
    folder = features / "fake"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"clips": [
        {"clip_id": "visible", "split": "validation", "label": 0},
        {"clip_id": "locked", "split": "test", "label": 0}]}))
    monkeypatch.setattr(model_lab, "FEATURE_ROOT", features)
    monkeypatch.setattr(model_lab, "load_model", lambda _: object())
    listed = model_lab.list_clips()["feature_sets"][0]
    assert [c["clip_id"] for c in listed["clips"]] == ["visible"]
    with pytest.raises(ValueError, match="test"):
        model_lab.replay("unused", "fake", "locked")
    with TestClient(create_debug_app()) as client:
        assert client.get("/").status_code == 200
        assert client.get("/dbg/health").json()["ui_available"]
        result = client.post("/dbg/replay", json={"model": "unused", "feature_set": "fake", "clip": "locked"})
        assert result.status_code == 400 and "test" in result.json()["error"]
