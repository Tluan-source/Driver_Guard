"""Regression: the engine must reproduce the golden vectors (same file the Android port uses)."""
import gzip
import json
from pathlib import Path

from driverguard.config import load_config
from driverguard.engine import DriverGuardEngine
from driverguard.schemas import FrameSignals

GOLDEN = Path(__file__).parent / "golden"


def _check(name: str) -> None:
    meta = json.loads((GOLDEN / "config.json").read_text())
    eng = DriverGuardEngine(load_config(), driver_id="GOLD", trip_id="golden",
                            trip_start_epoch_s=meta["epoch"], tz_offset_h=meta["tz_offset_h"])
    with gzip.open(GOLDEN / f"{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            t = eng.step(FrameSignals.from_row(row["in"])).to_dict()
            exp = row["out"]
            assert t["risk_level"] == exp["risk_level"], (name, i, t["risk_level"], exp)
            assert t["reason_codes"] == exp["reason_codes"], (name, i)
            assert (t["alert"]["level"] if t["alert"] else None) == exp["alert"], (name, i)
            assert t["closure_ms"] == exp["closure_ms"], (name, i)
            if exp["perclos_proxy"] is None:
                assert t["perclos_proxy"] is None
            else:
                assert abs(t["perclos_proxy"] - exp["perclos_proxy"]) < 1e-3, (name, i)


def test_golden_drowsy():
    _check("drowsy")


def test_golden_distracted():
    _check("distracted")


def test_golden_degraded():
    _check("degraded")
