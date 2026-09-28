"""Generate golden test vectors: synthetic FrameSignals in -> expected engine decisions out.

    python scripts/make_golden.py

Why: the Android (Kotlin) port of the temporal + risk engine must reproduce the Python reference
decision-for-decision. Both implementations replay tests/golden/*.jsonl.gz and must agree on
risk_level / reason_codes / alert at every tick (floats within tolerance).

Regenerate ONLY when a decision rule or default threshold changes on purpose, and say so in the
commit message — a silent golden update hides a behaviour change.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driverguard.config import load_config  # noqa: E402
from driverguard.engine import DriverGuardEngine  # noqa: E402
from driverguard.schemas import FrameSignals  # noqa: E402
from driverguard.sim import generate  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden"
SCENARIOS = ("drowsy", "distracted", "degraded")
EPOCH, TZ = 1_789_959_600, 7
OUT_KEYS = ("risk_level", "reason_codes", "eye_closed", "closure_ms", "perclos_proxy", "risk_score", "calibrated")


def expected(tick: dict) -> dict:
    out = {k: tick[k] for k in OUT_KEYS}
    out["alert"] = tick["alert"]["level"] if tick["alert"] else None
    return out


def build(name: str) -> list[dict]:
    eng = DriverGuardEngine(load_config(), driver_id="GOLD", trip_id="golden", trip_start_epoch_s=EPOCH, tz_offset_h=TZ)
    rows = []
    for s in generate(name, fps=15.0, seed=0):
        row = s.to_row()  # rounded, JSON-safe input — expected output is computed FROM this exact input
        rows.append({"in": row, "out": expected(eng.step(FrameSignals.from_row(row)).to_dict())})
    return rows


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "config.json").write_text(json.dumps(
        {"epoch": EPOCH, "tz_offset_h": TZ, "config": load_config().model_dump(mode="json")}, indent=2))
    for name in SCENARIOS:
        rows = build(name)
        with gzip.open(GOLDEN / f"{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} ticks")


if __name__ == "__main__":
    main()
