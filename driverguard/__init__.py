"""DriverGuard — on-device driver monitoring.

DriverGuard = Reliable Signals + Temporal Evidence + Deterministic Safety Logic.

Layers (each one testable on its own):
    capture     -> frames + wall-clock timestamps
    perception  -> FrameSignals (per-frame numbers, no temporal logic)
    temporal    -> eye/yawn/head/phone trackers over time windows
    calibration -> per-driver adaptive baselines
    risk        -> deterministic state machine + reason codes
    trip        -> whole-trip fatigue trend memory
    api         -> metadata-only FastAPI / WebSocket
"""

__version__ = "0.1.0"
