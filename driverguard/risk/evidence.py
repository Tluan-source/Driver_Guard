"""Evidence -> candidate levels + risk score.

Two kinds of candidates:
  * immediate: the rule already contains its own duration (e.g. eyes closed >= 2 s, looking
    down >= 2.5 s, phone persistent >= 1.5 s) -> may raise the level on this tick;
  * persistent: window statistics / score -> must hold for risk.enter_persist_ms first.

The weighted score is kept for the dashboard, the operating-point curves and for the
"learned fusion weights" experiment (Hướng 4). A learned model may replace `score_weights`,
but the final decision stays in the deterministic state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DriverGuardConfig
from ..schemas import Reason, RiskLevel
from ..temporal.trackers import EyeState, HeadState, PhoneState, YawnState
from .context import ContextState


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class Candidate:
    level: RiskLevel
    reason: Reason
    immediate: bool


@dataclass
class Evidence:
    components: dict[str, float]
    score: float
    candidates: list[Candidate] = field(default_factory=list)
    info: list[Reason] = field(default_factory=list)


def assess(cfg: DriverGuardConfig, eye: EyeState, yawn: YawnState, head: HeadState, phone: PhoneState,
           ctx: ContextState, trend_rising: bool) -> Evidence:
    r, e = cfg.risk, cfg.eye
    s = ctx.sensitivity
    parked = ctx.parked is True

    comp = {
        "eye": _clamp01(eye.closure_ms / e.microsleep_ms),
        "perclos": _clamp01(((eye.perclos_proxy or 0.0) - r.perclos_normal) / (r.perclos_warning - r.perclos_normal)),
        "yawn": _clamp01(yawn.yawns_in_window / (r.frequent_yawn_count + 1)),
        "head": max(_clamp01(head.off_road_ms / cfg.head.distraction_ms), 0.5 if head.nods_in_window else 0.0),
        "phone": _clamp01(phone.persistence),
    }
    if parked:  # driver may legitimately rest / use the phone while parked
        comp["eye"] = comp["head"] = comp["phone"] = 0.0
    wsum = sum(r.weights.values()) or 1.0
    score = 100.0 * sum(r.weights[k] * v for k, v in comp.items()) / wsum

    c: list[Candidate] = []
    info: list[Reason] = []
    if parked:
        info.append(Reason.PARKED)
    if ctx.circadian_low:
        info.append(Reason.CIRCADIAN_LOW)

    if not parked:
        # ---- drowsiness: eye closure (strong evidence)
        if eye.closure_ms >= e.microsleep_ms * s:
            c.append(Candidate(RiskLevel.CRITICAL, Reason.MICROSLEEP, True))
        elif eye.closure_ms >= e.long_closure_ms * s:
            c.append(Candidate(RiskLevel.WARNING, Reason.PROLONGED_EYE_CLOSURE, True))
        # ---- distraction
        if head.distracted and head.reason is not None:
            c.append(Candidate(RiskLevel.WARNING, head.reason, True))
        if phone.active:
            c.append(Candidate(RiskLevel.WARNING, Reason.PHONE_USE, True))
        # ---- nod combined with elevated closure statistics
        if head.nod and (eye.perclos_proxy or 0) >= r.perclos_caution * s:
            c.append(Candidate(RiskLevel.WARNING, Reason.HEAD_NOD, True))

    # ---- window statistics (persistent)
    if eye.perclos_proxy is not None:
        if eye.perclos_proxy >= r.perclos_warning * s:
            c.append(Candidate(RiskLevel.WARNING, Reason.HIGH_PERCLOS, False))
        elif eye.perclos_proxy >= r.perclos_caution * s:
            c.append(Candidate(RiskLevel.CAUTION, Reason.ELEVATED_PERCLOS, False))
    if yawn.yawns_in_window >= r.frequent_yawn_count:
        c.append(Candidate(RiskLevel.CAUTION, Reason.FREQUENT_YAWN, False))
    if head.nods_in_window >= 2:
        c.append(Candidate(RiskLevel.CAUTION, Reason.HEAD_NOD, False))
    if ctx.long_drive:
        c.append(Candidate(RiskLevel.CAUTION, Reason.LONG_DRIVE, False))
    if trend_rising:
        c.append(Candidate(RiskLevel.CAUTION, Reason.FATIGUE_TREND_RISING, False))
    if score >= r.warning_enter * s:
        c.append(Candidate(RiskLevel.WARNING, Reason.RISK_SCORE, False))
    elif score >= r.caution_enter * s:
        c.append(Candidate(RiskLevel.CAUTION, Reason.RISK_SCORE, False))

    return Evidence(comp, score, c, info)
