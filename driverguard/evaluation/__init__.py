"""Evaluation harness: event matching, subject-wise splits, interval labels, operating curves,
UTA-RLDD video-level protocol, runtime benchmark."""
from .events import Interval, MatchResult, alerts_from_ticks, false_alerts_per_hour, intervals_from_ticks, match_events
from .splits import assert_disjoint, driver_split, group_kfold, leave_one_subject_out

__all__ = [
    "Interval", "MatchResult", "alerts_from_ticks", "false_alerts_per_hour", "intervals_from_ticks", "match_events",
    "assert_disjoint", "driver_split", "group_kfold", "leave_one_subject_out",
]
