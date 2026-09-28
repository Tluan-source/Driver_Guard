"""Subject-independent splits. NEVER split by frame (identity leakage makes scores optimistic).

    UTA-RLDD   official 5 folds (12 participants each)
    DMD        leave-one-subject-out / GroupKFold by subject (only 14 subjects -> report CIs)
    YawDD      subject-wise
    StateFarm  group by driver id (subject column in driver_imgs_list.csv)
    Internal   driver-wise 70/15/15 or GroupKFold
"""
from __future__ import annotations

import random
from collections.abc import Sequence


def group_kfold(subjects: Sequence[str], k: int, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    subs = sorted(set(subjects))
    if k > len(subs):
        raise ValueError(f"k={k} > number of subjects ({len(subs)})")
    rng = random.Random(seed)
    rng.shuffle(subs)
    folds = [subs[i::k] for i in range(k)]
    return [(sorted(s for j, f in enumerate(folds) if j != i for s in f), sorted(folds[i])) for i in range(k)]


def leave_one_subject_out(subjects: Sequence[str]) -> list[tuple[list[str], list[str]]]:
    subs = sorted(set(subjects))
    return [([s for s in subs if s != t], [t]) for t in subs]


def driver_split(subjects: Sequence[str], ratios=(0.70, 0.15, 0.15), seed: int = 0):
    subs = sorted(set(subjects))
    random.Random(seed).shuffle(subs)
    n = len(subs)
    a = int(round(ratios[0] * n))
    b = a + int(round(ratios[1] * n))
    return sorted(subs[:a]), sorted(subs[a:b]), sorted(subs[b:])


def assert_disjoint(*groups: Sequence[str]) -> None:
    seen: dict[str, int] = {}
    for gi, g in enumerate(groups):
        for s in g:
            if s in seen and seen[s] != gi:
                raise AssertionError(f"subject {s!r} appears in split {seen[s]} and {gi} — identity leakage")
            seen[s] = gi
