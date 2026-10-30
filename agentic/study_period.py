"""Study period: the span of a night the models were trained on, from 30 min before the first sleep
epoch to 30 min after the last.

* `reference_span` applies that rule to an expert hypnogram (used only for calibration and evaluation).
* `estimate_span` estimates it WITHOUT a hypnogram, from per-epoch predicted stages (pass 1 over the
  whole recording): the first and last "sustained" sleep runs - at least `run` predicted non-W epochs,
  allowing W gaps of at most `gap` epochs inside a run - padded by `pad` epochs.

All indices are original-recording epochs; [start, end) half-open.
"""
from __future__ import annotations

import numpy as np

W = 0                                                     # stage index of Wake (order W, N1, N2, N3, REM)


def reference_span(labels: np.ndarray, pad: int = 60) -> tuple[int, int] | None:
    sleep = np.flatnonzero((labels != "W") & (labels != "INVALID"))
    if len(sleep) == 0:
        return None
    return max(0, int(sleep[0]) - pad), min(len(labels), int(sleep[-1]) + pad + 1)


def _sustained_runs(asleep: np.ndarray, run: int, gap: int) -> list[tuple[int, int]]:
    """[start, end) of sleep runs after bridging W gaps <= gap, keeping runs with >= run sleep epochs."""
    idx = np.flatnonzero(asleep)
    if len(idx) == 0:
        return []
    runs, s, prev, count = [], idx[0], idx[0], 1
    for i in idx[1:]:
        if i - prev - 1 <= gap:
            count += 1
        else:
            if count >= run:
                runs.append((int(s), int(prev) + 1))
            s, count = i, 1
        prev = i
    if count >= run:
        runs.append((int(s), int(prev) + 1))
    return runs


def sleep_bounds(pred: np.ndarray, run: int, gap: int, merge: int | None = None) -> tuple[int, int] | None:
    """Unpadded [start, end) of the sleep period.
    merge=None (v0.1): first to last sustained run - fails on long recordings with daytime false sleep.
    merge=k   (v0.2): the MAIN sleep episode - sustained runs separated by <= k wake epochs are merged,
                      and the cluster with the most sleep epochs is kept."""
    asleep = np.asarray(pred) != W
    runs = _sustained_runs(asleep, run, gap)
    if not runs:
        return None
    if merge is None:
        return runs[0][0], runs[-1][1]
    clusters = [[runs[0]]]
    for r in runs[1:]:
        if r[0] - clusters[-1][-1][1] <= merge:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    best = max(clusters, key=lambda c: int(asleep[c[0][0]:c[-1][1]].sum()))
    return best[0][0], best[-1][1]


def estimate_span(pred: np.ndarray, run: int, gap: int, pad: int = 60, merge: int | None = None) -> tuple[int, int] | None:
    """pred: (n,) predicted stage indices. None when no sustained sleep exists (-> R04 period_not_found)."""
    b = sleep_bounds(pred, run, gap, merge)
    return None if b is None else (max(0, b[0] - pad), min(len(pred), b[1] + pad))
