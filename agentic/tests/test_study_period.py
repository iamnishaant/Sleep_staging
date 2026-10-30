import numpy as np

from agentic.study_period import estimate_span, reference_span


def test_reference_span_pads_and_clips():
    lab = np.array(["W"] * 100 + ["N2"] * 10 + ["W"] * 100 + ["INVALID"] * 5, dtype=object)
    assert reference_span(lab) == (40, 170)
    assert reference_span(np.array(["W", "INVALID"], dtype=object)) is None


def test_estimate_ignores_short_runs_and_bridges_gaps():
    p = np.zeros(400, dtype=int)
    p[50:53] = 2                       # 3-epoch blip: not sustained
    p[100:110] = 2; p[111:120] = 2     # one W gap of 1 epoch inside a sustained run
    assert estimate_span(p, run=10, gap=2) == (40, 180)
    assert estimate_span(p, run=10, gap=0) == (40, 170)     # no bridging: 100:110 counts, 111:120 (9) does not
    assert estimate_span(np.zeros(50, dtype=int), run=10, gap=2) is None


def test_main_episode_ignores_daytime_false_sleep():
    p = np.zeros(2000, dtype=int)
    p[100:130] = 2                     # daytime false "sleep", far from the night
    p[1000:1800] = 2                   # the night
    assert estimate_span(p, run=10, gap=2, merge=None) == (40, 1860)        # v0.1 stretches to the daytime run
    assert estimate_span(p, run=10, gap=2, merge=60) == (940, 1860)         # v0.2 keeps the main episode
