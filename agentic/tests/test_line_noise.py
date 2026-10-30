"""V5 (policy 0.2): the quality step routes line noise near 50 Hz to W10 (warn) or Q03 (review only)."""
from types import SimpleNamespace

import numpy as np
import pytest

from agentic.core.atomic import write_json
from agentic.core.policy import Policy
from agentic.signals import FS, line_noise_ratio
from agentic.workflows import edf_report

EPOCHS = 40


def _eeg(noise_uv: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(EPOCHS * 30 * FS) / FS
    base = 20e-6 * np.sin(2 * np.pi * 2.0 * t) + 10e-6 * rng.standard_normal(t.size)   # slow wave + noise
    return (base + noise_uv * 1e-6 * np.sin(2 * np.pi * 49.0 * t)).reshape(EPOCHS, 30 * FS)


def _ctx(tmp_path, eeg):
    for d in ("study_period", "preprocess", "quality"):
        (tmp_path / d).mkdir(exist_ok=True)
    write_json(tmp_path / "study_period" / "span.json", {"span": [0, EPOCHS]})
    np.save(tmp_path / "preprocess" / "eeg.npy", eeg)
    return SimpleNamespace(earlier=lambda s: tmp_path / s, out=lambda n: tmp_path / "quality" / n,
                           policy=Policy.load())


@pytest.mark.parametrize("noise_uv,next_step,reasons", [
    (0.0, "stage_pass2", []),
    (6.0, "stage_pass2", ["W10"]),
    (40.0, "review_only", ["Q03"]),
])
def test_line_noise_routing(tmp_path, noise_uv, next_step, reasons):
    eeg = _eeg(noise_uv)
    d = edf_report.quality(_ctx(tmp_path, eeg))
    assert (d.next, d.reasons) == (next_step, reasons), line_noise_ratio(eeg)


def test_ratio_is_zero_for_empty_input():
    assert line_noise_ratio(np.zeros((0, 30 * FS))) == 0.0
