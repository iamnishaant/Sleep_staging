"""Where the student model plugs in (raw-EDF path, Phase 2).

A release names a staging adapter (`staging.adapter` in config/releases/*.yaml). An adapter turns a
preprocessed recording into per-epoch probabilities; everything downstream (tables -> evidence packet
-> report) is driven by the release's files, not by agent code. Swapping the student model therefore
means: write an adapter for its inputs (if they differ) and a new release file. The workflows, runner,
recovery, report path and V-HTML do not change.

"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class StagingAdapter(Protocol):
    name: str
    inputs: tuple[str, ...]          # channel names this adapter needs, e.g. ("EEG Fpz-Cz", "EOG horizontal")

    def predict(self, signals: dict[str, np.ndarray], spectral: np.ndarray) -> np.ndarray:
        """Per-epoch class probabilities, shape (n_epochs, 5), order W, N1, N2, N3, REM."""
        ...


from .n4kd import N4kdAdapter  # noqa: E402
from .n6adv_full import N6advFullAdapter  # noqa: E402

ADAPTERS: dict[str, type] = {"n4kd_eeg": N4kdAdapter, "n6adv_full": N6advFullAdapter}
