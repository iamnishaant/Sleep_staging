"""Staging adapter for N4kd (EEG Fpz-Cz + 34 spectral features).

Loads the checkpoint with the project's own loader (distillation/evaluate_student.load_student, imported
read-only) and predicts exactly as validation did: non-overlapping 256-epoch windows.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from .. import REPO_ROOT

WINDOW = 256


class N4kdAdapter:
    name = "n4kd_eeg"
    inputs = ("EEG Fpz-Cz",)

    def __init__(self, checkpoint):
        sys.path.insert(0, str(REPO_ROOT / "distillation"))
        from evaluate_student import load_student
        self.model, self.ck, self.scale = load_student(checkpoint)

    @torch.no_grad()
    def predict(self, signals: dict, spectral: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(signals["EEG Fpz-Cz"], dtype=torch.float32) * self.scale
        s = torch.as_tensor(spectral, dtype=torch.float32)
        n = min(len(t), len(s))
        out = np.empty((n, 5), dtype=np.float32)
        for st in range(0, n, WINDOW):
            en = min(st + WINDOW, n)
            out[st:en] = F.softmax(self.model(t[st:en].unsqueeze(0), s[st:en].unsqueeze(0)).float(), -1).squeeze(0).numpy()
        return out
