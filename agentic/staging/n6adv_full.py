"""Staging adapter for the Stage 3 advanced student (EEG + EOG, arm "full").

175,599-parameter model: N4kd backbone + EOG branch + time-frequency branch + local context, all
gated closed at init (ADDING EOG VERSION - FINAL SUMMARY/README.md). Checkpoint sha256
61e94bfc8aaf547bb0ca97e7ec8e48edefda1a594f6555912314a87961cd2cb0 (STAGE3_REPORT.md), the model chosen
for release v2 (ROADMAP R3).

The training module lives under a space-containing folder name, so it is loaded by path, read-only,
exactly as "ADDING EOG VERSION - PIPELINE INTEGRATION/make_prob_caches.py" already does.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import torch

from .. import REPO_ROOT

TRAINER_PATH = (REPO_ROOT / "ADDING EOG VERSION - ADVANCED STUDENT" / "kaggle_train_student_advanced.py")
WINDOW = 256


def _load_trainer():
    spec = importlib.util.spec_from_file_location("advtr_n6adv", TRAINER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class N6advFullAdapter:
    name = "n6adv_full"
    inputs = ("EEG Fpz-Cz", "EOG horizontal")

    def __init__(self, checkpoint):
        A = _load_trainer()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = A.build_model("full")
        self.model.load_state_dict(ck["eval_weights"])
        self.model.eval()
        std = ck["eog_feature_standardiser"]
        self.mu, self.sd = torch.tensor(std["mu"]), torch.tensor(std["sd"])
        self.lowpass, self.eog_features = A.lowpass, A.eog_epoch_features
        self.eeg_scale, self.eog_scale = A.EEG_SCALE, A.EOG_SCALE

    @torch.no_grad()
    def predict(self, signals: dict, spectral: np.ndarray) -> np.ndarray:
        n = min(len(signals["EEG Fpz-Cz"]), len(signals["EOG horizontal"]), len(spectral))
        eeg_v = np.asarray(signals["EEG Fpz-Cz"][:n], dtype=np.float64)
        eog_v = np.asarray(signals["EOG horizontal"][:n], dtype=np.float64)
        eeg_scaled = eeg_v * self.eeg_scale
        eog_scaled = (eog_v * self.eog_scale).reshape(-1)
        L = eeg_v.shape[1]
        eog_lp = np.ascontiguousarray(self.lowpass(eog_scaled)).reshape(n, L)  # sosfiltfilt can yield negative strides
        feat = (torch.as_tensor(self.eog_features(eog_lp, eeg_scaled), dtype=torch.float32) - self.mu) / self.sd
        feat = feat.clamp(-6, 6)
        xe = torch.as_tensor(eeg_scaled, dtype=torch.float32)
        xo = torch.as_tensor(eog_lp, dtype=torch.float32)
        xs = torch.as_tensor(spectral[:n], dtype=torch.float32)
        out = np.empty((n, 5), dtype=np.float32)
        for st in range(0, n, WINDOW):
            en = min(st + WINDOW, n)
            logits = self.model(xe[st:en].unsqueeze(0), xs[st:en].unsqueeze(0), None,
                                xo=xo[st:en].unsqueeze(0), xf=feat[st:en].unsqueeze(0))
            out[st:en] = torch.softmax(logits.float(), -1).squeeze(0).numpy()
        return out
