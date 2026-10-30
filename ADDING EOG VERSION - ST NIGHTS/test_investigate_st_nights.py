"""
    cd "ADDING EOG VERSION - ST NIGHTS"
    python -m pytest -q -p no:cacheprovider test_investigate_st_nights.py
"""
import sys
sys.dont_write_bytecode = True
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import investigate_st_nights as S                                  # noqa: E402

REC = "SC4001E0-PSG"
HAVE = (S.MC / "tensors" / f"{REC}.pt").exists() and (S.SPEC / f"{REC}_spectral.pt").exists()


def test_stuck_flags():
    x = np.random.default_rng(0).normal(0, 2e-5, (4, 3000))
    x[1] = 3e-7 * np.random.default_rng(1).normal(size=3000)        # flat
    x[2, 500:600] = x[2, 500]                                       # 1.00 s stuck
    x[3, 500:598] = x[3, 500]                                       # 0.98 s - not stuck
    assert S.stuck_flags(x).tolist() == [False, True, True, False]


def test_boundary_free_gain_one_is_identity():
    Sc = np.random.default_rng(0).random((5, 34)).astype(np.float32)
    assert S.spectral_for_gain(Sc, np.zeros((5, 3000)), 1.0) is Sc


@pytest.mark.skipif(not HAVE, reason="Sleep-EDF tensors not present")
def test_stft_block_reproduces_the_cache():
    x = torch.load(S.MC / "tensors" / f"{REC}.pt").numpy()[:200, 0].astype(np.float64)
    Sc = torch.load(S.SPEC / f"{REC}_spectral.pt").numpy()[:200]
    assert np.allclose(S.stft_block(x), Sc[:, 24:], rtol=1e-3, atol=1e-6)


@pytest.mark.skipif(not HAVE, reason="Sleep-EDF tensors not present")
def test_gain_transform_dwt_algebra():
    x = torch.load(S.MC / "tensors" / f"{REC}.pt").numpy()[:50, 0].astype(np.float64)
    Sc = torch.load(S.SPEC / f"{REC}_spectral.pt").numpy()[:50]
    g = 0.7
    out = S.spectral_for_gain(Sc, x, g).astype(np.float64)
    assert np.allclose(out[:, 0:24:4], Sc[:, 0:24:4] * g * g, rtol=1e-5)
    assert np.allclose(out[:, 3:24:4], Sc[:, 3:24:4] * g * g, rtol=1e-5)
    assert np.allclose(out[:, 1:24:4], np.log(out[:, 0:24:4] + 1e-12), atol=1e-4)
    assert np.allclose(out[:, 24:], S.stft_block(x * g), rtol=1e-4)


# ------------------------------------------------- the preprocessing fix ---
import fixed_preprocessing as FP                                   # noqa: E402

RAW = Path("F:/sleep edf signals/physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry")


def _night():
    # 70 unscored lead-in epochs, wake, sleep with 3 "Movement time" epochs inside, wake
    lab = ["INVALID"] * 70 + ["W"] * 80 + ["N1", "N2", "N2"] * 40 + ["INVALID"] * 3 + ["REM"] * 30 + ["W"] * 90
    sig = np.repeat(np.arange(len(lab), dtype=float), 30)          # each signal epoch carries its own index
    return np.array(lab, dtype=object), sig


def test_original_logic_misaligns():
    lab, sig = _night()
    s, e, stored = FP.epoch_bounds_original(lab)
    signal_epoch_index = sig[s * 30:e * 30].reshape(-1, 30)[:, 0].astype(int)
    true = lab[signal_epoch_index]
    assert (true != stored).sum() > 0                              # labels do not match their signal


def test_fixed_logic_is_aligned_by_construction():
    lab, sig = _night()
    epochs, labels = FP.extract_epochs_and_trim_fixed(sig, lab, 30)
    idx = epochs[:, 0].astype(int)
    assert len(labels) == len(epochs)
    assert [("?" if t == "INVALID" else t) for t in lab[idx]] == labels
    assert labels.count("?") == 3                                  # the mid-night unscored epochs, kept and marked


def test_fixed_trims_on_real_time():
    lab, _ = _night()
    s, e, _ = FP.epoch_bounds_fixed(lab)
    first_sleep = 70 + 80
    assert s == first_sleep - 60                                   # 30 real minutes before sleep onset


@pytest.mark.skipif(not RAW.exists(), reason="raw Sleep-EDF files not present")
def test_fixed_logic_on_a_real_misaligned_recording():
    import mne
    mne.set_log_level("ERROR")
    import verify_hypnogram_alignment as V
    raw = mne.io.read_raw_edf(RAW / "ST7191J0-PSG.edf", preload=False)
    lab = V.full_labels(raw.n_times, raw.info["sfreq"], mne.read_annotations(RAW / "ST7191JR-Hypnogram.edf"))
    s, e, stored = FP.epoch_bounds_original(lab)
    assert (lab[s:e] != stored).mean() > 0.5                       # the original: most labels wrong on this night
    s2, e2, window = FP.epoch_bounds_fixed(lab)
    assert all(w == ("?" if t == "INVALID" else t) for w, t in zip(window, lab[s2:e2]))
