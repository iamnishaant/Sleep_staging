"""Raw-recording signal handling for Path B: read EDF, epoch, 34 training features, signal quality.

The 34 spectral features are copied from code/Phase2_48/spectral_preprocess.py (that script cannot be
imported: it creates directories on import). tools/p2_parity.py proves the copy reproduces the stored
training features exactly.
"""
from __future__ import annotations

import numpy as np

EPOCH_SEC = 30
FS = 100
SPE = EPOCH_SEC * FS                       # samples per epoch the models were trained on


# ---------------------------------------------------------------- reading
def read_channel(edf_path, channel: str) -> tuple[np.ndarray, float]:
    """One channel in volts and its sampling rate. Raises KeyError if the channel is absent."""
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    if channel not in raw.ch_names:
        raise KeyError(channel)
    return raw.get_data(picks=[channel])[0], float(raw.info["sfreq"])


def to_100hz(x: np.ndarray, sfreq: float) -> np.ndarray:
    if sfreq == FS:
        return x
    import mne
    return mne.filter.resample(x, up=FS, down=sfreq, verbose=False)


def epochs(x: np.ndarray) -> np.ndarray:
    """(n_samples,) -> (n_epochs, 3000), dropping a trailing partial epoch."""
    n = len(x) // SPE
    return x[: n * SPE].reshape(n, SPE)


# ---------------------------------------------------------------- 34 training features
def _dwt_features(e):
    import pywt
    out = []
    for c in pywt.wavedec(e, "db4", level=5):
        energy = np.sum(c ** 2)
        p = (c ** 2) / (np.sum(c ** 2) + 1e-12)
        out += [energy, np.log(energy + 1e-12), -np.sum(p * np.log(p + 1e-12)), np.var(c)]
    return out


def _stft_features(e):
    from scipy.signal import stft
    f, _t, z = stft(e, fs=FS, window="hann", nperseg=2 * FS, noverlap=FS)
    pm = np.mean(np.abs(z) ** 2, axis=1)
    total = np.sum(pm)
    band = lambda lo, hi: np.sum(pm[np.logical_and(f >= lo, f <= hi)])
    d, t, a, b, g = band(0.5, 4), band(4, 8), band(8, 13), band(13, 30), band(30, 45)
    ps = pm / (np.sum(pm) + 1e-12)
    entropy = -np.sum(ps * np.log(ps + 1e-12))
    cum = np.cumsum(pm)
    sef95 = f[np.where(cum >= 0.95 * cum[-1])[0][0]]
    ratios = [d / (b + 1e-12), t / (a + 1e-12), (d + t) / (a + b + 1e-12)]
    return [d / total, t / total, a / total, b / total, g / total, entropy, sef95] + ratios


def spectral_features(ep: np.ndarray) -> np.ndarray:
    """(n, 3000) volts -> (n, 34) float32, exactly as the training data were made."""
    return np.array([_dwt_features(e) + _stft_features(e) for e in ep], dtype=np.float32)


# ---------------------------------------------------------------- quality
def unusable_epochs(ep: np.ndarray, flat_std_v: float = 1e-6, stuck_run: int = 100) -> np.ndarray:
    """True where an epoch is flat (std < 1 uV) or stuck (>= 1 s of identical consecutive samples).
    The same rule as the advanced trainer's quality_mask."""
    flat = ep.std(axis=1) < flat_std_v
    same = np.diff(ep, axis=1) == 0
    # a run of `stuck_run` identical samples = `stuck_run - 1` consecutive zero differences
    k = stuck_run - 1
    csum = np.cumsum(np.pad(same, ((0, 0), (1, 0))).astype(np.int32), axis=1)
    stuck = ((csum[:, k:] - csum[:, :-k]) == k).any(axis=1)
    return flat | stuck


def line_noise_ratio(ep: np.ndarray) -> float:
    """Night median of per-epoch Welch power in 44-50 Hz over 0.5-30 Hz, at 100 Hz (V5).
    Resampling removes 60 Hz, so only noise at or near 50 Hz (the Nyquist frequency) can reach the model."""
    from scipy.signal import welch
    if len(ep) == 0:
        return 0.0
    f, p = welch(ep, fs=FS, nperseg=256, axis=-1)
    hi = p[:, (f >= 44) & (f <= 50)].sum(-1)
    lo = p[:, (f >= 0.5) & (f <= 30)].sum(-1)
    return float(np.median(hi / np.maximum(lo, 1e-20)))


def quality_summary(bad: np.ndarray) -> dict:
    """Per-night numbers the policy thresholds act on."""
    longest, run = 0, 0
    for b in bad:
        run = run + 1 if b else 0
        longest = max(longest, run)
    return {"epochs": int(len(bad)), "unusable_fraction": float(bad.mean()) if len(bad) else 1.0,
            "longest_unusable_run_min": longest * EPOCH_SEC / 60}
