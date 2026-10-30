"""
A corrected version of the label/signal trimming used by the Sleep-EDF
preprocessing - for any FUTURE dataset build. The originals are left untouched:

    code/Phase1_48/Phase1reworked/preprocess.py :: extract_epochs_and_trim
    distillation/preprocess_multichannel.py     :: epoch_bounds

THE BUG IN THE ORIGINAL
    labels = labels[labels != "INVALID"]        # unscored epochs removed ...
    start_ep, end_ep = <on the shortened list>  # ... trim points on the SHORTENED list
    data = raw[start_ep*spe : end_ep*spe]       # ... applied to the FULL signal
Each removed epoch pushes the labels one epoch ahead of the signal.

THE FIX
    Never delete epochs. Trim on the full label array (so 30 minutes means 30
    real minutes), cut the signal with the same indices, and keep unscored
    epochs in place as "?" so training and scoring can ignore them. Label i
    then always belongs to signal epoch i, by construction.

WHEN TO USE IT
    Only if the datasets are ever rebuilt from the raw EDF files. For the
    existing tensors, results/corrected_labels_*.json already give the correct
    label of every stored epoch, so nothing needs to be regenerated to train or
    evaluate correctly.
"""

import numpy as np

EPOCH_SEC = 30
PRE_POST_WAKE_MIN = 30
UNSCORED = "?"


def epoch_bounds_fixed(labels):
    """
    labels: full-night array of stage strings, "INVALID" where never scored.
    Returns (start_ep, end_ep, trimmed_labels) where start/end index the FULL
    array - the same indices that cut the signal - and unscored epochs inside
    the window are kept as "?". Returns (None, None, None) with no sleep.
    """
    labels = np.asarray(labels, dtype=object)
    sleep = np.flatnonzero((labels != "W") & (labels != "INVALID"))
    if len(sleep) == 0:
        return None, None, None
    pre = int(PRE_POST_WAKE_MIN * 60 / EPOCH_SEC)
    start = max(0, int(sleep[0]) - pre)
    end = min(len(labels), int(sleep[-1]) + pre + 1)
    window = labels[start:end].copy()
    window[window == "INVALID"] = UNSCORED
    return start, end, window


def extract_epochs_and_trim_fixed(signal, labels, samples_per_epoch):
    """signal: (n_samples,) or (C, n_samples). Returns (epochs, labels) aligned 1:1."""
    start, end, window = epoch_bounds_fixed(labels)
    if start is None:
        return None, None
    data = np.asarray(signal)[..., start * samples_per_epoch:end * samples_per_epoch]
    n = data.shape[-1] // samples_per_epoch
    epochs = data[..., :n * samples_per_epoch].reshape(*data.shape[:-1], n, samples_per_epoch)
    if epochs.ndim == 3:                                     # (C, n, spe) -> (n, C, spe)
        epochs = epochs.transpose(1, 0, 2)
    return epochs, list(window[:n])


def epoch_bounds_original(labels):
    """The ORIGINAL (buggy) logic, reproduced verbatim for the tests only."""
    labels = np.asarray(labels, dtype=object)
    labels = labels[labels != "INVALID"]
    sleep_idx = np.where(labels != "W")[0]
    if len(sleep_idx) == 0:
        return None, None, None
    pre = int(PRE_POST_WAKE_MIN * 60 / EPOCH_SEC)
    start = max(0, sleep_idx[0] - pre)
    end = min(len(labels), sleep_idx[-1] + pre + 1)
    return start, end, labels[start:end]
