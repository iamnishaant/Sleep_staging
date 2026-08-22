"""
Unit tests for preprocess_multichannel.

The headline test is test_equivalent_to_original: `epoch_bounds()` was
refactored out of `extract_epochs_and_trim()` in
code/Phase1_48/Phase1reworked/preprocess.py so that one sample slice can serve
several channels. If the refactor is not EXACTLY equivalent, channel 0 of the
regenerated tensors will not match the stored single-channel ones, splits.json
will describe different epochs, and every published number becomes
incomparable - silently, because nothing raises.

`--verify` on the preprocessing script checks the same property against real
tensors. This checks it against hypnogram shapes that may not occur in
Sleep-EDF but would break the logic if they did.

Run:  python distillation/test_preprocess_multichannel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_multichannel import (EPOCH_SEC, PRE_POST_WAKE_MIN,       # noqa: E402
                                     epoch_bounds)

SPE = 3000
STAGES = ["W", "N1", "N2", "N3", "REM", "INVALID"]

PASS: list[str] = []
FAIL: list[str] = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")


def original_extract(data_1ch, labels, samples_per_epoch):
    """Verbatim from code/Phase1_48/Phase1reworked/preprocess.py."""
    valid_mask = labels != "INVALID"
    labels = labels[valid_mask]
    sleep_idx = np.where(labels != "W")[0]
    if len(sleep_idx) == 0:
        return None, None
    pre_epochs = int((PRE_POST_WAKE_MIN * 60) / EPOCH_SEC)
    start_ep = max(0, sleep_idx[0] - pre_epochs)
    end_ep = min(len(labels), sleep_idx[-1] + pre_epochs + 1)
    labels = labels[start_ep:end_ep]
    data = data_1ch[start_ep * samples_per_epoch:end_ep * samples_per_epoch]
    return data.reshape(-1, samples_per_epoch), labels.tolist()


def new_extract(data_1ch, labels, spe):
    """What preprocess_multichannel does, reduced to one channel."""
    start_ep, end_ep, lab = epoch_bounds(labels)
    if start_ep is None:
        return None, None
    d = data_1ch[start_ep * spe:end_ep * spe]
    n_ep = len(d) // spe
    return d[:n_ep * spe].reshape(n_ep, spe), lab.tolist()


def hypnograms():
    """Awkward shapes first, then randomised ones."""
    yield "all wake", np.array(["W"] * 50, dtype=object)
    yield "all invalid", np.array(["INVALID"] * 20, dtype=object)
    yield "sleep at first epoch", np.array(["N2"] + ["W"] * 60, dtype=object)
    yield "sleep at last epoch", np.array(["W"] * 60 + ["N2"], dtype=object)
    yield "invalid at both ends", np.array(
        ["INVALID"] * 10 + ["W"] * 5 + ["N2"] * 5 + ["INVALID"] * 10, dtype=object)
    yield "single epoch", np.array(["N3"], dtype=object)
    yield "brief REM in long wake", np.array(
        ["W"] * 200 + ["REM"] * 3 + ["W"] * 200, dtype=object)
    yield "sleep shorter than the 30-min pad", np.array(
        ["W"] * 5 + ["N2"] * 2 + ["W"] * 5, dtype=object)

    rng = np.random.default_rng(0)
    for i in range(300):
        n = int(rng.integers(1, 400))
        yield (f"random {i}",
               rng.choice(STAGES, size=n, p=[.35, .08, .28, .07, .12, .10]).astype(object))


def main() -> int:
    print(__doc__.strip().splitlines()[1])
    print()

    rng = np.random.default_rng(1)
    n_ok = n_none = 0
    bad_labels = bad_shape = bad_data = bad_none = bad_align = 0

    for _, labels in hypnograms():
        data = rng.standard_normal(len(labels) * SPE)
        a = original_extract(data, labels.copy(), SPE)
        b = new_extract(data, labels.copy(), SPE)

        if a[0] is None or b[0] is None:
            if not (a[0] is None and b[0] is None):
                bad_none += 1
            else:
                n_none += 1
            continue
        if a[1] != b[1]:
            bad_labels += 1
            continue
        if a[0].shape != b[0].shape:
            bad_shape += 1
            continue
        if not np.array_equal(a[0], b[0]):
            bad_data += 1
            continue
        if a[0].shape[0] != len(a[1]):
            bad_align += 1
            continue
        n_ok += 1

    total = n_ok + n_none
    print(f"test_equivalent_to_original  ({total} hypnograms)")
    check("labels identical to the original extraction", bad_labels == 0)
    check("epoch counts identical", bad_shape == 0)
    check("sample data identical", bad_data == 0)
    check("agrees on which nights have no sleep", bad_none == 0)
    check("epoch count matches label count", bad_align == 0)
    print(f"        {n_ok} produced output, {n_none} correctly returned None")

    print("\ntest_multichannel_reshape")
    # (C, n_ep*spe) -> (n_ep, C, spe) must keep each channel's samples contiguous
    n_ep, C = 7, 3
    flat = np.stack([np.arange(n_ep * SPE) + c * 1_000_000 for c in range(C)])
    ep = flat.reshape(C, n_ep, SPE).transpose(1, 0, 2)
    check("shape is (n_epochs, n_channels, samples)", ep.shape == (n_ep, C, SPE))
    check("channel 0 epoch 0 is the first 3000 samples of channel 0",
          np.array_equal(ep[0, 0], flat[0, :SPE]))
    check("channel 2 epoch 3 is the right slice of channel 2",
          np.array_equal(ep[3, 2], flat[2, 3 * SPE:4 * SPE]))
    check("no channel bleed", all(
        (ep[:, c, :] // 1_000_000 == c).all() for c in range(C)))
    check("channel 0 extraction matches the single-channel layout",
          np.array_equal(ep[:, 0, :], flat[0].reshape(n_ep, SPE)))

    print("\n" + "-" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
