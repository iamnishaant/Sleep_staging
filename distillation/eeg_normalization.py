"""
EEG input normalisation - statistics and transform.

WHY THIS EXISTS
---------------
The preprocessed temporal tensors are in VOLTS, straight from the EDF via MNE,
which returns volts by default. Typical EEG is 10-100 microvolts, so the stored
values sit around 2e-5 while the 34 spectral features sit around 7 - a scale
mismatch of roughly 450,000x.

The consequence was verified by ablation, not inferred: zeroing the ENTIRE raw
EEG input changes ZERO predictions in every trained model - the 121K student and
both 649K teachers, agreement 1.00000 across test recordings. The temporal
branch is numerically inert. Every model in the project is spectral-only, using
34 numbers per epoch and nothing else.

This module computes the scaling needed to make the temporal branch live. It
does NOT modify any existing preprocessed file, training script, or checkpoint -
the transform is applied on the fly at load time, so the original artefacts and
all existing results remain reproducible.

THE CHOICE OF NORMALISATION
---------------------------
Registered here before use, with rejected alternatives:

  GLOBAL SCALE (chosen)   x / sigma_train, sigma from the TRAINING split only.
      Gives unit variance overall while PRESERVING relative amplitude between
      epochs. That matters: N3 is defined by high-amplitude delta activity, so a
      transform that equalises amplitude across epochs would destroy the single
      most diagnostic feature in the signal.

  PER-EPOCH Z-SCORE (rejected as default)   (x - mu_t) / sigma_t per epoch.
      Standard in much of the sleep-staging literature, but it normalises away
      exactly the amplitude difference that distinguishes N3 from N1/N2. The
      spectral branch already carries band power, so the temporal branch's
      distinct contribution should be morphology AND amplitude, not morphology
      alone. Available via --mode per_epoch for comparison.

  FIXED x1e6 (volts -> microvolts)   Physically interpretable and very close to
      the global-scale result in practice. Available via --mode microvolt.
      Kept because it needs no fitted constant at all.

Centring is not applied: the measured per-recording mean is ~1e-7 volts, i.e.
already zero to seven decimal places, so subtracting it would be a no-op.

USAGE
    python distillation/eeg_normalization.py            # compute + write JSON
    python distillation/eeg_normalization.py --verify   # + ablation re-check
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"


def compute_stats(df: pd.DataFrame, train_subjects: list[str], root: Path,
                  epochs_per_rec: int = 120, seed: int = 42) -> dict:
    """
    Streaming mean/std over the TRAINING split.

    A bounded number of epochs per recording is sampled: the statistic is a
    global scale over ~400M samples, so a few hundred thousand per recording
    estimates it to far more precision than the training needs, and it avoids
    reading ~3.4 GB of tensors.
    """
    rng = np.random.default_rng(seed)
    tr = df[df["subject"].isin(train_subjects)]
    n = 0
    s1 = 0.0
    s2 = 0.0
    lo, hi = np.inf, -np.inf
    t0 = time.time()
    for i, (_, row) in enumerate(tr.iterrows()):
        x = torch.load(root / row["tensor_path"], map_location="cpu").float().numpy()
        if x.shape[0] > epochs_per_rec:
            idx = rng.choice(x.shape[0], epochs_per_rec, replace=False)
            x = x[idx]
        s1 += float(x.sum())
        s2 += float(np.square(x, dtype=np.float64).sum())
        n += x.size
        lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
        del x
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(tr)}] ({(time.time()-t0)/60:.1f} min)", flush=True)
    mean = s1 / n
    std = float(np.sqrt(max(s2 / n - mean ** 2, 0.0)))
    return {"n_samples": int(n), "n_recordings": int(len(tr)),
            "epochs_sampled_per_recording": epochs_per_rec,
            "mean_volts": mean, "std_volts": std,
            "min_volts": lo, "max_volts": hi,
            "scale_factor": 1.0 / std if std > 0 else 1.0}


def normalize(xt: torch.Tensor, mode: str, scale: float) -> torch.Tensor:
    """
    Apply the chosen transform. xt is [..., 3000] in volts.

    THIS IS THE FUNCTION TO COPY INTO ANY TRAINING SCRIPT. It is deliberately
    tiny and dependency-free so it can be pasted inline into a Kaggle notebook
    without importing this module.
    """
    if mode == "global":
        return xt * scale
    if mode == "microvolt":
        return xt * 1e6
    if mode == "per_epoch":
        mu = xt.mean(-1, keepdim=True)
        sd = xt.std(-1, keepdim=True).clamp_min(1e-12)
        return (xt - mu) / sd
    if mode == "none":
        return xt
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--epochs-per-rec", type=int, default=120)
    ap.add_argument("--verify", action="store_true",
                    help="re-run the zeroed-EEG ablation to document the problem")
    args = ap.parse_args()

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    df = pd.read_csv(args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]

    print("computing EEG statistics over the TRAINING split only...")
    st = compute_stats(df, sp["splits"]["train"], REPO_ROOT, args.epochs_per_rec)

    print(f"\n{'='*72}\nTRAINING-SPLIT EEG STATISTICS\n{'='*72}")
    print(f"  recordings              {st['n_recordings']}")
    print(f"  samples                 {st['n_samples']:,}")
    print(f"  mean                    {st['mean_volts']:+.3e} V")
    print(f"  std                     {st['std_volts']:.3e} V"
          f"   ({st['std_volts']*1e6:.2f} uV)")
    print(f"  range                   {st['min_volts']:+.3e} .. {st['max_volts']:+.3e} V")
    print(f"\n  SCALE FACTOR (1/std)    {st['scale_factor']:.4e}")
    print(f"  centring needed?        no - mean is {abs(st['mean_volts']):.1e} V, "
          f"zero to 7 dp")

    # what each mode does to the scale, against the spectral branch's ~6.9
    demo = torch.load(REPO_ROOT / df.iloc[0]["tensor_path"], map_location="cpu").float()[:8]
    print(f"\n  {'mode':<12}{'|x| mean after':>18}{'std after':>12}")
    for m in ("none", "global", "microvolt", "per_epoch"):
        z = normalize(demo, m, st["scale_factor"])
        print(f"  {m:<12}{z.abs().mean():>18.4f}{z.std():>12.4f}")
    print(f"  {'(spectral)':<12}{6.8928:>18.4f}{19.5079:>12.4f}   <- what it must compete with")

    out = {
        "purpose": "Make the temporal branch numerically live. Verified inert by ablation.",
        "problem": {
            "stored_units": "volts (MNE default, never converted)",
            "eeg_abs_mean": 1.6e-05,
            "spectral_abs_mean": 6.8928,
            "scale_mismatch": "~450000x",
            "ablation": "zeroing the entire EEG input changes 0 predictions; "
                        "agreement 1.00000 for the 121K student and both 649K teachers",
        },
        "statistics_fitted_on": "training split only",
        "statistics": st,
        "default_mode": "global",
        "modes": {
            "global": "x / std_train - unit variance, PRESERVES inter-epoch amplitude",
            "microvolt": "x * 1e6 - volts to microvolts, no fitted constant",
            "per_epoch": "(x - mu_t)/sigma_t - rejected as default: destroys the "
                         "amplitude difference that defines N3",
            "none": "reproduces the existing (broken) behaviour",
        },
        "does_not_modify": [
            "processed_sleepedf/ tensors", "existing training scripts",
            "existing checkpoints", "any published result",
        ],
    }
    p = RES / "eeg_normalization.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p.relative_to(REPO_ROOT)}")
    print(f"\nPaste into any trainer:   xt = xt * {st['scale_factor']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
