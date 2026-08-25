"""
Regenerate the Sleep-EDF tensors with EOG alongside the EEG.

WHY
---
Everything in this project is staged from ONE EEG derivation (Fpz-Cz), and in
the shipped model even that reaches the network only as 34 precomputed band
powers. AASM scoring does not work that way. Two of the five stages are defined
partly by the eyes:

    REM   rapid eye movements are definitional
    N1    slow rolling eye movements are the marker

and REM and N1 are exactly where this model is weakest - N1 F1 0.39, and
`separability_verdict.json` established that N1 is REPRESENTATION-bound, not
threshold-bound: no decision rule on the current probabilities improves N1 F1
by more than +0.0086. Representation-bound means only new information helps.

The 34 spectral features cannot be that new information - they are computed
FROM the same EEG, so they are a lossy summary of data the model already has.
`eeg_normalization_ablation.json` says this itself: "A genuinely independent
second modality (EOG) would test the fusion hypothesis properly; EEG-derived
spectral features cannot, since they share a source."

EOG horizontal is present in every Sleep-EDF recording, at 100 Hz, already
downloaded alongside the EEG, and has never been used.

WHAT THIS PRODUCES
------------------
    processed_sleepedf_mc/tensors/<rec>.pt     (n_epochs, n_channels, 3000)
    processed_sleepedf_mc/index.csv            same schema plus `channels`
    distillation/results/multichannel_norm.json  per-channel scale, TRAIN only

THE CHECK THAT MATTERS
----------------------
`--verify` re-derives channel 0 and compares it against the EXISTING
single-channel tensors, element by element. If they do not match, the epoching
or trimming has drifted and every comparison against a stored result is
invalid - which is precisely the class of silent bug this pipeline has already
hit twice. Run it before training on anything this produces.

The epoching, labelling and trimming below are copied unchanged from
code/Phase1_48/Phase1reworked/preprocess.py for that reason. Do not "improve"
them here; a change makes the new tensors incomparable to every published
number.

A NOTE ON SAMPLING RATES
------------------------
Sleep-EDF PSG files mix rates: EEG Fpz-Cz, EEG Pz-Oz and EOG horizontal are at
100 Hz, while EMG submental, Resp oro-nasal and Temp rectal are at 1 Hz. MNE
reads an EDF at the maximum rate across channels and repeats samples for the
slower ones, so a 1 Hz channel would arrive looking like a 100 Hz signal that
is piecewise constant. This script REFUSES any channel whose native rate is not
the maximum, rather than silently training on 100x-repeated values.

USAGE
    # needs the raw EDFs - not in this repo, ~8.1 GB
    #   https://physionet.org/content/sleep-edfx/1.0.0/
    #   or the Kaggle dataset if you preprocess where you train

    python distillation/preprocess_multichannel.py --edf-root <dir>
    python distillation/preprocess_multichannel.py --edf-root <dir> --verify
    python distillation/preprocess_multichannel.py --edf-root <dir> \\
        --channels "EEG Fpz-Cz" "EOG horizontal" "EEG Pz-Oz"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"

EPOCH_SEC = 30
PRE_POST_WAKE_MIN = 30
DEFAULT_CHANNELS = ["EEG Fpz-Cz", "EOG horizontal"]

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
INVALID_STAGES = {"Sleep stage ?", "Movement time"}


# ---------------------------------------------------------------------------
# COPIED UNCHANGED from code/Phase1_48/Phase1reworked/preprocess.py.
# Any edit here silently invalidates every comparison against a stored result.
# ---------------------------------------------------------------------------
def find_hypnogram(psg_file, directory):
    base = psg_file[:6]
    pattern = re.compile(rf"{base}..-Hypnogram.edf")
    for f in os.listdir(directory):
        if pattern.fullmatch(f):
            return os.path.join(directory, f)
    return None


def build_epoch_labels(raw, annotations):
    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times
    samples_per_epoch = int(EPOCH_SEC * sfreq)
    n_epochs = total_samples // samples_per_epoch

    labels = np.array(["INVALID"] * n_epochs, dtype=object)
    for desc, onset, duration in zip(annotations.description,
                                     annotations.onset,
                                     annotations.duration):
        if desc in INVALID_STAGES or desc not in STAGE_MAP:
            continue
        start_epoch = int(onset // EPOCH_SEC)
        end_epoch = int((onset + duration) // EPOCH_SEC)
        labels[start_epoch:end_epoch] = STAGE_MAP[desc]
    return labels, samples_per_epoch


def epoch_bounds(labels):
    """The trimming half of extract_epochs_and_trim, split out so the sample
    slice can be applied to several channels at once."""
    valid_mask = labels != "INVALID"
    labels = labels[valid_mask]
    sleep_idx = np.where(labels != "W")[0]
    if len(sleep_idx) == 0:
        return None, None, None
    pre_epochs = int((PRE_POST_WAKE_MIN * 60) / EPOCH_SEC)
    start_ep = max(0, sleep_idx[0] - pre_epochs)
    end_ep = min(len(labels), sleep_idx[-1] + pre_epochs + 1)
    return start_ep, end_ep, labels[start_ep:end_ep]
# ---------------------------------------------------------------------------


def native_rates(psg_path) -> dict[str, float]:
    """
    Per-channel sampling rate, parsed from the EDF header directly.

    Deliberately not via mne's internals. An earlier version called
    `mne.io.edf.edf._read_edf_header`, whose signature changed in mne 1.12 and
    broke - and because the caller swallowed the exception, the effect was to
    silently stop checking sampling rates rather than to fail. The header layout
    below is fixed by the EDF spec and will not move.

        256 bytes  fixed header, with ns (number of signals) in the last 4
        ns*16      labels
        ns*80      transducer
        ns*8       physical dimension
        ns*8*4     physical min/max, digital min/max
        ns*80      prefiltering
        ns*8       samples per data record   <- what we want
    """
    with open(psg_path, "rb") as f:
        head = f.read(256)
        n_records = int(head[236:244].decode("ascii", "ignore").strip() or 0)
        record_sec = float(head[244:252].decode("ascii", "ignore").strip() or 1.0)
        ns = int(head[252:256].decode("ascii", "ignore").strip() or 0)
        if ns <= 0 or record_sec <= 0:
            return {}
        labels = [f.read(16).decode("ascii", "ignore").strip() for _ in range(ns)]
        f.read(ns * (80 + 8 + 8 + 8 + 8 + 8 + 80))             # skip to n_samps
        n_samps = [int(f.read(8).decode("ascii", "ignore").strip() or 0)
                   for _ in range(ns)]
    del n_records
    return {lab: n / record_sec for lab, n in zip(labels, n_samps)}


def process(edf_dirs, channels, out_root: Path, limit=None):
    import mne
    from tqdm import tqdm

    tensor_dir = out_root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    records, skipped = [], []

    for directory in edf_dirs:
        psg_files = sorted(glob.glob(os.path.join(directory, "*-PSG.edf")))
        if limit:
            psg_files = psg_files[:limit]
        for psg_path in tqdm(psg_files, desc=f"{Path(directory).name}"):
            fname = os.path.basename(psg_path)
            hyp_path = find_hypnogram(fname, directory)
            if hyp_path is None:
                skipped.append((fname, "no hypnogram"))
                continue

            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            missing = [c for c in channels if c not in raw.ch_names]
            if missing:
                skipped.append((fname, f"missing channels {missing}"))
                continue

            # refuse channels MNE upsampled to reach the file's max rate
            rates = native_rates(psg_path)
            if rates:
                top = max(rates.values())
                slow = [c for c in channels if rates.get(c, top) < top]
                if slow:
                    skipped.append((fname, f"channels below max rate {slow}"))
                    continue

            annotations = mne.read_annotations(hyp_path)
            raw.set_annotations(annotations)

            labels, spe = build_epoch_labels(raw, annotations)
            start_ep, end_ep, labels = epoch_bounds(labels)
            if start_ep is None or len(labels) == 0:
                skipped.append((fname, "no sleep epochs"))
                continue

            data = raw.get_data(picks=channels)                # (C, n_samples), volts
            sl = slice(start_ep * spe, end_ep * spe)
            data = data[:, sl]
            n_ep = data.shape[1] // spe
            # (C, n_ep, spe) -> (n_ep, C, spe)
            epochs = data[:, :n_ep * spe].reshape(len(channels), n_ep, spe).transpose(1, 0, 2)

            assert epochs.shape[0] == len(labels), (
                f"{fname}: {epochs.shape[0]} epochs against {len(labels)} labels")

            rec = fname.replace(".edf", "")
            torch.save(torch.tensor(epochs, dtype=torch.float32),
                       tensor_dir / f"{rec}.pt")
            records.append({
                "tensor_path": f"{out_root.name}/tensors/{rec}.pt",
                "stage_sequence": " ".join(labels),
                "spectral": f"processed_sleepedf/spectral/{rec}_spectral.pt",
                "channels": "|".join(channels),
            })
    return records, skipped


def verify(out_root: Path, old_root: Path, n=None) -> int:
    """Channel 0 of the new tensors must equal the existing single-channel ones."""
    new_idx = out_root / "index.csv"
    if not new_idx.exists():
        raise SystemExit(f"Missing {new_idx}. Run without --verify first.")
    df = pd.read_csv(new_idx)
    if n:
        df = df.head(n)

    print(f"verifying channel 0 against {old_root}/tensors/ on {len(df)} recordings")
    ok = mismatch = missing = 0
    for _, row in df.iterrows():
        rec = Path(row["tensor_path"]).stem
        old_p = REPO_ROOT / old_root / "tensors" / f"{rec}.pt"
        new_p = REPO_ROOT / row["tensor_path"]
        if not old_p.exists():
            missing += 1
            continue
        old = torch.load(old_p, map_location="cpu")
        new = torch.load(new_p, map_location="cpu")[:, 0, :]
        if old.shape != new.shape:
            print(f"  {rec}: SHAPE {tuple(old.shape)} vs {tuple(new.shape)}")
            mismatch += 1
            continue
        d = (old - new).abs().max().item()
        if d > 1e-9:
            print(f"  {rec}: max abs difference {d:.3e}")
            mismatch += 1
        else:
            ok += 1

    print(f"\n  identical : {ok}")
    print(f"  mismatched: {mismatch}")
    print(f"  missing   : {missing}")
    if mismatch:
        print("\nFAIL. The epoching or trimming has drifted from the original "
              "pipeline. Every comparison against a stored result would be "
              "invalid. Do not train on these tensors.")
        return 1
    print("\nPASS. Channel 0 reproduces the existing tensors exactly, so the new "
          "tensors are comparable to every published number and splits.json "
          "still applies.")
    return 0


def fit_norm(out_root: Path, splits_path: Path) -> dict:
    """Per-channel 1/std over the TRAIN split only, mirroring eeg_normalization.py."""
    sp = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    df = pd.read_csv(out_root / "index.csv")
    df["rec"] = [Path(p).stem for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    train = df[df["subject"].isin(sp["train"])]

    chans = train.iloc[0]["channels"].split("|")
    # Welford over recordings rather than concatenating 170k epochs into memory
    n = np.zeros(len(chans))
    s = np.zeros(len(chans))
    ss = np.zeros(len(chans))
    for _, row in train.iterrows():
        t = torch.load(REPO_ROOT / row["tensor_path"], map_location="cpu").double()
        flat = t.permute(1, 0, 2).reshape(len(chans), -1)
        n += flat.shape[1]
        s += flat.sum(1).numpy()
        ss += (flat ** 2).sum(1).numpy()
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-30))

    out = {
        "fitted_on": "train split only",
        "n_recordings": int(len(train)),
        "n_subjects": int(train["subject"].nunique()),
        "channels": chans,
        "mean_volts": [float(v) for v in mean],
        "std_volts": [float(v) for v in std],
        "scale": [float(1.0 / v) for v in std],
        "note": "Multiply each channel by its `scale` to reach unit variance. "
                "Per-CHANNEL, not per-epoch: a per-epoch z-score would normalise "
                "away the amplitude difference that defines N3.",
        "reference": "single-channel EEG scale from eeg_normalization.json is "
                     "15849.46; channel 0 here should land close to it.",
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "multichannel_norm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c, sc in zip(chans, out["scale"]):
        print(f"  {c:<20} scale {sc:>14,.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf-root", help="directory containing sleep-cassette/ and "
                                       "sleep-telemetry/")
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--out", default="processed_sleepedf_mc")
    ap.add_argument("--old", default="processed_sleepedf")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap recordings per directory. For a quick check.")
    ap.add_argument("--verify", action="store_true",
                    help="Compare channel 0 against the existing tensors and exit.")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out
    if args.verify:
        return verify(out_root, Path(args.old), args.limit)

    if not args.edf_root:
        raise SystemExit(
            "--edf-root is required.\n\n"
            "The raw EDFs are not in this repository (~8.1 GB). Get them from\n"
            "  https://physionet.org/content/sleep-edfx/1.0.0/\n"
            "or preprocess on Kaggle, where the dataset is already available and\n"
            "where the trainers run anyway.\n\n"
            "Expected layout:\n"
            "  <edf-root>/sleep-cassette/*.edf\n"
            "  <edf-root>/sleep-telemetry/*.edf")

    root = Path(args.edf_root)
    dirs = [str(root / "sleep-cassette"), str(root / "sleep-telemetry")]
    missing = [d for d in dirs if not Path(d).is_dir()]
    if missing:
        raise SystemExit(f"Not found: {missing}")

    try:
        import mne                                             # noqa: F401
        from tqdm import tqdm                                  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"{e}. pip install mne tqdm")

    print(f"channels : {args.channels}")
    print(f"output   : {out_root}")
    records, skipped = process(dirs, args.channels, out_root, args.limit)

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_root / "index.csv", index=False)
    print(f"\n{len(records)} recordings written, {len(skipped)} skipped")
    for f, why in skipped[:10]:
        print(f"  skipped {f}: {why}")

    print(f"\nper-channel normalisation, TRAIN split only:")
    fit_norm(out_root, Path(args.splits))

    print(f"\nNEXT: python distillation/preprocess_multichannel.py --verify")
    print(f"      Channel 0 must reproduce {args.old}/tensors/ exactly before "
          f"anything is trained on this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
