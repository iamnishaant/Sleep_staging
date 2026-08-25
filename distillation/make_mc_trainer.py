"""
Derive kaggle_train_student_mc.py from kaggle_train_student_v2.py.

WHAT CHANGES, AND WHY IT IS SHORT
---------------------------------
The multiscale encoder already accepts (B, T, C, L) and already takes
`in_channels`. So going multi-channel is a data-shape change, not an
architecture change:

  1. index/tensor root      processed_sleepedf -> processed_sleepedf_mc
  2. EEG_SCALE (a scalar)   -> CHANNEL_SCALE (one per channel), because EOG and
                               Pz-Oz have different variances from Fpz-Cz and a
                               single scalar would leave two of the three
                               branches numerically small - the exact defect
                               that made the original temporal branch inert
  3. collate                allocates (B, L, C, 3000) instead of (B, L, 3000)
  4. the model gets in_channels=len(CHANNELS)
  5. SEED becomes settable from argv, because the ensemble needs K runs that
     differ only in seed

Everything else - splits, schedule, class weighting, early stopping, the
evaluation, the zeroed-input ablation - is inherited untouched.

CHANNEL SELECTION IS A CONFIG LINE, NOT A REPROCESS
---------------------------------------------------
preprocess_multichannel.py writes all three 100 Hz channels. CHANNELS picks
which of them the model sees, so EEG-only, EEG+EOG and EEG+EOG+Pz-Oz are three
config edits rather than three preprocessing runs.

USAGE
    python distillation/make_mc_trainer.py
    python distillation/make_mc_trainer.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "kaggle_train_student_v2.py"
DST = HERE / "kaggle_train_student_mc.py"

Q = '"' * 3

DOCSTRING = Q + '''
================================================================================
KAGGLE NOTEBOOK - MULTI-CHANNEL STUDENT  (experiment "N3mc")
================================================================================
Adds EOG and a second EEG derivation to the multiscale student. Same encoder,
same schedule, same splits, seed 42.

WHY EOG
-------
AASM scoring defines two of the five stages partly by the eyes: rapid eye
movements mark REM, slow rolling eye movements mark N1. Those are the two
classes this model is weakest at (test F1 0.78 and 0.44).

separability_verdict.json established that N1 is REPRESENTATION-bound - no
decision rule on the existing probabilities improves N1 F1 by more than +0.0086.
Representation-bound means only new information helps, and the 34 spectral
features cannot be that information because they are computed FROM the same
EEG. EOG is the first genuinely independent measurement this project has used.

WHY Pz-Oz AS WELL
-----------------
A second cortical derivation, not a summary of the first. Cheap to include once
the tensors are multi-channel. If it does not earn its ~6,000 parameters, drop
it - CHANNELS is a config line.

PER-CHANNEL SCALING IS NOT OPTIONAL
-----------------------------------
The raw EDF is in volts. A single scalar tuned for Fpz-Cz would leave EOG and
Pz-Oz numerically small, and check_branch_live.py already measured what happens
to a numerically small input: it goes INERT, silently, with no error and no
warning. Scales are 1/std per channel, fitted on the TRAINING split only, in
results/multichannel_norm.json.

WHAT SUCCESS LOOKS LIKE
-----------------------
    beat  : student_N2multiscale, validation macro-F1 0.7236, kappa 0.6735
            (same encoder, EEG only - so this isolates the added channels)
    watch : N1 and REM specifically. If the added channels help anywhere, it is
            there. A gain concentrated in W or N2 would suggest the model is
            just using more capacity rather than more information.

Held-out TEST must NOT be used to decide anything here.

The zeroed-input ablation at the end zeroes ALL channels together. To attribute
per channel, re-run it zeroing one at a time - that is a separate script, not a
knob here.

RUNNING K SEEDS FOR THE ENSEMBLE
--------------------------------
    python kaggle_train_student_mc.py        # seed 42
    python kaggle_train_student_mc.py 1      # seed 1, separate OUT_DIR

Each seed writes to student_N3mc_s<SEED>/, so K runs coexist. The ensemble that
supplies soft targets is built from those checkpoints; see the distillation
plan in the project notes.
================================================================================
''' + Q

CONFIG_BLOCK = '''EXPERIMENT     = "N3mc"

# Which of the preprocessed channels the model sees. All three are written by
# preprocess_multichannel.py; this line selects, it does not reprocess.
CHANNELS       = ["EEG Fpz-Cz", "EOG horizontal", "EEG Pz-Oz"]

# 1/std per channel over the TRAIN split, from results/multichannel_norm.json.
# A single scalar would leave EOG and Pz-Oz numerically small, and a numerically
# small input goes INERT - measured, not assumed (check_branch_live.py).
CHANNEL_SCALE  = None      # filled in by make_mc_trainer.py from the artefact

DATA_DIR       = "processed_sleepedf_mc"'''


def channel_scales(channels):
    """Read the fitted per-channel scales, so they cannot drift from the data."""
    p = HERE / "results" / "multichannel_norm.json"
    if not p.exists():
        raise SystemExit(
            f"Missing {p}.\nRun preprocess_multichannel.py first - the per-channel "
            f"scales are fitted on the TRAIN split there, and hardcoding them here "
            f"would let them drift from the tensors they describe.")
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    have = d["channels"]
    missing = [c for c in channels if c not in have]
    if missing:
        raise SystemExit(f"{p.name} has no scale for {missing}; it covers {have}.")
    return [round(d["scale"][have.index(c)], 2) for c in channels]


def build() -> str:
    src = SRC.read_text(encoding="utf-8")

    end = src.index(Q, src.index(Q) + 3) + 3
    src = DOCSTRING + src[end:]

    chans = ["EEG Fpz-Cz", "EOG horizontal", "EEG Pz-Oz"]
    cfg = CONFIG_BLOCK.replace(
        "CHANNEL_SCALE  = None      # filled in by make_mc_trainer.py from the artefact",
        f"CHANNEL_SCALE  = {channel_scales(chans)}")

    src, n1 = re.subn(r'^EXPERIMENT     = "N2multiscale"$', cfg, src, flags=re.M)
    src, n2 = re.subn(
        r'^EEG_SCALE      = 15849\.46$',
        '# Superseded by CHANNEL_SCALE above. Set to None so inherited code that\n'
        '# still references it fails loudly rather than silently scaling by 1.\n'
        'EEG_SCALE      = None', src, flags=re.M)
    src, n3 = re.subn(
        r'^SEED           = 42$',
        'import sys as _sys\n'
        '# The ensemble needs K runs differing only in seed:\n'
        '#     python kaggle_train_student_mc.py 1\n'
        'SEED           = int(_sys.argv[1]) if len(_sys.argv) > 1 else 42', src, flags=re.M)
    if n1 != 1 or n2 != 1 or n3 != 1:
        raise SystemExit(f"config substitution failed ({n1}, {n2}, {n3})")

    src, n4 = re.subn(
        r'        xt = xt\[:n\]\.float\(\) \* EEG_SCALE          # <<< THE ONE CHANGE\n'
        r'        return xt, xs\[:n\]\.float\(\), y\[:n\]',
        '        # (n, C, 3000) * (1, C, 1) - per channel, not one scalar\n'
        '        xt = xt[:n].float() * _CH_SCALE\n'
        '        return xt, xs[:n].float(), y[:n]', src)
    src, n5 = re.subn(
        r'    xt = torch\.zeros\(B, L, batch\[0\]\[0\]\.shape\[1\]\)',
        '    xt = torch.zeros(B, L, *batch[0][0].shape[1:])      # (B, L, C, 3000)', src)
    src, n6 = re.subn(
        r'self\.temporal_encoder = MultiScaleEpochEncoder\(embed=embed, n_tokens=n_tokens\)',
        'self.temporal_encoder = MultiScaleEpochEncoder(\n'
        '            embed=embed, n_tokens=n_tokens, in_channels=len(CHANNELS))', src)
    src, n7 = re.subn(r'"processed_sleepedf/tensors/%s\.pt"',
                      'DATA_DIR + "/tensors/%s.pt"', src)
    if not all((n4, n5, n6, n7)):
        raise SystemExit(f"body substitution failed ({n4}, {n5}, {n6}, {n7})")

    src = src.replace(
        "NUM_CLASSES = 5\nSTAGES = [",
        "_CH_SCALE = torch.tensor(CHANNEL_SCALE, dtype=torch.float32).view(1, -1, 1)\n\n"
        "NUM_CLASSES = 5\nSTAGES = [", 1)
    src = src.replace('"encoder": "multiscale",',
                      '"encoder": "multiscale", "channels": CHANNELS,\n'
                      '                 "channel_scale": CHANNEL_SCALE,', 1)
    # one OUT_DIR per seed so K runs coexist
    src = src.replace('out = Path(OUT_ROOT) / f"student_{EXPERIMENT}"',
                      'out = Path(OUT_ROOT) / f"student_{EXPERIMENT}_s{SEED}"', 1)
    # the header printed the scalar EEG_SCALE, which is now None
    src = re.sub(
        r'    print\(f"  EEG_SCALE = \{EEG_SCALE\}.*?\)\n',
        '    print(f"  channels  = {CHANNELS}")\n'
        '    print(f"  scales    = {CHANNEL_SCALE}")\n'
        '    print(f"  seed      = {SEED}")\n', src)
    src = src.replace('student_{EXPERIMENT}/student_best.pt',
                      'student_{EXPERIMENT}_s{SEED}/student_best.pt')
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        stale = (not DST.exists()) or DST.read_text(encoding="utf-8") != text
        print(f"{DST.name}: {'STALE - regenerate' if stale else 'up to date'}")
        return 1 if stale else 0
    DST.write_text(text, encoding="utf-8")
    print(f"wrote {DST.name} ({len(text.splitlines())} lines)")
    print("changed: docstring, CHANNELS/CHANNEL_SCALE/DATA_DIR, seed from argv, "
          "dataset scaling, collate shape, in_channels, per-seed OUT_DIR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
