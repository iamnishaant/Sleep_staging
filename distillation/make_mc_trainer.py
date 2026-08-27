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

ONE CONFOUND, STATED UP FRONT
-----------------------------
student_N2multiscale was trained with a gradient-accumulation bug that this
trainer fixes. Its loss was already `sum / nv` - this chunk's share of the
full-batch mean - and it was then multiplied by `nvc / nv` again. Measured
against a true full-batch backward: gradients came out exactly 0.5x too small
with balanced chunks, and MIS-DIRECTED (cosine 0.9916) whenever padding made
the chunks uneven, because each chunk was effectively weighted by nvc^2 rather
than nvc. AdamW normalises away a uniform rescale; it cannot fix a direction.

So "N3mc beats N2multiscale" would confound the added channels with the
gradient fix. To get a clean read, re-run N2multiscale from the regenerated
kaggle_train_student_v2.py first - it is ~20 minutes - and use ITS number as
the bar. The bar constants below are the OLD (buggy-run) figures until that
happens; update them when it does.

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
    # DATA RESOLUTION. The inherited code does `find_file("index.csv")` and takes
    # the first match, then assumes `index.parent.parent` contains everything.
    # Both assumptions break here:
    #
    #   1. With processed_sleepedf AND processed_sleepedf_mc both attached,
    #      sorted() puts the SINGLE-channel index first ('/' is 0x2F, '_' is
    #      0x5F). The run would read the wrong index - and because both indexes
    #      carry the same recordings and stage sequences, it would WORK, quietly,
    #      which is worse than failing.
    #   2. The multi-channel tensors (8 GB) and the spectral features (32 MB)
    #      are naturally two separate Kaggle datasets, so they do not share a
    #      parent directory.
    #
    # Both are resolved explicitly below, and the index is checked to be the one
    # whose `channels` column matches CHANNELS. Absolute paths are stored, so
    # the dataset's `root` argument becomes irrelevant.
    # Matches the block v2 now emits, which resolves its own index explicitly.
    # Kept as an exact-match replace so a change in v2 breaks this loudly rather
    # than leaving mc silently reading a single-channel index.
    _old_paths = (
        '    index = find_file("processed_sleepedf/index.csv")\n'
        '    root = index.parent.parent\n'
        '    print(f"  index     {index}")\n'
        '    if "channels" in pd.read_csv(index, nrows=1).columns:\n'
        '        raise SystemExit(f"{index} is a MULTI-CHANNEL index, but this "\n'
        '                         f"trainer is single-channel.\\n"\n'
        '                         f"Attach processed_sleepedf, or run "\n'
        '                         f"kaggle_train_student_mc.py instead.")\n'
        '    df = pd.read_csv(index)\n'
        '    df["rec"] = [str(p).replace("\\\\", "/").rsplit("/", 1)[-1][:-3] '
        'for p in df["tensor_path"]]\n'
        '    df["subject"] = df["rec"].str[:5]\n'
        '    df["tensor_path"] = ["processed_sleepedf/tensors/%s.pt" % r for r in df["rec"]]\n'
        '    df["spectral"] = ["processed_sleepedf/spectral/%s_spectral.pt" % r '
        'for r in df["rec"]]')
    _new_paths = (
        '    index = find_file(f"{DATA_DIR}/index.csv")\n'
        '    tensor_dir = index.parent / "tensors"\n'
        '    spec_dir = find_file("processed_sleepedf/spectral")\n'
        '    print(f"  index     {index}")\n'
        '    print(f"  tensors   {tensor_dir}")\n'
        '    print(f"  spectral  {spec_dir}")\n'
        '    if not tensor_dir.is_dir():\n'
        '        raise SystemExit(f"{tensor_dir} is not a directory. Attach the "\n'
        '                         f"{DATA_DIR} dataset (tensors/ + index.csv).")\n'
        '\n'
        '    df = pd.read_csv(index)\n'
        '    if "channels" not in df.columns:\n'
        '        raise SystemExit(f"{index} has no `channels` column, so it is not a "\n'
        '                         f"multi-channel index. Attach {DATA_DIR}, produced by "\n'
        '                         f"preprocess_multichannel.py.")\n'
        '    _have = str(df.iloc[0]["channels"]).split("|")\n'
        '    if _have[:len(CHANNELS)] != CHANNELS:\n'
        '        raise SystemExit(f"Channel mismatch.\\n  index has  {_have}\\n"\n'
        '                         f"  CHANNELS   {CHANNELS}\\n"\n'
        '                         f"CHANNELS must be a leading slice of what was "\n'
        '                         f"preprocessed, in the same order.")\n'
        '\n'
        '    df["rec"] = [str(p).replace("\\\\", "/").rsplit("/", 1)[-1][:-3] '
        'for p in df["tensor_path"]]\n'
        '    df["subject"] = df["rec"].str[:5]\n'
        '    # absolute, so the two datasets need not share a parent\n'
        '    df["tensor_path"] = [str(tensor_dir / f"{r}.pt") for r in df["rec"]]\n'
        '    df["spectral"] = [str(spec_dir / f"{r}_spectral.pt") for r in df["rec"]]\n'
        '    _missing = [p for p in list(df["tensor_path"]) + list(df["spectral"])\n'
        '                if not Path(p).exists()]\n'
        '    if _missing:\n'
        '        raise SystemExit(f"{len(_missing)} input files missing, first: "\n'
        '                         f"{_missing[:3]}")')
    if src.count(_old_paths) != 1:
        raise SystemExit(f"data-path fix did not apply ({src.count(_old_paths)} matches).")
    src = src.replace(_old_paths, _new_paths, 1)

    # CHANNELS may be a leading slice of what was preprocessed, so the dataset
    # must trim. This is what makes "EEG only" / "EEG+EOG" / "all three" a config
    # edit rather than three preprocessing runs.
    src = src.replace(
        '        xt = xt[:n].float() * _CH_SCALE',
        '        xt = xt[:n, :_keep].float() * _CH_SCALE')
    src = src.replace(
        "class SleepDataset(Dataset):",
        "# How many leading channels of the stored tensor the model consumes.\n"
        "# CHANNELS may be a leading slice of what was preprocessed - main()\n"
        "# verifies that against the index's `channels` column before training.\n"
        "_keep = len(CHANNELS)\n\n\n"
        "class SleepDataset(Dataset):", 1)

    if not all((n4, n5, n6)):
        raise SystemExit(f"body substitution failed ({n4}, {n5}, {n6})")

    # RESUME GUARD. The inherited check compares ck["eeg_scale"] against
    # EEG_SCALE, and both are None here - so `None != None` is False and the
    # guard can never fire. A stale student_last.pt would resume silently onto
    # DIFFERENT per-channel scales, which is exactly the failure this file
    # argues against everywhere else: a numerically wrong channel goes inert
    # with no error and no warning.
    _old_guard = (
        '        if ck.get("eeg_scale") != EEG_SCALE or ck.get("schedule_shape") != '
        '{"epochs": EPOCHS, "steps": steps}:\n'
        '            raise SystemExit(f"Cannot resume: config changed "\n'
        '                             f"(scale {ck.get(\'eeg_scale\')} -> {EEG_SCALE}). '
        'Delete {out}.")')
    _new_guard = (
        '        _want = {"channels": CHANNELS, "channel_scale": CHANNEL_SCALE,\n'
        '                 "schedule_shape": {"epochs": EPOCHS, "steps": steps}}\n'
        '        _have = {k: ck.get(k) for k in _want}\n'
        '        if _have != _want:\n'
        '            raise SystemExit("Cannot resume: config changed.\\n"\n'
        '                             f"  checkpoint {_have}\\n"\n'
        '                             f"  this run   {_want}\\n"\n'
        '                             f"Delete {out} to start fresh.")')
    if src.count(_old_guard) != 1:
        raise SystemExit(f"resume-guard fix did not apply "
                         f"({src.count(_old_guard)} matches).")
    src = src.replace(_old_guard, _new_guard, 1)

    # The inherited bars are student_baseline_E0 (0.6881) and student_N1norm
    # (0.6821). Neither is the right comparison here: this run's own docstring
    # says the bar is student_N2multiscale, which is the same encoder on EEG
    # alone, so it isolates the added channels. Printing one bar while claiming
    # another is how a result gets read against the wrong reference.
    src, nb = re.subn(
        r'^BASELINE_VAL_MACRO_F1 = 0\.6881\nBASELINE_VAL_KAPPA    = 0\.6389$',
        '# student_N2multiscale: SAME encoder, EEG only. The bar that isolates\n'
        '# the added channels, and the one this experiment is judged against.\n'
        'BASELINE_VAL_MACRO_F1 = 0.7236\n'
        'BASELINE_VAL_KAPPA    = 0.6735', src, flags=re.M)
    src, nc = re.subn(
        r'^N1NORM_VAL_MACRO_F1   = 0\.6821\nN1NORM_VAL_KAPPA      = 0\.6323$',
        '# student_baseline_E0: the original shipped model, EEG branch inert.\n'
        '# Kept as the long-run reference, not as the bar.\n'
        'N1NORM_VAL_MACRO_F1   = 0.6881\n'
        'N1NORM_VAL_KAPPA      = 0.6389', src, flags=re.M)
    src = src.replace(
        'vs student_N1norm (same EEG scale, old encoder) - "\n'
        '          "the bar that isolates the encoder:',
        'vs student_baseline_E0 (the original shipped model, "\n'
        '          "EEG branch inert):')
    src = src.replace('EXPERIMENT {EXPERIMENT}  (MULTI-SCALE TEMPORAL ENCODER)',
                      'EXPERIMENT {EXPERIMENT}  (MULTI-CHANNEL: EEG + EOG + Pz-Oz)')
    src = src.replace('EXPERIMENT {EXPERIMENT}  -  temporal encoder A/B',
                      'EXPERIMENT {EXPERIMENT}  -  added-channel A/B')
    src = src.replace(
        'print(f"  parameters: {npar:,}  '
        '(N1norm was 121,099; the encoder costs +18,507)")',
        'print(f"  parameters: {npar:,}  '
        '(N2multiscale was 139,606; {len(CHANNELS)} channels cost +{npar-139606:,})")')
    if nb != 1 or nc != 1:
        raise SystemExit(f"bar substitution failed ({nb}, {nc}).")

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
