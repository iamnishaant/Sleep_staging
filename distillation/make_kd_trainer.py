"""
Derive kaggle_train_student_kd.py from kaggle_train_student_v2.py.

WHAT THIS IS FOR
----------------
The soft-label run. It distils the ENSEMBLE teacher into a single student, and
it is the first configuration in this project where the teacher is actually
better than the student:

    ensemble (run1 + run2 + baseline_E0)   val kappa 0.6931   test 0.7140
    best single member                     val kappa 0.6763   test 0.6992

Every earlier distillation attempt lost because the teacher was WORSE than the
student - test kappa 0.6055 teaching a 0.6449 student. That is why the soft term
dragged the student down, and why fixing the loss, the temperature or alpha was
never going to help.

WHY THE EEG-ONLY STUDENT IS THE TARGET
--------------------------------------
Run 2 showed the extra channels buy nothing measurable (validation kappa
-0.0122, paired per-recording p=0.865 on test). They stay in the ENSEMBLE,
where their decorrelated errors are worth something, but the student distilled
out of it takes single-channel input: same accuracy, fewer parameters
(139,606 vs 151,606), better compression (4.65x vs 4.28x), and one channel to
acquire at inference instead of three.

That asymmetry is the point of distillation. The teacher may be as awkward as
it likes; the student is what ships.

WHAT CHANGES vs kaggle_train_student_v2.py
------------------------------------------
  1. the docstring
  2. EXPERIMENT, plus ALPHA / T / TEACHER_TAG
  3. the dataset also loads the cached teacher logits for the same epochs
  4. collate carries them
  5. the loss becomes distillation_loss, lifted verbatim from kd_loss.py

Splits, schedule, seed, class weighting, early stopping, the evaluation and the
zeroed-EEG ablation are inherited untouched. ALPHA=1.0 disables the soft term
entirely and reproduces the hard-label run through the SAME trainer, which is
what makes the comparison isolate the training signal rather than a pile of
confounds.

THE T**2 TERM IS MANDATORY
--------------------------
Softening logits by T shrinks the soft gradient by 1/T^2. Without the
correction, training still runs and still converges - it just barely distils,
and nothing in the loss curve reveals it. kd_loss.py asserts the scaling
explicitly (test_kd_loss.py, 13/13) rather than trusting it, and the function is
copied here verbatim so the two cannot drift.

USAGE
    python distillation/make_kd_trainer.py
    python distillation/make_kd_trainer.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "kaggle_train_student_v2.py"
DST = HERE / "kaggle_train_student_kd.py"

CV_TEACHER_GUARD = '\n    # ---- CV / cohort splits and a shared teacher do not mix (roadmap 0.1, 3.1) ----\n    # The ensemble teacher was trained on the ORIGINAL 69-subject train split.\n    #   CV_FOLD:      12-16 of each fold\'s 17 validation subjects are inside it.\n    #   COHORT_SPLIT: 15 of the 22 held-out ST subjects are inside it (68%).\n    # Either way the student inherits the teacher\'s memorisation of the very\n    # subjects it is then scored on. Generating the missing logits would hide\n    # that, not fix it.\n    if (CV_FOLD is not None or COHORT_SPLIT) and ALPHA < 1.0:\n        _which = "CV_FOLD" if CV_FOLD is not None else "COHORT_SPLIT"\n        _detail = ("12-16 of this fold\'s 17 validation subjects"\n                   if CV_FOLD is not None else\n                   "15 of the 22 held-out ST subjects (68% of the test cohort)")\n        raise SystemExit(\n            f"{_which} is set and the soft term is active (ALPHA < 1.0).\\n"\n            "\\n"\n            f"  Refused. teacher_logits_{TEACHER_TAG} comes from a teacher trained on\\n"\n            f"  the original 69-subject split, and {_detail}\\n"\n            "  are inside that set. Distilling from it would inflate the result\\n"\n            "  through the teacher - the same leakage, one level out, that this\\n"\n            "  project was started to correct.\\n"\n            "\\n"\n            "  Two valid routes:\\n"\n            "    (a) student-side questions - normalisation, augmentation, EOG,\\n"\n            "        capacity, cohort transfer - need no teacher. Set ALPHA = 1.0.\\n"\n            "    (b) KD hyperparameters (alpha, T) need a teacher trained on the\\n"\n            "        same split being evaluated. For CV that is 5 folds x 3\\n"\n            "        ensemble members; for the cohort split it is an SC-only\\n"\n            "        teacher. Until those exist, report alpha/T as selected on\\n"\n            "        the original split.\\n")\n\n'
KD = HERE / "kd_loss.py"

Q = '"' * 3

DOCSTRING = Q + '''
================================================================================
KAGGLE NOTEBOOK - DISTIL THE ENSEMBLE  (experiment "N4kd")
================================================================================
Trains the single-channel multiscale student against SOFT TARGETS from an
ensemble teacher, instead of against one-hot labels.

WHY THIS ONE SHOULD WORK WHEN THE OTHERS DID NOT
------------------------------------------------
Distillation has cost accuracy every previous time it was tried here:

    student_baseline_E0   (hard labels)   test kappa 0.6449
    student_distilled_E1b (soft, T=3)     test kappa 0.6134
    teacher_E1b           (the teacher)   test kappa 0.6055

The cause was never the loss, the temperature or alpha. The TEACHER WAS WORSE
THAN THE STUDENT, so half the gradient pulled a good model toward a bad one.

The ensemble teacher is not:

    ensemble (run1 + run2 + baseline_E0)   val kappa 0.6931   test 0.7140
    best single member                     val kappa 0.6763   test 0.6992

It is better without being bigger - three models averaged, no new architecture -
which is the only property distillation actually requires. Averaging cancels
decorrelated error, and members differing in INPUT and ENCODER decorrelate far
more than members differing only in seed: measured, run1+run2 gained +0.0033
over the best member on validation, and adding the weaker-but-architecturally-
different baseline_E0 took that to +0.0168.

SOFT TARGETS ARE ALSO THE HONEST TARGET
---------------------------------------
Inter-scorer agreement on AASM staging is about kappa 0.76, and far worse for
N1 - human scorers agree on N1 roughly 25-45% of the time. A one-hot "N1"
records one technician's opinion as certainty. Where the ensemble members split
0.45/0.40/0.15 across N1/N2/W, that soft target is closer to the truth than the
one-hot is. That is the clinical argument for soft labels, and it only holds
when the uncertainty means something - which a single weaker teacher's does not.

WHY A SINGLE-CHANNEL STUDENT
----------------------------
Run 2 showed EOG and Pz-Oz buy nothing measurable (validation kappa -0.0122;
paired per-recording p=0.865 on test). They stay in the TEACHER, where their
decorrelated errors are worth something, but the student takes one channel:
139,606 params against 151,606, 4.65x compression against 4.28x, and one
electrode to acquire at inference rather than three.

SETUP
-----
1. Attach the sleepedf dataset (processed_sleepedf/...).
2. Attach the teacher logits as a second dataset:
       distillation/results/teacher_logits_ENSmc/   (137 files, 3.2 MB)
3. Accelerator -> GPU T4 x2.
4. Run. ~20 min.

WHAT SUCCESS LOOKS LIKE
-----------------------
    beat  : student_N2multiscale_fix, validation macro-F1 0.7249, kappa 0.6763
            (the SAME architecture on hard labels - so this isolates the
            training signal, which is the whole question)
    watch : whether the gap to the ensemble teacher (val 0.6931) closes. A
            student that recovers most of it at 4.65x compression is the result;
            one that lands back at the hard-label number means the ensemble's
            advantage does not transfer, which is worth reporting as it stands.

ALPHA=1.0 disables the soft term and reproduces the hard-label run through this
same trainer. If anything looks surprising, run that first - a difference that
survives alpha=1.0 is not about distillation.

Held-out TEST must NOT be used to decide anything here.

GENERATED FILE - do not edit by hand.
Regenerate with: python distillation/make_kd_trainer.py
================================================================================
''' + Q

CONFIG = '''EXPERIMENT     = "N4kd"

# Soft-target weighting. alpha is the weight on the HARD cross-entropy term:
#   alpha = 0.5  half hard, half soft - the usual starting point
#   alpha = 1.0  soft term OFF; reproduces the hard-label run in this trainer,
#                which is the control the comparison needs
ALPHA          = 0.5
T              = 3.0

# Which cached teacher to distil from. Built by build_ensemble.py --cache-train,
# read from teacher_logits_<TEACHER_TAG>/.
TEACHER_TAG    = "ENSmc"'''

KD_LOSS_HEADER = '''# ---------------------------------------------------------------- KD LOSS ---
# Lifted verbatim from kd_loss.py by make_kd_trainer.py so the two cannot drift.
# Asserted by test_kd_loss.py (13/13), including the T**2 scaling below.
#
# THE T**2 TERM IS MANDATORY. Softening by T shrinks the soft gradient by 1/T^2;
# without the correction training still runs and still converges, it just barely
# distils, and the loss curve looks entirely normal while it happens.
'''


def extract_kd_loss() -> str:
    src = KD.read_text(encoding="utf-8")
    m = re.search(r"^def distillation_loss\b.*?(?=\n\n\ndef |\Z)", src, re.S | re.M)
    if not m:
        raise SystemExit("Could not find distillation_loss in kd_loss.py")
    return m.group(0).rstrip()


def build() -> str:
    src = SRC.read_text(encoding="utf-8")

    end = src.index(Q, src.index(Q) + 3) + 3
    src = DOCSTRING + src[end:]

    src, n1 = re.subn(r'^EXPERIMENT     = "N2multiscale"$', CONFIG, src, flags=re.M)
    if n1 != 1:
        raise SystemExit(f"config substitution failed ({n1})")

    # --- KD loss, next to the model -----------------------------------------
    marker = "# -------------------------------------------------------------------- DATA ---"
    if src.count(marker) != 1:
        raise SystemExit("DATA marker not found once")
    src = src.replace(marker, KD_LOSS_HEADER + extract_kd_loss() + "\n\n\n" + marker, 1)

    # --- dataset also yields the teacher's soft targets ----------------------
    _old_ds = (
        '    def __init__(self, df, root, window=256, overlap=0):\n'
        '        self.df = df.reset_index(drop=True); self.root = Path(root)\n'
        '        self.window = window; self.stride = max(1, window - overlap)')
    _new_ds = (
        '    def __init__(self, df, root, window=256, overlap=0, teacher_dir=None):\n'
        '        self.df = df.reset_index(drop=True); self.root = Path(root)\n'
        '        # teacher_dir=None yields ZERO logits, which is what validation\n'
        '        # wants: the soft term is a training signal, and scoring the\n'
        '        # student against its teacher would measure imitation rather than\n'
        '        # agreement with the expert.\n'
        '        self.teacher_dir = Path(teacher_dir) if teacher_dir else None\n'
        '        self.window = window; self.stride = max(1, window - overlap)')
    if src.count(_old_ds) != 1:
        raise SystemExit(f"dataset __init__ substitution failed ({src.count(_old_ds)})")
    src = src.replace(_old_ds, _new_ds, 1)

    _old_get = (
        '        n = min(len(y), xt.shape[0], xs.shape[0])\n'
        '        xt = xt[:n].float() * EEG_SCALE          # <<< THE ONE CHANGE\n'
        '        return xt, xs[:n].float(), y[:n]')
    _new_get = (
        '        n = min(len(y), xt.shape[0], xs.shape[0])\n'
        '        if self.teacher_dir is not None:\n'
        '            d = np.load(self.teacher_dir / f"{row[\'rec\']}.npz")\n'
        '            tl = torch.from_numpy(d["logits"][st:en].astype(np.float32))\n'
        '            n = min(n, tl.shape[0])\n'
        '            tl = tl[:n]\n'
        '        else:\n'
        '            tl = torch.zeros(n, NUM_CLASSES)\n'
        '        xt = xt[:n].float() * EEG_SCALE\n'
        '        return xt, xs[:n].float(), tl, y[:n]')
    if src.count(_old_get) != 1:
        raise SystemExit(f"dataset __getitem__ substitution failed ({src.count(_old_get)})")
    src = src.replace(_old_get, _new_get, 1)

    # --- collate carries them ------------------------------------------------
    _old_col = (
        'def collate(batch):\n'
        '    L, B = max(b[0].shape[0] for b in batch), len(batch)\n'
        '    xt = torch.zeros(B, L, batch[0][0].shape[1])\n'
        '    xs = torch.zeros(B, L, batch[0][1].shape[1])\n'
        '    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)\n'
        '    mask = torch.ones(B, L, dtype=torch.bool)\n'
        '    for i, (a, sp, c) in enumerate(batch):\n'
        '        n = a.shape[0]\n'
        '        xt[i, :n], xs[i, :n], y[i, :n], mask[i, :n] = a, sp, c, False\n'
        '    return xt, xs, y, mask')
    _new_col = (
        'def collate(batch):\n'
        '    L, B = max(b[0].shape[0] for b in batch), len(batch)\n'
        '    xt = torch.zeros(B, L, batch[0][0].shape[1])\n'
        '    xs = torch.zeros(B, L, batch[0][1].shape[1])\n'
        '    tl = torch.zeros(B, L, NUM_CLASSES)\n'
        '    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)\n'
        '    mask = torch.ones(B, L, dtype=torch.bool)\n'
        '    for i, (a, sp, t, c) in enumerate(batch):\n'
        '        n = a.shape[0]\n'
        '        xt[i, :n], xs[i, :n], tl[i, :n], y[i, :n], mask[i, :n] = a, sp, t, c, False\n'
        '    return xt, xs, tl, y, mask')
    if src.count(_old_col) != 1:
        raise SystemExit(f"collate substitution failed ({src.count(_old_col)})")
    src = src.replace(_old_col, _new_col, 1)

    # --- every loop that unpacks a batch -------------------------------------
    src = src.replace('for xt, xs, y, mask in loader:', 'for xt, xs, _tl, y, mask in loader:')
    src = src.replace('for xt, xs, y, mask in dl_tr:', 'for xt, xs, tl, y, mask in dl_tr:')

    # --- the loss ------------------------------------------------------------
    _old_loss = (
        '                with torch.amp.autocast(device_type="cuda", enabled=amp):\n'
        '                    lg = model(xt[sl], xs[sl], mask[sl])\n'
        '                    loss = F.cross_entropy(lg.reshape(-1, NUM_CLASSES), y[sl].reshape(-1),\n'
        '                                           weight=cw, ignore_index=IGNORE_INDEX,\n'
        '                                           reduction="sum") / nv')
    _new_loss = (
        '                with torch.amp.autocast(device_type="cuda", enabled=amp):\n'
        '                    lg = model(xt[sl], xs[sl], mask[sl])\n'
        '                    # distillation_loss reduces to a MEAN over this chunk, so\n'
        '                    # the chunk is weighted by its share of the batch\'s valid\n'
        '                    # tokens. That is the correct pairing - the sum/nv form\n'
        '                    # inherited from v2 must NOT also be weighted, which is the\n'
        '                    # bug make_v2_trainer fixes.\n'
        '                    loss = distillation_loss(lg, tl[sl], y[sl], T=T, alpha=ALPHA,\n'
        '                                             class_weights=cw) * (nvc / nv).float()')
    if src.count(_old_loss) != 1:
        raise SystemExit(f"loss substitution failed ({src.count(_old_loss)})")
    src = src.replace(_old_loss, _new_loss, 1)

    # --- wire the teacher directory in main() --------------------------------
    _old_mk = (
        '    ds_tr = SleepDataset(tr, root, WINDOW_SIZE, OVERLAP)\n'
        '    ds_va = SleepDataset(va, root, WINDOW_SIZE, 0)')
    _new_mk = (
        '    tdir = None\n'
        '    if ALPHA < 1.0:\n'
        '        tdir = find_file(f"teacher_logits_{TEACHER_TAG}").resolve()\n'
        '        print(f"  teacher   {tdir}")\n'
        '        have = {p.stem for p in tdir.glob("*.npz")}\n'
        '        miss = sorted(set(tr["rec"]) - have)\n'
        '        if miss:\n'
        '            raise SystemExit(f"{len(miss)} training recordings have no cached "\n'
        '                             f"teacher logits, first: {miss[:3]}")\n'
        '        print(f"  soft targets for all {len(tr)} training recordings")\n'
        '    else:\n'
        '        print("  ALPHA=1.0 - soft term DISABLED, this is the hard-label control")\n'
        '\n'
        '    # Validation gets no teacher: the student is scored against the EXPERT,\n'
        '    # never against its teacher.\n'
        '    ds_tr = SleepDataset(tr, root, WINDOW_SIZE, OVERLAP, teacher_dir=tdir)\n'
        '    ds_va = SleepDataset(va, root, WINDOW_SIZE, 0)')
    if src.count(_old_mk) != 1:
        raise SystemExit(f"dataset construction substitution failed ({src.count(_old_mk)})")
    src = src.replace(_old_mk, _new_mk, 1)

    # The input-scale sanity check unpacks a sample. The dataset now yields four
    # things, not three. Caught by test_trainer_smoke, which is the entire reason
    # that test exists - this is a one-line crash five minutes into a GPU run.
    _old_probe = '    _xt, _xs, _y = ds_tr[0]'
    _new_probe = ('    _xt, _xs, _tl, _y = ds_tr[0]\n'
                  '    if ALPHA < 1.0:\n'
                  '        # a teacher that is all zeros means the cache was found but is\n'
                  '        # empty for this recording, which would train against a uniform\n'
                  '        # soft target and look like nothing in particular went wrong\n'
                  '        if float(_tl.abs().sum()) == 0.0:\n'
                  '            raise SystemExit("Teacher logits for the first training "\n'
                  '                             "sample are all zero. The cache is present "\n'
                  '                             "but empty - check teacher_logits_"\n'
                  '                             f"{TEACHER_TAG}.")\n'
                  '        print(f"  teacher logits |x| mean {_tl.abs().mean():.4f} "\n'
                  '              f"(soft targets are live)")')
    if src.count(_old_probe) != 1:
        raise SystemExit(f"sanity-probe substitution failed ({src.count(_old_probe)})")
    src = src.replace(_old_probe, _new_probe, 1)

    # THE TEACHER LOGITS MUST GO TO THE DEVICE TOO.
    #
    # Renaming the loop variable is not enough: xt/xs/y/mask are moved, tl was
    # not, so distillation_loss indexed a CPU tensor with a CUDA mask and died
    # a few seconds into the first epoch. test_trainer_smoke could not catch it
    # because it runs on CPU, where every tensor is already on one device - a
    # device-placement bug is structurally invisible there. Guarded statically
    # instead; see test_trainer_smoke.check_device_moves.
    _old_dev = ('            xt, xs = xt.to(device, non_blocking=True), xs.to(device, non_blocking=True)\n'
                '            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)')
    _new_dev = ('            xt, xs = xt.to(device, non_blocking=True), xs.to(device, non_blocking=True)\n'
                '            tl = tl.to(device, non_blocking=True)\n'
                '            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)')
    if src.count(_old_dev) != 1:
        raise SystemExit(f"device-move substitution failed ({src.count(_old_dev)})")
    src = src.replace(_old_dev, _new_dev, 1)

    # The inherited bars are student_baseline_E0's. The bar for THIS experiment
    # is the same architecture trained on hard labels, which is what isolates
    # the training signal - the whole question the run exists to answer.
    _old_bars = ('BASELINE_VAL_MACRO_F1 = 0.6881\n'
                 'BASELINE_VAL_KAPPA    = 0.6389')
    _new_bars = ('# student_N2multiscale_fix: the SAME architecture on hard labels.\n'
                 '# Beating this is what shows the soft targets did the work.\n'
                 'BASELINE_VAL_MACRO_F1 = 0.7249\n'
                 'BASELINE_VAL_KAPPA    = 0.6763')
    if src.count(_old_bars) != 1:
        raise SystemExit(f"bar substitution failed ({src.count(_old_bars)})")
    src = src.replace(_old_bars, _new_bars, 1)
    src = src.replace('N1NORM_VAL_MACRO_F1   = 0.6821\nN1NORM_VAL_KAPPA      = 0.6323',
                      '# The ensemble teacher this student is distilled FROM. The student is\n'
                      '# not expected to reach it; how much of the gap it closes is the result.\n'
                      'N1NORM_VAL_MACRO_F1   = 0.7370\nN1NORM_VAL_KAPPA      = 0.6931')
    src = src.replace('vs student_N1norm (same EEG scale, old encoder) - "\n'
                      '          "the bar that isolates the encoder:',
                      'vs the ENSEMBLE TEACHER it was distilled from "\n'
                      '          "(a 3-model average, not a shippable model):')

    # --- config echo and checkpoint provenance -------------------------------
    src = src.replace('EXPERIMENT {EXPERIMENT}  -  temporal encoder A/B',
                      'EXPERIMENT {EXPERIMENT}  -  distil the ensemble teacher')
    src = src.replace('    print(f"  seed      = {SEED}")',
                      '    print(f"  seed      = {SEED}")\n'
                      '    print(f"  alpha     = {ALPHA}   T = {T}   teacher = {TEACHER_TAG}")')
    # Guarded: this replace silently stopped applying on 30 Aug when
    # '"cv_fold": CV_FOLD' was inserted ahead of it, and the kd trainer
    # quietly began recording a literal alpha=1.0 instead of ALPHA.
    _prov = '"eeg_scale": EEG_SCALE, "cv_fold": CV_FOLD, "alpha": 1.0, "T": 3.0,'
    assert src.count(_prov) == 1, f'provenance anchor not found ({src.count(_prov)})'
    src = src.replace(_prov,
                      '"eeg_scale": EEG_SCALE, "cv_fold": CV_FOLD, "alpha": ALPHA, "T": T,\n'
                      '                 "teacher_tag": TEACHER_TAG,', 1)
    # ---- refuse CV + shared teacher (roadmap 0.1) ----
    # Anchor on `tdir = None`, which is at 4-space indent and precedes the
    # whole `if ALPHA < 1.0: ... else: ...` structure. Anchoring inside that
    # structure (on the 8-space `have = ...` line) silently orphaned the
    # cache-completeness check behind the guard's raise.
    _anchor = '    tdir = None\n'
    assert src.count(_anchor) == 1, 'tdir anchor not found'
    src = src.replace(_anchor, CV_TEACHER_GUARD.lstrip('\n') + _anchor, 1)

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
    print("changed: docstring, ALPHA/T/TEACHER_TAG, kd_loss, dataset, collate, "
          "batch unpacking, loss, teacher wiring, checkpoint provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
